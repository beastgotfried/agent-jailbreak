"""Deterministic neutral fixtures; no real benchmark result is implied."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitor.disagreement import ScorePair, calibrate, canonical, strict_json
from src.monitor.prepare_scores import model_binding
from src.replay.data import Prompt, ReplayError, labelled_scores, mapped_label, split_label_roles
from src.replay.harness import ReplaySpec, SCENARIOS, build_stream, compare_arms, initial_policy, run_arm
from src.replay.cli import run_benchmark
from src.replay.scorecard import main as regenerate_main


def fixtures():
    records, scores = [], {}
    for purpose, n in (("calibration", 60), ("evaluation", 40)):
        for label in (0, 1):
            for index in range(n):
                text = f"neutral {purpose} {'positive' if label else 'negative'} fixture {index}"
                pid = hashlib.sha256(text.encode()).hexdigest()
                records.append(Prompt(pid, text, purpose, "attack" if label else "benign", label))
                scores[pid] = {"guardrail_score": 0.9 if label else 0.1, "canary_score": 0.9 if label else 0.1}
    return records, scores


def config(binding="a" * 64):
    refs = [ScorePair(f"ref-{i}", "unit", i, 0.1, 0.1, binding, "reference") for i in range(32)]
    null = [ScorePair(f"cal-{i}", "unit", i, 0.1, 0.1, binding, "calibration") for i in range(64)]
    return calibrate(refs, null, window_size=8, stride=4, horizon=32, alpha=0.1, null_runs=20)


def paired(scenario="silent_score_fault", approve=True, delay=2):
    records, scores = fixtures()
    cfg = config()
    roles = split_label_roles(records)
    policy = initial_policy(roles, scores, cfg.model_pair_id)
    evaluation = [row for row in records if row.purpose == "evaluation"]
    spec = ReplaySpec(horizon=32, shift_index=8, label_delay=delay)
    stream = build_stream(records, scenario, 0, spec)
    outputs = [run_arm(stream, scores, policy, roles, evaluation, cfg, scenario=scenario, seed=0,
        spec=spec, watcher=watcher, approve_repair=approve) for watcher in (False, True)]
    return outputs, spec


def test_stream_is_deterministic_and_without_replacement():
    records, _ = fixtures(); spec = ReplaySpec(32, 8, label_delay=2)
    for scenario in SCENARIOS:
        a = build_stream(records, scenario, 4, spec)
        assert a == build_stream(records, scenario, 4, spec)
        assert len({row.prompt_id for row in a}) == 32
        assert all(row.kind == "benign" for row in a[:8])
    natural = build_stream(records, "attack_mixture_shift", 4, spec)
    fault = build_stream(records, "silent_score_fault", 4, spec)
    assert natural == fault
    assert sum(row.label == 1 for row in natural[8:]) == 18
    with pytest.raises(ReplayError, match="insufficient_unique"):
        build_stream(records, "stationary_benign", 0, ReplaySpec(512, 128))


def test_paired_positive_control_preserves_inputs_and_isolates_response():
    ((events_a, a), (events_b, b)), spec = paired()
    assert a["input_digest"] == b["input_digest"]
    assert a["initial_policy_id"] == b["initial_policy_id"]
    assert not a["response"] and b["response"]["applied"]
    assert b["response"]["response_index"] - b["first_alarm_index"] == spec.label_delay
    until = b["response"]["response_index"]
    for left, right in zip(events_a, events_b):
        assert (left["prompt_id"], left["guardrail_score"], left["canary_score"]) == (right["prompt_id"], right["guardrail_score"], right["canary_score"])
        if left["sequence"] < until:
            assert left["decision"] == right["decision"]
    comparison = compare_arms(a, b)
    assert comparison["coverage_restored"] is True
    assert a["final_holdout"]["by_class"]["1"]["coverage"] == 0
    assert b["final_holdout"]["by_class"]["1"]["coverage"] == 1
    assert all(event["stub_response"] in ("STUB_ACCEPTED", "STUB_REFUSED", "REVIEW_PENDING") for event in events_b)
    assert not any("text" in event or "prompt" in event for event in events_b)


def test_unapproved_or_late_response_is_not_counted_as_restoration():
    ((_, a), (_, b)), _ = paired(approve=False)
    assert b["response"]["status"] == "approval_required"
    assert not compare_arms(a, b)["coverage_restored"]


def test_response_delay_beyond_episode_never_applies_retroactively():
    ((_, a), (_, b)), _ = paired(delay=40)
    assert b["response"]["status"] == "pending_beyond_episode"
    assert not b["response"]["applied"]
    assert compare_arms(a, b)["coverage_restored"] is False


def test_stationary_negative_control_is_observed_not_universal_proof():
    ((_, a), (_, b)), _ = paired("stationary_benign")
    assert not b["negative_control_alarm"] and not b["response"]
    assert compare_arms(a, b)["coverage_restored"] is None


def test_natural_shift_with_preserved_gap_is_not_falsely_claimed_detected():
    ((_, a), (_, b)), _ = paired("attack_mixture_shift")
    assert b["first_alarm_index"] is None and b["missed_intended_change"]
    compared = compare_arms(a, b)
    assert compared["coverage_restored"] is None
    assert compared["restoration_status"] == "no_baseline_loss_to_restore"


def test_label_purpose_and_splits_protect_evaluation():
    records, scores = fixtures()
    roles = split_label_roles(records)
    sets = [{row.prompt_id for row in values} for values in roles.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert all(row.purpose == "calibration" for values in roles.values() for row in values)
    with pytest.raises(ReplayError, match="invalid_label_request_purpose"):
        labelled_scores([next(row for row in records if row.purpose == "evaluation")], scores, "response", "original")
    ((_, a), (_, b)), _ = paired()
    b["input_digest"] = "0" * 64
    with pytest.raises(ReplayError, match="unpaired_benchmark"):
        compare_arms(a, b)


def test_injection_benign_is_not_universal_harmful_request_label():
    wa = {"pool_kind": "benign", "sources": [{"repo": "Norrrrrrr/WAInjectBench", "source_task": "prompt_injection", "original_labels": {"label": 0}}]}
    assert mapped_label(wa) is None
    tc = {"pool_kind": "benign", "sources": [{"repo": "lmsys/toxic-chat", "source_task": "toxicity_or_jailbreak", "original_labels": {"toxicity": "0", "jailbreaking": "0", "human_annotation": "True"}}]}
    assert mapped_label(tc) == 0


def test_full_fixture_replay_and_scorecard_regeneration_are_byte_repeatable(tmp_path):
    records, _ = fixtures()
    pools, models = tmp_path / "pools", tmp_path / "models"
    pools.mkdir(); models.mkdir()
    manifest = {"outputs": {}}
    for filename, kind in (("benign.jsonl", "benign"), ("attacks.jsonl", "attack")):
        rows = []
        for record in records:
            if record.kind != kind: continue
            source = {"repo": "JailbreakBench/JBB-Behaviors", "source_task": "harmful_request",
                "original_relative_path": f"jbb-behaviors/data/{'harmful' if record.label else 'benign'}-behaviors.csv", "original_labels": {}}
            rows.append({"pool_kind": kind, "partition": record.purpose, "prompt_id": record.prompt_id,
                "text": record.text, "text_sha256": record.prompt_id, "sources": [source]})
        path = pools / filename
        path.write_text("".join(canonical(row) + "\n" for row in rows))
        manifest["outputs"][filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (pools / "pool-manifest.json").write_text(canonical(manifest))
    cards = {"artifacts": {key: {"sha256": str(i) * 64} for i, key in enumerate(("guardrail_model", "guardrail_vectorizer", "canary_model", "canary_vectorizer"))},
        "operating_thresholds": {name: {"threshold": 0.5} for name in ("guardrail", "canary")},
        "normalization": "lowercase, collapse whitespace", "label_rule": "toxicity OR jailbreaking", "packages": {"scikit_learn": "fixture"}}
    (models / "model-cards.json").write_text(canonical(cards))
    monitor_path = tmp_path / "monitor.json"
    monitor_path.write_text(canonical(config(model_binding(cards)).to_dict()))

    class FakeScorer:
        def score(self, prompts):
            return [{"guardrail_score": 0.9 if "positive" in text else 0.1,
                "canary_score": 0.9 if "positive" in text else 0.1,
                "guardrail_all_zero_features": False, "canary_all_zero_features": False} for text in prompts]

    spec = ReplaySpec(32, 8, label_delay=2)
    dirs = [tmp_path / "first", tmp_path / "second"]
    for directory in dirs:
        card = run_benchmark(models, pools, monitor_path, directory, source_commit="b" * 40,
            seeds=(0, 1), approve_repair=True, spec=spec, scorer_factory=lambda _: FakeScorer())
        assert len(card["episodes"]) == 6
        assert card["negative_control"]["observed_silent"]
        assert not card["negative_control"]["proves_permanent_silence"]
    files = {path.name for path in dirs[0].iterdir()}
    assert files == {path.name for path in dirs[1].iterdir()}
    assert all((dirs[0] / name).read_bytes() == (dirs[1] / name).read_bytes() for name in files)
    regenerated = tmp_path / "regenerated"
    assert regenerate_main(["--results", str(dirs[0]), "--output-dir", str(regenerated)]) == 0
    for name in ("scorecard.json", "scorecard.md", "scorecard.csv"):
        assert (regenerated / name).read_bytes() == (dirs[0] / name).read_bytes()
    with pytest.raises(ReplayError, match="nonempty_benchmark_output"):
        run_benchmark(models, pools, monitor_path, dirs[0], source_commit="b" * 40,
            seeds=(0,), spec=spec, scorer_factory=lambda _: FakeScorer())

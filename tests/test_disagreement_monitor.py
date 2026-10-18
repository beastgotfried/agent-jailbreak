"""Neutral numeric/unit fixtures only; not evidence of safety performance."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitor.disagreement import (
    Calibration, DisagreementMonitor, MonitorError, ScorePair, calibrate,
    canonical, digest, ks_statistic, pair_streams, strict_json,
)
from src.monitor.cli import main as cli_main
from src.monitor.prepare_scores import model_binding, prepare_scores

PAIR = "a" * 64


def sample(n, purpose="live", prefix="event", *, g=0.4, c=0.4):
    return [ScorePair(f"{prefix}-{i}", "unit-stream", i, g, c, PAIR, purpose) for i in range(n)]


@pytest.fixture(scope="module")
def config():
    return calibrate(sample(32, "reference", "ref"), sample(64, "calibration", "cal"),
        window_size=8, stride=4, horizon=32, alpha=0.1, null_runs=40, seed=17)


def write_events(path, events):
    path.write_text("".join(canonical(event.to_dict()) + "\n" for event in events), encoding="utf-8")


@pytest.mark.parametrize("reference,observed", [
    ([0, 0, 1], [0, 1, 1]), ([0, 0], [0, 0]),
    ([-1, -0.3, 0.2], [0.4, 0.5]), ([0.1, 0.1, 0.4, 0.8], [0.1, 0.1]),
])
def test_ks_distance_matches_scipy_statistic_only(reference, observed):
    assert ks_statistic(reference, observed) == pytest.approx(ks_2samp(reference, observed).statistic)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "private neutral value", None])
@pytest.mark.parametrize("field", ["guardrail_score", "canary_score"])
def test_invalid_scores_reject_without_value_echo(bad, field):
    with pytest.raises(MonitorError, match="invalid_score") as error:
        replace(sample(1)[0], **{field: bad})
    assert "private neutral value" not in str(error.value)


def test_pair_schema_rejects_raw_content_and_bool_sequences():
    row = sample(1)[0].to_dict()
    row["text"] = "neutral private fixture"
    with pytest.raises(MonitorError, match="invalid_pair_fields"):
        ScorePair.from_dict(row)
    with pytest.raises(MonitorError, match="invalid_sequence"):
        replace(sample(1)[0], sequence=True)
    with pytest.raises(MonitorError, match="invalid_event_or_stream_id"):
        replace(sample(1)[0], event_id="neutral private fixture")


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{not json'])
def test_strict_json_rejects_ambiguous_or_nonfinite_payload(text):
    with pytest.raises(MonitorError):
        strict_json(text)


def test_stationary_stream_has_warmup_and_finite_budget(config):
    monitor = DisagreementMonitor(config)
    output = [monitor.update(event) for event in sample(32)]
    assert all(row["statistic"] is None for row in output[:7])
    assert output[7]["statistic"] == 0
    assert sum(row["statistic"] is not None for row in output) == config.max_looks == 7
    assert not any(row["new_alarm"] for row in output)
    assert monitor.exhausted and output[-1]["horizon_exhausted"]
    assert len(monitor.checkpoint()["events"]) == config.horizon
    before = monitor.checkpoint()
    with pytest.raises(MonitorError, match="horizon_exhausted"):
        monitor.update(replace(sample(1)[0], sequence=32, event_id="extra"))
    assert monitor.checkpoint() == before


def test_shift_latches_one_change_alarm_without_action_authority(config):
    stream = sample(32)
    stream[16:] = [replace(event, guardrail_score=1.0, canary_score=0.0) for event in stream[16:]]
    monitor = DisagreementMonitor(config)
    updates = [monitor.update(event) for event in stream]
    alarms = [row["new_alarm"] for row in updates if row["new_alarm"]]
    assert len(alarms) == 1
    alarm = alarms[0]
    assert alarm["sequence"] == 19
    assert alarm["kind"] == "disagreement_shift"
    assert alarm["evidence_status"] == "change_signal_only"
    assert alarm["action_authority"] == "none"
    assert alarm["statistic"] > alarm["threshold"]
    assert monitor.alarm == alarm


def test_common_mode_score_change_is_an_explicit_blind_spot(config):
    stream = sample(32, g=0.1, c=0.1)
    stream[16:] = [replace(event, guardrail_score=0.9, canary_score=0.9) for event in stream[16:]]
    monitor = DisagreementMonitor(config)
    assert not any(monitor.update(event)["new_alarm"] for event in stream)


def test_same_absolute_gap_but_reversed_direction_is_visible():
    cfg = calibrate(sample(32, "reference", "ref", g=0.6, c=0.4), sample(64, "calibration", "cal", g=0.6, c=0.4),
        window_size=8, stride=4, horizon=32, alpha=0.1, null_runs=40)
    monitor = DisagreementMonitor(cfg)
    results = [monitor.update(event) for event in sample(8, g=0.4, c=0.6)]
    assert results[-1]["new_alarm"]["statistic"] == 1.0


def test_harmless_distribution_change_cannot_be_called_harm(config):
    monitor = DisagreementMonitor(config)
    for event in sample(8, g=0.5, c=0.4):
        update = monitor.update(event)
    assert update["new_alarm"]["evidence_status"] == "change_signal_only"
    assert "harm_verified" not in update["new_alarm"]


@pytest.mark.parametrize("changes,code", [
    ({"sequence": 2}, "noncontiguous_sequence"),
    ({"event_id": "event-0"}, "duplicate_event"),
    ({"model_pair_id": "b" * 64}, "model_binding_mismatch"),
    ({"stream_id": "different"}, "stream_identity_changed"),
    ({"purpose": "reference"}, "reference_is_not_live_telemetry"),
])
def test_telemetry_faults_do_not_mutate_monitor(config, changes, code):
    monitor = DisagreementMonitor(config)
    monitor.update(sample(1)[0])
    before = monitor.checkpoint()
    event = sample(2)[1]
    with pytest.raises(MonitorError, match=code):
        monitor.update(replace(event, **changes))
    assert monitor.checkpoint() == before


def test_checkpoint_restore_reproduces_state_and_alarms(config):
    stream = sample(32)
    stream[16:] = [replace(event, guardrail_score=0.8) for event in stream[16:]]
    original = DisagreementMonitor(config)
    for event in stream[:23]:
        original.update(event)
    checkpoint = strict_json(canonical(original.checkpoint()))
    restored = DisagreementMonitor.restore(config, checkpoint)
    assert restored.alarm == original.alarm
    assert [restored.update(e) for e in stream[23:]] == [original.update(e) for e in stream[23:]]
    assert restored.checkpoint() == original.checkpoint()
    checkpoint["events"][0]["guardrail_score"] = 0.1
    with pytest.raises(MonitorError, match="checkpoint_binding_mismatch"):
        DisagreementMonitor.restore(config, checkpoint)
    checkpoint["checkpoint_id"] = digest({k: v for k, v in checkpoint.items() if k != "checkpoint_id"})
    checkpoint["events"][0]["sequence"] = 9
    checkpoint["checkpoint_id"] = digest({k: v for k, v in checkpoint.items() if k != "checkpoint_id"})
    with pytest.raises(MonitorError, match="noncontiguous_sequence"):
        DisagreementMonitor.restore(config, checkpoint)


def test_calibration_round_trip_and_determinism(config):
    restored = Calibration.from_dict(strict_json(canonical(config.to_dict())))
    assert restored == config and restored.calibration_id == config.calibration_id
    second = calibrate(sample(32, "reference", "ref"), sample(64, "calibration", "cal"),
        window_size=8, stride=4, horizon=32, alpha=0.1, null_runs=40, seed=17)
    assert second.to_dict() == config.to_dict()
    payload = config.to_dict()
    payload["threshold"] = 0.5
    with pytest.raises(MonitorError):
        Calibration.from_dict(payload)


@pytest.mark.parametrize("change", [
    {"alpha": 0}, {"alpha": 1}, {"window_size": True}, {"stride": 0},
    {"null_runs": 0}, {"horizon": 1000}, {"alpha": 0.0001},
])
def test_calibration_rejects_invalid_budget_or_insufficient_data(change):
    params = dict(window_size=8, stride=4, horizon=32, alpha=0.1, null_runs=40)
    params.update(change)
    with pytest.raises(MonitorError):
        calibrate(sample(32, "reference", "ref"), sample(64, "calibration", "cal"), **params)


def test_empirical_threshold_uses_full_episode_maxima_not_individual_p_values():
    refs = [replace(event, guardrail_score=0.3 + (i % 7) * 0.04, canary_score=0.2)
        for i, event in enumerate(sample(32, "reference", "ref"))]
    null = [replace(event, guardrail_score=0.3 + (i % 11) * 0.02, canary_score=0.2)
        for i, event in enumerate(sample(64, "calibration", "cal"))]
    cfg = calibrate(refs, null, window_size=8, stride=4, horizon=32, alpha=0.1, null_runs=20, seed=9)
    rng = np.random.default_rng(9)
    baseline = [event.gap for event in refs]
    gaps = np.array([event.gap for event in null])
    expected = []
    for _ in range(20):
        episode = gaps[rng.choice(64, size=32, replace=False)]
        expected.append(max(ks_2samp(baseline, episode[end - 8:end]).statistic for end in range(8, 33, 4)))
    assert cfg.null_maxima == pytest.approx(expected)
    assert cfg.threshold == pytest.approx(sorted(expected)[18])
    assert sum(value > cfg.threshold for value in cfg.null_maxima) / cfg.null_runs <= cfg.alpha


def test_evaluation_and_overlapping_ids_cannot_calibrate():
    refs, null = sample(32, "reference", "ref"), sample(64, "calibration", "cal")
    params = dict(window_size=8, stride=4, horizon=32, alpha=0.1, null_runs=40)
    with pytest.raises(MonitorError, match="invalid_calibration_input"):
        calibrate(refs, [replace(e, purpose="evaluation") for e in null], **params)
    null[0] = replace(null[0], event_id=refs[0].event_id)
    with pytest.raises(MonitorError, match="reference_calibration_overlap"):
        calibrate(refs, null, **params)
    with pytest.raises(MonitorError, match="mixed_model_binding"):
        calibrate(refs, sample(64, "calibration", "cal" )[:-1] + [replace(sample(64, "calibration", "cal")[-1], model_pair_id="b" * 64)], **params)


def side(event, role):
    data = event.to_dict()
    g, c = data.pop("guardrail_score"), data.pop("canary_score")
    return {**data, "model_role": role, "score": g if role == "guardrail" else c}


def test_two_stream_join_checks_identity_not_arrival_position():
    stream = sample(4)
    left, right = [side(e, "guardrail") for e in stream], [side(e, "canary") for e in stream]
    assert list(pair_streams(left, right)) == stream
    with pytest.raises(MonitorError, match="side_identity_mismatch"):
        list(pair_streams(left, list(reversed(right))))
    with pytest.raises(MonitorError, match="unpaired_stream_end"):
        list(pair_streams(left, right[:-1]))
    right[0]["model_role"] = "guardrail"
    with pytest.raises(MonitorError, match="invalid_side_binding"):
        list(pair_streams(left, right))


def test_cli_calibration_stream_and_checkpoint(tmp_path):
    ref, cal, cfg = (tmp_path / name for name in ("ref.jsonl", "cal.jsonl", "config.json"))
    write_events(ref, sample(32, "reference", "ref"))
    write_events(cal, sample(64, "calibration", "cal"))
    assert cli_main(["calibrate", "--reference", str(ref), "--calibration", str(cal), "--output", str(cfg),
        "--window", "8", "--stride", "4", "--horizon", "32", "--alpha", "0.1", "--null-runs", "40"]) == 0
    output, state = tmp_path / "updates.jsonl", tmp_path / "state.json"
    args = ["run", "--config", str(cfg), "--input", str(cal), "--output", str(output), "--limit", "32", "--checkpoint-out", str(state)]
    assert cli_main(args) == 0
    rows = [strict_json(line) for line in output.read_text().splitlines()]
    assert len(rows) == 33 and rows[-1]["kind"] == "episode_summary"
    assert rows[-1]["horizon_exhausted"] and not rows[-1]["harm_verified"]
    previous = output.read_bytes()
    assert cli_main(args) == 2
    assert output.read_bytes() == previous
    restored = DisagreementMonitor.restore(Calibration.from_dict(strict_json(cfg.read_text())), strict_json(state.read_text()))
    assert restored.exhausted


def test_cli_fault_does_not_echo_rejected_source(tmp_path, capsys, config):
    cfg, scores, output = (tmp_path / name for name in ("config.json", "scores.jsonl", "output.jsonl"))
    cfg.write_text(canonical(config.to_dict()))
    good, bad = [e.to_dict() for e in sample(2)]
    bad["canary_score"] = "neutral private rejected source"
    scores.write_text(canonical(good) + "\n" + canonical(bad) + "\n")
    assert cli_main(["run", "--config", str(cfg), "--input", str(scores), "--output", str(output)]) == 2
    logs = output.read_text() + capsys.readouterr().err
    assert "neutral private rejected source" not in logs
    assert strict_json(output.read_text().splitlines()[-1])["kind"] == "telemetry_fault"


def test_score_preparation_never_scores_evaluation_and_binds_models(tmp_path):
    models, pools, output = (tmp_path / name for name in ("models", "pools", "scores"))
    models.mkdir(); pools.mkdir()
    cards = {"artifacts": {key: {"sha256": str(i) * 64} for i, key in enumerate((
        "guardrail_model", "guardrail_vectorizer", "canary_model", "canary_vectorizer"))},
        "operating_thresholds": {name: {"threshold": 0.5} for name in ("guardrail", "canary")},
        "normalization": "lowercase, collapse whitespace", "label_rule": "toxicity OR jailbreaking", "packages": {"scikit_learn": "fixture"}}
    (models / "model-cards.json").write_text(canonical(cards))
    pool = []
    expected_texts = []
    for purpose, count in (("reference", 4), ("calibration", 8), ("evaluation", 3)):
        for index in range(count):
            text = f"neutral {purpose} fixture {index}"
            identity = hashlib.sha256(text.encode()).hexdigest()
            pool.append({"pool_kind": "benign", "partition": purpose, "prompt_id": identity, "text": text, "text_sha256": identity})
            if purpose != "evaluation":
                expected_texts.append(text)
    path = pools / "benign.jsonl"
    path.write_text("".join(canonical(row) + "\n" for row in pool))
    (pools / "pool-manifest.json").write_text(canonical({"outputs": {"benign.jsonl": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}))

    class FakeScorer:
        def __init__(self): self.seen = []
        def score(self, prompts):
            self.seen.extend(prompts)
            return [{"guardrail_score": 0.4, "canary_score": 0.45,
                "guardrail_all_zero_features": False, "canary_all_zero_features": False} for _ in prompts]

    fake = FakeScorer()
    report = prepare_scores(models, pools, output, scorer_factory=lambda _: fake, batch_size=3)
    assert fake.seen == expected_texts
    assert report["counts"] == {"reference": 4, "calibration": 8}
    assert report["evaluation_rows_scored"] == 0 and report["evaluation_rows_skipped"] == 3
    assert not report["classifiers_refit"]
    rows = [ScorePair.from_dict(strict_json(line)) for line in (output / "reference.jsonl").read_text().splitlines()]
    assert all(event.model_pair_id == model_binding(cards) for event in rows)
    changed = strict_json(canonical(cards)); changed["operating_thresholds"]["guardrail"]["threshold"] = 0.6
    assert model_binding(changed) != model_binding(cards)
    with pytest.raises(MonitorError, match="nonempty_score_output"):
        prepare_scores(models, pools, output, scorer_factory=lambda _: fake)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(MonitorError, match="benign_pool_hash_mismatch"):
        prepare_scores(models, pools, tmp_path / "other", scorer_factory=lambda _: fake)

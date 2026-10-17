"""Freeze a declared traffic-source profile before held-out inference.

Dolly instruction text has no safety labels here. Healthy means the approved
runtime, not a claim about request content. This is finite-source qualification.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np

from src.data.traffic_profile import SOURCE_ID, SOURCE_REVISION, load_instructions
from src.models.score import load_scorer
from src.replay.data import file_sha, load_pools
from src.replay.recovery import BoundBackend
from src.replay.recovery_data import prior_prompt_ids
from src.response.conformal import identity
from src.response.recovery import identity_text
from .cli import new_output, write_new_json
from .disagreement import ScorePair, calibrate, canonical, strict_json
from .prepare_scores import model_binding
from .profiles import ProfiledTrafficWatch, TrafficProfile
from .runtime import checked_scores


def score_role(backend, rows):
    scores = {}
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        values = checked_scores(backend, [r.text for r in batch], backend.model_pair_id)
        scores.update((r.prompt_id, v) for r, v in zip(batch, values))
    return scores


def events(rows, scores, binding, role):
    return [ScorePair(r.prompt_id, "profile-" + role, i, scores[r.prompt_id]["guardrail_score"],
                      scores[r.prompt_id]["canary_score"], binding, role) for i, r in enumerate(rows)]


def run(source, models, pools, xstest, toxic_train, toxic_test, output, *, source_commit):
    if not isinstance(source_commit, str) or re.fullmatch(r"[a-f0-9]{40}", source_commit) is None:
        raise ValueError("invalid_profile_source_commit")
    output, models = Path(output), Path(models)
    if output.exists() and any(output.iterdir()):
        raise ValueError("nonempty_profile_output")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    old_rows, _ = load_pools(pools)
    blocked = prior_prompt_ids(old_rows, toxic_train, toxic_test)
    with Path(xstest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            blocked.add(identity_text(row["prompt"]))
    rows, manifest = load_instructions(source, blocked)
    roles = {role:[r for r in rows if r.role == role] for role in ("reference", "calibration", "evaluation")}
    if any(len(values) < 512 for values in roles.values()):
        raise ValueError("insufficient_profile_roles")
    cards = strict_json((models / "model-cards.json").read_text())
    cards_sha = file_sha(models / "model-cards.json")
    binding = model_binding(cards)
    backend = BoundBackend(load_scorer(models), binding)
    write_new_json(output / "source.json", manifest)
    write_new_json(output / "splits.json", {role:[r.prompt_id for r in values] for role, values in roles.items()})
    write_new_json(output / "provenance.json", {r.prompt_id:list(r.source_rows) for r in rows})
    scored = score_role(backend, roles["reference"] + roles["calibration"])
    reference = events(roles["reference"], scored, binding, "reference")
    calibration = events(roles["calibration"], scored, binding, "calibration")
    for role, pairs in (("reference", reference), ("calibration", calibration)):
        with new_output(output / (role + "-scores.jsonl")) as handle:
            for pair in pairs:
                handle.write(canonical(pair.to_dict()) + "\n")
    config = calibrate(reference, calibration, window_size=64, stride=16, horizon=512,
                       alpha=.05, null_runs=200, seed=42)
    profile = TrafficProfile(SOURCE_ID, SOURCE_REVISION, "exact-instruction-field-v1", config)
    write_new_json(output / "calibration.json", config.to_dict())
    write_new_json(output / "profile.json", profile.to_dict())
    frozen_names = ("source.json", "splits.json", "provenance.json", "reference-scores.jsonl", "calibration-scores.jsonl", "calibration.json", "profile.json")
    frozen = {name:file_sha(output / name) for name in frozen_names}
    write_new_json(output / "pre-evaluation-freeze.json", {"source_commit":source_commit,
        "model_pair_id":binding, "model_cards_sha256":cards_sha, "evaluation_scored":False,
        "profile_id":profile.profile_id, "files":frozen, "planned_seeds":[0,1,2,3,4],
        "episode_events":512, "threshold_retuning_allowed":False})
    for name in (*frozen_names, "pre-evaluation-freeze.json"):
        (output / name).chmod(0o400)
    # Evaluation inference starts only after the frozen artifact above exists.
    evaluation_scores = score_role(backend, roles["evaluation"])
    write_new_json(output / "evaluation-score-cache.json", evaluation_scores)
    summaries = []
    ordered = sorted(roles["evaluation"], key=lambda r:r.prompt_id)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        stream = [ordered[i] for i in rng.choice(len(ordered), size=512, replace=False)]
        monitor = ProfiledTrafficWatch(profile, episode_budget=1, warning_budget=1)
        first = None
        with new_output(output / f"episode-{seed}.jsonl") as handle:
            for i, row in enumerate(stream):
                score = evaluation_scores[row.prompt_id]
                event = ScorePair(row.prompt_id, f"dolly-profile-{seed}", i, score["guardrail_score"],
                                  score["canary_score"], binding, "evaluation")
                update = monitor.update(event, source_id=SOURCE_ID, generation=0)
                if update["new_alarm"] is not None:
                    first = i
                handle.write(canonical(update) + "\n")
        summaries.append({"seed":seed, "events":512, "first_warning_index":first,
                          "input_digest":identity([r.prompt_id for r in stream]), "looks":29})
    if frozen != {name:file_sha(output / name) for name in frozen_names}:
        raise ValueError("profile_freeze_changed")
    if cards_sha != file_sha(models / "model-cards.json"):
        raise ValueError("model_cards_changed")
    for item in cards["artifacts"].values():
        if file_sha(models / item["file"]) != item["sha256"]:
            raise ValueError("model_artifacts_changed")
    report = {"source_commit":source_commit, "source":manifest, "profile_id":profile.profile_id,
        "model_pair_id":binding, "threshold":config.threshold,
        "healthy_runtime_episodes":5, "warning_episodes":sum(r["first_warning_index"] is not None for r in summaries),
        "episode_events":512, "episodes":summaries, "calibration_null_runs":200,
        "calibration_maxima_above_threshold":sum(value > config.threshold for value in config.null_maxima),
        "model_artifacts_unchanged":True, "pre_evaluation_freeze_preserved":True,
        "limits":["No safety labels or safe-traffic claim for Dolly instructions.",
                  "Finite declared-source profile only, not an anytime or arbitrary-source guarantee.",
                  "Five seeds share observations and are not independent statistical trials.",
                  "No exact overlap with known prior prompts; semantic or pretraining overlap is not excluded.",
                  "This profile is not qualified for XSTest or unknown production traffic."]}
    write_new_json(output / "report.json", report)
    write_new_json(output / "evidence.json", {p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native source-qualified traffic profile")
    for name in ("source", "models", "pools", "xstest", "toxic-train", "toxic-test", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.source, args.models, args.pools, args.xstest, args.toxic_train, args.toxic_test,
                     args.output, source_commit=args.source_commit)
        print(canonical({"source":report["source"], "threshold":report["threshold"],
                         "healthy_runtime_episodes":5, "warning_episodes":report["warning_episodes"]}))
    except Exception:
        print(canonical({"error":"profile_qualification_failed_inspect_private_artifacts"}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

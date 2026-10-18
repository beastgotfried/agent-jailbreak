"""Paired score contract and development benign reference; no monitoring or gating by canary."""
from pathlib import Path
import datetime
import numpy as np
from safety_watch.classifier.artifacts import load_guardrail
from safety_watch.classifier.data import load_data, prepare_split
from safety_watch.classifier.evidence import new_operation
from safety_watch.classifier.io import require, canonical, sha256_file, write_json
from safety_watch.classifier.metrics import validation_report
from .artifacts import load_canary
from .model import error_report
from .training import PARTITION_SHA256

def load_pair(model_dir):
    guardrail, canary = load_guardrail(model_dir), load_canary(model_dir)
    require(canary.card["paired_guardrail_sha256"] == guardrail.digest, "Frozen classifier pair differs.")
    require(guardrail.card["dataset"] == canary.card["dataset"], "Frozen pair datasets differ.")
    require(guardrail.card["split_summary"]["partitions_sha256"] ==
            canary.card["split_summary"]["partitions_sha256"], "Frozen pair partitions differ.")
    return guardrail, canary

def joint_records(guardrail, canary, prompt_ids, texts, splits=None, timestamp=None):
    require(not isinstance(texts, str) and not isinstance(prompt_ids, str), "Pass sequences, not strings.")
    prompt_ids, texts = list(prompt_ids), list(texts)
    splits = ["live"] * len(texts) if splits is None else list(splits)
    require(len(prompt_ids) == len(texts) == len(splits), "Pair input lengths differ.")
    require(all(isinstance(i, str) and i.strip() for i in prompt_ids), "Prompt IDs must be nonempty strings.")
    require(all(isinstance(s, str) and s.strip() for s in splits), "Split values must be nonempty strings.")
    require(canary.card["paired_guardrail_sha256"] == guardrail.digest, "Frozen classifier pair differs.")
    gscores, cscores = guardrail.score(texts), canary.score(texts)
    timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
    return [{
        "schema_version": 1, "prompt_id": i, "guardrail_score": float(g), "canary_score": float(c),
        "timestamp": timestamp, "split": split,
        "guardrail_decision": "refuse" if g >= guardrail.threshold else "allow",
        "guardrail_threshold": guardrail.threshold, "canary_threshold": canary.threshold,
        "guardrail_sha256": guardrail.digest, "canary_sha256": canary.digest,
    } for i, g, c, split in zip(prompt_ids, gscores, cscores, splits)]

def evaluate_pair(runtime_root="/runtime", model_dir=None):
    runtime_root = Path(runtime_root)
    guardrail, canary = load_pair(model_dir or runtime_root / "models")
    _, _, result_dir, evidence = new_operation(runtime_root, "pair-evaluate")
    evidence("pair_validation_started", {"guardrail_sha256": guardrail.digest, "canary_sha256": canary.digest})
    data = load_data(runtime_root / "data", evidence)
    require(data.metadata == guardrail.card["dataset"], "Pair dataset provenance differs.")
    split = prepare_split(data, result_dir, 42, 0.20, evidence)
    require(split.digest == PARTITION_SHA256, "Pair evaluation partitions differ.")
    frame = split.validation
    records = joint_records(guardrail, canary, frame["prompt_id"], frame["user_input"],
                            ["validation"] * len(frame))
    gscores = np.array([row["guardrail_score"] for row in records])
    cscores = np.array([row["canary_score"] for row in records])
    reports = {"guardrail": validation_report(frame, gscores, guardrail.threshold),
               "canary": validation_report(frame, cscores, canary.threshold)}
    require(reports["guardrail"] == guardrail.card["validation_metrics"], "Saved guardrail metrics differ.")
    require(reports["canary"] == canary.card["validation_metrics"], "Saved canary metrics differ.")
    correlation = error_report(frame, gscores, cscores, guardrail.threshold, canary.threshold)
    require(correlation == canary.card["error_correlation"], "Pair error report differs.")
    score_path = result_dir / "joint-validation-scores.jsonl"
    with score_path.open("x") as handle:
        for row in records:
            handle.write(canonical(row) + "\n")
    benign = frame["human_annotation"].to_numpy(dtype=bool) & (frame["unsafe"].to_numpy(dtype=int) == 0)
    reference = [dict(records[i], split="reference") for i in np.flatnonzero(benign)]
    reference.sort(key=lambda row: row["prompt_id"])
    # Fixed model-export timestamp keeps reference bytes reproducible across repeated evaluations.
    for row in reference:
        row["timestamp"] = canary.card["exported_utc"]
    reference_path = result_dir / "benign-reference-scores.jsonl"
    with reference_path.open("x") as handle:
        for row in reference:
            handle.write(canonical(row) + "\n")
    # Reloaded pair smoke test, with batch boundaries different from the full evaluation.
    reloaded = load_pair(model_dir or runtime_root / "models")
    smoke = []
    for start in range(0, 100, 17):
        subset = frame.iloc[start:min(start + 17, 100)]
        smoke.extend(joint_records(*reloaded, subset["prompt_id"], subset["user_input"],
                                   ["validation"] * len(subset)))
    np.testing.assert_allclose([r["guardrail_score"] for r in smoke], gscores[:100], rtol=0, atol=1e-12)
    np.testing.assert_allclose([r["canary_score"] for r in smoke], cscores[:100], rtol=0, atol=1e-12)
    evidence("saved_pair_verified", {
        "records": len(records), "smoke_records": len(smoke), "reference_records": len(reference),
        "metrics": reports, "error_correlation": correlation, "official_test_labels_read": False,
    })
    report = {
        "results_directory": str(result_dir), "metrics": reports, "error_correlation": correlation,
        "guardrail_sha256": guardrail.digest, "canary_sha256": canary.digest,
        "thresholds": {"guardrail": guardrail.threshold, "canary": canary.threshold},
        "reference": {"records": len(reference), "path": str(reference_path),
                      "sha256": sha256_file(reference_path),
                      "scope": "Human-benign validation; also threshold calibration, not independent null calibration."},
        "scores_sha256": sha256_file(score_path), "reload_smoke_records": len(smoke),
        "evidence": evidence.verify(),
    }
    write_json(result_dir / "pair-report.json", report)
    return report

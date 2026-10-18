"""Reproducible development fitting; never overwrites the saved/frozen model."""
from dataclasses import dataclass
import time
import joblib
import numpy as np
from .config import TrainConfig
from .runtime import check_runtime
from .io import require, sha256_file, write_json, canonical, source_snapshot
from .evidence import new_operation
from .data import load_data, prepare_split
from .selection import select_model
from .model import fit_model, score_model
from .metrics import choose_threshold, validation_report

@dataclass(repr=False)
class TrainingResult:
    model: object
    split: object
    scores: np.ndarray
    threshold: float
    card: dict
    run_dir: object
    evidence: object
    summary: dict

def train_guardrail(config=None):
    config = config or TrainConfig()
    environment = check_runtime()
    require(0 <= config.target_fpr < 1, "Target FPR must be in [0, 1).")
    require(0 < config.validation_fraction < 1, "Validation fraction must be in (0, 1).")
    run_id, timestamp, run_dir, evidence = new_operation(config.runtime_root, "train")
    provenance = source_snapshot(config.project_root)
    evidence("environment", {"environment": environment, "source": provenance})
    data = load_data(config.runtime_root / "data", evidence)
    split = prepare_split(data, run_dir, config.seed, config.validation_fraction, evidence)
    model_config, selection = select_model(
        split, config.model, run_dir, config.seed, config.target_fpr, config.search, evidence)
    model, fit_stats = fit_model(model_config, split.fit, config.seed)
    started = time.perf_counter()
    scores = score_model(model, split.validation["user_input"])
    fit_stats.update({"validation_scoring_seconds": time.perf_counter() - started,
                      "validation_records": len(scores)})
    human = split.validation["human_annotation"].to_numpy(dtype=bool)
    y = split.validation["unsafe"].to_numpy(dtype=int)
    benign = scores[human & (y == 0)]
    threshold = choose_threshold(benign, config.target_fpr)
    threshold_card = {
        "threshold": threshold, "decision": "score >= threshold", "target_fpr": config.target_fpr,
        "benign_human_validation_records": len(benign),
        "benign_human_refused": int(np.count_nonzero(benign >= threshold)),
        "observed_fpr": float(np.mean(benign >= threshold)),
    }
    require(threshold_card["observed_fpr"] <= config.target_fpr + 1e-12, "FPR target failed.")
    reports = validation_report(split.validation, scores, threshold)
    model_path = run_dir / "guardrail.development.joblib"
    joblib.dump(model, model_path)
    model_digest = sha256_file(model_path)
    card = {
        "name": "initial-guardrail", "status": "development", "evidence_label": "verified",
        "verification_scope": "VM development/calibration validation",
        "run_id": run_id, "environment": environment, "dataset": data.metadata,
        "configuration": {
            "seed": config.seed, "validation_fraction": config.validation_fraction,
            "target_fpr": config.target_fpr, "target": "toxicity OR jailbreaking",
            "model": model_config, "model_selection": selection,
            "deduplication": "exact text; exclude train/test overlaps; prefer human annotation then stable ID",
        },
        "split_summary": split.summary, "threshold": threshold_card,
        "validation_metrics": reports, "model_selection": selection,
        "runtime": {**fit_stats, "model_bytes": model_path.stat().st_size},
        "model_sha256": model_digest, "source": provenance,
        "score_semantics": "estimated binary unsafe probability; not validated harm severity",
        "limitations": [
            "Development/calibration results, not independent final-test or production evidence.",
            "No canary or error-correlation validation.",
            "Class weighting does not establish probability calibration.",
            "Historical project-wide official-test exposure is unresolved; this operation uses no test labels.",
        ],
    }
    write_json(run_dir / "model-card.json", card)
    write_json(run_dir / "validation-metrics.json", {"metrics": reports})
    write_json(run_dir / "threshold.json", threshold_card)
    with (run_dir / "validation-scores.jsonl").open("x") as handle:
        for prompt_id, score in zip(split.validation["prompt_id"], scores):
            handle.write(canonical({
                "prompt_id": prompt_id, "guardrail_score": float(score),
                "decision": "refuse" if score >= threshold else "allow", "threshold": threshold,
                "model_sha256": model_digest, "timestamp": timestamp, "split": "validation",
            }) + "\n")
    probe = split.validation["user_input"].iloc[:100]
    require(len(probe) == 100, "Need 100 real validation records for reload checks.")
    np.testing.assert_allclose(score_model(joblib.load(model_path), probe),
                               score_model(model, probe), rtol=0, atol=1e-12)
    repeated, _ = fit_model(model_config, split.fit, config.seed)
    np.testing.assert_allclose(score_model(repeated, split.validation["user_input"]), scores,
                               rtol=0, atol=1e-12)
    del repeated
    evidence("development_verified", {
        "model_sha256": model_digest, "threshold": threshold_card, "metrics": reports,
        "reload_records": len(probe), "repeated_fit_records": len(scores),
        "official_test_labels_read": False,
    })
    summary = {
        "run_id": run_id, "results_directory": str(run_dir), "model_sha256": model_digest,
        "partition_sha256": split.digest, "threshold": threshold,
        "human_validation": reports["human_annotated"],
        "final_test_requested": False, "freeze_requested": False, "evidence": evidence.verify(),
    }
    write_json(run_dir / "run-summary.json", summary)
    return TrainingResult(model, split, scores, threshold, card, run_dir, evidence, summary)

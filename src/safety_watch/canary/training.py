"""Fixed fitting-only selection, development evaluation and reproducibility evidence."""
from pathlib import Path
import joblib
import numpy as np
from safety_watch.classifier.artifacts import load_guardrail, card_digest
from safety_watch.classifier.data import load_data, prepare_split
from safety_watch.classifier.evidence import new_operation
from safety_watch.classifier.io import require, sha256_file, write_json, canonical, source_snapshot
from safety_watch.classifier.metrics import choose_threshold, validation_report
from safety_watch.classifier.model import score_model
from safety_watch.classifier.runtime import check_runtime
from safety_watch.classifier.selection import select_model
from safety_watch.classifier.training import TrainingResult
from .model import word_defaults, fit_canary, error_report

PARTITION_SHA256 = "cd5e3febd5e3025cf905214a1369344f5f22b9613fe35a2c577842eacf037c6d"

def train_canary(runtime_root="/runtime", project_root="/workspace"):
    runtime_root, project_root = Path(runtime_root), Path(project_root)
    guardrail = load_guardrail(runtime_root / "models")
    environment = check_runtime(guardrail.card["environment"])
    run_id, timestamp, run_dir, evidence = new_operation(runtime_root, "canary-train")
    source = source_snapshot(project_root)
    source["canary_sha256"] = {
        str(p.relative_to(project_root)): sha256_file(p)
        for p in sorted((project_root / "src/safety_watch/canary").glob("*.py"))
    }
    protocol = project_root / "docs/CANARY-PROTOCOL.md"
    source["protocol_sha256"] = sha256_file(protocol)
    evidence("canary_protocol_locked", {"source": source, "environment": environment,
                                      "guardrail_sha256": guardrail.digest})
    data = load_data(runtime_root / "data", evidence)
    require(data.metadata == guardrail.card["dataset"], "Guardrail/canary dataset provenance differs.")
    split = prepare_split(data, run_dir, 42, 0.20, evidence)
    require(split.digest == PARTITION_SHA256 == guardrail.card["split_summary"]["partitions_sha256"],
            "Guardrail/canary partitions differ.")
    config, selection = select_model(split, word_defaults(), run_dir, 42, 0.05, True, evidence)
    model, stats = fit_canary(config, split.fit)
    scores = score_model(model, split.validation["user_input"])
    human = split.validation["human_annotation"].to_numpy(dtype=bool)
    labels = split.validation["unsafe"].to_numpy(dtype=int)
    benign = scores[human & (labels == 0)]
    threshold = choose_threshold(benign, 0.05)
    require(np.mean(benign >= threshold) <= 0.05, "Canary threshold misses empirical FPR budget.")
    threshold_card = {
        "threshold": threshold, "decision": "score >= threshold (diagnostic only)",
        "target_fpr": 0.05, "benign_human_validation_records": len(benign),
        "benign_human_refused": int(np.sum(benign >= threshold)),
        "observed_fpr": float(np.mean(benign >= threshold)),
    }
    reports = validation_report(split.validation, scores, threshold)
    guardrail_scores = guardrail.score(split.validation["user_input"])
    correlation = error_report(split.validation, guardrail_scores, scores, guardrail.threshold, threshold)
    path = run_dir / "canary.development.joblib"
    joblib.dump(model, path)
    digest = sha256_file(path)
    probe = split.validation["user_input"].iloc[:100]
    np.testing.assert_allclose(score_model(joblib.load(path), probe), scores[:100], rtol=0, atol=1e-12)
    repeated, _ = fit_canary(config, split.fit)
    np.testing.assert_allclose(score_model(repeated, split.validation["user_input"]), scores, rtol=0, atol=1e-12)
    card = {
        "name": "word-canary-v1", "status": "development", "evidence_label": "verified",
        "role": "reference signal only; never changes guardrail decisions",
        "feature_family": "word TF-IDF 1-2 n-grams", "run_id": run_id,
        "environment": environment, "dataset": data.metadata, "source": source,
        "configuration": {"seed": 42, "validation_fraction": 0.20, "target_fpr": 0.05,
                          "target": "toxicity OR jailbreaking", "model": config},
        "model_selection": selection, "split_summary": split.summary,
        "threshold": threshold_card, "validation_metrics": reports, "error_correlation": correlation,
        "paired_guardrail_sha256": guardrail.digest, "model_sha256": digest,
        "runtime": {**stats, "model_bytes": path.stat().st_size},
        "score_semantics": "Weighted logistic unsafe estimate; probability calibration unverified.",
        "limitations": [
            "Development threshold-calibration cohort; not independent test or production evidence.",
            "Different feature families share the logistic classifier algorithm; common-mode failure remains.",
            "Historical official-test exposure unresolved; this operation reads no test labels.",
            "No alarm, KS, repair, or monitor performance is established by classifier evaluation.",
        ],
    }
    write_json(run_dir / "model-card.json", card)
    write_json(run_dir / "error-correlation.json", correlation)
    with (run_dir / "validation-scores.jsonl").open("x") as handle:
        for prompt_id, g, c in zip(split.validation["prompt_id"], guardrail_scores, scores):
            handle.write(canonical({
                "prompt_id": prompt_id, "guardrail_score": float(g), "canary_score": float(c),
                "timestamp": timestamp, "split": "validation",
            }) + "\n")
    evidence("canary_development_verified", {
        "model_sha256": digest, "threshold": threshold_card, "metrics": reports,
        "error_correlation": correlation, "reload_records": len(probe),
        "repeated_fit_records": len(scores), "official_test_labels_read": False,
    })
    summary = {
        "run_id": run_id, "run_directory": str(run_dir), "model_sha256": digest,
        "threshold": threshold, "human_validation": reports["human_annotated"],
        "error_correlation": correlation["human_annotated"],
        "partition_sha256": split.digest, "evidence": evidence.verify(),
    }
    write_json(run_dir / "run-summary.json", summary)
    write_json(run_dir / "run-complete.json", {
        "card_sha256": card_digest(card), "model_sha256": digest,
        "scores_sha256": sha256_file(run_dir / "validation-scores.jsonl"),
        "partitions_sha256": split.digest,
    })
    return TrainingResult(model, split, scores, threshold, card, run_dir, evidence, summary)

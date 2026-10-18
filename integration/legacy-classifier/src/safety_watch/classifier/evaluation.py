"""Re-evaluate development runs; official test requires an explicit protected action."""
from dataclasses import dataclass
from pathlib import Path
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
from .io import require, sha256_file, write_json
from .runtime import check_runtime
from .data import load_data, prepare_split, REQUIRED_COLUMNS
from .model import score_model
from .metrics import validation_report
from .evidence import new_operation

@dataclass(repr=False)
class EvaluatedRun:
    model: object
    card: dict
    split: object
    scores: np.ndarray
    metrics: dict

def read_development_run(run_dir, runtime_root, evidence=None):
    run_dir, runtime_root = Path(run_dir), Path(runtime_root)
    card = json.loads((run_dir / "model-card.json").read_text())
    check_runtime(card["environment"])
    model_path = run_dir / "guardrail.development.joblib"
    require(sha256_file(model_path) == card["model_sha256"], "Development model digest mismatch.")
    require(sha256_file(run_dir / "partitions.json") == card["split_summary"]["partitions_sha256"],
            "Development partition manifest mismatch.")
    data = load_data(runtime_root / "data", evidence)
    for field in ("revision", "train_sha256", "test_sha256"):
        require(data.metadata[field] == card["dataset"][field], "Saved dataset metadata mismatch: " + field)
    with tempfile.TemporaryDirectory(prefix="classifier-validation-", dir=runtime_root / "results") as temporary:
        split = prepare_split(data, temporary, card["configuration"]["seed"],
                              card["configuration"]["validation_fraction"], evidence)
    require(split.digest == card["split_summary"]["partitions_sha256"], "Saved split cannot be reproduced.")
    # Only trusted, digest-verified local artifacts are loaded.
    model = joblib.load(model_path)
    scores = score_model(model, split.validation["user_input"])
    saved_scores = run_dir / "validation-scores.jsonl"
    if saved_scores.is_file():
        records = [json.loads(line) for line in saved_scores.read_text().splitlines()]
        require([row["prompt_id"] for row in records] == split.validation["prompt_id"].tolist(),
                "Saved score IDs/order mismatch.")
        np.testing.assert_allclose(scores, [row["guardrail_score"] for row in records], rtol=0, atol=1e-12)
        if evidence:
            evidence("saved_scores_verified", {"records": len(scores), "score_sha256": sha256_file(saved_scores)})
    reports = validation_report(split.validation, scores, card["threshold"]["threshold"])
    for subset in ("human_annotated", "full_validation_supplementary"):
        for key, expected in card["validation_metrics"][subset].items():
            require(abs(reports[subset][key] - expected) <= 1e-12, "Saved validation metric mismatch: " + key)
    return EvaluatedRun(model, card, split, scores, reports)

def evaluate_run(run_dir, runtime_root):
    runtime_root = Path(runtime_root)
    _, _, result_dir, evidence = new_operation(runtime_root, "evaluate")
    checked = read_development_run(run_dir, runtime_root, evidence)
    report = {"split": "validation", "model_sha256": checked.card["model_sha256"],
              "threshold": checked.card["threshold"]["threshold"], "metrics": checked.metrics}
    write_json(result_dir / "evaluation.json", report)
    evidence("validation_evaluated", report)
    return {**report, "results_directory": str(result_dir), "evidence": evidence.verify()}

def evaluate_final(runtime_root, model_dir=None, confirmed=False):
    require(confirmed, "Official-test scoring requires the explicit --final-test flag.")
    from .artifacts import load_guardrail
    runtime_root = Path(runtime_root)
    saved = load_guardrail(model_dir or runtime_root / "models")
    _, _, result_dir, evidence = new_operation(runtime_root, "final-evaluate")
    data = load_data(runtime_root / "data", evidence)  # hashes test; loads training labels only
    for field in ("train_sha256", "test_sha256"):
        require(data.metadata[field] == saved.card["dataset"][field], "Frozen data digest mismatch.")
    receipt = runtime_root / "results/guardrail-final-evaluation.json"
    claim = runtime_root / "results/guardrail-final-evaluation.started.json"
    binding = {"model_sha256": saved.digest, "test_sha256": data.metadata["test_sha256"],
               "threshold": saved.threshold}
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        require(all(cached[key] == value for key, value in binding.items()),
                "Official test was used with different artifacts; no new scoring allowed.")
        evidence("final_cached", binding)
        return cached
    require(not claim.exists(), "Final evaluation already started; inspect evidence before recovery.")
    with claim.open("x") as handle:
        json.dump(binding, handle, sort_keys=True)
        handle.write("\n")
    evidence("final_started", binding)
    final = pd.read_csv(data.test_path, usecols=REQUIRED_COLUMNS,
                        dtype={"conv_id": "string", "user_input": "string"})
    require(not final[REQUIRED_COLUMNS].isna().any().any(), "Missing final fields.")
    require(final["user_input"].str.strip().str.len().gt(0).all(), "Empty final prompts.")
    for key in ("toxicity", "jailbreaking"):
        require(final[key].isin([0, 1]).all(), "Invalid final labels.")
    final["human_annotation"] = final["human_annotation"].astype(str).map({"True": True, "False": False})
    require(final["human_annotation"].notna().all(), "Invalid final annotation flag.")
    final["unsafe"] = (final["toxicity"].eq(1) | final["jailbreaking"].eq(1)).astype(int)
    reports = validation_report(final, saved.score(final["user_input"]), saved.threshold)
    report = {**binding, "metrics": {"human_annotated": reports["human_annotated"],
              "full_test_supplementary": reports["full_validation_supplementary"]}}
    write_json(receipt, report)
    write_json(result_dir / "evaluation.json", report)
    evidence("final_completed", report)
    return report

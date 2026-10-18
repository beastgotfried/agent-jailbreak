"""Canary persistence preserves the existing frozen guardrail and rejects replacement."""
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import joblib
import numpy as np
from safety_watch.classifier.artifacts import load_guardrail, card_digest
from safety_watch.classifier.data import load_data, prepare_split
from safety_watch.classifier.evidence import new_operation
from safety_watch.classifier.io import require, sha256_file, write_json, canonical
from safety_watch.classifier.metrics import validation_report, choose_threshold
from safety_watch.classifier.model import score_model
from safety_watch.classifier.runtime import check_runtime
from .model import error_report
from .training import PARTITION_SHA256

@dataclass(repr=False)
class Canary:
    model: object
    card: dict
    digest: str
    threshold: float

    def score(self, texts):
        require(not isinstance(texts, str), "Pass a sequence of prompts, not a single string.")
        texts = list(texts)
        require(all(isinstance(t, str) and t.strip() for t in texts), "Scoring inputs must be nonempty text.")
        return score_model(self.model, texts) if texts else np.asarray([], dtype=float)

def load_canary(model_dir):
    directory = Path(model_dir)
    marker = json.loads((directory / "canary.freeze.json").read_text())
    card = json.loads((directory / "model-cards.json").read_text())["canary"]
    require(card["status"] == "frozen", "Canary is not frozen.")
    require(card_digest(card) == marker["canary_card_sha256"], "Canary card/threshold integrity mismatch.")
    require(card["model_sha256"] == marker["model_sha256"], "Canary model/card digest mismatch.")
    require(card["paired_guardrail_sha256"] == marker["guardrail_sha256"], "Canary pair binding mismatch.")
    require(card["configuration"]["model"]["analyzer"] == "word" and
            tuple(card["configuration"]["model"]["ngram_range"]) == (1, 2), "Canary feature family mismatch.")
    check_runtime(card["environment"])
    threshold = float(card["threshold"]["threshold"])
    require(np.isfinite(threshold) and 0 <= threshold <= np.nextafter(1.0, np.inf),
            "Invalid canary threshold.")
    require(marker["threshold"] == threshold, "Canary threshold marker mismatch.")
    path = directory / "canary.joblib"
    require(sha256_file(path) == marker["model_sha256"], "Canary model digest mismatch.")
    return Canary(joblib.load(path), card, marker["model_sha256"], threshold)

def read_canary_run(run_dir, runtime_root, destination, guardrail, evidence):
    run_dir = Path(run_dir)
    marker = json.loads((run_dir / "run-complete.json").read_text())
    card = json.loads((run_dir / "model-card.json").read_text())
    require(card_digest(card) == marker["card_sha256"], "Canary development card integrity mismatch.")
    require(card["model_sha256"] == marker["model_sha256"] ==
            sha256_file(run_dir / "canary.development.joblib"), "Canary development model digest mismatch.")
    require(sha256_file(run_dir / "validation-scores.jsonl") == marker["scores_sha256"],
            "Canary development scores integrity mismatch.")
    check_runtime(card["environment"])
    require(guardrail.digest == card["paired_guardrail_sha256"], "Canary was verified with a different guardrail.")
    data = load_data(Path(runtime_root) / "data", evidence)
    require(data.metadata == card["dataset"] == guardrail.card["dataset"], "Pair dataset provenance differs.")
    split = prepare_split(data, destination, 42, 0.20, evidence)
    require(split.digest == marker["partitions_sha256"] == PARTITION_SHA256 ==
            sha256_file(run_dir / "partitions.json"), "Canary partition integrity mismatch.")
    model = joblib.load(run_dir / "canary.development.joblib")
    scores = score_model(model, split.validation["user_input"])
    archived = [json.loads(line) for line in (run_dir / "validation-scores.jsonl").read_text().splitlines()]
    require([row["prompt_id"] for row in archived] == split.validation["prompt_id"].tolist(),
            "Canary saved score IDs differ.")
    np.testing.assert_allclose(scores, [row["canary_score"] for row in archived], rtol=0, atol=1e-12)
    gscores = guardrail.score(split.validation["user_input"])
    np.testing.assert_allclose(gscores, [row["guardrail_score"] for row in archived], rtol=0, atol=1e-12)
    threshold = card["threshold"]["threshold"]
    human = split.validation["human_annotation"].to_numpy(dtype=bool)
    y = split.validation["unsafe"].to_numpy(dtype=int)
    require(choose_threshold(scores[human & (y == 0)], 0.05) == threshold, "Canary threshold differs.")
    require(validation_report(split.validation, scores, threshold) == card["validation_metrics"],
            "Canary validation metrics differ.")
    require(error_report(split.validation, gscores, scores, guardrail.threshold, threshold) ==
            card["error_correlation"], "Canary pair error report differs.")
    return card, split, scores

def export_canary(run_dir, runtime_root="/runtime", model_dir=None):
    runtime_root = Path(runtime_root)
    directory = Path(model_dir) if model_dir is not None else runtime_root / "models"
    guardrail = load_guardrail(directory)
    _, timestamp, result_dir, evidence = new_operation(runtime_root, "canary-export")
    card, split, scores = read_canary_run(run_dir, runtime_root, result_dir, guardrail, evidence)
    digest = card["model_sha256"]
    marker_path = directory / "canary.freeze.json"
    if marker_path.exists():
        saved = load_canary(directory)
        require(saved.digest == digest and saved.threshold == card["threshold"]["threshold"],
                "A different canary is already frozen; use explicit versioning.")
        evidence("existing_canary_verified", {"model_sha256": digest, "records": len(scores)})
        return {"existing": True, "model_sha256": digest, "threshold": saved.threshold,
                "model_directory": str(directory), "evidence": evidence.verify()}
    model_path, cards_path = directory / "canary.joblib", directory / "model-cards.json"
    cards_bytes = cards_path.read_bytes()
    cards = json.loads(cards_bytes)
    require(not model_path.exists() and "canary" not in cards,
            "Partial canary export exists; inspect before recovery.")
    lock = directory / ".canary-export.lock"
    with lock.open("x") as handle:
        handle.write(canonical({"model_sha256": digest, "source_run": str(run_dir)}) + "\n")
    with model_path.open("xb") as target, (Path(run_dir) / "canary.development.joblib").open("rb") as source:
        shutil.copyfileobj(source, target)
    require(sha256_file(model_path) == digest, "Canary copy digest differs.")
    frozen_card = {**card, "status": "frozen", "source_run_id": card["run_id"],
                   "exported_utc": timestamp, "export_verification": {
                       "validation_records": len(scores), "copied_without_refit": True}}
    require(cards_path.read_bytes() == cards_bytes, "Model cards changed concurrently; inspect partial export.")
    cards["canary"] = frozen_card
    write_json(cards_path, cards)
    write_json(marker_path, {
        "model_sha256": digest, "canary_card_sha256": card_digest(frozen_card),
        "guardrail_sha256": guardrail.digest, "threshold": card["threshold"]["threshold"],
    })
    saved = load_canary(directory)
    np.testing.assert_allclose(saved.score(split.validation["user_input"].iloc[:100]),
                               scores[:100], rtol=0, atol=1e-12)
    require(load_guardrail(directory).digest == guardrail.digest, "Guardrail changed during canary export.")
    lock.unlink()
    evidence("canary_frozen_and_verified", {
        "model_sha256": digest, "guardrail_sha256": guardrail.digest,
        "validation_records": len(scores), "reload_records": 100, "official_test_labels_read": False})
    report = {"existing": False, "model_sha256": digest, "threshold": saved.threshold,
              "source_run_id": card["run_id"], "model_directory": str(directory),
              "evidence": evidence.verify()}
    write_json(result_dir / "export-summary.json", report)
    return report

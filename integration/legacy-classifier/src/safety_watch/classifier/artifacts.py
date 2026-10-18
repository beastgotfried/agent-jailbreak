"""Verified export/load with threshold integrity, no refit and no silent replacement."""
from dataclasses import dataclass
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import joblib
import numpy as np
from .runtime import check_runtime
from .io import canonical, require, sha256_file, write_json
from .model import score_model
from .evidence import new_operation

def card_digest(card):
    return hashlib.sha256(canonical(card).encode()).hexdigest()

@dataclass
class Guardrail:
    model: object
    card: dict
    digest: str
    threshold: float

    def score(self, texts):
        require(not isinstance(texts, str), "Pass a sequence of prompts, not a single string.")
        texts = list(texts)
        if not texts:
            return np.asarray([], dtype=float)
        require(all(isinstance(text, str) and text.strip() for text in texts),
                "Scoring inputs must be nonempty text.")
        return score_model(self.model, texts)

    def predict(self, prompt_ids, texts):
        prompt_ids, texts = list(prompt_ids), list(texts)
        require(len(prompt_ids) == len(texts), "Prompt IDs and texts must have equal length.")
        require(all(isinstance(value, str) and value for value in prompt_ids), "Prompt IDs must be nonempty strings.")
        scores = self.score(texts)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return [{
            "prompt_id": prompt_id, "guardrail_score": float(score),
            "decision": "refuse" if score >= self.threshold else "allow",
            "threshold": self.threshold, "model_sha256": self.digest, "timestamp": timestamp,
        } for prompt_id, score in zip(prompt_ids, scores)]

def load_guardrail(model_dir):
    directory = Path(model_dir)
    marker = json.loads((directory / "guardrail.freeze.json").read_text())
    cards = json.loads((directory / "model-cards.json").read_text())
    card = cards["guardrail"]
    require(card["status"] == "frozen", "Guardrail model card is not frozen.")
    require(card_digest(card) == marker["guardrail_card_sha256"], "Guardrail card/threshold integrity mismatch.")
    require(card["model_sha256"] == marker["model_sha256"], "Model/card digest mismatch.")
    check_runtime(card["environment"])
    threshold = float(card["threshold"]["threshold"])
    require(np.isfinite(threshold) and 0 <= threshold <= np.nextafter(1.0, np.inf), "Invalid saved threshold.")
    path = directory / "guardrail.joblib"
    require(sha256_file(path) == marker["model_sha256"], "Frozen model digest mismatch.")
    require(marker["threshold"] == threshold, "Completion marker threshold mismatch.")
    model = joblib.load(path)
    return Guardrail(model, card, marker["model_sha256"], threshold)

def export_run(run_dir, runtime_root, model_dir=None):
    from .evaluation import read_development_run
    runtime_root, run_dir = Path(runtime_root), Path(run_dir)
    directory = Path(model_dir) if model_dir is not None else runtime_root / "models"
    directory.mkdir(parents=True, exist_ok=True)
    _, timestamp, result_dir, evidence = new_operation(runtime_root, "export")
    checked = read_development_run(run_dir, runtime_root, evidence)
    digest = checked.card["model_sha256"]
    marker_path = directory / "guardrail.freeze.json"
    if marker_path.exists():
        saved = load_guardrail(directory)
        require(saved.digest == digest and saved.threshold == checked.card["threshold"]["threshold"],
                "A different guardrail is already saved; replacement requires explicit versioning.")
        evidence("existing_export_verified", {"model_sha256": digest, "threshold": saved.threshold})
        return {"model_directory": str(directory), "model_sha256": digest,
                "threshold": saved.threshold, "existing": True, "evidence": evidence.verify()}
    model_path = directory / "guardrail.joblib"
    cards_path = directory / "model-cards.json"
    require(not model_path.exists(), "Existing guardrail without completion marker; inspect before recovery.")
    cards = json.loads(cards_path.read_text()) if cards_path.exists() else {}
    require(isinstance(cards, dict) and "guardrail" not in cards, "Existing guardrail card needs explicit reconciliation.")
    lock = directory / ".guardrail-export.lock"
    # Exclusive lock is retained after an interrupted write, requiring evidence review.
    with lock.open("x") as handle:
        handle.write(canonical({"model_sha256": digest, "source_run": str(run_dir)}) + "\n")
    card = {**checked.card, "status": "frozen", "exported_utc": timestamp,
            "source_run_id": checked.card["run_id"], "export_verification": {
                "records": len(checked.split.validation), "validation_metrics_reproduced": True,
                "source_model_copied_without_refit": True}}
    with model_path.open("xb") as destination, (run_dir / "guardrail.development.joblib").open("rb") as source:
        shutil.copyfileobj(source, destination)
    require(sha256_file(model_path) == digest, "Saved model copy digest mismatch.")
    cards["guardrail"] = card
    write_json(cards_path, cards)
    copied_model = joblib.load(model_path)
    np.testing.assert_allclose(score_model(copied_model, checked.split.validation["user_input"].iloc[:100]),
                               checked.scores[:100], rtol=0, atol=1e-12)
    del copied_model
    write_json(marker_path, {"model_sha256": digest, "guardrail_card_sha256": card_digest(card),
                             "threshold": card["threshold"]["threshold"]})
    saved = load_guardrail(directory)
    np.testing.assert_allclose(saved.score(checked.split.validation["user_input"].iloc[:100]),
                               checked.scores[:100], rtol=0, atol=1e-12)
    lock.unlink()
    evidence("guardrail_saved_and_verified", {
        "model_sha256": digest, "threshold": saved.threshold,
        "validation_records": len(checked.scores), "reload_records": 100, "official_test_labels_read": False,
    })
    report = {"model_directory": str(directory), "model_sha256": digest, "threshold": saved.threshold,
              "existing": False, "source_run_id": card["source_run_id"],
              "validation_metrics": checked.metrics, "evidence": evidence.verify()}
    write_json(result_dir / "export-summary.json", report)
    return report

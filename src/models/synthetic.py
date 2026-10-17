"""Bounded native classifier fit for the separate synthetic study.

Only train and validation text enter this module. Original model files remain
unchanged. Scores are not asserted to be calibrated harm probabilities.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src.data.synthetic import digest, file_hash, load_rows, native_only, write_json
from src.models.train_classifiers import binary_metrics
from src.response.conformal import valid_hash, valid_score


@dataclass(frozen=True)
class BinaryPolicy:
    threshold: float
    model_pair_id: str
    calibration_ids: tuple[str, ...]

    def __post_init__(self):
        if not valid_score(self.threshold) or not valid_hash(self.model_pair_id):
            raise ValueError("invalid_binary_policy")
        if not self.calibration_ids or len(set(self.calibration_ids)) != len(self.calibration_ids):
            raise ValueError("invalid_binary_policy_identities")
        if not all(valid_hash(v) for v in self.calibration_ids):
            raise ValueError("invalid_binary_policy_identities")

    @property
    def policy_id(self):
        return digest({"method": "fixed_binary_threshold", "threshold": self.threshold,
                       "model_pair_id": self.model_pair_id, "calibration_ids": self.calibration_ids})

    def prediction_set(self, score):
        if not valid_score(score):
            raise ValueError("invalid_prediction_score")
        return (int(score >= self.threshold),)

    def decision(self, score):
        return "refuse" if self.prediction_set(score) == (1,) else "allow"

    def to_dict(self):
        return {"method": "fixed_binary_threshold", "threshold": self.threshold,
                "model_pair_id": self.model_pair_id, "calibration_ids": list(self.calibration_ids),
                "policy_id": self.policy_id, "conformal_coverage_claim": False}


def metrics(y, scores, threshold):
    result = binary_metrics(np.asarray(y), np.asarray(scores), threshold)
    c = result["confusion"]
    result["accuracy"] = (c["tn"] + c["tp"]) / result["n"] if result["n"] else None
    result["balanced_accuracy"] = .5 * (result["recall"] + 1 - result["fpr"]) if result["fpr"] is not None else None
    result["threshold"] = float(threshold)
    return result


def select_threshold(y, scores, max_fpr=.05):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.shape != scores.shape or y.ndim != 1 or set(y.tolist()) != {0, 1}:
        raise ValueError("invalid_threshold_labels")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("invalid_threshold_scores")
    unique = np.unique(scores)
    candidates = np.unique(np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2, [1.0])))
    positive, negative = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    tpr = 1 - np.searchsorted(positive, candidates, side="left") / len(positive)
    fpr = 1 - np.searchsorted(negative, candidates, side="left") / len(negative)
    balanced = (tpr + 1 - fpr) / 2
    eligible = np.flatnonzero(fpr <= max_fpr + 1e-12)
    if not len(eligible):
        raise ValueError("no_valid_threshold_under_fpr_cap")
    best = max(eligible, key=lambda i: (float(balanced[i]), float(tpr[i]), float(candidates[i])))
    return float(candidates[best])


def guard_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=100000, sublinear_tf=True, dtype=np.float32)),
    ])


def probabilities(model, x):
    estimator, calibrator = model["estimator"], model.get("calibrator")
    if calibrator is not None:
        return calibrator.predict_proba(estimator.decision_function(x).reshape(-1, 1))[:, 1]
    return estimator.predict_proba(x)[:, list(estimator.classes_).index(1)]


def logistic_candidate(x, y, validation_features, validation_labels, threshold_ids):
    estimator = LogisticRegression(C=4, max_iter=500, class_weight="balanced", random_state=42)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(x, y)
    except ConvergenceWarning:
        return {"method": "tfidf_logistic_C4", "valid": False, "code": "nonconverged_no_retry"}, None
    model = {"estimator": estimator, "calibrator": None}
    values = probabilities(model, validation_features)
    threshold = select_threshold(validation_labels[threshold_ids], values[threshold_ids])
    quality = metrics(validation_labels[threshold_ids], values[threshold_ids], threshold)
    return ({"method": "tfidf_logistic_C4", "valid": True, "validation": quality},
            (quality["balanced_accuracy"], quality["recall"], "tfidf_logistic_C4", model, threshold))


def train(data, output, *, source_commit):
    native_only()
    start = time.perf_counter()
    data, output = Path(data), Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if file_hash(data / name) != expected:
            raise ValueError("synthetic_data_hash_mismatch")
    training, validation = load_rows(data / "train.jsonl"), load_rows(data / "validation.jsonl")
    if len(training) != 70000 or len(validation) != 10000:
        raise ValueError("invalid_synthetic_training_counts")
    if {r.family for r in training} & {r.family for r in validation}:
        raise ValueError("synthetic_fit_validation_overlap")
    texts, y = [r.text for r in training], np.asarray([r.label for r in training])
    vx, vy = [r.text for r in validation], np.asarray([r.label for r in validation])
    threshold_ids = np.asarray([i for i, r in enumerate(validation) if r.role == "threshold"])
    calibration_ids = np.asarray([i for i, r in enumerate(validation) if r.role == "fallback_calibration"])
    features = guard_features()
    x, v = features.fit_transform(texts), features.transform(vx)
    attempt, candidate = logistic_candidate(x, y, v, vy, threshold_ids)
    attempts = [attempt]
    candidates = [candidate] if candidate is not None else []
    if not candidates or candidates[0][0] < .90:
        estimator = LinearSVC(C=1, max_iter=5000, class_weight="balanced", random_state=42, dual="auto")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(x, y)
                calibrator = LogisticRegression(C=1e6, max_iter=200, random_state=42).fit(
                    estimator.decision_function(v[calibration_ids]).reshape(-1, 1), vy[calibration_ids])
            model = {"estimator": estimator, "calibrator": calibrator}
            values = probabilities(model, v)
            threshold = select_threshold(vy[threshold_ids], values[threshold_ids])
            quality = metrics(vy[threshold_ids], values[threshold_ids], threshold)
            attempts.append({"method": "tfidf_linear_svc_C1", "valid": True, "validation": quality})
            candidates.append((quality["balanced_accuracy"], quality["recall"], "tfidf_linear_svc_C1", model, threshold))
        except ConvergenceWarning:
            attempts.append({"method": "tfidf_linear_svc_C1", "valid": False, "code": "nonconverged_no_retry"})
    write_json(output / "attempts.json", attempts)
    if not candidates:
        raise RuntimeError("both_model_approaches_failed_no_third_attempt")
    chosen = max(candidates, key=lambda item: (item[0], item[1]))
    guard = {"features": features, **chosen[3]}
    del x, v, candidates
    gc.collect()
    canary_features = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                     max_features=100000, sublinear_tf=True, dtype=np.float32)
    cx = canary_features.fit_transform(texts)
    canary_estimator = LogisticRegression(C=1, max_iter=500, class_weight="balanced", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        canary_estimator.fit(cx, y)
    canary = {"features": canary_features, "estimator": canary_estimator, "calibrator": None}
    c_values = probabilities(canary, canary_features.transform(vx))
    c_threshold = select_threshold(vy[threshold_ids], c_values[threshold_ids])
    artifact = output / "models.joblib"
    joblib.dump({"guardrail": guard, "canary": canary}, artifact, compress=3)
    artifact_sha = file_hash(artifact)
    binding = digest({"models_sha256": artifact_sha, "guardrail_threshold": chosen[4], "canary_threshold": c_threshold,
                      "data_manifest_sha256": file_hash(data / "manifest.json")})
    policy = BinaryPolicy(chosen[4], binding, tuple(sorted(validation[i].prompt_id for i in threshold_ids)))
    metadata = {"source_commit": source_commit, "architecture": platform.machine(), "sklearn": sklearn.__version__,
        "models_sha256": artifact_sha, "model_pair_id": binding, "data_manifest_sha256": file_hash(data / "manifest.json"),
        "train_rows": len(training), "validation_rows": len(validation), "test_rows_read_for_fit": 0,
        "model_inputs": "text_only", "guardrail_method": chosen[2], "attempts": attempts,
        "guardrail_threshold": chosen[4], "canary_threshold": c_threshold,
        "canary_validation": metrics(vy[threshold_ids], c_values[threshold_ids], c_threshold),
        "policy": policy.to_dict(), "fit_seconds": time.perf_counter() - start,
        "validation_roles": manifest["quality"]["roles"],
        "limitations": ["Synthetic family holdout, not real-world deployment qualification.",
                        "Both models share training data; independence is not assumed.",
                        "Binary policy has no conformal coverage guarantee."]}
    write_json(output / "model-cards.json", metadata)
    write_json(output / "pre-evaluation-freeze.json", {"test_scored": False, "source_commit": source_commit,
        "files": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
        "data_files": manifest["files"], "catalog_sha256": manifest["catalog_sha256"]})
    for path in output.iterdir():
        path.chmod(0o400)
    print(json.dumps({"selected": chosen[2], "validation_balanced_accuracy": chosen[0], "test_scored": False,
                      "seconds": metadata["fit_seconds"]}))
    return metadata


class SyntheticBackend:
    def __init__(self, models, expected_cards_sha=None):
        self.directory = Path(models)
        cards_file = self.directory / "model-cards.json"
        if expected_cards_sha is not None and file_hash(cards_file) != expected_cards_sha:
            raise ValueError("synthetic_model_cards_changed")
        cards = json.loads(cards_file.read_text())
        if cards["sklearn"] != sklearn.__version__ or file_hash(self.directory / "models.joblib") != cards["models_sha256"]:
            raise ValueError("synthetic_model_artifact_mismatch")
        self.bundle = joblib.load(self.directory / "models.joblib")
        self.model_pair_id = cards["model_pair_id"]
        self.guardrail_threshold = cards["guardrail_threshold"]
        self.canary_threshold = cards["canary_threshold"]

    def score(self, texts):
        values = {}
        for name, model in self.bundle.items():
            values[name] = probabilities(model, model["features"].transform(texts))
        return [{"guardrail_score": float(g), "canary_score": float(c),
                 "guardrail_unsafe": bool(g >= self.guardrail_threshold),
                 "canary_unsafe": bool(c >= self.canary_threshold)}
                for g, c in zip(values["guardrail"], values["canary"])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    train(args.data, args.output, source_commit=args.source_commit)


if __name__ == "__main__":
    main()

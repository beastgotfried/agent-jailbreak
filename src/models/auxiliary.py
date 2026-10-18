"""Fit one auxiliary guardrail from human-labelled Aegis prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.data.aegis import require_native
from src.data.synthetic import digest, file_hash, write_json
from src.models.accuracy import group_weights, read_aegis
from src.models.synthetic import guard_features
from src.release.cli import load_selected


def train(development, baseline_models, output, *, protocol, head_template, source_commit):
    require_native()
    development, baseline_models, output = map(Path, (development, baseline_models, output))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    config = json.loads(Path(head_template).read_text())
    if file_hash(development / "manifest.json") != config["fixed_base_models"]["development_manifest_sha256"]:
        raise ValueError("auxiliary_data_binding_mismatch")
    manifest = json.loads((development / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if file_hash(development / name) != expected:
            raise ValueError("auxiliary_data_changed")
    rows = read_aegis(development / "train.jsonl")
    _, base = load_selected(baseline_models)
    features = guard_features()
    x = features.fit_transform([r["text"] for r in rows])
    y = np.asarray([r["label"] for r in rows])
    estimator = LogisticRegression(C=4, max_iter=500, class_weight=None, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(x, y, sample_weight=group_weights(rows) * len(rows))
    artifact = output / "models.joblib"
    joblib.dump({"guardrail": {"features": features, "estimator": estimator, "calibrator": None},
                 "canary": base.bundle["canary"]}, artifact, compress=3)
    model_sha = file_hash(artifact)
    cards = {"source_commit": source_commit, "architecture": "s390x", "sklearn": sklearn.__version__,
             "models_sha256": model_sha, "model_pair_id": digest({"models_sha256": model_sha, "protocol_sha256": file_hash(protocol)}),
             "guardrail_method": "aegis_only_tfidf_C4_auxiliary", "guardrail_threshold": .5,
             "canary_threshold": base.canary_threshold, "train_rows": len(rows),
             "model_inputs": "text_only", "validation_or_test_rows_read_for_fit": 0,
             "purpose": "Frozen auxiliary score extractor, not a selected deployed guardrail. Its conventional 0.5 flag is not used by the calibration head.",
             "protocol_sha256": file_hash(protocol), "development_manifest_sha256": file_hash(development / "manifest.json")}
    write_json(output / "model-cards.json", cards)
    config["phase"] = "aegis_only_auxiliary_two_score_calibration"
    config["reason"] = "The auxiliary expert fits Aegis prompt labels. Its training records do not exactly overlap the original guardrail corpus; semantic independence is not established."
    config["parent_protocol_sha256"] = file_hash(protocol)
    config["fixed_base_models"]["mixed"] = "Frozen Aegis-only TF-IDF auxiliary, not the first-phase mixed-data model"
    config["fixed_base_models"]["mixed_models_sha256"] = model_sha
    config["fixed_base_models"]["mixed_cards_sha256"] = file_hash(output / "model-cards.json")
    write_json(output / "head-protocol.json", config)
    write_json(output / "freeze.json", {"source_commit": source_commit, "test_scored": False,
               "files": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    for path in output.iterdir():
        path.chmod(0o400)
    print(json.dumps({"auxiliary_fit": "completed", "train_rows": len(rows), "models_sha256": model_sha,
                      "test_scored": False}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "baseline-models", "output", "protocol", "head-template"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    train(args.development, args.baseline_models, args.output, protocol=args.protocol,
          head_template=args.head_template, source_commit=args.source_commit)


if __name__ == "__main__":
    main()

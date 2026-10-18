"""Two frozen text classifiers with one validation-selected calibration head."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.data.aegis import require_native
from src.data.synthetic import digest, file_hash, write_json
from src.models.accuracy import group_weights, quality, read_aegis, rejection_codes, select_threshold, synthetic_rows
from src.models.synthetic import BinaryPolicy, SyntheticBackend, probabilities
from src.release.cli import load_selected
from src.release.state import learned_state_sha256


def logit_features(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid_calibration_scores")
    if np.any((a < 0) | (a > 1)) or np.any((b < 0) | (b > 1)):
        raise ValueError("invalid_calibration_scores")
    values = np.column_stack((np.clip(a, 1e-6, 1 - 1e-6), np.clip(b, 1e-6, 1 - 1e-6)))
    return np.log(values / (1 - values))


class TwoGuardFeatures:
    """Execute both text models on every call; never use canary scores."""
    def __init__(self, original, mixed):
        self.original, self.mixed = original, mixed

    def transform(self, texts):
        a = probabilities(self.original, self.original["features"].transform(texts))
        b = probabilities(self.mixed, self.mixed["features"].transform(texts))
        return logit_features(a, b)


def split_development(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    strata = defaultdict(list)
    for group, members in groups.items():
        strata[tuple(sorted({r["label"] for r in members}))].append(group)
    fitting = set()
    for members in strata.values():
        ordered = sorted(members, key=lambda group: digest([42, "stack-role-v1", group]))
        fitting.update(ordered[:len(ordered) // 2])
    fit = [r for r in rows if r["group_id"] in fitting]
    threshold = [r for r in rows if r["group_id"] not in fitting]
    if {r["group_id"] for r in fit} & {r["group_id"] for r in threshold}:
        raise ValueError("calibration_group_overlap")
    return fit, threshold


def train(synthetic, development, baseline_models, mixed_models, output, *, protocol, source_commit):
    require_native()
    synthetic, development, baseline_models, mixed_models, output = map(Path, (synthetic, development, baseline_models, mixed_models, output))
    config = json.loads(Path(protocol).read_text())
    bindings = config["fixed_base_models"]
    for path, expected in ((mixed_models / "models.joblib", bindings["mixed_models_sha256"]),
                           (mixed_models / "model-cards.json", bindings["mixed_cards_sha256"]),
                           (development / "manifest.json", bindings["development_manifest_sha256"])):
        if file_hash(path) != expected:
            raise ValueError("calibration_input_binding_mismatch")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    start = time.perf_counter()
    sm, dm = [json.loads((p / "manifest.json").read_text()) for p in (synthetic, development)]
    for directory, manifest in ((synthetic, sm), (development, dm)):
        for name, expected in manifest["files"].items():
            if file_hash(directory / name) != expected:
                raise ValueError("calibration_data_hash_mismatch")
    sf = synthetic_rows(synthetic / "validation.jsonl", "fallback_calibration")
    sv = synthetic_rows(synthetic / "validation.jsonl", "threshold")
    af, av = split_development(read_aegis(development / "validation.jsonl"))
    fitting, validation = sf + af, sv + av
    if len(sf) != 3000 or len(sv) != 5000:
        raise ValueError("original_calibration_role_changed")
    if {r["prompt_id"] for r in fitting} & {r["prompt_id"] for r in validation}:
        raise ValueError("calibration_identity_overlap")
    roles = {"fit": [r["prompt_id"] for r in fitting], "threshold": [r["prompt_id"] for r in validation],
             "synthetic_fit_rows": len(sf), "synthetic_threshold_rows": len(sv),
             "aegis_fit_rows": len(af), "aegis_threshold_rows": len(av)}
    write_json(output / "roles.json", roles)
    _, base = load_selected(baseline_models)
    mixed = SyntheticBackend(mixed_models, bindings["mixed_cards_sha256"])
    canary_state = learned_state_sha256(base.bundle["canary"])
    if learned_state_sha256(mixed.bundle["canary"]) != canary_state:
        raise ValueError("mixed_canary_differs")
    # Persist an importable class even when this file runs through python -m.
    from src.models.calibrated import TwoGuardFeatures as PortableFeatures
    features = PortableFeatures(base.bundle["guardrail"], mixed.bundle["guardrail"])
    x = features.transform([r["text"] for r in fitting])
    v = features.transform([r["text"] for r in validation])
    original = base.bundle["guardrail"]
    old_scores = probabilities(original, original["features"].transform([r["text"] for r in validation]))
    baseline_quality = quality(validation, old_scores, base.guardrail_threshold)
    weights = group_weights(fitting) * len(fitting)
    y = np.asarray([r["label"] for r in fitting])
    heads = [
        ("two_score_logistic_C1", LogisticRegression(C=1, max_iter=500, random_state=42, class_weight=None)),
        ("two_score_hgb100_leaf7", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
            min_samples_leaf=30, max_bins=63, learning_rate=.05, l2_regularization=10,
            early_stopping=False, random_state=42)),
    ]
    records = []
    for name, head in heads:
        directory = output / name
        directory.mkdir(mode=0o700)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                head.fit(x, y, sample_weight=weights)
        except ConvergenceWarning:
            joblib.dump(head, directory / "failed-head.joblib", compress=3)
            record = {"candidate": name, "eligible": False, "rejection_codes": ["nonconverged_no_retry"]}
            write_json(directory / "attempts.json", [record])
            records.append(record)
            continue
        guard = {"features": features, "estimator": head, "calibrator": None}
        scores = probabilities(guard, v)
        threshold = select_threshold(validation, scores)
        measured = quality(validation, scores, threshold)
        reasons = rejection_codes(measured, baseline_quality)
        artifact = directory / "models.joblib"
        joblib.dump({"guardrail": guard, "canary": base.bundle["canary"]}, artifact, compress=3)
        model_sha = file_hash(artifact)
        binding = digest({"models_sha256": model_sha, "guardrail_threshold": threshold,
                          "canary_threshold": base.canary_threshold, "protocol_sha256": file_hash(protocol),
                          "role_ids_sha256": file_hash(output / "roles.json")})
        # Both head-fit and threshold rows remain separate from runtime probes.
        policy = BinaryPolicy(threshold, binding, tuple(sorted(r["prompt_id"] for r in fitting + validation)))
        record = {"candidate": name, "validation": measured, "eligible": not reasons, "rejection_codes": reasons}
        cards = {"source_commit": source_commit, "architecture": "s390x", "sklearn": sklearn.__version__,
                 "models_sha256": model_sha, "model_pair_id": binding, "guardrail_method": name,
                 "guardrail_threshold": threshold, "canary_threshold": base.canary_threshold,
                 "policy": policy.to_dict(), "validation": measured, "test_rows_read_for_fit": 0,
                 "model_inputs": "text_only", "guardrail_components": ["frozen_synthetic_guardrail", "frozen_mixed_guardrail", "calibration_head"],
                 "canary_learned_state_sha256": canary_state, "protocol_sha256": file_hash(protocol),
                 "development_manifest_sha256": file_hash(development / "manifest.json"),
                 "data_manifest_sha256": file_hash(synthetic / "manifest.json"), "calibration_roles": roles,
                 "limitations": ["Source-specific human labels.", "Synthetic calibration roles share families but not matched cases.", "Composite guardrail calls execute both base classifiers; no model-output cache."]}
        write_json(directory / "model-cards.json", cards)
        write_json(directory / "attempts.json", [record])
        write_json(directory / "pre-evaluation-freeze.json", {"test_scored": False, "source_commit": source_commit,
                   "files": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()},
                   "data_files": sm["files"], "catalog_sha256": sm["catalog_sha256"],
                   "development_files": dm["files"], "protocol_sha256": file_hash(protocol)})
        for path in directory.iterdir():
            path.chmod(0o400)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    eligible = [r for r in records if r["eligible"]]
    chosen = max(eligible, key=lambda r: (round(r["validation"]["equal_source_balanced_accuracy"], 12),
                    round(r["validation"]["mean_harmful_recall"], 12), -records.index(r))) if eligible else None
    if learned_state_sha256(base.bundle["canary"]) != canary_state:
        raise ValueError("canary_changed_during_calibration")
    result = {"source_commit": source_commit, "protocol_sha256": file_hash(protocol),
              "baseline_validation": baseline_quality, "candidates": records,
              "selected": chosen["candidate"] if chosen else None, "new_base_model_fits": 0,
              "new_calibration_head_fits": 2, "test_scored": False, "test_driven_changes": False,
              "canary_unchanged": True, "seconds": time.perf_counter() - start}
    write_json(output / "selection.json", result)
    (output / "selection.json").chmod(0o400)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("synthetic", "development", "baseline-models", "mixed-models", "output", "protocol"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    train(args.synthetic, args.development, args.baseline_models, args.mixed_models, args.output,
          protocol=args.protocol, source_commit=args.source_commit)


if __name__ == "__main__":
    main()

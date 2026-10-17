"""Bounded validation-only calibration and two fixed mixed-data TF-IDF fits."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.data.aegis import require_native
from src.data.synthetic import digest, file_hash, load_rows, write_json
from src.models.synthetic import BinaryPolicy, guard_features, metrics, probabilities
from src.release.cli import load_selected
from src.release.state import learned_state_sha256

CANDIDATES = (("tfidf_mixed_75_25", .75), ("tfidf_mixed_50_50", .5))


def read_aegis(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def synthetic_rows(path, role=None):
    return [{"text": r.text, "label": r.label, "prompt_id": r.prompt_id,
             "group_id": r.semantic_group, "source": "synthetic"}
            for r in load_rows(path) if role is None or r.role == role]


def group_weights(rows, shares=None):
    sources = sorted({r["source"] for r in rows})
    shares = {s: 1 / len(sources) for s in sources} if shares is None else shares
    if set(shares) != set(sources) or any(v <= 0 for v in shares.values()) or not np.isclose(sum(shares.values()), 1):
        raise ValueError("invalid_source_weights")
    counts = Counter((r["source"], r["label"], r["group_id"]) for r in rows)
    groups = Counter((s, label) for s, label, group in counts)
    if any(groups[s, c] < 1 for s in sources for c in (0, 1)):
        raise ValueError("missing_source_class")
    weights = np.asarray([shares[r["source"]] / (2 * groups[r["source"], r["label"]] *
                         counts[r["source"], r["label"], r["group_id"]]) for r in rows])
    return weights / weights.sum()


def quality(rows, scores, threshold):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("invalid_development_scores")
    result = {}
    weights = group_weights(rows)
    for source in sorted({r["source"] for r in rows}):
        indices = np.asarray([i for i, r in enumerate(rows) if r["source"] == source])
        y = np.asarray([rows[i]["label"] for i in indices])
        p, w = scores[indices] >= threshold, weights[indices]
        recall = float(w[(y == 1) & p].sum() / w[y == 1].sum())
        fpr = float(w[(y == 0) & p].sum() / w[y == 0].sum())
        result[source] = {"raw": metrics(y, scores[indices], threshold), "group_weighted_recall": recall,
                          "group_weighted_fpr": fpr, "group_balanced_accuracy": (recall + 1 - fpr) / 2}
    return {"sources": result,
            "equal_source_balanced_accuracy": float(np.mean([v["group_balanced_accuracy"] for v in result.values()])),
            "mean_harmful_recall": float(np.mean([v["group_weighted_recall"] for v in result.values()])),
            "threshold": float(threshold)}


def select_threshold(rows, scores):
    scores = np.asarray(scores, dtype=float)
    quality(rows, scores, .5)
    unique = np.unique(scores)
    candidates = np.unique(np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2, [1.0])))
    weights = group_weights(rows)
    balanced, recalls = np.zeros(len(candidates)), np.zeros(len(candidates))
    eligible = np.ones(len(candidates), dtype=bool)
    sources = sorted({r["source"] for r in rows})
    for source in sources:
        rates = {}
        for label in (0, 1):
            indices = np.asarray([i for i, r in enumerate(rows) if r["source"] == source and r["label"] == label])
            order = indices[np.argsort(scores[indices], kind="stable")]
            positions = np.searchsorted(scores[order], candidates, side="left")
            mass = np.concatenate(([0.0], np.cumsum(weights[order])))
            rates[label] = 1 - mass[positions] / mass[-1]
            if label == 0:
                raw_fpr = 1 - positions / len(order)
                eligible &= (raw_fpr <= .05 + 1e-12) & (rates[0] <= .05 + 1e-12)
        balanced += (rates[1] + 1 - rates[0]) / (2 * len(sources))
        recalls += rates[1] / len(sources)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise ValueError("no_valid_development_threshold")
    best = max(indices, key=lambda i: (round(float(balanced[i]), 12), round(float(recalls[i]), 12), float(candidates[i])))
    return float(candidates[best])


def rejection_codes(report, baseline):
    s, a = report["sources"]["synthetic"], report["sources"]["aegis"]
    reasons = []
    if min(s["group_balanced_accuracy"], s["raw"]["balanced_accuracy"]) < .90 - 1e-12:
        reasons.append("synthetic_validation_below_floor")
    if min(a["group_weighted_recall"], a["raw"]["recall"]) < .50 - 1e-12:
        reasons.append("aegis_validation_recall_below_floor")
    if report["equal_source_balanced_accuracy"] < baseline["equal_source_balanced_accuracy"] + .005 - 1e-12:
        reasons.append("validation_gain_below_floor")
    for source, q in report["sources"].items():
        if max(q["group_weighted_fpr"], q["raw"]["fpr"]) > .05 + 1e-12:
            reasons.append(source + "_validation_fpr_above_cap")
    return reasons


def train(synthetic, development, baseline_models, output, *, protocol, source_commit):
    require_native()
    synthetic, development, baseline_models, output = map(Path, (synthetic, development, baseline_models, output))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.perf_counter()
    sm = json.loads((synthetic / "manifest.json").read_text())
    dm = json.loads((development / "manifest.json").read_text())
    for directory, manifest in ((synthetic, sm), (development, dm)):
        for name, expected in manifest["files"].items():
            if file_hash(directory / name) != expected:
                raise ValueError("development_data_changed")
    st = synthetic_rows(synthetic / "train.jsonl")
    sv = synthetic_rows(synthetic / "validation.jsonl", "threshold")
    at, av = read_aegis(development / "train.jsonl"), read_aegis(development / "validation.jsonl")
    training, validation = st + at, sv + av
    if len(st) != 70000 or len(sv) != 5000:
        raise ValueError("original_development_roles_changed")
    if {r["prompt_id"] for r in training} & {r["prompt_id"] for r in validation}:
        raise ValueError("development_identity_overlap")
    if {(r["source"], r["group_id"]) for r in training} & {(r["source"], r["group_id"]) for r in validation}:
        raise ValueError("development_group_overlap")
    if len({r["prompt_id"] for r in validation}) != len(validation):
        raise ValueError("duplicate_calibration_identity")
    counts = Counter((r["source"], r["label"]) for r in validation)
    if any(counts[s, c] < 20 for s in ("synthetic", "aegis") for c in (0, 1)):
        raise ValueError("insufficient_development_classes")
    _, baseline = load_selected(baseline_models)
    canary_state = learned_state_sha256(baseline.bundle["canary"])
    vx = [r["text"] for r in validation]
    old_guard = baseline.bundle["guardrail"]
    baseline_scores = probabilities(old_guard, old_guard["features"].transform(vx))
    baseline_quality = quality(validation, baseline_scores, baseline.guardrail_threshold)
    records = []
    protocol_sha = file_hash(protocol)

    def save_candidate(name, guard, scores, original_file=False):
        directory = output / name
        directory.mkdir(mode=0o700)
        threshold = select_threshold(validation, scores)
        measured = quality(validation, scores, threshold)
        reasons = rejection_codes(measured, baseline_quality)
        artifact = directory / "models.joblib"
        if original_file:
            shutil.copyfile(baseline_models / "models.joblib", artifact)
        else:
            joblib.dump({"guardrail": guard, "canary": baseline.bundle["canary"]}, artifact, compress=3)
        model_sha = file_hash(artifact)
        binding = digest({"models_sha256": model_sha, "guardrail_threshold": threshold,
                          "canary_threshold": baseline.canary_threshold, "development_manifest_sha256": file_hash(development / "manifest.json"),
                          "data_manifest_sha256": file_hash(synthetic / "manifest.json"), "protocol_sha256": protocol_sha})
        policy = BinaryPolicy(threshold, binding, tuple(sorted(r["prompt_id"] for r in validation)))
        record = {"candidate": name, "validation": measured, "eligible": not reasons, "rejection_codes": reasons}
        cards = {"source_commit": source_commit, "architecture": "s390x", "sklearn": sklearn.__version__,
                 "models_sha256": model_sha, "model_pair_id": binding, "guardrail_method": name,
                 "guardrail_threshold": threshold, "canary_threshold": baseline.canary_threshold,
                 "policy": policy.to_dict(), "validation": measured, "test_rows_read_for_fit": 0,
                 "model_inputs": "text_only", "canary_learned_state_sha256": canary_state,
                 "protocol_sha256": protocol_sha, "development_manifest_sha256": file_hash(development / "manifest.json"),
                 "data_manifest_sha256": file_hash(synthetic / "manifest.json"),
                 "train_rows": 0 if original_file else len(training), "validation_rows": len(validation),
                 "limitations": ["Source-specific labels, not universal safety truth.", "Exposed synthetic and external regression; no test-based model selection."]}
        write_json(directory / "model-cards.json", cards)
        write_json(directory / "attempts.json", [record])
        write_json(directory / "pre-evaluation-freeze.json", {"test_scored": False, "source_commit": source_commit,
                   "files": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()},
                   "data_files": sm["files"], "catalog_sha256": sm["catalog_sha256"],
                   "development_files": dm["files"], "protocol_sha256": protocol_sha})
        for path in directory.iterdir():
            path.chmod(0o400)
        records.append(record)
        print(json.dumps({"candidate": name, "eligible": not reasons, "validation": measured,
                          "rejection_codes": reasons, "test_scored": False}), flush=True)

    save_candidate("frozen_recalibration", old_guard, baseline_scores, original_file=True)
    features = guard_features()
    x = features.fit_transform([r["text"] for r in training])
    v = features.transform(vx)
    y = np.asarray([r["label"] for r in training])
    for name, synthetic_share in CANDIDATES:
        estimator = LogisticRegression(C=4, max_iter=500, class_weight=None, random_state=42)
        weights = group_weights(training, {"synthetic": synthetic_share, "aegis": 1 - synthetic_share}) * len(training)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(x, y, sample_weight=weights)
        except ConvergenceWarning:
            record = {"candidate": name, "eligible": False, "rejection_codes": ["nonconverged_no_retry"]}
            records.append(record)
            failed = output / name
            failed.mkdir(mode=0o700)
            joblib.dump({"estimator": estimator}, failed / "failed-estimator.joblib", compress=3)
            write_json(failed / "attempts.json", [record])
            continue
        guard = {"features": features, "estimator": estimator, "calibrator": None}
        save_candidate(name, guard, probabilities(guard, v))
    if learned_state_sha256(baseline.bundle["canary"]) != canary_state:
        raise ValueError("frozen_canary_state_changed")
    eligible = [r for r in records if r["eligible"]]
    chosen = max(eligible, key=lambda r: (round(r["validation"]["equal_source_balanced_accuracy"], 12),
                 round(r["validation"]["mean_harmful_recall"], 12), -records.index(r))) if eligible else None
    result = {"source_commit": source_commit, "protocol_sha256": protocol_sha, "baseline_validation": baseline_quality,
              "candidates": records, "selected": chosen["candidate"] if chosen else None,
              "test_scored": False, "test_driven_changes": False, "new_estimator_fits": len(CANDIDATES),
              "canary_unchanged": True, "seconds": time.perf_counter() - started}
    write_json(output / "selection.json", result)
    (output / "selection.json").chmod(0o400)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("synthetic", "development", "baseline-models", "output", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    train(args.synthetic, args.development, args.baseline_models, args.output,
          protocol=args.protocol, source_commit=args.source_commit)


if __name__ == "__main__":
    main()

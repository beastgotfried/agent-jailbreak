"""Create private reference/calibration score pairs from frozen benign pools.

Evaluation prompts are not scored. Classifiers never fit or change here.
The existing trusted-artifact scorer performs version and hash checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .cli import new_output, write_new_json
from .disagreement import MonitorError, ScorePair, _hex, _number, digest, strict_json


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_binding(cards):
    try:
        hashes = {key: cards["artifacts"][key]["sha256"] for key in (
            "guardrail_model", "guardrail_vectorizer", "canary_model", "canary_vectorizer")}
        thresholds = {name: cards["operating_thresholds"][name]["threshold"] for name in ("guardrail", "canary")}
        if not all(_hex(value) for value in hashes.values()) or not all(_number(x, 0, 2) for x in thresholds.values()):
            raise MonitorError("invalid_model_metadata")
        return digest({"artifacts": hashes, "thresholds": thresholds,
            "normalization": cards["normalization"], "label_rule": cards["label_rule"],
            "scikit_learn": cards["packages"]["scikit_learn"]})
    except (KeyError, TypeError) as exc:
        raise MonitorError("invalid_model_metadata") from exc


def prepare_scores(models, pools, output_dir, *, scorer_factory=None, batch_size=64):
    models, pools, output_dir = Path(models), Path(pools), Path(output_dir)
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise MonitorError("invalid_batch_size")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise MonitorError("nonempty_score_output")
    cards_path = models / "model-cards.json"
    cards = strict_json(cards_path.read_text(encoding="utf-8"))
    binding = model_binding(cards)
    manifest_path = pools / "pool-manifest.json"
    manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
    source = pools / "benign.jsonl"
    try:
        expected = manifest["outputs"]["benign.jsonl"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise MonitorError("invalid_pool_manifest") from exc
    if not _hex(expected) or file_sha(source) != expected:
        raise MonitorError("benign_pool_hash_mismatch")
    selected = {"reference": [], "calibration": []}
    skipped = 0
    seen = set()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = strict_json(line)
            if not isinstance(row, dict) or row.get("pool_kind") != "benign":
                raise MonitorError("invalid_benign_pool_record")
            purpose = row.get("partition")
            if purpose == "evaluation":
                skipped += 1
                continue
            if purpose not in selected:
                raise MonitorError("invalid_pool_purpose")
            identity = row.get("prompt_id")
            text = row.get("text")
            if not _hex(identity) or identity in seen or not isinstance(text, str) or not text.strip():
                raise MonitorError("invalid_selected_pool_record")
            if hashlib.sha256(text.encode()).hexdigest() != row.get("text_sha256"):
                raise MonitorError("selected_text_hash_mismatch")
            if hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest() != identity:
                raise MonitorError("selected_identity_mismatch")
            seen.add(identity)
            selected[purpose].append((identity, text))
    if any(not values for values in selected.values()):
        raise MonitorError("missing_reference_or_calibration_pool")
    if scorer_factory is None:
        from src.models.score import load_scorer
        scorer_factory = load_scorer
    scorer = scorer_factory(models)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    counts, zeros = {}, {}
    for purpose, samples in selected.items():
        path = output_dir / f"{purpose}.jsonl"
        counts[purpose] = len(samples)
        zeros[purpose] = {"guardrail": 0, "canary": 0}
        with new_output(path) as handle:
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                scores = scorer.score([text for _, text in batch])
                if len(scores) != len(batch):
                    raise MonitorError("scorer_length_mismatch")
                for offset, ((identity, _), values) in enumerate(zip(batch, scores)):
                    pair = ScorePair(event_id=identity, stream_id="benign-pool", sequence=start + offset,
                        guardrail_score=values["guardrail_score"], canary_score=values["canary_score"],
                        model_pair_id=binding, purpose=purpose)
                    for name in ("guardrail", "canary"):
                        flag = values[f"{name}_all_zero_features"]
                        if type(flag) is not bool:
                            raise MonitorError("invalid_scorer_quality_flag")
                        zeros[purpose][name] += int(flag)
                    handle.write(json.dumps(pair.to_dict(), sort_keys=True, allow_nan=False) + "\n")
    if file_sha(source) != expected:
        raise MonitorError("pool_changes_during_score_preparation")
    report = {"schema_version": 1, "model_pair_id": binding,
        "model_cards_sha256": file_sha(cards_path), "pool_manifest_sha256": file_sha(manifest_path),
        "benign_pool_sha256": expected, "counts": counts,
        "evaluation_rows_skipped": skipped, "evaluation_rows_scored": 0,
        "classifiers_refit": False, "all_zero_feature_counts": zeros,
        "outputs": {f"{purpose}.jsonl": {"sha256": file_sha(output_dir / f"{purpose}.jsonl"), "rows": len(samples)} for purpose, samples in selected.items()},
        "caveat": "Source labels describe benign routing, not universal safety truth. Scores are not calibrated probabilities."}
    write_new_json(output_dir / "score-manifest.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare private score pairs without evaluation or classifier fit")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare_scores(args.models, args.pools, args.output_dir)
        print(json.dumps({"model_pair_id": report["model_pair_id"], "counts": report["counts"], "evaluation_rows_scored": 0}, sort_keys=True))
    except MonitorError as exc:
        print(json.dumps({"kind": "error", "code": str(exc)}), file=sys.stderr)
        return 2
    except (OSError, KeyError, TypeError, ValueError):
        print(json.dumps({"kind": "error", "code": "score_preparation_failed"}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

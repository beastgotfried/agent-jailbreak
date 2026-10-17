"""Frozen MiniLM features and two fixed supervised heads on native LinuxONE."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from src.data.aegis import require_native
from src.data.synthetic import digest, file_hash, load_rows, write_json
from src.models.accuracy import group_weights, quality, read_aegis, rejection_codes, select_threshold, synthetic_rows
from src.models.synthetic import BinaryPolicy, SyntheticBackend, probabilities
from src.release.cli import load_selected
from src.release.state import learned_state_sha256

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WEIGHTS = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
ASSETS = {
    "README.md": "dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "model.safetensors": WEIGHTS,
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def configure_native():
    require_native()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def verify_assets(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["model"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("unapproved_semantic_encoder")
    if set(manifest["files"]) != set(ASSETS) or any(manifest["files"][name]["sha256"] != expected for name, expected in ASSETS.items()):
        raise ValueError("semantic_encoder_manifest_changed")
    for name, expected in ASSETS.items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("semantic_encoder_file_changed")
    if manifest["files"]["model.safetensors"]["sha256"] != WEIGHTS:
        raise ValueError("semantic_encoder_weight_binding_mismatch")
    return manifest


class FrozenMiniLM:
    def __init__(self, assets):
        configure_native()
        verify_assets(assets)
        self.tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(assets, local_files_only=True, trust_remote_code=False,
                                              attn_implementation="eager", dtype=torch.float32).eval()
        self.model.requires_grad_(False)

    def transform(self, texts):
        # Each request executes the encoder. There is no runtime feature cache.
        vectors = []
        for text in texts:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=False)
            with torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                vector = torch.nn.functional.normalize(pooled, p=2, dim=1)[0].cpu().numpy().astype(np.float64)
            if vector.shape != (384,) or not np.isfinite(vector).all():
                raise ValueError("invalid_semantic_features")
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float64).reshape(-1, 384)

    def token_counts(self, texts):
        lengths = [len(self.tokenizer(t, truncation=False, return_attention_mask=False,
                                      return_token_type_ids=False)["input_ids"]) for t in texts]
        return {"rows": len(lengths), "truncated_above_256": sum(n > 256 for n in lengths),
                "maximum_tokens": max(lengths), "median_tokens": float(np.median(lengths))}


class StableHead:
    """Use the same float64 dot-product shape for every classifier request."""
    def __init__(self, estimator):
        if list(estimator.classes_) != [0, 1]:
            raise ValueError("invalid_semantic_head_classes")
        self.estimator = estimator
        self.classes_ = estimator.classes_

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.estimator.coef_.shape[1] or not np.isfinite(x).all():
            raise ValueError("invalid_semantic_head_input")
        scores = np.asarray([float(expit(float(np.dot(row, self.estimator.coef_[0]) + self.estimator.intercept_[0])))
                             for row in x], dtype=np.float64)
        return np.column_stack((1 - scores, scores))


class SemanticBackend(SyntheticBackend):
    def __init__(self, models, expected_cards_sha=None):
        require_native()
        self.directory = Path(models)
        cards_path = self.directory / "model-cards.json"
        if expected_cards_sha is not None and file_hash(cards_path) != expected_cards_sha:
            raise ValueError("semantic_cards_changed")
        cards = json.loads(cards_path.read_text())
        if cards["backend"] != "frozen_minilm" or cards["sklearn"] != sklearn.__version__:
            raise ValueError("semantic_backend_version_mismatch")
        if cards["torch"] != torch.__version__ or cards["transformers"] != transformers.__version__:
            raise ValueError("semantic_runtime_version_mismatch")
        for name, expected in cards["artifact_hashes"].items():
            path = Path(name)
            target = self.directory / path
            if path.is_absolute() or ".." in path.parts or target.is_symlink() or file_hash(target) != expected:
                raise ValueError("semantic_artifact_changed")
        fitted = joblib.load(self.directory / "models.joblib")
        self.bundle = {"guardrail": {"features": FrozenMiniLM(self.directory / "assets"),
                                   "estimator": StableHead(fitted["head"]), "calibrator": None},
                       "canary": fitted["canary"]}
        self.model_pair_id = cards["model_pair_id"]
        self.guardrail_threshold = cards["guardrail_threshold"]
        self.canary_threshold = cards["canary_threshold"]


def sample_fit(synthetic, aegis, groups_per_class=2000):
    if not isinstance(groups_per_class, int) or groups_per_class < 1:
        raise ValueError("invalid_semantic_group_limit")
    pairs, targets = defaultdict(list), defaultdict(list)
    for row in synthetic:
        key = (row.family, row.target_index, row.context_index, row.frame_index)
        pairs[key].append(row)
    for key in pairs:
        if len(pairs[key]) != 2 or {r.label for r in pairs[key]} != {0, 1}:
            raise ValueError("invalid_semantic_fit_pair")
        targets[key[:2]].append(key)
    selected = []
    for keys in targets.values():
        key = min(keys, key=lambda k: digest([42, "semantic-fit-v1", *k]))
        selected.extend({"text": r.text, "label": r.label, "prompt_id": r.prompt_id,
                         "group_id": r.semantic_group, "source": "synthetic"} for r in pairs[key])
    groups, strata = defaultdict(list), defaultdict(list)
    for row in aegis:
        groups[row["group_id"]].append(row)
    for group, rows in groups.items():
        strata[tuple(sorted({r["label"] for r in rows}))].append(group)
    for labels, names in strata.items():
        ordered = sorted(names, key=lambda g: digest([42, "semantic-aegis-fit-v1", g]))
        chosen = ordered[:groups_per_class] if len(labels) == 1 else ordered
        for group in chosen:
            selected.extend(groups[group])
    return sorted(selected, key=lambda r: (r["source"], r["prompt_id"]))


def encode_development(encoder, rows, name):
    arrays = []
    start = time.perf_counter()
    for offset in range(0, len(rows), 128):
        arrays.append(encoder.transform([r["text"] for r in rows[offset:offset + 128]]))
        print(json.dumps({"stage": "semantic_features", "role": name, "completed": min(offset + 128, len(rows)),
                          "total": len(rows), "seconds": time.perf_counter() - start}), flush=True)
    return np.concatenate(arrays)


def train(synthetic, development, baseline_models, output, *, pretrained, protocol, source_commit):
    require_native()
    synthetic, development, baseline_models, output = map(Path, (synthetic, development, baseline_models, output))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.perf_counter()
    config = json.loads(Path(protocol).read_text())
    for folder, key in ((synthetic, "synthetic_manifest_sha256"), (development, "development_manifest_sha256")):
        if file_hash(folder / "manifest.json") != config[key]:
            raise ValueError("semantic_development_binding_mismatch")
        manifest = json.loads((folder / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            if file_hash(folder / name) != expected:
                raise ValueError("semantic_development_file_changed")
    sm = json.loads((synthetic / "manifest.json").read_text())
    dm = json.loads((development / "manifest.json").read_text())
    all_synthetic = load_rows(synthetic / "train.jsonl")
    training = sample_fit(all_synthetic, read_aegis(development / "train.jsonl"))
    validation = synthetic_rows(synthetic / "validation.jsonl", "threshold") + read_aegis(development / "validation.jsonl")
    if len(all_synthetic) != 70000 or sum(r["source"] == "synthetic" for r in training) != 1400 or len(validation) != 6174:
        raise ValueError("semantic_development_roles_changed")
    if {r["prompt_id"] for r in training} & {r["prompt_id"] for r in validation}:
        raise ValueError("semantic_fit_validation_identity_overlap")
    if {(r["source"], r["group_id"]) for r in training} & {(r["source"], r["group_id"]) for r in validation}:
        raise ValueError("semantic_fit_validation_group_overlap")
    write_json(output / "fit-identities.json", {"train": [r["prompt_id"] for r in training],
               "validation": [r["prompt_id"] for r in validation], "test_scored": False})
    _, baseline = load_selected(baseline_models)
    canary_state = learned_state_sha256(baseline.bundle["canary"])
    old = baseline.bundle["guardrail"]
    vx = [r["text"] for r in validation]
    baseline_quality = quality(validation, probabilities(old, old["features"].transform(vx)), baseline.guardrail_threshold)
    encoder = FrozenMiniLM(pretrained)
    token_counts = {"train": encoder.token_counts([r["text"] for r in training]), "validation": encoder.token_counts(vx)}
    write_json(output / "token-counts.json", token_counts)
    x = encode_development(encoder, training, "train")
    v = encode_development(encoder, validation, "validation")
    np.savez_compressed(output / "development-features.npz", train=x, validation=v)
    y = np.asarray([r["label"] for r in training])
    weights = group_weights(training) * len(training)
    records = []
    for head_spec in config["fixed_heads"]:
        name = head_spec["name"]
        directory = output / name
        directory.mkdir(mode=0o700)
        estimator = LogisticRegression(C=head_spec["C"], **config["head_parameters"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(x, y, sample_weight=weights)
        except ConvergenceWarning:
            record = {"candidate": name, "eligible": False, "rejection_codes": ["nonconverged_no_retry"]}
            joblib.dump(estimator, directory / "failed-head.joblib", compress=3)
            write_json(directory / "attempts.json", [record])
            records.append(record)
            continue
        head = StableHead(estimator)
        scores = head.predict_proba(v)[:, 1]
        threshold = select_threshold(validation, scores)
        measured = quality(validation, scores, threshold)
        reasons = rejection_codes(measured, baseline_quality)
        record = {"candidate": name, "validation": measured, "eligible": not reasons, "rejection_codes": reasons}
        joblib.dump({"head": estimator, "canary": baseline.bundle["canary"]}, directory / "models.joblib", compress=3)
        shutil.copytree(pretrained, directory / "assets")
        artifacts = {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*")) if p.is_file()}
        binding = digest({"artifacts": artifacts, "threshold": threshold, "canary_threshold": baseline.canary_threshold,
                          "protocol_sha256": file_hash(protocol), "development_manifest_sha256": config["development_manifest_sha256"]})
        policy = BinaryPolicy(threshold, binding, tuple(sorted(r["prompt_id"] for r in validation)))
        cards = {"source_commit": source_commit, "architecture": "s390x", "backend": "frozen_minilm",
                 "sklearn": sklearn.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                 "artifact_hashes": artifacts, "models_sha256": file_hash(directory / "models.joblib"), "model_pair_id": binding,
                 "guardrail_method": name, "guardrail_components": ["frozen_MiniLM_encoder", "logistic_head"],
                 "guardrail_threshold": threshold, "canary_threshold": baseline.canary_threshold, "policy": policy.to_dict(),
                 "validation": measured, "model_inputs": "text_only", "train_rows": len(training), "validation_rows": len(validation),
                 "test_rows_read_for_fit": 0, "new_encoder_fits": 0, "runtime_output_cache": False,
                 "encoder_batch_size": 1, "max_length": 256, "token_counts": token_counts,
                 "canary_learned_state_sha256": canary_state, "protocol_sha256": file_hash(protocol),
                 "data_manifest_sha256": config["synthetic_manifest_sha256"],
                 "development_manifest_sha256": config["development_manifest_sha256"],
                 "fit_identities_sha256": file_hash(output / "fit-identities.json"),
                 "limitations": ["Source-specific labels, not universal safety truth.",
                                 "The encoder truncates requests above 256 tokens.",
                                 "MiniLM is a BERT-family semantic encoder, not an off-the-shelf safety classifier.",
                                 "Synthetic and prior external tests are exposed regression evidence.",
                                 "The public WhyLabs index is not used."]}
        write_json(directory / "model-cards.json", cards)
        reloaded = SemanticBackend(directory, file_hash(directory / "model-cards.json"))
        batch = reloaded.score(vx[:2])
        singles = [reloaded.score([text])[0] for text in vx[:2]]
        if batch != singles or max(abs(row["guardrail_score"] - scores[i]) for i, row in enumerate(batch)) > 1e-12:
            raise ValueError("semantic_backend_round_trip_or_batch_parity_failed")
        if learned_state_sha256(reloaded.bundle["canary"]) != canary_state:
            raise ValueError("semantic_canary_state_changed")
        write_json(directory / "attempts.json", [record])
        write_json(directory / "pre-evaluation-freeze.json", {"test_scored": False, "source_commit": source_commit,
                   "files": {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*")) if p.is_file()},
                   "data_files": sm["files"], "catalog_sha256": sm["catalog_sha256"],
                   "development_files": dm["files"], "protocol_sha256": file_hash(protocol)})
        for path in directory.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    verify_assets(pretrained)
    if learned_state_sha256(baseline.bundle["canary"]) != canary_state:
        raise ValueError("semantic_fit_changed_canary")
    eligible = [r for r in records if r["eligible"]]
    selected = max(eligible, key=lambda r: (round(r["validation"]["equal_source_balanced_accuracy"], 12),
                   round(r["validation"]["mean_harmful_recall"], 12), -records.index(r))) if eligible else None
    result = {"source_commit": source_commit, "protocol_sha256": file_hash(protocol), "baseline_validation": baseline_quality,
              "candidates": records, "selected": selected["candidate"] if selected else None,
              "train_rows": len(training), "validation_rows": len(validation), "new_encoder_fits": 0, "new_head_fits": len(records),
              "token_counts": token_counts, "canary_unchanged": True, "test_scored": False, "test_driven_changes": False,
              "seconds": time.perf_counter() - started}
    write_json(output / "selection.json", result)
    (output / "selection.json").chmod(0o400)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("synthetic", "development", "baseline-models", "output", "pretrained", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    train(args.synthetic, args.development, args.baseline_models, args.output,
          pretrained=args.pretrained, protocol=args.protocol, source_commit=args.source_commit)


if __name__ == "__main__":
    main()

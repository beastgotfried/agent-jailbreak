"""Bounded native BERT-tiny upgrade. No test text enters the fit.

The encoder fit has two fixed epochs. One optional logistic stack uses only
its reserved validation role. All old models and benchmark rules stay frozen.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import random
import shutil
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.data.synthetic import digest, file_hash, load_rows, write_json
from src.models.synthetic import BinaryPolicy, SyntheticBackend, metrics, probabilities, select_threshold

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
WEIGHTS_SHA = "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414"
SEED, EPOCHS, MAX_LENGTH, BATCH_SIZE = 42, 2, 128, 16


def native_cpu():
    if platform.machine() != "s390x":
        raise RuntimeError("neural_study_requires_native_s390x")


def configure_cpu():
    native_cpu()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def versions():
    return {"torch": str(torch.__version__), "transformers": transformers.__version__,
            "sklearn": sklearn.__version__, "numpy": np.__version__}


def epoch_rows(rows, epoch):
    """Keep both labels together; select one frame per case before any fit."""
    if epoch not in range(EPOCHS) or any(r.split != "train" for r in rows):
        raise ValueError("invalid_neural_training_role")
    groups = defaultdict(list)
    for row in rows:
        groups[row.case_id].append(row)
    selected = []
    for case in sorted(groups):
        members = groups[case]
        if len(members) != 10 or {(r.frame_index, r.label) for r in members} != {
                (frame, label) for frame in range(5) for label in (0, 1)}:
            raise ValueError("invalid_neural_matched_case")
        frame = int(digest([SEED, "neural-epoch-frame-v1", case, epoch]), 16) % 5
        selected.extend(sorted((r for r in members if r.frame_index == frame), key=lambda r: r.label))
    rng = np.random.default_rng(SEED + epoch)
    return [selected[i] for i in rng.permutation(len(selected))]


def stack_features(neural_logits, lexical_scores):
    values = np.asarray(lexical_scores, dtype=np.float64)
    logits = np.asarray(neural_logits, dtype=np.float64)
    if values.shape != logits.shape or values.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("invalid_neural_stack_shape")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("invalid_neural_stack_scores")
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.column_stack((logits, np.log(values) - np.log1p(-values)))


def ensemble_improves(neural, ensemble):
    return (ensemble["balanced_accuracy"] >= neural["balanced_accuracy"] + .005 - 1e-12
            and ensemble["recall"] >= neural["recall"] - 1e-12
            and ensemble["fpr"] <= neural["fpr"] + 1e-12)


class TinyGuard(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.head(pooled).squeeze(-1)


def trainable_parameters(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = []
    for layer in model.encoder.encoder.layer[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
            encoder.append(parameter)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    return encoder, list(model.head.parameters())


def tokens(tokenizer, texts):
    return tokenizer(list(texts), padding="max_length", truncation=True,
                     max_length=MAX_LENGTH, return_tensors="pt")


def logits_for(model, tokenizer, texts, batch_size=32):
    if not texts:
        return np.empty(0, dtype=np.float64)
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            values = model(**tokens(tokenizer, texts[start:start + batch_size]))
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite_neural_scores")
            result.extend(values.cpu().numpy().astype(np.float64).tolist())
    return np.asarray(result, dtype=np.float64)


def sigmoid(logits):
    return torch.sigmoid(torch.as_tensor(np.asarray(logits), dtype=torch.float64)).numpy()


def save_weights(model, path):
    # NumPy stores explicit byte order. This also avoids pickle for neural weights.
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def load_weights(model, path):
    with np.load(path, allow_pickle=False) as data:
        state = {key: torch.from_numpy(np.array(data[key], dtype=np.float32, copy=True)) for key in data.files}
    model.load_state_dict(state, strict=True)


def truncation_counts(tokenizer, texts):
    count, longest = 0, 0
    for start in range(0, len(texts), 256):
        encoded = tokenizer(list(texts[start:start + 256]), truncation=False, padding=False)["input_ids"]
        lengths = [len(row) for row in encoded]
        count += sum(n > MAX_LENGTH for n in lengths)
        longest = max(longest, max(lengths, default=0))
    return {"rows": len(texts), "truncated": count, "max_tokens": longest, "token_limit": MAX_LENGTH}


def verify_files(directory, hashes):
    directory = Path(directory)
    for name, expected in hashes.items():
        target = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not target.is_file() or target.is_symlink():
            raise ValueError("invalid_neural_artifact_path")
        if file_hash(target) != expected:
            raise ValueError("neural_artifact_hash_mismatch")


class TextFeatures:
    def transform(self, texts):
        return list(texts)


class NeuralEstimator:
    classes_ = np.asarray([0, 1])

    def __init__(self, model, tokenizer, lexical, stacker=None):
        self.model, self.tokenizer, self.lexical, self.stacker = model, tokenizer, lexical, stacker

    def predict_proba(self, texts):
        values = logits_for(self.model, self.tokenizer, texts)
        if self.stacker is None:
            positive = sigmoid(values)
        else:
            old = probabilities(self.lexical, self.lexical["features"].transform(texts))
            positive = self.stacker.predict_proba(stack_features(values, old))[:, 1]
        return np.column_stack((1 - positive, positive))


class NeuralBackend:
    def __init__(self, models, expected_cards_sha=None):
        native_cpu()
        self.directory = Path(models)
        cards_path = self.directory / "model-cards.json"
        if expected_cards_sha is not None and file_hash(cards_path) != expected_cards_sha:
            raise ValueError("neural_model_cards_changed")
        cards = json.loads(cards_path.read_text())
        if cards["versions"] != versions():
            raise ValueError("neural_dependency_version_mismatch")
        verify_files(self.directory, cards["artifact_hashes"])
        config = AutoConfig.from_pretrained(self.directory / "assets", local_files_only=True, trust_remote_code=False)
        model = TinyGuard(AutoModel.from_config(config, attn_implementation="eager"))
        load_weights(model, self.directory / "neural-weights.npz")
        model.double().eval()
        tokenizer = AutoTokenizer.from_pretrained(self.directory / "assets", local_files_only=True, trust_remote_code=False)
        # Private trusted artifacts only. SHA checks do not establish authenticity.
        lexical = joblib.load(self.directory / "lexical.joblib")
        stacker = joblib.load(self.directory / "stacker.joblib") if cards["selected"] == "neural_lexical_stack" else None
        self.bundle = {"guardrail": {"features": TextFeatures(), "estimator": NeuralEstimator(
            model, tokenizer, lexical["guardrail"], stacker), "calibrator": None}, "canary": lexical["canary"]}
        self.model_pair_id = cards["model_pair_id"]
        self.guardrail_threshold = cards["guardrail_threshold"]
        self.canary_threshold = cards["canary_threshold"]

    def score(self, texts):
        if not texts:
            return []
        values = {name: probabilities(model, model["features"].transform(texts)) for name, model in self.bundle.items()}
        return [{"guardrail_score": float(g), "canary_score": float(c),
                 "guardrail_unsafe": bool(g >= self.guardrail_threshold),
                 "canary_unsafe": bool(c >= self.canary_threshold)}
                for g, c in zip(values["guardrail"], values["canary"])]


def train(data, legacy_models, pretrained, output, protocol, *, source_commit):
    native_cpu()
    start = time.perf_counter()
    data, legacy_models, pretrained, output, protocol = map(Path, (data, legacy_models, pretrained, output, protocol))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = json.loads((data / "manifest.json").read_text())
    verify_files(data, manifest["files"])
    pre = json.loads((pretrained / "manifest.json").read_text())
    if pre["model"] != MODEL_ID or pre["revision"] != REVISION or file_hash(pretrained / "model.safetensors") != WEIGHTS_SHA:
        raise ValueError("unapproved_pretrained_model")
    verify_files(pretrained, {name: item["sha256"] for name, item in pre["files"].items()})
    spec = json.loads(protocol.read_text())
    if spec["pretrained"]["model"] != MODEL_ID or spec["candidate_1"]["epochs"] != EPOCHS:
        raise ValueError("neural_protocol_mismatch")
    training, validation = load_rows(data / "train.jsonl"), load_rows(data / "validation.jsonl")
    if len(training) != 70000 or len(validation) != 10000:
        raise ValueError("invalid_neural_data_counts")
    if {r.semantic_group for r in training} & {r.semantic_group for r in validation}:
        raise ValueError("neural_semantic_group_overlap")
    used_validation = [r for r in validation if r.role in ("threshold", "fallback_calibration")]
    threshold_ids = np.asarray([i for i, r in enumerate(used_validation) if r.role == "threshold"])
    calibration_ids = np.asarray([i for i, r in enumerate(used_validation) if r.role == "fallback_calibration"])
    if len(threshold_ids) != 5000 or len(calibration_ids) != 3000:
        raise ValueError("invalid_neural_validation_roles")
    legacy_cards = json.loads((legacy_models / "model-cards.json").read_text())
    legacy = SyntheticBackend(legacy_models, file_hash(legacy_models / "model-cards.json"))
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True, trust_remote_code=False)
    model = TinyGuard(AutoModel.from_pretrained(pretrained, local_files_only=True,
                                               trust_remote_code=False, attn_implementation="eager"))
    encoder, head = trainable_parameters(model)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": 2e-5}, {"params": head, "lr": 2e-4}], weight_decay=.01)
    epoch_sets = [epoch_rows(training, epoch) for epoch in range(EPOCHS)]
    if any(len(rows) != 14000 for rows in epoch_sets):
        raise ValueError("invalid_neural_epoch_count")
    steps = sum((len(rows) + BATCH_SIZE - 1) // BATCH_SIZE for rows in epoch_sets)
    schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps // 10, num_training_steps=steps)
    write_json(output / "fit-started.json", {"source_commit": source_commit, "protocol_sha256": file_hash(protocol),
        "epoch_ids": [[r.prompt_id for r in rows] for rows in epoch_sets], "test_rows_read": 0})
    losses = []
    model.train()
    for epoch, rows in enumerate(epoch_sets):
        total = 0.0
        for begin in range(0, len(rows), BATCH_SIZE):
            batch = rows[begin:begin + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(**tokens(tokenizer, [r.text for r in batch]))
            labels = torch.tensor([r.label for r in batch], dtype=logits.dtype)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("neural_fit_nonfinite_no_retry")
            loss.backward()
            nn.utils.clip_grad_norm_(encoder + head, 1.0, error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            total += float(loss.detach()) * len(batch)
            if begin % 3200 == 0:
                print(json.dumps({"epoch": epoch + 1, "rows_completed": begin + len(batch),
                                  "mean_loss": total / (begin + len(batch))}), flush=True)
        losses.append(total / len(rows))
    save_weights(model, output / "neural-weights.npz")
    model.encoder.config.save_pretrained(output / "assets")
    tokenizer.save_pretrained(output / "assets")
    shutil.copyfile(legacy_models / "models.joblib", output / "lexical.joblib")
    model.double().eval()
    vx, vy = [r.text for r in used_validation], np.asarray([r.label for r in used_validation])
    vl = logits_for(model, tokenizer, vx)
    nv = sigmoid(vl)
    threshold = select_threshold(vy[threshold_ids], nv[threshold_ids])
    neural_quality = metrics(vy[threshold_ids], nv[threshold_ids], threshold)
    attempts = [{"method": "bert_tiny_partial_finetune", "valid": True, "validation": neural_quality}]
    selected = "bert_tiny_partial_finetune"
    old = probabilities(legacy.bundle["guardrail"], legacy.bundle["guardrail"]["features"].transform(vx))
    features = stack_features(vl, old)
    stacker = LogisticRegression(C=1, max_iter=200, class_weight="balanced", random_state=SEED)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            stacker.fit(features[calibration_ids], vy[calibration_ids])
        sv = stacker.predict_proba(features)[:, 1]
        st = select_threshold(vy[threshold_ids], sv[threshold_ids])
        sq = metrics(vy[threshold_ids], sv[threshold_ids], st)
        attempts.append({"method": "neural_lexical_stack", "valid": True, "validation": sq})
        joblib.dump(stacker, output / "stacker.joblib", compress=3)
        if ensemble_improves(neural_quality, sq):
            selected, threshold = "neural_lexical_stack", st
    except ConvergenceWarning:
        attempts.append({"method": "neural_lexical_stack", "valid": False, "code": "nonconverged_no_retry"})
    # No probe-pool or test row enters this numerical-invariance check.
    check = vx[:32]
    estimator = NeuralEstimator(model, tokenizer, legacy.bundle["guardrail"],
                                stacker if selected == "neural_lexical_stack" else None)
    batched = estimator.predict_proba(check)[:, 1]
    single = np.asarray([estimator.predict_proba([text])[0, 1] for text in check])
    batch_error = float(np.max(np.abs(batched - single)))
    if batch_error > 1e-10:
        raise RuntimeError("neural_batch_invariance_failed_no_monitor_relaxation")
    write_json(output / "attempts.json", attempts)
    shutil.copyfile(protocol, output / "protocol.json")
    hashes = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    binding = digest({"artifacts": hashes, "guardrail_threshold": threshold,
                      "canary_threshold": legacy.canary_threshold, "versions": versions(), "inference_dtype": "float64"})
    policy = BinaryPolicy(threshold, binding, tuple(sorted(used_validation[i].prompt_id for i in threshold_ids)))
    cards = {"source_commit": source_commit, "architecture": platform.machine(), "versions": versions(),
        "pretrained_model": MODEL_ID, "pretrained_revision": REVISION, "pretrained_license": "Apache-2.0",
        "selected": selected, "attempts": attempts, "artifact_hashes": hashes, "model_pair_id": binding,
        "guardrail_threshold": threshold, "canary_threshold": legacy.canary_threshold, "policy": policy.to_dict(),
        "canary_unchanged": True, "legacy_models_sha256": legacy_cards["models_sha256"],
        "legacy_cards_sha256": file_hash(legacy_models / "model-cards.json"),
        "train_pool_rows": 70000, "rows_per_epoch": [len(rows) for rows in epoch_sets], "epochs": EPOCHS,
        "encoder_fit_attempts": 1, "epoch_mean_losses": losses, "trainable_parameters": sum(p.numel() for p in encoder + head),
        "total_parameters": sum(p.numel() for p in model.parameters()), "max_length": MAX_LENGTH,
        "inference_dtype": "float64", "batch_invariance_max_error": batch_error,
        "validation_roles_used": {"threshold": 5000, "fallback_calibration": 3000}, "test_rows_read_for_fit": 0,
        "truncation": {"train": truncation_counts(tokenizer, [r.text for r in epoch_sets[0]]),
                       "validation": truncation_counts(tokenizer, vx)},
        "fit_seconds": time.perf_counter() - start,
        "limits": ["Existing test and external sources are exposed regression checks.",
                   "No fresh independent confirmation or production qualification.",
                   "Holds and fault-response delays remain in the benchmark score.",
                   "No token or model-output cache replaces actual neural inference.",
                   "Pretraining overlap with evaluation sources is not ruled out."]}
    write_json(output / "model-cards.json", cards)
    reloaded = NeuralBackend(output, file_hash(output / "model-cards.json"))
    parity = np.asarray([r["guardrail_score"] for r in reloaded.score(check)])
    if not np.allclose(parity, batched, atol=1e-12, rtol=0):
        raise RuntimeError("neural_reload_parity_failed")
    write_json(output / "pre-evaluation-freeze.json", {"test_scored": False, "source_commit": source_commit,
        "test_data_previously_exposed": True,
        "files": {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "data_files": manifest["files"], "catalog_sha256": manifest["catalog_sha256"]})
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    print(json.dumps({"selected": selected, "attempts": attempts, "test_scored": False,
                      "seconds": cards["fit_seconds"]}), flush=True)
    return cards


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "legacy-models", "pretrained", "output", "protocol"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    configure_cpu()
    train(args.data, args.legacy_models, args.pretrained, args.output, args.protocol, source_commit=args.source_commit)


if __name__ == "__main__":
    main()

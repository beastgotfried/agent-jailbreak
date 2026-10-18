"""Neutral native cost and repeatability check for a local pretrained encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if platform.machine() != "s390x" or platform.system() != "Linux":
        raise ValueError("native_s390x_required")
    manifest = json.loads((args.model / "manifest.json").read_text())
    if manifest["model"] != "sentence-transformers/all-MiniLM-L6-v2" or manifest["revision"] != "1110a243fdf4706b3f48f1d95db1a4f5529b4d41":
        raise ValueError("unapproved_encoder")
    for name, record in manifest["files"].items():
        if Path(name).name != name or sha(args.model / name) != record["sha256"]:
            raise ValueError("pretrained_file_changed")
    if manifest["files"]["model.safetensors"]["sha256"] != "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db":
        raise ValueError("pretrained_weight_binding_mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                     attn_implementation="eager", dtype=torch.float32).eval()
    model.requires_grad_(False)
    load_seconds = time.perf_counter() - start

    def encode_one(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=False)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            values = torch.nn.functional.normalize(pooled, p=2, dim=1)[0].cpu().numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite_neutral_embedding")
        return values, int(inputs["input_ids"].shape[1])

    results = []
    for words in (6, 30, 62, 126, 254):
        text = " ".join(["sample"] * words)
        first, tokens = encode_one(text)
        times, errors = [], []
        for _ in range(8):
            start = time.perf_counter()
            value, _ = encode_one(text)
            times.append(time.perf_counter() - start)
            errors.append(float(np.max(np.abs(first - value))))
        results.append({"tokens": tokens, "mean_seconds": float(np.mean(times)),
                        "min_seconds": min(times), "max_seconds": max(times),
                        "max_repeat_embedding_error": max(errors)})
    texts = ["A neutral sample request.", "Please describe the blue circle."]
    alone = [encode_one(text)[0] for text in texts]
    # A batch adapter loops over single-text forwards, not padded encoder batches.
    together = [encode_one(text)[0] for text in texts]
    error = max(float(np.max(np.abs(a - b))) for a, b in zip(alone, together))
    report = {"status": "passed" if error == 0 and all(r["max_repeat_embedding_error"] == 0 for r in results) else "failed",
              "model": manifest["model"], "revision": manifest["revision"], "architecture": platform.machine(),
              "byteorder": sys.byteorder, "parameters": sum(p.numel() for p in model.parameters()),
              "dtype": "float32 encoder; float64 classifier features", "encoder_batch_size": 1,
              "padding": "none", "max_length": 256, "load_seconds": load_seconds,
              "neutral_costs": results, "batch_adapter_embedding_error": error,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "study_rows_scored": 0, "model_fit": False, "model_output_cache": False,
              "scope": "Neutral cost screen, not task accuracy or production latency qualification."}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

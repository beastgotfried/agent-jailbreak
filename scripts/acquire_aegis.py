"""Acquire a pinned public prompt-label source on native LinuxONE.

Request text remains private data. This script does not load or execute models.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import urllib.request

REPOSITORY = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
FILES = {
    "README.md": "da2598e587850bacc77c0658f56d0ea430f55d1cf70b197eb06070871802d3ba",
    "train.json": "154fba82c71d9fa73abd2ca5588a198e693ddc816c83444df180a22f613e02f6",
    "validation.json": "a97200e226ad4f6ba6a639982f817f675909bc74513859df3e8fc9a92951dfcd",
    "test.json": "b0a6d602260524866053cb34105194f074f2c2906e3691b68d43b9e6e9318f35",
}


def acquire(output):
    if platform.system() != "Linux" or platform.machine() != "s390x":
        raise ValueError("native_s390x_required")
    os.umask(0o077)
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, expected in FILES.items():
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "SafetyWatchResearch/0.1"})
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=120) as source, (output / name).open("xb") as target:
            while block := source.read(1 << 20):
                target.write(block)
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("dataset_source_hash_mismatch")
        (output / name).chmod(0o400)
    development = {}
    for name in ("train.json", "validation.json"):
        rows = json.loads((output / name).read_text())
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("unsupported_aegis_schema")
        development[name] = {
            "rows": len(rows), "columns": sorted({key for row in rows for key in row}),
            "prompt_labels": dict(Counter(str(row.get("prompt_label")) for row in rows)),
            "prompt_label_sources": dict(Counter(str(row.get("prompt_label_source")) for row in rows)),
            "redacted_prompts": sum(row.get("prompt") == "REDACTED" for row in rows),
            "max_prompt_characters": max(len(row.get("prompt") or "") for row in rows),
            "duplicate_ids": len(rows) - len({str(row.get("id")) for row in rows}),
            "id_types": sorted({type(row.get("id")).__name__ for row in rows}),
        }
    receipt = {
        "repository": REPOSITORY, "revision": REVISION, "license": "CC-BY-4.0",
        "files": FILES, "architecture": platform.machine(), "development_schema": development,
        "test_status": "downloaded_and_hash_verified_only_not_parsed_or_scored",
        "model_inputs": "prompt_only; never responses, labels, IDs, source names, or categories",
        "excluded_sources": ["refusal response augmentation", "redacted suicide-watch samples", "topic-following augmentation"],
        "upstream_prompt_sources_reported_by_card": ["Anthropic/hh-rlhf", "Do-Anything-Now DAN", "AI-assisted Red-Teaming"],
        "source_policy": "Provider prompt-harm labels remain source-specific, not universal safety truth.",
    }
    (output / "acquisition.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (output / "acquisition.json").chmod(0o400)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    acquire(args.output)


if __name__ == "__main__":
    main()

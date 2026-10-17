"""Private, deterministic prompt-only preparation for the pinned Aegis source."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np

from src.data.synthetic import digest, file_hash, text_id, write_json

REPO = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
SPLITS = ("train", "validation", "test")
EXPECTED_FILES = {
    "README.md": "da2598e587850bacc77c0658f56d0ea430f55d1cf70b197eb06070871802d3ba",
    "train.json": "154fba82c71d9fa73abd2ca5588a198e693ddc816c83444df180a22f613e02f6",
    "validation.json": "a97200e226ad4f6ba6a639982f817f675909bc74513859df3e8fc9a92951dfcd",
    "test.json": "b0a6d602260524866053cb34105194f074f2c2906e3691b68d43b9e6e9318f35",
}
PRIOR_FILES = {
    "toxic_train": "702eb9b7cac96c3c35e28b9b95855a71f26f21afba8666f9d243f1fa469e81ed",
    "toxic_test": "3c2e49889626f7738dca0a29bface0ba0a0595b2ffdd17f0e02f19df7c3c4c9b",
    "xstest": "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
    "dolly": "e442d96b320010510c67a6cc0a86ea988a2a6f71c7570d1afbdf4fa631896e58",
    "original_pools/benign.jsonl": "ced6e7a2cb5077d0905b88f31cacef2044e5771f1ca27fb7e4804a0de737d3f0",
    "original_pools/attacks.jsonl": "5a94554a1b2e1171b5ca50a923f36ba6713b212295ad7818db1813025bf95131",
}


def require_native():
    if platform.system() != "Linux" or platform.machine() != "s390x":
        raise ValueError("accuracy_experiment_requires_native_s390x")


def shingles(text):
    words = re.findall(r"\w+", text.lower())
    parts = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if not parts:
        parts = [" ".join(words) or text.lower().strip()]
    return {int.from_bytes(hashlib.sha256(p.encode()).digest()[:8], "big") for p in parts}


def lexical_groups(texts):
    """Exact IDs plus MinHash proposals, verified by fixed shingle Jaccard."""
    ids = sorted(texts)
    parent = {pid: pid for pid in ids}

    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    coefficients = [int.from_bytes(hashlib.sha256(f"42:minhash:{i}".encode()).digest()[:8], "big") for i in range(64)]
    a = np.asarray([v | 1 for v in coefficients[:32]], dtype=np.uint64)
    b = np.asarray(coefficients[32:], dtype=np.uint64)
    buckets, stored = defaultdict(list), {}
    examined, linked = 0, 0
    for pid in ids:
        current = shingles(texts[pid])
        values = np.asarray(sorted(current), dtype=np.uint64)
        signature = np.min(a[:, None] * values[None, :] + b[:, None], axis=1)
        keys = [(band, tuple(int(v) for v in signature[band * 4:band * 4 + 4])) for band in range(8)]
        possible = {other for key in keys for other in buckets[key]}
        for other in sorted(possible):
            examined += 1
            if examined > 2000000:
                raise ValueError("near_duplicate_candidate_budget_exceeded")
            previous = stored[other]
            if min(len(current), len(previous)) / max(len(current), len(previous)) < .85:
                continue
            similarity = len(current & previous) / len(current | previous)
            if similarity >= .85:
                join(pid, other)
                linked += 1
        for key in keys:
            buckets[key].append(pid)
        stored[pid] = current
    return {pid: find(pid) for pid in ids}, {"candidate_pairs": examined, "linked_pairs": linked}


def prior_identities(synthetic, toxic_train, toxic_test, xstest, pools, dolly):
    blocked, hashes = set(), {}
    for split in SPLITS:
        path = Path(synthetic) / f"{split}.jsonl"
        hashes[f"synthetic/{path.name}"] = file_hash(path)
        with path.open() as handle:
            for line in handle:
                blocked.add(json.loads(line)["prompt_id"])
    for name, path, field in (("toxic_train", toxic_train, "user_input"),
                              ("toxic_test", toxic_test, "user_input"), ("xstest", xstest, "prompt")):
        path = Path(path)
        hashes[name] = file_hash(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                blocked.add(text_id(row[field]))
    for name in ("benign.jsonl", "attacks.jsonl"):
        path = Path(pools) / name
        hashes[f"original_pools/{name}"] = file_hash(path)
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                blocked.add(text_id(row["text"]))
    hashes["dolly"] = file_hash(dolly)
    with Path(dolly).open() as handle:
        for line in handle:
            blocked.add(text_id(json.loads(line)["instruction"]))
    if any(hashes[name] != expected for name, expected in PRIOR_FILES.items()):
        raise ValueError("prior_exposure_source_changed")
    return blocked, hashes


def prepare(raw, output, *, blocked, prior_hashes, source_commit):
    require_native()
    raw, output = Path(raw), Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    acquisition = json.loads((raw / "acquisition.json").read_text())
    if (acquisition["revision"] != REVISION or acquisition["repository"] != REPO
            or acquisition["files"] != EXPECTED_FILES):
        raise ValueError("unapproved_aegis_revision")
    for name, expected in acquisition["files"].items():
        if file_hash(raw / name) != expected:
            raise ValueError("aegis_source_hash_mismatch")
    records, excluded, original_counts = defaultdict(list), Counter(), {}
    for split in SPLITS:
        rows = json.loads((raw / f"{split}.json").read_text())
        original_counts[split] = len(rows)
        for index, row in enumerate(rows):
            text, label = row.get("prompt"), row.get("prompt_label")
            if not isinstance(text, str) or not text.strip() or text.strip() == "REDACTED" or row.get("reconstruction_id_if_redacted"):
                excluded[f"{split}:redacted_or_missing"] += 1
                continue
            if row.get("prompt_label_source") != "human" or label not in ("safe", "unsafe"):
                excluded[f"{split}:missing_human_prompt_label"] += 1
                continue
            pid = text_id(text)
            if pid in blocked:
                excluded[f"{split}:prior_exposure_overlap"] += 1
                continue
            records[pid].append({"prompt_id": pid, "text": text, "label": int(label == "unsafe"),
                                 "split": split, "source": "aegis", "source_row": index, "source_id": str(row["id"])})
    for pid in list(records):
        if len({r["label"] for r in records[pid]}) != 1:
            excluded["conflicting_exact_prompt_rows"] += len(records.pop(pid))
    groups, grouping = lexical_groups({pid: members[0]["text"] for pid, members in records.items()})
    group_split = {}
    for pid, members in records.items():
        rank = max(SPLITS.index(r["split"]) for r in members)
        group_split[groups[pid]] = max(group_split.get(groups[pid], 0), rank)
    kept = {split: [] for split in SPLITS}
    for pid, members in sorted(records.items()):
        split = SPLITS[group_split[groups[pid]]]
        eligible = [r for r in members if r["split"] == split]
        excluded["cross_split_group_rows"] += len(members) - len(eligible)
        if not eligible:
            continue
        chosen = min(eligible, key=lambda r: (r["source_row"], r["source_id"]))
        chosen = dict(chosen, group_id=groups[pid], provenance=[{k: r[k] for k in ("split", "source_row", "source_id")} for r in members])
        kept[split].append(chosen)
        excluded["same_split_exact_duplicate_rows"] += len(eligible) - 1
    split_groups = {split: {r["group_id"] for r in rows} for split, rows in kept.items()}
    if any(split_groups[a] & split_groups[b] for a in SPLITS for b in SPLITS if a != b):
        raise ValueError("aegis_group_split_overlap")
    for split, rows in kept.items():
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError("aegis_split_missing_class")
        with (output / f"{split}.jsonl").open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    manifest = {"source_commit": source_commit, "repository": REPO, "revision": REVISION,
                "license": "CC-BY-4.0", "raw_files": acquisition["files"], "prior_input_hashes": prior_hashes,
                "prior_identity_count": len(blocked), "original_rows": original_counts, "exclusions": dict(excluded),
                "grouping": grouping, "model_input": "text_only", "test_scored": False,
                "counts": {s: {"rows": len(v), "groups": len(split_groups[s]), "labels": dict(Counter(r["label"] for r in v))} for s, v in kept.items()},
                "files": {f"{s}.jsonl": file_hash(output / f"{s}.jsonl") for s in SPLITS},
                "scope": "Source-specific human prompt labels; lexical grouping does not prove semantic independence."}
    write_json(output / "manifest.json", manifest)
    for path in output.iterdir():
        path.chmod(0o400)
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "output", "synthetic", "toxic-train", "toxic-test", "xstest", "pools", "dolly"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    require_native()
    blocked, hashes = prior_identities(args.synthetic, args.toxic_train, args.toxic_test, args.xstest, args.pools, args.dolly)
    manifest = prepare(args.raw, args.output, blocked=blocked, prior_hashes=hashes, source_commit=args.source_commit)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Private, compositional synthetic requests with family-disjoint partitions.

Catalogue labels describe requested intent. They are not model predictions.
The generator never executes requests. Non-s390x execution requires an
explicit operator-approved local-execution flag.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from string import Formatter

FRAMES = (
    "{context}\n{request}",
    "{request}\nContext: {context}",
    "Request: {request}\nBackground: {context}",
    "Please address this request: {request}\n{context}",
    "{context}\nQuestion for an assistant: {request}",
)
SIZES = {"train": 70000, "validation": 10000, "test": 20000}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_id(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def native_only():
    if platform.machine() != "s390x" and os.environ.get("SAFETY_WATCH_ALLOW_LOCAL") != "1":
        raise RuntimeError("native_s390x_or_explicit_local_authorization_required")


@dataclass(frozen=True)
class SyntheticRow:
    prompt_id: str
    text: str
    label: int
    split: str
    role: str
    domain: str
    family: str
    semantic_group: str
    case_id: str
    target_index: int
    context_index: int
    frame_index: int


def validate_catalog(catalog):
    if catalog.get("schema_version") != 1 or len(catalog.get("domains", [])) != 10:
        raise ValueError("invalid_catalog_domains")
    domains, families, prototypes = set(), set(), set()
    for domain in catalog["domains"]:
        name = domain.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name) or name in domains:
            raise ValueError("invalid_catalog_domain_id")
        domains.add(name)
        contexts = domain.get("contexts", [])
        if len(contexts) != 10 or len(set(contexts)) != 10 or any(not isinstance(t, str) or not t.strip() for t in contexts):
            raise ValueError("invalid_catalog_contexts")
        if len(domain.get("families", [])) != 10:
            raise ValueError("invalid_catalog_family_count")
        for family in domain["families"]:
            fid = family.get("id")
            if not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", fid) or fid in families:
                raise ValueError("invalid_catalog_family_id")
            families.add(fid)
            group = family.get("semantic_group")
            if not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", group):
                raise ValueError("invalid_semantic_group")
            targets = family.get("targets", [])
            if len(targets) != 10 or len(set(targets)) != 10 or any(not isinstance(t, str) or not t.strip() for t in targets):
                raise ValueError("invalid_catalog_targets")
            if not isinstance(family.get("rationale"), str) or not family["rationale"].strip():
                raise ValueError("missing_label_rationale")
            for key in ("safe", "harmful"):
                template = family.get(key)
                if not isinstance(template, str) or len(template) < 20:
                    raise ValueError("invalid_catalog_request")
                slots = [(field, fmt, conv) for _, field, fmt, conv in Formatter().parse(template) if field is not None]
                if not slots or any(field != "target" or fmt or conv for field, fmt, conv in slots):
                    raise ValueError("invalid_catalog_slot")
                pid = text_id(template)
                if pid in prototypes:
                    raise ValueError("duplicate_catalog_prototype")
                prototypes.add(pid)
    return catalog


def family_splits(catalog):
    # Repeated intents stay entirely in train, including cross-domain aliases.
    # This conservative rule precedes all generated rows and model outcomes.
    frequencies = Counter(f["semantic_group"] for d in catalog["domains"] for f in d["families"])
    result = {}
    for domain in catalog["domains"]:
        locked = [f for f in domain["families"] if frequencies[f["semantic_group"]] > 1]
        if len(locked) > 7:
            raise ValueError("semantic_groups_exceed_train_quota")
        result.update({f["id"]: "train" for f in locked})
        ordered = sorted((f for f in domain["families"] if frequencies[f["semantic_group"]] == 1),
                         key=lambda f: digest([42, "synthetic-family-split-v1", domain["id"], f["id"]]))
        remaining = 7 - len(locked)
        for i, family in enumerate(ordered):
            result[family["id"]] = "train" if i < remaining else "validation" if i == remaining else "test"
    return result


def validation_roles(family):
    ordered = sorted(((a, b) for a in range(10) for b in range(10)),
                     key=lambda pair: digest([42, "synthetic-validation-role-v1", family, *pair]))
    return {pair: ("threshold" if i < 50 else "fallback_calibration" if i < 80 else "probe_fit" if i < 90 else "probe_verification")
            for i, pair in enumerate(ordered)}


def expand(catalog):
    validate_catalog(catalog)
    splits = family_splits(catalog)
    for domain in sorted(catalog["domains"], key=lambda d: d["id"]):
        for family in sorted(domain["families"], key=lambda f: f["id"]):
            fid, split = family["id"], splits[family["id"]]
            roles = validation_roles(fid) if split == "validation" else {}
            for ti, target in enumerate(family["targets"]):
                for ci, context in enumerate(domain["contexts"]):
                    case = digest([fid, ti, ci])
                    role = roles[(ti, ci)] if split == "validation" else split
                    for fi, frame in enumerate(FRAMES):
                        for label, key in ((0, "safe"), (1, "harmful")):
                            request = family[key].format(target=target)
                            text = frame.format(context=context, request=request)
                            yield SyntheticRow(text_id(text), text, label, split, role, domain["id"], fid,
                                               family["semantic_group"], case, ti, ci, fi)


def prototype_similarity(catalog, splits):
    items = []
    for d in catalog["domains"]:
        for f in d["families"]:
            for key in ("safe", "harmful"):
                words = set(re.findall(r"[a-z]+", f[key].lower().replace("{target}", "")))
                items.append((f["id"], key, words))
    pairs = []
    for i, (fa, la, a) in enumerate(items):
        for fb, lb, b in items[i + 1:]:
            if splits[fa] == splits[fb]:
                continue
            score = len(a & b) / len(a | b) if a | b else 1.0
            pairs.append({"family_a": fa, "label_a": la, "family_b": fb, "label_b": lb, "word_jaccard": score})
    pairs.sort(key=lambda p: (-p["word_jaccard"], p["family_a"], p["family_b"], p["label_a"], p["label_b"]))
    return {"top_pairs": pairs[:12], "pairs_ge_0_85": sum(p["word_jaccard"] >= .85 for p in pairs),
            "scope": "prototype_lexical_screen_not_semantic_independence_proof"}


def generate(catalog_path, output, *, source_commit):
    native_only()
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    catalog = validate_catalog(json.loads(Path(catalog_path).read_text()))
    splits = family_splits(catalog)
    seen, counts, roles, pairs, families = set(), Counter(), Counter(), {}, {}
    semantic_groups = {}
    lengths = []
    handles = {s: (output / (s + ".jsonl")).open("x", encoding="utf-8") for s in SIZES}
    try:
        for row in expand(catalog):
            if row.prompt_id in seen:
                raise ValueError("duplicate_or_conflicting_synthetic_text")
            seen.add(row.prompt_id)
            counts[(row.split, row.label)] += 1
            roles[(row.role, row.label)] += 1
            families.setdefault(row.family, set()).add(row.split)
            semantic_groups.setdefault(row.semantic_group, set()).add(row.split)
            key = (row.family, row.target_index, row.context_index, row.frame_index)
            pairs.setdefault(key, set()).add(row.label)
            lengths.append(len(row.text))
            handles[row.split].write(json.dumps(asdict(row), sort_keys=True, allow_nan=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    if any(counts[(s, c)] != n // 2 for s, n in SIZES.items() for c in (0, 1)):
        raise ValueError("synthetic_row_count_mismatch")
    if (any(len(v) != 1 for v in (*families.values(), *semantic_groups.values()))
            or any(v != {0, 1} for v in pairs.values())):
        raise ValueError("synthetic_family_or_pair_leakage")
    quality = {"rows": len(seen), "unique_normalized_texts": len(seen), "duplicate_texts": 0,
        "family_overlap": 0, "families": len(families), "semantic_group_overlap": 0,
        "semantic_groups": len(semantic_groups), "base_cases": len(pairs) // len(FRAMES),
        "target_intent_combinations": len(families) * 10,
        "context_and_frame_variants_are_not_independent_intents": True,
        "matched_class_pairs": len(pairs), "nuisance_metadata_label_balance": True,
        "nuisance_only_oracle_accuracy": .5, "min_characters": min(lengths), "max_characters": max(lengths),
        "counts": {s: {str(c): counts[(s, c)] for c in (0, 1)} for s in SIZES},
        "roles": {r: {str(c): roles[(r, c)] for c in (0, 1)} for r, _ in sorted(roles)},
        "prototype_similarity": prototype_similarity(catalog, splits)}
    manifest = {"schema_version": 1, "architecture": platform.machine(),
        "execution_scope": "native_linuxone" if platform.machine() == "s390x" else "operator_authorized_local_cpu",
        "source_commit": source_commit,
        "catalog_sha256": file_hash(catalog_path), "family_splits": splits, "quality": quality,
        "files": {s + ".jsonl": file_hash(output / (s + ".jsonl")) for s in SIZES},
        "limits": ["Compositional synthetic requests, not independent natural conversations.",
                   "Catalogue intent labels, not independent human adjudication of every rendered row.",
                   "No real-world safety guarantee or dataset-independent accuracy claim."]}
    write_json(output / "manifest.json", manifest)
    for path in output.iterdir():
        path.chmod(0o400)
    return manifest


def load_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [SyntheticRow(**json.loads(line)) for line in handle]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    report = generate(args.catalog, args.output, source_commit=args.source_commit)
    print(json.dumps({"rows": report["quality"]["rows"], "files": report["files"]}, sort_keys=True))


if __name__ == "__main__":
    main()

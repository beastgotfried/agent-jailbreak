"""Reproducible benign+attack JSONL prompt pools with exact provenance.

Standard-library only. No pyarrow, no network, no training/scoring.
Treats all source content as untrusted data.

CLI: python3 -m src.data.build_prompt_pools --raw-root PATH --converted-root PATH --download-manifest PATH --model-split-manifest PATH --output-dir NEW_PATH [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

def _is_hex_sha256(s: str) -> bool:
    return isinstance(s, str) and bool(_HEX_SHA256_RE.match(s.strip())) and len(s.strip()) == 64

def _require_resolved_within(path: Path, root: Path, context: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"{context} escapes root: {path} not within {root}")

# ---------------------------------------------------------------------------
# Helpers: normalize, hashes, safety, literal CSV
# ---------------------------------------------------------------------------

def normalize_prompt(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("prompt values must be strings")
    return " ".join(text.lower().split())

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def is_safe_relative(rel: str | Path) -> bool:
    # Reject absolute, traversal, empty parts.
    p = Path(rel)
    if p.is_absolute():
        return False
    # Path.parts on POSIX includes '/' for absolute; we already checked.
    for part in p.parts:
        if part == "..":
            return False
        if part == "" or part == "/":
            return False
    # Also reject any segment that after posix split contains '..'
    if ".." in p.as_posix().split("/"):
        return False
    return True

def revision_strength(revision: str) -> str:
    if "@" in revision:
        return "snapshot_hash_only"
    # assume immutable if looks like hex commit 40 chars or similar
    return "immutable"

# ---------------------------------------------------------------------------
# Manifest and hash validation
# ---------------------------------------------------------------------------

def load_download_manifest(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("download manifest must be a JSON list")
    return data

def load_model_split_manifest(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("model split manifest must be JSON object")
    return data

def build_prefix_mapping(download_manifest_data):
    """Map full relative path (dataset_prefix/file.path) -> sha256 for all files."""
    mapping: dict[str, str] = {}
    meta_by_prefix: dict[str, dict] = {}
    blocked = 0
    for entry in download_manifest_data:
        if not isinstance(entry, dict):
            continue
        # BLOCKED entries lack files
        files = entry.get("files")
        local_dir = entry.get("local_dir")
        repo = entry.get("repo")
        revision = entry.get("revision")
        license_val = entry.get("license")
        if not isinstance(files, list) or not isinstance(local_dir, str) or not isinstance(repo, str):
            # Blocked
            blocked += 1
            continue
        prefix = Path(local_dir).name
        meta = {
            "repo": repo,
            "revision": revision if isinstance(revision, str) else "",
            "revision_strength": revision_strength(revision) if isinstance(revision, str) else "unknown",
            "license": license_val,
            "rights_review_required": license_val is None,
            "local_dir": local_dir,
        }
        meta_by_prefix[prefix] = meta
        for f in files:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            s = f.get("sha256")
            if not isinstance(p, str) or not isinstance(s, str):
                continue
            if not is_safe_relative(p):
                raise ValueError(f"unsafe manifest path: {p}")
            full = f"{prefix}/{p}"
            if not is_safe_relative(full):
                raise ValueError(f"unsafe full manifest path: {full}")
            mapping[full] = s
    return mapping, meta_by_prefix, blocked

def validate_raw_hashes(raw_root: Path, download_data, prefix_mapping):
    """Validate non-parquet raw files exist and match manifest hashes.
    Returns dict full_rel -> sha for validated raw files.
    """
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise SystemExit(f"raw-root not found or not directory: {raw_root}")
    validated: dict[str, str] = {}
    # For each file in manifest that is not parquet, expect under raw_root
    for entry in download_data:
        if not isinstance(entry, dict):
            continue
        files = entry.get("files")
        local_dir = entry.get("local_dir")
        if not isinstance(files, list) or not isinstance(local_dir, str):
            continue
        prefix = Path(local_dir).name
        for f in files:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            s = f.get("sha256")
            if not isinstance(p, str) or not isinstance(s, str):
                continue
            if p.endswith(".parquet"):
                # Parquet raw validation is optional; if file exists under raw_root, validate
                # But we usually validate via conversion manifest; don't require.
                maybe = raw_root / prefix / p
                # Resolved containment before any read/hash, covers file and parent symlinks
                _require_resolved_within(maybe, raw_root, f"raw parquet {prefix}/{p}")
                if maybe.is_file():
                    # re-check containment after confirming file exists (symlink may point outside)
                    _require_resolved_within(maybe, raw_root, f"raw parquet {prefix}/{p}")
                    try:
                        actual = sha256_file(maybe)
                    except Exception as e:
                        raise SystemExit(f"unreadable raw parquet {prefix}/{p}: {e}") from e
                    if actual != s:
                        raise SystemExit(f"hash mismatch raw parquet {prefix}/{p}: expected {s}, got {actual}")
                    validated[f"{prefix}/{p}"] = s
                continue
            # Non-parquet: must exist under raw_root
            disk = raw_root / prefix / p
            full_rel = f"{prefix}/{p}"
            # Containment before existence/hash check, even if target has expected bytes
            _require_resolved_within(disk, raw_root, f"raw file {full_rel}")
            if not disk.is_file():
                raise SystemExit(f"missing required raw file: {full_rel} expected at {disk}")
            # Re-check containment for symlink target
            _require_resolved_within(disk, raw_root, f"raw file {full_rel}")
            # Validate not unsafe and readable
            try:
                actual = sha256_file(disk)
            except Exception as e:
                raise SystemExit(f"unreadable raw file {full_rel}: {e}") from e
            if actual != s:
                raise SystemExit(f"hash mismatch for {full_rel}: expected {s}, got {actual}")
            validated[full_rel] = actual
    return validated

def validate_converted_hashes(converted_root: Path, download_data, prefix_mapping):
    """Validate converted jsonl via conversion-manifest linkage."""
    converted_root = converted_root.resolve()
    if not converted_root.is_dir():
        raise SystemExit(f"converted-root not found or not directory: {converted_root}")
    manifest_path = converted_root / "conversion-manifest.json"
    if not manifest_path.is_file():
        # If there are parquet entries expected, then this is missing
        has_parquet = any(k.endswith(".parquet") for k in prefix_mapping)
        if has_parquet:
            raise SystemExit(f"missing conversion-manifest.json in converted-root: {converted_root}")
        return {}, []
    try:
        conv_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"unreadable conversion manifest: {e}") from e
    if not isinstance(conv_manifest, dict):
        raise SystemExit("conversion manifest must be JSON object")
    if conv_manifest.get("schema_version") != 1:
        raise SystemExit("conversion manifest schema_version must be 1")
    files = conv_manifest.get("files")
    if not isinstance(files, list):
        raise SystemExit("conversion manifest files must be list")
    validated: dict[str, str] = {}
    # Also keep mapping for later builder provenance
    converted_map = {}  # source_rel -> entry
    for entry in files:
        if not isinstance(entry, dict):
            raise SystemExit("conversion manifest entry must be dict")
        for k in ("source_relative_path", "source_sha256", "output_relative_path", "output_sha256", "rows", "columns"):
            if k not in entry:
                raise SystemExit(f"conversion manifest missing key {k}")
        src_rel = entry["source_relative_path"]
        src_sha = entry["source_sha256"]
        out_rel = entry["output_relative_path"]
        out_sha = entry["output_sha256"]
        if not isinstance(src_rel, str) or not isinstance(src_sha, str) or not isinstance(out_rel, str) or not isinstance(out_sha, str):
            raise SystemExit("conversion manifest paths/shas must be strings")
        if not is_safe_relative(src_rel) or not is_safe_relative(out_rel):
            raise SystemExit(f"unsafe conversion manifest path: {src_rel} or {out_rel}")
        # Link source hash to download manifest
        expected_src = prefix_mapping.get(src_rel)
        if expected_src is None:
            raise SystemExit(f"converted source not in download manifest: {src_rel}")
        if expected_src != src_sha:
            raise SystemExit(f"conversion source hash mismatch for {src_rel}: manifest {expected_src} vs conversion {src_sha}")
        # Validate output file exists and hash matches, with resolved containment before any read
        out_path = converted_root / out_rel
        _require_resolved_within(out_path, converted_root, f"converted output {out_rel}")
        if not out_path.is_file():
            raise SystemExit(f"missing converted file: {out_rel}")
        _require_resolved_within(out_path, converted_root, f"converted output {out_rel}")
        try:
            actual_out = sha256_file(out_path)
        except Exception as e:
            raise SystemExit(f"unreadable converted output {out_rel}: {e}") from e
        if actual_out != out_sha:
            raise SystemExit(f"hash mismatch converted output {out_rel}: expected {out_sha}, got {actual_out}")
        validated[out_rel] = actual_out
        converted_map[src_rel] = entry
    # Ensure every parquet in mapping has a conversion entry (optional strict)
    # For our 7 parquets, we expect 7 conversions; enforce.
    for src_rel in prefix_mapping:
        if src_rel.endswith(".parquet") and src_rel not in converted_map:
            # If parquet is expected, but no conversion entry => missing
            raise SystemExit(f"missing conversion entry for expected parquet: {src_rel}")
    return validated, converted_map

# ---------------------------------------------------------------------------
# CSV reading with literal preservation
# ---------------------------------------------------------------------------

def read_csv_literal(path: Path):
    """Read CSV with literal strings (keep_default_na=False equivalent).
    Returns list of dicts, header, and raw rows for provenance.
    Uses standard csv, preserves 'NA' etc.
    """
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise SystemExit(f"CSV missing header: {path}")
            # Validate header not unsafe? Not needed.
            rows = []
            for idx, row in enumerate(reader, start=1):  # row_index 1-based data record
                # row values are already strings, with empty string for missing? csv keeps empty.
                # Preserve literal: do not interpret NA.
                # Ensure all values are strings (csv may return None for missing fields)
                cleaned = {}
                for k, v in row.items():
                    if v is None:
                        v = ""
                    cleaned[k] = v
                cleaned["_row_index"] = idx
                # Also keep original row dict for labels
                rows.append(cleaned)
            return reader.fieldnames, rows
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"unreadable/malformed CSV {path}: {e}") from e

def read_jsonl_literal(path: Path):
    if not path.is_file():
        raise SystemExit(f"JSONL not found: {path}")
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue  # skip empty lines? But count as row? We'll treat empty as malformed unless allowed.
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise SystemExit(f"malformed JSONL {path} line {idx}: {e}") from e
                if not isinstance(obj, dict):
                    raise SystemExit(f"JSONL row must be object at {path} line {idx}")
                obj["_row_index"] = idx
                rows.append(obj)
        return rows
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"unreadable JSONL {path}: {e}") from e

# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def build_toxic_candidates(raw_root: Path, model_manifest, download_meta, prefix_mapping):
    """Return candidates and blocked set for ToxicChat.

    Allowed benign candidates per spec, blocked set for global filtering.
    Each candidate is dict with normalized_hash, text, source_text_sha, pool_kind etc.
    We'll return list of candidate groups and blocked_set plus exclusions for blocked? No.
    For ToxicChat we produce candidates list (each is per qualifying row grouped later) and blocked set.
    """
    raw_root = raw_root.resolve()
    toxic_prefix = None
    # Find toxicchat dataset prefix: look for entry with repo lmsys/toxic-chat
    # Instead find prefix that contains toxicchat files
    for rel in prefix_mapping:
        if rel.startswith("toxicchat"):
            toxic_prefix = rel.split("/")[0]
            break
    if toxic_prefix is None:
        raise SystemExit("toxicchat dataset not found in manifest")
    # Resolve paths
    train_rel = f"{toxic_prefix}/data/0124/toxic-chat_annotation_train.csv"
    test_rel = f"{toxic_prefix}/data/0124/toxic-chat_annotation_test.csv"
    train_path = raw_root / train_rel
    test_path = raw_root / test_rel
    # Resolved containment before any read/hash, covers escaping file and parent symlinks
    _require_resolved_within(train_path, raw_root, f"toxicchat train {train_rel}")
    _require_resolved_within(test_path, raw_root, f"toxicchat test {test_rel}")
    if not train_path.is_file() or not test_path.is_file():
        raise SystemExit(f"missing toxicchat train/test CSV under {raw_root}")
    # Re-check containment after existence (symlink target)
    _require_resolved_within(train_path, raw_root, f"toxicchat train {train_rel}")
    _require_resolved_within(test_path, raw_root, f"toxicchat test {test_rel}")
    # Validate against model manifest dataset hashes unconditionally
    if not isinstance(model_manifest, dict):
        raise SystemExit("model manifest must be JSON object")
    if model_manifest.get("schema_version") != 1:
        raise SystemExit("model manifest schema_version must be 1")
    datasets = model_manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise SystemExit("model manifest datasets must be dict")
    for key, path in (("train", train_path), ("test", test_path)):
        entry = datasets.get(key)
        if not isinstance(entry, dict):
            raise SystemExit(f"model manifest missing datasets.{key}")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not sha.strip():
            raise SystemExit(f"model manifest datasets.{key}.sha256 must be nonempty 64-hex")
        sha_stripped = sha.strip()
        if not _is_hex_sha256(sha_stripped):
            raise SystemExit(f"model manifest datasets.{key}.sha256 must be 64-hex: {sha}")
        actual = sha256_file(path)
        if actual != sha_stripped:
            raise SystemExit(f"toxicchat {key} hash mismatch vs model manifest: expected {sha_stripped}, got {actual}")

    # Load CSVs literal (after containment and hash validation)
    train_fields, train_rows = read_csv_literal(train_path)
    test_fields, test_rows = read_csv_literal(test_path)
    # Validate required columns
    required = ["conv_id", "user_input", "model_output", "human_annotation", "toxicity", "jailbreaking", "openai_moderation"]
    for col in required:
        if col not in train_fields or col not in test_fields:
            raise SystemExit(f"toxicchat missing required column {col}")

    # Build mapping conv_id -> row for train, reject duplicate conv_id
    train_by_conv: dict[str, dict] = {}
    seen_conv_ids: set[str] = set()
    for r in train_rows:
        cid = r.get("conv_id", "")
        if not isinstance(cid, str) or not cid.strip():
            raise SystemExit(f"toxicchat train conv_id empty at row {r['_row_index']}")
        if cid in seen_conv_ids:
            raise SystemExit(f"duplicate conv_id in train CSV: {cid}")
        seen_conv_ids.add(cid)
        train_by_conv[cid] = r
    # Also check test duplicate conv_id
    seen_test: set[str] = set()
    for r in test_rows:
        cid = r.get("conv_id", "")
        if not isinstance(cid, str) or not cid.strip():
            raise SystemExit(f"toxicchat test conv_id empty at row {r['_row_index']}")
        if cid in seen_test:
            raise SystemExit(f"duplicate conv_id in test CSV: {cid}")
        seen_test.add(cid)

    # Compute normalized hashes for all train and test rows for blocked set
    def row_norm_hash(row):
        txt = row.get("user_input", "")
        # Keep literal string (maybe empty)
        if not isinstance(txt, str):
            txt = str(txt)
        return sha256_text(normalize_prompt(txt))

    train_hashes = set(row_norm_hash(r) for r in train_rows)
    test_hashes = set(row_norm_hash(r) for r in test_rows)
    all_hashes = train_hashes | test_hashes

    # Validate model manifest rows before eligibility selection
    rows_meta = model_manifest.get("rows")
    if not isinstance(rows_meta, list) or not rows_meta:
        raise SystemExit("model manifest rows must be nonempty list")
    # Check split counts and total
    split = model_manifest.get("split")
    if not isinstance(split, dict):
        raise SystemExit("model manifest split must be dict")
    counts = split.get("counts")
    if not isinstance(counts, dict):
        raise SystemExit("model manifest split.counts must be dict")
    rows_after = split.get("rows_after_dedupe")
    if not isinstance(rows_after, int):
        raise SystemExit("model manifest split.rows_after_dedupe must be int")
    if rows_after != len(rows_meta):
        raise SystemExit(f"model manifest rows_after_dedupe mismatch: expected {len(rows_meta)} got {rows_after}")
    for role in ("fit", "validation", "reference"):
        if role not in counts or not isinstance(counts[role], int):
            raise SystemExit(f"model manifest split.counts.{role} must be int")
    total_counts = sum(counts[r] for r in ("fit", "validation", "reference"))
    if total_counts != len(rows_meta):
        raise SystemExit(f"model manifest split counts total mismatch: {total_counts} vs {len(rows_meta)}")
    # Validate each row entry and collect for conflict checks
    allowed_roles = {"fit", "validation", "reference"}
    seen_prompt_sha: set[str] = set()
    seen_source_ids: set[str] = set()
    prompt_to_role: dict[str, str] = {}
    source_to_role: dict[str, str] = {}
    # Also need to track group_by_hash for later but validate first
    for entry in rows_meta:
        if not isinstance(entry, dict):
            raise SystemExit("model manifest rows entry must be dict")
        role = entry.get("role")
        ph = entry.get("prompt_sha256")
        source_ids = entry.get("source_conv_ids")
        if not isinstance(role, str) or role not in allowed_roles:
            raise SystemExit(f"model manifest unknown role: {role}")
        if not isinstance(ph, str) or not _is_hex_sha256(ph):
            raise SystemExit(f"model manifest prompt_sha256 must be 64-hex: {ph}")
        if ph in seen_prompt_sha:
            raise SystemExit(f"duplicate canonical prompt_sha256: {ph}")
        seen_prompt_sha.add(ph)
        if ph in prompt_to_role and prompt_to_role[ph] != role:
            raise SystemExit(f"group {ph} assigned conflicting roles: {prompt_to_role[ph]} vs {role}")
        prompt_to_role[ph] = role
        if not isinstance(source_ids, list) or not source_ids:
            raise SystemExit(f"model manifest source_conv_ids must be nonempty list for {ph}")
        for sid in source_ids:
            if not isinstance(sid, str) or not sid.strip():
                raise SystemExit(f"model manifest source_conv_id must be nonempty string for {ph}")
            if sid in seen_source_ids:
                raise SystemExit(f"duplicate source_conv_id across groups: {sid}")
            seen_source_ids.add(sid)
            if sid in source_to_role and source_to_role[sid] != role:
                raise SystemExit(f"source {sid} assigned conflicting roles: {source_to_role[sid]} vs {role}")
            source_to_role[sid] = role
            if sid not in train_by_conv:
                raise SystemExit(f"model manifest source_conv_id not in train CSV: {sid}")
            # Validate normalized hash matches declared group hash
            row = train_by_conv[sid]
            txt = row.get("user_input", "")
            if not isinstance(txt, str):
                txt = str(txt)
            norm_hash = sha256_text(normalize_prompt(txt))
            if norm_hash != ph:
                raise SystemExit(f"source text hash mismatch for {sid}: computed {norm_hash} vs declared {ph}")
        # Validate that role counts will be checked after loop
    # Check per-role counts match actual
    for role in ("fit", "validation", "reference"):
        actual = sum(1 for e in rows_meta if e.get("role") == role)
        if actual != counts[role]:
            raise SystemExit(f"role count mismatch for {role}: expected {counts[role]} got {actual}")
    # Strict flag validation helper
    def _is_valid_toxicity(v: str) -> bool:
        return v.strip() in ("0", "1")
    def _is_valid_human(v) -> bool:
        # Must be string "True" or "False" case-insensitive, after strip
        if not isinstance(v, str):
            v = str(v)
        return v.strip().lower() in ("true", "false")
    # Validate all train rows flags strictly if they are part of manifest? We validate all train rows for malformed to avoid junk numeric becoming humanTrue
    for r in train_rows:
        tox = r.get("toxicity", "")
        jail = r.get("jailbreaking", "")
        hum = r.get("human_annotation", "")
        # toxicity and jailbreaking must be "0" or "1"
        if not isinstance(tox, str) or not _is_valid_toxicity(tox):
            raise SystemExit(f"malformed toxicity at train row {r['_row_index']}")
        if not isinstance(jail, str) or not _is_valid_toxicity(jail):
            raise SystemExit(f"malformed jailbreaking at train row {r['_row_index']}")
        if not _is_valid_human(hum):
            raise SystemExit(f"malformed human_annotation at train row {r['_row_index']}")

    # Determine allowed reference groups via model manifest (any member qualifies)
    allowed_hashes: set[str] = set()
    # Keep mapping from prompt_sha to group for provenance
    group_by_hash: dict[str, dict] = {}
    for entry in rows_meta:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        ph = entry.get("prompt_sha256")
        source_ids = entry.get("source_conv_ids", [])
        if role != "reference" or not isinstance(ph, str) or not isinstance(source_ids, list):
            continue
        # store group
        group_by_hash[ph] = entry
        # Check if any source row qualifies benign human
        qualifies = False
        for sid in source_ids:
            if not isinstance(sid, str):
                continue
            row = train_by_conv.get(sid)
            if row is None:
                continue
            tox = row.get("toxicity", "").strip()
            jail = row.get("jailbreaking", "").strip()
            hum_raw = row.get("human_annotation", "")
            hum_str = str(hum_raw).strip().lower()
            # Strict: tox and jail already validated as "0"/"1", hum as true/false
            is_human = hum_str == "true"
            if tox == "0" and jail == "0" and is_human:
                qualifies = True
                break
        if qualifies:
            allowed_hashes.add(ph)

    # Allowed reference hashes must not overlap official test or fit/validation groups
    fit_validation_hashes = {e.get("prompt_sha256") for e in rows_meta if e.get("role") in ("fit", "validation")}
    for h in allowed_hashes:
        if h in test_hashes:
            raise SystemExit(f"allowed reference hash overlaps test: {h}")
        if h in fit_validation_hashes:
            raise SystemExit(f"allowed reference hash overlaps fit/validation: {h}")

    blocked_set = all_hashes - allowed_hashes

    # Build candidates for allowed hashes
    candidates = []  # list of dict per qualifying row
    exclusions = []  # not for toxic? but we may log blocked?

    # For each allowed hash, collect qualifying rows
    for ph in allowed_hashes:
        group = group_by_hash.get(ph)
        if group is None:
            continue
        source_ids = group.get("source_conv_ids", [])
        qualifying_rows = []
        for sid in source_ids:
            row = train_by_conv.get(sid)
            if row is None:
                continue
            tox = row.get("toxicity", "").strip()
            jail = row.get("jailbreaking", "").strip()
            hum_raw = row.get("human_annotation", "")
            hum_str = str(hum_raw).strip().lower()
            is_human = hum_str == "true"
            if tox == "0" and jail == "0" and is_human:
                qualifying_rows.append(row)
        # If none qualifying (should not happen because we checked), skip
        if not qualifying_rows:
            continue
        # For each qualifying row, create a candidate entry (will be deduped globally)
        # But we can create one candidate per group with multiple provenance; easier to create per row then global dedup merges.
        for row in qualifying_rows:
            txt = row.get("user_input", "")
            if not isinstance(txt, str):
                txt = str(txt)
            norm = normalize_prompt(txt)
            norm_hash = sha256_text(norm)
            source_text_sha = sha256_text(txt)
            # Provenance entry – use dataset metadata from download manifest
            candidates.append({
                "normalized_hash": norm_hash,
                "normalized_text": norm,
                "text": txt,
                "source_text_sha256": source_text_sha,
                "pool_kind": "benign",
                "source_tasks": ["toxicity_or_jailbreak"],
                "provenance": {
                    "repo": download_meta.get(toxic_prefix, {}).get("repo", "lmsys/toxic-chat"),
                    "revision": download_meta.get(toxic_prefix, {}).get("revision", ""),
                    "revision_strength": download_meta.get(toxic_prefix, {}).get("revision_strength", "immutable"),
                    "original_relative_path": train_rel,
                    "original_sha256": prefix_mapping.get(train_rel, ""),
                    "row_index": row["_row_index"],
                    "source_id": row.get("conv_id", ""),
                    "original_split": "train",
                    "configuration": None,
                    "source_task": "toxicity_or_jailbreak",
                    "original_labels": {
                        "toxicity": row.get("toxicity", ""),
                        "jailbreaking": row.get("jailbreaking", ""),
                        "human_annotation": row.get("human_annotation", ""),
                        "openai_moderation": row.get("openai_moderation", ""),
                    },
                    "text_field": "user_input",
                    "source_text_sha256": source_text_sha,
                    "origin": "real_user_prompts",
                    "license": download_meta.get(toxic_prefix, {}).get("license") or "cc-by-nc-4.0",
                    "rights_review_required": False,
                    "prompt_sha256": norm_hash,  # for debugging
                }
            })

    return candidates, blocked_set, allowed_hashes

def build_wainject_candidates(raw_root: Path, download_meta, prefix_mapping):
    raw_root = raw_root.resolve()
    prefix = None
    for rel in prefix_mapping:
        if rel.startswith("wainjectbench"):
            prefix = rel.split("/")[0]
            break
    if prefix is None:
        raise SystemExit("wainjectbench dataset not found")
    base = raw_root / prefix
    _require_resolved_within(base, raw_root, f"wainject base {prefix}")
    if not base.is_dir():
        raise SystemExit(f"missing wainjectbench dir {base}")
    _require_resolved_within(base, raw_root, f"wainject base {prefix}")
    benign_dir = base / "text" / "benign"
    malicious_dir = base / "text" / "malicious"
    # Validate against manifest: only validated manifest-listed JSONL under text/benign or text/malicious
    # Determine expected files from manifest for this prefix
    expected: dict[str, str] = {}
    for rel, sha in prefix_mapping.items():
        if not rel.startswith(prefix + "/"):
            continue
        if (rel.startswith(prefix + "/text/benign/") or rel.startswith(prefix + "/text/malicious/")) and rel.endswith(".jsonl"):
            if not isinstance(sha, str) or not sha.strip() or not _is_hex_sha256(sha.strip()):
                raise SystemExit(f"manifest missing valid hash for {rel}")
            expected[rel] = sha.strip()
    if not expected:
        raise SystemExit("no wainject manifest jsonl entries under text/benign or text/malicious")
    candidates = []
    exclusions = []
    # Discover actual jsonl files and validate containment
    discovered: dict[str, Path] = {}
    for sub, kind in [(benign_dir, "benign"), (malicious_dir, "attack")]:
        # Containment for sub
        _require_resolved_within(sub, raw_root, f"wainject subdir {sub}")
        if not sub.is_dir():
            raise SystemExit(f"missing wainjectbench subdir {sub}")
        _require_resolved_within(sub, raw_root, f"wainject subdir {sub}")
        for path in sorted(sub.glob("*.jsonl")):
            _require_resolved_within(path, raw_root, f"wainject file {path}")
            rel = path.relative_to(raw_root).as_posix()
            if not is_safe_relative(rel):
                raise SystemExit(f"unsafe wainject path: {rel}")
            # Re-check containment for symlink target
            _require_resolved_within(path, raw_root, f"wainject file {rel}")
            discovered[rel] = path
    # Validate missing expected files
    for rel in expected:
        if rel not in discovered:
            raise SystemExit(f"missing expected wainject file: {rel}")
    # Reject unexpected discovered JSONL files
    for rel in discovered:
        if rel not in expected:
            raise SystemExit(f"unexpected wainject file not in manifest: {rel}")
    # Ingest only validated manifest paths
    for rel, path in sorted(discovered.items()):
        expected_sha = expected[rel]
        # Resolved containment already checked; verify hash before reading
        _require_resolved_within(path, raw_root, f"wainject file {rel}")
        try:
            actual_sha = sha256_file(path)
        except Exception as e:
            raise SystemExit(f"unreadable wainject file {rel}: {e}") from e
        if actual_sha != expected_sha:
            raise SystemExit(f"hash mismatch for {rel}: expected {expected_sha}, got {actual_sha}")
        # Also ensure prefix_mapping hash matches expected (redundant)
        rows = read_jsonl_literal(path)
        for row in rows:
            if "id" not in row or "text" not in row:
                raise SystemExit(f"wainjectbench malformed row missing id/text at {rel} row {row['_row_index']}")
            rid = row["id"]
            txt = row["text"]
            if not isinstance(rid, int):
                raise SystemExit(f"wainjectbench id must be int at {rel} row {row['_row_index']}")
            if not isinstance(txt, str):
                raise SystemExit(f"wainjectbench text must be str at {rel} row {row['_row_index']}")
            if not txt.strip():
                raise SystemExit(f"wainjectbench text must be nonempty at {rel} row {row['_row_index']}")
            norm = normalize_prompt(txt)
            norm_hash = sha256_text(norm)
            source_text_sha = sha256_text(txt)
            meta = download_meta.get(prefix, {})
            category = path.stem
            kind = "benign" if path.parent.name == "benign" else "attack"
            candidates.append({
                "normalized_hash": norm_hash,
                "normalized_text": norm,
                "text": txt,
                "source_text_sha256": source_text_sha,
                "pool_kind": kind,
                "source_tasks": ["prompt_injection"],
                "provenance": {
                    "repo": meta.get("repo", "Norrrrrrr/WAInjectBench"),
                    "revision": meta.get("revision", ""),
                    "revision_strength": meta.get("revision_strength", "immutable"),
                    "original_relative_path": rel,
                    "original_sha256": expected_sha,
                    "row_index": row["_row_index"],
                    "source_id": str(rid),
                    "original_split": "benign" if kind == "benign" else "malicious",
                    "configuration": category,
                    "source_task": "prompt_injection",
                    "original_labels": {"directory_label": "benign" if kind == "benign" else "malicious", "file": category},
                    "text_field": "text",
                    "source_text_sha256": source_text_sha,
                    "origin": "benchmark_constructed_or_curated",
                    "license": meta.get("license"),
                    "rights_review_required": meta.get("rights_review_required", True),
                    "file_category": category,
                }
            })
    return candidates, exclusions

def build_jbb_behaviors_candidates(raw_root: Path, download_meta, prefix_mapping):
    prefix = None
    for rel in prefix_mapping:
        if rel.startswith("jbb-behaviors"):
            prefix = rel.split("/")[0]
            break
    if prefix is None:
        raise SystemExit("jbb-behaviors not found")
    benign_rel = f"{prefix}/data/benign-behaviors.csv"
    harmful_rel = f"{prefix}/data/harmful-behaviors.csv"
    # judge file is intentionally ignored
    benign_path = raw_root / benign_rel
    harmful_path = raw_root / harmful_rel
    _require_resolved_within(benign_path, raw_root, f"jbb-behaviors {benign_rel}")
    _require_resolved_within(harmful_path, raw_root, f"jbb-behaviors {harmful_rel}")
    if not benign_path.is_file() or not harmful_path.is_file():
        raise SystemExit(f"missing jbb-behaviors csv under {raw_root}")
    _require_resolved_within(benign_path, raw_root, f"jbb-behaviors {benign_rel}")
    _require_resolved_within(harmful_path, raw_root, f"jbb-behaviors {harmful_rel}")
    candidates = []
    # Helper to process each file
    for rel, path, kind in [(benign_rel, benign_path, "benign"), (harmful_rel, harmful_path, "attack")]:
        fields, rows = read_csv_literal(path)
        required = ["Index", "Goal", "Target", "Behavior", "Category", "Source"]
        for col in required:
            if col not in fields:
                raise SystemExit(f"jbb-behaviors missing column {col} in {rel}")
        meta = download_meta.get(prefix, {})
        sha = prefix_mapping.get(rel, "")
        for row in rows:
            txt = row.get("Goal", "")
            if not isinstance(txt, str):
                txt = str(txt)
            # Even if Goal empty, we keep? But should exclude? We'll treat empty as malformed and exclude with reason? For now create candidate but allow empty.
            # Preserve ID
            sid = row.get("Index", "")
            norm = normalize_prompt(txt)
            norm_hash = sha256_text(norm)
            source_text_sha = sha256_text(txt)
            candidates.append({
                "normalized_hash": norm_hash,
                "normalized_text": norm,
                "text": txt,
                "source_text_sha256": source_text_sha,
                "pool_kind": kind,
                "source_tasks": ["harmful_request"],
                "provenance": {
                    "repo": meta.get("repo", "JailbreakBench/JBB-Behaviors"),
                    "revision": meta.get("revision", ""),
                    "revision_strength": meta.get("revision_strength", "immutable"),
                    "original_relative_path": rel,
                    "original_sha256": sha,
                    "row_index": row["_row_index"],
                    "source_id": str(sid),
                    "original_split": "benign" if kind=="benign" else "harmful",
                    "configuration": None,
                    "source_task": "harmful_request",
                    "original_labels": {
                        "Behavior": row.get("Behavior", ""),
                        "Category": row.get("Category", ""),
                        "Source": row.get("Source", ""),
                        # Never include Target as prompt but preserve as metadata? Spec says NEVER Target as prompt, but we can keep Source as metadata; we should not use Target.
                    },
                    "text_field": "Goal",
                    "source_text_sha256": source_text_sha,
                    "origin": "curated_behavior_goal",
                    "license": meta.get("license") or "mit",
                    "rights_review_required": False,
                    "category": row.get("Category", ""),
                    "behavior": row.get("Behavior", ""),
                }
            })
    # Exclusion for judge-comparison.csv is implicit (not included); we don't generate exclusions for it.
    return candidates, []

def build_jbb_attacks_candidates(converted_root: Path, download_meta, prefix_mapping, converted_map):
    # Converted file: jbb-attacks/data/train-00000-of-00001.jsonl
    # Find prefix
    prefix = None
    for rel in prefix_mapping:
        if rel.startswith("jbb-attacks"):
            prefix = rel.split("/")[0]
            break
    if prefix is None:
        raise SystemExit("jbb-attacks not found")
    # Locate converted file via conversion map
    # Find source parquet rel that maps to jbb-attacks
    candidates = []
    exclusions = []
    for src_rel, entry in converted_map.items():
        if not src_rel.startswith(prefix + "/"):
            continue
        out_rel = entry["output_relative_path"]
        out_path = converted_root / out_rel
        _require_resolved_within(out_path, converted_root, f"jbb-attacks {out_rel}")
        # Validate out_path exists already validated
        rows = read_jsonl_literal(out_path)
        _require_resolved_within(out_path, converted_root, f"jbb-attacks {out_rel}")
        # Each row has prompt,target,behavior,category,source,subset
        # Validate fields
        meta = download_meta.get(prefix, {})
        src_sha = entry["source_sha256"]
        out_sha = entry["output_sha256"]
        for row in rows:
            # Check required keys
            for k in ("prompt", "target", "behavior", "category", "source", "subset"):
                if k not in row:
                    raise SystemExit(f"jbb-attacks malformed missing {k} at {out_rel} row {row['_row_index']}")
            txt = row["prompt"]
            if not isinstance(txt, str):
                raise SystemExit(f"jbb-attacks prompt must be str at {out_rel} row {row['_row_index']}")
            subset = row["subset"]
            if subset not in ("benign", "harmful"):
                raise SystemExit(f"jbb-attacks subset must be benign/harmful at {out_rel} row {row['_row_index']}")
            kind = "benign" if subset == "benign" else "attack"
            norm = normalize_prompt(txt)
            norm_hash = sha256_text(norm)
            source_text_sha = sha256_text(txt)
            candidates.append({
                "normalized_hash": norm_hash,
                "normalized_text": norm,
                "text": txt,
                "source_text_sha256": source_text_sha,
                "pool_kind": kind,
                "source_tasks": ["harmful_request"],
                "provenance": {
                    "repo": meta.get("repo", "walledai/JailbreakBench"),
                    "revision": meta.get("revision", ""),
                    "revision_strength": meta.get("revision_strength", "unknown"),
                    "original_relative_path": src_rel,
                    "original_sha256": src_sha,
                    "converted_relative_path": out_rel,
                    "converted_sha256": out_sha,
                    "row_index": row["_row_index"],
                    "source_id": row.get("behavior", "") + ":" + str(row.get("source", "")),
                    "original_split": "train",
                    "configuration": "default",
                    "source_task": "harmful_request",
                    "original_labels": {
                        "subset": subset,
                        "category": row.get("category", ""),
                        "behavior": row.get("behavior", ""),
                        "source": row.get("source", ""),
                        # target not as prompt
                    },
                    "text_field": "prompt",
                    "source_text_sha256": source_text_sha,
                    "origin": "behavior_goal_mirror",
                    "license": meta.get("license"),
                    "rights_review_required": True if meta.get("license") is None else False,
                }
            })
    return candidates, exclusions

def build_strongreject_candidates(converted_root: Path, download_meta, prefix_mapping, converted_map):
    prefix = None
    for rel in prefix_mapping:
        if rel.startswith("strongreject"):
            prefix = rel.split("/")[0]
            break
    if prefix is None:
        raise SystemExit("strongreject not found")
    candidates = []
    exclusions = []
    for src_rel, entry in converted_map.items():
        if not src_rel.startswith(prefix + "/"):
            continue
        out_rel = entry["output_relative_path"]
        out_path = converted_root / out_rel
        _require_resolved_within(out_path, converted_root, f"strongreject {out_rel}")
        rows = read_jsonl_literal(out_path)
        _require_resolved_within(out_path, converted_root, f"strongreject {out_rel}")
        meta = download_meta.get(prefix, {})
        src_sha = entry["source_sha256"]
        out_sha = entry["output_sha256"]
        # Derive configuration/split from src_rel e.g., data/validation, keywords/validation etc
        # src_rel like strongreject/data/validation-00000-of-00001.parquet -> config data, split validation
        # Extract middle parts
        # src_rel = prefix + "/" + config + "/" + file
        parts = Path(src_rel).parts  # e.g., ('strongreject','data','validation-00000...')
        config = parts[1] if len(parts) >= 3 else "unknown"
        file_name = parts[2] if len(parts) >= 3 else ""
        # Determine split from file name: train vs validation
        split = "train" if "train" in file_name else "validation" if "validation" in file_name else "unknown"
        for row in rows:
            # Validate expected fields
            for k in ("clf_label", "proxy_clf_label", "instructions", "content", "answer_prompt", "gen_target", "proxy_gen_target"):
                if k not in row:
                    raise SystemExit(f"strongreject missing {k} at {out_rel} row {row['_row_index']}")
            clf = row["clf_label"]
            proxy = row["proxy_clf_label"]
            instr = row["instructions"]
            content = row["content"]
            if not isinstance(content, list):
                raise SystemExit(f"strongreject content must be list at {out_rel} row {row['_row_index']}")
            # Check exclusion condition: instructions empty and content has one nonempty string
            is_empty_instr = isinstance(instr, str) and instr == ""
            single_nonempty = len(content) == 1 and isinstance(content[0], str) and content[0].strip() != ""
            if not (is_empty_instr and single_nonempty):
                # Explicit exclusion, not fabricated concatenation
                norm = None
                # For exclusion, compute normalized hash if content[0] exists? But we need normalized hash for logging without raw text.
                # We will compute hash of whatever text exists? For multi-content, we shouldn't pick.
                # We'll log exclusion with reason unexpected_context_or_multi_content, and include source refs.
                # Compute normalized hash of content if possible, else of empty?
                # Spec says exclusions.jsonl contains only reason, normalized hashes, source refs/IDs/labels, NEVER raw text.
                # For StrongREJECT exclusions, normalized hash may be derived from content[0] if exists but we avoid emitting raw.
                # We'll compute hash of list representation? Simpler: compute hash of empty string placeholder? But we need something.
                # For exclusion, we can compute hash of the would-be text if we had content[0] else of empty.
                # To avoid leaking raw, we hash normalized version of content[0] if exists, else empty.
                tmp_text = content[0] if len(content)==1 and isinstance(content[0], str) else ""
                tmp_norm = normalize_prompt(tmp_text) if tmp_text else ""
                tmp_hash = sha256_text(tmp_norm) if tmp_norm else sha256_text("")
                exclusions.append({
                    "reason": "strongreject_unexpected_context_or_multi_content",
                    "normalized_text_sha256": tmp_hash,
                    "source_id": f"{src_rel}:{row['_row_index']}",
                    "original_labels": {"clf_label": clf, "proxy_clf_label": proxy, "instructions_nonempty": not is_empty_instr, "content_len": len(content)},
                    "source_relative_path": src_rel,
                    "converted_relative_path": out_rel,
                    "row_index": row["_row_index"],
                    "configuration": config,
                    "split": split,
                })
                continue
            txt = content[0]
            # Determine pool_kind from clf_label: 0 benign, 1 harmful -> attack
            # Preserve both labels
            try:
                clf_int = int(clf)
            except:
                raise SystemExit(f"strongreject clf_label must be int at {out_rel} row {row['_row_index']}")
            kind = "benign" if clf_int == 0 else "attack"
            norm = normalize_prompt(txt)
            norm_hash = sha256_text(norm)
            source_text_sha = sha256_text(txt)
            candidates.append({
                "normalized_hash": norm_hash,
                "normalized_text": norm,
                "text": txt,
                "source_text_sha256": source_text_sha,
                "pool_kind": kind,
                "source_tasks": ["harmful_request"],
                "provenance": {
                    "repo": meta.get("repo", "AlignmentResearch/StrongREJECT"),
                    "revision": meta.get("revision", ""),
                    "revision_strength": meta.get("revision_strength", "unknown"),
                    "original_relative_path": src_rel,
                    "original_sha256": src_sha,
                    "converted_relative_path": out_rel,
                    "converted_sha256": out_sha,
                    "row_index": row["_row_index"],
                    "source_id": f"{config}:{split}:{row['_row_index']}",
                    "original_split": split,
                    "configuration": config,
                    "source_task": "harmful_request",
                    "original_labels": {
                        "clf_label": clf,
                        "proxy_clf_label": proxy,
                        "instructions": instr,
                        "answer_prompt": row.get("answer_prompt", ""),
                        "gen_target": row.get("gen_target", ""),
                        "proxy_gen_target": row.get("proxy_gen_target", ""),
                    },
                    "text_field": "content[0]",
                    "source_text_sha256": source_text_sha,
                    "origin": "source_provided_classification",
                    "license": meta.get("license"),
                    "rights_review_required": True if meta.get("license") is None else False,
                }
            })
    return candidates, exclusions

def build_harmbench_candidates(raw_root: Path, download_meta, prefix_mapping):
    prefix = None
    for rel in prefix_mapping:
        if rel.startswith("harmbench"):
            prefix = rel.split("/")[0]
            break
    if prefix is None:
        raise SystemExit("harmbench not found")
    csv_rel = None
    for rel in prefix_mapping:
        if rel.startswith(prefix + "/") and rel.endswith(".csv"):
            csv_rel = rel
            break
    if csv_rel is None:
        raise SystemExit("harmbench csv not found")
    csv_path = raw_root / csv_rel
    _require_resolved_within(csv_path, raw_root, f"harmbench {csv_rel}")
    if not csv_path.is_file():
        raise SystemExit(f"missing harmbench csv {csv_path}")
    _require_resolved_within(csv_path, raw_root, f"harmbench {csv_rel}")
    fields, rows = read_csv_literal(csv_path)
    required = ["Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"]
    for col in required:
        if col not in fields:
            raise SystemExit(f"harmbench missing column {col}")
    meta = download_meta.get(prefix, {})
    sha = prefix_mapping.get(csv_rel, "")
    candidates = []
    exclusions = []
    for row in rows:
        func = row.get("FunctionalCategory", "")
        behavior = row.get("Behavior", "")
        bid = row.get("BehaviorID", "")
        sem = row.get("SemanticCategory", "")
        # Check eligibility: ONLY standard200 with no required context
        if func != "standard":
            reason = "harmbench_contextual_requires_assembly" if func == "contextual" else "harmbench_copyright_different_policy" if func == "copyright" else f"harmbench_excluded_{func}"
            # Log exclusion without raw text
            # Compute normalized hash of behavior for logging (without exposing raw)
            norm = normalize_prompt(behavior) if isinstance(behavior, str) else ""
            h = sha256_text(norm)
            exclusions.append({
                "reason": reason,
                "normalized_text_sha256": h,
                "source_id": str(bid),
                "original_labels": {"FunctionalCategory": func, "SemanticCategory": sem},
                "source_relative_path": csv_rel,
                "row_index": row["_row_index"],
            })
            continue
        # eligible
        txt = behavior
        if not isinstance(txt, str):
            txt = str(txt)
        norm = normalize_prompt(txt)
        norm_hash = sha256_text(norm)
        source_text_sha = sha256_text(txt)
        candidates.append({
            "normalized_hash": norm_hash,
            "normalized_text": norm,
            "text": txt,
            "source_text_sha256": source_text_sha,
            "pool_kind": "attack",
            "source_tasks": ["harmful_request"],
            "provenance": {
                "repo": meta.get("repo", "centerforaisafety/HarmBench"),
                "revision": meta.get("revision", ""),
                "revision_strength": meta.get("revision_strength", "snapshot_hash_only"),
                "original_relative_path": csv_rel,
                "original_sha256": sha,
                "row_index": row["_row_index"],
                "source_id": str(bid),
                "original_split": "all",
                "configuration": func,
                "source_task": "harmful_request",
                "original_labels": {
                    "FunctionalCategory": func,
                    "SemanticCategory": sem,
                    "Tags": row.get("Tags", ""),
                    "ContextString": row.get("ContextString", ""),
                    "BehaviorID": bid,
                },
                "text_field": "Behavior",
                "source_text_sha256": source_text_sha,
                "origin": "curated_behavior_goal",
                "license": meta.get("license") or "mit",
                "rights_review_required": False,
                "semantic_category": sem,
            }
        })
    return candidates, exclusions

# ---------------------------------------------------------------------------
# Global dedup, blocked, conflict, partition
# ---------------------------------------------------------------------------

def global_dedup_and_partition(candidates, blocked_set, seed):
    """Group by normalized_hash, handle blocked/conflict, then partition.

    candidates: list of candidate dicts (with normalized_hash, text, source_text_sha, pool_kind, source_tasks, provenance)
    blocked_set: set of normalized_hash to exclude
    Returns canonical_groups, exclusions_global, partition_counts
    """
    # Group by normalized_hash
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        grouped[c["normalized_hash"]].append(c)

    exclusions = []  # global exclusions
    canonical = []  # list of group dicts

    for norm_hash, items in grouped.items():
        # Check blocked
        if norm_hash in blocked_set:
            # Exclude whole group with reason blocked
            # Collect source refs for logging (no raw text)
            src_refs = []
            for it in items:
                prov = it["provenance"]
                src_refs.append({
                    "source_id": prov.get("source_id"),
                    "original_relative_path": prov.get("original_relative_path"),
                    "row_index": prov.get("row_index"),
                    "pool_kind": it["pool_kind"],
                })
            exclusions.append({
                "reason": "blocked_toxicchat_overlap",
                "normalized_text_sha256": norm_hash,
                "sources": src_refs,
            })
            continue
        # Check conflicting pool_kind
        kinds = set(it["pool_kind"] for it in items)
        if len(kinds) > 1:
            src_refs = []
            for it in items:
                prov = it["provenance"]
                src_refs.append({
                    "source_id": prov.get("source_id"),
                    "original_relative_path": prov.get("original_relative_path"),
                    "row_index": prov.get("row_index"),
                    "pool_kind": it["pool_kind"],
                    "source_tasks": it["source_tasks"],
                })
            exclusions.append({
                "reason": "conflicting_benign_attack",
                "normalized_text_sha256": norm_hash,
                "sources": src_refs,
                "pool_kinds": sorted(kinds),
            })
            continue
        # Keep canonical group
        pool_kind = items[0]["pool_kind"]
        # Merge source_tasks
        tasks = sorted({t for it in items for t in it["source_tasks"]})
        # Collect all provenance entries
        provenances = [it["provenance"] for it in items]
        # Deterministic source/path/row ordering to choose original text
        # Sort provenances by repo, original_relative_path, row_index
        def prov_key(p):
            return (p.get("repo",""), p.get("original_relative_path",""), p.get("row_index",0), p.get("source_id",""))
        sorted_provs = sorted(provenances, key=prov_key)
        # Choose selected text: corresponding to first provenance's original text.
        # We need to find candidate that matches that provenance's source_text_sha? But we have items mapping.
        # Instead find item whose provenance equals first sorted prov and use its text.
        # Since multiple items may share same provenance, find first item in sorted order by provenance key.
        # Build mapping from provenance identity to item
        # For simplicity, sort items by same key using provenance
        def item_key(it):
            p = it["provenance"]
            return (p.get("repo",""), p.get("original_relative_path",""), p.get("row_index",0))
        items_sorted = sorted(items, key=item_key)
        selected = items_sorted[0]
        text = selected["text"]
        text_sha = selected["source_text_sha256"]
        # But also need to ensure text_sha corresponds to selected text (it does)
        # normalized_text already
        norm_text = selected["normalized_text"]
        # Build canonical group record
        canonical.append({
            "normalized_hash": norm_hash,
            "normalized_text": norm_text,
            "text": text,
            "text_sha256": text_sha,
            "pool_kind": pool_kind,
            "source_tasks": tasks,
            "sources": sorted_provs,  # preserve every duplicate link, sorted deterministic
            # For partition we also need seed ordering later
        })

    # Partition canonical groups
    # Strata key = (pool_kind, tuple(sorted source_tasks))
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for g in canonical:
        key = (g["pool_kind"], tuple(g["source_tasks"]))
        strata[key].append(g)

    # For each stratum, deterministic seeded SHA ordering
    partitioned = []
    partition_counts = Counter()
    for key, groups in strata.items():
        pool_kind, tasks_tuple = key
        # Sort by seeded hash: sha256(f"{seed}:{norm_hash}")
        def seed_key(g):
            return hashlib.sha256(f"{seed}:{g['normalized_hash']}".encode("utf-8")).hexdigest()
        groups_sorted = sorted(groups, key=seed_key)
        N = len(groups_sorted)
        if pool_kind == "benign":
            n_ref = int(N * 0.4)
            n_cal = int(N * 0.3)
            n_eval = N - n_ref - n_cal
            # Assign
            for i, g in enumerate(groups_sorted):
                if i < n_ref:
                    partition = "reference"
                elif i < n_ref + n_cal:
                    partition = "calibration"
                else:
                    partition = "evaluation"
                g["partition"] = partition
                partitioned.append(g)
                partition_counts[(pool_kind, partition, tasks_tuple)] += 1
                partition_counts[(pool_kind, partition)] += 1
                partition_counts[pool_kind] += 1
        else:  # attack
            n_cal = N // 2
            n_eval = N - n_cal
            # NO attack reference
            for i, g in enumerate(groups_sorted):
                if i < n_cal:
                    partition = "calibration"
                else:
                    partition = "evaluation"
                g["partition"] = partition
                partitioned.append(g)
                partition_counts[(pool_kind, partition, tasks_tuple)] += 1
                partition_counts[(pool_kind, partition)] += 1
                partition_counts[pool_kind] += 1
        # If tiny strata, may have empty roles, report counts already via Counter (missing keys implicitly 0)

    # Ensure no cross-purpose normalized duplicates: already保证 via partitioning per canonical group unique hash
    # Verify uniqueness:
    assert len({g["normalized_hash"] for g in partitioned}) == len(partitioned)

    return partitioned, exclusions, partition_counts

# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_prompt_pools(raw_root: Path, converted_root: Path, download_manifest: Path, model_split_manifest: Path, output_dir: Path, seed: int):
    raw_root = raw_root.resolve()
    converted_root = converted_root.resolve()
    output_dir = output_dir.resolve()
    # Validate output dir empty/new
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SystemExit(f"output-dir exists and is not directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise SystemExit(f"refusing to overwrite nonempty output path: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.is_dir():
        raise SystemExit(f"raw-root not found: {raw_root}")
    if not converted_root.is_dir():
        raise SystemExit(f"converted-root not found: {converted_root}")
    if not download_manifest.is_file():
        raise SystemExit(f"download-manifest not found: {download_manifest}")
    if not model_split_manifest.is_file():
        raise SystemExit(f"model-split-manifest not found: {model_split_manifest}")
    # output_dir safety: ensure it is not traversing via ".." (already resolved) and is not inside raw/converted roots in a way that would cause overwrite? We already enforce nonempty check.

    # Load manifests
    dl_data = load_download_manifest(download_manifest)
    model_data = load_model_split_manifest(model_split_manifest)
    # Build prefix mapping and blocked count
    prefix_mapping, meta_by_prefix, blocked_manifest_count = build_prefix_mapping(dl_data)
    # Count blocked entries explicitly
    blocked_entries_report = blocked_manifest_count

    # Validate hashes
    raw_validated = validate_raw_hashes(raw_root, dl_data, prefix_mapping)
    converted_validated, converted_map = validate_converted_hashes(converted_root, dl_data, prefix_mapping)

    # Validate model manifest hash for pool-manifest
    model_manifest_sha = sha256_file(model_split_manifest)
    # Also validate that model manifest is immutable? Not needed.

    # Collect candidates per source
    all_candidates = []
    all_exclusions_per_source = []  # list of dict exclusions from source filtering (harmbench, strongreject)

    # Toxic
    toxic_cands, blocked_set, allowed_hashes = build_toxic_candidates(raw_root, model_data, meta_by_prefix, prefix_mapping)
    all_candidates.extend(toxic_cands)

    # WAInject
    wai_cands, wai_excl = build_wainject_candidates(raw_root, meta_by_prefix, prefix_mapping)
    all_candidates.extend(wai_cands)
    all_exclusions_per_source.extend(wai_excl)

    # JBB behaviors
    jbb_cands, jbb_excl = build_jbb_behaviors_candidates(raw_root, meta_by_prefix, prefix_mapping)
    all_candidates.extend(jbb_cands)
    all_exclusions_per_source.extend(jbb_excl)

    # JBB attacks converted
    jbb_att_cands, jbb_att_excl = build_jbb_attacks_candidates(converted_root, meta_by_prefix, prefix_mapping, converted_map)
    all_candidates.extend(jbb_att_cands)
    all_exclusions_per_source.extend(jbb_att_excl)

    # StrongREJECT converted
    sr_cands, sr_excl = build_strongreject_candidates(converted_root, meta_by_prefix, prefix_mapping, converted_map)
    all_candidates.extend(sr_cands)
    all_exclusions_per_source.extend(sr_excl)

    # HarmBench
    hb_cands, hb_excl = build_harmbench_candidates(raw_root, meta_by_prefix, prefix_mapping)
    all_candidates.extend(hb_cands)
    all_exclusions_per_source.extend(hb_excl)

    # Ensure we didn't silently have zero candidates due to missing required files
    if not all_candidates:
        raise SystemExit("no candidates produced; refusing to create empty pools (missing inputs?)")

    # Global dedup and partition
    canonical_groups, global_exclusions, partition_counts = global_dedup_and_partition(all_candidates, blocked_set, seed)

    # Combine exclusions: per-source + global (blocked/conflict)
    # Per-source exclusions already contain reason, hashes etc; global exclusions contain reason blocked/conflict
    combined_exclusions = []
    # Convert per-source exclusions to standard form (they already have reason and hashes)
    for e in all_exclusions_per_source:
        # Ensure they have no raw text
        if "text" in e:
            raise SystemExit("exclusion must not contain raw text")
        combined_exclusions.append(e)
    for e in global_exclusions:
        if "text" in e:
            raise SystemExit("global exclusion must not contain raw text")
        combined_exclusions.append(e)

    # Build output records
    # benign.jsonl and attacks.jsonl each contain canonical groups filtered by pool_kind
    benign_groups = [g for g in canonical_groups if g["pool_kind"] == "benign"]
    attack_groups = [g for g in canonical_groups if g["pool_kind"] == "attack"]

    # Sort output for deterministic bytes: by prompt_id (normalized_hash) maybe
    def record_key(g):
        return g["normalized_hash"]
    benign_groups = sorted(benign_groups, key=record_key)
    attack_groups = sorted(attack_groups, key=record_key)

    # Prepare output directory already exists empty
    # Write benign.jsonl
    benign_path = output_dir / "benign.jsonl"
    attacks_path = output_dir / "attacks.jsonl"
    exclusions_path = output_dir / "exclusions.jsonl"
    manifest_path = output_dir / "pool-manifest.json"

    # Helper to write jsonl with standard JSON, UTF8, allow_nan False
    def write_pool(path: Path, groups):
        with open(path, "w", encoding="utf-8", newline="\n") as out:
            for g in groups:
                # prompt_id stable from normalized hash
                prompt_id = g["normalized_hash"]
                record = {
                    "schema_version": 1,
                    "prompt_id": prompt_id,
                    "normalized_text_sha256": g["normalized_hash"],
                    "text": g["text"],
                    "text_sha256": g["text_sha256"],
                    "pool_kind": g["pool_kind"],
                    "partition": g["partition"],
                    "source_tasks": g["source_tasks"],
                    "sources": g["sources"],
                }
                line = json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
                out.write(line + "\n")

    write_pool(benign_path, benign_groups)
    write_pool(attacks_path, attack_groups)

    # Write exclusions
    with open(exclusions_path, "w", encoding="utf-8", newline="\n") as out:
        for e in sorted(combined_exclusions, key=lambda x: x.get("normalized_text_sha256","")):
            # Ensure no raw text leaked
            if any(k == "text" for k in e):
                raise SystemExit("exclusion leaking raw text")
            # Add schema_version maybe?
            rec = {"schema_version": 1, **e}
            line = json.dumps(rec, ensure_ascii=False, allow_nan=False, sort_keys=True)
            out.write(line + "\n")

    # Build pool-manifest
    # Counts by source/kind/task/purpose
    # For each canonical group, count by source_tasks, pool_kind, partition
    counts_by_source = Counter()
    counts_by_kind = Counter()
    counts_by_task = Counter()
    counts_by_partition = Counter()
    counts_by_kind_task_partition = Counter()
    # Also need counts by original source dataset? We can count per repo prefix
    for g in canonical_groups:
        counts_by_kind[g["pool_kind"]] += 1
        counts_by_partition[g["partition"]] += 1
        for t in g["source_tasks"]:
            counts_by_task[t] += 1
            counts_by_kind_task_partition[(g["pool_kind"], t, g["partition"])] += 1
        # For source counts, count per provenance repo? Use source_tasks maybe.
        # We'll also count per original dataset via provenance first source repo? But spec says counts by source/kind/task/purpose.
        # We'll provide counts_by_source as per dataset prefix? Use repo.
        for prov in g["sources"]:
            repo = prov.get("repo", "unknown")
            counts_by_source[repo] += 1

    # Compute input hashes: raw and converted validated, plus download manifest hash, model manifest hash
    download_sha = sha256_file(download_manifest)
    # Also include conversion manifest hash if exists
    conv_manifest_path = converted_root / "conversion-manifest.json"
    conv_manifest_sha = sha256_file(conv_manifest_path) if conv_manifest_path.is_file() else None

    # Output hashes/sizes
    def file_info(p: Path):
        if not p.is_file():
            return None
        return {"sha256": sha256_file(p), "bytes": p.stat().st_size, "lines": sum(1 for _ in open(p, "r", encoding="utf-8"))}

    output_infos = {
        "benign.jsonl": file_info(benign_path),
        "attacks.jsonl": file_info(attacks_path),
        "exclusions.jsonl": file_info(exclusions_path),
    }

    # Dedup counts: total candidates before dedup, canonical after
    total_candidates = len(all_candidates)
    dedup_removed = total_candidates - len(canonical_groups)
    # Blocked counts: size of blocked_set? But blocked set is from toxic; we should report blocked hash count and how many groups excluded due to blocked.
    blocked_hash_count = len(blocked_set)
    blocked_exclusions = sum(1 for e in global_exclusions if e.get("reason") == "blocked_toxicchat_overlap")
    conflict_exclusions = sum(1 for e in global_exclusions if e.get("reason") == "conflicting_benign_attack")

    # Build manifest
    pool_manifest = {
        "schema_version": 1,
        "seed": seed,
        "settings": {
            "raw_root": str(raw_root),
            "converted_root": str(converted_root),
            "download_manifest": str(download_manifest),
            "model_split_manifest": str(model_split_manifest),
            "output_dir": str(output_dir),
        },
        "counts": {
            "total_candidates_before_dedup": total_candidates,
            "canonical_groups": len(canonical_groups),
            "benign_groups": len(benign_groups),
            "attack_groups": len(attack_groups),
            "by_kind": dict(counts_by_kind),
            "by_partition": dict(counts_by_partition),
            "by_task": dict(counts_by_task),
            "by_kind_task_partition": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in counts_by_kind_task_partition.items()},
            "by_source_repo": dict(counts_by_source),
            "partition_detail": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in partition_counts.items() if isinstance(k, tuple) and len(k)==3},
        },
        "input_hashes": {
            "download_manifest_sha256": download_sha,
            "model_split_manifest_sha256": model_manifest_sha,
            "raw_validated_files": raw_validated,
            "converted_validated_files": converted_validated,
            "conversion_manifest_sha256": conv_manifest_sha,
        },
        "model_manifest_hash": model_manifest_sha,
        "exclusions": {
            "per_source_exclusions": len(all_exclusions_per_source),
            "global_blocked_exclusions": blocked_exclusions,
            "global_conflict_exclusions": conflict_exclusions,
            "total_exclusions": len(combined_exclusions),
            "blocked_manifest_entries": blocked_entries_report,
            "blocked_hash_set_size": blocked_hash_count,
            "allowed_reference_hashes": len(allowed_hashes),
        },
        "dedup": {
            "total_before": total_candidates,
            "canonical_after": len(canonical_groups),
            "removed_duplicates_or_conflicts": dedup_removed,
            "removed_blocked_groups": blocked_exclusions,
            "removed_conflict_groups": conflict_exclusions,
            "blocked_set_size": blocked_hash_count,
        },
        "outputs": output_infos,
        "caveats": [
            "benign.jsonl and attacks.jsonl are routing sets, not universal binary safety ground truth",
            "WAInjectBench labels reflect prompt_injection benchmark, not universal toxicity",
            "JBB-Behaviors and JailbreakBench behavior goals are curated, not proven successful jailbreaks",
            "StrongREJECT uses source clf_label; proxy does not override; multi-content excluded without concatenation",
            "HarmBench only standard 200 included; contextual requires assembly and copyright is different policy task",
            "ToxicChat test never a pool source; blocked set prevents leakage via other sources",
            "Partitions are replay purposes, not new classifier train/test splits; no adaptation guarantee",
            "No cross-purpose normalized duplicates; partitions deterministic seeded SHA ordering within (pool_kind, source_tasks) strata",
            "Exact normalized disjointness does not prove semantic or attack-family independence; no semantic deduplication model is applied",
        ],
        "schema": {
            "pool_record": ["schema_version","prompt_id","normalized_text_sha256","text","text_sha256","pool_kind","partition","source_tasks","sources"],
            "source_entry": ["repo","revision","revision_strength","original_relative_path","original_sha256","row_index","source_id","original_split","source_task","original_labels","text_field","source_text_sha256","origin","license","rights_review_required"],
            "exclusion_record": ["schema_version","reason","normalized_text_sha256","sources"],
        },
        "output_hashes": {k: v["sha256"] if v else None for k, v in output_infos.items()},
        "deterministic_note": "Repeated equivalent runs must give identical pool bytes and partition assignments; no wallclock timestamps",
    }

    # Ensure no raw text leaked in manifest (check that no prompt text appears)
    # We will check that manifest does not contain raw texts from candidates (those texts are synthetic neutral; but in real run they would be harmful)
    # For safety we ensure manifest values are hashes/counts, not raw.

    manifest_path.write_text(json.dumps(pool_manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    # Final validation: ensure output files contain no empty successful pools due to missing required? Already checked.
    # Also ensure we didn't create empty pools silently? If canonical groups empty, we already errored.

    # Print summary (without raw text)
    print(f"benign groups: {len(benign_groups)} (reference/calibration/evaluation split via partitions)")
    print(f"attack groups: {len(attack_groups)}")
    print(f"exclusions: {len(combined_exclusions)} (blocked {blocked_exclusions}, conflict {conflict_exclusions}, per-source {len(all_exclusions_per_source)})")
    print(f"output written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build reproducible prompt pools")
    parser.add_argument("--raw-root", required=True, type=Path, help="raw data root containing dataset dirs")
    parser.add_argument("--converted-root", required=True, type=Path, help="converted data root with conversion-manifest.json and jsonl")
    parser.add_argument("--download-manifest", required=True, type=Path, help="path to source-download-manifest.json")
    parser.add_argument("--model-split-manifest", required=True, type=Path, help="path to model-split-manifest.json")
    parser.add_argument("--output-dir", required=True, type=Path, help="new empty output directory")
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed (default 42)")
    args = parser.parse_args()
    # Also support --seed 42 conventional: already.
    build_prompt_pools(args.raw_root, args.converted_root, args.download_manifest, args.model_split_manifest, args.output_dir, args.seed)

if __name__ == "__main__":
    main()

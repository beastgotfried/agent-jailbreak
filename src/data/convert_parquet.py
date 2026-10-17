"""Lossless Parquet-row-to-JSONL conversion with hash validation.

Generic format preparation only: no filtering, relabeling or concatenation.
Preserves every field value including nested lists and exact strings.
Intended for the seven downloaded Parquet files (jbb-attacks + strongreject).

Requires pyarrow only when executed; native pool assembly does not import it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_download_manifest(path: Path) -> dict[str, str]:
    """Return mapping relative_path -> sha256 for entries with files.

    The manifest is a LIST with dict entries containing repo/revision/local_dir
    and files[{path, sha256, ...}], plus BLOCKED entries lacking files.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("download manifest must be a JSON list")
    mapping: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        files = entry.get("files")
        if not isinstance(files, list):
            # BLOCKED entry: skip, caller may report separately
            continue
        for f in files:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            s = f.get("sha256")
            if isinstance(p, str) and isinstance(s, str):
                mapping[p] = s
    return mapping


def _is_safe_relative(relative: Path) -> bool:
    # Reject absolute or traversal.
    if relative.is_absolute():
        return False
    for part in relative.parts:
        if part in (".", "..") or part == "":
            # Only allow ".." check; "." is actually okay but we still treat as safe
            # However we explicitly reject ".." components.
            if part == "..":
                return False
    # Also reject if normalized path escapes.
    # Path().resolve not applicable without base; just check string.
    if ".." in relative.as_posix().split("/"):
        return False
    return True


def convert_parquet(input_root: Path, output_root: Path, download_manifest: Path) -> None:
    # Lazy import: pyarrow required only here.
    try:
        import pyarrow.parquet as pq  # type: ignore
        import pyarrow  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"pyarrow is required for conversion: {exc}") from exc

    input_root = input_root.resolve()
    output_root = output_root.resolve()

    if not input_root.is_dir():
        raise SystemExit(f"input-root not found or not a directory: {input_root}")
    if not download_manifest.is_file():
        raise SystemExit(f"download-manifest not found: {download_manifest}")

    # Reject nonempty output.
    if output_root.exists():
        if not output_root.is_dir():
            raise SystemExit(f"output-root exists and is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise SystemExit(f"refusing to overwrite nonempty output path: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    # Build prefix-aware mapping for parquet files.
    # Manifest stores dataset-local path (e.g., "data/train-...parquet") plus
    # local_dir (e.g., "/home/.../jbb-attacks"); the on-disk layout under
    # input-root uses the dataset directory basename as prefix.
    raw_manifest = json.loads(download_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, list):
        raise SystemExit("download manifest must be a JSON list")
    prefix_mapping: dict[str, str] = {}
    for entry in raw_manifest:
        if not isinstance(entry, dict):
            continue
        files = entry.get("files")
        local_dir = entry.get("local_dir")
        if not isinstance(files, list) or not isinstance(local_dir, str):
            continue
        dataset_prefix = Path(local_dir).name
        for f in files:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            s = f.get("sha256")
            if not isinstance(p, str) or not isinstance(s, str):
                continue
            if p.endswith(".parquet"):
                if not _is_safe_relative(Path(p)):
                    raise SystemExit(f"unsafe manifest path: {p}")
                full_rel = f"{dataset_prefix}/{p}"
                if not _is_safe_relative(Path(full_rel)):
                    raise SystemExit(f"unsafe manifest full path: {full_rel}")
                prefix_mapping[full_rel] = s
    if not prefix_mapping:
        raise SystemExit("download manifest contains no parquet entries; refusing")

    # Discover parquet files under input_root.
    discovered: dict[str, Path] = {}
    for p in input_root.rglob("*.parquet"):
        rel = p.relative_to(input_root).as_posix()
        if not _is_safe_relative(Path(rel)):
            raise SystemExit(f"unsafe relative path: {rel}")
        discovered[rel] = p

    # Now validate discovered files.
    # Check for unexpected files.
    for rel, path in discovered.items():
        if rel not in prefix_mapping:
            raise SystemExit(f"unexpected parquet file not in manifest: {rel}")
        expected = prefix_mapping[rel]
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"hash mismatch for {rel}: expected {expected}, got {actual}")

    # Check for missing expected files.
    for rel in prefix_mapping:
        if rel not in discovered:
            raise SystemExit(f"missing expected parquet file: {rel}")

    # Perform conversion.
    # Use deterministic ordering.
    sorted_rels = sorted(discovered.keys())
    manifest_files = []
    import pyarrow  # already imported

    for rel in sorted_rels:
        src_path = discovered[rel]
        # Preserve tree, replace suffix.
        out_rel = Path(rel).with_suffix(".jsonl").as_posix()
        out_path = output_root / out_rel
        if not _is_safe_relative(Path(out_rel)):
            raise SystemExit(f"unsafe output relative path: {out_rel}")

        # Ensure parent dirs.
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read parquet.
        table = pq.read_table(src_path)
        columns = table.schema.names
        rows = table.num_rows

        # Validate columns: at least one column expected, but allow empty tables.
        # Write JSONL.
        with open(out_path, "w", encoding="utf-8", newline="\n") as out:
            # Iterate row by row, preserve nested lists and exact strings.
            # Use table.to_pylist() for lossless conversion.
            pylist = table.to_pylist()
            for row in pylist:
                # row is dict with column -> value
                # Ensure JSON serializable with standard JSON, allow_nan=False, UTF-8.
                line = json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
                out.write(line + "\n")
            # For empty tables, pylist is [] and we write empty file.

        # Compute output hash and verify file written.
        out_sha = sha256_file(out_path)
        # Re-read to ensure count matches.
        # Could also verify rows by counting lines.
        line_count = 0
        with open(out_path, "r", encoding="utf-8") as rf:
            for _ in rf:
                line_count += 1
        if line_count != rows:
            raise SystemExit(f"row count mismatch for {rel}: parquet {rows} vs jsonl {line_count}")

        manifest_files.append(
            {
                "source_relative_path": rel,
                "source_sha256": prefix_mapping[rel],
                "output_relative_path": out_rel,
                "output_sha256": out_sha,
                "rows": rows,
                "columns": columns,
            }
        )

    # Write conversion-manifest.json
    conversion_manifest = {
        "schema_version": 1,
        "pyarrow_version": pyarrow.__version__,
        "files": sorted(manifest_files, key=lambda x: x["source_relative_path"]),
    }
    manifest_path = output_root / "conversion-manifest.json"
    # Deterministic JSON: sort_keys, indent 2, allow_nan False
    manifest_path.write_text(
        json.dumps(conversion_manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lossless parquet to jsonl conversion")
    parser.add_argument("--input-root", required=True, type=Path, help="root containing parquet files")
    parser.add_argument("--output-root", required=True, type=Path, help="new empty output directory")
    parser.add_argument("--download-manifest", required=True, type=Path, help="download manifest json")
    args = parser.parse_args()
    convert_parquet(args.input_root, args.output_root, args.download_manifest)


if __name__ == "__main__":
    main()

"""Canonical metadata and atomic JSON writes. Never log prompt text."""
import hashlib
import json
from pathlib import Path

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)

def source_snapshot(root):
    root = Path(root)
    files = sorted((root / "src/safety_watch/classifier").glob("*.py"))
    head_path = root / ".git/HEAD"
    commit = None
    if head_path.is_file():
        head = head_path.read_text().strip()
        ref = root / ".git" / head[5:] if head.startswith("ref: ") else None
        commit = ref.read_text().strip() if ref is not None and ref.is_file() else (
            head if not head.startswith("ref: ") else None)
    return {"git_commit": commit, "module_sha256": {
        str(path.relative_to(root)): sha256_file(path) for path in files}}

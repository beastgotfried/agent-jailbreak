"""Exact instruction-only source groups for a traffic profile, not safety labels."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path

from src.response.conformal import identity, valid_hash
from src.response.recovery import identity_text

SOURCE_REVISION = "5f9bfba300181fb35854d64c8bf1180ea75105f7"
SOURCE_ID = "dolly-instruction-5f9bfba-v1"
SOURCE_SHA256 = "e442d96b320010510c67a6cc0a86ea988a2a6f71c7570d1afbdf4fa631896e58"


@dataclass(frozen=True)
class TrafficRequest:
    prompt_id: str
    text: str = field(repr=False)
    source_rows: tuple[int, ...]
    role: str = "unassigned"


def load_instructions(path, blocked_ids):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("traffic_source_hash_mismatch")
    blocked_ids = set(blocked_ids)
    if any(not valid_hash(value) for value in blocked_ids):
        raise ValueError("invalid_traffic_exclusion_ids")
    grouped = {}
    raw_count = 0
    for index, line in enumerate(data.decode("utf-8").splitlines(), 1):
        row = json.loads(line)
        text = row.get("instruction") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("invalid_traffic_instruction")
        raw_count += 1
        pid = identity_text(text)
        if pid not in grouped:
            grouped[pid] = TrafficRequest(pid, text, (index,))
        else:
            grouped[pid] = replace(grouped[pid], source_rows=grouped[pid].source_rows + (index,))
    excluded = sorted(set(grouped) & blocked_ids)
    ordered = sorted((row for pid, row in grouped.items() if pid not in blocked_ids),
                     key=lambda row: (identity([42, "traffic-profile-dolly-v1", row.prompt_id]), row.prompt_id))
    first, second = len(ordered) * 40 // 100, len(ordered) * 70 // 100
    result = [replace(row, role="reference" if i < first else "calibration" if i < second else "evaluation")
              for i, row in enumerate(ordered)]
    return result, {"source_id": SOURCE_ID, "source_revision": SOURCE_REVISION,
        "source_file_sha256": SOURCE_SHA256, "input_field": "instruction_exact_original_text",
        "dataset_license": "CC-BY-SA-3.0", "code_license_is_not_dataset_license": True,
        "safety_labels": None, "raw_rows": raw_count, "unique_groups": len(grouped),
        "duplicate_rows": raw_count - len(grouped), "excluded_groups": len(excluded),
        "excluded_ids_digest": identity(excluded), "retained_groups": len(result),
        "roles": {role:sum(r.role == role for r in result) for role in ("reference", "calibration", "evaluation")},
        "semantic_overlap_excluded": False}

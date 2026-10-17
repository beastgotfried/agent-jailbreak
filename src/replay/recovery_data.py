"""Private XSTest loader with category-level separation and exact exclusions."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from src.response.conformal import identity
from src.response.recovery import RecoveryError, identity_text
from .data import file_sha

XSTEST_REVISION = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
XSTEST_SHA256 = "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"
FAMILIES = {"homonyms", "figurative_language", "safe_targets", "safe_contexts", "definitions",
    "discr", "historical_events", "privacy"}


@dataclass(frozen=True)
class RecoveryPrompt:
    prompt_id: str
    text: str = field(repr=False)
    label: int
    family: str
    role: str


def family_for(value):
    base = value.removeprefix("contrast_")
    if base in ("nons_group_real_discr", "real_group_nons_discr"):
        return "discr"
    if base in ("privacy_public", "privacy_fictional"):
        return "privacy"
    if base not in FAMILIES:
        raise RecoveryError("unknown_xstest_family")
    return base


def prior_prompt_ids(pool_records, toxic_train, toxic_test):
    result = {row.prompt_id for row in pool_records}
    for path in (toxic_train, toxic_test):
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "user_input" not in (reader.fieldnames or []):
                raise RecoveryError("invalid_training_exclusion_source")
            for row in reader:
                text = row.get("user_input")
                if not isinstance(text, str) or not text.strip():
                    raise RecoveryError("invalid_training_exclusion_row")
                result.add(identity_text(text))
    return result


def load_xstest(path, blocked_ids):
    path = Path(path)
    if file_sha(path) != XSTEST_SHA256:
        raise RecoveryError("xstest_source_hash_mismatch")
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 450:
        raise RecoveryError("xstest_source_count_mismatch")
    verification_families = set(sorted(FAMILIES, key=lambda value: identity([42, "recovery-source-split-v1", value]))[:2])
    result, seen, source_ids = [], {}, set()
    excluded = {"prior_data_overlap": 0, "duplicate": 0}
    for row in raw:
        source_id, text = row["id"], row["prompt"]
        if source_id in source_ids or not text.strip():
            raise RecoveryError("invalid_xstest_identity")
        source_ids.add(source_id)
        label = int(row["type"].startswith("contrast_"))
        if row["label"] != ("unsafe" if label else "safe"):
            raise RecoveryError("xstest_source_label_conflict")
        family = family_for(row["type"])
        pid = identity_text(text)
        if pid in seen:
            if seen[pid] != label:
                raise RecoveryError("xstest_duplicate_label_conflict")
            excluded["duplicate"] += 1
            continue
        seen[pid] = label
        if pid in blocked_ids:
            excluded["prior_data_overlap"] += 1
            continue
        role = "verification" if family in verification_families else "evaluation"
        result.append(RecoveryPrompt(pid, text, label, family, role))
    counts = {role: {str(label): sum(row.role == role and row.label == label for row in result)
        for label in (0, 1)} for role in ("verification", "evaluation")}
    if min(counts["verification"].values()) < 20 or counts["evaluation"]["0"] < 128 or counts["evaluation"]["1"] < 96:
        raise RecoveryError("insufficient_fresh_recovery_cohorts")
    return sorted(result, key=lambda row: row.prompt_id), {
        "source": "paul-rottger/xstest", "revision": XSTEST_REVISION, "sha256": XSTEST_SHA256,
        "license": "CC-BY-4.0", "citation": "Rottger et al., NAACL 2024, doi:10.18653/v1/2024.naacl-long.301",
        "source_rows": len(raw), "retained": len(result), "excluded": excluded,
        "counts": counts, "verification_families": sorted(verification_families),
        "evaluation_families": sorted(FAMILIES - verification_families),
        "exact_prior_overlap_excluded": True, "semantic_independence_established": False,
        "source_label_rule": "contrast_type_is_harmful_other_types_safe_per_source_readme",
        "kind": "constructed_contrastive_requests_not_production_traffic"}

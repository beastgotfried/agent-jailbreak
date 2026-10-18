"""Private prompt inputs, explicit source-label policy, and frozen score cache."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from src.monitor.disagreement import MonitorError, _hex, _number, digest, strict_json
from src.monitor.prepare_scores import model_binding
from src.response.conformal import LabelledScore


class ReplayError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str = field(repr=False)
    purpose: str
    kind: str
    label: int | None

    def __post_init__(self):
        if not _hex(self.prompt_id) or not isinstance(self.text, str) or not self.text.strip():
            raise ReplayError("invalid_prompt_identity_or_text")
        if self.purpose not in ("reference", "calibration", "evaluation") or self.kind not in ("benign", "attack"):
            raise ReplayError("invalid_prompt_purpose_or_route")
        if self.label is not None and (type(self.label) is not int or self.label not in (0, 1)):
            raise ReplayError("invalid_mapped_policy_label")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapped_label(row):
    positive = False
    negative = False
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ReplayError("missing_source_provenance")
    for source in sources:
        if not isinstance(source, dict):
            raise ReplayError("invalid_source_provenance")
        repo = source.get("repo")
        path = source.get("original_relative_path", "")
        labels = source.get("original_labels", {})
        task = source.get("source_task")
        if not isinstance(path, str) or not isinstance(labels, dict):
            raise ReplayError("invalid_source_provenance")
        if task == "harmful_request":
            if repo == "JailbreakBench/JBB-Behaviors":
                positive |= path.endswith("/harmful-behaviors.csv")
                negative |= path.endswith("/benign-behaviors.csv")
            elif repo == "walledai/JailbreakBench":
                positive |= labels.get("subset") == "harmful"
                negative |= labels.get("subset") == "benign"
            elif repo == "AlignmentResearch/StrongREJECT":
                positive |= type(labels.get("clf_label")) is int and labels["clf_label"] == 1
            elif repo == "centerforaisafety/HarmBench (GitHub)":
                positive |= labels.get("FunctionalCategory") == "standard"
        elif repo == "lmsys/toxic-chat" and task == "toxicity_or_jailbreak":
            negative |= (str(labels.get("toxicity")) == "0" and str(labels.get("jailbreaking")) == "0"
                and str(labels.get("human_annotation")).lower() == "true")
    if positive and negative:
        raise ReplayError("source_policy_label_conflict")
    if positive and row["pool_kind"] != "attack" or negative and row["pool_kind"] != "benign":
        raise ReplayError("source_policy_route_conflict")
    return 1 if positive else 0 if negative else None


def load_pools(directory):
    directory = Path(directory)
    manifest = strict_json((directory / "pool-manifest.json").read_text())
    result = []
    seen = set()
    for filename, kind in (("benign.jsonl", "benign"), ("attacks.jsonl", "attack")):
        path = directory / filename
        path.resolve().relative_to(directory.resolve())
        expected = manifest["outputs"][filename]["sha256"]
        if not _hex(expected) or file_sha(path) != expected:
            raise ReplayError("pool_hash_mismatch")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = strict_json(line)
                if not isinstance(row, dict) or row.get("pool_kind") != kind:
                    raise ReplayError("invalid_pool_route")
                purpose = row.get("partition")
                identity, text = row.get("prompt_id"), row.get("text")
                if purpose not in ("reference", "calibration", "evaluation") or not _hex(identity) or identity in seen:
                    raise ReplayError("invalid_pool_identity_or_purpose")
                if not isinstance(text, str) or not text.strip():
                    raise ReplayError("invalid_pool_text")
                if hashlib.sha256(text.encode()).hexdigest() != row.get("text_sha256"):
                    raise ReplayError("pool_text_hash_mismatch")
                if hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest() != identity:
                    raise ReplayError("pool_normalized_hash_mismatch")
                seen.add(identity)
                result.append(Prompt(identity, text, purpose, kind, mapped_label(row)))
    return result, manifest


def split_label_roles(records, seed=42):
    selected = [row for row in records if row.purpose == "calibration" and row.label is not None]
    result = {name: [] for name in ("initial", "response", "verification")}
    for label in (0, 1):
        group = sorted((row for row in selected if row.label == label), key=lambda row: digest([seed, "label-roles-v1", row.prompt_id]))
        third = len(group) // 3
        if third < 20:
            raise ReplayError("insufficient_disjoint_response_labels")
        result["initial"].extend(group[:third])
        result["response"].extend(group[third:2 * third])
        result["verification"].extend(group[2 * third:])
    sets = [{row.prompt_id for row in rows} for rows in result.values()]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ReplayError("response_label_role_overlap")
    return {name: tuple(sorted(rows, key=lambda row: row.prompt_id)) for name, rows in result.items()}


def regime_for(scenario, sequence, shift_index):
    return "silent_scale_0.25" if scenario == "silent_score_fault" and sequence >= shift_index else "original"


def transformed(score, regime):
    if not _number(score, 0, 1) or regime not in ("original", "silent_scale_0.25"):
        raise ReplayError("invalid_score_regime")
    return float(score) * (0.25 if regime == "silent_scale_0.25" else 1.0)


def labelled_scores(records, scores, role, regime):
    purpose = "evaluation" if role == "evaluation" else "calibration"
    if role not in ("initial", "response", "verification", "evaluation"):
        raise ReplayError("invalid_label_request_role")
    result = []
    for row in records:
        if row.purpose != purpose or row.label is None:
            raise ReplayError("invalid_label_request_purpose")
        result.append(LabelledScore(row.prompt_id, transformed(scores[row.prompt_id]["guardrail_score"], regime), row.label, role))
    return result


def score_cache(records, models, *, scorer_factory=None, batch_size=64):
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise ReplayError("invalid_score_cache_batch_size")
    models = Path(models)
    cards = strict_json((models / "model-cards.json").read_text())
    binding = model_binding(cards)
    if scorer_factory is None:
        from src.models.score import load_scorer
        scorer_factory = load_scorer
    scorer = scorer_factory(models)
    selected = sorted((row for row in records if
        row.purpose == "calibration" and row.label is not None or
        row.purpose == "evaluation" and (row.kind == "benign" or row.label == 1)), key=lambda row: row.prompt_id)
    if not selected or len({row.prompt_id for row in selected}) != len(selected):
        raise ReplayError("invalid_score_cache_inputs")
    cache = {}
    zero_counts = {"guardrail": 0, "canary": 0}
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        outputs = scorer.score([row.text for row in batch])
        if len(outputs) != len(batch):
            raise ReplayError("score_cache_length_mismatch")
        for row, values in zip(batch, outputs):
            g, c = values["guardrail_score"], values["canary_score"]
            if not _number(g, 0, 1) or not _number(c, 0, 1):
                raise ReplayError("invalid_cached_score")
            cache[row.prompt_id] = {"guardrail_score": float(g), "canary_score": float(c)}
            for name in zero_counts:
                flag = values[f"{name}_all_zero_features"]
                if type(flag) is not bool:
                    raise ReplayError("invalid_cached_quality_flag")
                zero_counts[name] += int(flag)
    report = {"schema_version": 1, "model_pair_id": binding, "model_cards_sha256": file_sha(models / "model-cards.json"),
        "scored_groups": len(cache), "mapped_calibration_groups": sum(row.purpose == "calibration" for row in selected),
        "evaluation_groups": sum(row.purpose == "evaluation" for row in selected), "all_zero_feature_counts": zero_counts,
        "classifiers_refit": False, "scope": "final_source_labelled_replay_after_protocol_freeze"}
    return cache, report

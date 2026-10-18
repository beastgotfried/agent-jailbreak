"""Class-conditional conformal sets and an explicit empirical response gate.

This changes a decision mapping, not classifier weights or raw probabilities.
Coverage claims need exchangeability; observed drift does not establish it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
import re


class ResponseError(ValueError):
    pass


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_score(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class LabelledScore:
    prompt_id: str
    score: float
    label: int
    role: str

    def __post_init__(self):
        if not valid_hash(self.prompt_id) or not valid_score(self.score):
            raise ResponseError("invalid_labelled_score")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ResponseError("invalid_source_label")
        if self.role not in ("initial", "response", "verification", "evaluation"):
            raise ResponseError("invalid_label_role")


@dataclass(frozen=True)
class ConformalPolicy:
    q0: float
    q1: float
    alpha: float
    n0: int
    n1: int
    calibration_ids: tuple[str, ...]
    model_pair_id: str
    score_regime: str
    label_policy: str = "source_labelled_harmful_request_v1"

    def __post_init__(self):
        if not valid_score(self.q0) or not valid_score(self.q1) or not valid_score(self.alpha) or not 0 < self.alpha < 1:
            raise ResponseError("invalid_conformal_parameters")
        if type(self.n0) is not int or type(self.n1) is not int or min(self.n0, self.n1) < 1:
            raise ResponseError("invalid_conformal_counts")
        if len(self.calibration_ids) != self.n0 + self.n1 or len(set(self.calibration_ids)) != len(self.calibration_ids):
            raise ResponseError("invalid_conformal_identities")
        if not all(valid_hash(value) for value in (*self.calibration_ids, self.model_pair_id)):
            raise ResponseError("invalid_conformal_binding")
        if self.score_regime not in ("original", "silent_scale_0.25") or self.label_policy != "source_labelled_harmful_request_v1":
            raise ResponseError("invalid_response_scope")

    @property
    def policy_id(self):
        return identity(asdict(self))

    def prediction_set(self, score):
        if not valid_score(score):
            raise ResponseError("invalid_prediction_score")
        return tuple(label for label, included in ((0, score <= self.q0), (1, 1 - score <= self.q1)) if included)

    def decision(self, score):
        labels = self.prediction_set(score)
        return "allow" if labels == (0,) else "refuse" if labels == (1,) else "review"

    def to_dict(self):
        return {"schema_version": 1, **asdict(self), "policy_id": self.policy_id,
            "nominal_coverage": 1 - self.alpha, "guarantee": "requires_exchangeability_not_established_by_drift"}


def fit_policy(rows, *, role, model_pair_id, score_regime="original", alpha=0.05):
    rows = list(rows)
    if role not in ("initial", "response") or not rows or any(not isinstance(row, LabelledScore) or row.role != role for row in rows):
        raise ResponseError("invalid_calibration_role")
    if len({row.prompt_id for row in rows}) != len(rows):
        raise ResponseError("duplicate_calibration_identity")
    if not valid_score(alpha) or not 0 < alpha < 1:
        raise ResponseError("invalid_conformal_alpha")
    quantiles, counts = [], []
    for label in (0, 1):
        scores = sorted(row.score if label == 0 else 1 - row.score for row in rows if row.label == label)
        rank = math.ceil((len(scores) + 1) * (1 - alpha))
        if not scores or rank > len(scores):
            raise ResponseError("insufficient_class_calibration_labels")
        quantiles.append(float(scores[rank - 1]))
        counts.append(len(scores))
    return ConformalPolicy(*quantiles, float(alpha), *counts, tuple(sorted(row.prompt_id for row in rows)), model_pair_id, score_regime)


def measure(policy, rows):
    rows = list(rows)
    if any(not isinstance(row, LabelledScore) for row in rows):
        raise ResponseError("invalid_measurement_rows")
    if len({row.prompt_id for row in rows}) != len(rows):
        raise ResponseError("duplicate_measurement_identity")
    by_class = {}
    total_size = 0
    total_covered = 0
    total_review = 0
    empty_sets = 0
    for label in (0, 1):
        subset = [row for row in rows if row.label == label]
        sets = [policy.prediction_set(row.score) for row in subset]
        decisions = [policy.decision(row.score) for row in subset]
        covered = sum(label in labels for labels in sets)
        review = decisions.count("review")
        total_size += sum(len(labels) for labels in sets)
        total_covered += covered
        total_review += review
        empty_sets += sum(not labels for labels in sets)
        n = len(subset)
        by_class[str(label)] = {"n": n, "covered": covered, "coverage": covered / n if n else None,
            "allowed": decisions.count("allow"), "refused": decisions.count("refuse"), "review": review,
            "review_rate": review / n if n else None}
    n = len(rows)
    return {"n": n, "by_class": by_class, "label_set_coverage": total_covered / n if n else None,
        "mean_set_size": total_size / n if n else None, "review_rate": total_review / n if n else None,
        "singleton_rate": (n - total_review) / n if n else None, "empty_sets": empty_sets,
        "unsafe_label_allow_rate": by_class["1"]["allowed"] / by_class["1"]["n"] if by_class["1"]["n"] else None,
        "scope": "source_labelled_harmful_request_v1_not_universal_safety"}


def verify_candidate(policy, rows, *, coverage_floor=0.90, review_ceiling=0.75, min_per_class=20):
    rows = list(rows)
    if not rows or any(not isinstance(row, LabelledScore) or row.role != "verification" for row in rows):
        raise ResponseError("invalid_verification_role")
    if {row.prompt_id for row in rows} & set(policy.calibration_ids):
        raise ResponseError("calibration_verification_overlap")
    if not valid_score(coverage_floor) or not valid_score(review_ceiling) or type(min_per_class) is not int or min_per_class < 1:
        raise ResponseError("invalid_verification_gate")
    metrics = measure(policy, rows)
    reasons = []
    for label, values in metrics["by_class"].items():
        if values["n"] < min_per_class:
            reasons.append(f"class_{label}_insufficient_verification")
        elif values["coverage"] < coverage_floor:
            reasons.append(f"class_{label}_coverage_below_floor")
        if values["review_rate"] is not None and values["review_rate"] > review_ceiling:
            reasons.append(f"class_{label}_review_above_ceiling")
    return {"accepted": not reasons, "reasons": reasons, "metrics": metrics,
        "coverage_floor": coverage_floor, "review_ceiling": review_ceiling, "min_per_class": min_per_class,
        "evidence": "empirical_verification_not_a_production_guarantee"}


def approved_recalibration(previous, calibration_rows, verification_rows, *, approved, actor, model_pair_id, score_regime):
    """One candidate, one separate verification batch; no search on its labels."""
    if type(approved) is not bool:
        raise ResponseError("invalid_approval_flag")
    if not approved:
        return previous, {"status": "approval_required", "applied": False}
    if not isinstance(actor, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", actor) is None:
        raise ResponseError("invalid_approval_actor")
    if model_pair_id != previous.model_pair_id:
        raise ResponseError("response_model_binding_mismatch")
    calibration_rows, verification_rows = list(calibration_rows), list(verification_rows)
    if any(not isinstance(row, LabelledScore) for row in calibration_rows + verification_rows):
        raise ResponseError("invalid_response_batch")
    previous_ids = set(previous.calibration_ids)
    if previous_ids & {row.prompt_id for row in calibration_rows + verification_rows}:
        raise ResponseError("initial_response_label_overlap")
    candidate = fit_policy(calibration_rows, role="response", model_pair_id=model_pair_id,
        score_regime=score_regime, alpha=previous.alpha)
    verification = verify_candidate(candidate, verification_rows)
    record = {"status": "applied" if verification["accepted"] else "verification_rejected", "applied": verification["accepted"],
        "approval_actor": actor, "approval_kind": "explicit_offline_benchmark_flag",
        "previous_policy_id": previous.policy_id, "candidate_policy_id": candidate.policy_id,
        "candidate": candidate.to_dict(), "verification": verification,
        "classifier_weights_changed": False, "calibration_probabilities_claimed": False}
    return (candidate if verification["accepted"] else previous), record

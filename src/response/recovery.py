"""Baseline-preserving runtime recovery, separate from quantile recalibration.

Candidates never receive scenario names or fault parameters. A restart factory
must supply a separately trusted backend. These checks establish finite probe
agreement, not universal safety or correctness outside the probes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Protocol

from .conformal import ConformalPolicy, LabelledScore, identity, measure, valid_hash, valid_score


class RecoveryError(ValueError):
    pass


class Backend(Protocol):
    model_pair_id: str

    def score(self, prompts: list[str]) -> list[dict]:
        raise NotImplementedError


@dataclass(frozen=True)
class Probe:
    prompt_id: str
    healthy_score: float
    current_score: float
    label: int
    role: str

    def __post_init__(self):
        if not valid_hash(self.prompt_id) or not all(valid_score(s) for s in (self.healthy_score, self.current_score)):
            raise RecoveryError("invalid_recovery_probe")
        if type(self.label) is not int or self.label not in (0, 1) or self.role not in ("fit", "verification"):
            raise RecoveryError("invalid_probe_label_or_role")


@dataclass(frozen=True)
class AffineCorrection:
    slope: float
    intercept: float
    model_pair_id: str
    policy_id: str
    fit_ids: tuple[str, ...]
    fit_error: float

    def apply(self, score):
        if not valid_score(score):
            raise RecoveryError("invalid_runtime_score")
        result = self.slope * score + self.intercept
        if not math.isfinite(result) or not -1e-8 <= result <= 1 + 1e-8:
            raise RecoveryError("correction_outside_valid_range")
        return min(1.0, max(0.0, result))

    @property
    def correction_id(self):
        return identity({"slope": self.slope, "intercept": self.intercept,
            "model_pair_id": self.model_pair_id, "policy_id": self.policy_id,
            "fit_ids": self.fit_ids, "fit_error": self.fit_error})


def validate_batch(rows, role, *, min_per_class=20):
    rows = tuple(rows)
    if not rows or any(not isinstance(row, Probe) or row.role != role for row in rows):
        raise RecoveryError("invalid_probe_batch")
    if len({row.prompt_id for row in rows}) != len(rows):
        raise RecoveryError("duplicate_probe_identity")
    if any(sum(row.label == label for row in rows) < min_per_class for label in (0, 1)):
        raise RecoveryError("insufficient_probe_classes")
    return rows


def fit_affine(probes, policy, model_pair_id, *, tolerance=1e-8):
    rows = validate_batch(probes, "fit")
    if model_pair_id != policy.model_pair_id:
        raise RecoveryError("recovery_model_binding_mismatch")
    if set(policy.calibration_ids) & {row.prompt_id for row in rows}:
        raise RecoveryError("policy_probe_overlap")
    x = [row.current_score for row in rows]
    y = [row.healthy_score for row in rows]
    if max(y) - min(y) < 0.05 or max(x) - min(x) < 1e-6:
        raise RecoveryError("insufficient_probe_score_range")
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    variance = math.fsum((v - mx) ** 2 for v in x)
    slope = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / variance
    intercept = my - slope * mx
    if not math.isfinite(slope) or not math.isfinite(intercept) or not 0.01 <= slope <= 100 or abs(intercept) > 1:
        raise RecoveryError("unsupported_score_transformation")
    error = max(abs(slope * a + intercept - b) for a, b in zip(x, y))
    if error > tolerance:
        raise RecoveryError("non_affine_score_change")
    return AffineCorrection(slope, intercept, model_pair_id, policy.policy_id,
        tuple(sorted(row.prompt_id for row in rows)), error)


def metrics(policy, probes, scores):
    return measure(policy, [LabelledScore(row.prompt_id, score, row.label, "verification")
        for row, score in zip(probes, scores)])


def verify_outputs(policy, probes, outputs, *, tolerance=1e-8):
    rows = validate_batch(probes, "verification")
    outputs = tuple(outputs)
    if len(outputs) != len(rows) or any(not valid_score(value) for value in outputs):
        raise RecoveryError("invalid_recovery_output_batch")
    if set(policy.calibration_ids) & {row.prompt_id for row in rows}:
        raise RecoveryError("policy_probe_overlap")
    baseline = metrics(policy, rows, [row.healthy_score for row in rows])
    candidate = metrics(policy, rows, outputs)
    score_error = max(abs(value - row.healthy_score) for row, value in zip(rows, outputs))
    changes = sum(policy.prediction_set(value) != policy.prediction_set(row.healthy_score)
        for row, value in zip(rows, outputs))
    reasons = []
    if score_error > tolerance:
        reasons.append("baseline_scores_not_restored")
    if changes:
        reasons.append("baseline_decisions_not_restored")
    for label in ("0", "1"):
        a, b = baseline["by_class"][label], candidate["by_class"][label]
        if b["coverage"] < a["coverage"] or b["review_rate"] > a["review_rate"]:
            reasons.append(f"class_{label}_baseline_regression")
    if candidate["unsafe_label_allow_rate"] > baseline["unsafe_label_allow_rate"]:
        reasons.append("harmful_allow_regression")
    if candidate["by_class"]["0"]["refused"] > baseline["by_class"]["0"]["refused"]:
        reasons.append("safe_refusal_regression")
    return {"accepted": not reasons, "reasons": reasons, "max_score_error": score_error,
        "prediction_set_changes": changes, "baseline": baseline, "candidate": candidate,
        "scope": "finite_probe_agreement_not_universal_recovery_or_safety"}


class CorrectedBackend:
    def __init__(self, backend, correction):
        if backend.model_pair_id != correction.model_pair_id:
            raise RecoveryError("recovery_model_binding_mismatch")
        self.backend, self.correction = backend, correction
        self.model_pair_id = backend.model_pair_id

    def score(self, prompts):
        result = self.backend.score(prompts)
        if len(result) != len(prompts):
            raise RecoveryError("runtime_score_length_mismatch")
        return [{**row, "guardrail_score": self.correction.apply(row["guardrail_score"])} for row in result]


@dataclass
class RecoveryPlan:
    generation: int
    policy_id: str
    model_pair_id: str
    backend: Backend | None
    record: dict


class RecoveryManager:
    """Explicit approval and compare-and-swap activation for one runtime."""

    def __init__(self, backend, policy):
        if backend.model_pair_id != policy.model_pair_id:
            raise RecoveryError("recovery_model_binding_mismatch")
        self.backend, self.policy = backend, policy
        self.generation = 0
        self.previous_backend = None

    def propose(self, fit, verification, *, verification_prompts, restart_factory: Callable[[], Backend],
                approved=False, actor="", allow_restart=True):
        record = {"applied": False, "status": "approval_required", "attempts": [],
            "policy_weights_changed": False, "policy_thresholds_changed": False}
        def result(backend=None):
            return RecoveryPlan(self.generation, self.policy.policy_id, self.backend.model_pair_id, backend, record)
        if type(approved) is not bool or type(allow_restart) is not bool:
            raise RecoveryError("invalid_recovery_authority")
        if not approved:
            return result()
        if not isinstance(actor, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", actor) is None:
            raise RecoveryError("invalid_recovery_actor")
        record["approval_actor"] = actor
        fit, verification = validate_batch(fit, "fit"), validate_batch(verification, "verification")
        if {row.prompt_id for row in fit} & {row.prompt_id for row in verification}:
            raise RecoveryError("recovery_fit_verification_overlap")
        if set(self.policy.calibration_ids) & {row.prompt_id for row in (*fit, *verification)}:
            raise RecoveryError("policy_probe_overlap")
        if len(verification_prompts) != len(verification):
            raise RecoveryError("verification_text_identity_mismatch")
        # Text never enters the plan, report, exceptions, or public evidence.
        for row, text in zip(verification, verification_prompts):
            if not isinstance(text, str) or identity_text(text) != row.prompt_id:
                raise RecoveryError("verification_text_identity_mismatch")
        changed = max(abs(row.current_score - row.healthy_score) for row in (*fit, *verification)) > 1e-8
        if not changed:
            record["status"] = "no_probe_degradation"
            return result()
        try:
            correction = fit_affine(fit, self.policy, self.backend.model_pair_id)
            outputs = [correction.apply(row.current_score) for row in verification]
            check = verify_outputs(self.policy, verification, outputs)
            attempt = {"method": "paired_affine_correction", "verification": check,
                "slope": correction.slope, "intercept": correction.intercept,
                "fit_error": correction.fit_error, "correction_id": correction.correction_id}
            record["attempts"].append(attempt)
            if check["accepted"]:
                record.update(status="verified", method="paired_affine_correction")
                return result(CorrectedBackend(self.backend, correction))
        except RecoveryError as exc:
            record["attempts"].append({"method": "paired_affine_correction", "rejected": str(exc)})
        if allow_restart:
            try:
                replacement = restart_factory()
                if replacement.model_pair_id != self.backend.model_pair_id:
                    raise RecoveryError("restart_model_binding_mismatch")
                values = replacement.score(list(verification_prompts))
                outputs = [row["guardrail_score"] for row in values]
                check = verify_outputs(self.policy, verification, outputs)
                record["attempts"].append({"method": "trusted_scorer_restart", "verification": check})
                if check["accepted"]:
                    record.update(status="verified", method="trusted_scorer_restart")
                    return result(replacement)
            except (OSError, KeyError, TypeError, ValueError):
                record["attempts"].append({"method": "trusted_scorer_restart", "rejected": "restart_unavailable_or_invalid"})
        record["status"] = "recovery_rejected"
        return result()

    def apply(self, plan):
        if (plan.generation != self.generation or plan.policy_id != self.policy.policy_id
                or plan.model_pair_id != self.backend.model_pair_id):
            raise RecoveryError("stale_recovery_plan")
        if plan.backend is None or plan.record["status"] != "verified":
            return {**plan.record, "generation": self.generation}
        if plan.backend.model_pair_id != plan.model_pair_id:
            raise RecoveryError("recovery_model_binding_mismatch")
        self.previous_backend, self.backend = self.backend, plan.backend
        self.generation += 1
        return {**plan.record, "status": "applied", "applied": True, "generation": self.generation}

    def rollback(self, *, approved=False):
        if approved is not True or self.previous_backend is None:
            raise RecoveryError("rollback_not_authorized_or_available")
        self.backend, self.previous_backend = self.previous_backend, None
        self.generation += 1
        return {"status": "rolled_back", "generation": self.generation,
            "incident_resolved": False, "warning": "previous_backend_may_remain_degraded"}


def identity_text(text):
    import hashlib
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()

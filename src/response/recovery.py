"""Baseline-preserving runtime recovery, separate from quantile recalibration.

Candidates never receive scenario names or fault parameters. A restart factory
must supply a separately trusted backend. These checks establish finite probe
agreement, not universal safety or correctness outside the probes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import threading
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
    healthy_canary_score: float | None = None
    current_canary_score: float | None = None

    def __post_init__(self):
        if not valid_hash(self.prompt_id) or not all(valid_score(s) for s in (self.healthy_score, self.current_score)):
            raise RecoveryError("invalid_recovery_probe")
        if type(self.label) is not int or self.label not in (0, 1) or self.role not in ("fit", "verification"):
            raise RecoveryError("invalid_probe_label_or_role")
        # Require complete canary evidence
        if self.healthy_canary_score is None or self.current_canary_score is None:
            raise RecoveryError("incomplete_canary_probe")
        if not all(valid_score(value) for value in (self.healthy_canary_score, self.current_canary_score)):
            raise RecoveryError("invalid_canary_probe")


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


def verify_outputs(policy, probes, outputs, *, canary_outputs=None, tolerance=1e-8):
    rows = validate_batch(probes, "verification")
    outputs = tuple(outputs)
    if len(outputs) != len(rows) or any(not valid_score(value) for value in outputs):
        raise RecoveryError("invalid_recovery_output_batch")
    if set(policy.calibration_ids) & {row.prompt_id for row in rows}:
        raise RecoveryError("policy_probe_overlap")
    # Require complete canary evidence
    if any(row.healthy_canary_score is None or row.current_canary_score is None for row in rows):
        raise RecoveryError("incomplete_canary_probe")
    if canary_outputs is None:
        raise RecoveryError("missing_canary_verification")
    canary_outputs = tuple(canary_outputs)
    if len(canary_outputs) != len(rows) or any(not valid_score(value) for value in canary_outputs):
        raise RecoveryError("invalid_canary_verification")
    baseline = metrics(policy, rows, [row.healthy_score for row in rows])
    candidate = metrics(policy, rows, outputs)
    score_error = max(abs(value - row.healthy_score) for row, value in zip(rows, outputs))
    changes = sum(policy.prediction_set(value) != policy.prediction_set(row.healthy_score)
        for row, value in zip(rows, outputs))
    reasons = []
    canary_error = max(abs(value - row.healthy_canary_score) for row, value in zip(rows, canary_outputs))
    if canary_error > tolerance:
        reasons.append("baseline_canary_scores_not_restored")
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
        "max_canary_score_error": canary_error, "pair_verified": True,
        "prediction_set_changes": changes, "baseline": baseline, "candidate": candidate,
        "scope": "finite_probe_agreement_not_universal_recovery_or_safety"}


def _validate_score_batch(prompts, values):
    if not isinstance(values, (list, tuple)):
        raise RecoveryError("runtime_score_length_mismatch")
    if len(values) != len(prompts):
        raise RecoveryError("runtime_score_length_mismatch")
    for row in values:
        if not isinstance(row, dict):
            raise RecoveryError("invalid_runtime_score_batch")
        if "guardrail_score" not in row or "canary_score" not in row:
            raise RecoveryError("invalid_runtime_score_batch")
        if not valid_score(row["guardrail_score"]) or not valid_score(row["canary_score"]):
            raise RecoveryError("invalid_runtime_score_batch")
        for flag in ("guardrail_unsafe", "canary_unsafe"):
            if flag in row and type(row[flag]) is not bool:
                raise RecoveryError("invalid_runtime_flag")


class CorrectedBackend:
    def __init__(self, backend, correction):
        if backend.model_pair_id != correction.model_pair_id:
            raise RecoveryError("recovery_model_binding_mismatch")
        self.backend, self.correction = backend, correction
        self.model_pair_id = backend.model_pair_id
        self.guardrail_threshold = getattr(backend, "guardrail_threshold", None)
        self.canary_threshold = getattr(backend, "canary_threshold", None)

    def score(self, prompts):
        result = self.backend.score(prompts)
        if len(result) != len(prompts):
            raise RecoveryError("runtime_score_length_mismatch")
        _validate_score_batch(prompts, result)
        corrected = []
        for row in result:
            score = self.correction.apply(row["guardrail_score"])
            value = {**row, "guardrail_score": score}
            if "guardrail_unsafe" in row:
                if not valid_score(self.guardrail_threshold):
                    raise RecoveryError("missing_binary_threshold_binding")
                value["guardrail_unsafe"] = score >= self.guardrail_threshold
            # keep canary_unsafe consistent with unchanged canary_score
            if "canary_unsafe" in row:
                thresh = getattr(self.backend, "canary_threshold", None)
                if valid_score(thresh):
                    value["canary_unsafe"] = row["canary_score"] >= thresh
                else:
                    # if no threshold, keep original if bool, else validate
                    if type(row["canary_unsafe"]) is not bool:
                        raise RecoveryError("invalid_runtime_flag")
            corrected.append(value)
        return corrected


@dataclass
class RecoveryPlan:
    generation: int
    policy_id: str
    model_pair_id: str
    backend: Backend | None
    record: dict
    handle: str | None = None


class RecoveryManager:
    """Explicit approval and compare-and-swap activation for one runtime."""

    def __init__(self, backend, policy):
        if backend.model_pair_id != policy.model_pair_id:
            raise RecoveryError("recovery_model_binding_mismatch")
        self._backend = backend
        self._policy = policy
        self._generation = 0
        self._previous_backend = None
        self._lock = threading.RLock()
        self._store = {}

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value):
        self._backend = value

    @property
    def policy(self):
        return self._policy

    @property
    def generation(self):
        return self._generation

    @property
    def previous_backend(self):
        return self._previous_backend

    def propose(self, fit, verification, *, verification_prompts, restart_factory: Callable[[], Backend],
                approved=False, actor="", allow_restart=True):
        with self._lock:
            generation, current_backend = self._generation, self._backend
        verification_prompts = list(verification_prompts)
        record = {"applied": False, "status": "approval_required", "attempts": [],
            "policy_weights_changed": False, "policy_thresholds_changed": False}

        def make_plan(backend=None, record_dict=record, handle=None):
            # defensive copy for public record
            return RecoveryPlan(generation, self.policy.policy_id, current_backend.model_pair_id, backend, dict(record_dict), handle)

        if type(approved) is not bool or type(allow_restart) is not bool:
            raise RecoveryError("invalid_recovery_authority")
        if not approved:
            # store handle for approval_required as well to allow apply to return same status
            handle = identity({"generation": generation, "policy_id": self.policy.policy_id, "model_pair_id": current_backend.model_pair_id, "status": "approval_required"})
            with self._lock:
                self._store[handle] = {
                    "generation": generation,
                    "policy_id": self.policy.policy_id,
                    "model_pair_id": current_backend.model_pair_id,
                    "candidate": None,
                    "verification_prompts": tuple(verification_prompts),
                    "verification_probes": tuple(verification) if isinstance(verification, (list, tuple)) else (),
                    "record": dict(record),
                    "consumed": False,
                }
            return make_plan(backend=None, record_dict=record, handle=handle)
        if not isinstance(actor, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", actor) is None:
            raise RecoveryError("invalid_recovery_actor")
        record["approval_actor"] = actor
        fit = validate_batch(fit, "fit")
        verification = validate_batch(verification, "verification")
        if {row.prompt_id for row in fit} & {row.prompt_id for row in verification}:
            raise RecoveryError("recovery_fit_verification_overlap")
        if set(self.policy.calibration_ids) & {row.prompt_id for row in (*fit, *verification)}:
            raise RecoveryError("policy_probe_overlap")
        if len(verification_prompts) != len(verification):
            raise RecoveryError("verification_text_identity_mismatch")
        for row, text in zip(verification, verification_prompts):
            if not isinstance(text, str) or identity_text(text) != row.prompt_id:
                raise RecoveryError("verification_text_identity_mismatch")
        # Refresh actual current verification observations before deciding no degradation
        fresh_values = None
        fresh_error = None
        try:
            fresh_values = current_backend.score(list(verification_prompts))
            _validate_score_batch(verification_prompts, fresh_values)
        except Exception as exc:
            # Preserve stale detection: treat as degraded if scoring fails
            fresh_error = str(exc)
            fresh_values = None
        # Probe evidence indicates degradation if any fit/verification current differs from healthy
        probe_changed = any(abs(row.current_score - row.healthy_score) > 1e-8 or
            abs(row.current_canary_score - row.healthy_canary_score) > 1e-8
            for row in (*fit, *verification))
        if fresh_values is not None:
            fresh_changed = any(abs(val["guardrail_score"] - row.healthy_score) > 1e-8 or
                abs(val["canary_score"] - row.healthy_canary_score) > 1e-8
                for row, val in zip(verification, fresh_values))
            changed = probe_changed or fresh_changed
        else:
            # If fresh scoring failed, consider degraded
            changed = True
        if fresh_values is not None and not changed:
            record["status"] = "no_probe_degradation"
            handle = identity({"generation": generation, "policy_id": self.policy.policy_id, "model_pair_id": current_backend.model_pair_id, "status": "no_probe_degradation", "verification_ids": tuple(sorted(row.prompt_id for row in verification))})
            with self._lock:
                self._store[handle] = {
                    "generation": generation,
                    "policy_id": self.policy.policy_id,
                    "model_pair_id": current_backend.model_pair_id,
                    "candidate": None,
                    "verification_prompts": tuple(verification_prompts),
                    "verification_probes": tuple(verification),
                    "record": dict(record),
                    "consumed": False,
                }
            return make_plan(backend=None, record_dict=record, handle=handle)
        # Try paired affine correction - actually execute candidate on verification_prompts
        try:
            correction = fit_affine(fit, self.policy, current_backend.model_pair_id)
            candidate = CorrectedBackend(current_backend, correction)
            outputs = candidate.score(list(verification_prompts))
            _validate_score_batch(verification_prompts, outputs)
            guardrail_outputs = [row["guardrail_score"] for row in outputs]
            canary_outputs = [row["canary_score"] for row in outputs]
            check = verify_outputs(self.policy, verification, guardrail_outputs,
                canary_outputs=canary_outputs)
            attempt = {"method": "paired_affine_correction", "verification": check,
                "slope": correction.slope, "intercept": correction.intercept,
                "fit_error": correction.fit_error, "correction_id": correction.correction_id}
            record["attempts"].append(attempt)
            if check["accepted"]:
                record.update(status="verified", method="paired_affine_correction")
                handle = identity({"generation": generation, "policy_id": self.policy.policy_id, "model_pair_id": current_backend.model_pair_id, "method": "paired_affine_correction", "correction_id": correction.correction_id, "verification_ids": tuple(sorted(row.prompt_id for row in verification))})
                with self._lock:
                    self._store[handle] = {
                        "generation": generation,
                        "policy_id": self.policy.policy_id,
                        "model_pair_id": current_backend.model_pair_id,
                        "candidate": candidate,
                        "verification_prompts": tuple(verification_prompts),
                        "verification_probes": tuple(verification),
                        "record": dict(record),
                        "consumed": False,
                    }
                return make_plan(backend=candidate, record_dict=record, handle=handle)
        except RecoveryError as exc:
            # Avoid exposing private text in rejected string
            msg = str(exc)
            if "private" in msg.lower():
                msg = "restart_unavailable_or_invalid"
            record["attempts"].append({"method": "paired_affine_correction", "rejected": msg})
        except Exception:
            record["attempts"].append({"method": "paired_affine_correction", "rejected": "candidate_scoring_failed"})
        if allow_restart:
            try:
                replacement = restart_factory()
                if replacement.model_pair_id != current_backend.model_pair_id:
                    raise RecoveryError("restart_model_binding_mismatch")
                values = replacement.score(list(verification_prompts))
                _validate_score_batch(verification_prompts, values)
                guardrail_outputs = [row["guardrail_score"] for row in values]
                canary_outputs = [row["canary_score"] for row in values]
                check = verify_outputs(self.policy, verification, guardrail_outputs,
                    canary_outputs=canary_outputs)
                record["attempts"].append({"method": "trusted_scorer_restart", "verification": check})
                if check["accepted"]:
                    record.update(status="verified", method="trusted_scorer_restart")
                    handle = identity({"generation": generation, "policy_id": self.policy.policy_id, "model_pair_id": current_backend.model_pair_id, "method": "trusted_scorer_restart", "verification_ids": tuple(sorted(row.prompt_id for row in verification))})
                    with self._lock:
                        self._store[handle] = {
                            "generation": generation,
                            "policy_id": self.policy.policy_id,
                            "model_pair_id": current_backend.model_pair_id,
                            "candidate": replacement,
                            "verification_prompts": tuple(verification_prompts),
                            "verification_probes": tuple(verification),
                            "record": dict(record),
                            "consumed": False,
                        }
                    return make_plan(backend=replacement, record_dict=record, handle=handle)
            except RecoveryError as exc:
                msg = str(exc)
                # redact private text
                if "private" in msg.lower():
                    msg = "restart_unavailable_or_invalid"
                if msg == "restart_model_binding_mismatch":
                    record["attempts"].append({"method": "trusted_scorer_restart", "rejected": msg})
                else:
                    # for other RecoveryErrors, treat redacted
                    record["attempts"].append({"method": "trusted_scorer_restart", "rejected": msg if "private" not in msg.lower() else "restart_unavailable_or_invalid"})
            except (OSError, KeyError, TypeError, ValueError):
                record["attempts"].append({"method": "trusted_scorer_restart", "rejected": "restart_unavailable_or_invalid"})
            except Exception:
                record["attempts"].append({"method": "trusted_scorer_restart", "rejected": "restart_unavailable_or_invalid"})
        record["status"] = "recovery_rejected"
        handle = identity({"generation": generation, "policy_id": self.policy.policy_id, "model_pair_id": current_backend.model_pair_id, "status": "recovery_rejected", "verification_ids": tuple(sorted(row.prompt_id for row in verification)), "attempts": len(record["attempts"])})
        with self._lock:
            self._store[handle] = {
                "generation": generation,
                "policy_id": self.policy.policy_id,
                "model_pair_id": current_backend.model_pair_id,
                "candidate": None,
                "verification_prompts": tuple(verification_prompts),
                "verification_probes": tuple(verification),
                "record": dict(record),
                "consumed": False,
            }
        return make_plan(backend=None, record_dict=record, handle=handle)

    def apply(self, plan):
        with self._lock:
            return self._apply_locked(plan)

    def _apply_locked(self, plan):
        if not isinstance(plan, RecoveryPlan):
            raise RecoveryError("invalid_recovery_plan")
        handle = getattr(plan, "handle", None)
        # Generation/policy/model must match current
        if (plan.generation != self._generation or plan.policy_id != self.policy.policy_id
                or plan.model_pair_id != self._backend.model_pair_id):
            raise RecoveryError("stale_recovery_plan")
        entry = self._store.get(handle) if handle is not None else None
        if entry is None:
            raise RecoveryError("stale_recovery_plan")
        if (entry["generation"] != self._generation or entry["policy_id"] != self.policy.policy_id
                or entry["model_pair_id"] != self._backend.model_pair_id):
            raise RecoveryError("stale_recovery_plan")
        if entry.get("consumed"):
            raise RecoveryError("stale_recovery_plan")
        # Use internal record, not caller-supplied plan.record
        internal = dict(entry["record"])
        candidate = entry["candidate"]
        # If not verified or no candidate, return without activation
        if candidate is None or internal.get("status") != "verified":
            entry["consumed"] = True
            return {**internal, "generation": self._generation}
        # Reverify actual candidate on internally retained verification requests
        verification = entry["verification_probes"]
        prompts = entry["verification_prompts"]
        try:
            values = candidate.score(list(prompts))
            _validate_score_batch(prompts, values)
            guardrail_outputs = [row["guardrail_score"] for row in values]
            canary_outputs = [row["canary_score"] for row in values]
            check = verify_outputs(self.policy, verification, guardrail_outputs, canary_outputs=canary_outputs)
            if not check["accepted"]:
                entry["consumed"] = True
                # Preserve explicit unresolved status without activation
                failure = dict(internal)
                failure.update(status="recovery_rejected", applied=False, reverification=check, warning="previous_backend_may_remain_degraded")
                return {**failure, "generation": self._generation}
        except Exception:
            entry["consumed"] = True
            failure = dict(internal)
            failure.update(status="recovery_rejected", applied=False, warning="previous_backend_may_remain_degraded", reverification_error="candidate_scoring_failed")
            return {**failure, "generation": self._generation}
        # Validate model binding mismatch for candidate
        if candidate.model_pair_id != entry["model_pair_id"]:
            raise RecoveryError("recovery_model_binding_mismatch")
        self._previous_backend, self._backend = self._backend, candidate
        self._generation += 1
        entry["consumed"] = True
        return {**internal, "status": "applied", "applied": True, "generation": self._generation}

    def rollback(self, *, approved=False):
        with self._lock:
            return self._rollback_locked(approved=approved)

    def _rollback_locked(self, *, approved):
        if approved is not True or self._previous_backend is None:
            raise RecoveryError("rollback_not_authorized_or_available")
        self._backend, self._previous_backend = self._previous_backend, None
        self._generation += 1
        return {"status": "rolled_back", "generation": self._generation,
            "incident_resolved": False, "warning": "previous_backend_may_remain_degraded"}


def identity_text(text):
    import hashlib
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()

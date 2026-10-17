"""Neutral recovery fixtures. No study prompt or real result is included."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.response.conformal import ConformalPolicy, identity
from src.response.recovery import (
    Probe, RecoveryError, RecoveryManager, RecoveryPlan, fit_affine, identity_text, verify_outputs,
)

BINDING = "a" * 64
POLICY = ConformalPolicy(0.3, 0.3, 0.05, 20, 20,
    tuple(identity(["calibration", i]) for i in range(40)), BINDING, "original")


def healthy(text):
    index = int(text.split()[-1])
    return (0.05 if "safe" in text.split() else 0.75) + index / 200


class StubBackend:
    model_pair_id = BINDING

    def __init__(self, transform=lambda value: value):
        self.transform = transform

    def score(self, prompts):
        return [{"guardrail_score": self.transform(healthy(text)), "canary_score": healthy(text)} for text in prompts]


class CountingBackend(StubBackend):
    def __init__(self, transform=lambda value: value):
        super().__init__(transform)
        self.calls = 0
        self.last_prompts = None

    def score(self, prompts):
        self.calls += 1
        self.last_prompts = list(prompts)
        return super().score(prompts)


def batch(role, transform):
    texts = [f"neutral {role} {label} example {i}" for label in ("safe", "harmful") for i in range(20)]
    rows = [Probe(identity_text(text), healthy(text), transform(healthy(text)),
        int("harmful" in text.split()), role, healthy(text), transform(healthy(text))) for text in texts]
    return rows, texts


def proposal(transform, *, approved=True, restart=lambda: StubBackend(), allow_restart=True):
    manager = RecoveryManager(StubBackend(transform), POLICY)
    fit, _ = batch("fit", transform)
    verification, texts = batch("verification", transform)
    plan = manager.propose(fit, verification, verification_prompts=texts,
        restart_factory=restart, approved=approved, actor="unit-test", allow_restart=allow_restart)
    return manager, plan, texts


@pytest.mark.parametrize("transform,slope,intercept", [
    (lambda s: s * 0.25, 4, 0),
    (lambda s: s * 0.6 + 0.1, 1 / 0.6, -0.1 / 0.6),
])
def test_probe_fit_recovers_unknown_affine_parameters(transform, slope, intercept):
    manager, plan, texts = proposal(transform)
    assert plan.record["method"] == "paired_affine_correction"
    correction = plan.record["attempts"][0]
    assert correction["slope"] == pytest.approx(slope)
    assert correction["intercept"] == pytest.approx(intercept)
    assert manager.generation == 0
    receipt = manager.apply(plan)
    assert receipt["applied"] and manager.generation == 1
    assert [x["guardrail_score"] for x in manager.backend.score(texts)] == pytest.approx([healthy(text) for text in texts])
    assert not receipt["policy_weights_changed"] and not receipt["policy_thresholds_changed"]


@pytest.mark.parametrize("transform", [lambda s: s ** 2, lambda s: 0.1, lambda s: 1 - s, lambda s: min(0.2, s)])
def test_non_affine_and_information_loss_faults_use_verified_restart(transform):
    manager, plan, texts = proposal(transform)
    assert plan.record["method"] == "trusted_scorer_restart"
    assert "rejected" in plan.record["attempts"][0]
    assert manager.apply(plan)["applied"]
    assert manager.backend.score(texts) == StubBackend().score(texts)


def test_failed_restart_is_rejected_without_changing_runtime():
    manager, plan, texts = proposal(lambda s: 0.1, restart=lambda: StubBackend(lambda s: 0.1))
    before = manager.backend
    result = manager.apply(plan)
    assert result["status"] == "recovery_rejected" and not result["applied"]
    assert manager.backend is before and manager.generation == 0
    assert "baseline_scores_not_restored" in result["attempts"][-1]["verification"]["reasons"]


def test_no_approval_does_not_run_probe_or_restart_work():
    manager = RecoveryManager(StubBackend(), POLICY)
    def prohibited():
        raise AssertionError("must not load replacement")
    plan = manager.propose([], [], verification_prompts=[], restart_factory=prohibited, approved=False)
    assert manager.apply(plan)["status"] == "approval_required"


def test_healthy_scores_do_not_trigger_repair():
    manager, plan, _ = proposal(lambda s: s)
    assert manager.apply(plan)["status"] == "no_probe_degradation"
    assert not plan.record["attempts"]


def test_verification_separates_score_match_from_set_match():
    rows, _ = batch("verification", lambda s: s)
    check = verify_outputs(POLICY, rows, [row.healthy_score + 0.001 for row in rows],
        canary_outputs=[row.healthy_canary_score for row in rows])
    assert check["prediction_set_changes"] == 0
    assert not check["accepted"] and check["reasons"] == ["baseline_scores_not_restored"]


def test_mismatched_probe_text_and_label_role_reject():
    manager = RecoveryManager(StubBackend(), POLICY)
    fit, _ = batch("fit", lambda s: s * 0.25)
    verification, texts = batch("verification", lambda s: s * 0.25)
    with pytest.raises(RecoveryError, match="verification_text_identity_mismatch"):
        manager.propose(fit, verification, verification_prompts=texts[::-1], restart_factory=StubBackend,
            approved=True, actor="unit-test")
    with pytest.raises(RecoveryError, match="invalid_probe_batch"):
        fit_affine(verification, POLICY, BINDING)
    overlap = [replace(row, role="verification") for row in fit]
    with pytest.raises(RecoveryError, match="recovery_fit_verification_overlap"):
        manager.propose(fit, overlap, verification_prompts=texts, restart_factory=StubBackend,
            approved=True, actor="unit-test")


def test_second_activation_of_same_plan_rejects_and_rollback_is_explicit():
    manager, plan, _ = proposal(lambda s: s * 0.25)
    previous = manager.backend
    manager.apply(plan)
    with pytest.raises(RecoveryError, match="stale_recovery_plan"):
        manager.apply(plan)
    with pytest.raises(RecoveryError, match="rollback_not_authorized"):
        manager.rollback()
    receipt = manager.rollback(approved=True)
    assert manager.backend is previous and not receipt["incident_resolved"]
    with pytest.raises(RecoveryError, match="stale_recovery_plan"):
        manager.apply(plan)


def test_invalid_restart_binding_and_private_error_redaction():
    def bad_restart():
        replacement = StubBackend()
        replacement.model_pair_id = "b" * 64
        return replacement
    manager, plan, _ = proposal(lambda s: 0.1, restart=bad_restart)
    assert manager.apply(plan)["status"] == "recovery_rejected"
    def private_restart():
        raise ValueError("private neutral text must not appear")
    manager, plan, _ = proposal(lambda s: 0.1, restart=private_restart)
    assert "private neutral text" not in str(plan.record)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_probe_scores_are_rejected(value):
    with pytest.raises(RecoveryError, match="invalid_recovery_probe"):
        Probe("c" * 64, value, 0.1, 0, "fit", 0.1, 0.1)


def test_final_requests_cannot_enter_fit_or_verification():
    with pytest.raises(RecoveryError, match="invalid_probe_label_or_role"):
        Probe("c" * 64, 0.1, 0.1, 0, "evaluation", 0.1, 0.1)


def test_affine_only_mode_retains_rejection_for_nonlinear_fault():
    manager, plan, _ = proposal(lambda s: s ** 2, allow_restart=False)
    assert manager.apply(plan)["status"] == "recovery_rejected"


def test_correction_does_not_silently_clip_invalid_future_scores():
    manager, plan, _ = proposal(lambda s: s * 0.25)
    manager.apply(plan)
    with pytest.raises(RecoveryError, match="correction_outside_valid_range"):
        manager.backend.correction.apply(0.9)


def test_new_method_end_to_end_and_unrecoverable_controls():
    from src.monitor.disagreement import ScorePair, calibrate
    from src.replay.recovery import SCENARIOS, run_episode, score_records
    from src.replay.recovery_data import RecoveryPrompt
    def records(role, n):
        return [RecoveryPrompt(identity_text(text), text, int(label == "harmful"), "neutral", role)
            for label in ("safe", "harmful") for i in range(n)
            for text in [f"neutral {role} {label} unique {i} example {i % 20}"]]
    fit = records("fit", 20)
    verification = records("verification", 20)
    evaluation = records("evaluation", 160)
    backend = StubBackend()
    healthy_scores = score_records(backend, fit + verification + evaluation)
    refs = [ScorePair(f"ref-{i}", "unit", i, 0.1, 0.1, BINDING, "reference") for i in range(80)]
    null = [ScorePair(f"null-{i}", "unit", i, 0.1, 0.1, BINDING, "calibration") for i in range(600)]
    config = calibrate(refs, null, window_size=64, stride=16, horizon=512, alpha=0.1, null_runs=20)
    for scenario in SCENARIOS:
        events, summary = run_episode(scenario, 0, healthy_backend=backend, restart_factory=StubBackend,
            policy=POLICY, config=config, fit=fit, verification=verification, evaluation=evaluation,
            healthy=healthy_scores, approved=True, actor="unit-test")
        assert len({row["prompt_id"] for row in events}) == len(events)
        if scenario in ("stationary_safe", "attack_mixture_no_fault", "common_mode_scale"):
            assert summary["first_alarm_index"] is None
            assert not summary["applied"]
        elif scenario == "persistent_restart_fault":
            assert summary["response"]["status"] == "recovery_rejected"
            assert not summary["baseline_output_restored"]
        else:
            assert summary["baseline_output_restored"]
            assert summary["final_baseline_prediction_set_mismatches"] == 0
            response_index = summary["response"]["response_index"]
            assert response_index - summary["first_alarm_index"] == 32
            assert all(row["A_decision"] == row["B_decision"] for row in events[:response_index])
            assert all(row["B_decision"] == row["healthy_decision"] for row in events[response_index:])
            assert summary["B"] == summary["baseline"]
        assert all("text" not in event for event in events)


def test_shared_guardrail_canary_fault_requires_full_pair_restart():
    transform = lambda value: value * 0.25
    fit, _ = batch("fit", transform)
    verification, texts = batch("verification", transform)
    # shared fault: both guardrail and canary scaled – use genuinely faulty backend
    class SharedFaultBackend(StubBackend):
        def score(self, prompts):
            return [{"guardrail_score": transform(healthy(t)), "canary_score": transform(healthy(t))} for t in prompts]
    fit = [replace(row, healthy_canary_score=row.healthy_score, current_canary_score=row.current_score) for row in fit]
    verification = [replace(row, healthy_canary_score=row.healthy_score, current_canary_score=row.current_score) for row in verification]
    manager = RecoveryManager(SharedFaultBackend(transform), POLICY)
    plan = manager.propose(fit, verification, verification_prompts=texts,
        restart_factory=StubBackend, approved=True, actor="unit-test")
    assert plan.record["method"] == "trusted_scorer_restart"
    assert plan.record["attempts"][0]["verification"]["reasons"] == ["baseline_canary_scores_not_restored"]
    assert plan.record["attempts"][-1]["verification"]["pair_verified"]
    assert manager.apply(plan)["applied"]


def test_canary_only_corruption_does_not_count_as_no_degradation():
    fit, _ = batch("fit", lambda s: s)
    verification, texts = batch("verification", lambda s: s)
    fit = [replace(row, healthy_canary_score=row.healthy_score, current_canary_score=0.1) for row in fit]
    verification = [replace(row, healthy_canary_score=row.healthy_score, current_canary_score=0.1) for row in verification]
    class CanaryBrokenBackend(StubBackend):
        def score(self, prompts):
            return [{"guardrail_score": healthy(t), "canary_score": 0.1} for t in prompts]
    manager = RecoveryManager(CanaryBrokenBackend(), POLICY)
    plan = manager.propose(fit, verification, verification_prompts=texts,
        restart_factory=StubBackend, approved=True, actor="unit-test")
    assert plan.record["method"] == "trusted_scorer_restart"


def test_corrected_binary_flag_matches_corrected_score():
    class FlagBackend(StubBackend):
        guardrail_threshold = 0.5
        def score(self, prompts):
            return [{**row, "guardrail_unsafe": row["guardrail_score"] >= 0.5} for row in super().score(prompts)]
    transform = lambda value: value * 0.25
    fit, _ = batch("fit", transform)
    verification, texts = batch("verification", transform)
    manager = RecoveryManager(FlagBackend(transform), POLICY)
    plan = manager.propose(fit, verification, verification_prompts=texts,
        restart_factory=StubBackend, approved=True, actor="unit-test")
    manager.apply(plan)
    rows = manager.backend.score(texts)
    assert all(row["guardrail_unsafe"] == (row["guardrail_score"] >= 0.5) for row in rows)


def test_fault_outside_probes_remains_unresolved_not_claimed_repaired():
    class SelectiveBackend(StubBackend):
        def score(self, prompts):
            return [{"guardrail_score": 0.01 if "evaluation" in text else healthy(text),
                "canary_score": healthy(text)} for text in prompts]
    manager = RecoveryManager(SelectiveBackend(), POLICY)
    fit, _ = batch("fit", lambda s: s)
    verification, texts = batch("verification", lambda s: s)
    plan = manager.propose(fit, verification, verification_prompts=texts,
        restart_factory=StubBackend, approved=True, actor="unit-test")
    assert manager.apply(plan)["status"] == "no_probe_degradation"
    assert manager.backend.score(["neutral evaluation harmful example 0"])[0]["guardrail_score"] == 0.01


def test_source_family_separation_and_unknown_values_fail():
    from src.replay.recovery_data import family_for
    assert family_for("contrast_homonyms") == family_for("homonyms")
    assert family_for("privacy_public") == family_for("privacy_fictional") == family_for("contrast_privacy")
    assert family_for("nons_group_real_discr") == family_for("real_group_nons_discr") == family_for("contrast_discr")
    with pytest.raises(RecoveryError, match="unknown_xstest_family"):
        family_for("private neutral invalid family")


# --- New coverage for API contract ---

def test_missing_canary_evidence_rejects():
    # Probe without canary must be rejected
    with pytest.raises(RecoveryError, match="incomplete_canary_probe"):
        Probe("c" * 64, 0.1, 0.1, 0, "fit")
    with pytest.raises(RecoveryError, match="incomplete_canary_probe"):
        Probe("c" * 64, 0.1, 0.1, 0, "fit", None, None)
    with pytest.raises(RecoveryError, match="incomplete_canary_probe"):
        Probe("c" * 64, 0.1, 0.1, 0, "fit", 0.1, None)
    # Missing canary in verification also rejects
    rows, _ = batch("verification", lambda s: s)
    with pytest.raises(RecoveryError, match="missing_canary_verification"):
        verify_outputs(POLICY, rows, [row.healthy_score for row in rows])
    with pytest.raises(RecoveryError, match="missing_canary_verification"):
        verify_outputs(POLICY, rows, [row.healthy_score for row in rows], canary_outputs=None)


def test_stale_complete_probes_detected():
    # Achieved gain stale: probes say 0.25 but actual is 0.5 with broken canary
    # Proposal should not accept affine correction based on stale values
    class BrokenBackend(StubBackend):
        def __init__(self):
            super().__init__(lambda s: s * 0.5)
        def score(self, prompts):
            return [{"guardrail_score": 0.5 * healthy(t), "canary_score": 0.0} for t in prompts]

    # Build probes that are complete but stale (archived)
    fit, fit_texts = batch("fit", lambda s: s * 0.25)
    verification, verify_texts = batch("verification", lambda s: s * 0.25)
    # probes say canary matches healthy (0.25 transformed), but actual backend has canary 0.0
    # This is the stale scenario from native-stale-probe-check.json
    manager = RecoveryManager(BrokenBackend(), POLICY)
    # Need to ensure we count actual candidate calls - we can use CountingBackend wrapper
    plan = manager.propose(fit, verification, verification_prompts=verify_texts,
        restart_factory=StubBackend, approved=True, actor="unit-test", allow_restart=False)
    # Should be rejected because actual correction applied to 0.5 gain would be wrong and canary not restored
    assert plan.record["status"] == "recovery_rejected"
    receipt = manager.apply(plan)
    assert not receipt["applied"]
    assert manager.generation == 0


def test_actual_candidate_execution_on_verification():
    # Verify that propose actually scores verification prompts with candidate
    counting = CountingBackend(lambda s: s * 0.25)
    # Also need counting for candidate? CorrectedBackend will call counting.score
    manager = RecoveryManager(counting, POLICY)
    fit, _ = batch("fit", lambda s: s * 0.25)
    verification, texts = batch("verification", lambda s: s * 0.25)
    counting.calls = 0
    plan = manager.propose(fit, verification, verification_prompts=texts,
        restart_factory=StubBackend, approved=True, actor="unit-test")
    # propose should have scored verification prompts at least twice:
    # once for fresh current, once for candidate verification
    assert counting.calls >= 2
    assert plan.record["status"] == "verified"
    # apply should rescore
    candidate_calls_before = counting.calls
    # For affine correction, candidate is CorrectedBackend wrapping counting, so scoring will increment counting.calls
    receipt = manager.apply(plan)
    assert receipt["applied"]
    assert counting.calls > candidate_calls_before


def test_unregistered_plan_rejects():
    manager = RecoveryManager(StubBackend(), POLICY)
    # manually constructed plan not issued by manager
    fake = RecoveryPlan(0, POLICY.policy_id, BINDING, StubBackend(), {"status": "verified"})
    with pytest.raises(RecoveryError, match="stale_recovery_plan"):
        manager.apply(fake)
    # also with correct handle but not in store
    fake2 = RecoveryPlan(0, POLICY.policy_id, BINDING, StubBackend(), {"status": "verified"}, handle="0"*64)
    with pytest.raises(RecoveryError, match="stale_recovery_plan"):
        manager.apply(fake2)


def test_modified_plan_backend_ignored():
    manager, plan, texts = proposal(lambda s: s * 0.25)
    assert plan.record["status"] == "verified"
    original_backend = plan.backend
    # Try to mutate plan.backend to faulty one
    # Since RecoveryPlan is mutable, this assignment will succeed, but manager.apply must ignore it
    faulty = StubBackend(lambda s: 0.1)
    plan.backend = faulty
    # Also try to mutate record status
    plan.record["status"] = "verified"
    plan.record["method"] = "paired_affine_correction"
    receipt = manager.apply(plan)
    # Should still apply the original correct candidate, not faulty
    assert receipt["applied"]
    assert manager.backend is not faulty
    # Score should be healthy, not faulty constant
    assert manager.backend.score(texts) == StubBackend().score(texts)
    # Ensure original backend unchanged via mutation
    assert original_backend is not faulty


def test_cross_manager_plan_rejects():
    m1 = RecoveryManager(StubBackend(lambda s: s * 0.25), POLICY)
    m2 = RecoveryManager(StubBackend(lambda s: s * 0.25), POLICY)
    fit, _ = batch("fit", lambda s: s * 0.25)
    verification, texts = batch("verification", lambda s: s * 0.25)
    plan = m1.propose(fit, verification, verification_prompts=texts, restart_factory=StubBackend, approved=True, actor="unit-test")
    assert plan.record["status"] == "verified"
    with pytest.raises(RecoveryError, match="stale_recovery_plan"):
        m2.apply(plan)


def test_repeated_activation_rejects():
    manager, plan, _ = proposal(lambda s: s * 0.25)
    receipt = manager.apply(plan)
    assert receipt["applied"]
    with pytest.raises(RecoveryError, match="stale_recovery_plan"):
        manager.apply(plan)
    # New proposal after generation increment should have new generation
    fit, _ = batch("fit", lambda s: s * 0.25)
    verification, texts = batch("verification", lambda s: s * 0.25)
    # Manager now has corrected backend (healthy). Need to create new fault for next proposal
    # Use a fresh manager with new fault to test repeated still fails
    # Already tested that second apply raises.


def test_candidate_change_before_apply_rejects():
    class MutableBackend(StubBackend):
        def __init__(self):
            super().__init__(lambda s: s * 0.25)
            self.gain = 0.25
        def score(self, prompts):
            return [{"guardrail_score": self.gain * healthy(t), "canary_score": healthy(t)} for t in prompts]
    backend = MutableBackend()
    manager = RecoveryManager(backend, POLICY)
    fit, _ = batch("fit", lambda s: s * 0.25)
    verification, texts = batch("verification", lambda s: s * 0.25)
    plan = manager.propose(fit, verification, verification_prompts=texts, restart_factory=StubBackend, approved=True, actor="unit-test")
    assert plan.record["status"] == "verified"
    # Mutate backend gain before apply - stale external backend state
    backend.gain = 0.5
    receipt = manager.apply(plan)
    # Should reject without activation because reverification fails
    assert not receipt["applied"]
    assert receipt["status"] == "recovery_rejected"
    assert manager.generation == 0
    assert manager.backend is backend


def test_malformed_candidate_outputs_rejected():
    class BadLengthBackend(StubBackend):
        def score(self, prompts):
            # return wrong length
            return [{"guardrail_score": 0.1, "canary_score": 0.1}]
    manager, _, _ = proposal(lambda s: s ** 2, allow_restart=False)
    # Now test malformed restart
    fit, _ = batch("fit", lambda s: s ** 2)
    verification, texts = batch("verification", lambda s: s ** 2)
    manager2 = RecoveryManager(StubBackend(lambda s: s ** 2), POLICY)
    plan = manager2.propose(fit, verification, verification_prompts=texts, restart_factory=lambda: BadLengthBackend(), approved=True, actor="unit-test")
    assert plan.record["status"] == "recovery_rejected"
    assert not manager2.apply(plan)["applied"]

    class BadScoreBackend(StubBackend):
        def score(self, prompts):
            return [{"guardrail_score": 2.0, "canary_score": 0.1} for _ in prompts]
    plan2 = manager2.propose(fit, verification, verification_prompts=texts, restart_factory=lambda: BadScoreBackend(), approved=True, actor="unit-test")
    assert plan2.record["status"] == "recovery_rejected"

    class RaisesBackend(StubBackend):
        def score(self, prompts):
            raise OSError("simulated failure")
    plan3 = manager2.propose(fit, verification, verification_prompts=texts, restart_factory=lambda: RaisesBackend(), approved=True, actor="unit-test")
    assert plan3.record["status"] == "recovery_rejected"
    assert "private" not in str(plan3.record).lower()


def test_runtime_error_candidate_before_activation_rejected():
    # Candidate that will raise during reverification
    class FlakyBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.should_raise = False
        def score(self, prompts):
            if self.should_raise:
                raise ValueError("flaky failure")
            return super().score(prompts)
    flaky = FlakyBackend()
    # Need a scenario where affine succeeds but flaky will be used as restart or affine wrapper?
    # Use affine path with mutable backend that becomes flaky after propose
    manager = RecoveryManager(StubBackend(lambda s: s * 0.25), POLICY)
    fit, _ = batch("fit", lambda s: s * 0.25)
    verification, texts = batch("verification", lambda s: s * 0.25)
    plan = manager.propose(fit, verification, verification_prompts=texts, restart_factory=lambda: flaky, approved=True, actor="unit-test")
    # For affine, candidate is CorrectedBackend wrapping original faulty backend, not flaky, so flaky not involved
    # Instead test restart path where flaky is the restart candidate and becomes faulty before apply
    manager2 = RecoveryManager(StubBackend(lambda s: s ** 2), POLICY)
    fit2, _ = batch("fit", lambda s: s ** 2)
    verification2, texts2 = batch("verification", lambda s: s ** 2)
    flaky2 = FlakyBackend()
    plan2 = manager2.propose(fit2, verification2, verification_prompts=texts2, restart_factory=lambda: flaky2, approved=True, actor="unit-test")
    assert plan2.record["status"] == "verified"
    # Make flaky raise on next score (reverification)
    flaky2.should_raise = True
    receipt = manager2.apply(plan2)
    assert not receipt["applied"]
    assert receipt["status"] == "recovery_rejected"


def test_binary_flag_consistency_validated():
    class FlagBackend(StubBackend):
        guardrail_threshold = 0.5
        def score(self, prompts):
            return [{**row, "guardrail_unsafe": row["guardrail_score"] >= 0.5} for row in super().score(prompts)]
    transform = lambda value: value * 0.25
    fit, _ = batch("fit", transform)
    verification, texts = batch("verification", transform)
    manager = RecoveryManager(FlagBackend(transform), POLICY)
    plan = manager.propose(fit, verification, verification_prompts=texts, restart_factory=StubBackend, approved=True, actor="unit-test")
    manager.apply(plan)
    rows = manager.backend.score(texts)
    assert all(row["guardrail_unsafe"] == (row["guardrail_score"] >= 0.5) for row in rows)
    # Also test that whole batch validation rejects missing flags
    class MissingFlagBackend(StubBackend):
        def score(self, prompts):
            return [{"guardrail_score": 0.9, "canary_score": 0.9, "guardrail_unsafe": "not_bool"} for _ in prompts]
    m2 = RecoveryManager(StubBackend(lambda s: s ** 2), POLICY)
    f2, _ = batch("fit", lambda s: s ** 2)
    v2, t2 = batch("verification", lambda s: s ** 2)
    p2 = m2.propose(f2, v2, verification_prompts=t2, restart_factory=lambda: MissingFlagBackend(), approved=True, actor="unit-test")
    assert p2.record["status"] == "recovery_rejected"


def test_whole_runtime_batch_validation():
    class MissingFieldBackend(StubBackend):
        def score(self, prompts):
            return [{"guardrail_score": 0.1} for _ in prompts]
    manager = RecoveryManager(StubBackend(lambda s: s ** 2), POLICY)
    fit, _ = batch("fit", lambda s: s ** 2)
    verification, texts = batch("verification", lambda s: s ** 2)
    plan = manager.propose(fit, verification, verification_prompts=texts, restart_factory=lambda: MissingFieldBackend(), approved=True, actor="unit-test")
    assert plan.record["status"] == "recovery_rejected"

    class BadScoreBackend(StubBackend):
        def score(self, prompts):
            return [{"guardrail_score": float("nan"), "canary_score": 0.1} for _ in prompts]
    plan2 = manager.propose(fit, verification, verification_prompts=texts, restart_factory=lambda: BadScoreBackend(), approved=True, actor="unit-test")
    assert plan2.record["status"] == "recovery_rejected"


def test_record_defensive_copy():
    manager, plan, _ = proposal(lambda s: s * 0.25)
    # Mutate public record copy should not affect internal
    original_status = plan.record["status"]
    plan.record["status"] = "tampered"
    plan.record["applied"] = True
    # Internal still verified
    receipt = manager.apply(plan)
    assert receipt["status"] == "applied"
    assert receipt["applied"] is True
    # Public record still shows tampered but apply used internal
    assert plan.record["status"] == "tampered"


def test_deterministic_handle():
    m1 = RecoveryManager(StubBackend(lambda s: s * 0.25), POLICY)
    m2 = RecoveryManager(StubBackend(lambda s: s * 0.25), POLICY)
    fit, _ = batch("fit", lambda s: s * 0.25)
    verification, texts = batch("verification", lambda s: s * 0.25)
    p1 = m1.propose(fit, verification, verification_prompts=texts, restart_factory=StubBackend, approved=True, actor="unit-test")
    p2 = m2.propose(fit, verification, verification_prompts=texts, restart_factory=StubBackend, approved=True, actor="unit-test")
    assert p1.handle == p2.handle
    assert p1.handle is not None
    # No random IDs
    assert "private" not in p1.handle.lower()

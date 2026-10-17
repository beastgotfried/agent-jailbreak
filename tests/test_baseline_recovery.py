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
    Probe, RecoveryError, RecoveryManager, fit_affine, identity_text, verify_outputs,
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


def batch(role, transform):
    texts = [f"neutral {role} {label} example {i}" for label in ("safe", "harmful") for i in range(20)]
    rows = [Probe(identity_text(text), healthy(text), transform(healthy(text)),
        int("harmful" in text.split()), role) for text in texts]
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
    check = verify_outputs(POLICY, rows, [row.healthy_score + 0.001 for row in rows])
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
        Probe("c" * 64, value, 0.1, 0, "fit")


def test_final_requests_cannot_enter_fit_or_verification():
    with pytest.raises(RecoveryError, match="invalid_probe_label_or_role"):
        Probe("c" * 64, 0.1, 0.1, 0, "evaluation")


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
    fit = [replace(row, healthy_canary_score=row.healthy_score, current_canary_score=row.current_score) for row in fit]
    verification = [replace(row, healthy_canary_score=row.healthy_score, current_canary_score=row.current_score) for row in verification]
    manager = RecoveryManager(StubBackend(transform), POLICY)
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
    manager = RecoveryManager(StubBackend(), POLICY)
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

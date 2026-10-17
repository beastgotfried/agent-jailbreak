"""Neutral numeric fixtures for set semantics and response authority."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.response.conformal import (
    ConformalPolicy, LabelledScore, ResponseError, approved_recalibration,
    fit_policy, identity, measure, verify_candidate,
)

BINDING = "a" * 64


def batch(role, n=24, scale=1.0):
    return [LabelledScore(identity([role, label, index]), (0.1 if label == 0 else 0.9) * scale, label, role)
        for label in (0, 1) for index in range(n)]


def fitted():
    return fit_policy(batch("initial"), role="initial", model_pair_id=BINDING)


def test_class_conditional_quantile_and_decision_sets():
    policy = fitted()
    assert policy.n0 == policy.n1 == 24
    assert policy.q0 == pytest.approx(0.1)
    assert policy.q1 == pytest.approx(0.1)
    assert policy.prediction_set(0.1) == (0,)
    assert policy.prediction_set(0.9) == (1,)
    assert policy.prediction_set(0.5) == ()
    assert policy.decision(0.5) == "review"
    assert policy.decision(0.1) == "allow"
    assert policy.decision(0.9) == "refuse"
    assert measure(policy, batch("evaluation"))["label_set_coverage"] == 1


def test_insufficient_class_labels_and_invalid_roles_fail():
    with pytest.raises(ResponseError, match="insufficient_class_calibration_labels"):
        fit_policy(batch("initial", n=4), role="initial", model_pair_id=BINDING)
    with pytest.raises(ResponseError, match="invalid_calibration_role"):
        fit_policy(batch("evaluation"), role="evaluation", model_pair_id=BINDING)
    rows = batch("initial")
    rows.append(rows[0])
    with pytest.raises(ResponseError, match="duplicate_calibration_identity"):
        fit_policy(rows, role="initial", model_pair_id=BINDING)


def test_positive_control_recalibration_requires_separate_verification():
    before = fitted()
    adapted = batch("response", scale=0.25)
    verification = batch("verification", scale=0.25)
    after, report = approved_recalibration(before, adapted, verification, approved=True,
        actor="unit-test", model_pair_id=BINDING, score_regime="silent_scale_0.25")
    assert report["applied"] and report["status"] == "applied"
    assert not report["classifier_weights_changed"]
    assert after.policy_id != before.policy_id
    assert verify_candidate(after, verification)["accepted"]
    assert measure(before, batch("evaluation", scale=0.25))["by_class"]["1"]["coverage"] == 0
    assert measure(after, batch("evaluation", scale=0.25))["by_class"]["1"]["coverage"] == 1
    assert before == fitted()


def test_no_approval_does_not_fit_or_apply_a_candidate():
    before = fitted()
    after, report = approved_recalibration(before, [], [], approved=False,
        actor="", model_pair_id=BINDING, score_regime="original")
    assert after is before and report == {"status": "approval_required", "applied": False}


def test_all_review_is_not_a_successful_repair():
    before = fitted()
    # Both quantiles become 1: full sets have coverage 1 but review rate 1.
    adversarial_fit = [replace(row, score=1.0 if row.label == 0 else 0.0) for row in batch("response")]
    candidate = fit_policy(adversarial_fit, role="response", model_pair_id=BINDING)
    assert candidate.prediction_set(0.5) == (0, 1)
    check = verify_candidate(candidate, batch("verification"))
    assert check["metrics"]["label_set_coverage"] == 1
    assert not check["accepted"]
    assert "class_0_review_above_ceiling" in check["reasons"]
    after, report = approved_recalibration(before, adversarial_fit, batch("verification"), approved=True,
        actor="unit-test", model_pair_id=BINDING, score_regime="original")
    assert after is before and report["status"] == "verification_rejected"


def test_label_leakage_and_model_changes_reject():
    before = fitted()
    with pytest.raises(ResponseError, match="invalid_verification_role"):
        verify_candidate(before, batch("evaluation"))
    reused = [replace(row, role="verification") for row in batch("initial")]
    with pytest.raises(ResponseError, match="calibration_verification_overlap"):
        verify_candidate(before, reused)
    with pytest.raises(ResponseError, match="initial_response_label_overlap"):
        approved_recalibration(before, [replace(row, role="response") for row in batch("initial")], batch("verification"),
            approved=True, actor="unit-test", model_pair_id=BINDING, score_regime="original")
    with pytest.raises(ResponseError, match="response_model_binding_mismatch"):
        approved_recalibration(before, batch("response"), batch("verification"), approved=True,
            actor="unit-test", model_pair_id="b" * 64, score_regime="original")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, -0.1, 1.1, "neutral private value"])
def test_invalid_scores_do_not_echo_values(score):
    with pytest.raises(ResponseError) as error:
        LabelledScore(identity("unit"), score, 0, "initial")
    assert "neutral private value" not in str(error.value)


def test_measure_empty_and_class_counts_are_explicit():
    metrics = measure(fitted(), [])
    assert metrics["n"] == 0 and metrics["label_set_coverage"] is None
    assert metrics["by_class"]["1"]["n"] == 0
    assert metrics["unsafe_label_allow_rate"] is None

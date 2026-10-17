"""Neutral fixtures establish mechanics, not classifier effectiveness."""
from collections import Counter
from itertools import islice
import warnings

import numpy as np
import pytest

from src.data.synthetic import digest, expand, family_splits, native_only, validate_catalog, validation_roles
from src.models.synthetic import BinaryPolicy, logistic_candidate, metrics, select_threshold
from src.replay.synthetic import build_stream, endpoint_rows, episode, run, select_probes, service_metrics
from sklearn.exceptions import ConvergenceWarning


def catalog_fixture():
    return {"schema_version": 1, "domains": [
        {"id": f"domain{d}", "contexts": [f"Neutral context {d} variant {i}." for i in range(10)],
         "families": [{"id": f"domain{d}_family{f}", "semantic_group": f"domain{d}_family{f}",
            "safe": f"Describe neutral topic {d}-{f} about {{target}}.",
            "harmful": f"Compare neutral topic {d}-{f} about {{target}}.",
            "targets": [f"item {i}" for i in range(10)], "rationale": "Neutral class fixture only."}
            for f in range(10)]} for d in range(10)]}


@pytest.fixture(scope="module")
def records():
    return [r for r in expand(catalog_fixture()) if r.split != "train"]


def test_family_split_and_roles_are_fixed_before_expansion():
    c = catalog_fixture()
    assert Counter(family_splits(c).values()) == {"train": 70, "validation": 10, "test": 20}
    assert Counter(validation_roles("fixture").values()) == {
        "threshold": 50, "fallback_calibration": 30, "probe_fit": 10, "probe_verification": 10}
    rows = list(islice(expand(c), 1000))
    assert len({r.family for r in rows}) == 1
    assert len({r.prompt_id for r in rows}) == 1000
    assert Counter(r.label for r in rows) == {0: 500, 1: 500}
    assert all(rows[i].case_id == rows[i + 1].case_id for i in range(0, 1000, 2))


def test_semantic_aliases_stay_in_train_across_domains():
    c = catalog_fixture()
    for d, f in ((0, 0), (0, 1), (1, 0)):
        c["domains"][d]["families"][f]["semantic_group"] = "shared-intent"
    result = family_splits(c)
    assert result["domain0_family0"] == result["domain0_family1"] == result["domain1_family0"] == "train"
    assert Counter(result.values()) == {"train": 70, "validation": 10, "test": 20}


def test_catalog_rejects_unsafe_format_fields_and_duplicate_prototypes():
    c = catalog_fixture()
    c["domains"][0]["families"][0]["safe"] = "Describe the neutral value {target.__class__}."
    with pytest.raises(ValueError, match="invalid_catalog_slot"):
        validate_catalog(c)
    c = catalog_fixture()
    c["domains"][0]["families"][1]["safe"] = c["domains"][0]["families"][0]["safe"]
    with pytest.raises(ValueError, match="duplicate_catalog_prototype"):
        validate_catalog(c)


def test_binary_policy_has_no_review_or_conformal_claim():
    p = BinaryPolicy(.5, digest("models"), (digest("threshold-row"),))
    assert p.decision(.49) == "allow" and p.decision(.5) == "refuse"
    assert p.prediction_set(.5) == (1,)
    assert p.to_dict()["conformal_coverage_claim"] is False
    with pytest.raises(ValueError, match="invalid_prediction_score"):
        p.decision(float("nan"))


def test_threshold_uses_balanced_accuracy_and_safe_fpr_cap():
    y = np.asarray([0] * 20 + [1] * 20)
    scores = np.asarray([.1] * 19 + [.8] + [.7] * 20)
    threshold = select_threshold(y, scores)
    result = metrics(y, scores, threshold)
    assert result["fpr"] == .05 and result["recall"] == 1.0
    assert result["accuracy"] == .975
    with pytest.raises(ValueError, match="invalid_threshold_scores"):
        select_threshold([0, 1], [float("nan"), .5])


def test_nonconvergence_records_one_failed_candidate_without_retry(monkeypatch):
    calls = []
    def warn_once(self, x, y):
        calls.append(1)
        warnings.warn("neutral convergence fixture", ConvergenceWarning)
    monkeypatch.setattr("src.models.synthetic.LogisticRegression.fit", warn_once)
    attempt, candidate = logistic_candidate(None, None, None, None, None)
    assert candidate is None and calls == [1]
    assert attempt == {"method": "tfidf_logistic_C4", "valid": False, "code": "nonconverged_no_retry"}


def test_local_execution_requires_explicit_operator_flag(monkeypatch):
    monkeypatch.setattr("src.data.synthetic.platform.machine", lambda: "arm64")
    monkeypatch.delenv("SAFETY_WATCH_ALLOW_LOCAL", raising=False)
    with pytest.raises(RuntimeError, match="explicit_local_authorization_required"):
        native_only()
    monkeypatch.setenv("SAFETY_WATCH_ALLOW_LOCAL", "1")
    native_only()


def test_external_regression_paths_are_required(monkeypatch, tmp_path):
    monkeypatch.setenv("SAFETY_WATCH_ALLOW_LOCAL", "1")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="required_external_regression_inputs_missing"):
        run(tmp_path, tmp_path, output, source_commit="0" * 40, toxic_test=tmp_path / "test.csv")
    assert not output.exists()


def test_streams_pair_arms_and_keep_fault_scenarios_on_same_inputs(records):
    test = [r for r in records if r.split == "test"]
    a, b = build_stream(test, "scale", 0), build_stream(test, "constant", 0)
    assert [r.prompt_id for r in a] == [r.prompt_id for r in b]
    assert len({r.prompt_id for r in a}) == 512
    assert Counter(r.label for r in a) == {0: 256, 1: 256}
    mixed = build_stream(test, "traffic_mix_without_fault", 0)
    assert Counter(r.label for r in mixed) == {0: 160, 1: 352}
    assert Counter(r.label for r in endpoint_rows(test)) == {0: 500, 1: 500}


def test_holds_do_not_count_as_correct_automatic_decisions(records):
    rows = records[:10]
    result = service_metrics(rows, ["hold"] * 10)
    assert result["operational_accuracy"] == 0
    assert result["automatic_coverage"] == 0
    assert result["hold_rate"] == 1


class NeutralFeatures:
    def transform(self, texts):
        return texts


class NeutralEstimator:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, texts):
        values = np.asarray([.8 if "Compare neutral" in text else .2 for text in texts])
        return np.column_stack((1 - values, values))


class NeutralBackend:
    model_pair_id = digest("neutral-models")
    guardrail_threshold = .5
    canary_threshold = .5

    def __init__(self):
        self.bundle = {"guardrail": {"features": NeutralFeatures(), "estimator": NeutralEstimator(), "calibrator": None}}

    def score(self, texts):
        return [{"guardrail_score": .8 if "Compare neutral" in text else .2,
                 "canary_score": .75 if "Compare neutral" in text else .25,
                 "guardrail_unsafe": "Compare neutral" in text, "canary_unsafe": "Compare neutral" in text}
                for text in texts]


@pytest.mark.parametrize("scenario", ["healthy", "scale", "scale_then_constant", "persistent_restart_fault"])
def test_full_native_runtime_with_binary_policy(records, scenario):
    test = [r for r in records if r.split == "test"]
    fit = select_probes(records, "probe_fit", 32)
    verification = select_probes(records, "probe_verification", 64)
    policy = BinaryPolicy(.5, NeutralBackend.model_pair_id,
                          tuple(r.prompt_id for r in records if r.role == "threshold"))
    all_rows = test + fit + verification
    values = NeutralBackend().score([r.text for r in all_rows])
    healthy = {r.prompt_id: value for r, value in zip(all_rows, values)}
    events, summary, endpoint = episode(scenario, 0, factory=NeutralBackend, policy=policy, test=test,
        endpoint=endpoint_rows(test), fit=fit, verification=verification, healthy=healthy)
    assert len(events) == 512 and len(endpoint) == 1000
    assert summary["pre_fault_confirmations"] == 0
    if scenario == "healthy":
        assert summary["A_stream"] == summary["B_stream"] and summary["generation"] == 0
    elif scenario == "persistent_restart_fault":
        assert summary["held"] and not summary["restored_service"]
        assert summary["B_final"]["operational_accuracy"] == 0
    else:
        assert summary["restored_service"]
        assert summary["B_stream"]["operational_accuracy"] > summary["A_stream"]["operational_accuracy"]
        assert summary["generation"] == (2 if scenario == "scale_then_constant" else 1)
        assert all(item["confirmation_delay"] == 15 for item in summary["causal_detection"])

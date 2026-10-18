"""Neutral calibration mechanics and native estimator portability checks."""
import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.models.calibrated import TwoGuardFeatures, logit_features, split_development


def test_logit_inputs_are_bounded_without_mutation():
    first, second = np.array([0., .2, 1.]), np.array([1., .5, 0.])
    result = logit_features(first, second)
    assert result.shape == (3, 2)
    assert np.isfinite(result).all()
    assert np.array_equal(first, [0., .2, 1.])
    with pytest.raises(ValueError, match="invalid_calibration_scores"):
        logit_features([np.nan], [0.])
    with pytest.raises(ValueError, match="invalid_calibration_scores"):
        logit_features([.1, .2], [.3])


def test_head_roles_keep_matched_labels_and_groups_together():
    rows = [{"group_id": f"g{i}", "prompt_id": f"{i}-{label}", "label": label}
            for i in range(10) for label in (0, 1)]
    fit, threshold = split_development(rows)
    reverse_fit, reverse_threshold = split_development(list(reversed(rows)))
    assert len(fit) == len(threshold) == 10
    assert {r["group_id"] for r in fit}.isdisjoint({r["group_id"] for r in threshold})
    assert {r["prompt_id"] for r in fit} == {r["prompt_id"] for r in reverse_fit}
    assert {r["prompt_id"] for r in threshold} == {r["prompt_id"] for r in reverse_threshold}
    for subset in (fit, threshold):
        assert sum(r["label"] == 0 for r in subset) == 5
        assert sum(r["label"] == 1 for r in subset) == 5


def test_composite_head_round_trip_and_batch_parity(tmp_path):
    texts = ["blue circle", "green square", "blue square", "green circle"]
    features = TfidfVectorizer()
    x = features.fit_transform(texts)
    estimator = LogisticRegression(C=1, random_state=42).fit(x, [0, 1, 0, 1])
    guard = {"features": features, "estimator": estimator, "calibrator": None}
    combined = TwoGuardFeatures(guard, guard)
    assert combined.__class__.__module__ == "src.models.calibrated"
    values = combined.transform(texts)
    head = HistGradientBoostingClassifier(max_iter=2, max_leaf_nodes=3, min_samples_leaf=1,
        max_bins=8, early_stopping=False, random_state=42).fit(values, [0, 1, 0, 1])
    path = tmp_path / "neutral.joblib"
    joblib.dump({"features": combined, "head": head}, path)
    loaded = joblib.load(path)
    batch = loaded["head"].predict_proba(loaded["features"].transform(texts))
    single = np.vstack([loaded["head"].predict_proba(loaded["features"].transform([text])) for text in texts])
    assert np.array_equal(batch, single)

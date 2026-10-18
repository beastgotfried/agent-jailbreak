"""Locked word-feature canary; never used as the decision gate."""
import resource
import time
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from safety_watch.classifier.io import require

def word_defaults():
    return {
        "analyzer": "word", "ngram_range": (1, 2), "lowercase": True,
        "min_df": 2, "max_features": 100_000, "sublinear_tf": True, "norm": "l2",
        "solver": "liblinear", "C": 1.0, "class_weight": None, "max_iter": 1000,
    }

def fit_canary(config, fit, seed=42):
    require(config["analyzer"] == "word" and tuple(config["ngram_range"]) == (1, 2),
            "Canary features must be word 1-2 n-grams.")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), lowercase=config["lowercase"],
            min_df=config["min_df"], max_features=config["max_features"],
            sublinear_tf=config["sublinear_tf"], norm=config["norm"])),
        ("classifier", LogisticRegression(
            solver=config["solver"], l1_ratio=0.0, C=config["C"],
            class_weight=config["class_weight"], max_iter=config["max_iter"],
            random_state=seed)),
    ])
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(fit["user_input"], fit["unsafe"])
    return model, {
        "training_seconds": time.perf_counter() - started,
        "feature_count": len(model.named_steps["tfidf"].vocabulary_),
        "solver_iterations": model.named_steps["classifier"].n_iter_.tolist(),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }

def error_report(frame, guardrail_scores, canary_scores, guardrail_threshold, canary_threshold):
    """Development held-out-from-fitting errors, not an independence guarantee."""
    y = frame["unsafe"].to_numpy(dtype=int)
    g = np.asarray(guardrail_scores) >= guardrail_threshold
    c = np.asarray(canary_scores) >= canary_threshold
    def cohort(mask):
        target, gp, cp = y[mask], g[mask], c[mask]
        ge, ce = gp != target, cp != target
        correlation = None
        if np.unique(ge).size == 2 and np.unique(ce).size == 2:
            correlation = float(np.corrcoef(ge.astype(float), ce.astype(float))[0, 1])
        unsafe = target == 1
        return {
            "records": len(target),
            "error_indicator_pearson_phi": correlation,
            "both_wrong": int(np.sum(ge & ce)),
            "guardrail_only_wrong": int(np.sum(ge & ~ce)),
            "canary_only_wrong": int(np.sum(~ge & ce)),
            "both_correct": int(np.sum(~ge & ~ce)),
            "prediction_disagreement": int(np.sum(gp != cp)),
            "prediction_disagreement_rate": float(np.mean(gp != cp)),
            "unsafe_records": int(np.sum(unsafe)),
            "shared_unsafe_misses": int(np.sum(unsafe & ~gp & ~cp)),
            "guardrail_misses_canary_flags": int(np.sum(unsafe & ~gp & cp)),
            "canary_misses_guardrail_flags": int(np.sum(unsafe & gp & ~cp)),
        }
    return {
        "human_annotated": cohort(frame["human_annotation"].to_numpy(dtype=bool)),
        "full_validation_supplementary": cohort(np.ones(len(y), dtype=bool)),
        "scope": "Development validation held out from fitting, used for thresholds; prior research exposure.",
        "interpretation": "Different feature families do not establish independent errors. Canary never gates.",
    }

"""Character TF-IDF and logistic-regression primitives shared by every entrypoint."""
import time
import warnings
import resource
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .io import require

def build_model(config, seed=42):
    require(config["analyzer"] == "char" and tuple(config["ngram_range"]) == (3, 5),
            "Guardrail feature family must stay character 3-5.")
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=config["analyzer"], ngram_range=tuple(config["ngram_range"]),
            lowercase=config["lowercase"], min_df=config["min_df"], max_features=config["max_features"],
            sublinear_tf=config["sublinear_tf"], norm=config["norm"])),
        ("classifier", LogisticRegression(
            solver=config["solver"], l1_ratio=0.0, C=config["C"], class_weight=config["class_weight"],
            max_iter=config["max_iter"], random_state=seed)),
    ])

def score_model(model, texts):
    classes = np.asarray(model.classes_)
    require(classes.size == 2 and set(classes.tolist()) == {0, 1}, "Model classes must be binary 0/1.")
    scores = model.predict_proba(texts)[:, int(np.flatnonzero(classes == 1)[0])]
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), "Invalid classifier scores.")
    return scores

def fit_model(config, fit, seed=42):
    model = build_model(config, seed)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(fit["user_input"], fit["unsafe"])
    stats = {
        "training_seconds": time.perf_counter() - started,
        "feature_count": len(model.named_steps["tfidf"].vocabulary_),
        "solver_iterations": model.named_steps["classifier"].n_iter_.tolist(),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    return model, stats

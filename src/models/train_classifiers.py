"""Train, freeze, and evaluate the guardrail and canary prompt classifiers.

Training and scoring protocol: docs/MODELS.md.
- Guardrail: TF-IDF char n-grams (3-5, max 100k features) + logistic regression.
- Canary:   TF-IDF word n-grams (1-2, max 60k features) + logistic regression.
- Label: unsafe = (toxicity == 1) OR (jailbreaking == 1).
- Train is partitioned into fit (65%) / validation (20%) / reference (15%).
  TF-IDF and the classifiers fit ONLY on fit rows. There is no refit after
  threshold selection. The reference partition is reserved (monitor
  calibration); this script never trains or thresholds on it.
- Threshold: maximum recall subject to empirical FPR <= 5% on the validation
  partition only. Test labels never select hyperparameters or thresholds.
- Test split is evaluated exactly once, after all choices are frozen. No
  adaptation on it.

Model outputs are scores, not calibrated probabilities and not validated harm
severity. Guardrail and canary share one algorithm family (logistic regression
on TF-IDF) with different views; their errors are measured, not assumed
independent.

Deterministic (fixed seed, lbfgs solver). No network access at train time.
Never prints or logs raw prompt text: counts, ids, and hashes only.

Run (from the repository root):

    TOXICCHAT_TRAIN=/path/to/toxicchat_train.csv \
    TOXICCHAT_TEST=/path/to/toxicchat_test.csv \
    OUT_DIR=/path/to/models \
    python3 -m src.models.train_classifiers
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ENV_TRAIN = "TOXICCHAT_TRAIN"
ENV_TEST = "TOXICCHAT_TEST"
ENV_OUT = "OUT_DIR"

RANDOM_SEED = 42
FIT_FRACTION = 0.65
VALIDATION_FRACTION = 0.20
REFERENCE_FRACTION = 0.15
TARGET_FPR = 0.05
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)

# Feature caps bound the sparse CPU workload on the native LinuxONE container.
GUARDRAIL_MAX_FEATURES = 100_000
CANARY_MAX_FEATURES = 60_000
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000

REQUIRED_COLUMNS = [
    "conv_id",
    "user_input",
    "model_output",
    "human_annotation",
    "toxicity",
    "jailbreaking",
    "openai_moderation",
]

CROSS_SPLIT_DEDUPE_RULE = "normalized prompt text appears in the test split"
WITHIN_TRAIN_GROUP_RULE = (
    "train rows grouped by normalized prompt text; groups with conflicting "
    "labels are excluded entirely; one row is kept per consistent group"
)


def normalize_prompt(text: object) -> str:
    """Lowercase and collapse whitespace.

    The frozen scorer imports this function to preserve feature semantics.
    """
    if not isinstance(text, str):
        raise ValueError("prompt values must be strings")
    return " ".join(text.lower().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _binary_values(series: pd.Series, *, boolean_strings: bool = False) -> np.ndarray:
    """Reject absent or malformed values without echoing their contents."""
    if boolean_strings:
        series = series.map(
            lambda value: {"true": 1, "false": 0}.get(value.strip().lower(), value)
            if isinstance(value, str) else value
        )
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = ~numeric.isin([0, 1])
    if invalid.any():
        positions = np.flatnonzero(invalid.to_numpy()).tolist()
        raise ValueError(
            f"{series.name} must contain binary 0/1 values; "
            f"invalid row positions: {positions}"
        )
    return numeric.to_numpy(dtype=int)


def compute_labels(df: pd.DataFrame) -> np.ndarray:
    """unsafe = (toxicity == 1) OR (jailbreaking == 1); invalid labels fail."""
    toxicity = _binary_values(df["toxicity"])
    jailbreaking = _binary_values(df["jailbreaking"])
    return ((toxicity == 1) | (jailbreaking == 1)).astype(int)


def human_annotation_mask(df: pd.DataFrame) -> np.ndarray:
    """Validate the binary annotation flag (also accepts True/False strings)."""
    return _binary_values(df["human_annotation"], boolean_strings=True).astype(bool)


def dedupe_against_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Drop train rows whose normalized prompt text appears in the test split.

    Returns the kept train rows and the sorted conv_ids of every removed row.
    """
    test_keys = set(_prompt_texts(test_df))
    train_keys = train_df["user_input"].map(normalize_prompt)
    removed_mask = train_keys.isin(test_keys)
    removed_ids = sorted(train_df.loc[removed_mask, "conv_id"].astype(str).tolist())
    kept = train_df.loc[~removed_mask].reset_index(drop=True)
    return kept, removed_ids


def group_within_train(
    train_df: pd.DataFrame, y: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Group train rows by normalized prompt text; resolve duplicates.

    Groups whose rows carry conflicting labels are excluded entirely (never
    silently relabeled). Consistent groups keep their first row. Grouping
    happens before partitioning, so identical prompts cannot straddle the
    fit/validation boundary and leak.
    """
    y = np.asarray(y)
    if y.shape != (len(train_df),) or not np.isin(y, [0, 1]).all():
        raise ValueError("group labels must be binary and aligned to train rows")
    groups: dict[str, list[int]] = {}
    for position, text in enumerate(_prompt_texts(train_df)):
        groups.setdefault(text, []).append(position)

    keep_positions = []
    conflicts = []
    duplicates = []
    for text, positions in groups.items():
        source_ids = train_df.iloc[positions]["conv_id"].astype(str).tolist()
        record = {"prompt_sha256": sha256_text(text), "source_conv_ids": source_ids}
        if len(np.unique(y[positions])) > 1:
            conflicts.append({**record, "labels": sorted(set(y[positions].tolist()))})
        else:
            keep_positions.append(positions[0])
            if len(positions) > 1:
                duplicates.append({**record, "kept_conv_id": source_ids[0]})

    stats = {
        "conflicting_groups": len(conflicts),
        "conflicting_rows_removed": sum(len(g["source_conv_ids"]) for g in conflicts),
        "conflicting_conv_ids": sorted(
            source_id for g in conflicts for source_id in g["source_conv_ids"]
        ),
        "conflicting_source_groups": conflicts,
        "duplicate_groups": len(duplicates),
        "duplicate_rows_removed": sum(len(g["source_conv_ids"]) - 1 for g in duplicates),
        "duplicate_source_groups": duplicates,
    }
    kept = train_df.iloc[keep_positions].reset_index(drop=True)
    return kept, y[keep_positions], stats


def partition_roles(y: np.ndarray) -> dict[str, np.ndarray]:
    """Deterministic stratified fit/validation/reference partition."""
    y = np.asarray(y)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("retained train rows must contain both binary classes")
    indices = np.arange(len(y))
    fitval_idx, reference_idx = train_test_split(
        indices,
        test_size=int(np.ceil(len(y) * REFERENCE_FRACTION)),
        stratify=y,
        random_state=RANDOM_SEED,
    )
    # Round holdout sizes from the full retained train count, not a floating
    # relative fraction (which can spuriously ceil an exact 20% to one extra row).
    fit_idx, validation_idx = train_test_split(
        fitval_idx,
        test_size=int(np.ceil(len(y) * VALIDATION_FRACTION)),
        stratify=y[fitval_idx],
        random_state=RANDOM_SEED,
    )
    roles = {
        "fit": np.sort(fit_idx),
        "validation": np.sort(validation_idx),
        "reference": np.sort(reference_idx),
    }
    if any(len(np.unique(y[indices])) != 2 for indices in roles.values()):
        raise ValueError("not enough rows of each class for the stratified three-way split")
    return roles


def build_vectorizer(
    analyzer: str, ngram_range: tuple[int, int], max_features: int
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram_range, max_features=max_features
    )


def build_classifier() -> LogisticRegression:
    # Thread limits belong to the execution environment, not global code state.
    return LogisticRegression(
        solver="lbfgs",
        C=LOGISTIC_C,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=LOGISTIC_MAX_ITER,
    )


def fit_model(
    texts: list[str], y: np.ndarray, analyzer: str, ngram_range: tuple[int, int],
    max_features: int,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit vectorizer + classifier. Fails loudly on any convergence warning."""
    vectorizer = build_vectorizer(analyzer, ngram_range, max_features)
    features = vectorizer.fit_transform(texts)
    classifier = build_classifier()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(features, y)
    except ConvergenceWarning as exc:
        raise RuntimeError(
            f"logistic regression did not converge within "
            f"{LOGISTIC_MAX_ITER} iterations ({analyzer} view); refusing to "
            f"freeze a non-converged model"
        ) from exc
    return vectorizer, classifier


def positive_proba(classifier: LogisticRegression, features) -> np.ndarray:
    positive_index = list(classifier.classes_).index(1)
    return classifier.predict_proba(features)[:, positive_index]


def choose_threshold(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = TARGET_FPR
) -> dict:
    """Maximum-recall threshold subject to empirical FPR <= target_fpr.

    Decision rule: ``score >= threshold`` predicts unsafe. Candidate
    thresholds are the unique observed scores plus an explicit finite sentinel
    just above their maximum. Among candidates under the FPR cap, the highest
    recall wins; recall ties break toward the largest threshold. The sentinel
    therefore wins when no positive can be recalled under the cap, predicting
    everything benign on validation (not necessarily on future inputs).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if (
        y_true.ndim != 1 or scores.shape != y_true.shape
        or not np.isin(y_true, [0, 1]).all()
        or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all()
    ):
        raise ValueError("threshold inputs must be aligned binary labels and finite 0-1 scores")
    if not np.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be between 0 and 1")
    negative_scores = scores[y_true == 0]
    positive_scores = scores[y_true == 1]
    if len(negative_scores) == 0:
        raise RuntimeError(
            "validation partition has no benign rows; cannot bound empirical FPR"
        )

    sentinel = float(np.nextafter(np.max(scores), np.inf))
    if len(positive_scores) == 0:
        threshold = sentinel
        recall = None
    else:
        candidates = np.append(np.unique(scores), sentinel)
        negative_hits = np.mean(negative_scores[:, None] >= candidates[None, :], axis=0)
        positive_hits = np.mean(positive_scores[:, None] >= candidates[None, :], axis=0)
        under_cap = negative_hits <= target_fpr
        best_recall = float(positive_hits[under_cap].max())
        tied = under_cap & (positive_hits == best_recall)
        threshold = float(candidates[tied].max())
        recall = float(np.mean(positive_scores >= threshold))
    return {
        "threshold": threshold,
        "achieved_fpr": float(np.mean(negative_scores >= threshold)),
        "recall": recall,
        "uses_above_max_sentinel": threshold == sentinel,
        "policy_note": (
            "max recall under empirical FPR cap; ties choose largest threshold, "
            "including the above-max sentinel; all-benign recall is undefined"
        ),
    }


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if not len(y_true):
        return {
            "n": 0, "positives": 0, "negatives": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "roc_auc": None, "average_precision": None, "fpr": None,
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) > 1
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    return {
        "n": int(len(y_true)),
        "positives": positives,
        "negatives": negatives,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(y_true, scores)) if positives else None,
        "fpr": float(fp / negatives) if negatives else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def error_correlation(
    y_true: np.ndarray, guardrail_pred: np.ndarray, canary_pred: np.ndarray
) -> dict:
    """Disagreement of *errors* between guardrail and canary on one split.

    Both models share one algorithm family. Correlation is measured, not
    assumed positive, zero, or evidence of independent failures.
    """
    err_g = guardrail_pred != y_true
    err_c = canary_pred != y_true
    n = int(len(y_true))
    counts = {
        "both_wrong": int(np.sum(err_g & err_c)),
        "only_guardrail_wrong": int(np.sum(err_g & ~err_c)),
        "only_canary_wrong": int(np.sum(~err_g & err_c)),
        "both_right": int(np.sum(~err_g & ~err_c)),
    }
    rates = {f"{name}_rate": (count / n if n else None) for name, count in counts.items()}
    if err_g.any() and (~err_g).any() and err_c.any() and (~err_c).any():
        pearson = float(np.corrcoef(err_g.astype(float), err_c.astype(float))[0, 1])
    else:
        pearson = None  # one error vector is constant; correlation undefined
    return {"n": n, **counts, **rates, "pearson_errors": pearson}


def _json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _training_command(train_csv: Path, test_csv: Path, out_dir: Path) -> str:
    thread_prefix = "".join(
        f"{name}={shlex.quote(os.environ[name])} "
        for name in THREAD_ENV_VARS if name in os.environ
    )
    return (
        thread_prefix
        + f"{ENV_TRAIN}={shlex.quote(str(train_csv))} "
        f"{ENV_TEST}={shlex.quote(str(test_csv))} "
        f"{ENV_OUT}={shlex.quote(str(out_dir))} "
        "python3 -m src.models.train_classifiers"
    )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"dataset CSV not found: {path}")
    # Preserve IDs and literal prompt strings such as "NA"; empty label cells
    # still fail binary validation below rather than becoming benign.
    df = pd.read_csv(path, dtype={"conv_id": "string", "user_input": "string"}, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{path}: dataset must not be empty")
    for column in ("conv_id", "user_input"):
        invalid = df[column].map(lambda value: not isinstance(value, str) or not value.strip())
        if invalid.any():
            raise ValueError(
                f"{column} must be nonempty strings; invalid row positions: "
                f"{np.flatnonzero(invalid.to_numpy()).tolist()}"
            )
    compute_labels(df)
    human_annotation_mask(df)
    return df


def _prompt_texts(df: pd.DataFrame) -> list[str]:
    return df["user_input"].map(normalize_prompt).tolist()


def run_training(train_csv: str | Path, test_csv: str | Path, out_dir: str | Path) -> dict:
    """Load, dedupe, partition, fit, select thresholds, evaluate, and persist.

    Returns the run-evidence dict (also written to ``out_dir/run-evidence.json``).
    """
    train_csv, test_csv, out_dir = (
        Path(path).expanduser().resolve() for path in (train_csv, test_csv, out_dir)
    )
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output path: {out_dir}")

    train_df = _load_csv(train_csv)
    test_df = _load_csv(test_csv)
    rows_loaded = {"train": int(len(train_df)), "test": int(len(test_df))}
    cross_split_groups = len(set(_prompt_texts(train_df)) & set(_prompt_texts(test_df)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dedupe stage 1: test text only excludes train overlaps; no test adaptation.
    train_df, cross_split_removed = dedupe_against_test(train_df, test_df)
    # Dedupe stage 2: group identical prompts; drop conflicting-label groups.
    train_df, y_train, group_stats = group_within_train(
        train_df, compute_labels(train_df)
    )
    train_texts = _prompt_texts(train_df)

    roles = partition_roles(y_train)
    role_of = np.empty(len(train_df), dtype=object)
    for role, indices in roles.items():
        role_of[indices] = role

    specs = {
        "guardrail": {
            "analyzer": "char",
            "ngram_range": (3, 5),
            "max_features": GUARDRAIL_MAX_FEATURES,
        },
        "canary": {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "max_features": CANARY_MAX_FEATURES,
        },
    }

    # Fit ONLY on the fit partition. No refit anywhere downstream.
    fit_texts = [train_texts[i] for i in roles["fit"]]
    y_fit = y_train[roles["fit"]]
    fitted = {}
    for name, spec in specs.items():
        vectorizer, classifier = fit_model(
            fit_texts, y_fit, spec["analyzer"], spec["ngram_range"],
            spec["max_features"],
        )
        fitted[name] = {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "n_features": int(len(vectorizer.vocabulary_)),
        }

    # Thresholds from the validation partition only; then frozen.
    val_texts = [train_texts[i] for i in roles["validation"]]
    y_val = y_train[roles["validation"]]
    for name, model in fitted.items():
        val_scores = positive_proba(
            model["classifier"], model["vectorizer"].transform(val_texts)
        )
        selection = choose_threshold(y_val, val_scores)
        val_pred = (val_scores >= selection["threshold"]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()
        model["threshold"] = selection["threshold"]
        model["threshold_selection"] = {
            **selection,
            "target_fpr": TARGET_FPR,
            "role": "validation",
            "validation_rows": int(len(y_val)),
            "validation_negatives": int(np.sum(y_val == 0)),
            "validation_confusion": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            },
        }

    # ---- Single evaluation on the untouched test split --------------------
    y_test = compute_labels(test_df)
    test_texts = _prompt_texts(test_df)
    test_human = human_annotation_mask(test_df)
    test_predictions = {}
    metrics = {}
    for name, model in fitted.items():
        scores = positive_proba(
            model["classifier"], model["vectorizer"].transform(test_texts)
        )
        test_predictions[name] = (scores >= model["threshold"]).astype(int)
        metrics[name] = {
            "headline_human_annotated": binary_metrics(
                y_test[test_human], scores[test_human], model["threshold"]
            ),
            "secondary_all_rows": binary_metrics(y_test, scores, model["threshold"]),
        }

    correlation = {
        "all_rows": error_correlation(
            y_test, test_predictions["guardrail"], test_predictions["canary"]
        ),
        "human_annotated": error_correlation(
            y_test[test_human],
            test_predictions["guardrail"][test_human],
            test_predictions["canary"][test_human],
        ),
    }

    # ---- Split manifest: all retained source ids, hashes, roles; no text. ----
    duplicate_sources = {
        group["prompt_sha256"]: group["source_conv_ids"]
        for group in group_stats["duplicate_source_groups"]
    }
    manifest_rows = [
        {
            "conv_id": str(train_df.loc[i, "conv_id"]),
            "source_conv_ids": duplicate_sources.get(
                sha256_text(train_texts[i]), [str(train_df.loc[i, "conv_id"])]
            ),
            "prompt_sha256": sha256_text(train_texts[i]),
            "role": str(role_of[i]),
        }
        for i in sorted(range(len(train_df)), key=lambda i: str(train_df.loc[i, "conv_id"]))
    ]
    datasets = {
        "train": {
            "path": str(train_csv),
            "sha256": sha256_file(train_csv),
            "rows_loaded": rows_loaded["train"],
        },
        "test": {
            "path": str(test_csv),
            "sha256": sha256_file(test_csv),
            "rows_loaded": rows_loaded["test"],
        },
    }
    created_utc = _utc_now()
    split = {
        "fractions": {
            "fit": FIT_FRACTION,
            "validation": VALIDATION_FRACTION,
            "reference": REFERENCE_FRACTION,
        },
        "stratified": True,
        "rounding": "ceil validation and reference fractions of retained train; fit gets remainder",
        "random_seed": RANDOM_SEED,
        "counts": {role: int(len(indices)) for role, indices in roles.items()},
        "rows_after_dedupe": int(len(train_df)),
        "note": (
            "TF-IDF and classifiers fit only on fit rows; thresholds are chosen "
            "on validation rows and then frozen; the reference partition is "
            "reserved for monitor calibration and is untouched here; the test "
            "split is evaluated exactly once and never selects anything"
        ),
    }
    manifest = {
        "schema_version": 1,
        "created_utc": created_utc,
        "random_seed": RANDOM_SEED,
        "datasets": datasets,
        "split": split,
        "rows": manifest_rows,
    }
    (out_dir / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # ---- Freeze artifacts --------------------------------------------------
    artifact_files = {
        "guardrail_model": "guardrail.joblib",
        "guardrail_vectorizer": "guardrail_vectorizer.joblib",
        "canary_model": "canary.joblib",
        "canary_vectorizer": "canary_vectorizer.joblib",
    }
    payload = {
        "guardrail_model": fitted["guardrail"]["classifier"],
        "guardrail_vectorizer": fitted["guardrail"]["vectorizer"],
        "canary_model": fitted["canary"]["classifier"],
        "canary_vectorizer": fitted["canary"]["vectorizer"],
    }
    artifacts = {}
    for key, filename in artifact_files.items():
        path = out_dir / filename
        joblib.dump(payload[key], path)
        artifacts[key] = {"file": filename, "sha256": sha256_file(path)}
    artifacts["split_manifest"] = {
        "file": "split-manifest.json",
        "sha256": sha256_file(out_dir / "split-manifest.json"),
    }

    packages = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }
    execution = {
        "machine": platform.machine(),
        "system": platform.system(),
        "thread_environment": {
            name: os.environ[name]
            for name in THREAD_ENV_VARS
            if name in os.environ
        },
    }
    dedupe = {
        "normalization": "lowercase, collapse whitespace",
        "cross_split": {
            "rule": CROSS_SPLIT_DEDUPE_RULE,
            "overlapping_prompt_groups": cross_split_groups,
            "removed_count": len(cross_split_removed),
            "removed_conv_ids": cross_split_removed,
        },
        "within_train": {"rule": WITHIN_TRAIN_GROUP_RULE, **group_stats},
    }
    thresholds = {
        "target_fpr": TARGET_FPR,
        "policy": (
            "maximum recall subject to empirical FPR <= target on the "
            "validation partition; recall ties break toward the largest "
            "threshold, including a finite above-max sentinel; sentinel wins "
            "when feasible recall is zero or the slice is all-benign; "
            "decision rule is score >= threshold; not a test FPR guarantee"
        ),
        "guardrail": fitted["guardrail"]["threshold_selection"],
        "canary": fitted["canary"]["threshold_selection"],
    }
    model_specs = {
        name: {
            "vectorizer": {
                "type": "TfidfVectorizer",
                "analyzer": specs[name]["analyzer"],
                "ngram_range": list(specs[name]["ngram_range"]),
                "max_features": specs[name]["max_features"],
            },
            "classifier": {
                "type": "LogisticRegression",
                "solver": "lbfgs",
                "C": LOGISTIC_C,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED,
                "max_iter": LOGISTIC_MAX_ITER,
            },
            "n_features": fitted[name]["n_features"],
        }
        for name in specs
    }
    limitations = [
        "model outputs are scores, not calibrated probabilities and not "
        "validated measures of harm severity",
        "guardrail and canary share one algorithm family (logistic regression "
        "on TF-IDF) with different feature views; error dependence is measured "
        "and independence is not assumed",
        "detection proves change, not correctness",
    ]

    command = _training_command(train_csv, test_csv, out_dir)
    model_cards = {
        "schema_version": 1,
        "created_utc": created_utc,
        "training_command": command,
        "packages": packages,
        "execution": execution,
        "datasets": datasets,
        "label_rule": "unsafe = (toxicity == 1) OR (jailbreaking == 1)",
        "labels": {"benign": 0, "unsafe": 1},
        "normalization": "lowercase, collapse whitespace",
        "models": model_specs,
        "operating_thresholds": thresholds,
        "split": split,
        "metrics": metrics,
        "error_correlation": correlation,
        "limitations": limitations,
        "artifacts": artifacts,
    }
    (out_dir / "model-cards.json").write_text(
        json.dumps(model_cards, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    evidence = {
        "schema_version": 1,
        "created_utc": created_utc,
        "training_command": command,
        "packages": packages,
        "execution": execution,
        "datasets": datasets,
        "label_rule": "unsafe = (toxicity == 1) OR (jailbreaking == 1)",
        "labels": {"benign": 0, "unsafe": 1},
        "normalization": "lowercase, collapse whitespace",
        "models": model_specs,
        "dedupe": dedupe,
        "split": split,
        "thresholds": thresholds,
        "metrics": metrics,
        "error_correlation": correlation,
        "limitations": limitations,
        "artifacts": {
            **artifacts,
            "model_cards": {
                "file": "model-cards.json",
                "sha256": sha256_file(out_dir / "model-cards.json"),
            },
        },
        "evaluation_note": "test scored once per model after all choices froze; no reference scoring",
    }
    (out_dir / "run-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    missing = [name for name in (ENV_TRAIN, ENV_TEST, ENV_OUT) if not os.environ.get(name)]
    if missing:
        print(
            "ERROR: required environment variables are unset: "
            + ", ".join(missing)
            + "\nUsage: "
            + _training_command(
                Path("/path/to/toxicchat_train.csv"),
                Path("/path/to/toxicchat_test.csv"),
                Path("/path/to/models"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)

    evidence = run_training(
        os.environ[ENV_TRAIN], os.environ[ENV_TEST], os.environ[ENV_OUT]
    )
    dedupe = evidence["dedupe"]
    print(
        f"train rows after dedupe: {evidence['split']['rows_after_dedupe']} "
        f"(cross-split removed {dedupe['cross_split']['removed_count']}, "
        f"conflicting-label rows removed "
        f"{dedupe['within_train']['conflicting_rows_removed']}, "
        f"duplicate rows removed {dedupe['within_train']['duplicate_rows_removed']})"
    )
    print(f"partition counts: {evidence['split']['counts']}")
    for name in ("guardrail", "canary"):
        thr = evidence["thresholds"][name]
        headline = evidence["metrics"][name]["headline_human_annotated"]
        print(
            f"{name}: threshold={thr['threshold']:.6f} "
            f"achieved_fpr={thr['achieved_fpr']:.4f} "
            f"validation_recall={thr['recall']} "
            f"headline f1={headline['f1']:.4f} roc_auc={headline['roc_auc']}"
        )
    print(f"artifacts written to: {os.environ[ENV_OUT]}")


if __name__ == "__main__":
    main()

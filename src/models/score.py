"""Frozen scorer for the replay harness.

Loads the guardrail and canary artifacts written by
src/models/train_classifiers.py and scores prompts at the frozen operating
thresholds. Scores are not calibrated probabilities or validated harm severity.
Deterministic; no network access; never echoes raw prompt text into its outputs.
Only load trusted artifacts: joblib can execute code. Hashes detect corruption,
not authenticity; matching metadata and hashes cannot make untrusted files safe.

Usage:

    from src.models.score import load_scorer

    scorer = load_scorer("/path/to/models")
    rows = scorer.score(["prompt one", "prompt two"])
    # -> [{"guardrail_score": ..., "guardrail_unsafe": ...,
    #      "canary_score": ..., "canary_unsafe": ...}, ...]
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .train_classifiers import normalize_prompt, positive_proba, sha256_file

GUARDRAIL_MODEL_FILE = "guardrail.joblib"
GUARDRAIL_VECTORIZER_FILE = "guardrail_vectorizer.joblib"
CANARY_MODEL_FILE = "canary.joblib"
CANARY_VECTORIZER_FILE = "canary_vectorizer.joblib"
MODEL_CARDS_FILE = "model-cards.json"


class FrozenScorer:
    """Load trusted guardrail/canary artifacts after version and hash checks."""

    def __init__(self, artifact_dir: str | Path):
        artifact_dir = Path(artifact_dir).expanduser()
        required = [
            GUARDRAIL_MODEL_FILE,
            GUARDRAIL_VECTORIZER_FILE,
            CANARY_MODEL_FILE,
            CANARY_VECTORIZER_FILE,
            MODEL_CARDS_FILE,
        ]
        missing = [name for name in required if not (artifact_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{artifact_dir}: missing frozen artifacts: {missing}"
            )

        cards = json.loads(
            (artifact_dir / MODEL_CARDS_FILE).read_text(encoding="utf-8")
        )
        trained_version = cards["packages"]["scikit_learn"]
        if trained_version != sklearn.__version__:
            raise ValueError(
                f"scikit-learn version mismatch: artifacts={trained_version}, "
                f"runtime={sklearn.__version__}; use the training environment"
            )

        # Check every required artifact before deserializing even the first one.
        expected_files = {
            "guardrail_model": GUARDRAIL_MODEL_FILE,
            "guardrail_vectorizer": GUARDRAIL_VECTORIZER_FILE,
            "canary_model": CANARY_MODEL_FILE,
            "canary_vectorizer": CANARY_VECTORIZER_FILE,
            "split_manifest": "split-manifest.json",
        }
        for key, filename in expected_files.items():
            entry = cards["artifacts"][key]
            if entry["file"] != filename:
                raise ValueError(f"unexpected artifact filename for {key}")
            path = artifact_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen artifact: {filename}")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {filename}; refusing to load")

        thresholds = cards["operating_thresholds"]
        self.guardrail_threshold = float(thresholds["guardrail"]["threshold"])
        self.canary_threshold = float(thresholds["canary"]["threshold"])
        if not np.isfinite([self.guardrail_threshold, self.canary_threshold]).all():
            raise ValueError("frozen thresholds must be finite")

        self._guardrail_model = joblib.load(artifact_dir / GUARDRAIL_MODEL_FILE)
        self._guardrail_vectorizer = joblib.load(
            artifact_dir / GUARDRAIL_VECTORIZER_FILE
        )
        self._canary_model = joblib.load(artifact_dir / CANARY_MODEL_FILE)
        self._canary_vectorizer = joblib.load(
            artifact_dir / CANARY_VECTORIZER_FILE
        )

    def score(self, prompts: list[str]) -> list[dict]:
        """Score prompts with both models at the frozen thresholds.

        Returns scores (0-1, not calibrated probabilities), binary decisions,
        and all-zero feature flags, in input order. An all-zero vector gives an
        intercept-only score; it is not evidence of benignness.
        """
        if not prompts:
            return []
        texts = [normalize_prompt(prompt) for prompt in prompts]
        guardrail_features = self._guardrail_vectorizer.transform(texts)
        canary_features = self._canary_vectorizer.transform(texts)
        guardrail_scores = positive_proba(self._guardrail_model, guardrail_features)
        canary_scores = positive_proba(self._canary_model, canary_features)
        guardrail_zero = guardrail_features.getnnz(axis=1) == 0
        canary_zero = canary_features.getnnz(axis=1) == 0
        return [
            {
                "guardrail_score": float(guardrail),
                "guardrail_unsafe": bool(guardrail >= self.guardrail_threshold),
                "guardrail_all_zero_features": bool(guardrail_zero[i]),
                "canary_score": float(canary),
                "canary_unsafe": bool(canary >= self.canary_threshold),
                "canary_all_zero_features": bool(canary_zero[i]),
            }
            for i, (guardrail, canary) in enumerate(zip(guardrail_scores, canary_scores))
        ]


def load_scorer(artifact_dir: str | Path) -> FrozenScorer:
    return FrozenScorer(artifact_dir)


def score(prompts: list[str], artifact_dir: str | Path) -> list[dict]:
    """One-shot convenience wrapper around load_scorer(...).score(...)."""
    return load_scorer(artifact_dir).score(prompts)

"""Tests for the classifier training pipeline (src/models/train_classifiers.py).

###############################################################################
# SYNTHETIC TEST FIXTURE — neutral nonsense strings generated in this file.
# This fixture exists only to exercise the pipeline mechanics in tests.
# It is NEVER used as model training or evaluation data (see AGENTS.md rule:
# no synthetic data anywhere in the study itself).
###############################################################################
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import auc, precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import score as score_mod  # noqa: E402
from src.models import train_classifiers as tc  # noqa: E402

TRAIN_ROWS_PER_CLASS = 40
TEST_ROWS_PER_CLASS = 12
# Four removed rows belong to three cross-split groups, not four groups.
OVERLAP_TRAIN_IDS = ["train-000-s", "train-001-u", "train-002-s", "train-overlap-extra"]
DUPLICATE_TRAIN_IDS = ["train-duplicate-1", "train-duplicate-2"]
CONFLICT_TRAIN_IDS = ["train-conflict-0", "train-conflict-1"]


def _fixture_prompt(split: str, row_id: str, unsafe: bool) -> str:
    if unsafe:
        return f"tracking fixture {split} {row_id} {row_id.replace('-', '')} violet triangle"
    return f"tracking fixture {split} {row_id} {row_id.replace('-', '')} yellow circle"


def _fixture_frame(split: str, rows_per_class: int) -> pd.DataFrame:
    rows = []
    for i in range(rows_per_class):
        for unsafe in (False, True):
            row_id = f"{split}-{i:03d}-{'u' if unsafe else 's'}"
            # Spread the three label combinations across unsafe rows.
            toxicity = 1 if unsafe and i % 3 != 1 else 0
            jailbreaking = 1 if unsafe and i % 3 != 0 else 0
            rows.append(
                {
                    "conv_id": row_id,
                    "user_input": _fixture_prompt(split, row_id, unsafe),
                    "model_output": "fixture model output",
                    "human_annotation": i % 4 != 3,  # both classes annotated
                    "toxicity": toxicity,
                    "jailbreaking": jailbreaking,
                    "openai_moderation": "{}",
                }
            )
    return pd.DataFrame(rows)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    train = _fixture_frame("train", TRAIN_ROWS_PER_CLASS)
    test = _fixture_frame("test", TEST_ROWS_PER_CLASS)

    # Consistent duplicates and a conflicting group must resolve BEFORE split.
    for source_id, kept_id in [
        (OVERLAP_TRAIN_IDS[-1], "train-000-s"),
        *[(source_id, "train-003-s") for source_id in DUPLICATE_TRAIN_IDS],
    ]:
        row = train.loc[train["conv_id"] == kept_id].iloc[0].to_dict()
        row.update(conv_id=source_id, user_input="  " + row["user_input"].upper() + "\t ")
        train.loc[len(train)] = row
    for label, source_id in enumerate(CONFLICT_TRAIN_IDS):
        row = train.iloc[0].to_dict()
        row.update(conv_id=source_id, user_input="conflicting fixture neutral oval", toxicity=label)
        train.loc[len(train)] = row

    # Three test prompts overlap four train rows; case/whitespace is immaterial.
    overlap_prompts = train.loc[
        train["conv_id"].isin(OVERLAP_TRAIN_IDS[:-1]), "user_input"
    ].tolist()
    for j, prompt in enumerate(overlap_prompts):
        mangled = "  " + prompt.upper() + "\t " if j == 0 else prompt
        test.loc[len(test)] = {
            "conv_id": f"test-overlap-{j}",
            "user_input": mangled,
            "model_output": "fixture model output",
            "human_annotation": True,
            "toxicity": 0,
            "jailbreaking": 0,
            "openai_moderation": "{}",
        }

    train_csv = root / "fixture_train.csv"
    test_csv = root / "fixture_test.csv"
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    return train_csv, test_csv


@pytest.fixture(scope="session")
def trained_runs(tmp_path_factory) -> SimpleNamespace:
    """Run the full training pipeline twice on the fixture (determinism check)."""
    root = tmp_path_factory.mktemp("trained-runs")
    train_csv, test_csv = _write_fixture(root)
    out_a, out_b = root / "out-a", root / "out-b"
    original_bytes = (train_csv.read_bytes(), test_csv.read_bytes())
    fit_calls, transform_calls, score_calls = [], [], []
    original_fit = tc.fit_model
    original_transform = tc.TfidfVectorizer.transform
    original_score = tc.positive_proba

    def record_fit(texts, *args):
        fit_calls.append(list(texts))
        return original_fit(texts, *args)

    def record_transform(vectorizer, texts):
        transform_calls.append(list(texts))
        return original_transform(vectorizer, texts)

    def record_score(classifier, features):
        score_calls.append(features.shape[0])
        return original_score(classifier, features)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tc, "fit_model", record_fit)
        patch.setattr(tc.TfidfVectorizer, "transform", record_transform)
        patch.setattr(tc, "positive_proba", record_score)
        evidence_a = tc.run_training(train_csv, test_csv, out_a)
        evidence_b = tc.run_training(train_csv, test_csv, out_b)
    assert original_bytes == (train_csv.read_bytes(), test_csv.read_bytes())
    return SimpleNamespace(
        train_csv=train_csv,
        test_csv=test_csv,
        out_a=out_a,
        out_b=out_b,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
        fit_calls=fit_calls,
        transform_calls=transform_calls,
        score_calls=score_calls,
    )


# ---- Unit-level checks -------------------------------------------------------


def test_label_logic():
    df = pd.DataFrame(
        {
            "toxicity": [1, 0, 1, 0, False, "1"],
            "jailbreaking": [0, 1, 1, 0, True, 0],
        }
    )
    assert tc.compute_labels(df).tolist() == [1, 1, 1, 0, 1, 1]


def test_normalization_is_idempotent_and_collapses_whitespace():
    assert tc.normalize_prompt("  Hello\t WORLD\n ") == "hello world"
    assert tc.normalize_prompt("hello world") == "hello world"


# ---- Pipeline-level checks ---------------------------------------------------


def test_dedupe_removes_test_overlaps_and_logs_them(trained_runs):
    dedupe = trained_runs.evidence_a["dedupe"]
    cross_split = dedupe["cross_split"]
    assert cross_split["removed_count"] == len(OVERLAP_TRAIN_IDS)
    assert cross_split["removed_conv_ids"] == sorted(OVERLAP_TRAIN_IDS)
    assert cross_split["overlapping_prompt_groups"] == 3
    assert cross_split["rule"]
    within = dedupe["within_train"]
    assert within["conflicting_groups"] == 1
    assert within["conflicting_rows_removed"] == 2
    assert within["conflicting_conv_ids"] == CONFLICT_TRAIN_IDS
    assert within["duplicate_groups"] == 1
    assert within["duplicate_rows_removed"] == 2
    assert within["duplicate_source_groups"][0]["kept_conv_id"] == "train-003-s"
    assert within["duplicate_source_groups"][0]["source_conv_ids"] == [
        "train-003-s", *DUPLICATE_TRAIN_IDS
    ]
    assert within["conflicting_source_groups"][0]["labels"] == [0, 1]
    assert trained_runs.evidence_a["split"]["rows_after_dedupe"] == 77


def test_determinism_two_runs_identical_scores(trained_runs):
    scorer_a = score_mod.load_scorer(trained_runs.out_a)
    scorer_b = score_mod.load_scorer(trained_runs.out_b)
    prompts = [
        "tracking fixture probe 001 violet triangle",
        "tracking fixture probe 002 yellow circle",
        "completely unseen fixture probe alpha",
    ]
    rows_a = scorer_a.score(prompts)
    rows_b = scorer_b.score(prompts)
    assert len(rows_a) == len(prompts)
    for row_a, row_b in zip(rows_a, rows_b):
        assert row_a == row_b  # exact float equality across independent runs
    assert scorer_a.guardrail_threshold == scorer_b.guardrail_threshold
    assert scorer_a.canary_threshold == scorer_b.canary_threshold


def test_artifacts_created_and_scorer_outputs_well_formed(trained_runs):
    expected = {
        "guardrail.joblib",
        "canary.joblib",
        "guardrail_vectorizer.joblib",
        "canary_vectorizer.joblib",
        "model-cards.json",
        "run-evidence.json",
        "split-manifest.json",
    }
    assert expected == {p.name for p in trained_runs.out_a.iterdir()}

    scorer = score_mod.load_scorer(trained_runs.out_a)
    rows = scorer.score(
        [
            "tracking fixture probe 003 violet triangle",
            "tracking fixture probe 004 yellow circle",
        ]
    )
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {
            "guardrail_score",
            "guardrail_unsafe",
            "canary_score",
            "canary_unsafe",
            "guardrail_all_zero_features",
            "canary_all_zero_features",
        }
        for key in ("guardrail_score", "canary_score"):
            assert 0.0 <= row[key] <= 1.0
        for key in (
            "guardrail_unsafe", "canary_unsafe",
            "guardrail_all_zero_features", "canary_all_zero_features",
        ):
            assert isinstance(row[key], bool)

    # Frozen thresholds are applied, not recomputed.
    for row in rows:
        assert row["guardrail_unsafe"] == (
            row["guardrail_score"] >= scorer.guardrail_threshold
        )
        assert row["canary_unsafe"] == (
            row["canary_score"] >= scorer.canary_threshold
        )

    # Separable fixture: both models should rank the unsafe-style probe higher.
    assert rows[0]["guardrail_score"] > rows[1]["guardrail_score"]
    assert rows[0]["canary_score"] > rows[1]["canary_score"]


def test_model_card_contains_required_keys(trained_runs):
    cards = json.loads((trained_runs.out_a / "model-cards.json").read_text())

    # Package versions pinned in the card.
    for key in ("python", "scikit_learn", "numpy", "scipy"):
        assert cards["packages"][key]

    # Provenance: command, timestamp, dataset hashes.
    assert "python3 -m src.models.train_classifiers" in cards["training_command"]
    assert cards["created_utc"].endswith("Z")
    for split in ("train", "test"):
        assert len(cards["datasets"][split]["sha256"]) == 64

    # SHA-256 of every model artifact, matching the bytes on disk.
    assert set(cards["artifacts"]) == {
        "guardrail_model",
        "guardrail_vectorizer",
        "canary_model",
        "canary_vectorizer",
        "split_manifest",
    }
    evidence = json.loads((trained_runs.out_a / "run-evidence.json").read_text())
    for entry in evidence["artifacts"].values():
        assert len(entry["sha256"]) == 64
        assert entry["sha256"] == tc.sha256_file(trained_runs.out_a / entry["file"])

    # Empirical validation FPR is capped, not merely close to the target.
    thresholds = cards["operating_thresholds"]
    assert thresholds["target_fpr"] == 0.05
    for name in ("guardrail", "canary"):
        assert 0.0 <= thresholds[name]["threshold"] <= np.nextafter(1.0, np.inf)
        assert 0.0 <= thresholds[name]["achieved_fpr"] <= 0.05
        conf = thresholds[name]["validation_confusion"]
        assert thresholds[name]["achieved_fpr"] == conf["fp"] / (conf["tn"] + conf["fp"])
        assert sum(conf.values()) == cards["split"]["counts"]["validation"]

    # Metrics for both models: headline (human-annotated) and secondary (all).
    metric_keys = {"precision", "recall", "f1", "roc_auc", "average_precision", "fpr", "confusion"}
    for name in ("guardrail", "canary"):
        for section in ("headline_human_annotated", "secondary_all_rows"):
            block = cards["metrics"][name][section]
            assert metric_keys.issubset(block)
            assert set(block["confusion"]) == {"tn", "fp", "fn", "tp"}

    # Headline slice is exactly the human-annotated test rows.
    test_df = pd.read_csv(trained_runs.test_csv)
    expected_headline_n = int(tc.human_annotation_mask(test_df).sum())
    assert (
        cards["metrics"]["guardrail"]["headline_human_annotated"]["n"]
        == expected_headline_n
    )
    assert cards["metrics"]["guardrail"]["secondary_all_rows"]["n"] == len(test_df)

    # Common-mode-failure disclosure inputs.
    for cohort in ("all_rows", "human_annotated"):
        correlation = cards["error_correlation"][cohort]
        assert "pearson_errors" in correlation
        assert sum(correlation[key] for key in (
            "both_wrong", "only_guardrail_wrong", "only_canary_wrong", "both_right"
        )) == correlation["n"]


def test_missing_env_vars_fail_fast(monkeypatch, capsys):
    for name in (tc.ENV_TRAIN, tc.ENV_TEST, tc.ENV_OUT):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        tc.main()
    assert excinfo.value.code != 0
    stderr = capsys.readouterr().err
    for name in (tc.ENV_TRAIN, tc.ENV_TEST, tc.ENV_OUT):
        assert name in stderr


@pytest.mark.parametrize("column", ["toxicity", "jailbreaking"])
@pytest.mark.parametrize("invalid", [None, np.nan, "", "malformed", -1, 2, 0.5, "true"])
def test_compute_labels_rejects_missing_or_nonbinary_labels(column, invalid):
    frame = pd.DataFrame({"toxicity": [0], "jailbreaking": [0]})
    frame[column] = [invalid]
    with pytest.raises(ValueError, match=column):
        tc.compute_labels(frame)


def test_human_annotation_accepts_only_binary_flags():
    frame = pd.DataFrame({"human_annotation": [True, False, "TRUE", " false ", 1, 0, "1", "0"]})
    assert tc.human_annotation_mask(frame).tolist() == [True, False] * 4
    for invalid in (None, "", "unknown", 2, -1, 0.5):
        with pytest.raises(ValueError, match="human_annotation"):
            tc.human_annotation_mask(pd.DataFrame({"human_annotation": [invalid]}))


@pytest.mark.parametrize("column,invalid", [
    ("toxicity", ""), ("jailbreaking", 2), ("human_annotation", "unknown"),
    ("user_input", " \t\n "), ("user_input", None), ("conv_id", ""),
])
def test_load_csv_validates_labels_flags_and_nonempty_prompts(tmp_path, column, invalid):
    frame = _fixture_frame("validation", 1)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = invalid
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        tc._load_csv(path)


def test_load_csv_preserves_literal_strings_and_requires_columns(tmp_path):
    frame = _fixture_frame("literal", 1)
    frame.loc[0, "conv_id"] = "00001"
    frame.loc[0, "user_input"] = "NA"
    path = tmp_path / "literal.csv"
    frame.to_csv(path, index=False)
    loaded = tc._load_csv(path)
    assert loaded.loc[0, "conv_id"] == "00001"
    assert loaded.loc[0, "user_input"] == "NA"
    frame.drop(columns="toxicity").to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        tc._load_csv(path)
    frame.iloc[:0].to_csv(path, index=False)
    with pytest.raises(ValueError, match="must not be empty"):
        tc._load_csv(path)


def test_invalid_overlap_label_is_not_silently_removed(tmp_path):
    train_csv, test_csv = _write_fixture(tmp_path)
    frame = pd.read_csv(train_csv)
    frame.loc[0, "toxicity"] = np.nan  # this row would otherwise be excluded
    frame.to_csv(train_csv, index=False)
    with pytest.raises(ValueError, match="toxicity"):
        tc.run_training(train_csv, test_csv, tmp_path / "out")


def test_group_within_train_uses_positions_not_dataframe_index():
    frame = pd.DataFrame({
        "conv_id": ["a", "b", "c", "d", "e"],
        "user_input": ["  Same PROMPT", "same\tprompt", "conflict", "CONFLICT", "other"],
    }, index=[8, 2, 9, 3, 4])
    kept, labels, stats = tc.group_within_train(frame, np.array([0, 0, 0, 1, 1]))
    assert kept["conv_id"].tolist() == ["a", "e"]
    assert labels.tolist() == [0, 1]
    assert stats["duplicate_rows_removed"] == 1
    assert stats["conflicting_source_groups"] == [{
        "prompt_sha256": tc.sha256_text("conflict"),
        "source_conv_ids": ["c", "d"], "labels": [0, 1],
    }]
    with pytest.raises(ValueError, match="aligned"):
        tc.group_within_train(frame, np.array([0]))


def test_partition_roles_is_deterministic_stratified_and_disjoint():
    y = np.array([0, 1] * 100)
    first, second = tc.partition_roles(y), tc.partition_roles(y)
    assert {role: len(indices) for role, indices in first.items()} == {
        "fit": 130, "validation": 40, "reference": 30,
    }
    assert sorted(np.concatenate(list(first.values()))) == list(range(len(y)))
    for role, indices in first.items():
        np.testing.assert_array_equal(indices, second[role])
        assert y[indices].mean() == 0.5
    with pytest.raises(ValueError):
        tc.partition_roles(np.array([0] * 10))
    with pytest.raises(ValueError):
        tc.partition_roles(np.array([0, 1]))


def test_choose_threshold_honors_inclusive_ties_and_exact_fpr_cap():
    y = np.array([0] * 20 + [1, 1])
    scores = np.array([0.1] * 19 + [0.8, 0.8, 0.9])
    selected = tc.choose_threshold(y, scores)
    assert selected["threshold"] == 0.8
    assert selected["achieved_fpr"] == 0.05
    assert selected["recall"] == 1.0
    assert not selected["uses_above_max_sentinel"]
    conf = tc.binary_metrics(y, scores, selected["threshold"])["confusion"]
    assert conf == {"tn": 19, "fp": 1, "fn": 0, "tp": 2}
    # The same tied score is infeasible with only ten benign rows.
    smaller = tc.choose_threshold(y[10:], scores[10:])
    assert smaller["threshold"] == 0.9
    assert smaller["achieved_fpr"] == 0.0
    assert smaller["recall"] == 0.5


def test_choose_threshold_recall_ties_choose_largest_threshold():
    selected = tc.choose_threshold(
        np.array([0, 0, 0, 1, 1]), np.array([0.1, 0.3, 0.6, 0.7, 0.9]), target_fpr=1.0
    )
    assert selected["threshold"] == 0.7
    assert selected["recall"] == 1.0
    assert selected["achieved_fpr"] == 0.0


@pytest.mark.parametrize("scores,y", [
    ([0.5, 0.5], [0, 1]),
    ([0.1] * 19 + [0.9, 0.05], [0] * 20 + [1]),
    ([1.0, 1.0], [0, 1]),
    ([0.2, 0.8], [0, 0]),
])
def test_choose_threshold_uses_explicit_above_max_sentinel(scores, y):
    selected = tc.choose_threshold(np.array(y), np.array(scores))
    assert selected["threshold"] == np.nextafter(max(scores), np.inf)
    assert selected["uses_above_max_sentinel"]
    assert selected["achieved_fpr"] == 0.0
    assert selected["recall"] == (0.0 if 1 in y else None)
    assert not (np.array(scores) >= selected["threshold"]).any()


@pytest.mark.parametrize("seed", range(10))
def test_choose_threshold_matches_exhaustive_policy(seed):
    rng = np.random.default_rng(seed)
    y = np.array([0] * 40 + [1] * 12)
    scores = rng.integers(0, 6, size=len(y)) / 5
    selected = tc.choose_threshold(y, scores)
    candidates = [*np.unique(scores), np.nextafter(max(scores), np.inf)]
    feasible = [t for t in candidates if np.mean(scores[y == 0] >= t) <= 0.05]
    best_recall = max(np.mean(scores[y == 1] >= t) for t in feasible)
    expected = max(t for t in feasible if np.mean(scores[y == 1] >= t) == best_recall)
    assert selected["threshold"] == expected
    assert selected["recall"] == best_recall
    assert selected["achieved_fpr"] <= 0.05


@pytest.mark.parametrize("y,scores,target", [
    ([0, 1], [0.1], 0.05), ([0, 2], [0.1, 0.9], 0.05),
    ([0, 1], [np.nan, 0.9], 0.05), ([0, 1], [0.1, np.inf], 0.05),
    ([0, 1], [0.1, 1.1], 0.05), ([0, 1], [0.1, 0.9], -0.01),
    ([0, 1], [0.1, 0.9], np.nan), ([0, 1], [0.1, 0.9], 1.01),
])
def test_choose_threshold_rejects_invalid_inputs(y, scores, target):
    with pytest.raises(ValueError):
        tc.choose_threshold(np.array(y), np.array(scores), target)


def test_choose_threshold_requires_benign_validation_rows():
    with pytest.raises(RuntimeError, match="no benign"):
        tc.choose_threshold(np.array([1, 1]), np.array([0.2, 0.9]))


def test_binary_metrics_names_average_precision_and_reports_fpr():
    y, scores = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])
    metrics = tc.binary_metrics(y, scores, 0.8)
    assert metrics["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == metrics["fpr"] == 0.5
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    precision, recall, _ = precision_recall_curve(y, scores)
    assert metrics["average_precision"] != pytest.approx(auc(recall, precision))
    assert "pr_auc" not in metrics
    single_class = tc.binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)
    assert single_class["roc_auc"] is None
    assert single_class["average_precision"] is None
    empty = tc.binary_metrics(np.array([], dtype=int), np.array([]), 0.5)
    assert empty["n"] == 0 and empty["fpr"] is None


def test_error_correlation_includes_common_error_contingency():
    result = tc.error_correlation(
        np.zeros(4, dtype=int), np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
    )
    for name in ("both_wrong", "only_guardrail_wrong", "only_canary_wrong", "both_right"):
        assert result[name] == 1
        assert result[name + "_rate"] == 0.25
    assert result["pearson_errors"] == pytest.approx(0)
    assert tc.error_correlation(np.array([0]), np.array([0]), np.array([1]))["pearson_errors"] is None


def test_manifest_has_source_provenance_hashes_and_isolated_roles(trained_runs):
    manifest = json.loads((trained_runs.out_a / "split-manifest.json").read_text())
    manifest_b = json.loads((trained_runs.out_b / "split-manifest.json").read_text())
    assert manifest["rows"] == manifest_b["rows"]
    assert manifest["random_seed"] == 42
    assert manifest["datasets"] == trained_runs.evidence_a["datasets"]
    assert manifest["split"]["fractions"] == {"fit": 0.65, "validation": 0.2, "reference": 0.15}
    train = pd.read_csv(trained_runs.train_csv).set_index("conv_id")
    hashes_by_role = {role: set() for role in ("fit", "validation", "reference")}
    source_ids = []
    for row in manifest["rows"]:
        expected = tc.sha256_text(tc.normalize_prompt(train.loc[row["conv_id"], "user_input"]))
        assert row["prompt_sha256"] == expected
        for source_id in row["source_conv_ids"]:
            assert tc.sha256_text(tc.normalize_prompt(train.loc[source_id, "user_input"])) == expected
        hashes_by_role[row["role"]].add(expected)
        source_ids.extend(row["source_conv_ids"])
    assert len(set.union(*hashes_by_role.values())) == 77
    assert len(source_ids) == len(set(source_ids)) == 79
    assert not set(source_ids) & set(OVERLAP_TRAIN_IDS + CONFLICT_TRAIN_IDS)
    for role, hashes in hashes_by_role.items():
        assert len(hashes) == manifest["split"]["counts"][role]
    all_json = "".join(path.read_text() for path in trained_runs.out_a.glob("*.json"))
    for prompt in train["user_input"]:
        assert prompt not in all_json
        assert tc.normalize_prompt(prompt) not in all_json
    for name in ("train", "test"):
        assert manifest["datasets"][name]["sha256"] == tc.sha256_file(getattr(trained_runs, name + "_csv"))


def test_fit_only_vocabulary_no_refit_no_reference_scoring_and_one_test_score(trained_runs):
    manifest = json.loads((trained_runs.out_a / "split-manifest.json").read_text())
    train = pd.read_csv(trained_runs.train_csv)
    role_ids = {
        role: {row["conv_id"] for row in manifest["rows"] if row["role"] == role}
        for role in ("fit", "validation", "reference")
    }
    texts_by_role = {
        role: tc._prompt_texts(train.loc[train["conv_id"].isin(ids)])
        for role, ids in role_ids.items()
    }
    test_texts = tc._prompt_texts(pd.read_csv(trained_runs.test_csv))
    assert len(trained_runs.fit_calls) == 4  # two models in each independent run
    assert all(texts == texts_by_role["fit"] for texts in trained_runs.fit_calls)
    assert len(trained_runs.transform_calls) == 8
    assert trained_runs.transform_calls.count(texts_by_role["validation"]) == 4
    assert trained_runs.transform_calls.count(test_texts) == 4
    assert trained_runs.score_calls == [len(texts_by_role["validation"])] * 2 + [len(test_texts)] * 2 + [
        len(texts_by_role["validation"])
    ] * 2 + [len(test_texts)] * 2
    for filename in ("guardrail_vectorizer.joblib", "canary_vectorizer.joblib"):
        vectorizer = joblib.load(trained_runs.out_a / filename)
        expected = tc.TfidfVectorizer(**vectorizer.get_params()).fit(texts_by_role["fit"])
        assert vectorizer.vocabulary_ == expected.vocabulary_
        assert "fixture model output" not in vectorizer.vocabulary_
    word_vectorizer = joblib.load(trained_runs.out_a / "canary_vectorizer.joblib")
    for role in ("validation", "reference"):
        assert all(source_id.replace("-", "") not in word_vectorizer.vocabulary_ for source_id in role_ids[role])


def test_frozen_test_metrics_and_both_cohorts_match_scorer(trained_runs):
    test = pd.read_csv(trained_runs.test_csv)
    rows = score_mod.load_scorer(trained_runs.out_a).score(test["user_input"].tolist())
    y = tc.compute_labels(test)
    human = tc.human_annotation_mask(test)
    for name in ("guardrail", "canary"):
        scores = np.array([row[name + "_score"] for row in rows])
        threshold = trained_runs.evidence_a["thresholds"][name]["threshold"]
        for section, mask in (("headline_human_annotated", human), ("secondary_all_rows", np.ones(len(y), dtype=bool))):
            assert trained_runs.evidence_a["metrics"][name][section] == tc.binary_metrics(y[mask], scores[mask], threshold)
    for section, mask in (("human_annotated", human), ("all_rows", np.ones(len(y), dtype=bool))):
        predictions = {
            name: np.array([row[name + "_unsafe"] for row in rows]) for name in ("guardrail", "canary")
        }
        assert trained_runs.evidence_a["error_correlation"][section] == tc.error_correlation(
            y[mask], predictions["guardrail"][mask], predictions["canary"][mask]
        )


def test_model_settings_are_pinned(trained_runs):
    cards = json.loads((trained_runs.out_a / "model-cards.json").read_text())
    for name, analyzer, ngrams, cap in (
        ("guardrail", "char", [3, 5], 100_000), ("canary", "word", [1, 2], 60_000),
    ):
        assert cards["models"][name]["vectorizer"] == {
            "type": "TfidfVectorizer", "analyzer": analyzer,
            "ngram_range": ngrams, "max_features": cap,
        }
        assert cards["models"][name]["classifier"] == {
            "type": "LogisticRegression", "C": 1.0, "class_weight": "balanced",
            "solver": "lbfgs", "max_iter": 1000, "random_state": 42,
        }
    assert cards["labels"] == {"benign": 0, "unsafe": 1}
    assert cards["normalization"] == "lowercase, collapse whitespace"


def test_convergence_warning_is_fatal(monkeypatch):
    class NonConvergingClassifier:
        def fit(self, features, labels):
            warnings.warn("fixture convergence failure", ConvergenceWarning)

    monkeypatch.setattr(tc, "build_classifier", NonConvergingClassifier)
    with pytest.raises(RuntimeError, match="refusing to freeze"):
        tc.fit_model(["neutral circle", "neutral triangle"], np.array([0, 1]), "word", (1, 2), 60_000)


def test_refuses_nonempty_output_without_modifying_it(tmp_path, trained_runs):
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        tc.run_training(trained_runs.train_csv, trained_runs.test_csv, trained_runs.out_a)
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        tc.run_training("missing-train", "missing-test", output)
    assert marker.read_text() == "unchanged"
    assert list(output.iterdir()) == [marker]


def test_scorer_empty_input_performs_no_inference(trained_runs, monkeypatch):
    scorer = score_mod.load_scorer(trained_runs.out_a)

    def unexpected(*args):
        pytest.fail("empty input must not transform or predict")

    monkeypatch.setattr(scorer._guardrail_vectorizer, "transform", unexpected)
    monkeypatch.setattr(scorer._canary_vectorizer, "transform", unexpected)
    assert scorer.score([]) == []


def test_scorer_normalization_zero_features_and_inclusive_decision(trained_runs):
    scorer = score_mod.load_scorer(trained_runs.out_a)
    prompt = "tracking fixture violet triangle"
    canonical, variant = scorer.score([prompt, " \t" + prompt.upper() + "\n "])
    assert canonical == variant
    scorer.guardrail_threshold = canonical["guardrail_score"]
    scorer.canary_threshold = canonical["canary_score"]
    inclusive = scorer.score([prompt])[0]
    assert inclusive["guardrail_unsafe"] and inclusive["canary_unsafe"]
    zeros = scorer.score(["", " \t\n ", "☃☃☃"])
    assert zeros[0] == zeros[1] == zeros[2]
    assert all(row["guardrail_all_zero_features"] and row["canary_all_zero_features"] for row in zeros)
    assert not canonical["guardrail_all_zero_features"]
    assert not canonical["canary_all_zero_features"]


@pytest.mark.parametrize("filename", [
    "guardrail.joblib", "guardrail_vectorizer.joblib", "canary.joblib",
    "canary_vectorizer.joblib", "split-manifest.json",
])
def test_scorer_rejects_corruption_before_any_joblib_load(tmp_path, trained_runs, monkeypatch, filename):
    output = tmp_path / "corrupted"
    shutil.copytree(trained_runs.out_a, output)
    with (output / filename).open("ab") as handle:
        handle.write(b"fixture corruption")
    calls = []
    monkeypatch.setattr(score_mod.joblib, "load", lambda path: calls.append(path))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        score_mod.load_scorer(output)
    assert calls == []


def test_scorer_rejects_sklearn_mismatch_before_joblib_load(tmp_path, trained_runs, monkeypatch):
    output = tmp_path / "mismatch"
    shutil.copytree(trained_runs.out_a, output)
    path = output / "model-cards.json"
    cards = json.loads(path.read_text())
    cards["packages"]["scikit_learn"] = "incompatible-fixture-version"
    path.write_text(json.dumps(cards))
    calls = []
    monkeypatch.setattr(score_mod.joblib, "load", lambda path: calls.append(path))
    with pytest.raises(ValueError, match="scikit-learn version mismatch"):
        score_mod.load_scorer(output)
    assert calls == []


@pytest.mark.parametrize("tamper", ["filename", "threshold"])
def test_scorer_rejects_invalid_metadata_before_joblib_load(tmp_path, trained_runs, monkeypatch, tamper):
    output = tmp_path / "invalid-metadata"
    shutil.copytree(trained_runs.out_a, output)
    path = output / "model-cards.json"
    cards = json.loads(path.read_text())
    if tamper == "filename":
        cards["artifacts"]["guardrail_model"]["file"] = "../unexpected.joblib"
    else:
        cards["operating_thresholds"]["guardrail"]["threshold"] = float("nan")
    path.write_text(json.dumps(cards))
    calls = []
    monkeypatch.setattr(score_mod.joblib, "load", lambda path: calls.append(path))
    with pytest.raises(ValueError):
        score_mod.load_scorer(output)
    assert calls == []


def test_command_records_external_thread_limits(monkeypatch):
    for name in tc.THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    command = tc._training_command(Path("train with spaces.csv"), Path("test.csv"), Path("fresh-models"))
    assert command.startswith("OMP_NUM_THREADS=1 ")
    assert "TOXICCHAT_TRAIN='train with spaces.csv'" in command
    assert command.endswith("python3 -m src.models.train_classifiers")


def test_cli_uses_env_and_existing_empty_output_without_prompt_logging(tmp_path, trained_runs, monkeypatch, capsys):
    output = tmp_path / "empty-output"
    output.mkdir()
    monkeypatch.setenv(tc.ENV_TRAIN, str(trained_runs.train_csv))
    monkeypatch.setenv(tc.ENV_TEST, str(trained_runs.test_csv))
    monkeypatch.setenv(tc.ENV_OUT, str(output))
    tc.main()
    captured = capsys.readouterr()
    assert "train rows after dedupe: 77" in captured.out
    assert "partition counts:" in captured.out
    assert "achieved_fpr=" in captured.out
    for prompt in pd.read_csv(trained_runs.train_csv)["user_input"]:
        assert prompt not in captured.out + captured.err
    assert (output / "run-evidence.json").is_file()

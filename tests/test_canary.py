"""Native canary tests use pinned real records; no generated prompt data."""
import io
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest
from safety_watch.classifier.artifacts import load_guardrail
from safety_watch.classifier.io import sha256_file
from safety_watch.canary import train_canary, export_canary, load_canary, load_pair, joint_records, evaluate_pair
from safety_watch.canary import artifacts
from safety_watch.canary.cli import main
from safety_watch.canary.model import error_report

ROOT = Path("/runtime")

@pytest.fixture(scope="session")
def canary_run():
    return train_canary(ROOT, "/workspace")

@pytest.fixture
def pair_dir(tmp_path, canary_run):
    directory = tmp_path / "models"
    directory.mkdir()
    for name in ("guardrail.joblib", "guardrail.freeze.json"):
        shutil.copyfile(ROOT / "models" / name, directory / name)
    cards = json.loads((ROOT / "models/model-cards.json").read_text())
    (directory / "model-cards.json").write_text(json.dumps({"guardrail": cards["guardrail"]}))
    export_canary(canary_run.run_dir, ROOT, directory)
    return directory

def test_word_family_threshold_and_fitting_only_selection(canary_run):
    result = canary_run
    vectorizer = result.model.named_steps["tfidf"]
    assert vectorizer.analyzer == "word" and vectorizer.ngram_range == (1, 2)
    assert len(result.split.fit) == 3832 and len(result.split.validation) == 959
    assert result.card["threshold"]["observed_fpr"] <= 0.05
    search = json.loads((result.run_dir / "model-selection.json").read_text())
    outer = set(result.split.validation["prompt_id"])
    fitting = set(result.split.fit["prompt_id"])
    checks = []
    assert len(search["leaderboard"]) == 12
    for fold in search["fold_membership"]:
        a, b = set(fold["fit_ids"]), set(fold["check_ids"])
        assert not a & b and not (a | b) & outer
        assert a | b == fitting
        checks.extend(fold["check_ids"])
    assert len(checks) == len(set(checks)) == len(fitting)
    assert sha256_file(result.run_dir / "canary.development.joblib") == result.card["model_sha256"]

def test_export_preserves_guardrail_and_reloads_exact_scores(pair_dir, canary_run):
    model = load_canary(pair_dir)
    guardrail = load_guardrail(pair_dir)
    assert sha256_file(pair_dir / "guardrail.joblib") == guardrail.digest
    assert (pair_dir / "guardrail.freeze.json").read_bytes() == (ROOT / "models/guardrail.freeze.json").read_bytes()
    original = json.loads((ROOT / "models/model-cards.json").read_text())["guardrail"]
    assert guardrail.card == original
    assert model.digest == canary_run.card["model_sha256"]
    np.testing.assert_allclose(model.score(canary_run.split.validation["user_input"]),
                               canary_run.scores, rtol=0, atol=1e-12)
    assert model.score([]).size == 0
    assert export_canary(canary_run.run_dir, ROOT, pair_dir)["existing"] is True
    with pytest.raises(RuntimeError, match="sequence"):
        model.score(canary_run.split.validation["user_input"].iloc[0])

@pytest.mark.parametrize("target", ["model", "threshold"])
def test_canary_tampering_rejected_before_deserialization(pair_dir, monkeypatch, target):
    if target == "model":
        with (pair_dir / "canary.joblib").open("ab") as handle:
            handle.write(b"artifact-integrity-check")
    else:
        path = pair_dir / "model-cards.json"
        cards = json.loads(path.read_text())
        cards["canary"]["threshold"]["threshold"] += 0.01
        path.write_text(json.dumps(cards))
    def forbidden(*args, **kwargs):
        raise AssertionError("Integrity failure must precede deserialization.")
    monkeypatch.setattr(artifacts.joblib, "load", forbidden)
    with pytest.raises(RuntimeError, match="digest|integrity"):
        load_canary(pair_dir)

def test_joint_stream_scores_and_guardrail_only_decisions(pair_dir, canary_run, monkeypatch, capsys):
    frame = canary_run.split.validation.iloc[:100]
    lines = [json.dumps({"prompt_id": row.prompt_id, "user_input": row.user_input, "split": "validation"})
             for row in frame.itertuples()]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    assert main(["score", "--model-dir", str(pair_dir), "--batch-size", "17"]) == 0
    captured = capsys.readouterr()
    rows = [json.loads(line) for line in captured.out.splitlines()]
    assert len(rows) == 100 and json.loads(captured.err)["records"] == 100
    assert [row["prompt_id"] for row in rows] == frame["prompt_id"].tolist()
    assert all(set(row) == {
        "schema_version", "prompt_id", "guardrail_score", "canary_score", "timestamp", "split",
        "guardrail_decision", "guardrail_threshold", "canary_threshold", "guardrail_sha256", "canary_sha256"
    } for row in rows)
    np.testing.assert_allclose([row["canary_score"] for row in rows], canary_run.scores[:100], rtol=0, atol=1e-12)
    guardrail = load_guardrail(pair_dir)
    np.testing.assert_allclose([row["guardrail_score"] for row in rows],
                               guardrail.score(frame["user_input"]), rtol=0, atol=1e-12)
    assert all(row["guardrail_decision"] ==
               ("refuse" if row["guardrail_score"] >= guardrail.threshold else "allow") for row in rows)
    assert all("user_input" not in row for row in rows)

def test_error_correlation_matches_direct_real_error_contingency(pair_dir, canary_run):
    guardrail, canary = load_pair(pair_dir)
    frame = canary_run.split.validation
    gscores = guardrail.score(frame["user_input"])
    report = error_report(frame, gscores, canary_run.scores, guardrail.threshold, canary.threshold)
    human = frame["human_annotation"].to_numpy(dtype=bool)
    y = frame["unsafe"].to_numpy(dtype=int)[human]
    ge = (gscores[human] >= guardrail.threshold) != y
    ce = (canary_run.scores[human] >= canary.threshold) != y
    got = report["human_annotated"]
    assert got["both_wrong"] == int(np.sum(ge & ce))
    assert got["both_correct"] + got["both_wrong"] + got["guardrail_only_wrong"] + got["canary_only_wrong"] == 527
    if np.unique(ge).size == 2 and np.unique(ce).size == 2:
        assert got["error_indicator_pearson_phi"] == pytest.approx(np.corrcoef(ge, ce)[0, 1])

def test_saved_pair_reference_is_benign_private_and_reproducible(pair_dir, canary_run):
    first = evaluate_pair(ROOT, pair_dir)
    second = evaluate_pair(ROOT, pair_dir)
    assert first["reference"]["records"] == 457
    assert first["reference"]["sha256"] == second["reference"]["sha256"]
    assert first["reload_smoke_records"] == 100
    reference = [json.loads(line) for line in Path(first["reference"]["path"]).read_text().splitlines()]
    frame = canary_run.split.validation
    wanted = set(frame.loc[frame["human_annotation"] & frame["unsafe"].eq(0), "prompt_id"])
    assert {row["prompt_id"] for row in reference} == wanted
    assert not wanted & set(canary_run.split.fit["prompt_id"])
    assert all(row["split"] == "reference" and "user_input" not in row for row in reference)
    assert first["error_correlation"] == canary_run.card["error_correlation"]

def test_partial_export_and_changed_pair_are_rejected(pair_dir, canary_run):
    marker = pair_dir / "canary.freeze.json"
    marker.unlink()
    with pytest.raises(RuntimeError, match="Partial"):
        export_canary(canary_run.run_dir, ROOT, pair_dir)

def test_canary_cli_exposes_no_official_test_action():
    from safety_watch.canary.cli import parser
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate", "--split", "test"])

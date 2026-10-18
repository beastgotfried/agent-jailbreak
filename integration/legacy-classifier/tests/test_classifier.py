"""Native integration tests use only real pinned dataset rows and existing artifacts."""
import io
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from safety_watch.classifier import TrainConfig, train_guardrail, export_run, load_guardrail
from safety_watch.classifier.evaluation import read_development_run, evaluate_final
from safety_watch.classifier.metrics import choose_threshold
from safety_watch.classifier.evidence import EvidenceLog, verify_evidence
from safety_watch.classifier.io import sha256_file
from safety_watch.classifier import artifacts
from safety_watch.classifier.cli import main
import sys

ROOT = Path("/runtime")
SOURCE = ROOT / "results/guardrail/20261017T061152Z-a17252db"

@pytest.fixture(scope="session")
def source():
    assert (SOURCE / "model-card.json").is_file(), "The verified reference run must be available on the VM."
    return read_development_run(SOURCE, ROOT)

@pytest.fixture(scope="session")
def trained():
    return train_guardrail(TrainConfig(runtime_root=ROOT, project_root="/workspace"))

@pytest.fixture
def saved(tmp_path, source):
    model_dir = tmp_path / "models"
    report = export_run(SOURCE, ROOT, model_dir)
    return model_dir, report

def test_modular_fit_exactly_reproduces_verified_notebook(source, trained):
    assert trained.split.digest == source.card["split_summary"]["partitions_sha256"]
    assert trained.threshold == source.card["threshold"]["threshold"]
    np.testing.assert_allclose(trained.scores, source.scores, rtol=0, atol=1e-12)
    assert trained.card["validation_metrics"] == source.metrics
    # Pickle serialization is not canonical across independent fits. Numeric behavior
    # must match; the saved artifact digest must match its own bytes.
    assert sha256_file(trained.run_dir / "guardrail.development.joblib") == trained.card["model_sha256"]

def test_partitions_are_disjoint_and_threshold_handles_real_score_ties(source):
    a, b = source.split.fit, source.split.validation
    assert set(a["prompt_id"]).isdisjoint(b["prompt_id"])
    assert set(a["text_sha256"]).isdisjoint(b["text_sha256"])
    human = b["human_annotation"].to_numpy(dtype=bool)
    y = b["unsafe"].to_numpy(dtype=int)
    benign = np.repeat(source.scores[human & (y == 0)], 2)
    cutoff = choose_threshold(benign, 0.05)
    assert np.mean(benign >= cutoff) <= 0.05

def test_export_copies_exact_model_and_is_idempotent(saved, source):
    directory, report = saved
    assert report["model_sha256"] == sha256_file(SOURCE / "guardrail.development.joblib")
    again = export_run(SOURCE, ROOT, directory)
    assert again["existing"] is True
    assert sha256_file(directory / "guardrail.joblib") == report["model_sha256"]
    model = load_guardrail(directory)
    np.testing.assert_allclose(model.score(source.split.validation["user_input"].iloc[:100]),
                               source.scores[:100], rtol=0, atol=1e-12)
    assert model.predict([], []) == []

def test_modified_threshold_is_rejected_before_pickle_load(saved, monkeypatch):
    directory, _ = saved
    path = directory / "model-cards.json"
    cards = json.loads(path.read_text())
    cards["guardrail"]["threshold"]["threshold"] += 0.01
    path.write_text(json.dumps(cards))
    def forbidden(*args, **kwargs):
        raise AssertionError("Integrity failure must be detected before pickle loading.")
    monkeypatch.setattr(artifacts.joblib, "load", forbidden)
    with pytest.raises(RuntimeError, match="integrity"):
        load_guardrail(directory)

def test_modified_model_is_rejected_before_pickle_load(saved, monkeypatch):
    directory, _ = saved
    path = directory / "guardrail.joblib"
    with path.open("ab") as handle:
        handle.write(b"metadata-tamper-check")
    def forbidden(*args, **kwargs):
        raise AssertionError("Integrity failure must be detected before pickle loading.")
    monkeypatch.setattr(artifacts.joblib, "load", forbidden)
    with pytest.raises(RuntimeError, match="digest"):
        load_guardrail(directory)

def test_scoring_cli_streams_scores_without_echoing_real_prompts(saved, source, monkeypatch, capsys):
    directory, _ = saved
    frame = source.split.validation.iloc[:3]
    lines = [
        json.dumps({"prompt_id": row.prompt_id, "user_input": row.user_input})
        for row in frame.itertuples()
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    assert main(["score", "--model-dir", str(directory), "--batch-size", "2"]) == 0
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert len(records) == 3
    assert [r["prompt_id"] for r in records] == frame["prompt_id"].tolist()
    assert all("user_input" not in r and "canary_score" not in r for r in records)
    np.testing.assert_allclose([r["guardrail_score"] for r in records], source.scores[:3], rtol=0, atol=1e-12)
    assert all(r["decision"] == ("refuse" if r["guardrail_score"] >= r["threshold"] else "allow")
               for r in records)

def test_final_test_requires_explicit_action_before_any_data_access(tmp_path, monkeypatch, capsys):
    from safety_watch.classifier import evaluation
    def forbidden(*args, **kwargs):
        raise AssertionError("No dataset may be read without the explicit final-test action.")
    monkeypatch.setattr(evaluation, "load_data", forbidden)
    with pytest.raises(RuntimeError, match="explicit"):
        evaluate_final(tmp_path, confirmed=False)
    assert main(["evaluate", "--split", "test"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "final-test" in captured.err

def test_evidence_modification_is_detected(tmp_path, source):
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    log("verified", {"model_sha256": source.card["model_sha256"]})
    log("verified_again", {"model_sha256": source.card["model_sha256"]})
    assert log.verify()["records"] == 2
    rows = log.path.read_text().splitlines()
    altered = json.loads(rows[0])
    altered["payload"]["metadata_modified"] = True
    rows[0] = json.dumps(altered)
    log.path.write_text("\n".join(rows) + "\n")
    with pytest.raises(RuntimeError, match="digest"):
        verify_evidence(log.path)

def test_export_different_reference_cannot_replace_saved_model(saved):
    directory, report = saved
    older = ROOT / "results/guardrail/20261017T060048Z-8d82a72b"
    assert (older / "model-card.json").is_file()
    with pytest.raises(RuntimeError, match="different guardrail"):
        export_run(older, ROOT, directory)
    assert sha256_file(directory / "guardrail.joblib") == report["model_sha256"]

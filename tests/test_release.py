"""Neutral release mechanics. These tests do not load private model artifacts."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.release import cli


def test_selection_names_the_frozen_tfidf_pair():
    value = cli.selected_release()
    assert value["guardrail_method"] == "tfidf_logistic_C4"
    assert value["model_pair_id"] == "adb5b660369f0a8b9024a3d725548aeca034d86eac72cbaa3e975a4a77fa2790"
    assert value["guardrail_threshold"] == 0.20832091569900513
    assert value["public_model_weights"] is False
    assert set(value["model_files"]) == {"models.joblib", "model-cards.json", "attempts.json", "pre-evaluation-freeze.json"}


def fixture_models(tmp_path, monkeypatch):
    spec = {"model_pair_id": "a" * 64, "guardrail_threshold": .5,
            "canary_threshold": .6, "guardrail_method": "neutral_fixture"}
    cards = tmp_path / "model-cards.json"
    cards.write_text(json.dumps(spec))
    artifact = tmp_path / "models.joblib"
    artifact.write_bytes(b"neutral fixture; not a pickle")
    spec["model_files"] = {p.name: cli.file_hash(p) for p in (cards, artifact)}
    monkeypatch.setattr(cli, "selected_release", lambda: spec)
    return spec


def test_hash_checks_precede_model_deserialization(tmp_path, monkeypatch):
    fixture_models(tmp_path, monkeypatch)
    (tmp_path / "models.joblib").write_bytes(b"changed")
    calls = []
    monkeypatch.setattr(cli, "SyntheticBackend", lambda *a, **k: calls.append(True))
    with pytest.raises(cli.ReleaseError, match="hash_mismatch"):
        cli.load_selected(tmp_path)
    assert not calls


def test_selected_loader_passes_pinned_cards_hash(tmp_path, monkeypatch):
    spec = fixture_models(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "SyntheticBackend", lambda path, expected: calls.append((path, expected)) or "fixture")
    result, backend = cli.load_selected(tmp_path)
    assert backend == "fixture" and result == spec
    assert calls == [(tmp_path, spec["model_files"]["model-cards.json"])]


def test_release_refuses_model_file_symlinks(tmp_path, monkeypatch):
    fixture_models(tmp_path, monkeypatch)
    (tmp_path / "models.joblib").rename(tmp_path / "target")
    (tmp_path / "models.joblib").symlink_to("target")
    with pytest.raises(cli.ReleaseError, match="missing_or_invalid"):
        cli.verify_selected(tmp_path)


def neutral_backend():
    return SimpleNamespace(model_pair_id="a" * 64, guardrail_threshold=.5, canary_threshold=.6,
        score=lambda texts: [{"guardrail_score": .2, "canary_score": .3,
                              "guardrail_unsafe": False, "canary_unsafe": False} for _ in texts])


def test_score_output_contains_no_text_and_executes_nothing():
    output = io.StringIO()
    cli.score_lines(neutral_backend(), io.StringIO('{"text":"Describe a neutral sample."}\n'), output)
    result = json.loads(output.getvalue())
    assert "neutral sample" not in output.getvalue()
    assert result["decision"] == "allow" and result["sequence"] == 0
    assert result["monitor_enabled"] is False and result["action_executed"] is False


@pytest.mark.parametrize("line", ['{"text":""}\n', '{"text":123}\n', '{"text":"neutral","label":0}\n', 'not-json\n'])
def test_invalid_request_has_no_decision_output(line):
    output = io.StringIO()
    with pytest.raises(cli.ReleaseError):
        cli.score_lines(neutral_backend(), io.StringIO(line), output)
    assert output.getvalue() == ""


def test_input_line_has_a_memory_bound():
    output = io.StringIO()
    with pytest.raises(cli.ReleaseError, match="too_large"):
        cli.score_lines(neutral_backend(), io.StringIO("x" * 70000), output)
    assert output.getvalue() == ""


def test_backend_exceptions_are_redacted(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "native_only", lambda: None)
    def fail(path):
        raise RuntimeError("private fixture content")
    monkeypatch.setattr(cli, "load_selected", fail)
    assert cli.main(["score", "--models", str(tmp_path)]) == 2
    result = capsys.readouterr()
    assert "private fixture content" not in result.err
    assert json.loads(result.err)["error"] == "release_operation_failed"

"""Neutral mechanics tests; no study requests or pretrained model execution."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.data.synthetic import digest, file_hash
from src.models.semantic import MODEL, REVISION, StableHead, sample_fit, verify_assets
from src.replay.accuracy import backend_type


def test_semantic_fit_subset_keeps_pairs_and_lexical_groups():
    synthetic = []
    for family in ("first", "second"):
        for target in range(2):
            for context in range(3):
                for frame in range(2):
                    for label in (0, 1):
                        synthetic.append(SimpleNamespace(family=family, target_index=target,
                            context_index=context, frame_index=frame, label=label,
                            prompt_id=digest([family, target, context, frame, label]),
                            semantic_group=family, text="Neutral fixture text."))
    aegis = [{"source": "aegis", "group_id": f"single-{label}-{i}", "label": label,
              "prompt_id": digest([label, i]), "text": "Neutral example."}
             for label in (0, 1) for i in range(3)]
    aegis.extend({"source": "aegis", "group_id": "mixed", "label": label,
                   "prompt_id": digest(["mixed", label]), "text": "Plain sample."} for label in (0, 1))
    result = sample_fit(synthetic, aegis, groups_per_class=1)
    assert result == sample_fit(synthetic[::-1], aegis[::-1], groups_per_class=1)
    assert len([r for r in result if r["source"] == "synthetic"]) == 8
    selected_ids = {r["prompt_id"] for r in result}
    selected_pairs = {}
    for row in synthetic:
        if row.prompt_id in selected_ids:
            selected_pairs.setdefault((row.family, row.target_index, row.context_index, row.frame_index), set()).add(row.label)
    assert len(selected_pairs) == 4
    assert all(labels == {0, 1} for labels in selected_pairs.values())
    selected_groups = {r["group_id"] for r in result if r["source"] == "aegis"}
    assert len(selected_groups) == 3 and "mixed" in selected_groups
    assert all((r["prompt_id"] in selected_ids) == (r["group_id"] in selected_groups) for r in aegis)
    with pytest.raises(ValueError, match="invalid_semantic_fit_pair"):
        sample_fit(synthetic[1:], aegis)
    with pytest.raises(ValueError, match="invalid_semantic_group_limit"):
        sample_fit(synthetic, aegis, groups_per_class=0)


def test_stable_semantic_head_matches_classifier_without_batch_changes():
    x = np.random.default_rng(42).normal(size=(30, 6))
    y = np.asarray([0, 1] * 15)
    estimator = LogisticRegression(C=1, random_state=42).fit(x, y)
    head = StableHead(estimator)
    batch = head.predict_proba(x)
    singles = np.concatenate([head.predict_proba(row.reshape(1, -1)) for row in x])
    assert np.array_equal(batch, singles)
    assert np.allclose(batch, estimator.predict_proba(x), atol=1e-14, rtol=0)
    assert head.predict_proba(np.empty((0, 6))).shape == (0, 2)
    with pytest.raises(ValueError, match="invalid_semantic_head_input"):
        head.predict_proba([[float("nan")] * 6])
    with pytest.raises(ValueError, match="invalid_semantic_head_input"):
        head.predict_proba([[0., 1.]])


def test_semantic_assets_are_bound_and_mutation_fails(tmp_path, monkeypatch):
    import src.models.semantic as module
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"neutral fixture bytes, not model weights")
    expected = file_hash(path)
    monkeypatch.setattr(module, "WEIGHTS", expected)
    monkeypatch.setattr(module, "ASSETS", {path.name: expected})
    manifest = {"model": MODEL, "revision": REVISION,
                "files": {path.name: {"sha256": expected}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify_assets(tmp_path) == manifest
    path.write_bytes(b"changed fixture bytes")
    with pytest.raises(ValueError, match="semantic_encoder_file_changed"):
        verify_assets(tmp_path)
    manifest["files"][path.name]["sha256"] = file_hash(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="semantic_encoder_manifest_changed"):
        verify_assets(tmp_path)


def test_encoder_executes_each_request_without_a_result_cache(monkeypatch):
    import torch
    import src.models.semantic as module
    calls = []

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs["padding"] is False and kwargs["max_length"] == 256
            return {"input_ids": torch.tensor([[len(text), 2, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 0]])}

    class Encoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            calls.append(input_ids.clone())
            hidden = input_ids.unsqueeze(-1).float() + torch.arange(384).float()
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(module, "configure_native", lambda: None)
    monkeypatch.setattr(module, "verify_assets", lambda _: None)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(module.AutoModel, "from_pretrained", lambda *a, **k: Encoder())
    encoder = module.FrozenMiniLM("neutral-fixture-only")
    first = encoder.transform(["sample", "plain"])
    again = np.concatenate([encoder.transform([text]) for text in ("sample", "plain")])
    assert len(calls) == 4
    assert first.dtype == np.float64 and first.shape == (2, 384)
    assert np.array_equal(first, again)
    assert np.allclose(np.linalg.norm(first, axis=1), 1, atol=1e-7)


def test_accuracy_backend_dispatch_is_explicit(tmp_path):
    from src.models.semantic import SemanticBackend
    from src.models.synthetic import SyntheticBackend
    card = tmp_path / "model-cards.json"
    card.write_text("{}")
    assert backend_type(tmp_path) is SyntheticBackend
    card.write_text(json.dumps({"backend": "frozen_minilm"}))
    assert backend_type(tmp_path) is SemanticBackend
    card.write_text(json.dumps({"backend": "unrecognized"}))
    with pytest.raises(ValueError, match="unsupported_accuracy_backend"):
        backend_type(tmp_path)

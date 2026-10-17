"""Neutral mechanics only. No pretrained download or study inference in tests."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from src.models import neural
from src.data.synthetic import SyntheticRow, digest


def neutral_rows():
    return [SyntheticRow(digest([case, frame, label]), f"Describe neutral sample {case} {frame} {label}.",
        label, "train", "train", "neutral", f"family-{case}", f"group-{case}", digest([case]), 0, 0, frame)
        for case in range(4) for frame in range(5) for label in (0, 1)]


def tiny_model():
    torch.manual_seed(42)
    config = transformers.BertConfig(vocab_size=12, hidden_size=16, num_hidden_layers=3,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    return neural.TinyGuard(transformers.AutoModel.from_config(config, attn_implementation="eager"))


class NeutralTokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.zeros((len(texts), 128), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, text in enumerate(texts):
            values = [1] + [2 + ord(c) % 10 for c in text[:30]] + [1]
            ids[i, :len(values)] = torch.tensor(values)
            mask[i, :len(values)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_epoch_selection_preserves_case_labels_and_only_train():
    rows = neutral_rows()
    chosen = neural.epoch_rows(rows, 0)
    assert len(chosen) == 8
    assert [r.prompt_id for r in chosen] == [r.prompt_id for r in neural.epoch_rows(list(reversed(rows)), 0)]
    for case in {r.case_id for r in chosen}:
        pair = [r for r in chosen if r.case_id == case]
        assert {r.label for r in pair} == {0, 1}
        assert len({r.frame_index for r in pair}) == 1
    with pytest.raises(ValueError, match="training_role"):
        neural.epoch_rows([replace(r, split="test") for r in rows], 0)
    with pytest.raises(ValueError, match="matched_case"):
        neural.epoch_rows(rows[:-1], 0)
    with pytest.raises(ValueError, match="training_role"):
        neural.epoch_rows(rows, 2)


def test_stack_features_are_finite_and_do_not_change_inputs():
    scores = np.asarray([0., .5, 1.])
    values = neural.stack_features(np.asarray([-2., 0., 2.]), scores)
    assert values.shape == (3, 2) and np.isfinite(values).all()
    np.testing.assert_array_equal(scores, [0., .5, 1.])
    with pytest.raises(ValueError):
        neural.stack_features([1], [float("nan")])
    with pytest.raises(ValueError):
        neural.stack_features([1, 2], [.2])


def test_ensemble_needs_gain_without_safety_or_refusal_regression():
    base = {"balanced_accuracy": .95, "recall": .95, "fpr": .05}
    assert neural.ensemble_improves(base, {"balanced_accuracy": .96, "recall": .96, "fpr": .04})
    assert not neural.ensemble_improves(base, {**base, "balanced_accuracy": .954})
    assert not neural.ensemble_improves(base, {"balanced_accuracy": .97, "recall": .94, "fpr": 0.})
    assert not neural.ensemble_improves(base, {"balanced_accuracy": .97, "recall": 1., "fpr": .06})


def test_only_last_two_encoder_blocks_and_head_are_trainable():
    model = tiny_model()
    encoder, head = neural.trainable_parameters(model)
    assert encoder and head
    assert all(p.requires_grad for p in encoder + head)
    assert not any(p.requires_grad for p in model.encoder.embeddings.parameters())
    assert not any(p.requires_grad for p in model.encoder.encoder.layer[0].parameters())
    assert not any(p.requires_grad for p in model.encoder.pooler.parameters())


def test_weight_round_trip_native_byte_order_and_batch_invariance(tmp_path):
    model, tokenizer = tiny_model().eval(), NeutralTokenizer()
    neural.save_weights(model, tmp_path / "weights.npz")
    loaded = tiny_model().eval()
    neural.load_weights(loaded, tmp_path / "weights.npz")
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[key])
    loaded.double()
    texts = ["Describe a neutral sample.", "Compare two neutral samples."]
    batch = neural.logits_for(loaded, tokenizer, texts)
    singles = np.asarray([neural.logits_for(loaded, tokenizer, [text])[0] for text in texts])
    np.testing.assert_allclose(batch, singles, rtol=0, atol=1e-12)
    assert neural.logits_for(loaded, tokenizer, []).size == 0


def test_artifact_verification_rejects_mutation_and_traversal(tmp_path):
    file = tmp_path / "example.txt"
    file.write_text("neutral")
    expected = neural.file_hash(file)
    neural.verify_files(tmp_path, {"example.txt": expected})
    file.write_text("changed")
    with pytest.raises(ValueError, match="hash_mismatch"):
        neural.verify_files(tmp_path, {"example.txt": expected})
    with pytest.raises(ValueError, match="artifact_path"):
        neural.verify_files(tmp_path, {"../example.txt": expected})


def test_native_gate_has_no_local_override(monkeypatch):
    monkeypatch.setattr(neural.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("SAFETY_WATCH_ALLOW_LOCAL", "1")
    with pytest.raises(RuntimeError, match="requires_native"):
        neural.native_cpu()


def test_guardrail_view_does_not_run_canary():
    from src.replay.synthetic import GuardrailView
    calls = []

    class Estimator:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, values):
            calls.extend(values)
            return np.asarray([[.8, .2] for _ in values])

    backend = SimpleNamespace(bundle={"guardrail": {"features": neural.TextFeatures(),
        "estimator": Estimator(), "calibrator": None}, "canary": object()},
        model_pair_id="a" * 64, guardrail_threshold=.5, canary_threshold=.4)
    values = GuardrailView(backend).score(["A neutral sample."])
    assert calls == ["A neutral sample."]
    assert values[0]["guardrail_score"] == .2 and values[0]["canary_score"] == 0

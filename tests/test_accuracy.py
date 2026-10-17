"""Neutral mechanics tests; no study inference or pretrained downloads."""
import copy

import numpy as np
import pytest

from src.data.aegis import lexical_groups, require_native
from src.data.synthetic import text_id
from src.models.accuracy import group_weights, quality, rejection_codes, select_threshold


def rows():
    return [{"source": source, "label": label, "group_id": f"g{label}-{i}",
             "prompt_id": f"{source}-{label}-{i}", "text": "neutral fixture"}
            for source in ("synthetic", "aegis") for label in (0, 1) for i in range(20)]


def test_weights_balance_sources_classes_and_repeated_groups():
    data = rows()
    data += [dict(data[0]) for _ in range(9)]
    weights = group_weights(data, {"synthetic": .75, "aegis": .25})
    assert weights.sum() == pytest.approx(1)
    for source, share in (("synthetic", .75), ("aegis", .25)):
        for label in (0, 1):
            mass = sum(w for r, w in zip(data, weights) if r["source"] == source and r["label"] == label)
            assert mass == pytest.approx(share / 2)
    repeated = sum(w for r, w in zip(data, weights) if r["source"] == "synthetic" and r["group_id"] == "g0-0")
    assert repeated == pytest.approx(.75 / 40)


def test_threshold_uses_each_source_cap_and_never_changes_scores():
    data = rows()
    scores = np.array([.2 if r["label"] == 0 else .8 for r in data])
    before = scores.copy()
    threshold = select_threshold(data, scores)
    report = quality(data, scores, threshold)
    assert .2 < threshold <= .8
    assert report["equal_source_balanced_accuracy"] == pytest.approx(1)
    assert np.array_equal(before, scores)
    for q in report["sources"].values():
        assert q["raw"]["fpr"] <= .05
        assert q["group_weighted_fpr"] <= .05


def test_threshold_matches_brute_force_eligible_objective():
    data = rows()
    scores = np.random.default_rng(8).uniform(0, 1, len(data))
    chosen = select_threshold(data, scores)
    u = np.unique(scores)
    thresholds = np.unique(np.r_[0., (u[:-1] + u[1:]) / 2, 1.])
    candidates = []
    for threshold in thresholds:
        q = quality(data, scores, threshold)
        if all(max(s["raw"]["fpr"], s["group_weighted_fpr"]) <= .05 + 1e-12 for s in q["sources"].values()):
            candidates.append(q)
    best = max(candidates, key=lambda q: (round(q["equal_source_balanced_accuracy"], 12), round(q["mean_harmful_recall"], 12), q["threshold"]))
    assert chosen == best["threshold"]


def test_invalid_scores_or_source_weights_reject():
    data = rows()
    with pytest.raises(ValueError, match="invalid_development_scores"):
        select_threshold(data, np.full(len(data), np.nan))
    with pytest.raises(ValueError, match="invalid_source_weights"):
        group_weights(data, {"synthetic": 1.0})
    with pytest.raises(ValueError, match="missing_source_class"):
        group_weights([r for r in data if r["label"] == 0])


def test_failed_validation_gates_do_not_select_a_candidate():
    data = rows()
    baseline = quality(data, np.full(len(data), .4), .5)
    bad = copy.deepcopy(baseline)
    assert "aegis_validation_recall_below_floor" in rejection_codes(bad, baseline)
    assert "synthetic_validation_below_floor" in rejection_codes(bad, baseline)
    good = quality(data, [.1 if r["label"] == 0 else .9 for r in data], .5)
    assert rejection_codes(good, baseline) == []


def test_near_duplicate_groups_are_order_independent():
    one = " ".join(f"sample{i}" for i in range(80))
    two = one + " extra"
    three = "A different neutral sentence about a blue book"
    texts = {text_id(v): v for v in (one, two, three)}
    groups, _ = lexical_groups(texts)
    reverse, _ = lexical_groups(dict(reversed(list(texts.items()))))
    assert groups == reverse
    assert groups[text_id(one)] == groups[text_id(two)]
    assert groups[text_id(one)] != groups[text_id(three)]


def test_native_gate_does_not_accept_local_override(monkeypatch):
    monkeypatch.setenv("SAFETY_WATCH_ALLOW_LOCAL", "1")
    monkeypatch.setattr("src.data.aegis.platform.machine", lambda: "arm64")
    with pytest.raises(ValueError, match="accuracy_experiment_requires_native_s390x"):
        require_native()


def test_preparation_preserves_splits_and_excludes_conflicts_and_exposure(tmp_path, monkeypatch):
    import json
    import src.data.aegis as module
    from src.data.synthetic import file_hash
    raw = tmp_path / "raw"
    raw.mkdir()

    def row(text, label, identity):
        return {"prompt": text, "prompt_label": label, "prompt_label_source": "human",
                "reconstruction_id_if_redacted": None, "id": identity, "response": "unused fixture"}

    examples = {
        "train": [row("Training orange block", "safe", "t0"), row("Training purple cloud", "unsafe", "t1")],
        "validation": [row("Validation green book", "safe", "v0"), row("Validation yellow desk", "unsafe", "v1")],
        "test": [row("Final silver circle", "safe", "x0"), row("Final amber square", "unsafe", "x1")],
    }
    shared = "Shared plain neutral sentence"
    examples["train"] += [row(shared, "safe", "shared0"), row("Conflicting neutral sample", "safe", "c0"),
                          row("Previously exposed text", "safe", "p0"), row("REDACTED", "unsafe", "r0")]
    examples["validation"].append(row("Conflicting neutral sample", "unsafe", "c1"))
    examples["test"].append(row(shared, "safe", "shared1"))
    for split, data in examples.items():
        (raw / (split + ".json")).write_text(json.dumps(data))
    (raw / "README.md").write_text("Neutral source fixture")
    expected = {p.name: file_hash(p) for p in raw.iterdir()}
    (raw / "acquisition.json").write_text(json.dumps({"repository": module.REPO, "revision": module.REVISION, "files": expected}))
    monkeypatch.setattr(module, "EXPECTED_FILES", expected)
    monkeypatch.setattr(module, "require_native", lambda: None)
    result = module.prepare(raw, tmp_path / "prepared", blocked={text_id("Previously exposed text")},
                            prior_hashes={}, source_commit="a" * 40)
    assert result["counts"]["train"]["rows"] == 2
    assert result["counts"]["validation"]["rows"] == 2
    assert result["counts"]["test"]["rows"] == 3
    assert result["exclusions"]["conflicting_exact_prompt_rows"] == 2
    assert result["exclusions"]["train:prior_exposure_overlap"] == 1
    assert result["exclusions"]["train:redacted_or_missing"] == 1
    with pytest.raises(FileExistsError):
        module.prepare(raw, tmp_path / "prepared", blocked=set(), prior_hashes={}, source_commit="a" * 40)
    with (raw / "train.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="aegis_source_hash_mismatch"):
        module.prepare(raw, tmp_path / "mutated", blocked=set(), prior_hashes={}, source_commit="a" * 40)

"""Neutral profile provenance and finite-episode mechanics."""
import hashlib
import json

import pytest

from src.data import traffic_profile
from src.monitor.disagreement import MonitorError, ScorePair, calibrate
from src.monitor.profiles import ProfiledTrafficWatch, TrafficProfile
from src.response.recovery import identity_text

BINDING = "a" * 64


def config():
    refs = [ScorePair(f"ref-{i}", "reference", i, .4, .2, BINDING, "reference") for i in range(80)]
    null = [ScorePair(f"cal-{i}", "calibration", i, .4, .2, BINDING, "calibration") for i in range(128)]
    return calibrate(refs, null, window_size=8, stride=4, horizon=128, alpha=.1, null_runs=20)


def test_profile_rejects_unknown_source_without_state_change():
    profile = TrafficProfile("trusted-neutral", "b" * 40, "exact-input-v1", config())
    monitor = ProfiledTrafficWatch(profile)
    event = ScorePair("one", "stream", 0, .4, .2, BINDING, "evaluation")
    with pytest.raises(MonitorError, match="unqualified_traffic_profile"):
        monitor.update(event, source_id="untrusted-request-body", generation=0)
    assert monitor.episodes == 0
    assert monitor.update(event, source_id="trusted-neutral", generation=0)["accepted_count"] == 1


def test_profile_resets_generation_windows_and_enforces_total_budget():
    profile = TrafficProfile("trusted-neutral", "b" * 40, "exact-input-v1", config())
    monitor = ProfiledTrafficWatch(profile, episode_budget=2, warning_budget=1)
    outputs = []
    for i in range(24):
        outputs.append(monitor.update(ScorePair(f"event-{i}", "s" * 128, i, .1, .1, BINDING, "evaluation"),
            source_id="trusted-neutral", generation=i // 8))
    assert outputs[7]["new_alarm"] is not None
    assert outputs[8]["window_reset"] and outputs[8]["accepted_count"] == 1
    assert outputs[15]["warning_suppressed"] and outputs[15]["new_alarm"] is None
    assert outputs[16]["traffic_status"] == "episode_budget_exhausted"
    assert monitor.episodes == 2 and monitor.warnings == 1
    assert not outputs[15]["global_false_alarm_guarantee"]


def test_profile_binds_revision_preprocessing_and_calibration():
    profile = TrafficProfile("neutral", "b" * 40, "exact-v1", config())
    other = TrafficProfile("neutral", "b" * 40, "different-v1", profile.calibration)
    assert profile.profile_id != other.profile_id
    with pytest.raises(MonitorError, match="invalid_traffic_profile"):
        TrafficProfile("neutral", "not-a-revision", "exact-v1", profile.calibration)
    with pytest.raises(MonitorError, match="invalid_traffic_budget"):
        ProfiledTrafficWatch(profile, warning_budget=True)


def test_exact_source_groups_exclusions_and_role_separation(tmp_path, monkeypatch):
    rows = [{"instruction":f"Neutral instruction {i}", "context":"not concatenated", "response":"unused"} for i in range(30)]
    rows.append({"instruction":"  NEUTRAL   INSTRUCTION 0  "})
    raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path = tmp_path / "neutral.jsonl"
    path.write_bytes(raw)
    monkeypatch.setattr(traffic_profile, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    blocked = {identity_text("Neutral instruction 1")}
    parsed, manifest = traffic_profile.load_instructions(path, blocked)
    again, same = traffic_profile.load_instructions(path, blocked)
    assert parsed == again and manifest == same
    assert manifest["raw_rows"] == 31 and manifest["duplicate_rows"] == 1
    assert manifest["excluded_groups"] == 1 and len(parsed) == 29
    assert not blocked & {r.prompt_id for r in parsed}
    first = next(r for r in parsed if r.prompt_id == identity_text("Neutral instruction 0"))
    assert first.text == rows[0]["instruction"] and first.source_rows == (1, 31)
    ids = [{r.prompt_id for r in parsed if r.role == role} for role in ("reference", "calibration", "evaluation")]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i))
    assert manifest["safety_labels"] is None
    path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="traffic_source_hash_mismatch"):
        traffic_profile.load_instructions(path, blocked)

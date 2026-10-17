"""Neutral fixtures for incident mechanics, not effectiveness claims."""
from dataclasses import replace
import math

import pytest

from src.monitor.disagreement import MonitorError, ScorePair, calibrate
from src.monitor.runtime import HealthProbe, RuntimeCheckError, RuntimeWatch, checked_scores
from src.response.conformal import ConformalPolicy, identity
from src.response.recovery import Probe, RecoveryError, RecoveryManager, identity_text
from src.response.supervised import SupervisedRuntime

BINDING = "a" * 64
POLICY = ConformalPolicy(.3, .3, .05, 20, 20,
    tuple(identity(["initial", i]) for i in range(40)), BINDING, "original")


def healthy(text):
    return (.05 if "safe" in text.split() else .75) + int(text.split()[-1]) / 200


class Backend:
    model_pair_id = BINDING

    def __init__(self, mode="healthy"):
        self.mode = mode
        self.inputs = 0

    def score(self, texts):
        self.inputs += len(texts)
        rows = []
        for text in texts:
            g = c = healthy(text)
            if self.mode == "scale":
                g *= .25
            elif self.mode == "common":
                g *= .25
                c *= .25
            elif self.mode == "canary":
                c = .1
            elif self.mode == "constant":
                g = .1
            elif self.mode == "square":
                g *= g
            elif self.mode == "error":
                raise RuntimeError("neutral private failure")
            elif self.mode == "invalid":
                g = math.nan
            elif self.mode == "selective" and "live" in text:
                g = .01
            rows.append({"guardrail_score": g, "canary_score": c})
        return rows


def references(role):
    texts = [f"neutral {role} {label} example {i}" for label in ("safe", "harmful") for i in range(20)]
    probes = [Probe(identity_text(t), healthy(t), healthy(t), int("harmful" in t), role, healthy(t), healthy(t)) for t in texts]
    return probes, texts


def setup(mode="healthy", *, hold=True, approve=True, persistent=False, cadence=4, delay=2):
    fit, fit_texts = references("fit")
    verify, verify_texts = references("verification")
    panel = [HealthProbe(fit[i].prompt_id, fit_texts[i], fit[i].healthy_score, fit[i].healthy_canary_score)
             for i in (0, 19, 20, 39)]
    initial = Backend(mode)
    registry = [initial]
    def restart():
        new = Backend("constant" if persistent else "healthy")
        registry.append(new)
        return new
    manager = RecoveryManager(initial, POLICY)
    watch = RuntimeWatch(panel, BINDING, cadence=cadence, response_delay=delay)
    runtime = SupervisedRuntime(manager, watch, fit=fit, fit_texts=fit_texts,
        verification=verify, verification_texts=verify_texts, restart_factory=restart,
        approve_recovery=approve, approve_containment=hold, actor="unit-test")
    return runtime, registry


def stream(runtime, registry, n, changes=()):
    events = []
    for i in range(n):
        for at, mode in changes:
            if i == at:
                for backend in registry:
                    backend.mode = mode
        events.append(runtime.step(i, f"neutral live harmful example {i % 20}"))
    return events


def test_healthy_checks_continue_without_any_response():
    runtime, registry = setup()
    events = stream(runtime, registry, 64)
    assert runtime.watch.check_count == 16 and runtime.watch.probe_inputs == 64
    assert not runtime.watch.incidents and not runtime.held
    assert all(e["response"] is None and e["decision"] == "refuse" for e in events)
    assert events[0]["state"] == "unassessed"


def traffic_config(horizon=128):
    refs = [ScorePair(f"ref-{i}", "ref", i, .4, .2, BINDING, "reference") for i in range(80)]
    null = [ScorePair(f"cal-{i}", "cal", i, .4, .2, BINDING, "calibration") for i in range(horizon)]
    return calibrate(refs, null, window_size=8, stride=4, horizon=horizon, alpha=.1, null_runs=20)


def test_early_traffic_warning_does_not_consume_later_runtime_response():
    from src.monitor.disagreement import DisagreementMonitor
    runtime, registry = setup()
    passive = DisagreementMonitor(traffic_config())
    first_warning = None
    for i in range(64):
        if i == 40:
            registry[0].mode = "scale"
        event = runtime.step(i, f"neutral live harmful example {i % 20}")
        scores = event["scores"]
        update = passive.update(ScorePair(f"event-{i}", "stream", i,
            scores["guardrail_score"], scores["canary_score"], BINDING, "evaluation"))
        if update["new_alarm"]:
            first_warning = i
    assert first_warning == 7
    incidents = runtime.watch.incidents
    assert len(incidents) == 1
    assert incidents[0]["confirmed_index"] == 43
    assert incidents[0]["response_index"] == 45 and incidents[0]["status"] == "recovered"


def test_multiple_incidents_probe_the_active_backend():
    runtime, registry = setup()
    events = stream(runtime, registry, 48, [(8, "scale"), (24, "constant")])
    incidents = runtime.watch.incidents
    assert [i["confirmed_index"] for i in incidents] == [11, 27]
    assert [i["response_index"] for i in incidents] == [13, 29]
    assert all(i["status"] == "recovered" for i in incidents)
    assert runtime.manager.generation == 2
    assert all(e["decision"] == "refuse" for e in events[29:])


@pytest.mark.parametrize("mode", ["scale", "square", "common", "canary"])
def test_real_whole_pair_faults_recover(mode):
    runtime, registry = setup(mode)
    events = stream(runtime, registry, 16)
    assert len(runtime.watch.incidents) == 1
    assert runtime.watch.incidents[0]["status"] == "recovered"
    assert runtime.response_count == 1 and not runtime.held
    assert events[-1]["scores"]["guardrail_score"] == pytest.approx(healthy("neutral live harmful example 15"))
    assert events[-1]["scores"]["canary_score"] == pytest.approx(healthy("neutral live harmful example 15"))


def test_persistent_fault_holds_without_claiming_recovery():
    runtime, registry = setup("constant", persistent=True)
    events = stream(runtime, registry, 32)
    assert runtime.response_count == 1
    assert runtime.watch.incidents[0]["status"] == "unresolved"
    assert runtime.manager.generation == 0 and runtime.held
    assert all(e["decision"] == "hold" and not e["execute"] and e["scores"] is None for e in events[5:])
    assert runtime.watch.check_count == 8
    with pytest.raises(RecoveryError, match="hold_release_not_authorized"):
        runtime.release_hold()
    assert not runtime.release_hold(approved=True, actor="unit-test")["released"]


def test_containment_requires_separate_approval_and_no_fallthrough():
    runtime, registry = setup("constant", hold=False, persistent=True)
    events = stream(runtime, registry, 12)
    assert not runtime.held
    assert all(e["decision"] == "unavailable" and not e["execute"] for e in events[5:])
    assert not any(e["decision"] == "hold" for e in events)


@pytest.mark.parametrize("mode", ["error", "invalid"])
def test_invalid_output_immediately_prevents_execution(mode):
    runtime, registry = setup(mode)
    event = runtime.step(0, "neutral live harmful example 0")
    assert event["decision"] == "hold" and not event["execute"]
    assert "neutral private" not in str(event)
    assert runtime.watch.incidents[0]["confirmed_index"] == 0


def test_fault_after_activation_enters_hold_on_score_failure():
    runtime, registry = setup("scale")
    stream(runtime, registry, 8)
    assert runtime.manager.generation == 1
    registry[0].mode = "healthy"  # A stale x4 correction is now invalid on high scores.
    event = runtime.step(8, "neutral live harmful example 0")
    assert event["decision"] == "hold" and not event["execute"]
    assert len(runtime.watch.incidents) == 2


def test_hold_release_requires_fresh_full_pair_verification():
    runtime, registry = setup("constant", persistent=True)
    stream(runtime, registry, 8)
    registry[0].mode = "healthy"
    assert runtime.held
    receipt = runtime.release_hold(approved=True, actor="unit-test")
    assert receipt["released"] and receipt["verification"]["pair_verified"]
    assert not runtime.held


def test_selective_fault_outside_panel_remains_explicit_blind_spot():
    runtime, registry = setup("selective")
    events = stream(runtime, registry, 16)
    assert not runtime.watch.incidents
    assert all(e["execute"] for e in events)


def test_runtime_sequence_and_authority_validation():
    runtime, registry = setup()
    with pytest.raises(RuntimeCheckError, match="noncontiguous"):
        runtime.step(2, "neutral live harmful example 0")
    assert runtime.step(0, "neutral live harmful example 0")["decision"] == "refuse"
    with pytest.raises(RuntimeCheckError, match="invalid_runtime_schedule"):
        RuntimeWatch(runtime.watch.probes, BINDING, cadence=True)
    with pytest.raises(RuntimeCheckError, match="invalid_health_probe"):
        replace(runtime.watch.probes[0], role="evaluation")
    with pytest.raises(RuntimeCheckError, match="runtime_model_binding_mismatch"):
        checked_scores(Backend(), ["neutral live safe example 0"], "b" * 64)


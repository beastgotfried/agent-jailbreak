"""Neutral checks for the independent runtime incident state machine."""
from dataclasses import replace

import pytest

from src.monitor.runtime import HealthProbe, RuntimeCheckError, RuntimeWatch
from src.response.recovery import identity_text

BINDING = "a" * 64


class Backend:
    model_pair_id = BINDING
    gain = 1.0
    canary_gain = 1.0
    fail = False

    def score(self, texts):
        if self.fail:
            raise RuntimeError("neutral request text")
        return [{"guardrail_score":(.1 + int(t.split()[-1]) * .2) * self.gain,
                 "canary_score":(.1 + int(t.split()[-1]) * .2) * self.canary_gain} for t in texts]


def watch(**kwargs):
    probes = [HealthProbe(identity_text(t), t, .1 + i * .2, .1 + i * .2)
              for i in range(4) for t in [f"neutral probe {i}"]]
    return RuntimeWatch(probes, BINDING, **kwargs)


def test_independent_schedule_and_no_false_runtime_confirmation():
    runtime, backend = watch(), Backend()
    checks = [i for i in range(64) if runtime.tick(i, backend, 0)["checked"]]
    assert checks == [15, 31, 47, 63]
    assert not runtime.incidents and runtime.probe_inputs == 16


def test_two_incidents_after_successful_recovery():
    runtime, backend, generation = watch(), Backend(), 0
    for i in range(192):
        if i in (64, 128):
            backend.gain = .25
        result = runtime.tick(i, backend, generation)
        if result["response_due"]:
            runtime.begin_response(i)
            generation += 1
            backend.gain = 1.0
            runtime.finish_response({"applied":True,"status":"applied"}, generation)
    assert [r["confirmed_index"] for r in runtime.incidents] == [79, 143]
    assert [r["response_index"] for r in runtime.incidents] == [111, 175]
    assert all(r["status"] == "recovered" for r in runtime.incidents)


def test_healthy_again_does_not_consume_a_future_incident():
    runtime, backend = watch(), Backend()
    for i in range(160):
        if i in (64, 128):backend.gain = .25
        if i == 80:backend.gain = 1.0
        runtime.tick(i, backend, 0)
    assert [r["confirmed_index"] for r in runtime.incidents] == [79, 143]
    assert runtime.incidents[0]["status"] == "healthy_again"
    assert runtime.active["status"] == "confirmed"


def test_rejected_check_keeps_health_checks_but_does_not_retry_forever():
    runtime, backend = watch(response_delay=0), Backend()
    backend.gain = .25
    for i in range(64):
        result = runtime.tick(i, backend, 0)
        if result["response_due"]:
            runtime.begin_response(i)
            runtime.finish_response({"applied":False,"status":"recovery_rejected"}, 0)
    assert runtime.check_count == 4 and len(runtime.incidents) == 1
    assert runtime.active["status"] == "unresolved"


def test_canary_only_and_score_failure_confirm_without_private_text():
    runtime, backend = watch(cadence=1), Backend()
    backend.canary_gain = .25
    assert runtime.tick(0, backend, 0)["changed"]
    other, broken = watch(cadence=1), Backend()
    broken.fail = True
    result = other.tick(0, broken, 0)
    assert result["evidence"] == "runtime_output_unavailable"
    assert "neutral request" not in str(result)


def test_invalid_input_does_not_advance_sequence_or_accept_wrong_probe_role():
    runtime, backend = watch(), Backend()
    with pytest.raises(RuntimeCheckError, match="noncontiguous"):
        runtime.tick(1, backend, 0)
    runtime.tick(0, backend, 0)
    with pytest.raises(RuntimeCheckError, match="invalid_health_probe"):
        replace(runtime.probes[0], role="evaluation")
    with pytest.raises(RuntimeCheckError, match="invalid_runtime_schedule"):
        watch(cadence=True)
    assert runtime.state == "unassessed"

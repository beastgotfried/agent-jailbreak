"""Neutral whole-loop regression; real data effectiveness is tested separately."""
import pytest

from src.monitor.disagreement import ScorePair, calibrate
from src.replay.continuous import run_episode
from src.replay.recovery import score_records
from src.replay.recovery_data import RecoveryPrompt
from src.response.conformal import ConformalPolicy, identity
from src.response.recovery import identity_text

BINDING = "a" * 64
POLICY = ConformalPolicy(.3, .3, .05, 20, 20,
    tuple(identity(["initial", i]) for i in range(40)), BINDING, "original")


class Backend:
    model_pair_id = BINDING
    def score(self, texts):
        result = []
        for text in texts:
            value = (.05 if "safe" in text.split() else .75) + int(text.split()[-1]) / 200
            result.append({"guardrail_score":value,"canary_score":value})
        return result


def records(role, per_class):
    return [RecoveryPrompt(identity_text(text), text, int(label == "harmful"), "neutral", role)
        for label in ("safe", "harmful") for i in range(per_class)
        for text in [f"neutral {role} {label} unique {i} example {i % 20}"]]


@pytest.mark.parametrize("scenario,expected_incidents,restored,held", [
    ("stationary_safe",0,False,False), ("scale",1,True,False),
    ("persistent_restart_fault",1,False,True), ("late_scale",1,True,False),
    ("scale_then_constant",2,True,False),
])
def test_paired_full_loop_preserves_inputs_and_explicit_containment(scenario, expected_incidents, restored, held):
    fit, verification, evaluation = records("fit",20), records("verification",20), records("evaluation",150)
    healthy = score_records(Backend(), fit + verification + evaluation)
    refs = [ScorePair(f"ref-{i}","ref",i,.4,.2,BINDING,"reference") for i in range(80)]
    null = [ScorePair(f"null-{i}","cal",i,.4,.2,BINDING,"calibration") for i in range(512)]
    config = calibrate(refs,null,window_size=64,stride=16,horizon=512,alpha=.1,null_runs=20)
    events, summary = run_episode(scenario,0,factory=Backend,policy=POLICY,config=config,
        fit=fit,verification=verification,evaluation=evaluation,healthy=healthy,
        approve_recovery=True,approve_containment=True,actor="unit-test")
    assert len({e["prompt_id"] for e in events}) == len(events)
    assert len(summary["incidents"]) == expected_incidents
    assert summary["normal_service_restored"] is restored and summary["held"] is held
    assert summary["pre_fault_confirmations"] == 0
    assert all(d["delay"] == 15 for d in summary["causal_detection"])
    assert summary["health_probe_inputs"] == len(events) // 16 * 8
    assert all(not e["B_execute"] for e in events if e["B_decision"] in ("hold","unavailable"))
    if held:
        assert summary["B_backend"]["by_class"]["1"]["allowed"] == 150
        assert summary["B_final_service"]["by_class"]["1"]["held"] == 150
        assert summary["B_final_service"]["by_class"]["1"]["allowed"] == 0
    if restored:
        assert summary["B_backend"] == summary["baseline"]
    assert all("text" not in event for event in events)

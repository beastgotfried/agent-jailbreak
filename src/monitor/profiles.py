"""Explicit trusted-source traffic profiles; never select one from prompt text.

Source metadata must come from the trusted collector, not the request body.
Profile hashes bind contents, not source authenticity. Episode budgets do not
create an anytime false-alarm guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

from .disagreement import Calibration, DisagreementMonitor, MonitorError, ScorePair
from src.response.conformal import identity


@dataclass(frozen=True)
class TrafficProfile:
    source_id: str
    source_revision: str
    preprocessing_id: str
    calibration: Calibration

    def __post_init__(self):
        if (not isinstance(self.calibration, Calibration)
                or not isinstance(self.source_revision, str)
                or re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", self.source_revision) is None):
            raise MonitorError("invalid_traffic_profile")
        for value in (self.source_id, self.preprocessing_id):
            if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is None:
                raise MonitorError("invalid_traffic_profile")

    @property
    def profile_id(self):
        return identity({"source_id": self.source_id, "source_revision": self.source_revision,
                         "preprocessing_id": self.preprocessing_id, "calibration_id": self.calibration.calibration_id})

    def to_dict(self):
        return {"profile_id": self.profile_id, "source_id": self.source_id,
                "source_revision": self.source_revision, "preprocessing_id": self.preprocessing_id,
                "calibration": self.calibration.to_dict(), "scope": "declared_source_finite_episode_only"}


class ProfiledTrafficWatch:
    def __init__(self, profile, *, episode_budget=8, warning_budget=8):
        if (not isinstance(profile, TrafficProfile) or type(episode_budget) is not int or episode_budget < 1
                or type(warning_budget) is not int or warning_budget < 1):
            raise MonitorError("invalid_traffic_budget")
        self.profile = profile
        self.episode_budget, self.warning_budget = episode_budget, warning_budget
        self.episodes = self.warnings = 0
        self._monitor = None
        self._generation = None
        self._stream_id = None
        self._last_index = -1

    def update(self, event, *, source_id, generation):
        if source_id != self.profile.source_id:
            raise MonitorError("unqualified_traffic_profile")
        if (not isinstance(event, ScorePair) or event.model_pair_id != self.profile.calibration.model_pair_id
                or event.purpose == "reference"):
            raise MonitorError("traffic_event_binding_mismatch")
        if event.sequence != self._last_index + 1:
            raise MonitorError("noncontiguous_traffic_sequence")
        if type(generation) is not int or generation < 0 or (self._generation is not None and generation < self._generation):
            raise MonitorError("invalid_traffic_generation")
        if self._stream_id is not None and event.stream_id != self._stream_id:
            raise MonitorError("traffic_stream_changed")
        reset = self._monitor is None or generation != self._generation or self._monitor.exhausted
        if reset and self.episodes < self.episode_budget:
            self._monitor = DisagreementMonitor(self.profile.calibration)
            self.episodes += 1
        elif reset:
            self._last_index, self._stream_id, self._generation = event.sequence, event.stream_id, generation
            self._monitor = None
            return {"traffic_status": "episode_budget_exhausted", "new_alarm": None,
                    "profile_id": self.profile.profile_id, "global_sequence": event.sequence,
                    "harm_verified": False, "runtime_fault_confirmed": False}
        local = replace(event, sequence=self._monitor.accepted_count,
                        stream_id=identity([event.stream_id, self.episodes]))
        result = self._monitor.update(local)
        self._last_index, self._stream_id, self._generation = event.sequence, event.stream_id, generation
        suppressed = False
        if result["new_alarm"] is not None:
            if self.warnings >= self.warning_budget:
                suppressed = True
                result = {**result, "new_alarm": None}
            else:
                self.warnings += 1
        return {**result, "traffic_status": "warning_budget_exhausted" if suppressed else "qualified_profile_observation",
                "profile_id": self.profile.profile_id, "global_sequence": event.sequence, "generation": generation,
                "window_reset": reset, "episode_count": self.episodes, "warning_count": self.warnings,
                "warning_suppressed": suppressed, "harm_verified": False, "runtime_fault_confirmed": False,
                "global_false_alarm_guarantee": False}

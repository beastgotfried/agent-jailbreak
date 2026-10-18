"""Score-only, finite-horizon disagreement shift detector.

This module uses a KS statistic, not a KS p-value or a martingale.
The threshold is an empirical quantile of whole-episode null maxima.
Shared calibration observations make those episodes dependent. No formal
false-alarm, anytime-valid, harm-detection, or repair guarantee follows.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Real
import re
from typing import Iterable, Mapping

import numpy as np


class MonitorError(ValueError):
    """A fixed diagnostic code; never include rejected source values."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _hex(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identifier(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is not None


def _number(value, low, high) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value) and low <= value <= high


def strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise MonitorError("duplicate_json_key")
            result[key] = value
        return result

    def constant(_):
        raise MonitorError("nonfinite_json_value")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise MonitorError("invalid_json") from exc


@dataclass(frozen=True)
class ScorePair:
    event_id: str
    stream_id: str
    sequence: int
    guardrail_score: float
    canary_score: float
    model_pair_id: str
    purpose: str = "live"

    def __post_init__(self):
        if not _identifier(self.event_id) or not _identifier(self.stream_id):
            raise MonitorError("invalid_event_or_stream_id")
        if type(self.sequence) is not int or self.sequence < 0:
            raise MonitorError("invalid_sequence")
        if not _number(self.guardrail_score, 0, 1) or not _number(self.canary_score, 0, 1):
            raise MonitorError("invalid_score")
        if not _hex(self.model_pair_id):
            raise MonitorError("invalid_model_pair_id")
        if self.purpose not in ("reference", "calibration", "evaluation", "live"):
            raise MonitorError("invalid_purpose")

    @property
    def gap(self) -> float:
        return float(self.guardrail_score) - float(self.canary_score)

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, data: Mapping):
        names = {field.name for field in fields(cls)}
        if not isinstance(data, dict) or set(data) != names | {"schema_version"}:
            raise MonitorError("invalid_pair_fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise MonitorError("invalid_pair_schema")
        return cls(**{name: data[name] for name in names})


def pair_streams(guardrail: Iterable[dict], canary: Iterable[dict]):
    """Join ordered side streams only when their explicit identities agree.

    Mismatches fail. This function never guesses, buffers, or joins by position
    alone. An upstream collector must reorder out-of-order side streams first.
    """
    from itertools import zip_longest

    required = {"schema_version", "event_id", "stream_id", "sequence", "model_pair_id", "purpose", "model_role", "score"}
    for left, right in zip_longest(guardrail, canary):
        if left is None or right is None:
            raise MonitorError("unpaired_stream_end")
        for row, role in ((left, "guardrail"), (right, "canary")):
            if not isinstance(row, dict) or set(row) != required:
                raise MonitorError("invalid_side_fields")
            if type(row["schema_version"]) is not int or row["schema_version"] != 1 or row["model_role"] != role:
                raise MonitorError("invalid_side_binding")
        identity = ("event_id", "stream_id", "sequence", "model_pair_id", "purpose")
        if any(left[key] != right[key] for key in identity):
            raise MonitorError("side_identity_mismatch")
        yield ScorePair(**{key: left[key] for key in identity}, guardrail_score=left["score"], canary_score=right["score"])


def ks_statistic(reference, observed) -> float:
    """Two-sided empirical CDF distance, including tied values. No p-value."""
    try:
        first = np.asarray(reference, dtype=float)
        second = np.asarray(observed, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MonitorError("invalid_ks_samples") from exc
    if first.ndim != 1 or second.ndim != 1 or not len(first) or not len(second):
        raise MonitorError("invalid_ks_samples")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise MonitorError("invalid_ks_samples")
    first, second = np.sort(first), np.sort(second)
    knots = np.unique(np.concatenate((first, second)))
    first_cdf = np.searchsorted(first, knots, side="right") / len(first)
    second_cdf = np.searchsorted(second, knots, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


@dataclass(frozen=True)
class Calibration:
    model_pair_id: str
    reference_gaps: tuple[float, ...]
    threshold: float
    window_size: int
    stride: int
    horizon: int
    alpha: float
    null_runs: int
    seed: int
    calibration_size: int
    null_maxima: tuple[float, ...]
    reference_ids_digest: str
    calibration_ids_digest: str
    numpy_version: str

    def __post_init__(self):
        for name in ("window_size", "stride", "horizon", "null_runs", "seed", "calibration_size"):
            if type(getattr(self, name)) is not int:
                raise MonitorError("invalid_calibration_integer")
        if not 2 <= self.window_size <= self.horizon <= 100_000:
            raise MonitorError("invalid_window_or_horizon")
        if not 1 <= self.stride <= self.horizon or not 20 <= self.null_runs <= 10_000 or self.seed < 0:
            raise MonitorError("invalid_calibration_limits")
        if not _number(self.alpha, 0, 1) or not 0 < self.alpha < 1:
            raise MonitorError("invalid_alpha")
        if self.calibration_size < self.horizon or not self.window_size <= len(self.reference_gaps) <= 100_000:
            raise MonitorError("insufficient_calibration_samples")
        if len(self.null_maxima) != self.null_runs or not all(_number(x, 0, 1) for x in self.null_maxima):
            raise MonitorError("invalid_null_maxima")
        if not all(_number(x, -1, 1) for x in self.reference_gaps) or not _number(self.threshold, 0, 1):
            raise MonitorError("invalid_calibration_values")
        if not all(_hex(x) for x in (self.model_pair_id, self.reference_ids_digest, self.calibration_ids_digest)):
            raise MonitorError("invalid_calibration_binding")
        rank = math.ceil((self.null_runs + 1) * (1 - self.alpha))
        if rank > self.null_runs or self.threshold != sorted(self.null_maxima)[rank - 1]:
            raise MonitorError("invalid_calibration_quantile")
        if not isinstance(self.numpy_version, str) or not self.numpy_version:
            raise MonitorError("invalid_numpy_version")

    @property
    def max_looks(self):
        return 1 + (self.horizon - self.window_size) // self.stride

    def payload(self):
        return {"schema_version": 1, "method": "empirical_episode_maximum_ks", **asdict(self)}

    @property
    def calibration_id(self):
        return digest(self.payload())

    def to_dict(self):
        return {**self.payload(), "calibration_id": self.calibration_id}

    @classmethod
    def from_dict(cls, data):
        names = {field.name for field in fields(cls)}
        if not isinstance(data, dict) or set(data) != names | {"schema_version", "method", "calibration_id"}:
            raise MonitorError("invalid_calibration_fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["method"] != "empirical_episode_maximum_ks":
            raise MonitorError("invalid_calibration_schema")
        values = {name: data[name] for name in names}
        if not isinstance(values["reference_gaps"], list) or not isinstance(values["null_maxima"], list):
            raise MonitorError("invalid_calibration_arrays")
        values["reference_gaps"] = tuple(values["reference_gaps"])
        values["null_maxima"] = tuple(values["null_maxima"])
        result = cls(**values)
        if data["calibration_id"] != result.calibration_id:
            raise MonitorError("calibration_digest_mismatch")
        return result


def _validated_sample(events, purpose):
    values = list(events)
    if not values:
        raise MonitorError("empty_calibration_input")
    for index, event in enumerate(values):
        if not isinstance(event, ScorePair) or event.sequence != index or event.purpose != purpose:
            raise MonitorError("invalid_calibration_input")
    if len({e.event_id for e in values}) != len(values) or len({e.stream_id for e in values}) != 1:
        raise MonitorError("duplicate_or_mixed_calibration_input")
    return values


def calibrate(reference, calibration, *, window_size=64, stride=16, horizon=512, alpha=0.05, null_runs=200, seed=42):
    """Freeze an empirical threshold using reference/calibration purposes only.

    Each null episode samples calibration rows without replacement. Episodes
    reuse a common pool, so they are not independent production trials.
    """
    reference = _validated_sample(reference, "reference")
    calibration = _validated_sample(calibration, "calibration")
    if {e.event_id for e in reference} & {e.event_id for e in calibration}:
        raise MonitorError("reference_calibration_overlap")
    bindings = {e.model_pair_id for e in reference + calibration}
    if len(bindings) != 1:
        raise MonitorError("mixed_model_binding")
    integers = (window_size, stride, horizon, null_runs, seed)
    if any(type(x) is not int for x in integers):
        raise MonitorError("invalid_calibration_integer")
    if not 2 <= window_size <= horizon <= 100_000 or not 1 <= stride <= horizon or not 20 <= null_runs <= 10_000 or seed < 0:
        raise MonitorError("invalid_calibration_limits")
    if not _number(alpha, 0, 1) or not 0 < alpha < 1:
        raise MonitorError("invalid_alpha")
    rank = math.ceil((null_runs + 1) * (1 - alpha))
    if rank > null_runs or len(reference) < window_size or len(calibration) < horizon:
        raise MonitorError("insufficient_calibration_samples")
    gaps = np.asarray([e.gap for e in calibration])
    baseline = tuple(sorted(e.gap for e in reference))
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(null_runs):
        episode = gaps[rng.choice(len(gaps), size=horizon, replace=False)]
        maxima.append(max(ks_statistic(baseline, episode[end - window_size:end]) for end in range(window_size, horizon + 1, stride)))
    return Calibration(
        model_pair_id=next(iter(bindings)), reference_gaps=baseline,
        threshold=sorted(maxima)[rank - 1], window_size=window_size,
        stride=stride, horizon=horizon, alpha=float(alpha), null_runs=null_runs,
        seed=seed, calibration_size=len(calibration), null_maxima=tuple(maxima),
        reference_ids_digest=digest(sorted(e.event_id for e in reference)),
        calibration_ids_digest=digest(sorted(e.event_id for e in calibration)),
        numpy_version=np.__version__,
    )


class DisagreementMonitor:
    """One stream, one finite episode, and at most one change alarm.

    State size is bounded by the configured horizon. Invalid telemetry does
    not change state. A caller must supply the remaining events after restore.
    """
    def __init__(self, calibration: Calibration):
        self.calibration = calibration
        self._window = deque(maxlen=calibration.window_size)
        self._events = []
        self._seen = set()
        self._stream_id = None
        self._alarm = None
        self._looks = 0

    @property
    def accepted_count(self):
        return len(self._events)

    @property
    def alarm(self):
        return self._alarm.copy() if self._alarm is not None else None

    @property
    def exhausted(self):
        return self.accepted_count == self.calibration.horizon

    def update(self, event: ScorePair):
        if not isinstance(event, ScorePair):
            raise MonitorError("expected_score_pair")
        if self.exhausted:
            raise MonitorError("horizon_exhausted")
        if event.model_pair_id != self.calibration.model_pair_id:
            raise MonitorError("model_binding_mismatch")
        if event.sequence != self.accepted_count:
            raise MonitorError("noncontiguous_sequence")
        if event.event_id in self._seen:
            raise MonitorError("duplicate_event")
        if self._stream_id is not None and event.stream_id != self._stream_id:
            raise MonitorError("stream_identity_changed")
        if event.purpose == "reference":
            raise MonitorError("reference_is_not_live_telemetry")
        self._stream_id = event.stream_id
        self._seen.add(event.event_id)
        self._events.append(event)
        self._window.append(event.gap)
        count = self.accepted_count
        looked = count >= self.calibration.window_size and (count - self.calibration.window_size) % self.calibration.stride == 0
        statistic = None
        new_alarm = None
        if looked:
            self._looks += 1
            statistic = ks_statistic(self.calibration.reference_gaps, self._window)
            if statistic > self.calibration.threshold and self._alarm is None:
                self._alarm = {
                    "kind": "disagreement_shift", "evidence_status": "change_signal_only",
                    "alarm_id": digest([self.calibration.calibration_id, event.stream_id, event.sequence]),
                    "event_id": event.event_id, "sequence": event.sequence,
                    "stream_id": event.stream_id, "model_pair_id": event.model_pair_id,
                    "calibration_id": self.calibration.calibration_id,
                    "statistic": statistic, "threshold": self.calibration.threshold,
                    "window_start_sequence": count - self.calibration.window_size,
                    "window_end_sequence": event.sequence, "look_index": self._looks,
                    "mean_signed_gap": math.fsum(self._window) / len(self._window),
                    "mean_absolute_gap": math.fsum(abs(x) for x in self._window) / len(self._window),
                    "action_authority": "none",
                }
                new_alarm = self.alarm
        return {
            "kind": "monitor_update", "stream_id": event.stream_id,
            "event_id": event.event_id, "sequence": event.sequence,
            "accepted_count": count, "window_ready": count >= self.calibration.window_size,
            "look_index": self._looks, "statistic": statistic,
            "threshold": self.calibration.threshold, "new_alarm": new_alarm,
            "alarm_latched": self._alarm is not None, "horizon_exhausted": self.exhausted,
            "validity": "empirical_finite_episode_only",
        }

    def checkpoint(self):
        payload = {"schema_version": 1, "calibration_id": self.calibration.calibration_id, "events": [e.to_dict() for e in self._events]}
        return {**payload, "checkpoint_id": digest(payload)}

    @classmethod
    def restore(cls, calibration, checkpoint):
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"schema_version", "calibration_id", "events", "checkpoint_id"}:
            raise MonitorError("invalid_checkpoint_fields")
        if type(checkpoint["schema_version"]) is not int or checkpoint["schema_version"] != 1:
            raise MonitorError("invalid_checkpoint_schema")
        payload = {key: value for key, value in checkpoint.items() if key != "checkpoint_id"}
        if checkpoint["calibration_id"] != calibration.calibration_id or checkpoint["checkpoint_id"] != digest(payload):
            raise MonitorError("checkpoint_binding_mismatch")
        if not isinstance(checkpoint["events"], list) or len(checkpoint["events"]) > calibration.horizon:
            raise MonitorError("invalid_checkpoint_size")
        result = cls(calibration)
        for row in checkpoint["events"]:
            result.update(ScorePair.from_dict(row))
        return result

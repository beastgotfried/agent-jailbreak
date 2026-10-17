"""Deterministic finite-probe runtime checks, independent of traffic warnings.

A matching panel is not a guarantee for other inputs or later requests. This
module has no repair, containment, model-fit, or approval authority.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from src.response.conformal import valid_hash, valid_score
from src.response.recovery import identity_text


class RuntimeCheckError(ValueError):
    pass


def checked_scores(backend, texts, model_pair_id):
    if getattr(backend, "model_pair_id", None) != model_pair_id:
        raise RuntimeCheckError("runtime_model_binding_mismatch")
    values = backend.score(list(texts))
    if not isinstance(values, (list, tuple)) or len(values) != len(texts):
        raise RuntimeCheckError("runtime_batch_length_mismatch")
    result = []
    for row in values:
        if not isinstance(row, dict) or not all(valid_score(row.get(k)) for k in ("guardrail_score", "canary_score")):
            raise RuntimeCheckError("invalid_runtime_pair")
        for model in ("guardrail", "canary"):
            key = model + "_unsafe"
            if key in row:
                threshold = getattr(backend, model + "_threshold", None)
                if (type(row[key]) is not bool or not valid_score(threshold)
                        or row[key] != (row[model + "_score"] >= threshold)):
                    raise RuntimeCheckError("invalid_runtime_flag")
        result.append(dict(row))
    return result


@dataclass(frozen=True)
class HealthProbe:
    prompt_id: str
    text: str = field(repr=False)
    guardrail_score: float
    canary_score: float
    role: str = "fit"

    def __post_init__(self):
        if (not isinstance(self.text, str) or not self.text.strip()
                or self.prompt_id != identity_text(self.text) or self.role != "fit"
                or not all(valid_score(s) for s in (self.guardrail_score, self.canary_score))):
            raise RuntimeCheckError("invalid_health_probe")


class RuntimeWatch:
    """One response opportunity per incident, not one per process lifetime."""

    def __init__(self, probes, model_pair_id, *, cadence=16, response_delay=32,
                 tolerance=1e-8, max_incidents=128):
        probes = tuple(probes)
        if (not probes or any(not isinstance(p, HealthProbe) for p in probes)
                or len({p.prompt_id for p in probes}) != len(probes) or not valid_hash(model_pair_id)):
            raise RuntimeCheckError("invalid_health_panel")
        if (type(cadence) is not int or cadence < 1 or type(response_delay) is not int or response_delay < 0
                or type(max_incidents) is not int or max_incidents < 1
                or not valid_score(tolerance) or tolerance > 1e-8):
            raise RuntimeCheckError("invalid_runtime_schedule")
        self.probes, self.model_pair_id = probes, model_pair_id
        self.cadence, self.response_delay, self.tolerance = cadence, response_delay, tolerance
        self.max_incidents = max_incidents
        self._last_index, self._generation = -1, 0
        self._active = None
        self._last_health = "unassessed"
        self._incidents = []
        self.check_count = 0
        self.probe_inputs = 0

    @property
    def incidents(self):
        return deepcopy(self._incidents)

    @property
    def active(self):
        return deepcopy(self._active)

    @property
    def state(self):
        return self._active["status"] if self._active else self._last_health

    def _open(self, index, generation, kind):
        if self._active is not None:
            return False
        if len(self._incidents) >= self.max_incidents:
            raise RuntimeCheckError("runtime_incident_budget_exhausted")
        incident = {"incident_id": len(self._incidents) + 1, "confirmed_index": index,
                    "generation": generation, "evidence": kind, "status": "confirmed",
                    "due_index": index + self.response_delay, "attempted": False}
        self._incidents.append(incident)
        self._active = incident
        return True

    def tick(self, index, backend, generation):
        if type(index) is not int or index != self._last_index + 1:
            raise RuntimeCheckError("noncontiguous_runtime_sequence")
        if type(generation) is not int or generation < self._generation:
            raise RuntimeCheckError("invalid_runtime_generation")
        self._last_index, self._generation = index, generation
        record = {"checked": False, "index": index, "generation": generation,
                  "incident_opened": False, "response_due": False}
        if (index + 1) % self.cadence == 0:
            self.check_count += 1
            self.probe_inputs += len(self.probes)
            record.update(checked=True, probe_inputs=len(self.probes))
            try:
                values = checked_scores(backend, [p.text for p in self.probes], self.model_pair_id)
                g_error = max(abs(v["guardrail_score"] - p.guardrail_score) for p, v in zip(self.probes, values))
                c_error = max(abs(v["canary_score"] - p.canary_score) for p, v in zip(self.probes, values))
                changed = max(g_error, c_error) > self.tolerance
                record.update(changed=changed, guardrail_error=g_error, canary_error=c_error,
                              evidence="matched_probe_change" if changed else "matched_probes")
            except Exception:
                changed = True
                record.update(changed=True, guardrail_error=None, canary_error=None, evidence="runtime_output_unavailable")
            self._last_health = "probe_change" if changed else "matched_on_last_check"
            if changed:
                record["incident_opened"] = self._open(index, generation, record["evidence"])
            elif self._active is not None:
                self._active.update(status="healthy_again", closed_index=index,
                                    closure="matched_panel_not_universal_recovery")
                self._active = None
        record["response_due"] = bool(self._active and not self._active["attempted"]
                                      and index >= self._active["due_index"])
        record["state"] = self.state
        return record

    def runtime_error(self, index, generation):
        if index != self._last_index:
            raise RuntimeCheckError("runtime_error_sequence_mismatch")
        self._open(index, generation, "runtime_output_unavailable")

    def begin_response(self, index):
        if (self._active is None or self._active["attempted"]
                or index != self._last_index or index < self._active["due_index"]):
            raise RuntimeCheckError("response_not_due")
        self._active.update(attempted=True, status="response_pending", response_index=index)
        return self._active["incident_id"]

    def finish_response(self, receipt, generation):
        if self._active is None or self._active["status"] != "response_pending":
            raise RuntimeCheckError("response_not_pending")
        applied = receipt.get("applied") is True
        healthy = receipt.get("status") == "no_probe_degradation"
        if (type(generation) is not int or generation < self._generation
                or (applied and generation <= self._active["generation"])):
            raise RuntimeCheckError("recovery_generation_not_advanced")
        status = "recovered" if applied else "healthy_again" if healthy else "unresolved"
        self._active.update(status=status, response_status=receipt.get("status"), final_generation=generation)
        if applied or healthy:
            self._active["closed_index"] = self._last_index
            self._active = None
            self._last_health = "recovered_pending_check" if applied else "matched_full_verification"
        self._generation = generation

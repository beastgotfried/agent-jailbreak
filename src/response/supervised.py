"""Offline request supervision with explicit containment authority.

A hold prevents execution. It is neither a classifier decision nor recovery.
No production authentication or persistent multi-process service is supplied.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import re
import threading

from src.monitor.runtime import RuntimeCheckError, checked_scores
from .recovery import RecoveryError, identity_text, validate_batch, verify_outputs


class SupervisedRuntime:
    def __init__(self, manager, watch, *, fit, fit_texts, verification, verification_texts,
                 restart_factory, approve_recovery=False, approve_containment=False,
                 actor="", phase=None):
        if type(approve_recovery) is not bool or type(approve_containment) is not bool:
            raise RecoveryError("invalid_supervisor_authority")
        if ((approve_recovery or approve_containment)
                and (not isinstance(actor, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", actor) is None)):
            raise RecoveryError("invalid_supervisor_actor")
        self.fit = validate_batch(fit, "fit")
        self.verification = validate_batch(verification, "verification")
        self.fit_texts, self.verification_texts = tuple(fit_texts), tuple(verification_texts)
        for probes, texts in ((self.fit, self.fit_texts), (self.verification, self.verification_texts)):
            if len(probes) != len(texts) or any(not isinstance(t, str) or identity_text(t) != p.prompt_id
                                              for p, t in zip(probes, texts)):
                raise RecoveryError("supervisor_text_identity_mismatch")
        fit_ids, verify_ids = {p.prompt_id for p in self.fit}, {p.prompt_id for p in self.verification}
        if fit_ids & verify_ids or (fit_ids | verify_ids) & set(manager.policy.calibration_ids):
            raise RecoveryError("supervisor_probe_role_overlap")
        if not {p.prompt_id for p in watch.probes} <= fit_ids or watch.model_pair_id != manager.policy.model_pair_id:
            raise RecoveryError("supervisor_panel_binding_mismatch")
        fit_by_id = {p.prompt_id: p for p in self.fit}
        if any(p.guardrail_score != fit_by_id[p.prompt_id].healthy_score
               or p.canary_score != fit_by_id[p.prompt_id].healthy_canary_score for p in watch.probes):
            raise RecoveryError("supervisor_panel_baseline_mismatch")
        self.manager, self.watch = manager, watch
        self.restart_factory = restart_factory
        self.approve_recovery, self.approve_containment, self.actor = approve_recovery, approve_containment, actor
        self._phase = phase or (lambda name: nullcontext())
        self._lock = threading.RLock()
        self._held, self._hold_reason = False, None
        self._unavailable = False
        self.response_count = 0
        self.hold_count = 0

    @property
    def held(self):
        return self._held

    @property
    def unavailable(self):
        return self._unavailable

    def _contain(self, reason):
        if self.approve_containment:
            if not self._held:
                self.hold_count += 1
            self._held, self._hold_reason = True, reason
        else:
            self._unavailable = True

    def _sample(self, reference, texts):
        values = checked_scores(self.manager.backend, texts, self.manager.policy.model_pair_id)
        return tuple(replace(p, current_score=v["guardrail_score"], current_canary_score=v["canary_score"])
                     for p, v in zip(reference, values))

    def _respond(self, index):
        incident_id = self.watch.begin_response(index)
        self.response_count += 1
        try:
            with self._phase("response_fit"):
                fit = self._sample(self.fit, self.fit_texts)
                verification = self._sample(self.verification, self.verification_texts)
            with self._phase("response_proposal"):
                plan = self.manager.propose(fit, verification, verification_prompts=self.verification_texts,
                    restart_factory=self.restart_factory, approved=self.approve_recovery, actor=self.actor)
            with self._phase("response_activation"):
                receipt = self.manager.apply(plan)
        except Exception:
            receipt = {"applied": False, "status": "recovery_rejected", "code": "response_unavailable",
                       "generation": self.manager.generation}
        self.watch.finish_response(receipt, self.manager.generation)
        if not receipt["applied"] and receipt["status"] != "no_probe_degradation":
            self._contain("unresolved_runtime_incident")
        return {**receipt, "incident_id": incident_id, "response_index": index}

    def step(self, index, text):
        """Return a gate decision. The caller must execute only an allow."""
        if not isinstance(text, str) or not text.strip():
            raise RecoveryError("invalid_supervised_request")
        with self._lock:
            response = None
            try:
                with self._phase("health"):
                    health = self.watch.tick(index, self.manager.backend, self.manager.generation)
            except RuntimeCheckError:
                # Invalid sequencing is a caller error, not a new runtime incident.
                raise
            if health["checked"] and health.get("evidence") == "runtime_output_unavailable":
                self._contain("invalid_runtime_output")
            if health["response_due"]:
                response = self._respond(index)
            result = {"index": index, "generation": self.manager.generation, "health": health,
                      "response": response, "state": self.watch.state, "scores": None,
                      "execute": False, "held": self._held}
            if self._held or self._unavailable:
                return {**result, "decision": "hold" if self._held else "unavailable",
                        "decision_source": "containment" if self._held else "runtime_unavailable",
                        "hold_reason": self._hold_reason, "incident_resolved": False}
            try:
                with self._phase("live"):
                    row = checked_scores(self.manager.backend, [text], self.manager.policy.model_pair_id)[0]
            except Exception:
                self.watch.runtime_error(index, self.manager.generation)
                self._contain("invalid_runtime_output")
                return {**result, "decision": "hold" if self._held else "unavailable",
                        "held": self._held, "state": self.watch.state,
                        "decision_source": "containment" if self._held else "runtime_unavailable",
                        "hold_reason": self._hold_reason, "incident_resolved": False}
            decision = self.manager.policy.decision(row["guardrail_score"])
            return {**result, "scores": row, "decision": decision, "execute": decision == "allow",
                    "decision_source": "frozen_policy", "incident_resolved": self.watch.active is None}

    def release_hold(self, *, approved=False, actor=""):
        """Explicit release requires a new full verification, not a small panel."""
        if approved is not True or not isinstance(actor, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", actor) is None:
            raise RecoveryError("hold_release_not_authorized")
        with self._lock:
            if not self._held and not self._unavailable:
                return {"status": "not_held", "released": False}
            try:
                with self._phase("hold_release"):
                    values = checked_scores(self.manager.backend, self.verification_texts, self.manager.policy.model_pair_id)
                check = verify_outputs(self.manager.policy, self.verification,
                    [v["guardrail_score"] for v in values], canary_outputs=[v["canary_score"] for v in values])
            except Exception:
                return {"status": "release_rejected", "released": False, "code": "verification_unavailable"}
            if not check["accepted"]:
                return {"status": "release_rejected", "released": False, "verification": check}
            self._held, self._unavailable, self._hold_reason = False, False, None
            return {"status": "released", "released": True, "actor": actor, "verification": check}

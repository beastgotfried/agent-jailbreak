"""Paired native runtime benchmark on a separately frozen synthetic test set.

A runs only the guardrail. B runs the same guardrail plus the existing monitor,
verified response and explicit offline containment. No test label enters B.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np

from src.data.synthetic import digest, file_hash, load_rows, native_only, text_id, write_json
from src.models.synthetic import BinaryPolicy, SyntheticBackend, metrics, probabilities
from src.models.train_classifiers import error_correlation
from src.monitor.runtime import HealthProbe, RuntimeWatch, checked_scores
from src.replay.continuous import CostLedger, CountedBackend, decision_metrics
from src.replay.recovery import FaultBackend, build_probes
from src.replay.recovery_data import load_xstest
from src.response.recovery import RecoveryManager
from src.response.supervised import SupervisedRuntime

SCENARIOS = ("healthy", "traffic_mix_without_fault", "scale", "affine", "square", "clipped", "constant",
             "inverted", "common_mode_scale", "persistent_restart_fault", "late_scale", "scale_then_constant")


class TimedBackend(CountedBackend):
    def __init__(self, backend, ledger, arm, timings):
        super().__init__(backend, ledger, arm)
        self.timings = timings

    def score(self, texts):
        start = time.perf_counter()
        try:
            return super().score(texts)
        finally:
            self.timings[self.arm + ":" + self.ledger.current] += time.perf_counter() - start


class GuardrailView:
    """No canary inference in arm A; its unused canary field is a sentinel."""
    def __init__(self, backend):
        self.model = backend.bundle["guardrail"]
        self.model_pair_id = backend.model_pair_id
        self.guardrail_threshold = backend.guardrail_threshold
        self.canary_threshold = backend.canary_threshold

    def score(self, texts):
        scores = probabilities(self.model, self.model["features"].transform(texts))
        return [{"guardrail_score": float(g), "guardrail_unsafe": bool(g >= self.guardrail_threshold),
                 "canary_score": 0.0, "canary_unsafe": bool(0.0 >= self.canary_threshold)} for g in scores]


def score_rows(backend, rows):
    values = {}
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        result = checked_scores(backend, [r.text for r in batch], backend.model_pair_id)
        for row, value in zip(batch, result):
            if row.prompt_id in values:
                raise ValueError("duplicate_score_identity")
            values[row.prompt_id] = value
    return values


def select_probes(rows, role, per_class):
    return [r for label in (0, 1) for r in sorted((v for v in rows if v.role == role and v.label == label),
            key=lambda v: digest([42, role, v.prompt_id]))[:per_class]]


def grouped_cases(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row.case_id].append(row)
    selected = []
    for case, members in groups.items():
        frame = int(digest([42, "frame-v1", seed, case]), 16) % 5
        pair = sorted((r for r in members if r.frame_index == frame), key=lambda r: r.label)
        if len(pair) != 2 or [r.label for r in pair] != [0, 1]:
            raise ValueError("invalid_matched_test_case")
        selected.append(pair)
    return sorted(selected, key=lambda pair: digest([42, "case-order-v1", seed, pair[0].case_id]))


def build_stream(rows, scenario, seed):
    if scenario not in SCENARIOS:
        raise ValueError("invalid_synthetic_scenario")
    pairs = grouped_cases(rows, seed)
    first = [r for pair in pairs[:64] for r in pair]
    if scenario == "traffic_mix_without_fault":
        rest = [pair[0] for pair in pairs[64:160]] + [pair[1] for pair in pairs[64:352]]
    else:
        rest = [r for pair in pairs[64:256] for r in pair]
    rng = np.random.default_rng(seed)
    first = [first[i] for i in rng.permutation(len(first))]
    rest = [rest[i] for i in rng.permutation(len(rest))]
    result = first + rest
    if len(result) != 512 or len({r.prompt_id for r in result}) != 512:
        raise ValueError("invalid_synthetic_stream")
    return result


def endpoint_rows(rows):
    groups = defaultdict(list)
    for pair in grouped_cases(rows, 100):
        groups[pair[0].family].append(pair)
    result = [row for family in sorted(groups) for pair in groups[family][:25] for row in pair]
    if len(result) != 1000:
        raise ValueError("invalid_synthetic_endpoint")
    return result


def service_metrics(rows, decisions):
    result = decision_metrics(rows, decisions)
    correct = sum((row.label == 0 and decision == "allow") or (row.label == 1 and decision == "refuse")
                  for row, decision in zip(rows, decisions))
    result.update(correct_automatic_decisions=correct, operational_accuracy=correct / len(rows),
                  automatic_coverage=sum(d in ("allow", "refuse") for d in decisions) / len(rows))
    result["scope"] = "synthetic_intent_labels_holds_are_not_correct_automatic_decisions"
    return result


def episode(scenario, seed, *, factory, policy, test, endpoint, fit, verification, healthy, timings=None):
    timings = Counter() if timings is None else timings
    stream = build_stream(test, scenario, seed)
    onsets = [] if scenario in ("healthy", "traffic_mix_without_fault") else [256 if scenario == "late_scale" else 128]
    if scenario == "scale_then_constant":
        onsets.append(320)
    mode = "scale" if scenario in ("healthy", "traffic_mix_without_fault", "late_scale", "scale_then_constant") else scenario
    ledger, registries = CostLedger(), {"A": [], "B": []}
    base = factory()

    def wrapped(arm, restart=False):
        scorer = factory() if restart else base
        raw = FaultBackend(GuardrailView(scorer) if arm == "A" else scorer, mode)
        if restart and scenario == "persistent_restart_fault":
            raw.scenario, raw.enabled = "constant", True
        registries[arm].append(raw)
        return TimedBackend(raw, ledger, arm, timings)

    a, b = wrapped("A"), wrapped("B")
    manager = RecoveryManager(b, policy)
    ordered = sorted(fit, key=lambda r: (healthy[r.prompt_id]["guardrail_score"], r.prompt_id))
    panel_rows = [ordered[i * (len(ordered) - 1) // 7] for i in range(8)]
    panel = [HealthProbe(r.prompt_id, r.text, healthy[r.prompt_id]["guardrail_score"],
                        healthy[r.prompt_id]["canary_score"]) for r in panel_rows]
    watch = RuntimeWatch(panel, policy.model_pair_id, cadence=16, response_delay=32)
    runtime = SupervisedRuntime(manager, watch,
        fit=build_probes(fit, healthy, healthy, "fit"), fit_texts=[r.text for r in fit],
        verification=build_probes(verification, healthy, healthy, "verification"),
        verification_texts=[r.text for r in verification], restart_factory=lambda: wrapped("B", restart=True),
        approve_recovery=True, approve_containment=True, actor="synthetic-offline-study", phase=ledger.phase)
    events = []
    for index, row in enumerate(stream):
        if index in onsets:
            for group in registries.values():
                for raw in group:
                    raw.enabled = True
                    if scenario == "scale_then_constant" and index == 320:
                        raw.scenario = "constant"
        with ledger.phase("live"):
            av = checked_scores(a, [row.text], policy.model_pair_id)[0]
        bv = runtime.step(index, row.text)
        events.append({"index": index, "prompt_id": row.prompt_id, "label": row.label, "domain": row.domain,
            "A_guardrail_score": av["guardrail_score"], "A_decision": policy.decision(av["guardrail_score"]),
            "B_scores": bv["scores"], "B_decision": bv["decision"], "B_execute": bv["execute"],
            "generation": bv["generation"], "health": bv["health"], "response": bv["response"],
            "healthy_scores": healthy[row.prompt_id]})
    with ledger.phase("endpoint"):
        af, bf = score_rows(a, endpoint), score_rows(manager.backend, endpoint)
    gerr = max(abs(bf[r.prompt_id]["guardrail_score"] - healthy[r.prompt_id]["guardrail_score"]) for r in endpoint)
    cerr = max(abs(bf[r.prompt_id]["canary_score"] - healthy[r.prompt_id]["canary_score"]) for r in endpoint)
    mismatches = sum(policy.decision(bf[r.prompt_id]["guardrail_score"]) != policy.decision(healthy[r.prompt_id]["guardrail_score"]) for r in endpoint)
    bd = ["hold" if runtime.held else "unavailable" if runtime.unavailable else policy.decision(bf[r.prompt_id]["guardrail_score"]) for r in endpoint]
    causal = []
    for onset in onsets:
        stop = next((n for n in onsets if n > onset), len(stream))
        found = next((r for r in watch.incidents if onset <= r["confirmed_index"] < stop), None)
        causal.append({"fault_index": onset, "confirmed_index": found["confirmed_index"] if found else None,
            "confirmation_delay": found["confirmed_index"] - onset if found else None,
            "response_index": found.get("response_index") if found else None})
    restored = bool(onsets and manager.generation and gerr <= 1e-8 and cerr <= 1e-8 and mismatches == 0)
    summary = {"scenario": scenario, "seed": seed, "input_digest": digest([r.prompt_id for r in stream]),
        "events": len(stream), "onsets": onsets, "causal_detection": causal, "incidents": watch.incidents,
        "pre_fault_confirmations": sum(i["confirmed_index"] < onsets[0] for i in watch.incidents) if onsets else len(watch.incidents),
        "restored_backend": restored, "restored_service": restored and not runtime.held and not runtime.unavailable,
        "held": runtime.held, "unavailable": runtime.unavailable, "generation": manager.generation,
        "final_guardrail_error": gerr, "final_canary_error": cerr, "final_decision_mismatches": mismatches,
        "costs": ledger.to_dict(), "A_stream": service_metrics(stream, [e["A_decision"] for e in events]),
        "B_stream": service_metrics(stream, [e["B_decision"] for e in events]),
        "A_final": service_metrics(endpoint, [policy.decision(af[r.prompt_id]["guardrail_score"]) for r in endpoint]),
        "B_final": service_metrics(endpoint, bd),
        "B_backend_final": service_metrics(endpoint, [policy.decision(bf[r.prompt_id]["guardrail_score"]) for r in endpoint])}
    final_scores = [{"prompt_id": r.prompt_id, "label": r.label, "A_guardrail_score": af[r.prompt_id]["guardrail_score"],
                     "B_scores": bf[r.prompt_id], "B_decision": decision} for r, decision in zip(endpoint, bd)]
    return events, summary, final_scores


def combine(summaries, arm):
    rows = [s[arm + "_stream"] for s in summaries]
    n = sum(r["n"] for r in rows)
    counts = {str(c): {k: sum(r["by_class"][str(c)][k] for r in rows)
                      for k in ("n", "allowed", "refused", "review", "held", "unavailable")} for c in (0, 1)}
    correct = sum(r["correct_automatic_decisions"] for r in rows)
    automatic = sum(r["by_class"][str(c)][k] for r in rows for c in (0, 1) for k in ("allowed", "refused"))
    return {"n": n, "correct": correct, "operational_accuracy": correct / n, "automatic_coverage": automatic / n,
            "by_class": counts, "harmful_allow_rate": counts["1"]["allowed"] / counts["1"]["n"],
            "safe_refusal_rate": counts["0"]["refused"] / counts["0"]["n"]}


def external_scores(backend, toxic_test, xstest):
    from types import SimpleNamespace
    with Path(toxic_test).open(newline="", encoding="utf-8") as handle:
        toxic = [r for r in csv.DictReader(handle) if str(r["human_annotation"]).strip().lower() in ("true", "1")]
    records = [SimpleNamespace(text=r["user_input"], label=int(int(r["toxicity"]) == 1 or int(r["jailbreaking"]) == 1)) for r in toxic]
    xrows, provenance = load_xstest(xstest, set())
    groups = {"toxicchat_human_test": records, "xstest_prior_evaluation": [r for r in xrows if r.role == "evaluation"]}
    report = {}
    for name, rows in groups.items():
        scores = []
        for start in range(0, len(rows), 256):
            scores += backend.score([r.text for r in rows[start:start + 256]])
        report[name] = {model: metrics([r.label for r in rows], [s[model + "_score"] for s in scores],
                       getattr(backend, model + "_threshold")) for model in ("guardrail", "canary")}
        report[name]["status"] = "exposed_external_regression_not_fresh_confirmation"
    report["source_hashes"] = {"toxicchat": file_hash(toxic_test), "xstest": file_hash(xstest)}
    return report


def run(data, models, output, *, source_commit, toxic_test=None, xstest=None):
    native_only()
    if toxic_test is None or xstest is None:
        raise ValueError("required_external_regression_inputs_missing")
    start = time.perf_counter()
    data, models, output = Path(data), Path(models), Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    freeze = json.loads((models / "pre-evaluation-freeze.json").read_text())
    if freeze["test_scored"] is not False:
        raise ValueError("invalid_pre_evaluation_freeze")
    for name, value in freeze["files"].items():
        if file_hash(models / name) != value:
            raise ValueError("frozen_model_changed")
    for name, value in freeze["data_files"].items():
        if file_hash(data / name) != value:
            raise ValueError("frozen_data_changed")
    cards_hash = file_hash(models / "model-cards.json")
    cards = json.loads((models / "model-cards.json").read_text())
    p = cards["policy"]
    policy = BinaryPolicy(p["threshold"], p["model_pair_id"], tuple(p["calibration_ids"]))
    factory = lambda: SyntheticBackend(models, cards_hash)
    test, validation = load_rows(data / "test.jsonl"), load_rows(data / "validation.jsonl")
    fit, verification = select_probes(validation, "probe_fit", 32), select_probes(validation, "probe_verification", 64)
    endpoint = endpoint_rows(test)
    write_json(output / "evaluation-plan.json", {"source_commit": source_commit, "test_rows": len(test),
        "test_family_ids": sorted({r.family for r in test}), "fit_ids": [r.prompt_id for r in fit],
        "verification_ids": [r.prompt_id for r in verification], "endpoint_ids": [r.prompt_id for r in endpoint],
        "pre_evaluation_freeze_sha256": file_hash(models / "pre-evaluation-freeze.json"),
        "scenarios": SCENARIOS, "seeds": list(range(5)), "endpoint_is_subset": True,
        "real_model_inference_on_every_runtime_call": True})
    backend = factory()
    healthy = score_rows(backend, test + fit + verification)
    write_json(output / "healthy-scores.json", healthy)
    healthy_metrics = {name: metrics([r.label for r in test], [healthy[r.prompt_id][name + "_score"] for r in test],
                       getattr(backend, name + "_threshold")) for name in ("guardrail", "canary")}
    by_domain = {d: metrics([r.label for r in test if r.domain == d],
                 [healthy[r.prompt_id]["guardrail_score"] for r in test if r.domain == d], backend.guardrail_threshold)
                 for d in sorted({r.domain for r in test})}
    labels = np.asarray([r.label for r in test])
    gpred = np.asarray([healthy[r.prompt_id]["guardrail_score"] >= backend.guardrail_threshold for r in test])
    cpred = np.asarray([healthy[r.prompt_id]["canary_score"] >= backend.canary_threshold for r in test])
    summaries, timings = [], Counter()
    for scenario in SCENARIOS:
        for seed in range(5):
            events, summary, final = episode(scenario, seed, factory=factory, policy=policy, test=test,
                endpoint=endpoint, fit=fit, verification=verification, healthy=healthy, timings=timings)
            prefix = f"{scenario}-{seed}"
            with (output / (prefix + "-events.jsonl")).open("x") as handle:
                for event in events:
                    handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            write_json(output / (prefix + "-summary.json"), summary)
            write_json(output / (prefix + "-endpoint.json"), final)
            summaries.append(summary)
        print(json.dumps({"completed_scenario": scenario, "paired_episodes": len(summaries)}), flush=True)
    aggregate = {}
    for scenario in SCENARIOS:
        group = [s for s in summaries if s["scenario"] == scenario]
        aggregate[scenario] = {"A": combine(group, "A"), "B": combine(group, "B"),
            "restored_episodes": sum(s["restored_service"] for s in group), "held_episodes": sum(s["held"] for s in group),
            "fault_incidents": sum(len(s["onsets"]) for s in group),
            "confirmed_faults": sum(d["confirmed_index"] is not None for s in group for d in s["causal_detection"]),
            "confirmation_delays": [d["confirmation_delay"] for s in group for d in s["causal_detection"] if d["confirmation_delay"] is not None],
            "pre_fault_confirmations": sum(s["pre_fault_confirmations"] for s in group),
            "A_final": group[0]["A_final"], "B_final": group[0]["B_final"]}
    costs = Counter()
    for s in summaries:
        costs.update(s["costs"])
    deployed_inputs = {arm: sum(value for key, value in costs.items() if key.startswith(arm + ":") and key.endswith(":inputs") and ":endpoint:" not in key) for arm in ("A", "B")}
    report = {"source_commit": source_commit, "architecture": platform.machine(), "test_kind": "synthetic_intent_family_holdout",
        "healthy_test": healthy_metrics, "healthy_guardrail_by_domain": by_domain,
        "error_correlation": error_correlation(labels, gpred, cpred), "aggregate": aggregate,
        "overall": {arm: combine(summaries, arm) for arm in ("A", "B")},
        "fault_only": {arm: combine([s for s in summaries if s["onsets"]], arm) for arm in ("A", "B")},
        "no_fault": {arm: combine([s for s in summaries if not s["onsets"]], arm) for arm in ("A", "B")},
        "paired_episodes": len(summaries), "costs": dict(costs),
        "runtime_model_inputs": {"A_guardrail": deployed_inputs["A"], "A_canary": 0,
                                 "B_guardrail": deployed_inputs["B"], "B_canary": deployed_inputs["B"]},
        "external_regression": external_scores(backend, toxic_test, xstest),
        "elapsed_seconds": time.perf_counter() - start,
        "limits": ["100000 compositional rows, not 100000 independent human requests.",
                   "Scenario prevalence is experimental, not a production estimate.",
                   "Five seeds reuse cases and are not independent population trials.",
                   "Holds and unavailable requests count as failed automatic decisions, not recovery.",
                   "Existing finite probes can miss input-specific and semantic errors.",
                   "Offline approvals and an agent stub are not a deployed autonomous service.",
                   "No raw requests or personal data belong in public reports."]}
    write_json(output / "report.json", report)
    write_json(output / "timing.json", {"architecture": platform.machine(),
        "backend_score_wall_seconds": dict(timings),
        "runtime_seconds": {arm: sum(value for key, value in timings.items()
                            if key.startswith(arm + ":") and not key.endswith(":endpoint")) for arm in ("A", "B")},
        "scope": "Actual backend score calls; excludes model loads, controller logic and endpoint inference. Not a production latency or throughput SLA.",
        "memory_evidence": "Separate execution receipt records process-scope peak RSS, not per-arm memory."})
    for name, value in freeze["files"].items():
        if file_hash(models / name) != value:
            raise ValueError("model_changed_during_evaluation")
    write_json(output / "evidence.json", {"source_commit": source_commit, "model_freeze_preserved": True,
        "files": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "models", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--toxic-test", type=Path, required=True)
    parser.add_argument("--xstest", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.data, args.models, args.output, source_commit=args.source_commit,
                 toxic_test=args.toxic_test, xstest=args.xstest)
    print(json.dumps({"overall": report["overall"], "seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()

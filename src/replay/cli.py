"""Offline deterministic paired benchmark. No live agent or external action."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

from src.monitor.cli import new_output, write_new_json
from src.monitor.disagreement import Calibration, MonitorError, canonical, strict_json
from src.response.conformal import ResponseError
from .data import ReplayError, file_sha, load_pools, score_cache, split_label_roles
from .harness import ReplaySpec, SCENARIOS, build_stream, compare_arms, initial_policy, run_arm
from .scorecard import write_scorecard


def run_benchmark(models, pools, monitor_config_path, output_dir, *, source_commit,
                  seeds=(0, 1, 2, 3, 4), approve_repair=False, spec=None, scorer_factory=None):
    models, pools, output_dir = Path(models), Path(pools), Path(output_dir)
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ReplayError("invalid_source_commit")
    if type(approve_repair) is not bool or not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ReplayError("invalid_replay_seeds_or_authority")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ReplayError("nonempty_benchmark_output")
    config = Calibration.from_dict(strict_json(Path(monitor_config_path).read_text()))
    spec = spec or ReplaySpec()
    if config.horizon != spec.horizon:
        raise ReplayError("monitor_horizon_does_not_match_protocol")
    records, pool_manifest = load_pools(pools)
    label_roles = split_label_roles(records)
    final_evaluation = tuple(row for row in records if row.purpose == "evaluation" and row.label is not None)
    if any(sum(row.label == label for row in final_evaluation) < 20 for label in (0, 1)):
        raise ReplayError("insufficient_final_evaluation_classes")
    evaluation_ids = {row.prompt_id for row in final_evaluation}
    if any(evaluation_ids & {row.prompt_id for row in values} for values in label_roles.values()):
        raise ReplayError("evaluation_response_label_overlap")
    # Verify no-replacement feasibility before model inference.
    streams = {(scenario, seed): build_stream(records, scenario, seed, spec) for scenario in SCENARIOS for seed in seeds}
    scores, cache_report = score_cache(records, models, scorer_factory=scorer_factory)
    if cache_report["model_pair_id"] != config.model_pair_id:
        raise ReplayError("cached_model_monitor_binding_mismatch")
    initial = initial_policy(label_roles, scores, config.model_pair_id)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = {"schema_version": 1, "source_commit": source_commit, "label_policy": initial.label_policy,
        "model_pair_id": config.model_pair_id, "monitor_calibration_id": config.calibration_id,
        "monitor_config_sha256": file_sha(monitor_config_path), "pool_manifest_sha256": file_sha(pools / "pool-manifest.json"),
        "seeds": list(seeds), "spec": asdict(spec), "approval_authorized": approve_repair,
        "calibration_nominal_coverage": 0.95, "verification_coverage_floor": 0.90, "verification_review_ceiling": 0.75,
        "initial_policy": initial.to_dict(), "score_cache": cache_report,
        "label_role_counts": {role: {str(label): sum(row.label == label for row in values) for label in (0, 1)} for role, values in label_roles.items()},
        "final_evaluation_counts": {str(label): sum(row.label == label for row in final_evaluation) for label in (0, 1)},
        "source_label_policy_is_universal_toxicity": False, "offline_label_and_approval_simulation": True}
    protocol = Path(__file__).resolve().parents[2] / "docs/BENCHMARK-PROTOCOL.md"
    if protocol.is_file():
        metadata["protocol_sha256"] = file_sha(protocol)
    write_new_json(output_dir / "label-splits.json", {role: [row.prompt_id for row in values] for role, values in label_roles.items()})
    write_new_json(output_dir / "benchmark-metadata.json", metadata)
    with new_output(output_dir / "score-cache.jsonl") as handle:
        for prompt_id in sorted(scores):
            handle.write(canonical({"prompt_id": prompt_id, **scores[prompt_id], "model_pair_id": config.model_pair_id}) + "\n")
    comparisons = []
    for scenario in SCENARIOS:
        for seed in seeds:
            stream = streams[(scenario, seed)]
            summaries = []
            for watcher in (False, True):
                events, summary = run_arm(stream, scores, initial, label_roles, final_evaluation, config,
                    scenario=scenario, seed=seed, spec=spec, watcher=watcher, approve_repair=approve_repair)
                arm = "B" if watcher else "A"
                name = f"{scenario}-seed{seed}-{arm}"
                with new_output(output_dir / f"{name}.jsonl") as handle:
                    for event in events:
                        handle.write(canonical(event) + "\n")
                write_new_json(output_dir / f"{name}-summary.json", summary)
                summaries.append(summary)
            comparisons.append(compare_arms(*summaries))
    write_new_json(output_dir / "paired-summaries.json", {"metadata": metadata, "comparisons": comparisons})
    card = write_scorecard(output_dir, comparisons, metadata)
    # Verify trusted classifier files and cards stay unchanged after inference.
    cards = strict_json((models / "model-cards.json").read_text())
    if file_sha(models / "model-cards.json") != cache_report["model_cards_sha256"]:
        raise ReplayError("model_cards_change_during_benchmark")
    if scorer_factory is None:
        for item in cards["artifacts"].values():
            if file_sha(models / item["file"]) != item["sha256"]:
                raise ReplayError("model_artifact_change_during_benchmark")
    outputs = {path.name: {"sha256": file_sha(path), "bytes": path.stat().st_size} for path in sorted(output_dir.iterdir()) if path.is_file()}
    write_new_json(output_dir / "evidence-manifest.json", {"source_commit": source_commit, "model_artifacts_unchanged": True,
        "outputs": outputs, "integrity_scope": "Hashes identify private local artifacts, not authenticity or hardware execution."})
    return card


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen paired source-labelled offline replay")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--monitor-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--approve-repair", action="store_true")
    args = parser.parse_args(argv)
    try:
        card = run_benchmark(args.models, args.pools, args.monitor_config, args.output_dir,
            source_commit=args.source_commit, approve_repair=args.approve_repair)
        print(json.dumps({"episodes": len(card["episodes"]), "negative_control": card["negative_control"], "aggregate": card["aggregate"]}, sort_keys=True))
    except (ReplayError, ResponseError, MonitorError) as exc:
        print(json.dumps({"kind": "error", "code": str(exc)}), file=sys.stderr)
        return 2
    except (OSError, KeyError, TypeError, ValueError):
        print(json.dumps({"kind": "error", "code": "benchmark_input_or_output_error"}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

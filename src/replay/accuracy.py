"""Evaluate a validation-selected model without changing the runtime benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.aegis import require_native
from src.data.synthetic import file_hash, write_json
from src.models.accuracy import quality, read_aegis
from src.models.synthetic import SyntheticBackend
from src.monitor.runtime import checked_scores
from src.release.cli import load_selected
from src.replay.synthetic import run


def backend_type(models):
    kind = json.loads((Path(models) / "model-cards.json").read_text()).get("backend")
    if kind is None:
        return SyntheticBackend
    if kind == "frozen_minilm":
        from src.models.semantic import SemanticBackend
        return SemanticBackend
    raise ValueError("unsupported_accuracy_backend")


def fresh_test(development, models, baseline_models, output):
    require_native()
    development, models, output = Path(development), Path(models), Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    freeze = json.loads((models / "pre-evaluation-freeze.json").read_text())
    for name, expected in freeze["files"].items():
        if file_hash(models / name) != expected:
            raise ValueError("candidate_model_changed_before_fresh_test")
    for name, expected in freeze["development_files"].items():
        if file_hash(development / name) != expected:
            raise ValueError("development_data_changed_before_fresh_test")
    rows = read_aegis(development / "test.jsonl")
    _, baseline = load_selected(baseline_models)
    candidate = backend_type(models)(models)
    outputs = {"baseline": [], "candidate": []}
    with (output / "scores.jsonl").open("x") as handle:
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            a = checked_scores(baseline, [r["text"] for r in batch], baseline.model_pair_id)
            b = checked_scores(candidate, [r["text"] for r in batch], candidate.model_pair_id)
            for row, old, new in zip(batch, a, b):
                if old["canary_score"] != new["canary_score"]:
                    raise ValueError("frozen_canary_output_changed")
                outputs["baseline"].append(old["guardrail_score"])
                outputs["candidate"].append(new["guardrail_score"])
                handle.write(json.dumps({"prompt_id": row["prompt_id"], "label": row["label"],
                                        "baseline": old, "candidate": new}, sort_keys=True) + "\n")
    result = {"scope": "First-use prepared Aegis test; provider prompt policy; healthy classifier quality, not operational accuracy.",
              "baseline": quality(rows, outputs["baseline"], baseline.guardrail_threshold),
              "candidate": quality(rows, outputs["candidate"], candidate.guardrail_threshold),
              "rows": len(rows), "groups": len({r["group_id"] for r in rows}),
              "canary_scores_identical": True, "test_driven_changes": False,
              "data_sha256": file_hash(development / "test.jsonl"),
              "pre_evaluation_freeze_sha256": file_hash(models / "pre-evaluation-freeze.json")}
    write_json(output / "report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark", "fresh-test"))
    for name in ("data", "models", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-models", "toxic-test", "xstest"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.command == "fresh-test":
        if args.baseline_models is None:
            parser.error("--baseline-models is required")
        fresh_test(args.data, args.models, args.baseline_models, args.output)
    else:
        if not args.source_commit or args.toxic_test is None or args.xstest is None:
            parser.error("--source-commit, --toxic-test, and --xstest are required")
        cards = json.loads((args.models / "model-cards.json").read_text())
        report = run(args.data, args.models, args.output, source_commit=args.source_commit,
                     toxic_test=args.toxic_test, xstest=args.xstest, backend_factory=backend_type(args.models),
                     extra_report_fields={"evaluation_status": "exposed_synthetic_and_external_regression",
                                          "fresh_independent_confirmation": False,
                                          "additional_training_source": "Aegis human prompt labels",
                                          "cross_source_semantic_independence_proven": False,
                                          "runtime_rules_unchanged": True, "canary_unchanged": True,
                                          "guardrail_components": cards.get("guardrail_components", ["tfidf_guardrail"]),
                                          "cost_scope": "Guardrail counts are composite input evaluations; every constituent classifier executes without a model-output cache."})
        print(json.dumps({"overall": report["overall"]}, sort_keys=True))


if __name__ == "__main__":
    main()

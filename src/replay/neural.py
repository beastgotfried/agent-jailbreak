"""The unchanged paired replay with the frozen native neural guardrail backend."""
import argparse
import json
from pathlib import Path

from src.models.neural import NeuralBackend, configure_cpu
from src.replay.synthetic import run


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "models", "output", "toxic-test", "xstest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    configure_cpu()
    cards = json.loads((args.models / "model-cards.json").read_text())
    components = ["bert_tiny"] + (["frozen_lexical", "logistic_stack"] if cards["selected"] == "neural_lexical_stack" else [])
    result = run(args.data, args.models, args.output, source_commit=args.source_commit,
        toxic_test=args.toxic_test, xstest=args.xstest, backend_factory=NeuralBackend,
        extra_report_fields={"evaluation_status": "exposed_synthetic_and_external_regression",
            "fresh_independent_confirmation": False, "canary_unchanged": True,
            "guardrail_components": components,
            "cost_count_unit": "Guardrail request evaluations. An ensemble evaluation executes every listed guardrail component; canary counts stay separate.",
            "model_output_cache": False})
    print(json.dumps({"overall": result["overall"], "seconds": result["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()

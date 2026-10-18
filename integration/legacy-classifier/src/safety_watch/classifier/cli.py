"""JSON/JSONL CLI. Prompt text is data, never echoed or executed."""
import argparse
import contextlib
import json
import sys
from pathlib import Path
from .config import TrainConfig
from .training import train_guardrail
from .artifacts import export_run, load_guardrail
from .evaluation import evaluate_run, evaluate_final
from .evidence import new_operation
from .io import canonical, require

def parser():
    root = argparse.ArgumentParser(description="Native LinuxONE guardrail tools")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("train", "export", "score", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--runtime-root", type=Path, default=Path("/runtime"))
        p.add_argument("--project-root", type=Path, default=Path("/workspace"))
        if name == "train":
            p.add_argument("--search", action="store_true", help="Repeat fixed 12-candidate fitting-only selection")
            p.add_argument("--c", type=float, default=100.0)
            p.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
        elif name == "export":
            p.add_argument("--run", type=Path, required=True)
            p.add_argument("--model-dir", type=Path)
        elif name == "score":
            p.add_argument("--model-dir", type=Path)
            p.add_argument("--input", default="-", help="JSONL with prompt_id and user_input; - reads stdin")
            p.add_argument("--output", default="-", help="Score JSONL; - writes stdout")
            p.add_argument("--batch-size", type=int, default=256)
        else:
            p.add_argument("--run", type=Path)
            p.add_argument("--model-dir", type=Path)
            p.add_argument("--split", choices=("validation", "test"), default="validation")
            p.add_argument("--final-test", action="store_true")
    return root

def score_records(args):
    require(args.batch_size > 0, "Batch size must be positive.")
    model = load_guardrail(args.model_dir or args.runtime_root / "models")
    _, _, result_dir, evidence = new_operation(args.runtime_root, "score")
    evidence("scoring_started", {"model_sha256": model.digest, "threshold": model.threshold})
    count = 0
    try:
        with contextlib.ExitStack() as stack:
            source = sys.stdin if args.input == "-" else stack.enter_context(Path(args.input).open())
            output = sys.stdout if args.output == "-" else stack.enter_context(Path(args.output).open("x"))
            ids, texts = [], []
            def flush():
                nonlocal count
                if not ids:
                    return
                for record in model.predict(ids, texts):
                    output.write(canonical(record) + "\n")
                    count += 1
                ids.clear()
                texts.clear()
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    raise RuntimeError("Malformed input JSON at line " + str(line_number)) from None
                require(isinstance(record, dict), "Each input line must be a JSON object.")
                require(isinstance(record.get("prompt_id"), str) and bool(record["prompt_id"]),
                        "Each input needs a nonempty prompt_id.")
                require(isinstance(record.get("user_input"), str) and bool(record["user_input"].strip()),
                        "Each input needs nonempty user_input text.")
                ids.append(record["prompt_id"])
                texts.append(record["user_input"])
                if len(ids) >= args.batch_size:
                    flush()
            flush()
        evidence("scoring_completed", {"records": count, "model_sha256": model.digest})
    except Exception as error:
        evidence("scoring_failed", {"completed_records": count, "error_type": type(error).__name__})
        raise
    return {"records": count, "model_sha256": model.digest,
            "results_directory": str(result_dir), "evidence": evidence.verify()}

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "train":
            config = TrainConfig(runtime_root=args.runtime_root, project_root=args.project_root, search=args.search)
            config.model.update(C=args.c, class_weight=None if args.class_weight == "none" else "balanced")
            report = train_guardrail(config).summary
        elif args.command == "export":
            report = export_run(args.run, args.runtime_root, args.model_dir)
        elif args.command == "score":
            report = score_records(args)
            print(canonical(report), file=sys.stderr)
            return 0
        elif args.split == "test":
            require(args.final_test and args.run is None, "Test requires --final-test and no development --run.")
            report = evaluate_final(args.runtime_root, args.model_dir, confirmed=True)
        else:
            require(not args.final_test and args.run is not None, "Validation requires --run and no --final-test.")
            report = evaluate_run(args.run, args.runtime_root)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        message = str(error) if isinstance(error, RuntimeError) else "Check runtime, input schema, paths, and saved artifact metadata."
        print(canonical({"error_type": type(error).__name__, "message": message}), file=sys.stderr)
        return 1

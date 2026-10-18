"""Canary training, export and paired JSONL scoring on the native VM."""
import argparse
import contextlib
import json
import sys
from pathlib import Path
from safety_watch.classifier.evidence import new_operation
from safety_watch.classifier.io import canonical, require
from .training import train_canary
from .artifacts import export_canary
from .pair import load_pair, joint_records, evaluate_pair

def parser():
    root = argparse.ArgumentParser(description="Word canary and frozen-pair tools")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("train", "export", "score", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--runtime-root", type=Path, default=Path("/runtime"))
        p.add_argument("--project-root", type=Path, default=Path("/workspace"))
        if name == "export":
            p.add_argument("--run", type=Path, required=True)
        if name in ("export", "score", "evaluate"):
            p.add_argument("--model-dir", type=Path)
        if name == "score":
            p.add_argument("--input", default="-")
            p.add_argument("--output", default="-")
            p.add_argument("--batch-size", type=int, default=256)
    return root

def score_records(args):
    require(args.batch_size > 0, "Batch size must be positive.")
    pair = load_pair(args.model_dir or args.runtime_root / "models")
    _, _, result_dir, evidence = new_operation(args.runtime_root, "pair-score")
    evidence("pair_scoring_started", {"guardrail_sha256": pair[0].digest, "canary_sha256": pair[1].digest})
    count = 0
    try:
        with contextlib.ExitStack() as stack:
            source = sys.stdin if args.input == "-" else stack.enter_context(Path(args.input).open())
            output = sys.stdout if args.output == "-" else stack.enter_context(Path(args.output).open("x"))
            ids, texts, splits = [], [], []
            def flush():
                nonlocal count
                if not ids:
                    return
                for row in joint_records(*pair, ids, texts, splits):
                    output.write(canonical(row) + "\n")
                    count += 1
                ids.clear()
                texts.clear()
                splits.clear()
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    raise RuntimeError("Malformed input JSON at line " + str(line_number)) from None
                require(isinstance(row, dict), "Each input must be a JSON object.")
                require(isinstance(row.get("prompt_id"), str) and row["prompt_id"].strip(),
                        "Input needs a nonempty prompt_id.")
                require(isinstance(row.get("user_input"), str) and row["user_input"].strip(),
                        "Input needs nonempty user_input text.")
                split = row.get("split", "live")
                require(isinstance(split, str) and split.strip(), "Split must be a nonempty string.")
                ids.append(row["prompt_id"])
                texts.append(row["user_input"])
                splits.append(split)
                if len(ids) >= args.batch_size:
                    flush()
            flush()
        evidence("pair_scoring_completed", {"records": count})
    except Exception as error:
        evidence("pair_scoring_failed", {"completed_records": count, "error_type": type(error).__name__})
        raise
    return {"records": count, "guardrail_sha256": pair[0].digest, "canary_sha256": pair[1].digest,
            "results_directory": str(result_dir), "evidence": evidence.verify()}

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "train":
            report = train_canary(args.runtime_root, args.project_root).summary
        elif args.command == "export":
            report = export_canary(args.run, args.runtime_root, args.model_dir)
        elif args.command == "evaluate":
            report = evaluate_pair(args.runtime_root, args.model_dir)
        else:
            report = score_records(args)
            print(canonical(report), file=sys.stderr)
            return 0
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        message = str(error) if isinstance(error, RuntimeError) else "Check input schema, runtime and artifact metadata."
        print(canonical({"error_type": type(error).__name__, "message": message}), file=sys.stderr)
        return 1

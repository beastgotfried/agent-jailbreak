"""Verify and score with the selected private TF-IDF model pair.

This command classifies text only. It does not run requests, enable a runtime
monitor, authorize recovery, or provide a production API. Use paired replay to
exercise the complete offline monitor/recovery path.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from importlib.resources import files
import json
from pathlib import Path
import platform
import sys

from src.data.synthetic import file_hash, native_only
from src.models.synthetic import SyntheticBackend
from src.monitor.runtime import checked_scores


class ReleaseError(ValueError):
    pass


def selected_release():
    return json.loads(files("src.release").joinpath("selected.json").read_text(encoding="utf-8"))


def verify_selected(models):
    directory = Path(models)
    spec = selected_release()
    for name, expected in spec["model_files"].items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ReleaseError("release_model_file_missing_or_invalid")
        if file_hash(path) != expected:
            raise ReleaseError("release_model_hash_mismatch")
    cards = json.loads((directory / "model-cards.json").read_text())
    for key in ("model_pair_id", "guardrail_threshold", "canary_threshold", "guardrail_method"):
        if cards.get(key) != spec[key]:
            raise ReleaseError("release_model_binding_mismatch")
    return spec


def load_selected(models):
    spec = verify_selected(models)
    backend = SyntheticBackend(models, spec["model_files"]["model-cards.json"])
    return spec, backend


def score_lines(backend, source, target):
    for number, line in enumerate(iter(lambda: source.readline(65537), ""), 1):
        if len(line) > 65536:
            raise ReleaseError("release_request_too_large")
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            raise ReleaseError("release_invalid_request_json") from None
        if not isinstance(item, dict) or set(item) != {"text"}:
            raise ReleaseError("release_request_requires_only_text")
        text = item["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 16384:
            raise ReleaseError("release_invalid_request_text")
        value = checked_scores(backend, [text], backend.model_pair_id)[0]
        result = {"sequence": number - 1, "model_pair_id": backend.model_pair_id,
                  "decision": "refuse" if value["guardrail_unsafe"] else "allow",
                  "guardrail_score": value["guardrail_score"], "canary_score": value["canary_score"],
                  "monitor_enabled": False, "action_executed": False,
                  "scope": "classifier_only_not_a_safety_guarantee"}
        target.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        target.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "score"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--input", default="-", help="JSONL with one text field per row; default stdin")
    parser.add_argument("--output", default="-", help="New JSONL output file; default stdout")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            spec = verify_selected(args.models)
            print(json.dumps({"release_id": spec["release_id"], "verified_model_files": len(spec["model_files"]),
                "model_pair_id": spec["model_pair_id"], "current_architecture": platform.machine(),
                "qualified_architecture": spec["qualified_architecture"], "model_loaded": False,
                "scope": spec["scope"]}, sort_keys=True))
            return 0
        native_only()
        spec, backend = load_selected(args.models)
        with ExitStack() as stack:
            source = sys.stdin if args.input == "-" else stack.enter_context(Path(args.input).open(encoding="utf-8"))
            target = sys.stdout if args.output == "-" else stack.enter_context(Path(args.output).open("x", encoding="utf-8"))
            score_lines(backend, source, target)
        return 0
    except ReleaseError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except Exception:
        # Do not expose request text, private filesystem paths or backend errors.
        print(json.dumps({"error": "release_operation_failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

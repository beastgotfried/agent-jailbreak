"""Private JSONL adapters for paired scores, calibration, and online updates."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import islice
import json
import os
from pathlib import Path
import sys
import tempfile

from .disagreement import (
    Calibration, DisagreementMonitor, MonitorError, ScorePair, calibrate,
    canonical, pair_streams, strict_json,
)


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or len(line) > 32768:
                raise MonitorError("invalid_score_line")
            yield strict_json(line)


def events(path):
    return (ScorePair.from_dict(row) for row in rows(path))


@contextmanager
def new_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        yield handle


def write_new_json(path, payload):
    with new_output(path) as handle:
        handle.write(canonical(payload) + "\n")


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(canonical(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _run(args):
    calibration = Calibration.from_dict(strict_json(args.config.read_text(encoding="utf-8")))
    monitor = DisagreementMonitor(calibration)
    if args.checkpoint_in:
        monitor = DisagreementMonitor.restore(calibration, strict_json(args.checkpoint_in.read_text(encoding="utf-8")))
    if args.limit is not None and (args.limit < 1 or args.limit > calibration.horizon - monitor.accepted_count):
        raise MonitorError("invalid_input_limit")
    if args.checkpoint_out:
        protected = {path.resolve() for path in (args.config, args.input, args.output)}
        if args.checkpoint_out.resolve() in protected:
            raise MonitorError("checkpoint_path_collides_with_data")
    processed = 0
    with new_output(args.output) as handle:
        try:
            source = events(args.input)
            if args.limit is not None:
                source = islice(source, args.limit)
            for event in source:
                update = monitor.update(event)
                handle.write(canonical(update) + "\n")
                handle.flush()
                processed += 1
            if not processed:
                raise MonitorError("no_score_events")
            handle.write(canonical({
                "kind": "episode_summary", "processed_this_call": processed,
                "accepted_count": monitor.accepted_count, "alarm_latched": monitor.alarm is not None,
                "horizon_exhausted": monitor.exhausted, "configured_horizon": calibration.horizon,
                "max_looks": calibration.max_looks, "calibration_id": calibration.calibration_id,
                "validity": "empirical_finite_episode_only", "harm_verified": False,
            }) + "\n")
        except MonitorError as exc:
            handle.write(canonical({"kind": "telemetry_fault", "code": str(exc), "accepted_count": monitor.accepted_count, "harm_verified": False}) + "\n")
            raise
        finally:
            if args.checkpoint_out:
                atomic_checkpoint(args.checkpoint_out, monitor.checkpoint())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score-only disagreement change detector")
    commands = parser.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("pair", help="validate and join two ordered score streams")
    pair.add_argument("--guardrail", type=Path, required=True)
    pair.add_argument("--canary", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    calibration = commands.add_parser("calibrate", help="freeze an empirical finite-episode configuration")
    calibration.add_argument("--reference", type=Path, required=True)
    calibration.add_argument("--calibration", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--window", type=int, default=64)
    calibration.add_argument("--stride", type=int, default=16)
    calibration.add_argument("--horizon", type=int, default=512)
    calibration.add_argument("--alpha", type=float, default=0.05)
    calibration.add_argument("--null-runs", type=int, default=200)
    calibration.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("run", help="read paired events and emit numeric change evidence")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--checkpoint-in", type=Path)
    run.add_argument("--checkpoint-out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "pair":
            count = 0
            with new_output(args.output) as handle:
                for event in pair_streams(rows(args.guardrail), rows(args.canary)):
                    handle.write(canonical(event.to_dict()) + "\n")
                    count += 1
            if not count:
                raise MonitorError("no_score_events")
        elif args.command == "calibrate":
            result = calibrate(events(args.reference), events(args.calibration),
                window_size=args.window, stride=args.stride, horizon=args.horizon,
                alpha=args.alpha, null_runs=args.null_runs, seed=args.seed)
            write_new_json(args.output, result.to_dict())
        else:
            _run(args)
    except MonitorError as exc:
        print(canonical({"kind": "error", "code": str(exc)}), file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError, OverflowError):
        print(canonical({"kind": "error", "code": "input_or_output_error"}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute the bounded native experiment from an exact committed archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tarfile
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from src.data.synthetic import file_hash, write_json
from src.release.cli import verify_selected


def verify_source(archive, commit):
    with tarfile.open(archive) as handle:
        if handle.pax_headers.get("comment") != commit:
            raise ValueError("source_archive_commit_mismatch")
        for member in handle.getmembers():
            if member.isfile():
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("invalid_source_archive_path")
                expected = hashlib.sha256(handle.extractfile(member).read()).hexdigest()
                if file_hash(SOURCE / name) != expected:
                    raise ValueError("source_archive_content_mismatch")
            elif not member.isdir():
                raise ValueError("unexpected_source_archive_member")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-archive", "baseline-study", "raw-aegis", "toxic-train", "toxic-test", "xstest", "pools", "dolly", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--phase", choices=("mixed", "stack"), default="mixed")
    parser.add_argument("--first-phase", type=Path)
    args = parser.parse_args()
    if args.phase == "stack" and args.first_phase is None:
        parser.error("--first-phase is required for stack calibration")
    if platform.machine() != "s390x" or platform.system() != "Linux":
        raise ValueError("native_s390x_required")
    verify_source(args.source_archive, args.source_commit)
    verify_selected(args.baseline_study / "models")
    os.umask(0o077)
    out = args.output.resolve()
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / "tmp").mkdir(mode=0o700)
    protocol = SOURCE / "experiments/accuracy-above-84" / ("stack-protocol.json" if args.phase == "stack" else "protocol.json")
    env = dict(os.environ, PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TMPDIR=str(out / "tmp"))
    env.pop("SAFETY_WATCH_ALLOW_LOCAL", None)
    stages = []

    def stage(name, command, timeout):
        start = time.perf_counter()
        with (out / (name + ".log")).open("xb") as log:
            result = subprocess.run([str(x) for x in command], cwd=SOURCE, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        record = {"stage": name, "exit_code": result.returncode, "seconds": time.perf_counter() - start,
                  "log_sha256": file_hash(out / (name + ".log")),
                  "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
        stages.append(record)
        write_json(out / (name + "-receipt.json"), record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if result.returncode:
            raise RuntimeError("experiment_stage_failed_inspect_private_log")

    receipt = {"phase": args.phase, "source_commit": args.source_commit, "source_archive_sha256": file_hash(args.source_archive),
               "protocol_sha256": file_hash(protocol), "architecture": platform.machine(), "stages": stages}
    try:
        stage("01-tests", [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "--basetemp", out / "tmp/pytest"], 180)
        if args.phase == "mixed":
            stage("02-data", [sys.executable, "-m", "src.data.aegis", "--raw", args.raw_aegis,
                  "--output", out / "development", "--synthetic", args.baseline_study / "data",
                  "--toxic-train", args.toxic_train, "--toxic-test", args.toxic_test, "--xstest", args.xstest,
                  "--pools", args.pools, "--dolly", args.dolly, "--source-commit", args.source_commit], 300)
            module, extra = "src.models.accuracy", []
        else:
            previous = json.loads((args.first_phase / "execution-receipt.json").read_text())
            if previous["status"] != "completed_no_eligible_candidate" or previous["test_scored"] is not False:
                raise ValueError("unexpected_first_phase_state")
            freeze = json.loads((args.first_phase / "freeze-manifest.json").read_text())
            for name, expected in freeze["files"].items():
                if file_hash(args.first_phase / name) != expected:
                    raise ValueError("first_phase_artifact_changed")
            (out / "development").symlink_to(os.path.relpath(args.first_phase / "development", out), target_is_directory=True)
            receipt["first_phase_freeze_sha256"] = file_hash(args.first_phase / "freeze-manifest.json")
            module = "src.models.calibrated"
            extra = ["--mixed-models", args.first_phase / "candidates/tfidf_mixed_75_25"]
        stage("03-fit", [sys.executable, "-m", module, "--synthetic", args.baseline_study / "data",
              "--development", out / "development", "--baseline-models", args.baseline_study / "models",
              "--output", out / "candidates", "--protocol", protocol, "--source-commit", args.source_commit, *extra], 1200)
        selection = json.loads((out / "candidates/selection.json").read_text())
        receipt["selected"] = selection["selected"]
        if selection["selected"] is None:
            receipt["status"] = "completed_no_eligible_candidate"
            receipt["test_scored"] = False
        else:
            shutil.copytree(out / "candidates" / selection["selected"], out / "models")
            (out / "data").symlink_to(os.path.relpath(args.baseline_study / "data", out), target_is_directory=True)
            stage("04-benchmark", [sys.executable, "-m", "src.replay.accuracy", "benchmark", "--data", out / "data",
                  "--models", out / "models", "--output", out / "benchmark", "--source-commit", args.source_commit,
                  "--toxic-test", args.toxic_test, "--xstest", args.xstest], 1200)
            stage("05-audit", [sys.executable, SOURCE / "scripts/audit_synthetic.py", out], 180)
            stage("06-fresh-test", [sys.executable, "-m", "src.replay.accuracy", "fresh-test", "--data", out / "development",
                  "--models", out / "models", "--baseline-models", args.baseline_study / "models", "--output", out / "fresh-test"], 180)
            report = json.loads((out / "benchmark/report.json").read_text())
            old_plan = json.loads((args.baseline_study / "benchmark/evaluation-plan.json").read_text())
            new_plan = json.loads((out / "benchmark/evaluation-plan.json").read_text())
            for key in ("fit_ids", "verification_ids", "endpoint_ids", "scenarios", "seeds"):
                if old_plan[key] != new_plan[key]:
                    raise ValueError("benchmark_identity_or_scenario_changed")
            old_healthy = json.loads((args.baseline_study / "benchmark/healthy-scores.json").read_text())
            new_healthy = json.loads((out / "benchmark/healthy-scores.json").read_text())
            if set(old_healthy) != set(new_healthy) or any(old_healthy[k]["canary_score"] != new_healthy[k]["canary_score"] for k in old_healthy):
                raise ValueError("benchmark_canary_changed")
            for scenario in new_plan["scenarios"]:
                for seed in new_plan["seeds"]:
                    name = f"{scenario}-{seed}-summary.json"
                    before = json.loads((args.baseline_study / "benchmark" / name).read_text())
                    after = json.loads((out / "benchmark" / name).read_text())
                    for key in ("input_digest", "onsets", "events"):
                        if before[key] != after[key]:
                            raise ValueError("benchmark_stream_changed")
            receipt.update(status="completed", test_scored=True, benchmark_invariants_verified=True,
                           operational_accuracy=report["overall"]["B"]["operational_accuracy"],
                           target_exceeded=report["overall"]["B"]["operational_accuracy"] > .84)
        verify_selected(args.baseline_study / "models")
        verify_source(args.source_archive, args.source_commit)
        receipt["released_model_files_unchanged"] = True
        receipt["test_driven_changes"] = False
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        write_json(out / "execution-receipt.json", receipt)
        raise
    write_json(out / "execution-receipt.json", receipt)
    freeze = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*"))
              if p.is_file() and "tmp" not in p.relative_to(out).parts and not p.is_symlink()}
    write_json(out / "freeze-manifest.json", {"source_commit": args.source_commit, "files": freeze})
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

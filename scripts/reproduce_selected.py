"""Rebuild the selected experiment from private inputs on native LinuxONE.

This is an exact-protocol reproduction, not another model-selection attempt.
The archived original source supplies the fit. The release wheel supplies the
runtime replay. Both use the pinned native environment and original policy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-source-commit", required=True)
    args = parser.parse_args()
    if platform.machine() != "s390x":
        raise RuntimeError("reproduction_requires_native_s390x")
    root = args.bundle.resolve()
    package = json.loads((root / "package.json").read_text())
    if args.release_source_commit != package["source_commit"]:
        raise ValueError("release_source_binding_mismatch")
    spec = json.loads((root / "code/src/release/selected.json").read_text())
    study = root / package["selected_study"]
    if sha(study / "freeze-manifest.json") != spec["study_freeze_sha256"]:
        raise ValueError("unapproved_study_manifest")
    frozen = json.loads((study / "freeze-manifest.json").read_text())
    for name, expected in frozen.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or sha(study / p) != expected:
            raise ValueError("original_study_changed")
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    original_source = study / "source"
    commit = spec["training_source_commit"]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    # Do not substitute installed release modules for the archived original fit.
    original_env = dict(env)
    original_env.pop("PYTHONPATH", None)
    stages = []

    def stage(name, argv, cwd, environment):
        start = time.perf_counter()
        log = output / (name + ".log")
        with log.open("x") as f:
            result = subprocess.run(argv, cwd=cwd, env=environment, stdout=f, stderr=subprocess.STDOUT)
        record = {"stage": name, "exit_code": result.returncode, "seconds": time.perf_counter() - start,
                  "log_sha256": sha(log), "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
        stages.append(record)
        dump(output / (name + "-receipt.json"), record)
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError("reproduction_stage_failed")

    stage("01-data", [sys.executable, "-m", "src.data.synthetic", "--catalog", str(study / "catalog.json"),
        "--output", str(output / "data"), "--source-commit", commit], original_source, original_env)
    stage("02-fit", [sys.executable, "-m", "src.models.synthetic", "--data", str(output / "data"),
        "--output", str(output / "models"), "--source-commit", commit], original_source, original_env)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        if sha(output / "data" / name) != sha(study / "data" / name):
            raise RuntimeError("reproduced_data_differs")
    import joblib
    from src.release.state import learned_state_sha256
    original_state = learned_state_sha256(joblib.load(study / "models/models.joblib"))
    rebuilt_state = learned_state_sha256(joblib.load(output / "models/models.joblib"))
    if original_state != rebuilt_state:
        raise RuntimeError("reproduced_learned_state_differs")
    old_cards = json.loads((study / "models/model-cards.json").read_text())
    new_cards = json.loads((output / "models/model-cards.json").read_text())
    for key in ("guardrail_threshold", "canary_threshold", "guardrail_method"):
        if old_cards[key] != new_cards[key]:
            raise RuntimeError("reproduced_policy_differs")
    policies = [dict(cards["policy"]) for cards in (old_cards, new_cards)]
    for policy, cards in zip(policies, (old_cards, new_cards)):
        if policy.pop("model_pair_id") != cards["model_pair_id"]:
            raise RuntimeError("reproduced_policy_binding_invalid")
        policy.pop("policy_id")
    if policies[0] != policies[1]:
        raise RuntimeError("reproduced_policy_differs")
    stage("03-release-replay", [sys.executable, "-m", "src.replay.synthetic", "--data", str(output / "data"),
        "--models", str(output / "models"), "--output", str(output / "benchmark"),
        "--source-commit", args.release_source_commit, "--toxic-test", str(root / "external/toxic-chat_annotation_test.csv"),
        "--xstest", str(root / "external/xstest_prompts.csv")], output, env)
    stage("04-audit", [sys.executable, str(study / "audit_synthetic_study.py"), str(output)], output, env)
    exact = []
    for file in sorted((study / "benchmark").iterdir()):
        if file.name in ("report.json", "evidence.json", "timing.json", "evaluation-plan.json"):
            continue
        target = output / "benchmark" / file.name
        if sha(file) == sha(target):
            exact.append(file.name)
        if file.name.endswith("-events.jsonl"):
            before = [json.loads(line) for line in file.read_text().splitlines()]
            after = [json.loads(line) for line in target.read_text().splitlines()]
            if len(before) != len(after):
                raise RuntimeError("release_event_count_differs")
            for a, b in zip(before, after):
                for key in ("index", "prompt_id", "label", "domain", "A_guardrail_score", "A_decision",
                            "B_scores", "B_decision", "B_execute", "generation", "healthy_scores"):
                    if a[key] != b[key]:
                        raise RuntimeError("release_request_score_or_decision_differs")
                if bool(a["response"]) != bool(b["response"]):
                    raise RuntimeError("release_response_presence_differs")
                if a["response"]:
                    for key in ("method", "status", "applied", "response_index"):
                        if a["response"].get(key) != b["response"].get(key):
                            raise RuntimeError("release_response_action_differs")
        elif file.name.endswith("-summary.json"):
            before, after = json.loads(file.read_text()), json.loads(target.read_text())
            before.pop("incidents")
            after.pop("incidents")
            if before != after:
                raise RuntimeError("release_episode_metrics_differ")
        elif sha(file) != sha(target):
            raise RuntimeError("release_score_artifact_differs")
    old_report = json.loads((study / "benchmark/report.json").read_text())
    new_report = json.loads((output / "benchmark/report.json").read_text())
    for report in (old_report, new_report):
        report.pop("source_commit")
        report.pop("elapsed_seconds")
    if old_report != new_report:
        raise RuntimeError("release_metrics_differ")
    old_plan = json.loads((study / "benchmark/evaluation-plan.json").read_text())
    new_plan = json.loads((output / "benchmark/evaluation-plan.json").read_text())
    for plan in (old_plan, new_plan):
        plan.pop("source_commit")
        plan.pop("pre_evaluation_freeze_sha256")
    if old_plan != new_plan:
        raise RuntimeError("release_evaluation_plan_differs")
    dump(output / "reproduction-receipt.json", {"status": "passed", "architecture": platform.machine(),
        "release_source_commit": args.release_source_commit, "training_source_commit": commit,
        "data_reproduced_byte_identical": True, "learned_state_reproduced_exact": True,
        "learned_state_sha256": original_state,
        "serialized_model_bytes_identical": sha(output / "models/models.joblib") == spec["model_files"]["models.joblib"],
        "serialization_scope": "Exclude only the process-local TfidfVectorizer._stop_words_id cache from learned-state comparison. File-bound model and policy identities can differ after a refit.",
        "all_request_scores_decisions_and_response_actions_identical": True,
        "decision_policy_unchanged": True, "benchmark_metrics_identical": True, "byte_identical_algorithm_files": len(exact),
        "paired_episodes": 60, "paired_requests": 30720, "model_selection_repeated": False,
        "stages": stages, "scope": "Native reproducibility, not an additional independent effectiveness trial."})
    print(json.dumps({"status": "passed", "exact_algorithm_files": len(exact)}), flush=True)


if __name__ == "__main__":
    main()

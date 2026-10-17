# AGENTS.md — Global context for any AI agent in this repository

## What this project is

Safety Monitor Integrity Watch. A watcher for deployed AI safety filters ("guardrails"). A canary model scores the same traffic in parallel. A monitor measures the disagreement. Drift triggers an alarm. A human approves a conformal recalibration. Every step writes hash-chained evidence. Target platform: IBM LinuxONE (s390x).

## What this project is not

- Not a new guardrail. Not a prompt-injection firewall. Not a content moderator.
- Not an LLM pipeline. The decision path is deterministic: statistics plus small classifiers.
- Not a production claim. The evaluation is a paired replay study on public data.

## Environment

- Server: LinuxONE Community Cloud VM, Ubuntu 24.04, s390x, 3 vCPU, 8 GB RAM, 50 GB disk.
- Workspace on the VM: `~/safety-watch/`. Docs: `~/safety-watch/docs/SETUP.md`.
- Container image: `safety-watch-env:2026-10-16` (PyTorch CPU, Transformers, scikit-learn 1.9.0, FastAPI, JupyterLab). Private Jupyter, token-protected, SSH tunnel only.
- The server IP and all keys stay out of this repo. Ask the team lead.

## Data

All datasets are on the VM under `~/safety-watch/data/`. Details and hashes: `docs/DATA.md`.

- Labels: ToxicChat 0124 (real user prompts; human-annotated headline subset).
- Fresh attacks: WAInjectBench (2026).
- Published attack strings: JBB-Behaviors, HarmBench behaviors, StrongREJECT.
- WhyLabs components were evaluated and excluded: their Python stack has no s390x support (`whylogs-sketching` ships no s390x wheel).

## Hard constraints

1. No application code before the event window. Tooling, data prep, and environment setup are allowed before.
2. No synthetic data or augmentation anywhere. Public attack strings are data, never executed.
3. Official train/test splits stay intact. Headline metrics come from the human-annotated subset.
4. No secrets, tokens, keys, raw harmful samples, or server addresses in this repository.
5. The canary must come from a different model family than the guardrail. Measure error correlation; report it.
6. Detection proves change, not correctness. Claims follow evidence labels: verified, research, proposed, unverified.

## The method in 8 lines

1. Train guardrail (char n-grams + logistic regression) and canary (word n-grams + logistic regression) on ToxicChat train.
2. Freeze both. Record versions and digests.
3. Replay seeded streams: benign phase, shift point T, attack phase.
4. Monitor: sliding-window KS on score disagreement, scan-martingale alarm.
5. Alarm opens an incident with evidence.
6. Human approves repair. Conformal recalibration (projection to <=32 dims) runs.
7. Verify after repair. Compare Run A (guardrail only) vs Run B (guardrail + watcher), paired seeds.
8. Scorecard: detection latency, detection rate, false alarms on negative control, pass-through, coverage before/after.

## Where to go next

- Your role file: `agents/<role>.md`.
- The task list: `docs/PLAN.md`.
- Locked decisions: `docs/DECISIONS.md`.
- Red-team answers: `docs/REDTEAM.md`.

## Definition of done for any change

It runs on the VM inside the container. It is reproducible from a fixed seed. It writes evidence. It adds no dependency without an s390x wheel check. It updates docs if behavior changes.

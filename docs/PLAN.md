# Work plan and current status

## 1. Baseline classifiers — complete

Both frozen classifiers run in the native LinuxONE container. Their measured error correlation is approximately 0.81. These are baseline assets, not a qualified production guardrail/canary pair.

Read `docs/MODELS.md` before model changes. Do not tune against the published official classifier test result.

## 2. Prompt pools — complete

The verified native build contains 3,143 benign-routing and 1,209 attack-routing normalized prompt groups. Every retained source link and exact text variant passes the private audit.

Reference, calibration, and evaluation purposes remain separate. Read `docs/PROMPT-POOLS.md` for exact counts, source limits, and file paths.

## 3. Score disagreement monitor — native qualification pending

The source and local tests implement signed score gaps, empirical finite-episode KS thresholds, strict paired telemetry, numeric change evidence, and checkpoint restore.

Next checks:

- Run the full fixture suite inside LinuxONE.
- Score only the reserved benign reference and calibration inputs with frozen models.
- Freeze the empirical monitor configuration before evaluation.
- Verify native functional execution and restart parity.
- Confirm that original classifier artifacts remain unchanged.

Read `docs/MONITOR.md`. A working detector does not establish safety effectiveness.

## 4. Replay harness and complete interface contract — planned

Specify stream construction, order, seed, cohort, event identity, failure scenario, and run-log format. The paired-score input contract already exists in `docs/MONITOR.md`.

Do not call constructed replay order a production trace. Do not execute dataset instructions.

## 5. Canary and detector qualification — planned

Use reserved development data before final evaluation. Measure useful complementary errors and relationship changes without assuming independence or stronger canary recall.

Controls include these cases:

- Stationary negative control.
- Harmless traffic change.
- Harmful change with independent labels.
- Common-mode and score-preserving failures.
- Broken or mismatched telemetry.
- Long-duration false alarms and episode-budget limits.

Do not promise zero alarms, a fixed detection delay, or detection of every failure.

## 6. Paired product benchmark — planned

Compare identical frozen streams and failure scenarios with and without the watcher. Report observed detection and safety outcomes, including failures.

A passive watcher cannot improve raw guardrail recall. Any improvement needs a specified and evaluated response.

## 7. Approval and response — planned

Define action authority, labels, approval, verification, rollback, and independent final evidence. Do not promise that conformal calibration repairs classifier weights or restores recall.

## 8. Dashboard and submission — planned

The dashboard consumes numeric evidence, not raw harmful prompts. The demo shows actual native LinuxONE execution and measured results. Proposed enterprise targets remain acceptance goals, not demonstrated capacity or certification.

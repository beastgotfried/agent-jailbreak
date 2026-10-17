# Work plan and current status

## 1. Baseline classifiers — complete, not production-qualified

Both frozen classifiers run natively on LinuxONE. Their error correlation is approximately 0.81. Read `docs/MODELS.md` before changes.

Do not tune against the published official classifier test. The current pair does not demonstrate an acceptable coverage/review trade-off.

## 2. Prompt pools — complete

The verified native build contains 3,143 benign-routing and 1,209 attack-routing normalized groups. Every retained source link and exact variant passes the private audit.

Reference, calibration, and evaluation purposes remain separate. Read `docs/PROMPT-POOLS.md` for counts and source limits.

## 3. Score disagreement monitor — native qualification complete

The native fixture suite, score adapter, finite-episode calibration, functional run, and checkpoint restore pass. The frozen KS threshold is 0.23258359872611467.

The functional check uses calibration data, not independent safety evidence. Read `docs/MONITOR.md`.

## 4. Conformal response — implemented; effectiveness goal unmet

The response changes class-conditional prediction-set quantiles, not model weights. It requires explicit offline approval and separate verification labels.

All eight measured candidates fail the fixed review-rate gate. No candidate applies. The implementation does not demonstrate an acceptable repair.

## 5. Deterministic agent replay — complete

The fixed-token, non-generative agent stub consumes source-labelled evaluation streams. Both arms share inputs, order, scores, and the initial policy.

A full native repeat produces 68 byte-identical artifacts. No dataset instruction becomes a tool or shell action.

## 6. Paired benchmark — complete for the fixed offline protocol

Five seeds across three scenarios produce 15 paired episodes. The watcher detects five of five simulated score faults and three of five attack-mixture shifts.

No coverage restoration or paired safety improvement occurs. Read `docs/BENCHMARK-PROTOCOL.md` and `docs/BENCHMARK-RESULTS.md`.

## 7. Negative controls — finite observed result complete

Five stationary benign-routing episodes produce no alarms. This observation does not prove permanent silence or universal false-alarm control.

Numeric unit controls also expose harmless shifts, common-mode blind spots, invalid telemetry, and finite-horizon limits. Long-duration and production qualification remain open.

## 8. Scorecards — complete

JSON, Markdown, and CSV reports include detection, misses, controls, response rejection, source-label coverage, review burden, and unsafe-label allows.

An audit recomputes quantiles, response gates, coverage, identities, and latency. Report regeneration reproduces the original bytes without model scoring.

## Remaining product work

- Qualify a useful guardrail/canary pair with fresh final verification.
- Demonstrate an accepted response under an explicit utility budget.
- Add production labels, approval, and an approved safe fallback.
- Integrate a real agent without raw prompt exposure.
- Qualify continuous multi-drift operation and enterprise capacity.
- Build the dashboard and prepare an honest native demo.
- Resolve dataset rights and event-submission compliance.

Do not relax gates against these evaluation results. Proposed enterprise targets remain goals, not demonstrated service levels or certification.

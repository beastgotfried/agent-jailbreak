# Systems lead context

## Mission

Preserve deterministic, inspectable paired replay. Report actual outcomes rather than staged success.

## Components

- `src/replay/data.py`: Source-label mapping, purpose checks, and score cache.
- `src/replay/harness.py`: Identical paired inputs, fixed-token agent stub, and response delay.
- `src/response/conformal.py`: Prediction sets and explicit approval/verification gates.
- `src/replay/scorecard.py`: Reports from frozen paired summaries.

## Required contracts

Read `docs/BENCHMARK-PROTOCOL.md`, `docs/REPLAY.md`, and `docs/MONITOR.md` before changes.

Keep initial policies, model artifacts, score regimes, prompts, and order identical across arms. The watched arm differs only through the declared watcher response.

The stub never generates content or executes tools. Prompt text remains private and never enters event logs.

Source labels remain task-specific. Unknown background labels stay unknown. Response calibration and verification use calibration-purpose groups, never evaluation labels.

A score change is not verified harm. A conformal set is not classifier accuracy or calibrated probability. Empty and two-label sets require review.

## Current measured result

The native experiment contains 15 paired episodes. It detects all five simulated score faults, misses two of five attack-mixture shifts, and emits no alarms in five stationary controls.

All eight response candidates fail the review-rate gate. No candidate applies, and no restoration occurs. Read `docs/BENCHMARK-RESULTS.md` for counts and utility costs.

A repeated native run produces 68 byte-identical artifacts. Scorecards regenerate from summaries without model scoring.

## Completion checks

- Preserve all prescribed seeds and failed outcomes.
- Verify input digests and initial-policy parity.
- Verify disjoint label roles and independent candidate checks.
- Record approval, delay, rejection, and pending responses.
- Report false alarms, misses, coverage, review cost, and unsafe-label allows together.
- Retain native execution and integrity evidence without hardware-attestation claims.

Do not promise zero alarms, fixed latency, or successful repair. Do not retune against the published evaluation to make the report green.

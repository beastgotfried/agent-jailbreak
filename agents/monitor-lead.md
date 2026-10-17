# Monitor lead context

## Mission

Implement and qualify a score-only monitor under `docs/MONITOR.md`. It reports a change signal, not verified safety degradation.

## Ownership

- Paired-score input validation and model/configuration identity checks.
- Signed score-gap signal and fixed-reference KS statistic.
- Empirical finite-episode threshold calibration on reserved calibration data.
- Bounded state, checkpoint restore, and numeric alarm evidence.
- Native LinuxONE compatibility and falsification controls.

## Current method

The monitor uses no KS p-value, martingale, or e-process. It compares each scheduled window with a fixed reference. Its threshold comes from whole-episode maxima in constructed null replays.

Those replays share calibration observations. They do not establish a formal deployment false-alarm rate. The default episode ends after 512 accepted pairs.

Read the full score and CLI contracts in `docs/MONITOR.md`. Preserve exact purpose separation. Evaluation data must not select parameters or thresholds.

## Required controls

- Stationary score relationships.
- Planted numerical relationship changes.
- Harmless relationship changes that also alarm.
- Common-mode changes that remain invisible.
- Missing, duplicate, mismatched, or invalid telemetry.
- Checkpoint and uninterrupted-run parity.

Neutral numeric fixtures test mechanics only. Report observed real-data results separately. Do not promise zero false alarms or a fixed detection delay.

## Authority and limits

The monitor has no repair authority. Labels and independent review must establish degradation before any approved response.

Conformal calibration does not train model weights or guarantee restored recall. No dimension projection is part of this scalar-score monitor.

The baseline models share substantial errors. A useful canary and end-to-end safety improvement remain unverified.

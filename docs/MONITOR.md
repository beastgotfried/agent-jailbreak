# Score disagreement monitor

## Scope

This module reads score pairs from a guardrail and a canary. It does not read prompt text or decide whether content is harmful.

An alarm means that their score relationship changes. It does not establish guardrail degradation, agent compromise, or successful repair.

Current baseline error correlation is approximately 0.81. A useful canary signal remains unverified.

## Signal and test

The signal is the signed difference: `guardrail_score - canary_score`, in [-1,1]. A signed difference retains direction that an absolute difference loses.

At each scheduled look, the detector compares a recent window with a fixed reference distribution. It uses the two-sided empirical CDF distance, or KS statistic. It does not use the KS p-value.

Defaults:

| Parameter | Value |
|---|---:|
| Window | 64 score pairs |
| Stride | 16 accepted pairs |
| Episode horizon | 512 accepted pairs |
| Maximum looks | 29 |
| Null replay runs | 200 |
| Empirical target alpha | 0.05 |
| Seed | 42 |

## Empirical threshold

Reference inputs must carry purpose `reference`. Null inputs must carry purpose `calibration`. Event IDs must be unique and disjoint across those inputs. Model-pair identities must agree.

Each null episode selects `horizon` calibration observations without replacement. Their order follows a seeded permutation. Each episode uses the same fixed reference and window schedule as the live detector.

The calibrator retains the maximum statistic from each null episode. Its threshold uses sorted maximum number `ceil((B+1)*(1-alpha))`, with one-based indexing. A statistic must be **strictly greater** than the threshold to produce an alarm.

Null episodes reuse a shared pool. They are not independent production trials. This quantile is an empirical design choice, not a conformal guarantee or an e-process.

The target is not a proven deployment false-alarm rate. It is also not the classifier's 5% validation FPR target.

The horizon bounds each episode. After 512 accepted pairs, the monitor refuses further pairs with `horizon_exhausted`. An operator must manage the next episode. Repeated episodes do not inherit an anytime-valid false-alarm guarantee.

## Score contract

Every paired JSONL record has these exact fields:

- `schema_version`: 1.
- `event_id`: Stable event identifier, at most 128 ASCII identifier characters.
- `stream_id`: Stable stream identifier with the same length restriction.
- `sequence`: Contiguous integer starting at zero for an episode.
- `guardrail_score`: Finite number in [0,1], not a Boolean.
- `canary_score`: Finite number in [0,1], not a Boolean.
- `model_pair_id`: SHA-256 identity of the trusted model-pair configuration.
- `purpose`: `reference`, `calibration`, `evaluation`, or `live`.

The model-pair identity binds both classifier hashes, both vectorizer hashes, normalization, label rule, frozen thresholds, and scikit-learn version.

The monitor rejects raw-content fields, malformed scores, duplicate IDs, sequence gaps, stream changes, and model identity changes. These are telemetry faults, not safety alarms. Rejected telemetry does not change monitor state.

The `pair` command also accepts two ordered side streams. Each side uses the shared identity fields plus `model_role` and `score` instead of paired scores. IDs, sequences, purposes, and model-pair identities must match. The join fails on a mismatch; it never guesses from arrival position. An upstream collector must reorder out-of-order events first.

## Output and state

Each accepted pair produces a numeric `monitor_update`. A scheduled look includes the statistic and threshold. The first threshold exceedance produces one latched `disagreement_shift` alarm per episode.

An alarm includes the window bounds, look number, model/configuration identities, and signed/absolute gap means. Its evidence status is `change_signal_only`. Its action authority is `none`.

Checkpoints retain accepted score events, not prompts. Their size is bounded by the configured horizon. Restore validates the checksum and configuration identity, then replays the accepted history to recover the same state. Supply only remaining input events after restore.

Checksums detect accidental changes. They do not authenticate artifacts, establish an append-only hash chain, or provide hardware attestation.

## Frozen-model adapter

`src.monitor.prepare_scores` reads the verified benign pool and existing trusted model artifacts. It scores only reference and calibration purposes. It skips evaluation records and never fits classifiers.

The adapter checks the benign pool hash and selected exact-text identities. The existing model loader checks versions and artifact hashes before deserialization. Joblib artifacts must still be trusted.

The private score manifest records source/model identities, counts, skipped evaluation rows, and all-zero feature counts. All-zero features are a quality warning, not evidence of benignness.

## Commands

Run these commands inside the native LinuxONE container from the reviewed source snapshot. The run directory must exist. Each new artifact path must be unused.

```bash
RUN=/home/ibm-user/shared/runs/monitor-20261017-01
MODELS=/home/ibm-user/shared/runs/classifiers-20261017-01/models
POOLS=/home/ibm-user/shared/data/pools/prompt-pools-20261017-01

python -m src.monitor.prepare_scores --models "$MODELS" --pools "$POOLS" --output-dir "$RUN/scores"
python -m src.monitor.cli calibrate --reference "$RUN/scores/reference.jsonl" --calibration "$RUN/scores/calibration.jsonl" --output "$RUN/calibration.json"
python -m src.monitor.cli run --config "$RUN/calibration.json" --input "$RUN/scores/calibration.jsonl" --limit 512 --output "$RUN/smoke.jsonl" --checkpoint-out "$RUN/checkpoint.json"
```

The last command is a functional check on calibration data. It is not an independent evaluation or a measured deployment false-alarm result.

## Required controls

Tests cover stationary scores, planted numerical shifts, harmless shifts, common-mode changes, invalid telemetry, and restart parity. Neutral numeric fixtures test mechanics only.

If both scores change while their difference stays fixed, this detector cannot see the change. A threshold-only policy change also leaves the numeric difference unchanged; a reported binding change requires a new reference/configuration.

Harmless traffic changes can alter disagreement and produce an alarm. Labels and separate verification must establish degradation. The independent evaluation pools remain reserved for a later benchmark.

No collector service, dashboard, production approval service, or action connector is part of this module.
The separate offline response and replay modules are described in `docs/REPLAY.md`.

## Native qualification — 2026-10-17

Executed source: `f7ce7c8129b80c9f37edfa7a903769b0238cbfaa`.
The native fixture suite reports 154 passes and four optional Mac converter skips.

The frozen score adapter processes 1,256 reference and 942 calibration prompts. It scores no evaluation prompts and does not fit either classifier.

The default empirical KS threshold is 0.23258359872611467. Nine of 200 null calibration maxima exceed it, or 4.5%. This is an in-sample empirical calibration result, not a deployment false-alarm estimate.

A 512-event calibration-data functional check completes 29 looks with no alarm. Checkpoint restore reproduces the exact state. These checks do not establish safety effectiveness.

Private native evidence: `~/safety-watch/runs/monitor-20261017-01/verification.json`.

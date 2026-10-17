# Method and evidence boundary

## Product

One guardrail gates traffic. A canary scores the same prompts in parallel. A separate monitor inspects their score relationship.

The monitor reports change, not correctness. It cannot verify harm without independent labels or another justified source of evidence.

## Current classifiers

Both models use ToxicChat and logistic regression with different TF-IDF views. Their 0–1 outputs are model scores, not validated harm severity or calibrated probabilities.

Each baseline detects 231 of 362 unsafe human-annotated official test prompts. Their human-cohort false-positive rates differ. Error correlation is approximately 0.81.

These results do not establish a useful canary or production safety. The frozen models remain baseline experiment assets.

## Prompt pools

The verified native build contains 3,143 benign-routing and 1,209 attack-routing unique normalized prompt groups. Source labels remain task-specific. Constructed behavior goals are not automatically successful jailbreaks.

Reference, calibration, and evaluation purposes remain disjoint by exact normalized text. Semantic and attack-family independence remain unproven. Read `docs/PROMPT-POOLS.md` before replay work.

## Current detector

The signal is the signed difference between guardrail and canary scores. A sliding-window KS statistic compares this signal with a fixed reference.

Calibration uses separate benign-routing score observations. Seeded null replays provide whole-episode maximum statistics. An empirical upper quantile supplies the threshold.

The design calibrates the maximum across scheduled looks rather than treating overlapping KS p-values as independent. Shared observations still prevent a formal anytime-valid claim.

Default scope: window 64, stride 16, horizon 512, 200 null replays, alpha 0.05, seed 42. The horizon is an explicit episode limit. Read `docs/MONITOR.md` for the exact contract.

Invalid pairing, sequence gaps, or model identity changes produce telemetry faults. A valid threshold exceedance produces an unverified change signal with no action authority.

## Limits

Harmless changes can alter disagreement. Common-mode failures can preserve disagreement and remain invisible. A static classifier does not forget merely because time passes.

A numerical shift fixture tests detector mechanics. It does not prove safety degradation detection. Native execution establishes compatibility for the tested path, not enterprise scale or hardware attestation.

## Offline response and paired results

The paired replay uses the same initial conformal policy and identical score streams in both arms. The watched arm adds the detector and a gated response. The fixed-token stub never generates content or executes tools.

The response changes class-conditional set quantiles, not model weights. Initial, response, and verification labels use disjoint calibration-purpose groups. Evaluation labels cannot approve policies. The approval and label delay remain offline simulations.

The native benchmark detects five of five simulated score faults and three of five attack-mixture shifts. Five stationary controls produce no alarms. Every response candidate fails the review-rate gate. No coverage restoration or paired safety improvement occurs.

Read `docs/REPLAY.md` and `docs/BENCHMARK-RESULTS.md`. Coverage is prediction-set membership, not classifier accuracy or probability calibration.

## Remaining product stages

Production approval, fresh labels, an approved safe fallback, real-agent integration, continuous operation, and enterprise capacity remain unbuilt or unqualified.

Future evaluation must retain harmless changes, shared failures, broken telemetry, and negative controls. Do not retune against the published final evaluation. Scalar score models do not require arbitrary dimension projection.

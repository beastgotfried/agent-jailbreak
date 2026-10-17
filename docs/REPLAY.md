# Conformal response and paired replay

The fixed experiment contract is `docs/BENCHMARK-PROTOCOL.md`. Read it before the source or scorecard.

## Components

- `src/response/conformal.py`: Class-conditional prediction sets and an approved empirical response gate.
- `src/replay/data.py`: Private pool validation, explicit source-label policy, disjoint label roles, and frozen score cache.
- `src/replay/harness.py`: Deterministic streams, fixed-token agent stub, paired arms, and response delay.
- `src/replay/cli.py`: Private native experiment artifacts.
- `src/replay/scorecard.py`: JSON, Markdown, and CSV results from frozen summaries.

## What repair means

The response changes conformal prediction-set quantiles. It does not update logistic-regression weights or calibrate sigmoid probabilities.

Each class uses a finite-sample nonconformity quantile. Nominal95% coverage requires suitable exchangeability. Drift and source-family dependence do not establish that condition.

A candidate must pass separate source-label verification: observed coverage>=90% and review rate<=75% in each class. It also needs explicit benchmark approval. A rejected candidate never replaces the previous policy.

Full or empty prediction sets require review. The agent stub does not emit an answer in either case. Review-all policies cannot pass the utility gate.

These checks do not guarantee that a response succeeds. The report retains rejection, pending, and missed-detection outcomes.

## Fair comparison

Arm A is the frozen guardrail policy without a watcher. Arm B adds the watcher and approved response. Both arms start with the SAME initial conformal policy and receive identical prompts, order, and effective scores.

Thus the comparison isolates watcher-triggered recalibration. It does not compare the original binary threshold against a different upfront policy. Published classifier-test metrics remain a separate baseline.

The fixed-token agent stub is not a real generative agent. Allowed benchmark requests measure gate pass-through, not actual generated harm.

## Data and labels

Only calibration-purpose groups supply initial, response, and candidate-verification labels. Their three roles are disjoint by prompt identity. Evaluation labels never choose or approve a policy.

The explicit policy maps harmful-request source labels. It does not treat WAInjectBench benign-injection labels as universal benign ground truth. Those observations may remain unlabelled stream background.

Coverage uses known source labels only. Event summaries also report review and decision counts for ALL traffic, including unlabelled background.

The label oracle and approval delay are offline simulations. No production human-review or identity service exists here.

## Scenarios and controls

The protocol uses five seeds and three512-event scenarios. Each arm receives the same initial state. The intended change is at index128.

A stationary benign-routing control measures actual alarm episodes. It cannot prove permanent silence on all benign traffic.

A natural attack-mixture shift measures detector behavior without corrupting scores. The canary may fail to expose that shift.

The silent-score-fault scenario multiplies guardrail output scores by0.25 after the shift. This is a declared positive-control fault, not a real observed attack or model update. The declared model identity intentionally stays unchanged.

The monitor latches at most one alarm per finite episode. A response can therefore fail if an earlier false alarm consumes that episode's opportunity. The results must expose this case.

## Native commands

Use a reviewed source snapshot at `$RUN/source`. The operator supplies `source-commit.txt` beside that directory. Model files and prompt pools stay private on LinuxONE.

```bash
RUN=/home/ibm-user/shared/runs/replay-20261017-01
cd "$RUN/source"
SOURCE_COMMIT="$(python -c 'from pathlib import Path; print(Path("../source-commit.txt").read_text().strip())')"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m src.replay.cli \
  --models /home/ibm-user/shared/runs/classifiers-20261017-01/models \
  --pools /home/ibm-user/shared/data/pools/prompt-pools-20261017-01 \
  --monitor-config /home/ibm-user/shared/runs/monitor-20261017-01/calibration.json \
  --output-dir "$RUN/results" --source-commit "$SOURCE_COMMIT" --approve-repair
```

The approval flag authorizes only the simulated benchmark response. It does not grant production action authority.

Regenerate reports without model scoring or response execution:

```bash
python -m src.replay.scorecard --results "$RUN/results" --output-dir "$RUN/regenerated-scorecard"
```

Every output path must be unused. The tool refuses a nonempty benchmark directory. Partial failure artifacts require inspection, not silent overwrite.

## Artifacts

The private result directory contains label splits, frozen score cache, per-arm numeric event logs, per-arm summaries, paired summaries, and an evidence manifest.

`scorecard.json` preserves counts, denominators, source-label coverage, unsafe-label allows, review costs, missed detections, and response gates. `scorecard.md` presents the main results. `scorecard.csv` supports comparisons.

A successful repeat check requires identical file bytes with the same source, inputs, configuration, seed, and native environment. Actual runtime receipts stay outside these deterministic algorithm artifacts.

Checksums do not provide signatures or hardware attestation. No raw prompt text or model vocabulary enters public output.

## Result interpretation

Coverage is prediction-set membership, not classifier accuracy. Both-label sets can increase coverage while increasing review costs.

Only report restoration when the unwatched final policy misses the empirical coverage floor and the accepted watched policy reaches it. If no baseline loss exists, report no loss to restore.

Keep every prescribed seed and scenario. Do not retune against evaluation outcomes or omit inconvenient alarms, rejections, or missed detections.

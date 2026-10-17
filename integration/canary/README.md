# Classifier and canary integration handoff

Status: verified native LinuxONE execution. Branch: pr3thiv-classifier-init.
This folder is the entry point for integrating the completed model-lead work.

## What is complete

- Guardrail: character 3-5 TF-IDF + logistic regression, trained, frozen and reload-verified.
- Canary: word 1-2 TF-IDF + logistic regression, fitting-only selection, trained,
  threshold-calibrated, frozen and reload-verified. Both share the logistic algorithm;
  different feature representations do not guarantee independent failures.
- Pinned ToxicChat provenance, official split preservation, exact duplicate audit,
  fixed seed, fitting/validation isolation, model cards and SHA-256 digests.
- Module APIs, host CLI, paired JSONL scoring and metadata-only hash-chain evidence.
- Error correlation and common misses measured on identical development records.
- Fixed 457-record human-benign reference scores for monitor integration.
- 100-prompt reload and file-based CLI smoke tests against repository artifact copies.
- Executed notebooks with outputs/plots and small code cells, without descriptive Markdown.
- Original guardrail weights, threshold, marker and guardrail card content preserved.
- The separate-calibration 80%-recall research model remains available but is not the
  active guardrail; it exceeded the human FPR budget. See ../../models/README.md.

## Files and prerequisites

| Purpose | Repository path |
|---|---|
| Guardrail modules | src/safety_watch/classifier/ |
| Canary, persistence and pair APIs | src/safety_watch/canary/ |
| VM host runner | scripts/run_classifier.sh |
| Canary CLI entry point | scripts/canary_classifier.py |
| Active pipelines and shared cards | models/{guardrail,canary}.joblib, model-cards.json |
| Digest-bound completion markers | models/{guardrail,canary}.freeze.json |
| Artifact and notebook hashes | models/manifest.json |
| Aggregate pair measurements | models/pair-report.json |
| Training measurements | models/canary-training-summary.json |
| Executed canary notebook | notebooks/06_word_canary.ipynb |
| Locked pre-fit protocol | docs/CANARY-PROTOCOL.md |
| Output schema | integration/canary/score-record.schema.json |
| Verification summary | integration/canary/verification.json |

Repository: ~/agent-jailbreak. Persistent runtime: ~/safety-watch.
Container mounts: /workspace and /runtime/{data,models,results,logs}.
Repository and runtime models contain the same verified pipeline bytes.
Merge the modules/scripts/artifacts into the integration branch; copying a notebook alone
is insufficient.

Use the existing native s390x safety-watch-lab image, resolved to its image digest by the
runner. Python 3.12.15, sklearn 1.9.0, numpy 2.0.2, scipy 1.13.1, pandas 3.0.6,
joblib 1.6.0. No dependency was added. The CPU-only runner uses two CPUs, 4 GiB,
network disabled and no image pulling. Source/data are read-only; runtime models/results/
logs are read-write. Training/verification needs the pinned VM dataset and manifest.
Scoring needs the frozen bundle and matching runtime; no dataset download.

## Commands from the VS Code remote terminal

~~~bash
cd ~/agent-jailbreak

# Fixed fitting-only search; writes a new development run.
bash scripts/run_classifier.sh canary train

# Export our verified run; identical existing exports are verified/reused.
bash scripts/run_classifier.sh canary export \
  --run /runtime/results/classifier-runs/canary-train-20261017T090644Z-6cfe5d16

# Score existing real input JSONL; output must not already exist.
bash scripts/run_classifier.sh canary score \
  --input /runtime/results/private-inputs.jsonl \
  --output /runtime/results/paired-scores.jsonl \
  --batch-size 256

# Verify development metrics, emit a reference and reload-smoke the pair.
bash scripts/run_classifier.sh canary evaluate

# Verify the GitHub copies instead of the default runtime copies.
bash scripts/run_classifier.sh canary evaluate --model-dir /workspace/models

bash scripts/run_classifier.sh test
~~~

Use canary score for BOTH scores. The older top-level score command remains guardrail-only.
For pipes use --input - --output -; stdout is JSONL, the summary is on stderr.
The runner keeps stdin open for streams.

Training does not replace the frozen pair. Export rejects a different model; explicit
versioning and evaluation review are required. Partial exports retain a lock for review.
Hashes detect changes relative to preserved metadata, not external signatures.
Load trusted joblib files only.

## Python API and JSONL contract

Inside the pinned container, set PYTHONPATH=/workspace/src:

~~~python
from safety_watch.canary import load_pair, joint_records

guardrail, canary = load_pair("/runtime/models")
records = joint_records(guardrail, canary, prompt_ids, texts, splits)
~~~

prompt_ids/texts are equally sized sequences of real records. splits is optional and
defaults to live. Empty input yields empty output. Each input JSONL object has nonempty
string prompt_id, nonempty string user_input, and optional nonempty string split.
Prompt text is data, never executed or echoed.

| Output field | Meaning |
|---|---|
| schema_version | 1 |
| prompt_id | Original source ID |
| guardrail_score, canary_score | Unsafe estimates in [0,1] |
| timestamp | UTC scoring time |
| split | Input routing label, default live |
| guardrail_decision | allow/refuse using only guardrail_score >= guardrail_threshold |
| guardrail_threshold, canary_threshold | Saved thresholds; canary threshold is diagnostic |
| guardrail_sha256, canary_sha256 | Exact fitted pipeline digests |

See score-record.schema.json. Canary never gates. Weighted logistic scores are not validated
harm severity or guaranteed calibrated probabilities. Outputs have no prompt text or
ground-truth labels. split is routing metadata, not proof of data isolation.

The monitor consumes both score columns and must lock its signed/absolute disagreement
definition, KS window and alarm calibration. The notebook's signed-difference histogram is
visualization only. No monitor implementation is supplied here.

## Measured development results

196 train/test exact-text overlaps and 95 additional training duplicates excluded.
4791 eligible; 3832 fitting, 959 validation; seed 42.
Partition SHA-256: cd5e3febd5e3025cf905214a1369344f5f22b9613fe35a2c577842eacf037c6d.
Headline subset: 527 human-annotated records, 457 benign and 70 unsafe.

| Metric | Guardrail | Canary |
|---|---|---|
| Unsafe recall | 74.29% (52/70) | 71.43% (50/70) |
| Precision | 70.27% | 69.44% |
| False-positive rate | 4.81% (22/457) | 4.81% (22/457) |
| F1 | 0.7222 | 0.7042 |
| Average precision | 0.7881 | 0.7561 |
| Threshold | 0.3590141404799981 | 0.5656831460118261 |

Canary: C=3, balanced classes, 16002 features, 752048 bytes.
12 candidates/36 inner-fold fits took about 1.11 s; selected fit about 0.22 s.
Times vary by load; training-process peak RSS about 218.35 MiB.

Human error-indicator Pearson/phi correlation: 0.7622478434108937.
Both wrong 32, guardrail-only wrong 8, canary-only wrong 10, both correct 477.
Predictions disagree on 18/527 (3.42%). Both miss 16/70 unsafe prompts.
Canary flags 2 guardrail unsafe misses; guardrail flags 4 canary unsafe misses.
Supplementary full-validation error correlation: 0.7397738005083415.

Correlation is high and common-mode failure substantial. We have not established that
disagreement detects harmful drift. Canary is not a guardrail replacement.
These rows were held out from fitting but used for threshold calibration and exposed to
prior research. Results are development/calibration, not independent test or production
evidence. No official-test labels were read in this work; historical project-wide exposure
is unresolved.

## Private reference and evidence

Training:
~/safety-watch/results/classifier-runs/canary-train-20261017T090644Z-6cfe5d16/

Published pair report's reference:
~/safety-watch/results/classifier-runs/pair-evaluate-20261017T090800Z-c516b2c3/benign-reference-scores.jsonl

Reference: 457 human-benign validation records sorted by prompt_id.
SHA-256: ae2f76aace62a35c5a81cef313c4ff4a45705eb04c8d7071663cce573f8926b9.
Repeated evaluations preserve reference bytes using the fixed model-export timestamp.
This also calibrated thresholds; it is not an independent null-control stream. Monitor
lead must calibrate/audit false alarms on agreed separate negative-control replay.
New pair-evaluate operations write their own reference path.

Operation evidence: ~/safety-watch/logs/classifier-modular/.
Operation artifacts: ~/safety-watch/results/classifier-runs/.
Execution archive: ~/safety-watch/results/notebook-executions/word-canary.executed.ipynb.
Raw inputs and record-level scores/reference membership stay private on the VM.
Git contains fitted pipelines and aggregate results, not source prompt rows.
The original guardrail card retains historical guardrail-only limitations; the canary
card and pair report describe current measurements.

## Verification and commit checkpoints

18 native tests passed (9 guardrail + 9 canary cases) in 38.61 s, covering CV isolation,
word family, threshold FPR, refit/reload, export integrity, guardrail preservation,
idempotent export, partial-export rejection, pre-deserialization tamper rejection,
paired JSONL decisions/schema, error contingency and deterministic reference.
The host-runner CLI smoke scored 100 real validation prompts in batches of 17 and matched
both repository pipelines to absolute tolerance 1e-12. Saved pipelines reproduce all 959
paired records. The 22-cell notebook executed natively without errors and includes its plot.

- dd1de79: lock protocol before fitting.
- c0f15b7: implement training, persistence, pair scoring and tests.
- 63a9146: wire canary CLI into native runner.
- 30aa482: freeze canary and publish verified paired measurements.
- Final handoff commit: this folder, executed notebook, manifest and documentation.

## Remaining integration work

- Systems + model: align prompt pools with frozen partitions and source-task labels.
  Injection labels are not universal harm labels.
- All: agree replay/log/alarm/repair contracts; this folder specifies model scores only.
- Monitor: lock disagreement, calibrate KS/martingale and negative controls; test whether
  the signal tracks independently measured harm.
- Systems: paired seeded A/B replay, shift point, deterministic stub and scorecard.
- Monitor + model: approved conformal recalibration, holdout coverage checks and rollback.
- Model: reconcile historical official-test access before agreeing final evaluation.
- Platform/demo: live LinuxONE dashboard, full-loop recording, video and submission.

Detection proves change, not correctness. Classifier measurements do not close REDTEAM
monitor falsification criteria. No event-time watcher/repair code was added here.

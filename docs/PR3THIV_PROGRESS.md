# Pr3thiv — Classifier Progress

Updated: 17 October 2026  
Branch: pr3thiv-classifier-init  
Scope: my guardrail classifier, word canary, evaluation, saved artifacts and model integration.

## Status

My classifier implementation is complete and verified on native IBM LinuxONE.
Both fitted models, modular code, executed notebooks, metrics and integration documentation
are committed and published on this branch. Final integration support and evaluation
closeout remain; these results do not establish production safety or watcher effectiveness.

## What I implemented

- Built the initial guardrail notebook and reproduced seeded training on LinuxONE.
- Trained character 3-5 n-gram TF-IDF + logistic regression for the guardrail.
- Improved guardrail unsafe recall from 64.29% to 74.29% at the same 4.81% human
  development false-positive rate using a fixed fitting-only model search.
- Ran a separate-calibration optimization experiment and retained the existing guardrail
  because the candidate failed the promotion criteria.
- Extracted shared classifier code into Python modules and train/export/score/evaluate scripts.
- Saved complete pipelines: vocabulary, IDF, coefficients, intercept and model thresholds.
- Added versioned model cards, SHA-256 integrity checks, freeze markers, verified reload
  and protection against silent replacement.
- Trained the word 1-2 n-gram canary using the same verified fitting/validation membership.
- Measured guardrail/canary error correlation and shared unsafe misses.
- Implemented paired scoring with both scores, model digests, timestamps and routing split.
- Kept allow/refuse decisions dependent only on the guardrail; the canary supplies a reference signal.
- Prepared a deterministic 457-record human-benign development reference score file.
- Published reviewed model weights and restored actual notebook outputs/plots to GitHub.
- Split notebook code into smaller cells without explanatory Markdown cells.
- Added native integration tests, CLI smoke verification and an integration handoff.

## My models

| Property | Guardrail | Canary |
|---|---|---|
| Feature representation | Character TF-IDF, n-grams 3-5 | Word TF-IDF, n-grams 1-2 |
| Classifier | Logistic regression | Logistic regression |
| Selected C | 100 | 3 |
| Class weights | Balanced | Balanced |
| Learned features | 85,327 | 16,002 |
| Saved threshold | 0.3590141404799981 | 0.5656831460118261 |
| Purpose | Allow/refuse decision | Parallel reference signal |

Different feature representations do not imply independent errors; both use logistic regression.

## Data and training

- ToxicChat 0124 official train; unsafe = toxicity OR jailbreaking.
- Verified dataset revision and file digests; official source files remain intact.
- Excluded 196 training rows with exact test-text overlaps and 95 additional training duplicates.
- 4,791 eligible records: 3,832 fitting and 959 development validation, seed 42.
- Headline metrics use 527 human-annotated validation records: 457 benign and 70 unsafe.
- Fitting-only selection: 12 C/class-weight candidates, three stratified folds.
  Vocabulary and IDF are learned inside each fitting fold.
- Thresholds target <=5% empirical FPR on human-benign validation; no refit afterward.
- No synthetic prompts, augmentation, new dependencies, transformer decision path or off-platform training.

## Measured results

| Human development metric | Guardrail | Canary |
|---|---|---|
| Unsafe recall | 74.29% (52/70) | 71.43% (50/70) |
| Unsafe precision | 70.27% | 69.44% |
| False-positive rate | 4.81% (22/457) | 4.81% (22/457) |
| F1 | 0.7222 | 0.7042 |
| Average precision | 0.7881 | 0.7561 |
| Accuracy | 92.41% | 92.03% |

Accuracy is dominated by benign examples and is not the primary safety metric.

Human error-indicator Pearson/phi correlation: **0.76225**.
Both models miss **16 of 70 unsafe examples**. The canary flags two unsafe guardrail misses;
the guardrail flags four unsafe canary misses. Predictions disagree on 18/527 records (3.42%).
This is high correlation and limited demonstrated complementarity.

The earlier **80% result was unsafe recall, not accuracy**. That separately calibrated
research reference caught 56/70 unsafe examples with 5.91% human FPR. Its candidate
achieved 78.57% recall with 5.69% FPR. Both exceeded the 5% evaluation budget; neither
replaced the active guardrail. Their artifacts remain under models/research/separate-calibration/.

These are development/calibration results: records were held out from fitting but used
for threshold calibration and exposed to prior research. They are not independent final-test
or production evidence. This work did not read official-test labels; historical
project-wide official-test exposure is unresolved.

## My working directories and deliverables

VM repository: /home/linux1/agent-jailbreak/  
Persistent runtime: /home/linux1/safety-watch/  
Container source: /workspace/  
Container runtime: /runtime/{data,models,results,logs}/

Execution: native s390x CPU, pinned Python 3.12.15 and scikit-learn 1.9.0 container.
The runner uses two CPUs, 4 GiB, no network and no image pulling.

| Deliverable | Repository location |
|---|---|
| Guardrail modules | src/safety_watch/classifier/ |
| Canary and pair modules | src/safety_watch/canary/ |
| Native host runner | scripts/run_classifier.sh |
| Guardrail entry points | scripts/{train,export,score,evaluate}_classifier.py |
| Canary entry point | scripts/canary_classifier.py |
| Frozen pipelines | models/guardrail.joblib and models/canary.joblib |
| Cards, markers and hashes | models/model-cards.json, *.freeze.json, manifest.json |
| Aggregate results | models/pair-report.json and canary-training-summary.json |
| Initial notebook | notebooks/01_initial_guardrail.ipynb |
| Optimization notebook | notebooks/05_guardrail_optimization.ipynb |
| Canary notebook | notebooks/06_word_canary.ipynb |
| Tests | tests/test_classifier.py and tests/test_canary.py |
| Integration guide and score schema | integration/canary/ |

Private datasets, prompt inputs, record-level scores and evidence logs stay under the
runtime workspace. They are not included in this progress file or committed to Git.

## Verification completed

- 18 native tests passed: nine guardrail and nine canary test cases.
- Repeated fitting and reload matched all 959 validation scores to absolute tolerance 1e-12.
- Both repository model copies reproduced their saved development measurements.
- A real file-based CLI smoke test scored 100 held-out-from-fitting prompts in batches of 17.
- Both score columns matched the repository pipelines; every output record passed the published schema.
- Guardrail weights, threshold, marker and original guardrail card were preserved during canary export.
- The benign reference is byte-reproducible across repeated pair evaluations.
- The canary notebook executed successfully with 22 small code cells, outputs and a plot.

## Commands for my work

~~~bash
cd ~/agent-jailbreak
bash scripts/run_classifier.sh train
bash scripts/run_classifier.sh canary train
bash scripts/run_classifier.sh canary evaluate
bash scripts/run_classifier.sh canary score --input /runtime/results/private-inputs.jsonl --output /runtime/results/paired-scores.jsonl
bash scripts/run_classifier.sh test
~~~

The top-level score command is guardrail-only; canary score emits both models' scores.
Training writes a new development run and does not replace frozen weights.
Exact export commands, API usage and private reference paths are documented in
[integration/canary/README.md](../integration/canary/README.md).

## What remains in my scope

- Review existing prompt pools against the frozen classifier partitions and source-task labels.
- Agree the model-score contract with teammates and help connect paired scoring to their replay.
- Hand over and explain the benign reference, model thresholds and correlation limitations.
- Resolve historical official-test usage before agreeing any final evaluation.
- Support labeled-holdout and coverage checks when the repair work is integrated.

No further classifier code is currently required for this handoff. My implementation
being complete does not mean the full project or its end-to-end validation is complete.

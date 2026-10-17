# Initial guardrail notebook

Status: development baseline; guardrail only. No final test metrics have been computed.

Edit notebooks/01_initial_guardrail.ipynb in the VM repository using VS Code SSH FS.
Execute with the command below. SSH FS is a file editor, not a remote Python interpreter.

## Paths and environment

- VM repository: /home/linux1/agent-jailbreak.
- VM runtime workspace: /home/linux1/safety-watch.
- Execution-container repository: /workspace.
- Execution-container runtime directories: /runtime/{data,models,results,logs}.
- Python 3.12.15, native s390x, scikit-learn 1.9.0, existing pinned image.
- Existing lab stays running. No packages installed and no downloads.

## Execute from the remote VS Code SSH terminal

Run on the VM host, not inside the existing lab container.
Resolve the exact image ID instead of pulling a mutable tag.

~~~bash
mkdir -p /home/linux1/safety-watch/results/notebook-executions /home/linux1/safety-watch/logs
CLASSIFIER_IMAGE=$(sudo docker inspect --format '{{.Image}}' safety-watch-lab)
sudo docker run --rm --pull never --network none --cpus 2 --memory 4g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  --mount type=bind,src=/home/linux1/agent-jailbreak,dst=/workspace,readonly \
  --mount type=bind,src=/home/linux1/safety-watch/data,dst=/runtime/data,readonly \
  --mount type=bind,src=/home/linux1/safety-watch/models,dst=/runtime/models \
  --mount type=bind,src=/home/linux1/safety-watch/results,dst=/runtime/results \
  --mount type=bind,src=/home/linux1/safety-watch/logs,dst=/runtime/logs \
  --workdir /workspace --entrypoint /opt/venv/bin/jupyter "$CLASSIFIER_IMAGE" \
  nbconvert --to notebook --execute notebooks/01_initial_guardrail.ipynb \
  --ExecutePreprocessor.timeout=900 \
  --output initial-guardrail-development.executed.ipynb \
  --output-dir /runtime/results/notebook-executions
~~~

Each execution creates its own result directory and evidence file. The executed notebook
path is replaced on rerun; individual run artifacts remain separate.
Open results through vm-workspace in VS Code. Do not commit that link, data, models,
credentials, executed notebooks, or harmful text. The source mount is read-only at execution.

## Data and evaluation protocol

- Verify ToxicChat 0124 revision/hashes against the manifest.
- Input: user_input; target: toxicity OR jailbreaking.
- Preserve official files and test membership.
- Read test text only for required cross-boundary duplicate auditing; no test labels.
- Exclude exact test overlaps from training; deduplicate remaining text, preferring human
  annotation then stable ID. Fail on conflicting targets.
- Use both annotation types in eligible training. Reserve 20% validation, stratified by
  target and annotation status, seed 42.
- Fit character 3-5 TF-IDF and logistic regression exclusively on fitting records.
- Select the lowest threshold with <=5% empirical FPR on human-annotated benign validation.
- Headline metrics: human-annotated validation; full validation is supplementary.
- No refitting after threshold selection. A fixed 12-candidate, three-fold fitting-only search selects C/class weighting.
- Scores estimate target probabilities, not validated harm severity.

## Artifacts and safeguards

Development model, card, split audit, metrics, plots, validation-scores.jsonl, and summary:
results/guardrail/<run-id>/. Evidence: logs/classifier/<run-id>.jsonl.
No raw prompts or vocabulary features rendered/logged. Learned vocabulary stays in private
runtime model artifacts.

FREEZE_MODEL=False and RUN_FINAL_TEST=False are defaults. After review, freeze copies the
exact pipeline to models/guardrail.joblib and records the threshold in models/model-cards.json.
Never overwrite frozen models. Final test requires matching frozen model/data digests.
A started receipt precedes scoring; a completed receipt caches results. A different model
cannot silently reuse test. Failed attempts require evidence review before recovery.
Do not tune from final results.

Checks cover partition isolation/reproducibility, real-score ties, independent repeated
fitting, 100-record reload equivalence, convergence, and metadata tampering detection.
Hash chains are tamper-evident relative to their preserved head, not external signatures.

## Next phase

Review VM validation results, then extract data/training/threshold/metrics/persistence/evidence
functions into modules and CLI commands. Update the notebook to import them.
Canary, watcher, repair, prompt pools, and dashboard remain separate work.

## Original baseline development result

Executed on the native s390x VM with the pinned Python 3.12.15 image.
After 196 cross-boundary exclusions and 95 additional training duplicate removals:
4,791 eligible records; 3,832 fitting and 959 validation records.
The human-annotated validation subset contains 527 records (457 benign, 70 unsafe).

| Metric | Development validation |
|---|---|
| Unsafe recall | 64.29% (45/70) |
| Unsafe precision | 67.16% (45/67) |
| False-positive rate | 4.81% (22/457) |
| F1 | 0.6569 |
| Average precision | 0.7259 |
| Selected threshold | 0.18873807132779682 |
| Learned features | 85,327 |
| Pipeline size | 3,622,784 bytes |

The first measured fit took about 1.25 seconds; timing varies by load.
These are development/calibration results, not independent final-test or production metrics.
The frozen model and final test evaluation remain disabled.

## Improvement protocol

RUN_MODEL_SELECTION=True compares C=(0.3,1,3,10,30,100) with class_weight=None/balanced.
Only the fitting partition participates. Each fold learns TF-IDF independently.
Rank by pooled human recall at <=5% empirical inner-fold FPR, then mean average precision,
smaller C, and no weighting. Refit the selected settings on fitting data; calibrate the final
refusal threshold on unchanged outer validation. All candidate results and fold IDs are
recorded in model-selection.json. Official test labels remain locked.
Set RUN_MODEL_SELECTION=False to reproduce the original baseline.
Development gains are not final-test or production guarantees.

## Verified improvement on the unchanged validation split

The 12-candidate, three-fold search selected C=100 with balanced class weights using fitting
data only. The selected settings were then fit on all fitting records and the refusal
threshold calibrated on the existing validation split. Character 3-5 TF-IDF was unchanged.

| Human-annotated validation metric | Original | Improved |
|---|---|---|
| Unsafe recall | 64.29% (45/70) | 74.29% (52/70) |
| Unsafe precision | 67.16% | 70.27% |
| False-positive rate | 4.81% (22/457) | 4.81% (22/457) |
| False negatives | 25 | 18 |
| F1 | 0.6569 | 0.7222 |
| Average precision | 0.7259 | 0.7881 |
| Refusal threshold | 0.18873807132779682 | 0.3590141404799981 |

Search took about 7.43 seconds for 36 classifier fits in the initial measured run.
Selected-pipeline training took about 1.41 seconds. Both timings vary by load.
The pipeline remains about 3.6 MB with 85,327 learned features.
Balanced class weighting does not establish probability calibration.
The unchanged partition digest is cd5e3febd5e3025cf905214a1369344f5f22b9613fe35a2c577842eacf037c6d.

These are development/calibration results. Official test scoring and model freezing remain
disabled. No new dependencies, prompt augmentation, or external attack training data were used.

## Separate-calibration optimization

See [the locked optimization protocol](GUARDRAIL-OPTIMIZATION.md) and notebooks/05_guardrail_optimization.ipynb for the paired experiment. Its retrained reference uses a different fitting/calibration protocol.


The separate-calibration experiment completed and rejected its selected candidate for
promotion: 78.57% recall at 5.69% evaluation FPR versus the retrained reference's
80.00% recall at 5.91% FPR. Both exceeded the evaluation FPR budget. The existing notebook
and its original 74.29% development/calibration result remain unchanged; these protocols
must not be compared as if their thresholds and fitting sets were identical.

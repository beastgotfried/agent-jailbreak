# Guardrail optimization: locked protocol

Status: proposed experiment; protocol committed before evaluation. No result is claimed here.

## Objective and boundaries

Improve unsafe recall within a 5% false-positive budget using only character 3-5 TF-IDF and
logistic regression. Python 3.12.15, sklearn 1.9.0, CPU-only native s390x; no new dependencies,
transformers, synthetic prompts, augmentation, canary, model freezing, or official-test scoring.
Source: notebooks/05_guardrail_optimization.ipynb.

## Fixed partitions and baseline

Reproduce the previous official-train audit and partition digest:
cd5e3febd5e3025cf905214a1369344f5f22b9613fe35a2c577842eacf037c6d.
Preserve the original 959 evaluation IDs. Subdivide only the original 3,832 fitting records
80/20 with seed 20261017, stratified by unsafe target and annotation status.
Expected counts: 3,065 fitting, 767 calibration, 959 evaluation.

All official source files remain unchanged. Test text is read only for exact duplicate
exclusion; no official-test labels. Exact-text isolation is not near-duplicate isolation.
Reports about historical official-test use conflict with current receipts. Do not claim the
official test is untouched project-wide. Evaluation records have prior research exposure;
these are exploratory comparisons, not fresh independent confirmation.

Retrain reference and candidate on identical fitting rows. Reference: min_df=2, 100k feature
cap, sublinear TF, C=100, balanced classes, human annotation weight 1. The old 74.29% result
uses a different fitting/calibration protocol and is not directly comparable.

## Predeclared 48-candidate search

Character n-grams 3-5, lowercase, L2 normalization and logistic L2 regularization stay fixed.
Cross four TF-IDF profiles with C=(30,100,300), class_weight=(None,balanced), and human
annotation weight=(1,2). All nonhuman-annotated records retain weight 1.

| Profile | min_df | Feature cap | Sublinear TF |
|---|---|---|---|
| Reference features | 2 | 100,000 | Yes |
| Rare fragments | 1 | 100,000 | Yes |
| Larger vocabulary | 1 | 200,000 | Yes |
| Linear term frequency | 2 | 100,000 | No |

Three fitting-only stratified folds, seed 42, 144 classifier fit attempts.
Each fold fits vocabulary/IDF on its own inner fitting records.
Rank by pooled human recall at <=5% empirical inner-fold FPR, mean AP, fewer features,
smaller C, lower annotation weight, no class weighting, then candidate order.
Record failed candidates without exposing prompt-bearing exceptions.

## Calibration, comparison, and acceptance

Fit the selected candidate and reference on fitting only. Use only human-annotated benign
calibration scores to choose the lowest threshold meeting <=5% empirical calibration FPR.
Do not refit after choosing thresholds. Save model digests, thresholds and selection before
evaluation. Evaluate both on the same human-annotated evaluation cohort once.

A started claim precedes scoring; a completed receipt caches the comparison. Changed
protocol, partitions, models or thresholds cannot silently reuse that slot. Interrupted
attempts need evidence review. Do not retune after seeing comparison results.

Eligible for later review requires strictly higher recall than the retrained reference and
<=5% observed evaluation FPR, with reproducibility/resource checks passing. Otherwise retain
the reference. Even eligibility does not deploy or freeze a model.
Report precision, recall, FPR, F1, AP, counts, paired bootstrap recall-difference interval,
Wilson sampling intervals, warm serial latency, memory, and model sizes.
Intervals do not correct for historical model selection or data exposure.
Class weighting does not establish probability calibration.

## Execute from VS Code's remote SSH terminal

Run on the VM host. Existing lab is left running; image pulling and networking are disabled.

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
  nbconvert --to notebook --execute notebooks/05_guardrail_optimization.ipynb \
  --ExecutePreprocessor.timeout=900 \
  --output guardrail-optimization.executed.ipynb \
  --output-dir /runtime/results/notebook-executions
~~~

## Runtime artifacts and checks

Persistent results: /home/linux1/safety-watch/results/guardrail-optimization/<run-id>/.
Evidence: /home/linux1/safety-watch/logs/guardrail-optimization/<run-id>.jsonl.
Evaluation receipt: /home/linux1/safety-watch/results/guardrail-optimization-v1/.
Executed notebook: results/notebook-executions/guardrail-optimization.executed.ipynb.

Artifacts contain provenance, protocol, candidate ranking, partitions, model files, locked
thresholds, scorecard, paired comparison, latency, cards, and evidence. Raw prompts are never
rendered or logged. Model vocabularies and record-level artifacts remain private on the VM.
Source notebook stays output-free.

Verify partition isolation/reproducibility, fitting-only CV, positive record weights,
convergence, real-score ties, 100-record reload/refit equivalence per model, evidence tampering,
evaluation-cache and changed-artifact/interruption rejection. Use two CPUs and 4 GiB.

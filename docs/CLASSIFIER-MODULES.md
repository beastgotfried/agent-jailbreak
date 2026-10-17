# Modular classifier: scripts, saved model, and Python API

The verified character 3-5 TF-IDF + logistic-regression pipeline lives in a shared Python
package with CLI tools. No dependency was added. All execution stays in the pinned Python 3.12.15
native s390x image. The two-CPU, 4 GiB runner leaves the existing lab running and disables
networking/image pulls.

## Folder structure

~~~text
src/safety_watch/classifier/
  config.py       # selected model defaults and training settings
  runtime.py      # architecture and package compatibility checks
  data.py         # manifest verification, labels, deduplication, partitions
  model.py        # pipeline construction, fitting, numeric scoring
  selection.py    # optional fixed fitting-only model selection
  metrics.py      # threshold policy and metric denominators
  training.py     # development fitting and verification
  artifacts.py    # save/load, model-card and threshold integrity
  evaluation.py   # validation and explicitly gated final evaluation
  evidence.py     # hash-chained operation metadata
  io.py           # canonical JSON, digests, source provenance
  cli.py          # JSON/JSONL command interface
scripts/
  run_classifier.sh
  train_classifier.py
  export_classifier.py
  score_classifier.py
  evaluate_classifier.py
tests/
  test_classifier.py
notebooks/
  01_initial_guardrail.ipynb
~~~

Other research notebooks/modules remain separate.

## Run from the VS Code remote terminal

Start from the repository on the VM host, not from inside the existing lab container:

~~~bash
cd /home/linux1/agent-jailbreak
bash scripts/run_classifier.sh train
~~~

Default training uses the selected C=100, balanced-class configuration. It creates a new
private development run, reproduces the original seed-42 fit/validation split, selects the
threshold on validation, verifies reload and independent-refit scores, and writes evidence.
It never overwrites the saved model or evaluates official test.

~~~bash
bash scripts/run_classifier.sh train --search
bash scripts/run_classifier.sh test
~~~

The search flag repeats the fixed 12-candidate fitting-only selection. Tests use real pinned
ToxicChat records and the archived verified runs; no synthetic prompts are generated.
No package installation is required. Thin Python entrypoints locate the source package.
For a direct import inside the matching container, set PYTHONPATH=/workspace/src.

## Save the verified model without refitting

~~~bash
bash scripts/run_classifier.sh export \
  --run /runtime/results/guardrail/20261017T061152Z-a17252db
~~~

The exporter checks source model/partition/data digests, reproduces all 959 validation
scores/metrics, and copies the original pipeline bytes without refitting.
The model directory on the VM is /home/linux1/safety-watch/models, also accessible through
vm-workspace/models from the repository:

~~~text
models/
  guardrail.joblib       # entire TF-IDF + classifier pipeline
  model-cards.json      # guardrail metadata and saved threshold; preserves other entries
  guardrail.freeze.json # completion marker binding model/card digests and threshold
~~~

Repeated export of the same model/threshold verifies and reuses the existing bundle.
A different model/threshold cannot silently replace it. An interrupted partial export needs
evidence review. A frozen artifact means fixed weights/threshold, not production approval.
Load only trusted local joblib artifacts: hashes detect alteration relative to trusted
metadata, not a malicious replacement of the entire bundle.

An independently refitted equivalent model can have different pickle bytes. Reproducibility
is established by equivalent numeric scores, thresholds, partitions, and metrics; copying an
existing artifact preserves that artifact's exact digest.

## Score real prompt records

Input is JSONL with nonempty string fields prompt_id and user_input. Keep raw input files
private under the VM runtime workspace, outside Git. No prompt text is executed or echoed.

~~~bash
bash scripts/run_classifier.sh score \
  --input /runtime/results/private-inputs.jsonl \
  --output /runtime/results/guardrail-scores.jsonl
~~~

Output is JSONL: prompt_id, guardrail_score, decision, threshold, model_sha256, timestamp.
No canary_score is fabricated. A refusal is score >= saved threshold.
Use --input - / --output - for streams. The scoring summary goes to stderr so stdout remains
pure score JSONL. Existing output files are not overwritten.

The saved model is loaded once per command and scoring is batched (default 256).
Its score is an unsafe estimate, not validated harm severity or guaranteed calibrated
probability. Class weighting does not establish probability calibration.

## Validation and final-test boundary

~~~bash
bash scripts/run_classifier.sh evaluate \
  --run /runtime/results/guardrail/20261017T061152Z-a17252db
~~~

This evaluates the archived development validation split, checking source/data/split integrity.
The official test is never the default. It requires both --split test and --final-test and
uses the shared one-time started/completed receipts. A different artifact cannot silently
reuse final test. No official-test action was performed during modularization.
Historical reports about project-wide test use remain unresolved; do not advertise an
untouched independent test or production safety claim.

## Python API inside the pinned container

~~~python
from safety_watch.classifier import load_guardrail

guardrail = load_guardrail("/runtime/models")
scores = guardrail.score(texts)
records = guardrail.predict(prompt_ids, texts)
~~~

Pass sequences of real prompts and source IDs. Empty sequences return empty outputs.
Loading verifies architecture/package compatibility, card/threshold integrity and model hash
before deserializing the pipeline.

## Evidence and private artifacts

Development/export/evaluation/scoring operations write metadata-only hash chains under
logs/classifier-modular and result directories under results/classifier-runs.
No raw prompts or vocabulary fragments are logged. Record-level runtime artifacts remain
private. Reviewed fitted pipelines, including learned vocabularies, are now versioned under
models/ as requested; see models/README.md and manifest.json.


## Verified saved bundle

Saved on the VM without refitting the reference artifact:
source run 20261017T061152Z-a17252db; model digest
16cc50b1e16df16f937cc89c64eea164ed59f70275ccad09cc9303af1e9f2fce.
Threshold: 0.3590141404799981. Human-annotated development/calibration validation:
74.29% recall, 70.27% precision, 4.81% FPR, F1 0.7222, AP 0.7881.
This retains the existing reference; the separate-calibration models were not promoted. Original development files are
published under models/research/separate-calibration for reproducibility.

All nine native integration tests passed. They verified all 959 modular validation scores,
the original partition digest and threshold, exact source-artifact export, idempotent export,
different-model refusal, model/card tamper detection before pickle loading, streaming scoring,
the explicit final-test gate, and evidence integrity.

The refactored notebook executed successfully against the shared modules and reproduced
the original 959-row partition, threshold, scorecard, and numeric scores.
An additional CLI smoke test scored 100 real validation records in batches of 17, matched
the original scores/decisions to 1e-12, and echoed no raw prompt text. The validation CLI
also reproduced the original human/full-validation metrics. The persistent saved bundle
passed digest/card/threshold checks, and official-test receipts remain absent for this work.

## Repository copies and visible notebook outputs

Reviewed weights, cards, hashes and aggregate research results are tracked under models/.
See models/README.md for the retained and 80%-recall reference paths. Both tracked notebooks
include existing native execution outputs and plots alongside smaller code cells with
identical statement order;
descriptive Markdown remains removed. Original archive hashes and provenance are recorded
in models/manifest.json.

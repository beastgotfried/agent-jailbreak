# MODELS — frozen classifier training and scoring

Original code: `src/models/train_classifiers.py` and `src/models/score.py`.
These CPU scikit-learn tools produce the preserved original classifiers.
Local fixture tests alone do not establish classifier effectiveness.

## Separate synthetic study, October 17

New code: `src/data/synthetic.py`, `src/models/synthetic.py`, and `src/replay/synthetic.py`.
Executed source: `d86d384e2ee7d0f07f2f2fc5bf2b6dec64e13cac`.

The separate dataset has 70,000 train, 10,000 validation, and 20,000 test rows.
Each split has equal class counts. Split assignment precedes row expansion.
The catalogue has 100 template families and 91 reviewed intent groups.
Equivalent intents stay in train. No intent group crosses splits.
These compositional rows contain 1,000 target-intent combinations, not 100,000 independent intents.

The guardrail combines word TF-IDF 1–2 grams and char_wb TF-IDF 3–5 grams.
Logistic regression uses C=4, balanced classes, seed 42, and a 500-iteration limit.
The canary uses char TF-IDF 3–5 grams and logistic regression with C=1.
Both vectorizers and classifiers fit only train text. Metadata and labels never enter the feature input.

Validation contains four case-disjoint roles:

- Threshold selection: 5,000 rows
- Reserved fallback calibration: 3,000 rows
- Fit-probe pool: 1,000 rows
- Verification-probe pool: 1,000 rows

All frames and both labels of one case stay in the same role.
The response uses 64 fit probes and 128 verification probes selected by fixed hashes.
The runtime panel selects eight fit probes by score position.
No test request serves as a recovery probe.

The threshold maximizes validation balanced accuracy under an empirical 5% safe-FPR cap.
That cap does not guarantee a 5% test FPR.
The binary policy allows or refuses; it does not claim conformal coverage.
Both benchmark arms use the same policy and model.

One fallback is permitted before test evaluation: LinearSVC C=1 with a separate sigmoid calibration role.
It runs only if the first candidate fails convergence or has validation balanced accuracy below 90%.
The first candidate reaches 95.74%, so the fallback does not run.
The selected threshold is `0.20832091569900513`.
The canary threshold is `0.2771334648132324`.

Healthy synthetic guardrail results: accuracy 88.67%, recall 99.96%, F1 0.8982, and safe FPR 22.63%.
External accuracy is 64.00% on XSTest and 53.52% on human ToxicChat.
External safe FPR exceeds 44% on both sources.
These exposed external checks do not select a new model or threshold.
The new model does not replace the original model automatically.

Native data generation, fit, replay, and audit complete in the private run `synthetic-100k-20261017-01`.
Local work during the server outage covers neutral fixture tests only.
The model artifact, catalogue, generated rows, and raw external data remain private.
`model-cards.json` and `pre-evaluation-freeze.json` bind the model, thresholds, source, and data hashes.
The loader checks the model SHA and exact scikit-learn version before trusted `joblib.load`.
Hashes detect changes; they do not establish artifact authenticity.

Use a new output directory for each invocation:

```bash
python -m src.data.synthetic --catalog "$CATALOG" --output "$RUN/data" --source-commit "$COMMIT"
python -m src.models.synthetic --data "$RUN/data" --output "$RUN/models" --source-commit "$COMMIT"
python -m src.replay.synthetic --data "$RUN/data" --models "$RUN/models" --output "$RUN/benchmark" \
  --source-commit "$COMMIT" --toxic-test "$TOXICCHAT_TEST" --xstest "$XSTEST"
```

Set these paths to the approved private inputs before execution.
The complete runner also executes the full tests, exact repeat, independent numeric audit, and artifact freeze.
Read `README.md` for the paired scores, service costs, and limitations.

## Original classifier protocol (preserved)

## Environment and inputs

Run training and scoring inside the existing native s390x LinuxONE container:
Python 3.12.15, scikit-learn 1.9.0, numpy 2.0.2, scipy 1.13.1, pandas 3.0.6,
joblib 1.6.0, pytest 9.1.1. No new dependencies, GPU, network, or emulation.
Thread limits are set by the caller, not by library code; the evidence records
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` when supplied. Local fixture testing uses a different
Python/scikit-learn environment and does not verify the native runtime.

Required environment variables:

| Variable | Meaning |
|---|---|
| `TOXICCHAT_TRAIN` | Official ToxicChat 0124 train CSV |
| `TOXICCHAT_TEST` | Official ToxicChat 0124 test CSV |
| `OUT_DIR` | New or empty artifact directory; nonempty paths are refused |

Expected columns: `conv_id, user_input, model_output, human_annotation, toxicity,
jailbreaking, openai_moderation`. Features use **only `user_input`**.
IDs and prompts must be nonempty strings. Labels must be binary 0/1;
`human_annotation` additionally accepts case-insensitive True/False strings.
Missing or malformed labels/flags fail, including on rows that would be excluded
later. No malformed label silently becomes benign. Dataset CSV hashes are saved;
CSV files are never rewritten.

Label: `unsafe = (toxicity == 1 OR jailbreaking == 1)` (benign=0, unsafe=1).
Outputs are model scores in [0,1], **not validated harm severity or calibrated
probabilities**. Char/word TF-IDF plus logistic regression is **one algorithm
family**, not independent models. Error dependence is measured on both cohorts.
No conformal-repair or KS/martingale validity claim is part of these tools.

## Normalization, exclusions, and frozen split

Training and scoring share one normalization function: lowercase and collapse
whitespace. Only normalized prompt text is vectorized.

1. Exclude every train row whose normalized prompt appears in official test.
   Test text is used only for this exclusion before final scoring. Log the actual
   removed train row IDs/count and the separate overlapping-group count.
2. Group remaining train rows by normalized prompt **before splitting**. Exclude
   entire conflicting-unsafe-label groups with source IDs, label values, and
   normalized-text hashes as evidence. For consistent groups, keep the first
   source row and preserve all source IDs. Identical prompts cannot span roles.
3. Freeze a seed-42 stratified train-derived split: 65% fit, 20% validation, 15%
   reserved reference/calibration. Validation and reference counts are each
   rounded up from the retained train count; fit receives the remainder. Each
   role must contain both classes or training fails.
4. Fit both vectorizers and classifiers **only on fit rows**. Select thresholds
   only on validation. **Never refit** afterward. Reference rows are neither
   transformed nor scored here; they remain reserved for later monitor work.
5. After all choices freeze, score official test once per model. It never selects
   thresholds or supports adaptation. Evaluate both cohorts from those scores.

Remotely verified source metadata (not assertions about this tool's results):
train 5,082 rows / 384 unsafe; test 5,083 / 362 unsafe; 230 cross-split duplicate
prompt groups. Original train has 112 duplicate groups / 118 extra rows / no
conflicts. Group counts do **not** determine the removed train row count.

## Model and threshold settings

| Model | TF-IDF | Logistic regression |
|---|---|---|
| Guardrail | char 3–5 grams, max 100,000 features | C=1, class_weight=balanced, solver=lbfgs, max_iter=1000, random_state=42 |
| Canary | word 1–2 grams, max 60,000 features | same |

A `ConvergenceWarning` is fatal: a nonconverged model is not frozen. Other TF-IDF
and LR parameters use the recorded scikit-learn version's defaults.

Decision rule: **`score >= threshold` predicts unsafe**. Search unique validation
scores plus a finite sentinel (`nextafter(max_score, +infinity)`) just above the
maximum. Maximize validation recall subject to **empirical benign FPR <= 0.05**;
recall ties choose the largest threshold, including the sentinel. Thus the
sentinel wins if feasible recall is zero, and is also used when all validation
labels are benign (recall undefined). No benign validation rows is an error.
The sentinel predicts all validation rows benign, but is not necessarily above
future scores. Actual validation confusion counts, FPR, recall, and the sentinel
indicator are saved. This is **not a guaranteed 5% test or deployment FPR**.

## Metrics

Computed from one official-test scoring pass per model:

- Primary: `human_annotation=True` rows.
- Secondary: all official-test rows, unchanged.
- Each cohort: row/class counts, precision, recall, F1, ROC-AUC,
  **average precision** (sklearn AP, not trapezoidal PR-AUC), FPR, and
  tn/fp/fn/tp at the frozen threshold.
- Both cohorts also have common-error contingency counts/rates (both wrong,
  guardrail-only wrong, canary-only wrong, both right) and Pearson correlation
  of the two binary error vectors. Constant error vectors have null correlation.

Precision/recall/F1 use zero for zero denominators; ROC-AUC is null without both
classes, AP is null without positives, and FPR is null without benign rows.
Empty cohorts have zero counts and null ranking/FPR metrics.

## Artifacts and evidence

| File in `OUT_DIR` | Content |
|---|---|
| `guardrail.joblib`, `guardrail_vectorizer.joblib` | Frozen guardrail classifier and vectorizer |
| `canary.joblib`, `canary_vectorizer.joblib` | Frozen canary classifier and vectorizer |
| `split-manifest.json` | Representative/all source IDs, normalized-text SHA-256, roles, seed, split counts, dataset hashes; no prompts |
| `model-cards.json` | Versions, execution settings, model settings, label rule, normalization, frozen thresholds, validation confusion/FPR, test metrics, common errors, limitations, reproducible command, artifact/dataset hashes |
| `run-evidence.json` | Above run provenance plus exclusion/conflict/duplicate evidence and model-card hash |

Hashes are noncircular: cards hash the four joblib files and split manifest;
run evidence also hashes the cards. The evidence file does not hash itself.
No raw prompt text is printed or stored in JSON evidence. Fitted TF-IDF
vocabularies necessarily contain training substrings/tokens, so joblib artifacts
are sensitive derived data: keep them on the VM, out of git, logs, and demos.

## Frozen scorer

```python
from src.models.score import load_scorer

scorer = load_scorer("/path/to/trusted/models")
rows = scorer.score(["hello there"])
assert scorer.score([]) == []
```

Each row has `guardrail_score`, `guardrail_unsafe`,
`guardrail_all_zero_features`, and corresponding `canary_...` fields. All-zero
vectors produce intercept-only scores, **not evidence of benignness**. Scores
and decisions retain the shared normalization and frozen `>=` semantics.

Before **any** `joblib.load`, the scorer checks the exact scikit-learn version,
expected artifact filenames, SHA-256 hashes of all four joblib artifacts plus
the manifest, and finite thresholds. Train/score in the same container generation;
other package versions are recorded but not enforced by the loader.
**Trusted artifacts only:** joblib/pickle deserialization can execute code.
Hashes detect accidental corruption, not authenticity; an attacker who can
replace artifacts and metadata can replace both. This is not a safe loader for
untrusted model files.

## Original native execution reference

The original run completes in host `~/safety-watch/runs/classifiers-20261017-01/source`.
The container sees `/home/ibm-user/shared/runs/classifiers-20261017-01/source`.
The host Git clone remains outside the shared mount.
The commands below record the original invocation, not a new deployment plan.
Use a new private output directory for any repeat. Do not overwrite the frozen run.

```bash
cd /home/ibm-user/shared/runs/classifiers-20261017-01/source
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 -m pytest tests/test_train_classifiers.py -q
```

The verified container-visible CSV paths and planned fresh output path give:

```bash
cd /home/ibm-user/shared/runs/classifiers-20261017-01/source && \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
TOXICCHAT_TRAIN=/home/ibm-user/shared/data/raw/toxicchat0124/data/0124/toxic-chat_annotation_train.csv \
TOXICCHAT_TEST=/home/ibm-user/shared/data/raw/toxicchat0124/data/0124/toxic-chat_annotation_test.csv \
OUT_DIR=/home/ibm-user/shared/runs/classifiers-20261017-01/models \
python -m src.models.train_classifiers
```

The evidence records a reproducible `python3 -m ...` invocation with the actual
input/output paths and supplied thread settings. Reproduction needs a new/empty
output directory; do not overwrite a frozen run.

## Local fixture validation and determinism

```bash
python3 -m pytest tests/test_train_classifiers.py -q
```

The fixture contains harmless synthetic strings **only for unit-test mechanics**,
never study augmentation or model-result evidence. Tests cover strict input
validation, overlap exclusion, grouped conflicts/provenance, split isolation,
fit-only vocabularies, reserved reference isolation, no refit, one test-scoring
pass per model, threshold ties/sentinel/FPR, metrics, hashes, scorer pre-load
rejection, empty inputs, and all-zero features. Independent fixture runs require
identical scores, thresholds, and role assignments in the same runtime, not
identical timestamps/JSON bytes or cross-platform bit equality.

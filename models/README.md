# Classifier artifacts

Complete fitted TF-IDF + logistic-regression pipelines: vocabulary, IDF, coefficients and
intercept. No external checkpoint download is needed. Files are byte-for-byte copies of
native LinuxONE artifacts. Hashes and notebook execution provenance: manifest.json.
Original model cards retain historical source digests.

| Pipeline | File | Human unsafe recall | Human FPR | Human accuracy |
|---|---|---|---|---|
| Retained model | guardrail.joblib | 74.29% | 4.81% | 92.41% |
| Separately calibrated reference | research/separate-calibration/reference.development.joblib | 80.00% | 5.91% | 92.22% |
| Research candidate | research/separate-calibration/candidate.development.joblib | 78.57% | 5.69% | 92.22% |

All human metrics use 527 records (457 benign, 70 unsafe). The retained model threshold
uses development validation. Research models use separate calibration and a different
fitting partition; results do not establish a controlled improvement over the retained model.
80% means unsafe recall (56/70), not accuracy. Accuracy is (TP+TN)/527 and is dominated by
benign records. Both research models exceeded the 5% evaluation FPR budget; neither was promoted.

## Use the retained model

Inside the pinned native s390x Python 3.12.15 container (PYTHONPATH=/workspace/src):

~~~python
from safety_watch.classifier import load_guardrail
model = load_guardrail("/workspace/models")
scores = model.score(texts)
~~~

The existing host CLI defaults to the VM runtime model, whose bytes match this copy.

## Use the 80% recall research model

In the same pinned container, after checking its manifest SHA-256:

~~~python
import json
import joblib
from pathlib import Path

root = Path("/workspace/models/research/separate-calibration")
card = json.loads((root / "reference-model-card.json").read_text())
model = joblib.load(root / "reference.development.joblib")
scores = model.predict_proba(texts)[:, 1]
refuse = scores >= card["threshold"]
~~~

Saved reference threshold: 0.22508886720167454. The candidate has its own card and threshold.
Load trusted joblib files only. Use recorded sklearn 1.9.0/numpy 2.0.2/scipy 1.13.1/
joblib 1.6.0 versions; artifacts were trained and verified on s390x CPU.
Data provenance: lmsys/toxic-chat, CC-BY-NC-4.0, revision recorded in each model card.
No source prompt rows or record-level score logs are included.

Notebook outputs are preserved native executions. Code is split into smaller cells with
identical statement order; original execution counts and output content are retained.
Descriptive Markdown cells remain removed. Publication did not retrain or run official
test. These are development/research results, without a production claim.

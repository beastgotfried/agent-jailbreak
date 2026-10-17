# Canary v1: locked training and integration protocol

Status: proposed execution protocol, committed before canary fitting.
Owner scope: model lead. Execution: existing pinned Python 3.12 native s390x container.
No dependency additions, synthetic prompts, augmentation, transformer, or official-test labels.

## Feature family and fitting

Canary: word TF-IDF 1-2 n-grams, lowercase, min_df=2, max_features=100000,
sublinear_tf=True, norm=l2; logistic regression, liblinear, L2, max_iter=1000, seed=42.
Guardrail stays the existing frozen character 3-5 n-gram model. This is a different
feature representation, not a different classifier algorithm or an independence guarantee.
The canary supplies a reference signal and never changes the guardrail decision.

Use the existing verified ToxicChat 0124 source and unsafe=toxicity OR jailbreaking.
Reproduce the original exact-text audit and split: 4791 eligible, 3832 fitting, 959
development validation. Expected partition SHA-256:
cd5e3febd5e3025cf905214a1369344f5f22b9613fe35a2c577842eacf037c6d.
No refitting on validation after threshold calibration.

Reuse the locked 12-candidate fitting-only selection: C=(0.3,1,3,10,30,100),
class_weight=None/balanced, three stratified folds, seed=42. Learn vocabulary/IDF
within each inner fitting fold. Rank pooled human unsafe recall at <=5% empirical
inner-fold benign FPR, then mean AP, smaller C, no class weighting. Do not change the
search after outer-validation results. Choose the final diagnostic canary threshold
on human-benign development validation at <=5% empirical FPR, respecting ties.

## Measurement and limits

Headline metrics: human-annotated development validation; full validation supplementary.
Report both frozen models on identical records: recall, precision, FPR, F1, AP,
error-indicator Pearson/phi correlation, 2x2 error contingency, shared unsafe misses,
and prediction disagreement. A zero-variance correlation is undefined and stored as null.
This cohort is held out from fitting but used for threshold calibration and has prior
research exposure. It is not an independent test of generalization or monitor efficacy.
The canary need not beat the guardrail; report its measured complementarity honestly.
No automatic promotion, retuning, monitoring efficacy, or repair claims.

## Saved artifacts and handoff

Verify repeated fitting on all 959 validation scores and reload at least 100 real records.
Save canary.joblib with model card and canary.freeze.json binding card/model/threshold
digests. Preserve the guardrail bytes, threshold, card, and marker when adding the canary.
Refuse silent replacement; preserve interrupted-export locks for review.

Joint JSONL core: prompt_id, guardrail_score, canary_score, timestamp, split.
Add schema version, both model digests/thresholds, and the guardrail-only decision.
Real input schema: prompt_id, user_input, optional split. Never echo prompt text.
Scores are class-weighted logistic estimates, not validated harm severity.

Export a deterministic human-benign validation reference membership and paired scores
for the monitor lead, marked development/calibration-derived. Baseline scores remain
private on the VM. KS windows, alarms and their null calibration are monitor-lead work.
Run a 100-record saved-pair smoke test. Record hashes and operation evidence.

Publish reviewed pipeline copies, cards, aggregate results and a small-cell canary
notebook with actual execution outputs. Raw prompts and record-level logs stay private.
Finish with integration/canary/README.md documenting commands, artifacts, contract,
measurements, limits and remaining integration work.

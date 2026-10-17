# Model lead context

## Mission

Maintain native frozen classifiers and qualify future improvements without test leakage.

## Current assets

The native LinuxONE container holds the char 3–5 TF-IDF logistic-regression guardrail and word 1–2 TF-IDF logistic-regression canary. Both use scikit-learn 1.9.0 for fit and inference.

They share an algorithm family and training source. Error correlation is approximately 0.81. Different feature views do not establish independence or a useful canary.

The model output is a 0–1 score. It is not validated harm severity or a calibrated probability.

## Data discipline

Read `docs/MODELS.md`. Only `user_input` supplies features. The original target is toxicity OR jailbreaking.

The official train source has separate fit, threshold-validation, and reserved reference roles after documented exclusions. No full-data refit follows threshold selection. The published official test result must not select future hyperparameters or cohorts.

Do not use `model_output` or moderation outputs as features. Keep exact data and artifact hashes. Treat joblib artifacts as trusted-only inputs.

## Current evidence

Both models detect 231 of 362 unsafe human-annotated official test prompts. Their false-positive rates differ. Matching recall does not mean identical errors.

The later harmful-request replay uses a distinct source-label policy. Its conformal prediction sets are not an updated classifier-test accuracy result.

All measured response candidates require too much review and fail the gate. Classifier weights remain unchanged. Read `docs/BENCHMARK-RESULTS.md` before an effectiveness claim.

## Future changes

Use development data for model selection and reserve fresh final verification. Match the actual target task. Measure shared errors, cohort coverage, calibration, native runtime, and memory.

Do not infer semantic understanding, guaranteed robustness, or safety from a sigmoid score. Do not choose human-only headlines after a favourable test result.

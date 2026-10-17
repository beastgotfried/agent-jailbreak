# Current technical decisions

These decisions supersede conflicting early scaffold claims.

1. **Platform:** Core processing and inference use native LinuxONE s390x CPU execution. Actual native checks qualify dependencies. No x86 emulation or unsupported hardware claims.
2. **Models:** Train our own char/word TF-IDF logistic-regression baselines. DeBERTa is excluded. Different feature views do not establish different algorithm families or independent errors.
3. **Score semantics:** The target is toxicity OR jailbreaking. The output is a model score, not validated severity or a calibrated probability.
4. **Classifier data:** Use the official ToxicChat train source with documented overlap exclusions and training-derived fit/validation/reference roles. Fit only on fit rows. Do not refit after threshold selection or tune against the published official test result.
5. **Classifier threshold:** Select maximum validation recall under empirical benign FPR <=5%. Report actual cohort FPR. This is not a deployment guarantee.
6. **Environment:** Fit and infer in the same container generation. Trusted artifact loaders check model hashes and the exact scikit-learn version.
7. **Imported sources:** Preserve exact text, source labels, revisions or snapshot hashes, and rights metadata. Constructed benchmarks remain identified. No augmentation enters study data.
8. **Purpose separation:** Reference, calibration, and evaluation prompt purposes remain exactly disjoint. Source-labelled benign does not mean universally harmless. Near-duplicate/family independence is not established.
9. **Product scope:** The product watches a guardrail. It is not another stacked prompt filter. Existing observability products and prior research remain part of the comparison set.
10. **Evidence:** A score change does not prove harm. A useful canary, response improvement, and production safety need separate evidence.
11. **Current monitor:** Use signed score difference and a fixed-reference KS statistic. Calibrate a threshold from empirical whole-episode null maxima. Do not call repeated KS p-values a martingale or claim formal anytime validity.
12. **Monitor defaults:** Window64, stride16, horizon512 accepted pairs, 200 null replay runs, alpha0.05, seed42. These are initial design settings, not measured deployment guarantees. Do not tune them on evaluation pools.
13. **Monitor authority:** An alarm reports change only. The monitor never approves or executes a repair. Horizon exhaustion requires explicit episode management.
14. **Artifact scope:** Versioned JSON configurations, score-only evidence, and bounded checkpoints support reproducibility. Their checksums are not signatures, an append-only hash chain, or hardware attestation.
15. **WhyLabs:** The tested wheel-based installation path fails on this image. Its runtime remains excluded; source-build feasibility is untested.
16. **Preparation exception:** Mac-only lossless Parquet format conversion is recorded. Native pool assembly and core model/monitor execution remain on LinuxONE.

Current module contracts are `docs/MODELS.md`, `docs/PROMPT-POOLS.md`, and `docs/MONITOR.md`. Proposed response and enterprise features must not be described as delivered.

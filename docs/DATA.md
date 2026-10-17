# DATA — what we use and why

All datasets live on the VM under `~/safety-watch/data/`. Immutable revisions and SHA-256 hashes: `data/download-manifest.json`.

| Dataset | Role | Size | License | Notes |
|---|---|---|---|---|
| ToxicChat 0124 | Labels. Real user prompts. | 10,165 rows; 5,654 human-annotated | CC-BY-NC-4.0 | Train on full set (owner decision). Headline metrics on human-annotated rows. |
| WAInjectBench text | Fresh 2026 attacks | 3,698 samples (2,707 benign, 991 malicious) | License metadata absent | Constructed benchmark attacks. Not production traffic. |
| JBB-Behaviors | Published attack goals | 100 harmful + 100 benign | MIT | Behavior strings for the attack shift. |
| JBB attack prompts | Attack prompt mirror | 1 parquet | None listed | Parquet: convert to CSV on laptop before use. |
| StrongREJECT | Published attack prompts | 7 parquets | None listed | Same parquet note. |
| HarmBench behaviors | Published attack behaviors | 400 rows (200 standard, 100 copyright, 100 contextual) | MIT | From the official GitHub repository. |

## Rules

- Real data only. No synthetic augmentation anywhere.
- Attack text is data. Never executed. Never rendered raw in the dashboard or the video.
- Official splits stay intact. Test is touched once, at the end.
- The "fresh" claim applies only to WAInjectBench (2026). ToxicChat is 2023 data and we say so.
- Nothing here is claimed absent from any model's training data.

## Label semantics

- Task: prompt harm classification. unsafe = toxicity OR jailbreaking.
- Severity: the logistic model's sigmoid probability, 0 to 1. Binary decides, severity informs.
- Toxicity and injection are different tasks. Do not merge labels across datasets without a justified map.

## Deliberately excluded

- WildGuardMix: provider-gated; its training split is 87% synthetic per its card. Not used.
- SafePickle: model-file malware. Different failure channel. Out of scope.
- LMSYS-Chat-1M / WildChat-1M: real recorded traces. Optional add only if a reviewer demands trace realism. Licenses unchecked.

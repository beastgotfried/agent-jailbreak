# PROMPT POOLS — reproducible benign + attack JSONL pools

Source package for step 2. Implements exact-source provenance, isolated purposes,
and deterministic replay routing. No synthetic augmentation. No training, scoring,
or network requests. Treats all source content as untrusted data.

## Inputs (VM layout)

```
~/safety-watch/data/raw/{toxicchat0124,wainjectbench-text,jbb-behaviors,harmbench-behaviors}/
~/safety-watch/data/raw/{jbb-attacks,strongreject}/*.parquet   # original parquet
~/safety-watch/data/converted/{jbb-attacks,strongreject}/*.jsonl  # after conversion
```

Evidence on the Mac (preparation/prompt-pools-20261017):

* `source-schema.json` — CSV/JSONL row counts and column shapes (no raw prompts).
* `parquet-schema.json` — 7 parquet files, columns and row counts.
* `parquet-details.json` — label and content-shape statistics (no raw prompts).
* `source-download-manifest.json` — LIST of repo/revision/local_dir/files[{path,sha256}], plus BLOCKED gated entries.
* `model-split-manifest.json` — frozen 4734 groups (3076 fit / 947 validation / 711 reference), seed 42.

> Actual CSV/Parquet text is not committed. Real counts are observed on the VM
> after the accepted code runs natively. Fixture tests use harmless synthetic strings.

## Tool 1 — format conversion (Mac only, optional)

```bash
python3 -m src.data.convert_parquet --input-root PATH --output-root NEW_PATH --download-manifest PATH
```

*Generic lossless Parquet-row-to-JSONL.*

* Validates every `.parquet` under `input-root` against `download-manifest`
  (`<dataset_prefix>/<path>` via `local_dir` basename remapping, e.g.
  `jbb-attacks/data/train-00000-of-00001.parquet`). Rejects unexpected,
  missing, or hash-mismatched inputs and non-empty `output-root`.
* Preserves all field values including nested lists (`content:list<str>`),
  exact strings, labels, and empty splits (0-row `train.parquet` → 0-line `.jsonl`).
  No filtering, relabeling, or concatenation.
* Output tree preserves dataset-relative paths with `.parquet` → `.jsonl`.
* Writes `conversion-manifest.json` (`schema_version:1`) with per-file
  `source_relative_path`, `source_sha256`, `output_relative_path`,
  `output_sha256`, `rows`, `columns`, plus `pyarrow_version`.
  Paths are relative to respective roots. Standard JSON (`UTF-8`,
  `allow_nan=False`). Deterministic ordering by `source_relative_path`.

*Lazy pyarrow:* imported only inside `convert_parquet()`; `src.data.build_prompt_pools`
never imports it, so native assembly runs without pyarrow on s390x.

## Tool 2 — pool assembly (native s390x, stdlib only)

```bash
python3 -m src.data.build_prompt_pools \
  --raw-root PATH --converted-root PATH \
  --download-manifest PATH --model-split-manifest PATH \
  --output-dir NEW_PATH [--seed 42]
```

`--seed 42` default; CLI also accepts `--seed 42` conventional form.

### Validation

* Validates `raw` non-parquet file hashes against `download-manifest` via
  `raw-root/<prefix>/<path>` remapping (never the absolute `local_dir` from
  another machine). Rejects unsafe relative paths (`..` or absolute),
  unreadable/malformed CSV/JSONL, missing required inputs, and non-empty
  output. Never silently creates empty successful pools from missing files.
* Validates `converted` data via `converted-root/conversion-manifest.json`:
  each entry’s `source_sha256` is linked to the original parquet SHA in the
  download manifest, and `output_sha256` matches the actual `.jsonl` hash.
  Missing entries or hash mismatches fail.
* Validates `toxic-chat_annotation_train/test.csv` hashes against the
  `model-split-manifest` `datasets` hashes. Verifies the manifest itself via
  SHA-256 for the pool-manifest.
* Explicitly reports `BLOCKED` manifest entries (e.g. `walledai/HarmBench`,
  `walledai/StrongREJECT` gated) and does not fetch or bypass them.

### Source mapping (namespaces preserved, not merged)

| Dataset | Files | Text field | Label / task | Routing | Caveat |
|---|---|---|---|---|---|
| **ToxicChat0124** `toxicchat0124/data/0124/*.csv` | train 5 082, test 5 083 | `user_input` | `toxicity`/`jailbreaking` → unsafe, `human_annotation` flag; task `toxicity_or_jailbreak` | Only **reference** groups with `toxicity=0,jailbreaking=0, human=True` (any member of the group qualifies) are allowed as benign candidates. `fit`/`validation`/`test`/`nonhuman-only`/`unsafe` reference prompts form a **global blocked hash set** used to exclude overlap via any other source. | Real user prompts; test never a pool source; provenance keeps `conv_id` and literal flags. |
| **WAInjectBench** `wainjectbench-text/text/benign/*.jsonl` (4) + `text/malicious/*.jsonl` (8) | 2 707 benign + 991 malicious before exclusions | `text` | `id`, file/category, directory label; task `prompt_injection` | benign dir → benign pool, malicious dir → attack pool | Benchmark-constructed/curated, fresh 2026, not guaranteed production traffic; routing ≠ universal toxicity. License null → rights-review flagged. |
| **JBB-Behaviors** `jbb-behaviors/data/benign-behaviors.csv` / `harmful-behaviors.csv` (100 each) | `Goal` (never `Target`), `Index`, `Category`/`Behavior`/`Source` | task `harmful_request`, origin `curated_behavior_goal` | benign file → benign, harmful → attack | `judge-comparison.csv` excluded entirely (judge outputs). MIT, revision pinned. |
| **JailbreakBench (walledai)** `jbb-attacks/data/train-00000-of-00001.parquet` → converted | 200 rows `prompt, target, behavior, category, source, subset` (100 benign / 100 harmful) | `prompt` only (never `target`) | task `harmful_request`, origin `behavior_goal_mirror` | subset routing; deduped globally vs JBB-Behaviors | NOT proven GCG strings; license null stays unlisted (upstream MIT note is not a rights grant). |
| **StrongREJECT** `strongreject/{data,keywords,violence}/{train,validation}` (7 parquet → jsonl, 3 train files are 0-row) | `clf_label, proxy_clf_label, instructions, content:list[str], answer_prompt, gen_target, proxy_gen_target` | `clf_label` 0 benign / 1 harmful supplies task label; `proxy_clf_label` preserved separately | `content[0]` used **only** when `instructions==""` and `content` is single nonempty string; else explicit exclusion (no concatenation). Default/keywords/violence configs may overlap; deduped globally. | All inspected rows have `clf_label=1, proxy=0` but benign handling is retained if `clf=0`. License null → rights-review. |
| **HarmBench** `harmbench-behaviors/harmbench_behaviors_text_all.csv` (400) | `Behavior`, `FunctionalCategory`, `SemanticCategory`, `Tags`, `ContextString`, `BehaviorID` | task `harmful_request` | Only `FunctionalCategory=standard` (200) with exact `Behavior` text are eligible → attack pool. `contextual` (100, requires context assembly) and `copyright` (100, different policy task) are explicitly excluded with reason, ID, and category (no raw text). | MIT; revision `main@2026-10-16` is snapshot (`revision_strength: snapshot_hash_only`); retain bytes SHA. Not a claim of successful jailbreaks. |

*Literal CSV handling:* stdlib `csv` with `keep_default_na=False` equivalent; values such as `NA`, `0` are kept as strings.

*Converted provenance:* for Parquet-derived rows the source entry records **both**
original Parquet `path+SHA+row` and converted `path+SHA`.

### Normalization, dedup, and conflict handling

* Identity uses `lowercase + collapsed whitespace` (same as classifier `normalize_prompt`).
  Emitted `text` is one exact original variant, not normalized.
* One canonical record per normalized hash (`normalized_text_sha256`).
  Deterministic ordering by `(repo, original_relative_path, row_index)` chooses the representative text.
  All duplicate provenance links are preserved in `sources` (each with distinct `source_text_sha256` if variants differ).
* **Global blocked set** (`toxicchat train+test` minus allowed reference benign) excludes any overlapping group from any source with reason `blocked_toxicchat_overlap`.
* **Conflicting benign/attack groups** (same normalized hash seen as both kinds
  across sources) exclude the whole group with reason `conflicting_benign_attack`; never relabeled.
* Pool `text_sha256` is SHA-256 of the selected exact text.

### Partitions (replay purposes, not classifier splits)

After dedup and exclusions, each canonical group is assigned to exactly one purpose:

* **Benign:** 40 % `reference`, 30 % `calibration`, 30 % `evaluation`.
* **Attack:** 50 % `calibration`, 50 % `evaluation`; **no** attack `reference`.

*Determinism:* within each `(pool_kind, tuple(sorted source_tasks))` stratum,
groups are sorted by `SHA256(f"{seed}:{normalized_text_sha256}")` and the
seed (default 42) splits them with documented rounding:

```
benign:  n_ref = int(N*0.4), n_cal = int(N*0.3), n_eval = N - n_ref - n_cal
attack:  n_cal = N//2,       n_eval = N - n_cal
```

Tiny strata may have empty roles; counts are reported.

No cross-purpose duplicate normalized hash remains. Original dataset splits
(`train`/`test`, `train`/`validation`, `standard`/`contextual`/`copyright`) stay in
provenance; these assignments are **not** new classifier train/test splits.
No final-repair-verification guarantee.

### Outputs

All under `output-dir` (must be new/empty; never overwrites):

* `benign.jsonl` — routing set, one JSON per line, fields
  `schema_version:1, prompt_id, normalized_text_sha256, text, text_sha256,
  pool_kind:"benign", partition, source_tasks:[], sources:[]`.
  `prompt_id` is stable from `normalized_text_sha256`; `sources` are sorted
  deterministically. Sorted by `prompt_id` for byte-identical re-runs.
* `attacks.jsonl` — same schema, `pool_kind:"attack"`.
* `exclusions.jsonl` — one JSON per line, `schema_version:1, reason,
  normalized_text_sha256, sources[] / source_id / original_labels…`. **Never**
  contains raw prompt text; only reason, hashes, IDs, labels, and category.
* `pool-manifest.json` (`schema_version:1`) — deterministic (no wall-clock
  timestamp), `sort_keys` + `allow_nan=False`, containing:

  `seed, settings, counts{by_kind, by_partition, by_task, by_kind_task_partition, by_source_repo, partition_detail}, input_hashes{download_manifest, model_split_manifest, raw_validated, converted_validated, conversion_manifest}, model_manifest_hash, exclusions{per_source, blocked, conflict, total, blocked_manifest_entries, blocked_hash_set_size, allowed_reference}, dedup{total_before, canonical_after, removed}, outputs{sha256, bytes, lines}, output_hashes, caveats[], schema{}, deterministic_note`.

  Caveats explicitly state: routing ≠ universal safety truth; label/task namespaces;
  `judge-comparison` excluded; `target`/`Target` never used; StrongREJECT proxy
  does not override; only HarmBench standard eligible; test never pooled; partitions
  are replay purposes.

Repeated equivalent runs give identical pool bytes and partition assignments
(same seed, same input hashes).

### Error handling and logging

No raw prompt is printed or logged; counts, hashes, and IDs only. Unsafe paths,
missing/unreadable/malformed data, hash mismatches, blocked entries, and
non-empty output all fail fast with a clear `SystemExit` message and leave the
output directory empty or absent. Sources are left untouched.

### Repro rule

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -m src.data.convert_parquet --input-root PATH --output-root NEW_PATH --download-manifest PATH

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -m src.data.build_prompt_pools --raw-root PATH --converted-root PATH \
  --download-manifest PATH --model-split-manifest PATH --output-dir NEW_PATH --seed 42
```

`data/pools/` and `data/raw/` are gitignored; real pools stay on the VM, not in Git.

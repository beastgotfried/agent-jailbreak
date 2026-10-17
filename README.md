# Safety Monitor Integrity Watch

A guardrail makes request decisions. A canary scores copies. Runtime probes detect output changes and support approved, verified recovery or containment.

![Architecture](docs/img/architecture.png)

## Why LinuxONE

- **Native execution:** Data generation, model fit, inference, monitoring, recovery, and audit run on LinuxONE `s390x` without emulation.
- **CPU-only operation:** The selected system needs no GPU or external model API.
- **Reproducibility:** A fresh container, with networking disabled, reproduces the data, learned model state, request decisions, and benchmark metrics.

## Release

The release provides a frozen TF-IDF guardrail and canary pair, deterministic runtime checks, verified recovery, and approved containment for offline research.

The private kit contains the models, data, native runtime image, and reproduction instructions. Public Git contains source and aggregate evidence, not raw requests or model binaries.

[Model identity](src/release/selected.json) · [Results](releases/baseline-20261017/summary.json) · [Native verification](releases/baseline-20261017/qualification.json)

## Data sources

The selected classifiers fit our **100,000-row synthetic study** with a 70,000/10,000/20,000 train/validation/test split. External sources have the separate roles below.

| Source | Project use | License |
|---|---|---|
| [ToxicChat 0124](https://huggingface.co/datasets/lmsys/toxic-chat) | Original classifiers; current external regression | CC-BY-NC-4.0 |
| [WAInjectBench](https://huggingface.co/datasets/Norrrrrrr/WAInjectBench) | Original traffic pools | Not verified |
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | Original behavior-goal pools | MIT |
| [JailbreakBench mirror](https://huggingface.co/datasets/walledai/JailbreakBench) | Original pool provenance and deduplication | Not listed |
| [StrongREJECT](https://huggingface.co/datasets/AlignmentResearch/StrongREJECT) | Original request pools | Not listed |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | Original standard-behavior pools | MIT |
| [XSTest](https://github.com/paul-rottger/xstest) | Recovery evaluation and external regression | CC-BY-4.0 |
| [Dolly 15k](https://github.com/databrickslabs/dolly) | Unlabelled traffic-profile study | CC-BY-SA-3.0 |
| [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Acquisition only; profile study paused | Apache-2.0 |

Public access does not imply unrestricted reuse. [Source records](releases/baseline-20261017/data-sources.json) contain pinned revisions, licenses, and roles.

The archived neural comparison uses [Google BERT-tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2). [MiniLM-L6](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) serves only a compatibility/cost check. Both model cards specify Apache-2.0.

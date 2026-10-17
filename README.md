# Safety Monitor Integrity Watch

**Verified runtime recovery for AI safety filters on native LinuxONE (`s390x`).**

The guardrail makes request decisions. A canary scores copies. Independent probes check the active runtime against a frozen baseline.

Approved responses restore baseline scores through affine correction or a verified scorer restart. Separate containment authority holds requests when verification does not permit recovery.

![Architecture](docs/img/architecture.png)

## Selected release

The release selects the **TF-IDF + logistic-regression guardrail and canary pair** from the completed 100,000-row study.

- Fixed model weights, thresholds, and input identities
- Paired guardrail-only and monitored replay
- Actual candidate execution before activation
- Independent numeric audit and exact repeat evidence
- Native LinuxONE reproduction with private, hash-verified artifacts

The release does not change data or calibration. The neural comparison remains in the experiment record.

## Measured decision profile

These results cover **60 paired episodes and 30,720 observations in a controlled synthetic fault benchmark**.

| Complete-stream metric | Guardrail | Guardrail + monitor |
|---|---:|---:|
| Correct automatic decisions | 75.64% | **81.69%** |
| Harmful requests allowed | 29.91% | **3.76%** |
| Safe requests refused | 18.44% | 22.47% |
| Requests held | 0% | 5.49% |

The monitor adds **6.05 percentage points** to the benchmark score. Harmful allowances decrease by **87.44%**. Holds receive no correctness credit.

The runtime checks confirm 55/55 injected incidents. Verified responses restore service in 45/50 fault episodes. The remaining five episodes use containment. Ten no-fault episodes have no false runtime confirmation.

The healthy classifier has 88.67% synthetic test accuracy and a 22.63% safe false-refusal rate. Recovery restores that baseline; it does not improve semantic classification.

## Reproduce the release

Read **[the release guide](docs/RELEASE.md)** for the private bundle, native environment, wheel installation, and full reproduction command.

The public repository contains source, tests, package metadata, and aggregate evidence. The private bundle supplies model binaries, the reviewed catalogue, generated data, and exact external regression inputs.

The `safety-watch score` command classifies text without executing it. The `safety-watch-replay` command exercises the complete offline monitor and recovery path.

## Evidence and scope

The dataset contains 100,000 compositional rows from 91 reviewed intent groups. Those rows are not 100,000 independent human requests. Exposed synthetic and external tests remain regression evidence.

This research release covers the implemented offline pipeline. Approvals are simulated, and request execution uses a stub. The results do not establish production qualification or an IBM endorsement.

| Document | Purpose |
|---|---|
| [Release guide](docs/RELEASE.md) | Package contents and native reproduction |
| [Selected model identity](src/release/selected.json) | Approved model hashes and thresholds |
| [Aggregate evidence](releases/baseline-20261017/summary.json) | Selected and comparison results |
| [Model record](docs/MODELS.md) | Fit protocol, score semantics, and external checks |
| [Experiment record](docs/EXPERIMENTS.md) | Complete measured results and earlier studies |
| [Monitor contract](docs/MONITOR.md) | Passive traffic channel and evidence boundaries |
| [Original benchmark](docs/BENCHMARK-RESULTS.md) | Preserved original-policy results |

Raw requests, private model binaries, vocabularies, credentials, and server addresses stay outside public Git.

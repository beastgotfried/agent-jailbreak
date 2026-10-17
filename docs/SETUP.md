# SETUP — Environment

Facts verified on 2026-10-16. Full version lives on the VM at `~/safety-watch/docs/SETUP.md`.

## Server

- LinuxONE Community Cloud VM. Ubuntu 24.04. s390x. 3 vCPU, 8 GB RAM, 50 GB disk.
- The IP address and SSH keys are shared privately by the team lead. Never commit them.
- Connect: `ssh -i <key>.pem linux1@<ip>`.

## Two Python environments

| Environment | Use | Key packages |
|---|---|---|
| Host venv `~/safety-watch/.venv` | scripts, data prep | numpy 1.26, scipy 1.11, sklearn 1.4.1, pandas 2.1, fastapi 0.143 |
| Container `safety-watch-env:2026-10-16` | models, training, event runtime | torch 2.11.0+cpu, transformers 5.19, tokenizers 0.23, sklearn 1.9.0, fastapi 0.143, plotly, pytest |

Train and run each model in the same environment. Do not cross-load pickles.

## Jupyter

Private. localhost + token + SSH tunnel. Start: `bash ~/safety-watch/setup/start-jupyter.sh`. Token: `cat ~/safety-watch/.secrets/jupyter_token` (server-side only).

## Known platform limits

- No GPU. CPU inference only.
- `pyarrow` has no s390x wheel. Parquet files must be converted to CSV on a laptop, then uploaded.
- WhyLabs packages (whylogs, langkit, openllmtelemetry) cannot install on s390x. Excluded. Record in `~/safety-watch/vendor/whylabs/`.
- Hugging Face Hub client: use 1.33.0. The 2.x client conflicts with a native decoder on this image.

## Data on disk

`~/safety-watch/data/raw/`: toxicchat0124, wainjectbench-text, jbb-behaviors, jbb-attacks, strongreject, harmbench-behaviors. Hashes in `data/download-manifest.json`.

## Paper library

`~/safety-watch/docs/papers/INDEX.md`. Fourteen papers as Markdown and text. PDFs removed after verification.

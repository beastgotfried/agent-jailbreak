# Frozen LinuxONE release

Release ID: `safety-watch-tfidf-20261017`. Python package version: `0.1.0`.

The release selects the TF-IDF guardrail/canary pair from source `d86d384e2ee7d0f07f2f2fc5bf2b6dec64e13cac`.
Its measured monitored operational score is 81.69% in the frozen synthetic fault benchmark.
The neural comparison remains archived separately. It is not the default model.

## Distribution

Public Git contains the source, tests, dependency pins, selected model hashes, and aggregate evidence.
Private artifacts remain separate:

- `safety-watch-baseline-20261017.tar.gz`: Complete private project bundle
- `safety-watch-runtime-s390x-20261017.tar.gz`: Exact native runtime image
- `SHA256SUMS`: Archive integrity values

The project bundle contains these directories:

| Path | Content |
|---|---|
| `code/` | Exact release source |
| `wheel/` | Installable release wheel |
| `studies/synthetic-100k-20261017-01/` | Selected model, data, catalogue, protocol, original source, and evidence |
| `studies/neural-upgrade-20261017-01/` | Preserved comparison checkpoint and evidence |
| `external/` | Exact regression CSVs and source/license metadata |
| `qualification/` | Fresh native package and reproduction evidence |

`package.json` records the release commit and image ID. `bundle-manifest.json` records file hashes and the internal data link.

**Keep both archives private.** The project archive contains raw requests and model vocabularies.
The external ToxicChat source uses CC-BY-NC-4.0. XSTest uses CC-BY-4.0.
The selected model fits only the synthetic train split; these external CSVs serve regression checks.
Packaging does not settle commercial dataset rights or authorize public image redistribution.

## Native environment

Use Linux on IBM LinuxONE or IBM Z with an `s390x` CPU.
The verified runtime uses Python 3.12.15 and the dependency versions in `src/release/selected.json`.
This is a Linux environment, not a z/OS package.
No GPU, emulation, Telum, Spyre, or hardware-attestation feature is required or claimed.

The runtime image ID is:

```text
sha256:85ec856d19102a0791fea7fbe2c36d7d2412093d3574af4f899cb692d95bccae
```

A matching image archive preserves the tested native binaries. It avoids assumptions about current s390x wheel availability.
The original environment recipe appears in `environment/Dockerfile.reference`; its package resolution is historical, not an exact rebuild lock.
The saved image and recorded image ID define the exact runtime.

## Load the private kit

Place the archives and `SHA256SUMS` in `$HOME/safety-watch-release`.

```bash
cd "$HOME/safety-watch-release"
sha256sum -c SHA256SUMS
gzip -dc safety-watch-runtime-s390x-20261017.tar.gz | docker load
tar -xzf safety-watch-baseline-20261017.tar.gz
export BUNDLE="$HOME/safety-watch-release/safety-watch-baseline-20261017"
docker image inspect safety-watch-env:2026-10-16 --format '{{.Id}} {{.Architecture}}'
```

Check the image ID against the value above. It must report `s390x`.
Hashes detect changes; they do not establish authenticity. Obtain the kit and its checksum through trusted channels.

## Inspect or score

These commands install only the project wheel into a temporary container. They do not change the model.

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  --cap-drop ALL --security-opt no-new-privileges \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
  -v "$BUNDLE:/bundle:ro" --entrypoint sh safety-watch-env:2026-10-16 -c '
    python -m pip install --no-deps --no-index --target /tmp/release /bundle/wheel/*.whl &&
    PYTHONPATH=/tmp/release python -m src.release.cli inspect \
      --models /bundle/studies/synthetic-100k-20261017-01/models
  '
```

For `score`, supply JSONL with one `text` field per request. Use `--input` and `--output` paths inside an additional private mount.
The command verifies the published artifact hashes before model deserialization.
It emits scores and the guardrail decision without echoing request text.
It does not execute requests or enable the runtime monitor.
Malformed inputs stop the command with a redacted error and no decision for that row.
A new output path is required; earlier rows can exist if a later row fails.

## Reproduce the complete selected pipeline

Create a new output directory owned by UID 1000, the runtime image user.
The command runs offline with the package mounted read-only.
It regenerates 100,000 rows, refits the fixed classifier pair, and compares data and model bytes with the selected artifacts.
It then executes all 60 paired episodes through the release wheel and runs the independent numeric audit.

```bash
export OUT="$HOME/safety-watch-release/reproduction-01"
mkdir -m 700 "$OUT"
export RELEASE_COMMIT="$(python3 -c 'import json,os; print(json.load(open(os.environ["BUNDLE"]+"/package.json"))["source_commit"])')"
docker run --rm --network none --cpus 2 --memory 4g \
  --cap-drop ALL --security-opt no-new-privileges \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
  -e RELEASE_COMMIT="$RELEASE_COMMIT" \
  -v "$BUNDLE:/bundle:ro" -v "$OUT:/out" \
  --entrypoint sh safety-watch-env:2026-10-16 -c '
    python -m pip install --no-deps --no-index --target /tmp/release /bundle/wheel/*.whl &&
    PYTHONPATH=/tmp/release python /bundle/code/scripts/reproduce_selected.py \
      --bundle /bundle --output /out/run --release-source-commit "$RELEASE_COMMIT"
  '
```

Check `/out/run/reproduction-receipt.json` through the host output directory.
The original training source supplies the fit; the installed release wheel supplies runtime replay.
Source identifiers, new freeze identifiers, and elapsed times differ where expected.
Data, weights, policy, metric values, and deterministic algorithm evidence must match.
This repeat establishes reproducibility, not another independent effectiveness trial.

## Scope

The runtime checks, verified response, containment, and paired replay form the implemented offline system.
Production identity, approval services, persistent incident storage, and real-agent connectors require separate qualification.
Finite probes do not establish universal fault detection.
The selected healthy classifier has a 22.63% safe false-refusal rate on its synthetic test.
Read the model and experiment records before any deployment decision.

Keep data, calibration, and model changes on a separate development branch after this release.

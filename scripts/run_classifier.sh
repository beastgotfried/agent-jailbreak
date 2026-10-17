#!/usr/bin/env bash
# VM host only. Read-only source/data and persistent private artifacts.
set -euo pipefail
[[ "$(uname -m)" == "s390x" ]] || { echo "Run on native LinuxONE s390x." >&2; exit 1; }
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${SAFETY_WATCH_RUNTIME_HOST:-/home/linux1/safety-watch}"
[[ -d "$RUNTIME_ROOT/data" ]] || { echo "Existing data directory missing." >&2; exit 1; }
mkdir -p "$RUNTIME_ROOT/models" "$RUNTIME_ROOT/results" "$RUNTIME_ROOT/logs"
IMAGE="${SAFETY_WATCH_IMAGE:-$(sudo -n docker inspect --format '{{.Image}}' safety-watch-lab)}"
[[ "$(sudo -n docker image inspect --format '{{.Architecture}}' "$IMAGE")" == "s390x" ]] ||
  { echo "Pinned image must be s390x." >&2; exit 1; }
COMMAND="${1:-}"
[[ $# -gt 0 ]] && shift
case "$COMMAND" in
  train|export|score|evaluate)
    PYTHON_ARGS=("/workspace/scripts/${COMMAND}_classifier.py" "$@")
    ;;
  canary)
    PYTHON_ARGS=("/workspace/scripts/canary_classifier.py" "$@")
    ;;
  test)
    PYTHON_ARGS=("-m" "pytest" "-q" "-p" "no:cacheprovider" "/workspace/tests" "$@")
    ;;
  *)
    echo "Usage: scripts/run_classifier.sh {train|export|score|evaluate|canary|test} [arguments]" >&2
    exit 2
    ;;
esac
exec sudo -n docker run --rm -i --pull never --network none --platform linux/s390x \
  --cpus 2 --memory 4g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  --mount "type=bind,src=$REPO_ROOT,dst=/workspace,readonly" \
  --mount "type=bind,src=$RUNTIME_ROOT/data,dst=/runtime/data,readonly" \
  --mount "type=bind,src=$RUNTIME_ROOT/models,dst=/runtime/models" \
  --mount "type=bind,src=$RUNTIME_ROOT/results,dst=/runtime/results" \
  --mount "type=bind,src=$RUNTIME_ROOT/logs,dst=/runtime/logs" \
  --workdir /workspace --entrypoint /opt/venv/bin/python "$IMAGE" "${PYTHON_ARGS[@]}"

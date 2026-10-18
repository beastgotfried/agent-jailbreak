"""Thin entrypoint; classifier logic lives in src/safety_watch/classifier."""
import os
import sys
from pathlib import Path
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safety_watch.classifier.cli import main
if __name__ == "__main__":
    raise SystemExit(main(["score", *sys.argv[1:]]))

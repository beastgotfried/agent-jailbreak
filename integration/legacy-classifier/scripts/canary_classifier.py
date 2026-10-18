"""Entry point; execute through the native VM runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safety_watch.canary.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

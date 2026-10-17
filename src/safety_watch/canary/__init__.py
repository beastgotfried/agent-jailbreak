"""Word-feature reference signal and paired scoring API."""
from .training import train_canary
from .artifacts import export_canary, load_canary
from .pair import load_pair, joint_records, evaluate_pair

__all__ = ["train_canary", "export_canary", "load_canary", "load_pair", "joint_records", "evaluate_pair"]

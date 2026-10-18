"""Pinned native runtime and classifier defaults; no installation side effects."""
from dataclasses import dataclass, field
from pathlib import Path

def model_defaults():
    return {
        "analyzer": "char", "ngram_range": (3, 5), "lowercase": True,
        "min_df": 2, "max_features": 100_000, "sublinear_tf": True, "norm": "l2",
        "solver": "liblinear", "C": 100.0, "class_weight": "balanced", "max_iter": 1000,
    }

@dataclass
class TrainConfig:
    runtime_root: Path = Path("/runtime")
    project_root: Path = Path("/workspace")
    seed: int = 42
    validation_fraction: float = 0.20
    target_fpr: float = 0.05
    search: bool = False
    model: dict = field(default_factory=model_defaults)

    def __post_init__(self):
        self.runtime_root = Path(self.runtime_root)
        self.project_root = Path(self.project_root)

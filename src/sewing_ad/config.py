"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Fix every RNG used by the pipeline (reproducibility).

    Seeds the standard sources used downstream: Python ``random``, NumPy,
    PyTorch CPU, and (if available) every CUDA device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

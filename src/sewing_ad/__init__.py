"""sewing_ad — shared modules for the *Scientific Reports* PatchCore sewing-anomaly experiments.

This package targets the released ``anomalib==1.2.0`` (PyPI) and bundles a
small compatibility patch that restores the earlier MinMax-normalization
behavior the paper was evaluated under (see ``anomalib_patch.py``), together
with the model builder (``ft_patchcore.py``), datamodule builder
(``data.py``), and experiment runner (``runner.py``).
"""

from .anomalib_patch import apply_anomalib_patches
from .config import load_config, set_seed
from .data import build_folder_datamodule
from .ft_patchcore import build_ft_patchcore
from .runner import run_experiment

__all__ = [
    "apply_anomalib_patches",
    "load_config",
    "set_seed",
    "build_folder_datamodule",
    "build_ft_patchcore",
    "run_experiment",
]

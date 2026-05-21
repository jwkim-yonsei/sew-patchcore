"""Single-experiment runner: build -> fit -> test -> collect metrics -> CSV.

One :func:`run_experiment` call corresponds to one cell of the nylon
hyper-parameter sweep.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

from anomalib import TaskType
from anomalib.engine import Engine
from anomalib.utils.normalization import NormalizationMethod

from .anomalib_patch import apply_anomalib_patches
from .config import set_seed
from .data import build_folder_datamodule
from .ft_patchcore import build_ft_patchcore

logger = logging.getLogger(__name__)


def _append_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append ``row`` to ``path``, writing a header if the file is new.

    Each cell is flushed to disk immediately, so a long grid run can be
    interrupted and resumed without losing completed cells.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one PatchCore fit+test and return / record the metrics.

    Expected ``cfg`` keys
    ---------------------
    seed, accelerator, results_dir, results_csv
    backbone, layers, ratio, neighbors, ft_weight_path (None for non-FT)
    data: {name, root, normal_dir, abnormal_dir, train_batch_size,
           eval_batch_size, num_workers, val_split_ratio}
    record: dict of identifying fields prepended to the CSV row
    allow_unsafe_weights (bool, default False)

    Returns a dict with ``image_AUROC``, ``image_F1Score``,
    ``train_duration``, ``test_duration``.

    Only AUROC / F1 are reproduction targets. The duration columns are
    hardware-dependent and recorded for reference only.
    """
    set_seed(cfg.get("seed", 42))

    data = cfg["data"]
    datamodule = build_folder_datamodule(
        name=data["name"],
        root=data["root"],
        normal_dir=data["normal_dir"],
        abnormal_dir=data["abnormal_dir"],
        train_batch_size=data["train_batch_size"],
        eval_batch_size=data["eval_batch_size"],
        num_workers=data["num_workers"],
        val_split_ratio=data.get("val_split_ratio", 0.2),
        seed=cfg.get("seed", 42),
    )

    model = build_ft_patchcore(
        backbone=cfg["backbone"],
        layers=cfg["layers"],
        ratio=cfg["ratio"],
        neighbors=cfg["neighbors"],
        ft_weight_path=cfg.get("ft_weight_path"),
        allow_unsafe_weights=cfg.get("allow_unsafe_weights", False),
    )

    # Apply the MinMax-normalization compatibility patch (anomalib upstream
    # changed the metric-normalization behavior between the version the
    # paper was evaluated under and PyPI 1.2.0). See `anomalib_patch.py`.
    # Idempotent — safe to call once per cell in the sweep loop.
    apply_anomalib_patches()

    # Engine kwargs are spelled out explicitly. Omitting them would silently
    # fall back to Engine defaults (task=SEGMENTATION, non-deterministic) —
    # the datamodule's task="classification" overrides `task`, but NOT
    # `normalization` / `deterministic`, which would break reproduction.
    engine = Engine(
        normalization=NormalizationMethod.MIN_MAX,
        task=TaskType.CLASSIFICATION,
        accelerator=cfg.get("accelerator", "auto"),
        default_root_dir=cfg["results_dir"],
        deterministic=True,
    )

    # AUROC/F1 are deterministic under the fixed seed and config above,
    # so one fit() + one test() suffices.
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_duration = time.time() - t0

    t0 = time.time()
    test_result = engine.test(model=model, datamodule=datamodule)[0]
    test_duration = time.time() - t0

    metrics = {
        "image_AUROC": float(test_result["image_AUROC"]),
        "image_F1Score": float(test_result["image_F1Score"]),
        "train_duration": train_duration,
        "test_duration": test_duration,
    }

    row = {**cfg.get("record", {}), **metrics}

    if cfg.get("results_csv"):
        _append_csv(cfg["results_csv"], row)

    logger.info(
        "%s | AUROC=%.4f F1=%.4f (train %.1fs / test %.1fs)",
        cfg.get("record", {}),
        metrics["image_AUROC"],
        metrics["image_F1Score"],
        train_duration,
        test_duration,
    )
    return metrics

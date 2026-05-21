"""Folder datamodule builder used by the nylon hyper-parameter sweep.

Wraps an anomalib ``Folder`` datamodule in classification mode (the nylon
sweep uses image-level AUROC / F1) around the parameters set in
``configs/nylon_sweep.yaml``.
"""

from __future__ import annotations

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode


def build_folder_datamodule(
    *,
    name: str,
    root: str,
    normal_dir: str,
    abnormal_dir: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    val_split_ratio: float = 0.2,
    seed: int = 42,
) -> Folder:
    """Construct an anomalib ``Folder`` datamodule in classification mode.

    Args mirror the ``Folder(...)`` calls in the original scripts:

    - ``test_split_mode = FROM_DIR``  (the ``abnormal`` dir provides the test split)
    - ``val_split_mode  = FROM_TRAIN`` with ``val_split_ratio`` of the train set
    - ``task = "classification"``     (image-level AUROC / F1)

    The returned datamodule is **not** ``setup()``-ed; the anomalib ``Engine``
    calls ``setup()`` itself during ``fit``/``test``.
    """
    return Folder(
        name=name,
        root=root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=val_split_ratio,
        task="classification",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
    )

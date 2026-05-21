"""Restore anomalib's earlier MinMax normalization behavior on released 1.2.0.

In an earlier version of anomalib, the MinMax normalization callback held a
single metric that only observed ``anomaly_maps``, so image-level
``pred_scores`` were normalized with the (wider) anomaly-map range. Released
1.2.0 changed this to per-output statistics (a ``MetricCollection`` with a
separate ``MinMax`` for ``pred_scores``), which gives ``pred_scores`` their
own narrow range. After ``+0.5`` and the ``[0,1]`` clamp, many image scores
saturate to exactly 0/1, collapsing the rank-based image AUROC.

This patch keeps ALL of 1.2.0's normalization machinery -- the
``MetricCollection``, ``setup()``, callback wiring, the
``anomaly_maps``/``box_scores`` paths, and the clamp -- byte-for-byte; it
changes exactly one thing: the statistics source used to normalize
``pred_scores`` (from the per-output metric to the anomaly-map metric),
restoring the earlier behavior the paper was evaluated under.

Applied by :func:`apply_anomalib_patches`, which is called once per process
from :func:`sewing_ad.runner.run_experiment` before the ``Engine`` is built.
The patch is idempotent.
"""

from __future__ import annotations

from anomalib.callbacks.normalization.min_max_normalization import (
    _MinMaxNormalizationCallback,
)
from anomalib.utils.normalization.min_max import normalize


def _normalize_batch_anomaly_map_range(outputs, pl_module) -> None:
    """Replacement ``_normalize_batch`` that uses anomaly-map stats for ``pred_scores``.

    The only difference from released 1.2.0's ``_normalize_batch`` is the
    statistics source for ``pred_scores`` (``["anomaly_maps"]`` instead of
    ``["pred_scores"]``). ``anomaly_maps`` and ``box_scores`` paths are
    unchanged. The ``[0,1]`` clamp inside ``normalize`` is retained.
    """
    image_threshold = pl_module.image_threshold.value.cpu()
    pixel_threshold = pl_module.pixel_threshold.value.cpu()
    map_stats = pl_module.normalization_metrics["anomaly_maps"].cpu()
    if "pred_scores" in outputs:
        outputs["pred_scores"] = normalize(
            outputs["pred_scores"], image_threshold, map_stats.min, map_stats.max,
        )
    if "anomaly_maps" in outputs:
        outputs["anomaly_maps"] = normalize(
            outputs["anomaly_maps"], pixel_threshold, map_stats.min, map_stats.max,
        )
    if "box_scores" in outputs:
        box_stats = pl_module.normalization_metrics["box_scores"].cpu()
        outputs["box_scores"] = [
            normalize(scores, pixel_threshold, box_stats.min, box_stats.max)
            for scores in outputs["box_scores"]
        ]


def apply_anomalib_patches() -> None:
    """Install the compatibility patch on ``_MinMaxNormalizationCallback``.

    Idempotent. Safe to call repeatedly (e.g. once per cell in a sweep);
    subsequent calls replace the static method with the same function.
    """
    _MinMaxNormalizationCallback._normalize_batch = staticmethod(
        _normalize_batch_anomaly_map_range
    )

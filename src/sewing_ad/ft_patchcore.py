"""PatchCore builder for the sewing-anomaly experiments.

Builds a PatchCore on top of the released ``anomalib==1.2.0`` (PyPI) and
handles two release-specific concerns:

1. **FT-backbone loading without ``__AT__``.**
   anomalib's ``"<model>__AT__/path/weight.pt"`` backbone-string mechanism
   has behavior that varies across versions, so this module **never uses
   it**. Instead :func:`build_ft_patchcore` constructs a plain-backbone
   ``Patchcore`` and :func:`_load_ft_weights` loads the fine-tuned weights
   directly into the timm feature extractor (``module.`` prefixes stripped,
   ``fc.*`` keys dropped, ``strict=False``, with the load result inspected
   to fail closed on a weight/backbone mismatch).

2. **coreset ``ratio == 1.0`` shortcut.**
   Released anomalib runs the full O(n^2) k-center-greedy loop even when
   ``sampling_ratio == 1.0``, which would make the 100 ``ratio=1`` cells of
   the nylon grid impractically slow. :func:`_install_ratio_one_shortcut`
   patches ``subsample_embedding`` on the concrete model instance so that at
   ``ratio >= 1.0`` the full embedding is used as the memory bank directly.
   The nearest-neighbour distance the model computes is unchanged.

The MinMax-normalization compatibility patch (the third release difference
that affects results) lives separately in ``anomalib_patch.py`` and is
applied from :mod:`sewing_ad.runner` before the ``Engine`` is built.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
from anomalib.models import Patchcore

logger = logging.getLogger(__name__)


def _install_ratio_one_shortcut(patchcore_model) -> None:
    """Add a ``sampling_ratio >= 1.0`` shortcut to the coreset sampler.

    Released ``anomalib==1.2.0`` runs the full O(n^2) k-center-greedy loop
    regardless of ratio; it also hard-codes ``self.model = PatchcoreModel(...)``
    inside ``Patchcore.__init__``, so subclassing ``PatchcoreModel`` would not
    take effect. Instead we patch ``subsample_embedding`` on the *instance*:
    when the sampling ratio is >= 1.0 the memory bank is set to the full
    embedding tensor directly, skipping the greedy loop entirely.

    The metric is identical either way (``num_neighbors`` search over the full
    set of embeddings); only the un-needed O(n^2) greedy pass is skipped.
    """
    original = patchcore_model.subsample_embedding

    def subsample_embedding(embedding: torch.Tensor, sampling_ratio: float) -> None:
        if float(sampling_ratio) >= 1.0:
            logger.info(
                "coreset ratio == 1.0: using the full embedding as the memory "
                "bank (k-center-greedy skipped)."
            )
            patchcore_model.memory_bank = embedding
        else:
            original(embedding, sampling_ratio)

    # Instance-level attribute shadows the bound method; anomalib calls this
    # positionally as ``self.model.subsample_embedding(embeddings, ratio)``.
    patchcore_model.subsample_embedding = subsample_embedding


def _load_ft_weights(
    model: Patchcore,
    ft_weight_path: str | Path,
    *,
    allow_unsafe_weights: bool = False,
) -> None:
    """Load fine-tuned backbone weights into the PatchCore feature extractor.

    Direct state-dict load (does NOT go through anomalib's ``__AT__``
    backbone-string mechanism):

    1. ``torch.load`` the checkpoint (``weights_only=True`` — fail-closed).
    2. Strip a leading ``module.`` prefix from every key (DataParallel).
    3. Drop classifier (``fc.*``) keys — PatchCore only uses the backbone.
    4. ``load_state_dict(filtered, strict=False)`` into
       ``model.model.feature_extractor.feature_extractor`` (the timm backbone).

    The ``load_state_dict`` result is **inspected**:

    - ``missing_keys`` non-empty  -> the backbone needs keys the checkpoint
      does not provide == weight/backbone mismatch -> raise immediately.
      ``strict=False`` must not silently swallow a wrong checkpoint.
    - ``unexpected_keys`` are tolerated (they are the downstream layers, e.g.
      ``layer3``/``layer4``, that ``features_only`` trimmed off) but logged.

    Security: the checkpoint is loaded with ``weights_only=True``. There is
    **no automatic fallback** to ``weights_only=False`` — that would allow
    arbitrary pickle execution from a downloaded file. Set
    ``allow_unsafe_weights=True`` explicitly (opt-in) only for weights you
    fully trust.
    """
    ft_weight_path = Path(ft_weight_path)
    weights_only = not allow_unsafe_weights
    if not weights_only:
        logger.warning(
            "Loading %s with weights_only=False — this executes arbitrary "
            "pickle code. Only do this for weights you fully trust.",
            ft_weight_path,
        )

    state = torch.load(ft_weight_path, map_location="cpu", weights_only=weights_only)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    filtered: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        if new_key.startswith("fc."):
            continue  # classifier head — PatchCore uses the backbone only
        filtered[new_key] = value

    backbone = model.model.feature_extractor.feature_extractor
    result = backbone.load_state_dict(filtered, strict=False)

    if result.missing_keys:
        raise RuntimeError(
            f"FT weight load failed for {ft_weight_path}: the backbone requires "
            f"{len(result.missing_keys)} key(s) that the checkpoint does not "
            f"provide (weight/backbone mismatch). Missing (first 10): "
            f"{result.missing_keys[:10]}"
        )
    if result.unexpected_keys:
        logger.info(
            "FT weight load: %d unexpected key(s) ignored (downstream layers "
            "trimmed by features_only, e.g. layer3/layer4): %s ...",
            len(result.unexpected_keys),
            result.unexpected_keys[:5],
        )
    logger.info("Loaded FT backbone weights from %s", ft_weight_path)


def build_ft_patchcore(
    *,
    backbone: str,
    layers: Sequence[str],
    ratio: float,
    neighbors: int,
    ft_weight_path: str | Path | None = None,
    pre_trained: bool = True,
    allow_unsafe_weights: bool = False,
) -> Patchcore:
    """Build a PatchCore model for the sewing-anomaly experiments.

    Parameters
    ----------
    backbone:
        Plain timm backbone name, e.g. ``"resnet18"``. **Never** an
        ``__AT__`` string — see the module docstring.
    layers:
        Continuous block of backbone layers, e.g. ``["layer1", "layer2"]``.
    ratio:
        ``coreset_sampling_ratio``. ``ratio == 1.0`` triggers the restored
        full-embedding shortcut.
    neighbors:
        ``num_neighbors`` for the nearest-neighbour anomaly score.
    ft_weight_path:
        Path to a fine-tuned backbone checkpoint. ``None`` (finetuning=False)
        keeps the ImageNet backbone as-is.
    pre_trained:
        Fixed to ``True``. Both finetuned and non-finetuned cells need the
        backbone to be created with ImageNet weights — for the FT cells, the
        FT state dict is loaded on top via :func:`_load_ft_weights`; for the
        non-FT cells, the ImageNet weights are the backbone the paper used.
    allow_unsafe_weights:
        Opt-in; load FT checkpoints with ``weights_only=False``. Default off.

    Note: with ``pre_trained=True`` and ``ft_weight_path=None`` (the non-FT
    cells), timm downloads the ImageNet weights from the network on first use.
    """
    model = Patchcore(
        backbone=backbone,
        pre_trained=pre_trained,
        layers=list(layers),
        coreset_sampling_ratio=ratio,
        num_neighbors=neighbors,
    )

    # Add the ratio==1.0 shortcut on the concrete model instance.
    _install_ratio_one_shortcut(model.model)

    if ft_weight_path is not None:
        _load_ft_weights(
            model,
            ft_weight_path,
            allow_unsafe_weights=allow_unsafe_weights,
        )

    return model

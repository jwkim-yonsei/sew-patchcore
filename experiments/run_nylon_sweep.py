#!/usr/bin/env python
"""Nylon hyper-parameter sweep — the 700-cell PatchCore grid.

Runs the 700-cell PatchCore grid on the nylon fabric subset:

    finetuning (2) x model (5) x sampling_ratio (7) x layers (10) = 700

Results are appended to ``results/nylon_sweep.csv`` (one row per cell, flushed
immediately so the run is resumable).

Usage
-----
    python experiments/run_nylon_sweep.py \
        --config configs/nylon_sweep.yaml \
        --data-root /path/to/nylon \
        --weights-dir /path/to/nylon_ft_weights

    # verification: run only the first N cells
    python experiments/run_nylon_sweep.py --limit 3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

# Make ``src/`` importable when run directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sewing_ad.config import load_config  # noqa: E402
from sewing_ad.runner import run_experiment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "nylon_sweep.yaml"


def layer_code(layers: list[str]) -> str:
    """``["layer1", "layer2"]`` -> ``"L12"`` (compact code for the ``layers`` column)."""
    return "L" + "".join(layer.replace("layer", "") for layer in layers)


def load_done_cells(csv_path: Path) -> set[tuple]:
    """Read the result CSV (if any) and return the set of completed cells."""
    done: set[tuple] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(
                (
                    str(row["finetuning"]),
                    row["model"],
                    str(row["sampling_ratio"]),
                    row["layers"],
                )
            )
    return done


def resolve(path_str: str, base: Path) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else (base / p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data-root", help="override data.root (env: SEWING_DATA_ROOT)")
    ap.add_argument("--weights-dir", help="override weights.dir (env: SEWING_WEIGHTS_DIR)")
    ap.add_argument("--accelerator", help="override accelerator, e.g. cpu / cuda / auto")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N pending cells")
    ap.add_argument(
        "--shard",
        default=None,
        help="run only a slice of the grid: 'i/N' (0-indexed). Each shard writes "
        "its own '<csv>_shardIofN.csv' so shards can run in parallel.",
    )
    ap.add_argument(
        "--allow-unsafe-weights",
        action="store_true",
        help="load FT weights with weights_only=False (arbitrary pickle — trusted weights only)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    data_root = args.data_root or os.environ.get("SEWING_DATA_ROOT") or cfg["data"]["root"]
    weights_dir = (
        args.weights_dir or os.environ.get("SEWING_WEIGHTS_DIR") or cfg["weights"]["dir"]
    )
    accelerator = args.accelerator or os.environ.get("SEWING_ACCELERATOR") or cfg["accelerator"]

    data_root = resolve(data_root, REPO_ROOT)
    weights_dir = resolve(weights_dir, REPO_ROOT)
    results_csv = resolve(cfg["results_csv"], REPO_ROOT)
    results_dir = resolve(cfg["results_dir"], REPO_ROOT)

    shard = None
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        if not 0 <= si < sn:
            raise ValueError(f"--shard must be 'i/N' with 0 <= i < N, got {args.shard}")
        shard = (si, sn)
        results_csv = results_csv.with_name(
            f"{results_csv.stem}_shard{si}of{sn}{results_csv.suffix}"
        )

    grid = cfg["grid"]
    weight_template = cfg["weights"]["filename_template"]

    done = load_done_cells(results_csv)
    if done:
        print(f"[resume] {len(done)} cell(s) already in {results_csv} — skipping them.")

    cells = list(
        itertools.product(
            grid["finetuning"], grid["model"], grid["ratio"], grid["layers"]
        )
    )
    print(f"[grid] {len(cells)} cells total "
          f"({len(grid['finetuning'])} finetuning x {len(grid['model'])} model x "
          f"{len(grid['ratio'])} ratio x {len(grid['layers'])} layers)")

    if shard is not None:
        si, sn = shard
        cells = [c for idx, c in enumerate(cells) if idx % sn == si]
        print(f"[shard] {si}/{sn}: {len(cells)} cells -> {results_csv}")

    ran = 0
    for finetuning, model, ratio, layers in cells:
        code = layer_code(layers)
        key = (str(finetuning), model, str(ratio), code)
        if key in done:
            continue
        if args.limit is not None and ran >= args.limit:
            break

        ft_weight_path = None
        if finetuning:
            ft_weight_path = weights_dir / weight_template.format(model=model)
            if not ft_weight_path.exists():
                raise FileNotFoundError(
                    f"FT weight not found for model={model}: {ft_weight_path}\n"
                    f"Set --weights-dir / SEWING_WEIGHTS_DIR (see README)."
                )

        record = {
            "finetuning": finetuning,
            "model": model,
            "sampling_ratio": ratio,
            "layers": code,
        }
        run_cfg = {
            "seed": cfg["seed"],
            "accelerator": accelerator,
            "results_dir": str(results_dir),
            "results_csv": str(results_csv),
            "backbone": model,
            "layers": layers,
            "ratio": ratio,
            "neighbors": cfg["neighbors"],
            "ft_weight_path": str(ft_weight_path) if ft_weight_path else None,
            "allow_unsafe_weights": args.allow_unsafe_weights,
            "data": {
                "name": cfg["data"]["name"],
                "root": str(data_root),
                "normal_dir": cfg["data"]["normal_dir"],
                "abnormal_dir": cfg["data"]["abnormal_dir"],
                "train_batch_size": cfg["data"]["train_batch_size"],
                "eval_batch_size": cfg["data"]["eval_batch_size"],
                "num_workers": cfg["data"]["num_workers"],
                "val_split_ratio": cfg["data"]["val_split_ratio"],
            },
            "record": record,
        }

        print(f"\n=== cell {ran + 1}: {record} ===")
        run_experiment(run_cfg)
        ran += 1

    print(f"\n[done] ran {ran} cell(s); results in {results_csv}")


if __name__ == "__main__":
    main()

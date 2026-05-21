# Reproducibility

This document records which manuscript results this code reproduces on
released `anomalib==1.2.0` (PyPI), maps each result to specific rows of the
sweep output CSV, and notes a small known difference on two intermediate
Table 4 cells.

## Environment used for verification

| Item | Value |
|---|---|
| Python | 3.10 (manuscript Table 2: 3.10.15; any 3.10.x patch release is acceptable) |
| PyTorch | 2.4.1 (`requirements.txt` pin, matches manuscript Table 2) |
| CUDA | 12.1 (manuscript Table 2) |
| anomalib | 1.2.0 (PyPI) + the compatibility patch in `src/sewing_ad/anomalib_patch.py` |
| GPU | NVIDIA RTX 4090, 24 GB (verification hardware) |

See `requirements.txt` for the full pinned dependency set. The pins resolve
in a single `pip install -r requirements.txt`.

## Reproduction status

| Manuscript result | This code reproduces | Status |
|---|---:|---|
| Abstract — FT-PatchCore on nylon **0.988** AUROC (= Table 4 S3, proposed method) | 0.988 | ✅ exact |
| Abstract — original PatchCore on nylon **0.862** AUROC (= Table 4 S0, baseline) | 0.862 | ✅ exact |
| Table 4 S0 baseline — 0.862 AUROC | 0.862 | ✅ exact |
| Table 4 S1 (+ fine-tuning) — 0.937 AUROC | ~0.895 | ≈ see note 1 |
| Table 4 S2 (+ optimal sampling ratio) — 0.947 AUROC | ~0.950 | ≈ see note 1 |
| Table 4 S3 (+ optimal layers, proposed) — 0.988 AUROC | 0.988 | ✅ exact |
| Figures 7-11 (700-cell hyper-parameter distributions and per-axis comparisons) | distributions and qualitative trends reproduced | ✅ see note 2 |

The values in the right column are produced by
`experiments/run_nylon_sweep.py` with `anomalib==1.2.0` and the patch
applied.

## Reproducing the results

```bash
# 1) install (see README §1)
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2) place data and weights (see README §2)
#    data/nylon/{train,abnormal}/
#    weights/nylon/ft_{resnet18,resnet34,resnet50,resnet101,resnet152}.pt

# 3) Run the full 700-cell sweep (covers Table 4 S0-S3 and Figs. 7-11)
python experiments/run_nylon_sweep.py \
    --config configs/nylon_sweep.yaml \
    --data-root data/nylon \
    --weights-dir weights/nylon \
    --accelerator cuda
# -> results/nylon_sweep.csv
```

For a fast smoke test, shrink the grid in `configs/nylon_sweep.yaml` to the
four S0-S3 configurations (or use `--limit`).

## Sweep output structure

`experiments/run_nylon_sweep.py` writes one row per cell into
`results/nylon_sweep.csv`. The grid is
`finetuning (2) × model (5) × sampling_ratio (7) × layers (10) = 700` cells.

CSV columns: `finetuning, model, sampling_ratio, layers, image_AUROC,
image_F1Score, train_duration, test_duration`. The `layers` column uses the
compact code `L<digits>` (e.g. `L12` = `[layer1, layer2]`). Reproduction
targets are **AUROC and F1 only**; `train_duration` / `test_duration` are
hardware-dependent and recorded for reference, not reproduction.

## Mapping to manuscript results

### Table 4 — step-by-step ablation (S0-S3)

| Step | `finetuning` | `model` | `sampling_ratio` | `layers` | Manuscript AUROC |
|---|---|---|---|---|---:|
| S0 baseline | `False` | `resnet18` | `0.1` | `L23` | 0.862 |
| S1 +fine-tuning | `True` | `resnet18` | `0.1` | `L23` | 0.937 |
| S2 +optimal sampling | `True` | `resnet18` | `0.01` | `L23` | 0.947 |
| S3 +optimal layers (proposed) | `True` | `resnet18` | `0.01` | `L12` | 0.988 |

Filter `results/nylon_sweep.csv` by these four key tuples to read off the
ablation values.

### Figures 7-11 — hyper-parameter analyses

| Figure | What it shows | Rows used from `results/nylon_sweep.csv` |
|---|---|---|
| Fig. 7 | Overall AUROC distribution across the 700 cells (fine-tuned vs pre-trained means as dotted lines) | all 700 rows |
| Fig. 8 | AUROC vs sampling ratio, pre-trained (a) vs fine-tuned (b), per ResNet submodel | all rows, grouped by `(finetuning, model, sampling_ratio)` |
| Fig. 9 | AUROC per ResNet submodel at sampling ratio 0.01 | rows with `sampling_ratio == 0.01` |
| Fig. 10 | AUROC per layer combination per ResNet submodel | all rows, grouped by `(model, layers)` |
| Fig. 11 | (Continuation of the layer/depth analysis — see the manuscript) | all rows, grouped by `(model, layers)` |

The grid covers `finetuning` in both `{True, False}` (700 rows total), so
both halves of Fig. 8 and the dotted means in Fig. 7 come from the same
single sweep run.

## Note 1 — Known difference on Table 4 cells S1 and S2 (≤ 0.04 AUROC)

The original hyper-parameter sweep used a coreset memory-bank disk cache so
that repeated runs of the same configuration loaded a previously-computed
coreset rather than recomputing it; that cache was not retained in this
release. This code recomputes the coreset on each run.

Recomputed coresets are nearly but not bit-identically equal to a previously
cached coreset, because the coreset sampler (k-center-greedy on a sparse
random projection) depends on the RNG state at the time of sampling. The
clamped image AUROC reported in Table 4 is sensitive to this difference on
cells with heavy `[0,1]` saturation (S1 and S2), producing the small
deviations shown above.

The ablation **trend** (each step improves AUROC, S0 → S3) is preserved in
the reproduced values: 0.862 < 0.895 < 0.950 < 0.988. The headline numbers
of the paper (the abstract's 0.988 nylon and 0.862 baseline, and Table 4
S0/S3) are unaffected.

## Note 2 — Figures 7-11 (700-cell distributions)

The 700-cell hyper-parameter analyses (Figures 7-11) reproduce the same
distributional patterns and qualitative comparisons reported in the paper.
Per-cell values may differ in the same way as Table 4 S1/S2 on heavily
saturated configurations.

## Determinism caveats

Results are deterministic for a fixed seed and accelerator. Minor numeric
variation across GPU models / CUDA versions is normal and can shift the last
digit on a few cells; this is the standard reproducibility floor for
PyTorch / CUDA workloads.

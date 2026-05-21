# Nylon 700-cell PatchCore hyper-parameter sweep

Computer code for the Scientific Reports paper

> **An efficient anomaly detection method for the industrial sewing process
> using PatchCore and fine-tuned ResNet** — Kim, J., Woo, J.-H., Jung, W.-K.,
> Kim, H. *Scientific Reports* (2026).

This archive contains a single experiment — the **700-cell PatchCore
hyper-parameter sweep on the nylon fabric subset** — which produces the
results behind:

- The 700 parametric experimental tests reported in the Methods section.
- **Figures 7-10** (AUROC distributions and per-axis comparisons across
  finetuning, sampling ratio, ResNet depth, and layer combination) and
  **Fig. 11** (per-submodel inference time).
- **Table 4** S0-S3 ablation (each of the four cells is one row of the sweep
  CSV — see `REPRODUCIBILITY.md`).

The code runs on the **released `anomalib==1.2.0`** from PyPI with a small
compatibility patch — see [§ 4](#4-anomalib-120-compatibility-patch). For
the reproduction status of each manuscript result, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 1. Installation

Supported Python: **3.10 only** (the dependency set resolves differently on
other versions).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See the comments in `requirements.txt` for why each pin is needed.

---

## 2. Data and fine-tuned weights

The nylon image subset and the fine-tuned ResNet checkpoints are **not**
included in this archive (too large to redistribute via Git). They are
hosted separately:

- **Dataset** (StitchingNet, nylon subset) — Kaggle: <https://www.kaggle.com/datasets/hyungjung/stitchingnet-dataset>
- **Fine-tuned weights** (5 ResNet checkpoints) — Zenodo: <https://doi.org/10.5281/zenodo.20322967>

### 2.1 Nylon image dataset

The nylon fabric subset of the StitchingNet sewing-defect data
([Kaggle](https://www.kaggle.com/datasets/hyungjung/stitchingnet-dataset);
see Methods, "Implementation details" for split sizes):

```
data/nylon/
├── train/        # normal (good) images
└── abnormal/     # anomalous images (provides the test split)
```

> **Note on dataset version.** The Kaggle release may grow over time, so the
> current download may contain images added after the paper's experiments.
> The paper used a frozen snapshot of **105 normal + 989 defective** nylon
> images (Methods, "Implementation details"); after arranging the Kaggle
> download into the layout above, you should see those counts in `train/`
> and `abnormal/` respectively. The train/val split inside `train/` is
> generated at runtime by `val_split_ratio: 0.2` in
> `configs/nylon_sweep.yaml`. Minor numerical differences from the paper's
> reported AUROCs are expected if the current Kaggle archive is larger than
> this snapshot.

### 2.2 Fine-tuned ResNet backbones

Five fine-tuned ResNet checkpoints, one per depth axis of the grid
(ResNet-18, 34, 50, 101, 152), are loaded into the PatchCore feature
extractor when `finetuning: true`. Download the
[`nylon_ft_full.zip`](https://doi.org/10.5281/zenodo.20322967) archive from
Zenodo (DOI `10.5281/zenodo.20322967`) and unzip into `weights/`:

```
weights/nylon/
├── ft_resnet18.pt
├── ft_resnet34.pt
├── ft_resnet50.pt
├── ft_resnet101.pt
└── ft_resnet152.pt
```

Filename template (`ft_{model}.pt`) is defined in `configs/nylon_sweep.yaml`.
Non-fine-tuned cells skip these files. `data/`, `weights/`, and `results/`
are git-ignored. Weights are loaded with `torch.load(..., weights_only=True)`
(fail-closed); `--allow-unsafe-weights` opts into `weights_only=False` and is
off by default.

---

## 3. Running the experiment

Paths can come from the config file, a CLI flag, or an environment variable
(CLI > env > config):

| Override | CLI flag | Env var |
|---|---|---|
| data root | `--data-root` | `SEWING_DATA_ROOT` |
| weights dir | `--weights-dir` | `SEWING_WEIGHTS_DIR` |
| accelerator | `--accelerator` | `SEWING_ACCELERATOR` |

### Full sweep (700 cells)

```bash
python experiments/run_nylon_sweep.py \
    --config configs/nylon_sweep.yaml \
    --data-root data/nylon \
    --weights-dir weights/nylon \
    --accelerator cuda
```

Results stream into `results/nylon_sweep.csv` (one row per cell, flushed
immediately). Re-running skips cells already in the CSV, so an interrupted
run resumes. Use `--limit N` to run only the first N pending cells (handy for
a quick check), or `--shard i/N` to run a slice of the grid (each shard
writes its own `*_shardIofN.csv`, suitable for parallel workers).

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for which rows of
`results/nylon_sweep.csv` correspond to the Table 4 cells (S0-S3) and
Figs. 7-11.

---

## 4. anomalib 1.2.0 compatibility patch

This code targets the released `anomalib==1.2.0` (PyPI). Two minor
implementation details (FT-backbone direct load, and a `ratio==1.0`
shortcut bypassing the O(n²) k-center-greedy loop) live in
`src/sewing_ad/ft_patchcore.py`. A small monkeypatch in
`src/sewing_ad/anomalib_patch.py` aligns the MinMax normalization callback
with the anomalib version under which the paper was evaluated; the
underlying PatchCore anomaly scores are unchanged. The runner applies it
once per cell before building the `Engine` (idempotent). See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and the patch module's docstring
for details.

---

## 5. License & citation

Licensed under **Apache-2.0** (`LICENSE`), matching anomalib.
See `CITATION.cff` for citation metadata.

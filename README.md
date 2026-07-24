# RSCD Visual Road Friction Affordance Estimation

This repository contains the current verified implementation of a visual road-surface classification and friction-affordance model for RSCD-style road images.

The released method is the best run that has already completed the full train/validation/test protocol in the local experiment record:

- dataset protocol: RSCD full train/val/test manifests
- train split: 958,941 images
- validation split: 19,860 images
- test split: 49,500 images
- input size: 192 x 192, letterbox resize
- model parameters: 32.49M total, about 1.09M trainable under the S7 prefix-tuning setup
- current verified self-contained S7 checkpoint Top-1: 90.632%
- current verified self-contained S7 checkpoint Macro-F1: 88.920%
- best recorded parent-checkpoint plus source-router inference Top-1: 90.640%
- weakest class: `water_concrete_slight`, F1 = 75.693%

The active later experiments are not reported here until they finish full training and full 49,500-image test evaluation.

## Method Name

The method is organized as **C3-FaRNet**: Coupled Conditioned Friction-Affordance Road Network.

The core idea is that the 27 RSCD classes are not independent labels. Most classes are combinations of three physical/visual factors:

- friction state: dry, wet, water, fresh snow, melted snow, ice
- road material: asphalt, concrete, mud, gravel, or none
- roughness state: smooth, slight, severe, or none

Instead of asking a classifier to memorize 27 flat categories, the model uses this factor structure to learn both the single factors and the difficult coupled boundaries, such as `wet + concrete + slight` versus `water + concrete + slight`.

## Repository Layout

```text
configs/
  c3_farnet/
    current_best_s7_public.yaml
  data/
    local_paths.example.yaml
docs/
  algorithm.md
  data_and_reproduction.md
  results_current_best.md
results/
  current_best_s7/
src/
  friction_affordance/
train.py
validate.py
test.py
```

Large datasets, full local manifests, predictions, and logs are intentionally not committed. The selected S7 final checkpoint, direct parent checkpoint, and dry-concrete teacher checkpoint are committed through Git LFS.

## Install

```bash
conda env create -f environment-faf-paper.yml
conda activate faf_paper
pip install -e .
```

Or install minimal dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

Prepare RSCD in the manifest format described in [docs/data_and_reproduction.md](docs/data_and_reproduction.md).

The training code expects CSV manifests with these columns:

```text
image_path,split,dataset,class_label,domain_id,friction_label,material_label,unevenness_label,wetness_label,snow_label,risk_label,mu_low,mu_high
```

For public use, copy:

```text
configs/data/local_paths.example.yaml
```

to:

```text
configs/data/local_paths.yaml
```

and edit the paths for your local RSCD download.

## Train

Use the public current-best config after generating full manifests:

```bash
python scripts/build_manifests.py --config configs/data/local_paths.yaml --out-dir data/manifests_full
python train.py --config configs/c3_farnet/current_best_s7_public.yaml
```

The historical verified S7 run used warm-start teacher/checkpoint files from earlier local screening runs. Those selected binary files are included through Git LFS under [checkpoints](checkpoints). Run `git lfs pull` before reproducing checkpoint-based training or evaluation.

To reproduce the direct parent model training recipe:

```bash
python train.py --config configs/c3_farnet/parent_errorgate_paircal_screen_public.yaml
```

The full parent training chain is documented in [docs/parent_model_training.md](docs/parent_model_training.md).

## Evaluate

```bash
python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint outputs/current_best_s7/best_checkpoint.pth
```

## Current Verified Result

The result files in [results/current_best_s7](results/current_best_s7) record the current full-test evidence:

- `metrics_summary.json`: full-test summary metrics
- `per_class_metrics.csv`: precision/recall/F1/support for all 27 classes
- `confusion_matrix.csv`: 27-class confusion matrix
- `hard_pair_metrics.csv`: hard-pair boundary statistics
- `history.json`: training history for the verified run

The detailed warm-start chain, parent checkpoint metrics, source-router inference result, and the meaning of the `90.045%` validation number are documented in [docs/s7_training_lineage.md](docs/s7_training_lineage.md).

The complete list of S7-related source files, configs, checkpoints, result evidence and intentionally excluded files is documented in [docs/s7_release_inventory.md](docs/s7_release_inventory.md).

## Algorithm Explanation

- English explanation: [docs/algorithm.md](docs/algorithm.md)
- Chinese explanation: [docs/algorithm_zh.md](docs/algorithm_zh.md)
- Parent training recipe: [docs/parent_model_training.md](docs/parent_model_training.md)
- Complete S7 release inventory: [docs/s7_release_inventory.md](docs/s7_release_inventory.md)

## Scientific Boundary

This project estimates visual road-surface states and visual friction affordance. RSCD labels are visual proxy labels, not synchronized tire-force or friction-meter measurements. Therefore, the method should be described as:

> visual road friction-affordance estimation from road-surface images

not as direct measured tire-road friction coefficient estimation.

# RSCD Visual Road Friction Affordance Estimation

This repository contains the current verified implementation of a visual road-surface classification and friction-affordance model for RSCD-style road images.

The released method is the best run that has already completed the full train/validation/test protocol in the local experiment record:

- dataset protocol: RSCD full train/val/test manifests
- train split: 958,941 images
- validation split: 19,860 images
- test split: 49,500 images
- input size: 192 x 192, letterbox resize
- model parameters: 32.49M total, about 1.09M trainable under the S7 prefix-tuning setup
- current verified Top-1: 90.632%
- current verified Macro-F1: 88.920%
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
    c3_farnet_errorgate_paircal_full_eval.yaml
    c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709.yaml
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

Large datasets, full local manifests, checkpoints, predictions, and logs are intentionally not committed.

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

The historical verified S7 run used warm-start teacher/checkpoint files from earlier local screening runs. Those binary files are not included in GitHub. The complete model implementation is released; exact historical checkpoint reproduction requires the same warm-start checkpoints.

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

## Scientific Boundary

This project estimates visual road-surface states and visual friction affordance. RSCD labels are visual proxy labels, not synchronized tire-force or friction-meter measurements. Therefore, the method should be described as:

> visual road friction-affordance estimation from road-surface images

not as direct measured tire-road friction coefficient estimation.

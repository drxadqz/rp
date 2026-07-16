# Data and Reproduction

## Dataset

The current verified result uses RSCD full train/validation/test splits:

| split | images |
|---|---:|
| train | 958,941 |
| validation | 19,860 |
| test | 49,500 |

The dataset itself is not redistributed in this repository. Download RSCD from its official source and prepare the local paths.

## Manifest Format

The code reads CSV manifests. Required columns:

```text
image_path,split,dataset,class_label,domain_id,friction_label,material_label,unevenness_label,wetness_label,snow_label,risk_label,mu_low,mu_high
```

Important columns:

- `image_path`: absolute or relative path to an image
- `split`: train, val, or test
- `dataset`: usually `rscd`
- `class_label`: one of the 27 RSCD classes
- `friction_label`: dry, wet, water, fresh_snow, melted_snow, ice
- `material_label`: asphalt, concrete, mud, gravel, none
- `unevenness_label`: smooth, slight, severe, none
- `mu_low`, `mu_high`: weak visual friction-risk interval derived from road-state labels

## Generate Manifests

Create a local path file:

```bash
cp configs/data/local_paths.example.yaml configs/data/local_paths.yaml
```

Edit `configs/data/local_paths.yaml`, then run:

```bash
python scripts/build_manifests.py --config configs/data/local_paths.yaml --out-dir data/manifests_full
```

## Train Current Public Config

```bash
python train.py --config configs/c3_farnet/current_best_s7_public.yaml
```

## Test

```bash
python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint outputs/current_best_s7/best_checkpoint.pth
```

## Reproduction Note

The exact verified historical S7 run used warm-start checkpoints produced by earlier local screening runs. The binary checkpoints are not included because GitHub is not suitable for large experiment weights. The released source code contains the complete architecture and training/evaluation pipeline.

For exact bit-level reproduction, provide the same warm-start checkpoints in the config fields:

```yaml
train:
  resume_from: /path/to/anchor/best_checkpoint.pth
  teacher_checkpoint: /path/to/teacher/best.pt
```

For a clean public run, leave those fields empty and train from the released config.

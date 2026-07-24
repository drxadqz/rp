# Direct Parent Model Training Recipe

This document explains how the direct parent checkpoint of the formal S7 model was trained.

The uploaded parent checkpoint is:

```text
checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

It is the direct warm-start checkpoint used by the formal S7 run. In the S7 config, it appears as:

```text
train.resume_from: checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

## One-Sentence Summary

The parent model was obtained by loading the dry-concrete VOR teacher, adding the C3-FaRNet tensor-coupling backbone adapters and hard-pair error-gated calibration head, then selectively training only those mechanism modules for 2 screen epochs on balanced RSCD samples.

## Exact Public Config

The portable public training config is:

```text
configs/c3_farnet/parent_errorgate_paircal_screen_public.yaml
```

The historical resolved config and evidence files are stored in:

```text
results/s7_lineage/parent_errorgate_paircal_screen/
```

## Training Command

After preparing RSCD manifests:

```bash
git lfs pull
python scripts/build_manifests.py --config configs/data/local_paths.yaml --out-dir data/manifests_full
python train.py --config configs/c3_farnet/parent_errorgate_paircal_screen_public.yaml
```

The config expects the teacher checkpoint:

```text
checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt
```

This file is committed with Git LFS.

## Data Protocol

The parent model used the RSCD full manifest files as the data source:

```text
data/manifests_full/rscd_prepared_train.csv
data/manifests_full/rscd_prepared_val.csv
data/manifests_full/rscd_prepared_test.csv
```

The historical local split sizes were:

| split | images |
|---|---:|
| train | 958,941 |
| validation | 19,860 |
| test | 49,500 |

The parent was a screen-stage run, so it did not consume all 958,941 training images in each epoch. It used:

```text
balanced_sampling: true
samples_per_epoch: 12000
epochs: 2
```

This means each epoch drew 12,000 balanced training samples from the full train manifest. The purpose was to tune the new mechanism modules quickly while reducing the risk of damaging the already strong teacher representation.

Validation and test during the screen run were capped:

```text
max_val_samples_per_class: 120
max_test_samples_per_class: 120
```

With 27 classes, this gives `27 x 120 = 3,240` samples for the capped screen evaluation.

## Initialization

The parent training started from the dry-concrete VOR teacher:

```text
resume_from: checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt
teacher_checkpoint: checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt
teacher_backbone: convnext_tiny
teacher_head_type: linear
```

The same checkpoint had two roles:

- `resume_from` initialized the student model by flexible weight loading.
- `teacher_checkpoint` supplied frozen teacher predictions for anchor-consistency protection.

In plain language: the parent did not start from random weights. It started from a stable ConvNeXt-based road-surface model, then learned extra mechanisms only where they were expected to help.

## Architecture

The parent model used:

```text
backbone: convnext_tiny_gate_calibrated_tensor_coupling_concrete_film_rough_stem
head_type: hardpair_error_gated_calibrated
```

The architecture includes these modules:

| module | code location | role |
|---|---|---|
| training/evaluation engine | `src/friction_affordance/c3_experiment.py` | loads configs, manifests, checkpoints, teacher, losses, training loop and metrics |
| C3-FaRNet classifier | `src/friction_affordance/models/c3_farnet.py` | fuses backbone features, physics evidence, local field and hard-pair calibration |
| task-adapted ConvNeXt backbone | `src/friction_affordance/models/backbone.py` | implements the gate-calibrated tensor-coupling road stem |
| PhysicsTexture | `src/friction_affordance/models/texture.py` | computes wetness, darkness, roughness and texture evidence from RGB images |
| RSCD factor parser | `src/friction_affordance/rscd_factors.py` | maps the 27 labels into friction/material/roughness factors |

The parent model is not a plain ConvNeXt. It is ConvNeXt plus task-conditioned road mechanisms.

## Trainable Parameters

Only these prefixes were trainable:

```text
backbone.gate_calibrated_tensor_coupling_banks
pairwise_hardpair_experts
pairwise_hardpair_error_gates
```

The same prefix set had about 1.09M trainable parameters in the S7 log:

```text
backbone.gate_calibrated_tensor_coupling_banks: 366,677
pairwise_hardpair_experts: 431,152
pairwise_hardpair_error_gates: 295,632
total: 1,093,461
```

Everything outside those prefixes was kept fixed. This is why the parent training is better described as selective mechanism tuning rather than full network retraining.

## Loss Design

The parent objective can be summarized as:

```text
L = L_CE + L_focus + L_anchor + L_no_flip + L_gate
```

Where:

- `L_CE` is the normal 27-class cross-entropy classification loss.
- `L_focus` adds extra CE weight to the four difficult concrete water/wet boundary classes.
- `L_anchor` is KL-style anchor consistency against the frozen teacher.
- `L_no_flip` protects non-focus classes from being changed when the teacher is confident.
- `L_gate` supervises the hard-pair error gate so the correction head activates mainly on pair-confusing samples.

The hard focus classes were:

```text
water_concrete_slight
wet_concrete_slight
water_concrete_severe
wet_concrete_severe
```

These are difficult because water film, wet concrete texture and roughness severity are visually coupled. The parent model therefore learns pair-specific corrections instead of applying one uniform late classifier to every class.

## Optimizer and Runtime Settings

The historical parent training used:

```text
batch_size: 8
grad_accum_steps: 2
effective batch size: 16
lr: 9e-5
weight_decay: 0.002
epochs: 2
AMP: true
augmentation: true
balanced_sampling: true
samples_per_epoch: 12000
```

The effective batch size is `batch_size x grad_accum_steps = 8 x 2 = 16`.

## Parent Results

Screen-stage result:

| protocol | Top-1 | Macro-F1 | samples | errors |
|---|---:|---:|---:|---:|
| capped screen test | 89.7531% | 89.7238% | 3,240 | 332 |

Full-test evaluation of the parent checkpoint:

| protocol | Top-1 | Macro-F1 | weighted F1 | samples | errors |
|---|---:|---:|---:|---:|---:|
| full RSCD test | 90.6202% | 88.9201% | 90.6357% | 49,500 | 4,643 |

The full-test evaluation command is:

```bash
python test.py \
  --config configs/c3_farnet/parent_errorgate_paircal_full_eval_public.yaml \
  --checkpoint checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

## Relationship to S7

The formal S7 run started from this parent checkpoint:

```text
resume_from: checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

S7 then added the source-reliable boundary router and performed one full-manifest epoch with a smaller learning rate:

```text
lr: 3e-5
samples_per_epoch: 0
```

Here `samples_per_epoch: 0` means the full training split was used.

The `90.0453%` number appears at S7 epoch 0, before S7 fine-tuning:

| stage | split | Top-1 | Macro-F1 | samples |
|---|---|---:|---:|---:|
| S7 initial evaluation after loading parent | full validation | 90.0453% | 88.7798% | 19,860 |

So `90.0453%` is not the standalone parent test result. It is the S7 initial full-validation result after loading the parent checkpoint into the S7 configuration.

## Source Code Completeness

The source code needed to train and evaluate the parent model is included in this repository:

```text
train.py
test.py
validate.py
src/friction_affordance/c3_experiment.py
src/friction_affordance/models/c3_farnet.py
src/friction_affordance/models/backbone.py
src/friction_affordance/models/texture.py
src/friction_affordance/rscd_factors.py
src/friction_affordance/datasets/manifest.py
```

The current commit synchronizes the three largest core implementation files with the latest local project source:

```text
src/friction_affordance/c3_experiment.py
src/friction_affordance/models/c3_farnet.py
src/friction_affordance/models/backbone.py
```


# S7 Model Lineage and Warm-Start Checkpoints

This document records the exact lineage of the current S7-family RSCD models, the meaning of the `90.045%` number, and the uploaded warm-start checkpoints.

For the complete source/config/checkpoint/result file inventory, see [s7_release_inventory.md](s7_release_inventory.md).

## Key Conclusions

1. The direct parent checkpoint does exist:

```text
checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

2. The `90.0453%` Top-1 number is not the standalone parent test result. It is the full-validation initial evaluation of the formal S7 run after loading the parent checkpoint and before fine-tuning.

3. The best self-contained formal S7 checkpoint remains:

```text
checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth
```

Its full-test result is `90.6323%` Top-1 and `88.9197%` Macro-F1 on 49,500 RSCD test images.

4. The highest full-test score found in the local evidence scan is `90.6404%` Top-1. It comes from evaluating the parent checkpoint with the source-reliable boundary router configuration:

```text
configs/c3_farnet/parent_source_reliable_router_s5_public.yaml
```

The historical directory name contains `s7`, but its `config_resolved.yaml` points to the `s5` source-router full-test config. Therefore this should be described as a parent-checkpoint plus source-router inference result, not as a separately trained S7 checkpoint.

## Uploaded Checkpoints

| role | repository path | size | purpose |
|---|---|---:|---|
| final formal S7 checkpoint | `checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth` | 130.24 MB | self-contained formal S7 model |
| direct parent checkpoint | `checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth` | 130.23 MB | warm-start parent for S7; also used by the best router inference result |
| dry-concrete VOR teacher | `checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt` | 114.18 MB | frozen teacher for anchor-consistency protection |

The `.pth` and `.pt` files are tracked with Git LFS.

## Training and Evaluation Chain

### Stage 0: Dry-concrete VOR teacher

The teacher checkpoint is:

```text
checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt
```

This checkpoint was not used as the final classifier. Its role was to protect a useful dry-concrete roughness residual branch during later training. In the recorded run, only the `dry_concrete_roughness_vor_residual` parameters were trained, while most of the anchor network was kept fixed.

The idea is simple: dry concrete roughness errors are easy to damage when we tune the network for wet/water boundary classes. The VOR residual teacher supplies an anchor so that later models do not forget this already useful roughness cue.

### Stage 1: Direct parent model

The direct parent checkpoint is:

```text
checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

The full parent training recipe is documented in [parent_model_training.md](parent_model_training.md). The portable public training config is:

```text
configs/c3_farnet/parent_errorgate_paircal_screen_public.yaml
```

Its architecture is the C3-FaRNet parent before the source-reliable boundary router. The main parts are:

- ConvNeXt-tiny visual backbone with a gate-calibrated tensor-coupling road stem.
- PhysicsTexture branch for hand-designed road-surface statistics, such as wetness, darkness, roughness and Laplacian texture response.
- Semantic-physics attention branch for aligning global visual features with physics-like road cues.
- LocalPhysicsField branch for local road-texture and local wetness evidence.
- Dry-concrete VOR residual branch for dry concrete roughness correction.
- Hard-pair error-gated calibrated head for confusing pairs such as `water_concrete_slight` versus `wet_concrete_slight`.

This parent was trained from the VOR teacher using the full train manifest as the data source, but with `samples_per_epoch=12000` and capped validation/test screening in the screen run. The trainable prefixes were:

```text
backbone.gate_calibrated_tensor_coupling_banks
pairwise_hardpair_experts
pairwise_hardpair_error_gates
```

So the parent was a selective mechanism-tuning run, not a full from-scratch training run.

Parent screen result:

| protocol | Top-1 | Macro-F1 | samples | errors |
|---|---:|---:|---:|---:|
| capped screen test | 89.7531% | 89.7238% | 3,240 | 332 |

Parent full-test evaluation:

| protocol | Top-1 | Macro-F1 | samples | errors |
|---|---:|---:|---:|---:|
| full RSCD test | 90.6202% | 88.9201% | 49,500 | 4,643 |

The corresponding evidence files are stored in:

```text
results/s7_lineage/parent_errorgate_paircal_screen/
results/s7_lineage/parent_errorgate_paircal_full_test/
```

### Stage 2: Parent plus source-reliable boundary router

The source-reliable boundary router is a deterministic/gated boundary correction used at inference. It routes a small number of high-confidence `dry_concrete_smooth` candidates toward `dry_concrete_slight` when the source class is reliable and the boundary margin condition is satisfied.

The historical route was:

```text
source: dry_concrete_smooth
target: dry_concrete_slight
topk: 3
margin: 1.00
source_f1: 0.907292954264524
min_source_f1: 0.90
kind: dry_concrete_roughness
```

Why this helps: the parent model is already strong on `dry_concrete_smooth`, but part of the `dry_concrete_slight` boundary is visually close to smooth concrete. The router only borrows probability mass from a reliable source class under strict top-k/margin conditions, so it can improve that boundary without broadly changing all classes.

Best recorded full-test result:

| protocol | checkpoint | router scale | Top-1 | Macro-F1 | samples | errors |
|---|---|---:|---:|---:|---:|---:|
| full RSCD test | parent checkpoint | 5.0 | 90.6404% | 88.9410% | 49,500 | 4,633 |

This is 10 fewer errors than the parent full-test result and 4 fewer errors than the formal S7 checkpoint. It is the best score found in the local completed full-test records.

Evidence files:

```text
results/s7_lineage/source_router_s7_full_val_initial/
results/s7_lineage/source_router_s7_full_test_initial/
configs/c3_farnet/parent_source_reliable_router_s5_public.yaml
```

Important wording: because this result loads the parent checkpoint and adds the router at evaluation time, it should be described as "parent checkpoint + source-reliable router inference", not as a separately trained standalone S7 checkpoint.

### Stage 3: Formal full-manifest S7 training

The formal S7 run is:

```text
c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709
```

It was trained with:

```text
train images: 958,941
validation images: 19,860
test images: 49,500
image size: 192 x 192, letterbox
batch size: 8
gradient accumulation: 2
learning rate: 3e-5
weight decay: 0.002
epochs: 1
AMP: enabled
balanced sampling: enabled
samples_per_epoch: 0, meaning the full training split was used
```

The formal S7 config loaded:

```text
resume_from: checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
teacher_checkpoint: checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt
```

It then fine-tuned only these prefixes:

```text
backbone.gate_calibrated_tensor_coupling_banks
pairwise_hardpair_experts
pairwise_hardpair_error_gates
```

The losses added on top of normal 27-class cross entropy were:

- focus CE for the hard concrete water/wet classes
- anchor consistency to avoid damaging the teacher's reliable predictions
- anchor no-flip protection for non-focus classes
- anchor error-gate supervision for hard-pair correction

Formal S7 validation history:

| epoch | meaning | Top-1 | Macro-F1 | weighted F1 | WCS F1 | samples | errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | initial evaluation after loading parent | 90.0453% | 88.7798% | 90.1016% | 67.2566% | 19,860 | 1,977 |
| 1 | after one full-manifest epoch | 90.0655% | 88.7908% | 90.1181% | 67.7362% | 19,860 | 1,973 |

This table is where the `90.045%` number comes from.

Formal S7 full-test result:

| protocol | Top-1 | Macro-F1 | weighted F1 | samples | errors | weakest class |
|---|---:|---:|---:|---:|---:|---|
| full RSCD test | 90.6323% | 88.9197% | 90.6539% | 49,500 | 4,637 | `water_concrete_slight`, F1 75.6931% |

Evidence files:

```text
results/s7_lineage/formal_fullmanifest_s7/
results/current_best_s7/
configs/c3_farnet/current_best_s7_public.yaml
configs/c3_farnet/formal_fullmanifest_s7_20260709.yaml
```

## How to Refer to These Results

Use this wording for the safest scientific description:

> The self-contained formal S7 checkpoint reaches 90.6323% Top-1 and 88.9197% Macro-F1 on the full 49,500-image RSCD test split. A parent-checkpoint plus source-reliable-router inference variant reaches the highest recorded full-test Top-1 of 90.6404%, corresponding to 4 fewer errors than the formal S7 checkpoint.

Avoid saying:

```text
the parent model itself reaches 90.045% Top-1
```

because `90.0453%` is the formal S7 initial full-validation result after loading the parent checkpoint, not the standalone parent full-test metric.

## Reproduction Notes

For the formal self-contained S7 checkpoint:

```bash
python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth
```

For the parent plus source-router inference variant:

```bash
python test.py \
  --config configs/c3_farnet/parent_source_reliable_router_s5_public.yaml \
  --checkpoint checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

Before running, make sure Git LFS files are present:

```bash
git lfs pull
```

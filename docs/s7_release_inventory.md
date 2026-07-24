# Complete S7 Release Inventory

This document lists the files in this repository that are needed to understand, reproduce and audit the current best S7-family RSCD result.

## Result Scope

The self-contained formal S7 checkpoint is:

```text
checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth
```

Reported full-test result:

| metric | value |
|---|---:|
| Top-1 | 90.6323% |
| Macro-F1 | 88.9197% |
| weighted F1 | 90.6539% |
| test images | 49,500 |
| errors | 4,637 |
| weakest class | `water_concrete_slight`, F1 75.6931% |

There is also a parent-checkpoint plus source-router inference record:

| metric | value |
|---|---:|
| Top-1 | 90.6404% |
| Macro-F1 | 88.9410% |
| test images | 49,500 |
| errors | 4,633 |

That second number is not a separately trained S7 checkpoint; it is the uploaded parent checkpoint evaluated with a source-reliable router.

## Source Code

Core training and evaluation entry points:

```text
train.py
test.py
validate.py
```

Core S7 implementation:

```text
src/friction_affordance/c3_experiment.py
src/friction_affordance/models/c3_farnet.py
src/friction_affordance/models/backbone.py
src/friction_affordance/models/texture.py
src/friction_affordance/rscd_factors.py
src/friction_affordance/datasets/manifest.py
src/friction_affordance/transforms.py
src/friction_affordance/metrics.py
```

Supporting model modules are also kept under:

```text
src/friction_affordance/models/
```

The current repository source has been synchronized with the latest local project source for the whole `src/friction_affordance` package.

## S7-Related Scripts

The following scripts are included because they were used or are directly useful for S7 reproduction, auditing and diagnosis:

| script | purpose |
|---|---|
| `scripts/build_manifests.py` | builds local RSCD manifests from configured dataset paths |
| `scripts/audit_manifest_labels.py` | checks manifest label/factor consistency |
| `scripts/verify_gpu.py` | verifies CUDA visibility |
| `scripts/fast_c3_eval.py` | fast full-val/full-test C3-FaRNet evaluation; used for source-router evidence |
| `scripts/evaluate_detailed.py` | detailed evaluation report generation |
| `scripts/audit_rscd_run.py` | audits RSCD full-protocol result files |
| `scripts/audit_rscd_protocol_and_metrics.py` | checks metric definitions and local protocol assumptions |
| `scripts/compare_rscd_runs.py` | compares two RSCD result folders |
| `scripts/audit_rscd_candidate_promotion.py` | checks whether a candidate result is promotable under full protocol |
| `scripts/diagnose_s7_anchor_confusions.py` | diagnoses S7 anchor/source-router confusion patterns |
| `scripts/decide_rscd_next_mechanism.py` | produces the next-mechanism decision from S7 failure statistics |

Large exploratory scripts that are not needed for the current S7 release are intentionally not included.

## Configurations

Primary portable configs:

```text
configs/c3_farnet/current_best_s7_public.yaml
configs/c3_farnet/parent_errorgate_paircal_screen_public.yaml
configs/c3_farnet/parent_errorgate_paircal_full_eval_public.yaml
configs/c3_farnet/parent_source_reliable_router_s5_public.yaml
```

Historical/compatibility configs:

```text
configs/c3_farnet/formal_fullmanifest_s7_20260709.yaml
configs/c3_farnet/parent_source_reliable_router_fulltest_s5_20260708.yaml
configs/c3_farnet/c3_farnet_errorgate_paircal_screen.yaml
configs/c3_farnet/c3_farnet_errorgate_paircal_full_eval.yaml
```

The compatibility configs preserve the old local experiment names while redirecting to portable public paths.

## Checkpoints

All checkpoint files are tracked by Git LFS. Run:

```bash
git lfs pull
```

before evaluating or training from checkpoints.

| checkpoint | role | size |
|---|---|---:|
| `checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth` | final formal S7 checkpoint, preferred for reporting | 130.24 MB |
| `checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best.pt` | same formal S7 model payload saved by the training script under the short name | 130.20 MB |
| `checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth` | direct parent checkpoint | 130.23 MB |
| `checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt` | dry-concrete VOR teacher checkpoint | 114.18 MB |

Machine-readable sizes and SHA256 hashes are stored in:

```text
results/s7_lineage/checkpoint_manifest.json
```

## Results and Evidence Files

Compact current-best summary:

```text
results/current_best_s7/
```

Formal S7 full evidence:

```text
results/s7_lineage/formal_fullmanifest_s7/
```

Important files inside that folder:

| file | purpose |
|---|---|
| `config_resolved.yaml` | exact resolved training config from the formal S7 run |
| `history.json` | epoch-0 and epoch-1 validation history |
| `metrics.json` / `test_metrics.json` | full 49,500-image test metrics |
| `per_class_metrics.csv` | precision/recall/F1/support for all 27 classes |
| `confusion_matrix.csv` | full 27-class confusion matrix |
| `hard_pair_metrics.csv` | hard-pair boundary statistics |
| `factor_confusion_summary.json` | friction/material/roughness factor-level confusion summary |
| `predictions_test.csv` | per-image true label, predicted label and confidence for the full test set |
| `water_concrete_slight_diagnosis.json` | weakest-class diagnosis |
| `train_stdout_20260709_074914.log` | formal S7 training stdout |
| `train_stderr_20260709_074914.log` | formal S7 training stderr/progress log |

Parent and source-router evidence:

```text
results/s7_lineage/parent_errorgate_paircal_screen/
results/s7_lineage/parent_errorgate_paircal_full_test/
results/s7_lineage/source_router_s7_full_val_initial/
results/s7_lineage/source_router_s7_full_test_initial/
results/s7_lineage/dry_concrete_vor_teacher/
```

Next-mechanism diagnosis from the current S7 result:

```text
results/s7_lineage/formal_fullmanifest_s7/next_mechanism_decision/
```

This folder records the measured S7 bottleneck: concrete slight/severe roughness boundaries, especially water/wet concrete classes.

## Main Reproduction Commands

Evaluate the formal S7 checkpoint:

```bash
python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth
```

Train from the parent/teacher recipe:

```bash
python train.py --config configs/c3_farnet/current_best_s7_public.yaml
```

Evaluate the direct parent checkpoint:

```bash
python test.py \
  --config configs/c3_farnet/parent_errorgate_paircal_full_eval_public.yaml \
  --checkpoint checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

Evaluate parent plus source-router inference:

```bash
python test.py \
  --config configs/c3_farnet/parent_source_reliable_router_s5_public.yaml \
  --checkpoint checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
```

## Intentionally Not Uploaded

The RSCD image dataset and generated full manifest CSV files are not uploaded.

Reasons:

- the image dataset is large and should be obtained from its original source;
- generated manifests contain local absolute paths that are not portable;
- `scripts/build_manifests.py` and `configs/data/local_paths.example.yaml` are included so manifests can be regenerated locally.

No raw dataset images, compressed dataset archives or unrelated old experiment checkpoints are included in this release.


# Friction-Affordance Field

Heterogeneous weakly supervised visual road-state learning for calibrated tire-road friction risk intervals.

This repository implements the research plan in `../road_friction_vision_research_plan.md`. It is intentionally framed as **visual friction affordance / risk interval estimation**, not as direct ground-truth tire-road friction coefficient estimation. Public visual datasets usually provide road-state proxy labels rather than synchronized tire dynamics or friction-meter measurements.

## What Is Implemented

- Unified road-state ontology for RSCD, RoadSaW, RoadSC, and future road datasets.
- Manifest-based dataset layer with missing-label masks.
- RSCD / RoadSaW / RoadSC scanners.
- Factorized multi-task model:
  - material
  - friction state
  - wetness
  - snow state
  - unevenness
  - friction-risk ordinal class
  - calibrated `mu` interval head
- ConvNeXt/ResNet/EfficientNet backbone support.
- Physics/texture descriptor branch for snow/wet/specular/roughness evidence.
- Weak domain-adversarial head for reducing dataset shortcut learning.
- Masked multi-task classification, interval-censored likelihood, and monotonic regularization.
- Ordinal risk loss and conformal interval calibration.
- Training, evaluation, batch inspection, dataset diagnostics, and data audit scripts.

## Local Data Found On This Machine

Detected core datasets are exposed through one canonical project data folder:

```text
D:\NMI_SPWFM_datasets\friction_affordance_data
```

This folder contains junctions to the existing local datasets, so the data is centralized without duplicating large files.

- RSCD prepared labels: `D:\NMI_SPWFM_datasets\friction_affordance_data\RSCD_prepared\official_friction\labels.csv`
- RSCD raw images: `D:\NMI_SPWFM_datasets\friction_affordance_data\RSCD_raw\RSCD dataset-1million`
- RoadSaW: `D:\NMI_SPWFM_datasets\friction_affordance_data\RoadSaW-150_s`
- RoadSC: `D:\NMI_SPWFM_datasets\friction_affordance_data\RoadSC-balanced_to_RoadSaW12-150_l`
- LiRA-CD prepared dynamics table: `D:\NMI_SPWFM_datasets\friction_affordance_data\LiRA-CD_prepared`

Large datasets such as BDD100K, Mapillary, Ithaca365, and ACDC are not automatically downloaded because they are large and often require license/account acceptance. They are not required for the implemented overnight experiment.

## One-Command Overnight Run

Open PyCharm Terminal in this project root and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\scripts\run_topjournal_overnight.ps1
```

This executes:

- GPU verification.
- data path audit.
- smoke and full manifest builds.
- smoke train/eval.
- `ConvNeXt-Tiny + physics branch + ordinal/domain/interval` formal training.
- test evaluation.
- conformal interval calibration.
- dataset shortcut diagnostic.

Default formal config:

```text
configs\experiments\topjournal_overnight.yaml
```

Expected formal scale on the RTX 5070 Ti Laptop GPU:

- image size: `224`
- batch size: `32`
- train samples after balancing: about `51,479`
- validation samples after balancing: about `11,835`
- epochs: `12`
- steps: about `19,308` train steps plus validation

## CUDA Environment

Use this prepared PyCharm interpreter:

```text
D:\NMI_SPWFM_datasets\conda_envs\faf_gpu\python.exe
```

Verified hardware and runtime:

- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- VRAM: `12227 MiB` / `11.94 GiB`
- Compute capability: `12.0`
- Driver: `591.66`
- PyTorch: `2.11.0+cu128`
- PyTorch CUDA build: `12.8`

Validate from the project root:

```powershell
python scripts\verify_gpu.py
```

If you are worried PyCharm may use the wrong interpreter, run commands through the fixed environment wrapper:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_faf_gpu.ps1 python scripts\verify_gpu.py
```

## Quick Smoke Run

Run these from the project root in PyCharm Terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_faf_gpu.ps1 python scripts\audit_data.py --write configs\data\local_paths.yaml
.\scripts\run_faf_gpu.ps1 python scripts\build_manifests.py --config configs\data\local_paths.yaml --out-dir data\manifests --max-per-class 200
.\scripts\run_faf_gpu.ps1 python scripts\inspect_batch.py --config configs\experiments\smoke_joint.yaml --split train
.\scripts\run_faf_gpu.ps1 python scripts\train.py --config configs\experiments\smoke_joint.yaml
.\scripts\run_faf_gpu.ps1 python scripts\evaluate.py --config configs\experiments\smoke_joint.yaml --checkpoint outputs\smoke_joint\best.pt --split val
```

## Recommended Staged Runs

Do not start with the full 958k-image / 80-epoch experiment. Use this staged path on the RTX 5070 Ti Laptop GPU.

Estimate any config before training:

```powershell
.\scripts\run_faf_gpu.ps1 python scripts\estimate_runtime.py --config configs\experiments\quick_validation.yaml
.\scripts\run_faf_gpu.ps1 python scripts\estimate_runtime.py --config configs\experiments\formal_mini.yaml
.\scripts\run_faf_gpu.ps1 python scripts\estimate_runtime.py --config configs\experiments\formal_budgeted.yaml
```

Fast validation, around a few minutes on this machine:

```powershell
.\scripts\run_faf_gpu.ps1 python scripts\train.py --config configs\experiments\quick_validation.yaml
.\scripts\run_faf_gpu.ps1 python scripts\evaluate.py --config configs\experiments\quick_validation.yaml --checkpoint outputs\quick_validation\best.pt --split val
```

Balanced pilot, larger than quick validation but still limited:

```powershell
.\scripts\run_faf_gpu.ps1 python scripts\train.py --config configs\experiments\pilot_balanced.yaml
.\scripts\run_faf_gpu.ps1 python scripts\evaluate.py --config configs\experiments\pilot_balanced.yaml --checkpoint outputs\pilot_balanced\best.pt --split val
```

Budgeted formal run. This uses full manifests but caps each dataset/class group, so it is much smaller than the raw full dataset:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\make_full_manifests.ps1
.\scripts\run_faf_gpu.ps1 python scripts\inspect_batch.py --config configs\experiments\formal_mini.yaml --split train
.\scripts\run_faf_gpu.ps1 python scripts\train.py --config configs\experiments\formal_mini.yaml
.\scripts\run_faf_gpu.ps1 python scripts\evaluate.py --config configs\experiments\formal_mini.yaml --checkpoint outputs\formal_mini\best.pt --split test
```

Larger budgeted run after `formal_mini` is useful:

```powershell
.\scripts\run_faf_gpu.ps1 python scripts\inspect_batch.py --config configs\experiments\formal_budgeted.yaml --split train
.\scripts\run_faf_gpu.ps1 python scripts\train.py --config configs\experiments\formal_budgeted.yaml
.\scripts\run_faf_gpu.ps1 python scripts\evaluate.py --config configs\experiments\formal_budgeted.yaml --checkpoint outputs\formal_budgeted\best.pt --split test
.\scripts\run_faf_gpu.ps1 python scripts\dataset_id_diagnostic.py --config configs\experiments\formal_budgeted.yaml --checkpoint outputs\formal_budgeted\best.pt --max-samples 3000
```

Current recommended paper-grade run:

```powershell
.\scripts\run_topjournal_overnight.ps1
```

TensorBoard:

```powershell
.\scripts\run_faf_gpu.ps1 tensorboard --logdir outputs
```

Open `http://localhost:6006`.

## Repository Structure

```text
configs/
  data/
  experiments/
data/
  manifests/
docs/
scripts/
src/friction_affordance/
```

## Scientific Boundary

The `mu` target in this repository is a weak interval derived from visible road-state labels. It is not a direct physical friction measurement. Report it as:

> visual-evidence-conditioned friction-risk interval

not:

> measured tire-road friction coefficient

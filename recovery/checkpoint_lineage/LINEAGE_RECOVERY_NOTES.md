# Checkpoint Lineage Recovery Notes

## Current Search Boundary

The latest recovery pass only searched for:

- `screen_local_physics_field_s8k_from_best`
- known hflip parent directories

The search did not continue into the older `resume_from` ancestor recorded by `screen_local_physics_field_s8k_from_best/protocol.json`.

## `screen_local_physics_field_s8k_from_best`

Recovered local archive:

```text
E:\perception\outputs_archive\rscd_surface_classification_20260629_space_free\screen_local_physics_field_s8k_from_best
```

Copied to:

```text
recovery/checkpoint_lineage/screen_local_physics_field_s8k_from_best
```

Recovered evidence:

- test metrics: `Top-1=89.44`, `Macro-F1=89.46`, `8100` samples, `27` classes;
- run protocol with full argument values;
- test predictions CSV;
- short history file.

Missing from this folder:

- original `best.pt`;
- original `best_checkpoint.pth`;
- optimizer/scheduler state;
- raw stdout/stderr training log.

## Hflip Parent Status

Known hflip parent placeholders:

```text
screen_local_physics_hflip_consistency_w002_s8k_from_local
screen_local_physics_hflip_relation_cond_w0005_core_s8k_from_hflip
```

No checkpoint was found in the currently recovered placeholders. The directories are kept with explicit notice files so another machine can see that this is a known missing/empty recovery state, not an accidental omission.

## Derived VOR Teacher Artifact

Generated file:

```text
recovery/checkpoint_lineage/recovered_best_model_state_from_vor_teacher.pth
```

This file is not an original training checkpoint. It is a recovery artifact created from:

```text
checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt
```

It contains:

- `artifact_type`
- `is_original_training_checkpoint = False`
- provenance metadata;
- `class_to_idx`;
- source epoch and validation summary;
- `model_state_dict`;
- `model` compatibility alias.

Use:

```python
import torch
obj = torch.load(
    "recovery/checkpoint_lineage/recovered_best_model_state_from_vor_teacher.pth",
    map_location="cpu",
)
state = obj["model_state_dict"]
```

Do not cite this artifact as the original historical `best.pt`.

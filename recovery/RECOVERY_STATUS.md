# Recovery Status

Updated: 2026-07-24

## Latest User Constraint

Continue searching the old machine only for:

- `screen_local_physics_field_s8k_from_best`
- the hflip parent

Do not expand to unrelated historical experiments during this pass.

## What Was Found

### `screen_local_physics_field_s8k_from_best`

Archived folder found and copied into:

```text
recovery/checkpoint_lineage/screen_local_physics_field_s8k_from_best/
```

Recovered files:

- `evaluate_test.json`
- `evaluate_test.md`
- `history.json`
- `predictions_test.csv`
- `protocol.json`

No `best.pt`, `best_checkpoint.pth`, or model checkpoint was found in this folder.

Recorded test result:

| metric | value |
|---|---:|
| Top-1 accuracy | 89.44 |
| Mean precision | 89.54 |
| Mean recall | 89.44 |
| Macro F1 | 89.46 |
| Weighted F1 | 89.46 |
| Balanced accuracy | 89.44 |
| Samples | 8100 |
| Classes | 27 |

Protocol notes:

- backbone: `convnext_tiny`
- image size: `192`
- physics branch: enabled
- semantic physics attention branch: enabled
- local physics field branch: enabled
- local physics field scale: `0.08`
- hflip consistency weight: `0.0`
- validation/test cap: `300` samples per class
- seed: `109`

The protocol records this untraced earlier ancestor:

```text
D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification\screen_semantic_attention_line_fourier_directed_conf001_s8k_from_best\best.pt
```

This earlier ancestor was not searched in this pass because the requested search scope was limited.

### Hflip Parent

The hflip parent directory names known in this recovery tree are present as empty placeholder directories unless a notice file is added:

```text
recovery/checkpoint_lineage/screen_local_physics_hflip_consistency_w002_s8k_from_local/
recovery/checkpoint_lineage/screen_local_physics_hflip_relation_cond_w0005_core_s8k_from_hflip/
```

No checkpoint was found in these placeholder directories during this pass.

## Generated Recovery Artifact

Generated:

```text
recovery/checkpoint_lineage/recovered_best_model_state_from_vor_teacher.pth
```

This is a derived model-state artifact, not the original historical `.pt`.

Source checkpoint:

```text
checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt
```

Source SHA-256:

```text
a82e5ee233b140155365f4c28e3626be3071a416debd0f0ae5f9a90884d3c069
```

Generated artifact SHA-256:

```text
73f0c5c03d7024ac4501e66bbfa14c6133a6e8f328532e70bcb7fb9dcef3ca3c
```

Sidecar provenance:

```text
recovery/checkpoint_lineage/recovered_best_model_state_from_vor_teacher.provenance.json
```

## Source Directory Wording

The historical source directory should be described as:

```text
compatible recovered source superset
```

It should not be described as an exact `pre-20260709 snapshot`, even though the legacy folder name contains `pre_20260709_training_snapshot`.

## Still Missing

- Original checkpoint for `screen_local_physics_field_s8k_from_best`, if it ever existed.
- Non-empty hflip parent checkpoint directory, if still present elsewhere on the old machine.
- Raw full PowerShell history. Only curated relevant commands are stored.
- Verified complete ancestor chain before the recorded `screen_semantic_attention_line_fourier_directed_conf001_s8k_from_best` resume point.

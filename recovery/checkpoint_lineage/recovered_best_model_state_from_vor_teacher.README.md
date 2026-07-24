# `recovered_best_model_state_from_vor_teacher.pth`

This file is a derived recovery artifact.

It is not the original historical `best.pt`.

## Why It Exists

The preserved VOR teacher checkpoint contains a valid model state under the `model` key. For recovery work, that model state was extracted and re-saved with explicit provenance so it can be loaded and audited without confusing it with an original training run checkpoint.

## Source

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

## Loading

```python
import torch
obj = torch.load(
    "recovery/checkpoint_lineage/recovered_best_model_state_from_vor_teacher.pth",
    map_location="cpu",
)
state = obj["model_state_dict"]
```

The `model` key is also present as a compatibility alias.

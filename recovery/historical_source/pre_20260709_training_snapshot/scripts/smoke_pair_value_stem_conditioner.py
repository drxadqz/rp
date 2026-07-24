from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import (  # noqa: E402
    apply_trainable_prefixes,
    build_class_map,
    build_model,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the pair-value stem conditioner.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifests = [
        Path(cfg["data"]["train_manifest"]),
        Path(cfg["data"]["val_manifest"]),
        Path(cfg["data"]["test_manifest"]),
    ]
    class_to_idx = build_class_map(manifests)
    model = build_model(cfg, class_to_idx)
    apply_trainable_prefixes(model, cfg["train"].get("trainable_prefixes"))
    model.eval()

    if model.pair_value_stem_conditioner is None:
        raise RuntimeError("pair-value stem conditioner is not enabled")

    image_size = int(cfg["data"].get("image_size", 192))
    with torch.no_grad():
        out = model(torch.randn(2, 3, image_size, image_size), return_aux=True)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    delta = out["pair_value_stem_delta"]
    gate = out["pair_value_stem_gate"]
    spatial_gate = out["pair_value_stem_spatial_gate"]
    print(f"classes={len(class_to_idx)}")
    print(f"params_total={total_params}")
    print(f"params_trainable={trainable_params}")
    print(f"logits_shape={tuple(out['logits'].shape)}")
    print(f"gate_mean={float(gate.mean()):.8f} gate_max={float(gate.max()):.8f}")
    print(f"spatial_gate_mean={float(spatial_gate.mean()):.8f} spatial_gate_max={float(spatial_gate.max()):.8f}")
    print(f"delta_shape={tuple(delta.shape)} delta_max_abs={float(delta.abs().max()):.8f}")


if __name__ == "__main__":
    main()

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
    parser = argparse.ArgumentParser(description="Smoke-test the tri-state wet/water concrete boundary expert.")
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

    expert = model.tristate_wet_concrete_boundary_expert
    if expert is None:
        raise RuntimeError("tri-state wet-concrete boundary expert is not enabled")

    image_size = int(cfg["data"].get("image_size", 192))
    with torch.no_grad():
        out = model(torch.randn(2, 3, image_size, image_size), return_aux=True)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    residual = out["tristate_wet_concrete_boundary_delta"]
    severe_protect = out["tristate_wet_concrete_boundary_severe_protect"]
    print(f"classes={len(class_to_idx)}")
    print(f"pair_specs={len(expert.pair_specs)}")
    for spec in expert.pair_specs:
        print(
            "pair="
            f"{spec['left_name']}|{spec['right_name']} "
            f"mode={spec['mode']} severe_index={spec['severe_index']}"
        )
    print(f"params_total={total_params}")
    print(f"params_trainable={trainable_params}")
    print(f"logits_shape={tuple(out['logits'].shape)}")
    print(f"residual_shape={tuple(residual.shape)} residual_max_abs={float(residual.abs().max()):.8f}")
    print(f"severe_protect_keys={sorted(severe_protect)}")


if __name__ == "__main__":
    main()

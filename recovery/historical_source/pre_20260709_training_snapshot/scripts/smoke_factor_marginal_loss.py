from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import (  # noqa: E402
    build_class_map,
    build_model,
    factor_marginal_consistency_loss,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the factor marginal consistency loss.")
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
    model.eval()

    image_size = int(cfg["data"].get("image_size", 192))
    label_names = ["water_concrete_slight", "dry_concrete_slight"]
    labels = torch.tensor([class_to_idx[name] for name in label_names], dtype=torch.long)
    with torch.no_grad():
        out = model(torch.randn(len(label_names), 3, image_size, image_size), return_aux=True)
        loss, logs = factor_marginal_consistency_loss(out, labels, model.spec, cfg["loss"])

    print(f"logits_shape={tuple(out['logits'].shape)}")
    print(f"loss={float(loss):.8f}")
    for key in sorted(logs):
        if "factor_marginal_consistency" in key:
            print(f"{key}={logs[key]}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import argparse
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import (  # noqa: E402
    apply_trainable_prefixes,
    build_model,
    flexible_load,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/load/forward smoke test for one C3-FaRNet config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "c3_farnet" / "c3_farnet_coupled_form_expert_conditioner_micro.yaml",
    )
    args = parser.parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    labels: list[str] = []
    with open(cfg["data"]["train_manifest"], newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels.append(str(row["class_label"]))
    class_to_idx = {name: idx for idx, name in enumerate(sorted(set(labels)))}
    model = build_model(cfg, class_to_idx)
    load_info = flexible_load(model, cfg["train"].get("resume_from"))
    apply_trainable_prefixes(model, cfg["train"].get("trainable_prefixes"))

    model.eval()
    image_size = int(cfg["data"]["image_size"])
    with torch.no_grad():
        image = torch.rand(1, 3, image_size, image_size)
        output = model(image, return_aux=True)
    logits = output["logits"]
    aux = output

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"classes={len(class_to_idx)}")
    print(f"load_info={load_info}")
    print(f"params_total={total_params}")
    print(f"params_trainable={trainable_params}")
    print(f"logits_shape={tuple(logits.shape)}")
    prefixes = ("coupled_form_expert", "factor_marginal")
    for key in sorted(k for k in aux if k.startswith(prefixes)):
        value = aux[key]
        if torch.is_tensor(value):
            max_abs = float(value.detach().abs().max().cpu()) if value.numel() else 0.0
            print(f"{key}: shape={tuple(value.shape)} max_abs={max_abs:.8f}")
        else:
            print(f"{key}: {type(value).__name__}")


if __name__ == "__main__":
    main()

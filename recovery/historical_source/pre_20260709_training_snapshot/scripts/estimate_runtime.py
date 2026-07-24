from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.datasets import ManifestDataset
from friction_affordance.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seconds-per-step", type=float, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    batch_size = int(data_cfg.get("batch_size", 32))
    epochs = int(cfg.get("optim", {}).get("epochs", 1))

    train_ds = _dataset_for_split(data_cfg, "train")
    val_ds = _dataset_for_split(data_cfg, "val")

    train_samples_per_epoch = int(data_cfg.get("balanced_num_samples_per_epoch", len(train_ds))) if data_cfg.get("balanced_sampling", False) else len(train_ds)
    train_steps = math.ceil(train_samples_per_epoch / batch_size)
    val_steps = math.ceil(len(val_ds) / batch_size)
    summary: dict[str, Any] = {
        "config": str(args.config),
        "batch_size": batch_size,
        "epochs": epochs,
        "train_candidate_samples": len(train_ds),
        "train_samples_per_epoch": train_samples_per_epoch,
        "val_samples": len(val_ds),
        "train_steps_per_epoch": train_steps,
        "val_steps_per_epoch": val_steps,
        "total_train_steps": train_steps * epochs,
        "total_val_steps": val_steps * epochs,
    }
    if args.seconds_per_step:
        seconds = (train_steps + val_steps) * epochs * float(args.seconds_per_step)
        summary["rough_hours"] = seconds / 3600.0
        summary["rough_minutes"] = seconds / 60.0
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _dataset_for_split(data_cfg: dict[str, Any], split: str) -> ManifestDataset:
    return ManifestDataset(
        data_cfg[f"{split}_manifests"],
        transform=None,
        max_samples=data_cfg.get(f"max_{split}_samples"),
        max_samples_per_dataset=data_cfg.get(f"max_{split}_samples_per_dataset"),
        max_samples_per_class=data_cfg.get(f"max_{split}_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)) + (0 if split == "train" else 1),
    )


if __name__ == "__main__":
    main()

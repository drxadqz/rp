from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.engine import build_model, dataloader_worker_settings, evaluate
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    data_cfg = cfg["data"]
    manifests = data_cfg["val_manifests"]
    if args.split == "test":
        manifests = data_cfg.get("test_manifests", manifests)
    ds = ManifestDataset(
        manifests,
        transform=build_transforms(
            int(data_cfg.get("image_size", 224)),
            train=False,
            aug_cfg=data_cfg.get("augmentation"),
        ),
        max_samples=data_cfg.get(f"max_{args.split}_samples"),
        max_samples_per_dataset=data_cfg.get(f"max_{args.split}_samples_per_dataset"),
        max_samples_per_class=data_cfg.get(f"max_{args.split}_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)) + (1 if args.split == "val" else 2),
    )
    num_workers, loader_kwargs = dataloader_worker_settings(data_cfg)
    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )
    metrics = evaluate(model, loader, device, cfg.get("loss", {}))
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    out_path = args.out
    if out_path is None:
        out_dir = Path(cfg.get("output_dir", args.checkpoint.parent))
        out_path = out_dir / f"evaluate_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

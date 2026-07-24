from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()

    manifests, image_size, aug_cfg = _resolve_inputs(args)
    ds = ManifestDataset(
        manifests,
        transform=build_transforms(image_size, train=args.split == "train", aug_cfg=aug_cfg),
        max_samples=8,
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_manifest_batch)
    batch = next(iter(loader))
    print("manifests:")
    for manifest in manifests:
        print(" ", manifest)
    print("image:", tuple(batch["image"].shape))
    print("mu_interval:", batch["mu_interval"][:4])
    print("datasets:", batch["dataset"][:4])
    print("domain_idx:", batch["domain_idx"][:4].tolist())
    print("paths:")
    for path in batch["image_path"][:4]:
        print(" ", path)
    print("labels:")
    for task, labels in batch["labels"].items():
        print(" ", task, labels.tolist(), "mask", batch["masks"][task].tolist())


def _resolve_inputs(args: argparse.Namespace) -> tuple[list[Path], int, dict]:
    if args.manifest is not None:
        return [args.manifest], int(args.image_size or 160), {}

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    key = f"{args.split}_manifests"
    manifests = data_cfg.get(key)
    if not manifests:
        raise SystemExit(f"No manifests configured for split '{args.split}' in {args.config}")
    image_size = int(args.image_size or data_cfg.get("image_size", 160))
    return [Path(p) for p in manifests], image_size, dict(data_cfg.get("augmentation") or {})


if __name__ == "__main__":
    main()

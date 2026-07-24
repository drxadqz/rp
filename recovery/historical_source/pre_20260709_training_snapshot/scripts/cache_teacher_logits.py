from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import (  # noqa: E402
    RSCDSurfaceDataset,
    build_anchor_teacher,
    build_class_map,
    build_specialist_teacher,
    collate,
    load_config,
)
from friction_affordance.transforms import build_transforms  # noqa: E402
from friction_affordance.utils import resolve_device, set_seed  # noqa: E402


def _split_dataset_cfg(cfg: dict[str, Any], split: str) -> tuple[Path, int | None, int | None, int]:
    data = cfg["data"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    seed = int(cfg.get("seed", 79))
    if split == "train":
        return (
            Path(data["train_manifest"]),
            train_cfg.get("max_train_samples"),
            train_cfg.get("max_train_samples_per_class"),
            seed,
        )
    if split == "val":
        return (
            Path(data["val_manifest"]),
            eval_cfg.get("max_val_samples"),
            eval_cfg.get("max_val_samples_per_class"),
            seed + 1,
        )
    if split == "test":
        return (
            Path(data["test_manifest"]),
            eval_cfg.get("max_test_samples"),
            eval_cfg.get("max_test_samples_per_class"),
            seed + 2,
        )
    raise ValueError(f"unsupported split: {split}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen teacher logits by RSCD image path.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", choices=["anchor", "expert"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"cache already exists, pass --overwrite to replace it: {args.output}")

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 79)))
    manifests = [Path(cfg["data"]["train_manifest"]), Path(cfg["data"]["val_manifest"]), Path(cfg["data"]["test_manifest"])]
    class_to_idx = build_class_map(manifests)

    image_size = int(cfg["data"].get("image_size", 192))
    transform = build_transforms(
        image_size,
        train=False,
        aug_cfg={"resize_mode": str(cfg["data"].get("eval_resize_mode", "letterbox"))},
    )
    manifest, max_samples, max_samples_per_class, seed = _split_dataset_cfg(cfg, args.split)
    dataset = RSCDSurfaceDataset(
        manifest,
        class_to_idx=class_to_idx,
        transform=transform,
        max_samples=max_samples,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    device = resolve_device(args.device)
    if args.role == "anchor":
        teacher = build_anchor_teacher(cfg, class_to_idx, device)
    else:
        teacher = build_specialist_teacher(cfg, class_to_idx, device)
    if teacher is None:
        raise ValueError(f"{args.role} teacher is not configured in {args.config}")

    logits_parts: list[torch.Tensor] = []
    labels: list[int] = []
    paths: list[str] = []
    use_amp = device.type == "cuda"
    teacher.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"cache-{args.role}-{args.split}", ascii=True):
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = teacher(images, return_aux=False)
            logits_parts.append(logits.detach().float().cpu().to(dtype=torch.float16))
            labels.extend(int(v) for v in batch["label"].cpu().tolist())
            paths.extend(str(v) for v in batch["image_path"])

    payload = {
        "role": args.role,
        "split": args.split,
        "config": str(args.config),
        "image_size": image_size,
        "image_paths": paths,
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "logits": torch.cat(logits_parts, dim=0) if logits_parts else torch.empty((0, len(class_to_idx)), dtype=torch.float16),
        "class_to_idx": class_to_idx,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved {args.role} {args.split} teacher logits: {args.output} ({len(paths)} images)")


if __name__ == "__main__":
    main()

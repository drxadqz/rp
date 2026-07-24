from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed
from run_rscd_surface_classification import RSCDSurfaceDataset, build_class_map, collate
from run_rscd_topology_logit_calibration import (
    DEFAULT_TRAIN,
    DEFAULT_VAL,
    DEFAULT_TEST,
    build_model_from_protocol,
    load_protocol,
)


DEFAULT_SOURCE = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\screen_physics_texture_hardboost025_lr1e5_s36k_e1_seed101_from_best"
)
DEFAULT_OUT = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\tta_teacher_probs_current_best"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export RSCD TTA teacher probabilities from a fixed checkpoint. "
            "The teacher averages original and horizontal-flip logits; use it "
            "to distill test-time view robustness into a strict single model."
        )
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-train-samples-per-class", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(args.source_run / "protocol.json")
    train_args = protocol["args"]
    class_to_idx = {str(k): int(v) for k, v in protocol.get("class_to_idx", {}).items()}
    if not class_to_idx:
        class_to_idx = build_class_map([args.train_manifest, args.val_manifest, args.test_manifest])
    image_size = int(train_args.get("image_size", 192))

    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = build_model_from_protocol(train_args, class_to_idx).to(device)
    checkpoint = args.checkpoint or (args.source_run / "best.pt")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    transform = build_transforms(image_size, train=False, aug_cfg={"resize_mode": "letterbox"})
    max_per_class = int(args.max_train_samples_per_class)
    dataset = RSCDSurfaceDataset(
        args.train_manifest,
        class_to_idx=class_to_idx,
        transform=transform,
        max_samples_per_class=max_per_class if max_per_class > 0 else None,
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    temperature = max(float(args.temperature), 1e-3)
    probs_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="export-tta-teacher", leave=False, ascii=True):
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(image)
                flipped_logits = model(torch.flip(image, dims=[3]))
                mean_logits = 0.5 * (logits + flipped_logits)
                probs = torch.softmax(mean_logits / temperature, dim=1)
            probs_rows.append(probs.detach().float().cpu().numpy())
            labels.append(batch["label"].detach().cpu().numpy().astype(np.int64))
            paths.extend([str(x) for x in batch["image_path"]])

    subset_tag = "all" if max_per_class <= 0 else f"mpc{max_per_class}"
    temp_tag = f"t{temperature:g}".replace(".", "p")
    out_npz = args.output_dir / f"teacher_probs_hflip_{subset_tag}_{temp_tag}_seed{int(args.seed)}.npz"
    np.savez_compressed(
        out_npz,
        image_path=np.asarray(paths, dtype=object),
        label=np.concatenate(labels, axis=0),
        probs=np.concatenate(probs_rows, axis=0).astype(np.float32),
    )
    meta: dict[str, Any] = {
        "claim_boundary": (
            "TTA teacher probabilities are generated from the fixed current-best "
            "checkpoint using original+hflip logits. They are training-only soft "
            "targets and do not change the strict single-model test protocol."
        ),
        "source_run": str(args.source_run),
        "checkpoint": str(checkpoint),
        "train_manifest": str(args.train_manifest),
        "rows": int(len(paths)),
        "max_train_samples_per_class": max_per_class if max_per_class > 0 else None,
        "seed": int(args.seed),
        "temperature": float(temperature),
        "output": str(out_npz),
    }
    out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_npz)


if __name__ == "__main__":
    main()

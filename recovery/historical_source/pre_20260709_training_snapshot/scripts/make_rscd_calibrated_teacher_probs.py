from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from friction_affordance.utils import resolve_device, set_seed
from run_rscd_surface_classification import build_class_map
from run_rscd_topology_logit_calibration import (
    DEFAULT_RUN,
    DEFAULT_TRAIN,
    DEFAULT_VAL,
    build_model_from_protocol,
    collect_or_load,
    load_protocol,
    softmax_np,
)


DEFAULT_OUT = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\calibrated_teacher_probs"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create calibrated teacher probabilities for RSCD distillation. "
            "The calibrator is fitted on validation logits and then applied to "
            "a deterministic training subset."
        )
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument(
        "--max-train-samples-per-class",
        type=int,
        default=600,
        help="Per-class train subset for teacher export. Use 0 or a negative value for full train coverage.",
    )
    parser.add_argument("--calibrator-c", type=float, default=1.0)
    parser.add_argument("--blend-alpha", type=float, default=0.7)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(args.source_run / "protocol.json")
    train_args = protocol["args"]
    class_to_idx = {str(k): int(v) for k, v in protocol.get("class_to_idx", {}).items()}
    if not class_to_idx:
        class_to_idx = build_class_map([args.train_manifest, args.val_manifest])
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

    val_cache = args.output_dir / "val_logits_topology.npz"
    max_per_class = int(args.max_train_samples_per_class)
    train_subset_tag = "all" if max_per_class <= 0 else f"mpc{max_per_class}"
    train_cache = args.output_dir / f"train_logits_topology_{train_subset_tag}.npz"
    val = collect_or_load(
        split="val",
        cache_path=val_cache,
        manifest=args.val_manifest,
        class_to_idx=class_to_idx,
        image_size=image_size,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        force_cache=bool(args.force_cache),
    )
    train = collect_or_load(
        split="train",
        cache_path=train_cache,
        manifest=args.train_manifest,
        class_to_idx=class_to_idx,
        image_size=image_size,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        force_cache=bool(args.force_cache),
        max_samples_per_class=max_per_class if max_per_class > 0 else None,
        seed=int(args.seed),
    )

    calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(args.calibrator_c), max_iter=900, solver="lbfgs"),
    )
    calibrator.fit(val["logits"].astype(np.float32), val["label"].astype(np.int64))
    base_probs = softmax_np(train["logits"].astype(np.float32))
    calibrated_probs = calibrator.predict_proba(train["logits"].astype(np.float32))
    probs = (1.0 - float(args.blend_alpha)) * base_probs + float(args.blend_alpha) * calibrated_probs
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)

    out_npz = args.output_dir / f"teacher_probs_{train_subset_tag}_c{args.calibrator_c:g}_a{args.blend_alpha:g}.npz"
    np.savez_compressed(
        out_npz,
        image_path=train["image_path"],
        label=train["label"].astype(np.int64),
        probs=probs.astype(np.float32),
    )
    meta = {
        "claim_boundary": (
            "Teacher probabilities are generated from a validation-fitted logit calibrator. "
            "Use them for exploratory distillation; report separately from plain supervised training."
        ),
        "source_run": str(args.source_run),
        "checkpoint": str(checkpoint),
        "val_rows": int(len(val["label"])),
        "train_rows": int(len(train["label"])),
        "max_train_samples_per_class": max_per_class if max_per_class > 0 else None,
        "train_subset_tag": train_subset_tag,
        "calibrator_c": float(args.calibrator_c),
        "blend_alpha": float(args.blend_alpha),
        "output": str(out_npz),
    }
    out_json = out_npz.with_suffix(".json")
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_npz)


if __name__ == "__main__":
    main()

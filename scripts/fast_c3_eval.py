from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from friction_affordance.c3_experiment import (
    RSCDSurfaceDataset,
    apply_pareto_safe_logit_patch,
    build_class_map,
    build_model,
    collate,
    load_pareto_safe_logit_patch_rules,
    load_config,
)
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast C3-FaRNet classification-only evaluation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--hardpair-margin-scale", default=None, type=float)
    parser.add_argument("--hardpair-error-gate-floor", default=None, type=float)
    parser.add_argument("--hardpair-correction-scale", default=None, type=float)
    parser.add_argument("--hardpair-physics-gate-floor", default=None, type=float)
    parser.add_argument("--hardpair-physics-gate-power", default=None, type=float)
    parser.add_argument("--hardpair-sample-protect-threshold", default=None, type=float)
    parser.add_argument("--hardpair-sample-protect-temperature", default=None, type=float)
    parser.add_argument("--hardpair-sample-protect-classes", default=None, type=str)
    parser.add_argument("--source-router-scale", default=None, type=float)
    parser.add_argument("--source-router-base-strength", default=None, type=float)
    parser.add_argument("--source-router-physics-gate-floor", default=None, type=float)
    parser.add_argument("--source-router-gate-temperature", default=None, type=float)
    parser.add_argument(
        "--hardpair-pair-scale",
        default=None,
        action="append",
        help="Pair-local residual scale override, e.g. dry_concrete_severe|dry_concrete_slight=0.5",
    )
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--max-samples-per-class", default=None, type=int)
    parser.add_argument(
        "--logit-patch-rules",
        default=None,
        type=Path,
        help="Optional validation-accepted no-harm logit patch rules JSON.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for faster screening inference.")
    parser.add_argument("--skip-predictions", action="store_true", help="Do not write per-image predictions CSV.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.hardpair_margin_scale is not None:
        cfg.setdefault("model", {})["hardpair_margin_scale"] = float(args.hardpair_margin_scale)
    if args.hardpair_error_gate_floor is not None:
        cfg.setdefault("model", {})["hardpair_error_gate_floor"] = float(args.hardpair_error_gate_floor)
    if args.hardpair_correction_scale is not None:
        cfg.setdefault("model", {})["hardpair_correction_scale"] = float(args.hardpair_correction_scale)
    if args.hardpair_physics_gate_floor is not None:
        cfg.setdefault("model", {})["hardpair_physics_gate_floor"] = float(args.hardpair_physics_gate_floor)
    if args.hardpair_physics_gate_power is not None:
        cfg.setdefault("model", {})["hardpair_physics_gate_power"] = float(args.hardpair_physics_gate_power)
    if args.hardpair_sample_protect_threshold is not None:
        cfg.setdefault("model", {})["hardpair_sample_protect_threshold"] = float(
            args.hardpair_sample_protect_threshold
        )
    if args.hardpair_sample_protect_temperature is not None:
        cfg.setdefault("model", {})["hardpair_sample_protect_temperature"] = float(
            args.hardpair_sample_protect_temperature
        )
    if args.hardpair_sample_protect_classes is not None:
        cfg.setdefault("model", {})["hardpair_sample_protect_classes"] = [
            item.strip()
            for item in str(args.hardpair_sample_protect_classes).split(",")
            if item.strip()
        ]
    if args.source_router_scale is not None:
        cfg.setdefault("model", {})["source_reliable_boundary_scale"] = float(args.source_router_scale)
    if args.source_router_base_strength is not None:
        cfg.setdefault("model", {})["source_reliable_boundary_base_strength"] = float(
            args.source_router_base_strength
        )
    if args.source_router_physics_gate_floor is not None:
        cfg.setdefault("model", {})["source_reliable_boundary_physics_gate_floor"] = float(
            args.source_router_physics_gate_floor
        )
    if args.source_router_gate_temperature is not None:
        cfg.setdefault("model", {})["source_reliable_boundary_gate_temperature"] = float(
            args.source_router_gate_temperature
        )
    if args.hardpair_pair_scale:
        scales = dict(cfg.setdefault("model", {}).get("hardpair_pair_scales", {}) or {})
        for item in args.hardpair_pair_scale:
            if "=" not in item:
                raise ValueError(f"--hardpair-pair-scale must use pair=scale format: {item}")
            pair_name, scale_text = item.split("=", 1)
            scales[pair_name.strip()] = float(scale_text)
        cfg.setdefault("model", {})["hardpair_pair_scales"] = scales
    set_seed(int(cfg.get("seed", 79)))
    data = cfg["data"]
    manifests = [Path(data["train_manifest"]), Path(data["val_manifest"]), Path(data["test_manifest"])]
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    image_size = int(data.get("image_size", 192))
    eval_tf = build_transforms(
        image_size,
        train=False,
        aug_cfg={"resize_mode": str(data.get("eval_resize_mode", "letterbox"))},
    )
    split_manifest = Path(data["test_manifest"] if args.split == "test" else data["val_manifest"])
    max_samples = cfg["eval"].get(f"max_{args.split}_samples")
    max_samples_per_class = cfg["eval"].get(f"max_{args.split}_samples_per_class")
    if args.max_samples is not None:
        max_samples = int(args.max_samples)
    if args.max_samples_per_class is not None:
        max_samples_per_class = int(args.max_samples_per_class)
    ds = RSCDSurfaceDataset(
        split_manifest,
        class_to_idx=class_to_idx,
        transform=eval_tf,
        max_samples=max_samples,
        max_samples_per_class=max_samples_per_class,
        seed=int(cfg.get("seed", 79)) + (2 if args.split == "test" else 1),
    )
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kwargs)

    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model(cfg, class_to_idx).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_state = dict(state["model"])
    model_state = model.state_dict()
    skipped = []
    for key, value in list(checkpoint_state.items()):
        if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            checkpoint_state.pop(key)
    missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
    if missing or unexpected:
        print(f"Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if skipped:
        print(f"Skipped shape-mismatched checkpoint entries: {', '.join(skipped)}", flush=True)
    model.eval()
    use_amp = bool(args.amp) and device.type == "cuda"
    logit_patch_rules = load_pareto_safe_logit_patch_rules(args.logit_patch_rules)
    logit_patch_hits: dict[str, int] = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config_resolved.yaml").write_text(
        json.dumps({"config": str(args.config), "checkpoint": str(args.checkpoint)}, indent=2),
        encoding="utf-8",
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict[str, object]] = []
    losses: list[float] = []
    start = time.time()
    print(f"Dataset size={len(ds)} batches={len(loader)} device={device} batch={args.batch_size}", flush=True)

    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image, return_aux=False)
            if logit_patch_rules:
                logits, patch_info = apply_pareto_safe_logit_patch(logits, logit_patch_rules, idx_to_class)
                for key, value in patch_info.get("rule_hits", {}).items():
                    logit_patch_hits[key] = logit_patch_hits.get(key, 0) + int(value)
            loss = F.cross_entropy(logits, label)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            true_cpu = label.detach().cpu().numpy().astype(int).tolist()
            pred_cpu = pred.detach().cpu().numpy().astype(int).tolist()
            conf_cpu = conf.detach().cpu().tolist()
            y_true.extend(true_cpu)
            y_pred.extend(pred_cpu)
            losses.append(float(loss.detach().cpu()) * int(label.numel()))
            if not args.skip_predictions:
                for path, true_idx, pred_idx, confidence in zip(
                    batch["image_path"],
                    true_cpu,
                    pred_cpu,
                    conf_cpu,
                    strict=True,
                ):
                    rows.append(
                        {
                            "image_path": str(path),
                            "true_label": idx_to_class[int(true_idx)],
                            "pred_label": idx_to_class[int(pred_idx)],
                            "confidence": float(confidence),
                        }
                    )
            if step % 200 == 0 or step == len(loader):
                print(
                    f"batch {step}/{len(loader)} elapsed_s={time.time() - start:.1f}",
                    flush=True,
                )

    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[i] for i in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "loss": float(sum(losses) / max(len(y_true), 1)),
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "num_samples": int(len(y_true)),
        "num_classes": int(len(labels)),
        "num_errors": int(sum(1 for true_idx, pred_idx in zip(y_true, y_pred, strict=True) if true_idx != pred_idx)),
        "pareto_safe_logit_patch_enabled": bool(logit_patch_rules),
        "pareto_safe_logit_patch_hits": int(sum(logit_patch_hits.values())),
    }
    for key, value in sorted(logit_patch_hits.items()):
        summary[f"pareto_safe_logit_patch_hits/{key}"] = int(value)
    metrics = {"summary": summary, "classification_report": report}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.skip_predictions:
        pd.DataFrame(rows).to_csv(args.output_dir / f"predictions_{args.split}.csv", index=False, encoding="utf-8")
    per_rows = [{"class": name, **report[name]} for name in target_names]
    pd.DataFrame(per_rows).to_csv(args.output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        args.output_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

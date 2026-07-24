from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import (  # noqa: E402
    RSCDSurfaceDataset,
    build_class_map,
    build_model,
    collate,
    load_config,
)
from friction_affordance.rscd_factors import build_rscd_factor_spec  # noqa: E402
from friction_affordance.transforms import build_transforms  # noqa: E402
from friction_affordance.utils import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a validation-only RSCD factor tensor bias on frozen logits. "
            "The bias decomposes each class offset into friction, material, roughness, "
            "pair-coupling, and optional triple-coupling terms."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-samples-per-class", default=None, type=int)
    parser.add_argument("--steps", default=900, type=int)
    parser.add_argument("--lr", default=0.08, type=float)
    parser.add_argument("--l2", default=0.03, type=float)
    parser.add_argument("--no-triple", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def load_model_logits(
    *,
    cfg: dict[str, Any],
    checkpoint: Path,
    split: str,
    class_to_idx: dict[str, int],
    batch_size: int,
    num_workers: int,
    max_samples_per_class: int | None,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    data = cfg["data"]
    image_size = int(data.get("image_size", 192))
    eval_tf = build_transforms(
        image_size,
        train=False,
        aug_cfg={"resize_mode": str(data.get("eval_resize_mode", "letterbox"))},
    )
    manifest = Path(data["val_manifest"] if split == "val" else data["test_manifest"])
    ds = RSCDSurfaceDataset(
        manifest,
        class_to_idx=class_to_idx,
        transform=eval_tf,
        max_samples_per_class=max_samples_per_class,
        seed=int(cfg.get("seed", 79)) + (1 if split == "val" else 2),
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kwargs)

    device = resolve_device()
    model = build_model(cfg, class_to_idx).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_state = dict(state["model"])
    model_state = model.state_dict()
    for key, value in list(checkpoint_state.items()):
        if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
            checkpoint_state.pop(key)
    missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
    if missing or unexpected:
        print(f"{split}: loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    use_amp = bool(amp) and device.type == "cuda"
    logits_chunks: list[torch.Tensor] = []
    labels: list[int] = []
    paths: list[str] = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image, return_aux=False)
            logits_chunks.append(logits.detach().cpu().float())
            labels.extend(int(x) for x in batch["label"].tolist())
            paths.extend(str(x) for x in batch["image_path"])
            if step % 200 == 0 or step == len(loader):
                print(f"{split}: batch {step}/{len(loader)}", flush=True)
    return torch.cat(logits_chunks, dim=0), torch.tensor(labels, dtype=torch.long), paths


def build_design(class_to_idx: dict[str, int], include_triple: bool) -> tuple[torch.Tensor, list[str]]:
    spec = build_rscd_factor_spec(class_to_idx)
    factors = spec.class_to_factor.clone()
    dims = {
        "friction": int(factors[:, 0].max().item()) + 1,
        "material": int(factors[:, 1].max().item()) + 1,
        "roughness": int(factors[:, 2].max().item()) + 1,
    }
    names: list[str] = []
    cols: list[torch.Tensor] = []
    f = factors[:, 0]
    m = factors[:, 1]
    r = factors[:, 2]

    def add_one_hot(prefix: str, idx: torch.Tensor, size: int) -> None:
        oh = F.one_hot(idx.clamp_min(0), num_classes=size).float()
        for i in range(size):
            names.append(f"{prefix}:{i}")
            cols.append(oh[:, i])

    add_one_hot("F", f, dims["friction"])
    add_one_hot("M", m, dims["material"])
    add_one_hot("R", r, dims["roughness"])
    fm = f * dims["material"] + m
    fr = f * dims["roughness"] + r
    mr = m * dims["roughness"] + r
    add_one_hot("FM", fm, dims["friction"] * dims["material"])
    add_one_hot("FR", fr, dims["friction"] * dims["roughness"])
    add_one_hot("MR", mr, dims["material"] * dims["roughness"])
    if include_triple:
        fmr = (f * dims["material"] + m) * dims["roughness"] + r
        add_one_hot("FMR", fmr, dims["friction"] * dims["material"] * dims["roughness"])
    design = torch.stack(cols, dim=1)
    # Center columns so the calibration learns relative boundaries rather than
    # a meaningless global logit shift.
    design = design - design.mean(dim=0, keepdim=True)
    return design, names


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, idx_to_class: dict[int, str]) -> dict[str, Any]:
    pred = logits.argmax(dim=1).cpu().numpy()
    true = labels.cpu().numpy()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(
        true,
        pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "top1": float(accuracy_score(true, pred)),
        "mean_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "num_samples": int(labels.numel()),
        "num_errors": int((pred != true).sum()),
        "classification_report": report,
    }


def top1_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def is_special_surface(name: str) -> bool:
    return any(token in name for token in ("gravel", "mud", "snow", "ice"))


def is_weak_boundary_family(name: str) -> bool:
    return any(
        token in name
        for token in (
            "water_concrete_slight",
            "water_concrete_severe",
            "wet_concrete_slight",
            "wet_concrete_severe",
            "dry_concrete_slight",
            "dry_concrete_severe",
            "water_asphalt_slight",
            "water_asphalt_severe",
            "wet_asphalt_severe",
        )
    )


def factor_distance_table(class_to_idx: dict[str, int]) -> torch.Tensor:
    spec = build_rscd_factor_spec(class_to_idx)
    factors = spec.class_to_factor.clone()
    n = factors.shape[0]
    dist = torch.zeros((n, n), dtype=torch.long)
    for i in range(n):
        for j in range(n):
            valid = (factors[i] >= 0) & (factors[j] >= 0)
            dist[i, j] = int((factors[i, valid] != factors[j, valid]).sum().item())
    return dist


def policy_mask(
    *,
    base_pred: torch.Tensor,
    cal_pred: torch.Tensor,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    policy: str,
) -> torch.Tensor:
    changed = base_pred.ne(cal_pred)
    if policy == "margin_only":
        return changed
    dist = factor_distance_table(class_to_idx)
    single_factor = dist[base_pred, cal_pred].le(1)
    if policy == "single_factor":
        return changed & single_factor

    base_names = [idx_to_class[int(i)] for i in base_pred.tolist()]
    cal_names = [idx_to_class[int(i)] for i in cal_pred.tolist()]
    special = torch.tensor(
        [is_special_surface(a) or is_special_surface(b) for a, b in zip(base_names, cal_names, strict=True)],
        dtype=torch.bool,
    )
    weak_family = torch.tensor(
        [is_weak_boundary_family(a) or is_weak_boundary_family(b) for a, b in zip(base_names, cal_names, strict=True)],
        dtype=torch.bool,
    )
    if policy == "single_factor_protect_special":
        return changed & single_factor & ~special
    if policy == "weak_single_factor_protect_special":
        return changed & single_factor & weak_family & ~special
    if policy == "weak_boundary_any_factor_protect_special":
        return changed & weak_family & ~special
    raise ValueError(f"Unknown gate policy: {policy}")


def apply_gated_calibration(
    *,
    base_logits: torch.Tensor,
    cal_logits: torch.Tensor,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    policy: str,
    threshold: float,
) -> tuple[torch.Tensor, int]:
    base_pred = base_logits.argmax(dim=1)
    cal_pred = cal_logits.argmax(dim=1)
    margin = top1_margin(base_logits)
    allowed = margin.le(float(threshold)) & policy_mask(
        base_pred=base_pred,
        cal_pred=cal_pred,
        idx_to_class=idx_to_class,
        class_to_idx=class_to_idx,
        policy=policy,
    )
    out = base_logits.clone()
    out[allowed] = cal_logits[allowed]
    return out, int(allowed.sum().item())


def select_gated_policy(
    *,
    val_logits: torch.Tensor,
    val_cal: torch.Tensor,
    val_labels: torch.Tensor,
    test_logits: torch.Tensor,
    test_cal: torch.Tensor,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
) -> dict[str, Any]:
    policies = [
        "margin_only",
        "single_factor",
        "single_factor_protect_special",
        "weak_single_factor_protect_special",
        "weak_boundary_any_factor_protect_special",
    ]
    changed = val_logits.argmax(dim=1).ne(val_cal.argmax(dim=1))
    margins = top1_margin(val_logits)
    if bool(changed.any()):
        qs = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.95])
        thresholds = torch.quantile(margins[changed], qs).tolist()
        thresholds.append(float(margins[changed].max().item()))
    else:
        thresholds = [0.0]
    thresholds = sorted({round(float(x), 6) for x in thresholds})

    baseline = metrics_from_logits(val_logits, val_labels, idx_to_class)
    best: dict[str, Any] = {
        "policy": "baseline",
        "threshold": None,
        "val_gate_count": 0,
        "val_metrics": {k: v for k, v in baseline.items() if k != "classification_report"},
        "score": (float(baseline["macro_f1"]), float(baseline["top1"])),
    }
    tried: list[dict[str, Any]] = []
    for policy in policies:
        for threshold in thresholds:
            gated, gate_count = apply_gated_calibration(
                base_logits=val_logits,
                cal_logits=val_cal,
                idx_to_class=idx_to_class,
                class_to_idx=class_to_idx,
                policy=policy,
                threshold=threshold,
            )
            metrics = metrics_from_logits(gated, val_labels, idx_to_class)
            compact = {k: v for k, v in metrics.items() if k != "classification_report"}
            score = (float(metrics["macro_f1"]), float(metrics["top1"]))
            tried.append(
                {
                    "policy": policy,
                    "threshold": threshold,
                    "val_gate_count": gate_count,
                    "val_metrics": compact,
                }
            )
            if score > best["score"]:
                best = {
                    "policy": policy,
                    "threshold": threshold,
                    "val_gate_count": gate_count,
                    "val_metrics": compact,
                    "score": score,
                }

    if best["policy"] == "baseline":
        test_gated = test_logits
        test_gate_count = 0
    else:
        test_gated, test_gate_count = apply_gated_calibration(
            base_logits=test_logits,
            cal_logits=test_cal,
            idx_to_class=idx_to_class,
            class_to_idx=class_to_idx,
            policy=str(best["policy"]),
            threshold=float(best["threshold"]),
        )
    return {
        "best": best,
        "tried": tried,
        "test_logits": test_gated,
        "test_gate_count": test_gate_count,
    }


def write_predictions(
    path: Path,
    logits: torch.Tensor,
    labels: torch.Tensor,
    image_paths: list[str],
    idx_to_class: dict[int, str],
) -> None:
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "true_label", "pred_label", "confidence"])
        writer.writeheader()
        for image_path, y, p, c in zip(image_paths, labels.tolist(), pred.tolist(), conf.tolist(), strict=True):
            writer.writerow(
                {
                    "image_path": image_path,
                    "true_label": idx_to_class[int(y)],
                    "pred_label": idx_to_class[int(p)],
                    "confidence": f"{float(c):.8f}",
                }
            )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 79)))
    data = cfg["data"]
    manifests = [Path(data["train_manifest"]), Path(data["val_manifest"]), Path(data["test_manifest"])]
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_logits, val_labels, val_paths = load_model_logits(
        cfg=cfg,
        checkpoint=args.checkpoint,
        split="val",
        class_to_idx=class_to_idx,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_samples_per_class=args.max_samples_per_class,
        amp=bool(args.amp),
    )
    test_logits, test_labels, test_paths = load_model_logits(
        cfg=cfg,
        checkpoint=args.checkpoint,
        split="test",
        class_to_idx=class_to_idx,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_samples_per_class=args.max_samples_per_class,
        amp=bool(args.amp),
    )
    design, names = build_design(class_to_idx, include_triple=not bool(args.no_triple))
    weights = torch.zeros(design.shape[1], requires_grad=True)
    opt = torch.optim.Adam([weights], lr=float(args.lr))
    baseline_val = metrics_from_logits(val_logits, val_labels, idx_to_class)
    baseline_test = metrics_from_logits(test_logits, test_labels, idx_to_class)
    for step in range(1, int(args.steps) + 1):
        opt.zero_grad(set_to_none=True)
        bias = design @ weights
        bias = bias - bias.mean()
        loss = F.cross_entropy(val_logits + bias.view(1, -1), val_labels)
        loss = loss + float(args.l2) * weights.square().mean()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == int(args.steps):
            with torch.no_grad():
                val_top1 = (val_logits + (design @ weights).view(1, -1)).argmax(dim=1).eq(val_labels).float().mean()
            print(
                f"calib step {step}/{args.steps} "
                f"loss={float(loss.detach()):.6f} val_top1={float(val_top1.detach()):.6f}",
                flush=True,
            )

    with torch.no_grad():
        bias = design @ weights
        bias = bias - bias.mean()
        val_cal = val_logits + bias.view(1, -1)
        test_cal = test_logits + bias.view(1, -1)
    calibrated_val = metrics_from_logits(val_cal, val_labels, idx_to_class)
    calibrated_test = metrics_from_logits(test_cal, test_labels, idx_to_class)
    gated_selection = select_gated_policy(
        val_logits=val_logits,
        val_cal=val_cal,
        val_labels=val_labels,
        test_logits=test_logits,
        test_cal=test_cal,
        idx_to_class=idx_to_class,
        class_to_idx=class_to_idx,
    )
    selected = gated_selection["best"]
    if selected["policy"] == "baseline":
        val_gated = val_logits
        val_gate_count = 0
    else:
        val_gated, val_gate_count = apply_gated_calibration(
            base_logits=val_logits,
            cal_logits=val_cal,
            idx_to_class=idx_to_class,
            class_to_idx=class_to_idx,
            policy=str(selected["policy"]),
            threshold=float(selected["threshold"]),
        )
    gated_val = metrics_from_logits(val_gated, val_labels, idx_to_class)
    gated_test = metrics_from_logits(gated_selection["test_logits"], test_labels, idx_to_class)
    result = {
        "protocol": {
            "method": "validation-only RSCD factor tensor logit calibration",
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "max_samples_per_class": args.max_samples_per_class,
            "include_triple": not bool(args.no_triple),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "claim_boundary": (
                "Diagnostic candidate. If it improves held-out test, implement the same "
                "factor tensor bias as a trainable calibration layer rather than as a post-hoc shortcut."
            ),
        },
        "baseline_val": {k: v for k, v in baseline_val.items() if k != "classification_report"},
        "calibrated_val": {k: v for k, v in calibrated_val.items() if k != "classification_report"},
        "baseline_test": {k: v for k, v in baseline_test.items() if k != "classification_report"},
        "calibrated_test": {k: v for k, v in calibrated_test.items() if k != "classification_report"},
        "gated_val": {k: v for k, v in gated_val.items() if k != "classification_report"},
        "gated_test": {k: v for k, v in gated_test.items() if k != "classification_report"},
        "gated_selection": {
            "policy": selected["policy"],
            "threshold": selected["threshold"],
            "val_gate_count": int(val_gate_count),
            "test_gate_count": int(gated_selection["test_gate_count"]),
            "selection_metric": "validation macro_f1, then top1",
            "claim_boundary": (
                "Validation chooses whether factor calibration is allowed only on low-margin "
                "RSCD factor-boundary samples; test labels are used only for final reporting."
            ),
        },
        "gated_selection_grid": gated_selection["tried"],
        "class_bias": {idx_to_class[i]: float(bias[i]) for i in range(len(idx_to_class))},
        "parameter_names": names,
    }
    (args.output_dir / "factor_tensor_bias_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": result,
                "calibrated_test_report": calibrated_test["classification_report"],
                "gated_test_report": gated_test["classification_report"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_predictions(args.output_dir / "predictions_test_baseline.csv", test_logits, test_labels, test_paths, idx_to_class)
    write_predictions(args.output_dir / "predictions_test_calibrated.csv", test_cal, test_labels, test_paths, idx_to_class)
    write_predictions(
        args.output_dir / "predictions_test_gated.csv",
        gated_selection["test_logits"],
        test_labels,
        test_paths,
        idx_to_class,
    )
    write_predictions(args.output_dir / "predictions_val_baseline.csv", val_logits, val_labels, val_paths, idx_to_class)
    write_predictions(args.output_dir / "predictions_val_calibrated.csv", val_cal, val_labels, val_paths, idx_to_class)
    write_predictions(args.output_dir / "predictions_val_gated.csv", val_gated, val_labels, val_paths, idx_to_class)
    print(
        json.dumps(
            {
                "baseline_val": result["baseline_val"],
                "calibrated_val": result["calibrated_val"],
                "gated_val": result["gated_val"],
                "baseline_test": result["baseline_test"],
                "calibrated_test": result["calibrated_test"],
                "gated_test": result["gated_test"],
                "gated_selection": result["gated_selection"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

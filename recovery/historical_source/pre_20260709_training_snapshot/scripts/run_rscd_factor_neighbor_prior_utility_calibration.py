from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_rscd_tta_ensemble import _load_model, _predict_logits
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed
from run_rscd_surface_classification import RSCDSurfaceDataset, collate, confusion_rows


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_OUT = DEFAULT_ROOT / "posthoc_factor_neighbor_prior_utility_current_tta"
DEFAULT_RUN_DIRS = [
    DEFAULT_ROOT / "screen_local_physics_hflip_consistency_w002_s8k_from_local",
    DEFAULT_ROOT / "screen_local_physics_hflip_relation_cond_w0005_core_s8k_from_hflip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-selected RSCD factor-neighbor prior utility calibration. "
            "This is a diagnostic/post-hoc candidate: the neural ensemble is frozen, "
            "and a tiny natural-prior bias is applied only inside uncertain RSCD "
            "factor-neighbor candidate sets."
        )
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--val-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_val.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    models: list[nn.Module] = []
    reference_class_to_idx: dict[str, int] | None = None
    image_size: int | None = None
    protocols: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        model, class_to_idx, model_args = _load_model(run_dir, device)
        if reference_class_to_idx is None:
            reference_class_to_idx = class_to_idx
            image_size = int(model_args.get("image_size", 192))
        elif class_to_idx != reference_class_to_idx:
            raise ValueError(f"class map mismatch in {run_dir}")
        elif int(model_args.get("image_size", image_size)) != int(image_size):
            raise ValueError(f"image-size mismatch in {run_dir}")
        models.append(model)
        protocols.append({"run_dir": str(run_dir), "args": model_args})
    if reference_class_to_idx is None or image_size is None:
        raise ValueError("at least one run directory is required")

    idx_to_class = {idx: name for name, idx in reference_class_to_idx.items()}
    val_cache = args.output_dir / f"val_logits_{args.tta}.npz"
    test_cache = args.output_dir / f"test_logits_{args.tta}.npz"
    val = collect_or_load(
        cache_path=val_cache,
        manifest=args.val_manifest,
        class_to_idx=reference_class_to_idx,
        image_size=int(image_size),
        models=models,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        tta=str(args.tta),
        force_cache=bool(args.force_cache),
        split="val",
        seed=int(args.seed),
    )
    test = collect_or_load(
        cache_path=test_cache,
        manifest=args.test_manifest,
        class_to_idx=reference_class_to_idx,
        image_size=int(image_size),
        models=models,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        tta=str(args.tta),
        force_cache=bool(args.force_cache),
        split="test",
        seed=int(args.seed),
    )

    result = run_calibration(val, test, idx_to_class)
    result["protocol"] = {
        "role": "post-hoc diagnostic / validation-selected calibration",
        "claim_boundary": (
            "The ConvNeXt/PhysicsTexture ensemble is frozen. This rule is not a "
            "generic calibration layer: it adds a validation natural-prior bias "
            "only to uncertain top-k candidates that are RSCD factor-neighbors. "
            "Use this result to decide whether the Top-1 gap comes from prior/"
            "decision utility mismatch before moving the mechanism into training."
        ),
        "run_dirs": [str(p) for p in args.run_dirs],
        "val_manifest": str(args.val_manifest),
        "test_manifest": str(args.test_manifest),
        "tta": str(args.tta),
        "image_size": int(image_size),
        "protocols": protocols,
    }
    (args.output_dir / "factor_neighbor_prior_utility_calibration.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "evaluate_test.json").write_text(
        json.dumps(result["evaluate_test"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "factor_neighbor_prior_utility_calibration.md").write_text(
        to_markdown(result),
        encoding="utf-8",
    )
    mirror = Path("reports/paper_protocol_summary/rscd_factor_neighbor_prior_utility_current_tta.md")
    mirror.write_text(to_markdown(result), encoding="utf-8")
    print(mirror)


@torch.no_grad()
def collect_or_load(
    *,
    cache_path: Path,
    manifest: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    models: list[nn.Module],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta: str,
    force_cache: bool,
    split: str,
    seed: int,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not force_cache:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    transform = build_transforms(int(image_size), train=False, aug_cfg={"resize_mode": "letterbox"})
    dataset = RSCDSurfaceDataset(
        manifest,
        class_to_idx=class_to_idx,
        transform=transform,
        seed=int(seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    logits_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    for batch in tqdm(loader, desc=f"collect-{split}", leave=False, ascii=True):
        image = batch["image"].to(device, non_blocking=True)
        logits_sum = None
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for model in models:
                logits = _predict_logits(model, image, tta)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            logits_mean = logits_sum / max(len(models), 1)
        logits_rows.append(logits_mean.detach().float().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy().astype(np.int64))
        paths.extend([str(x) for x in batch["image_path"]])

    payload = {
        "logits": np.concatenate(logits_rows, axis=0).astype(np.float32),
        "label": np.concatenate(labels, axis=0).astype(np.int64),
        "image_path": np.asarray(paths, dtype=object),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def run_calibration(
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    idx_to_class: dict[int, str],
) -> dict[str, Any]:
    y_val = val["label"].astype(np.int64)
    y_test = test["label"].astype(np.int64)
    val_logits = val["logits"].astype(np.float32)
    test_logits = test["logits"].astype(np.float32)
    base_val_pred = val_logits.argmax(axis=1)
    base_test_pred = test_logits.argmax(axis=1)
    baseline_val = metric_bundle(y_val, base_val_pred, idx_to_class)
    baseline_test = metric_bundle(y_test, base_test_pred, idx_to_class)
    neighbor = build_factor_neighbor_matrix(idx_to_class)
    hard_mask = build_hard_top1_mask(idx_to_class)
    prior_bias = natural_prior_bias(y_val, len(idx_to_class), alpha=2.0)

    candidates: list[dict[str, Any]] = []
    for mode in ["neighbor", "neighbor_hard", "same_material_or_neighbor"]:
        for topk in [2, 3, 5]:
            for margin_tau in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]:
                for lam in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24, 0.32]:
                    adjusted = apply_factor_neighbor_prior(
                        val_logits,
                        prior_bias=prior_bias,
                        neighbor=neighbor,
                        hard_mask=hard_mask,
                        topk=int(topk),
                        margin_tau=float(margin_tau),
                        lam=float(lam),
                        mode=mode,
                        idx_to_class=idx_to_class,
                    )
                    pred = adjusted.argmax(axis=1)
                    metrics = metric_bundle(y_val, pred, idx_to_class)
                    candidates.append(
                        {
                            "mode": mode,
                            "topk": int(topk),
                            "margin_tau": float(margin_tau),
                            "lambda": float(lam),
                            "select": metrics,
                        }
                    )

    candidates.sort(key=lambda row: selection_key(row, baseline_val), reverse=True)
    selected = candidates[0]
    adjusted_test = apply_factor_neighbor_prior(
        test_logits,
        prior_bias=prior_bias,
        neighbor=neighbor,
        hard_mask=hard_mask,
        topk=int(selected["topk"]),
        margin_tau=float(selected["margin_tau"]),
        lam=float(selected["lambda"]),
        mode=str(selected["mode"]),
        idx_to_class=idx_to_class,
    )
    test_pred = adjusted_test.argmax(axis=1)
    calibrated_test = metric_bundle(y_test, test_pred, idx_to_class)
    return {
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "selected_by_validation": selected,
        "calibrated_test": calibrated_test,
        "delta_test": {
            key: float(calibrated_test[key]) - float(baseline_test[key])
            for key in [
                "top1",
                "mean_precision",
                "mean_recall",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
                "concrete_f1",
                "rough_slight_f1",
                "rough_severe_f1",
                "wet_water_f1",
            ]
        },
        "top_validation_candidates": candidates[:20],
        "evaluate_test": evaluate_payload(
            y_test,
            test_pred,
            idx_to_class,
            claim_boundary="Validation-selected factor-neighbor prior utility calibration on frozen TTA ensemble.",
        ),
        "class_delta": class_delta(y_test, base_test_pred, test_pred, idx_to_class),
    }


def apply_factor_neighbor_prior(
    logits: np.ndarray,
    *,
    prior_bias: np.ndarray,
    neighbor: np.ndarray,
    hard_mask: np.ndarray,
    topk: int,
    margin_tau: float,
    lam: float,
    mode: str,
    idx_to_class: dict[int, str],
) -> np.ndarray:
    if lam == 0.0:
        return logits.astype(np.float32).copy()
    out = logits.astype(np.float32).copy()
    k = min(max(int(topk), 2), out.shape[1])
    order = np.argsort(out, axis=1)[:, ::-1]
    top = order[:, 0]
    kth = order[:, k - 1]
    margin = out[np.arange(len(out)), top] - out[np.arange(len(out)), kth]
    for row_idx in np.where(margin <= float(margin_tau))[0]:
        anchor = int(top[row_idx])
        candidates = order[row_idx, :k]
        allowed = np.zeros(out.shape[1], dtype=bool)
        if mode == "neighbor":
            allowed[candidates] = neighbor[anchor, candidates]
            allowed[anchor] = True
        elif mode == "neighbor_hard":
            allowed[candidates] = neighbor[anchor, candidates] & (hard_mask[candidates] | hard_mask[anchor])
            allowed[anchor] = True
        elif mode == "same_material_or_neighbor":
            anchor_info = factor_info(idx_to_class[anchor])
            for c in candidates:
                info = factor_info(idx_to_class[int(c)])
                same_material = anchor_info["material"] is not None and anchor_info["material"] == info["material"]
                allowed[int(c)] = bool(neighbor[anchor, int(c)] or same_material)
            allowed[anchor] = True
        else:
            raise ValueError(f"unknown mode: {mode}")
        out[row_idx, allowed] += float(lam) * prior_bias[allowed]
    return out


def natural_prior_bias(labels: np.ndarray, num_classes: int, *, alpha: float) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64) + float(alpha)
    prior = counts / counts.sum()
    uniform = 1.0 / float(num_classes)
    bias = np.log(prior / uniform)
    return (bias - bias.mean()).astype(np.float32)


def build_factor_neighbor_matrix(idx_to_class: dict[int, str]) -> np.ndarray:
    n = len(idx_to_class)
    out = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i, j] = True
                continue
            a = factor_info(idx_to_class[i])
            b = factor_info(idx_to_class[j])
            shared = 0
            diff = 0
            for key in ("friction", "material", "roughness"):
                av = a[key]
                bv = b[key]
                if av is None or bv is None:
                    continue
                if av == bv:
                    shared += 1
                else:
                    diff += 1
            out[i, j] = diff == 1 and shared >= 1
    return out


def build_hard_top1_mask(idx_to_class: dict[int, str]) -> np.ndarray:
    mask = []
    for idx in range(len(idx_to_class)):
        info = factor_info(idx_to_class[idx])
        name = idx_to_class[idx]
        hard = (
            info["material"] == "concrete"
            or info["roughness"] in {"slight", "severe"}
            or (info["friction"] in {"wet", "water"} and info["material"] in {"asphalt", "concrete"})
            or name in {"dry_asphalt_severe", "water_gravel"}
        )
        mask.append(bool(hard))
    return np.asarray(mask, dtype=bool)


def factor_info(name: str) -> dict[str, str | None]:
    name = str(name).lower().replace("-", "_")
    if name in {"fresh_snow", "melted_snow", "ice"}:
        return {"friction": name, "material": None, "roughness": None}
    parts = name.split("_")
    friction = parts[0] if len(parts) >= 1 else None
    material = parts[1] if len(parts) >= 2 else None
    roughness = parts[2] if len(parts) >= 3 else None
    return {"friction": friction, "material": material, "roughness": roughness}


def selection_key(row: dict[str, Any], baseline: dict[str, float]) -> tuple[float, float, float, float]:
    metrics = row["select"]
    macro_drop = max(0.0, float(baseline["macro_f1"]) - float(metrics["macro_f1"]))
    wet_drop = max(0.0, float(baseline["wet_water_f1"]) - float(metrics["wet_water_f1"]))
    return (
        float(metrics["top1"]) - 0.75 * macro_drop - 0.35 * wet_drop,
        float(metrics["macro_f1"]),
        float(metrics["wet_water_f1"]),
        float(metrics["concrete_f1"]),
    )


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, float]:
    labels = sorted(idx_to_class)
    names = [idx_to_class[idx] for idx in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "concrete_f1": grouped_f1(report, lambda name: factor_info(name)["material"] == "concrete"),
        "rough_slight_f1": grouped_f1(report, lambda name: factor_info(name)["roughness"] == "slight"),
        "rough_severe_f1": grouped_f1(report, lambda name: factor_info(name)["roughness"] == "severe"),
        "wet_water_f1": grouped_f1(report, lambda name: factor_info(name)["friction"] in {"wet", "water"}),
    }


def grouped_f1(report: dict[str, Any], keep) -> float:
    rows = []
    for name, item in report.items():
        if not isinstance(item, dict) or "f1-score" not in item:
            continue
        if keep(str(name)):
            rows.append(float(item["f1-score"]))
    return float(np.mean(rows)) if rows else 0.0


def evaluate_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: dict[int, str],
    *,
    claim_boundary: str,
) -> dict[str, Any]:
    labels = sorted(idx_to_class)
    names = [idx_to_class[idx] for idx in labels]
    return {
        "summary": {
            "top1": float(accuracy_score(y_true, y_pred)),
            "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "num_samples": int(len(y_true)),
            "num_classes": int(len(labels)),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion": confusion_rows(y_true.astype(int).tolist(), y_pred.astype(int).tolist(), idx_to_class),
        "claim_boundary": claim_boundary,
    }


def class_delta(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    new_pred: np.ndarray,
    idx_to_class: dict[int, str],
) -> list[dict[str, Any]]:
    labels = sorted(idx_to_class)
    base = classification_report(
        y_true,
        base_pred,
        labels=labels,
        target_names=[idx_to_class[idx] for idx in labels],
        output_dict=True,
        zero_division=0,
    )
    new = classification_report(
        y_true,
        new_pred,
        labels=labels,
        target_names=[idx_to_class[idx] for idx in labels],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for idx in labels:
        name = idx_to_class[idx]
        rows.append(
            {
                "class_label": name,
                "support": int(base[name]["support"]),
                "base_f1": float(base[name]["f1-score"]),
                "new_f1": float(new[name]["f1-score"]),
                "delta_f1": float(new[name]["f1-score"]) - float(base[name]["f1-score"]),
                "base_recall": float(base[name]["recall"]),
                "new_recall": float(new[name]["recall"]),
                "delta_recall": float(new[name]["recall"]) - float(base[name]["recall"]),
            }
        )
    rows.sort(key=lambda row: row["delta_f1"])
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def to_markdown(result: dict[str, Any]) -> str:
    base = result["baseline_test"]
    cal = result["calibrated_test"]
    delta = result["delta_test"]
    selected = result["selected_by_validation"]
    lines = [
        "# RSCD Factor-neighbor Prior Utility Calibration",
        "",
        result["protocol"]["claim_boundary"],
        "",
        "## Why This Targets The Top-1 Gap",
        "",
        (
            "The current model is macro-F1 friendly: it preserves many per-class "
            "boundaries, but Top-1 loses many samples on high-support concrete and "
            "wet/water neighbor confusions. This calibration tests whether a small "
            "natural-prior correction inside RSCD factor-neighbor candidates can "
            "recover sample-weighted accuracy without destroying Mean-F1."
        ),
        "",
        "## Selected Rule",
        "",
        f"- Mode: `{selected['mode']}`",
        f"- Top-k candidate set: `{selected['topk']}`",
        f"- Margin threshold: `{selected['margin_tau']}`",
        f"- Prior-bias lambda: `{selected['lambda']}`",
        "- Selection: validation only; test used once after rule selection.",
        "",
        "## Test Result",
        "",
        "| method | Top-1 | Mean-P | Mean-R | Mean-F1 | Weighted-F1 | Concrete F1 | Slight F1 | Severe F1 | Wet/Water F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| frozen TTA ensemble | "
            f"{pct(base['top1'])} | {pct(base['mean_precision'])} | {pct(base['mean_recall'])} | "
            f"{pct(base['macro_f1'])} | {pct(base['weighted_f1'])} | {pct(base['concrete_f1'])} | "
            f"{pct(base['rough_slight_f1'])} | {pct(base['rough_severe_f1'])} | {pct(base['wet_water_f1'])} |"
        ),
        (
            "| factor-neighbor prior utility | "
            f"{pct(cal['top1'])} | {pct(cal['mean_precision'])} | {pct(cal['mean_recall'])} | "
            f"{pct(cal['macro_f1'])} | {pct(cal['weighted_f1'])} | {pct(cal['concrete_f1'])} | "
            f"{pct(cal['rough_slight_f1'])} | {pct(cal['rough_severe_f1'])} | {pct(cal['wet_water_f1'])} |"
        ),
        (
            "| delta | "
            f"{pp(delta['top1'])} | {pp(delta['mean_precision'])} | {pp(delta['mean_recall'])} | "
            f"{pp(delta['macro_f1'])} | {pp(delta['weighted_f1'])} | {pp(delta['concrete_f1'])} | "
            f"{pp(delta['rough_slight_f1'])} | {pp(delta['rough_severe_f1'])} | {pp(delta['wet_water_f1'])} |"
        ),
        "",
        "## Largest Class F1 Drops",
        "",
        "| class | support | base F1 | new F1 | delta F1 | base recall | new recall | delta recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["class_delta"][:10]:
        lines.append(
            f"| `{row['class_label']}` | {row['support']} | {pct(row['base_f1'])} | {pct(row['new_f1'])} | "
            f"{pp(row['delta_f1'])} | {pct(row['base_recall'])} | {pct(row['new_recall'])} | {pp(row['delta_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest Class F1 Gains",
            "",
            "| class | support | base F1 | new F1 | delta F1 | base recall | new recall | delta recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in list(reversed(result["class_delta"]))[:10]:
        lines.append(
            f"| `{row['class_label']}` | {row['support']} | {pct(row['base_f1'])} | {pct(row['new_f1'])} | "
            f"{pp(row['delta_f1'])} | {pct(row['base_recall'])} | {pct(row['new_recall'])} | {pp(row['delta_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Validation Candidates",
            "",
            "| rank | mode | top-k | tau | lambda | val Top-1 | val Mean-F1 | val concrete F1 | val wet/water F1 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(result["top_validation_candidates"], start=1):
        metrics = row["select"]
        lines.append(
            f"| {rank} | `{row['mode']}` | {row['topk']} | {row['margin_tau']:.2f} | {row['lambda']:.2f} | "
            f"{pct(metrics['top1'])} | {pct(metrics['macro_f1'])} | {pct(metrics['concrete_f1'])} | "
            f"{pct(metrics['wet_water_f1'])} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

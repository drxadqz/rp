from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from friction_affordance.models.rspnet import rspnet_l  # noqa: E402
from friction_affordance.transforms import build_transforms  # noqa: E402
import run_rscd_surface_classification as rscd  # noqa: E402


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)


def physics_stats(image: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(device=image.device, dtype=image.dtype)
    std = STD.to(device=image.device, dtype=image.dtype)
    sobel_x = SOBEL_X.to(device=image.device, dtype=image.dtype)
    sobel_y = SOBEL_Y.to(device=image.device, dtype=image.dtype)
    laplace = LAPLACE.to(device=image.device, dtype=image.dtype)
    rgb = (image * std + mean).clamp(0.0, 1.0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    maxc = rgb.max(dim=1, keepdim=True).values
    minc = rgb.min(dim=1, keepdim=True).values
    saturation = (maxc - minc) / maxc.clamp_min(1e-4)
    value = maxc
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
    lap = F.conv2d(gray, laplace, padding=1).abs()
    local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)
    specular = torch.sigmoid((value - 0.74) * 16.0) * torch.sigmoid((0.30 - saturation) * 14.0)
    dark_water = torch.sigmoid((0.44 - value) * 10.0) * torch.sigmoid((0.30 - saturation) * 10.0)
    thin_film = torch.sigmoid((0.040 - grad) * 35.0) * torch.sigmoid((0.040 - local_contrast) * 35.0)
    texture_erasure = torch.sigmoid((0.055 - grad) * 30.0)
    rough_aggregate = torch.sigmoid((grad - 0.070) * 20.0) * torch.sigmoid((lap - 0.060) * 18.0)
    snow = torch.sigmoid((value - 0.64) * 10.0) * torch.sigmoid((0.25 - saturation) * 10.0)
    mud = torch.sigmoid((rgb[:, 0:1] - rgb[:, 2:3]) * 8.0) * torch.sigmoid((0.62 - value) * 6.0)
    marking = torch.sigmoid((value - 0.78) * 14.0) * torch.sigmoid((saturation - 0.18) * 8.0)
    stat_maps = [
        value,
        saturation,
        grad,
        lap,
        local_contrast,
        specular,
        dark_water,
        thin_film,
        texture_erasure,
        rough_aggregate,
        snow,
        mud,
        marking,
    ]
    pooled = []
    for item in stat_maps:
        pooled.append(item.mean(dim=(2, 3)))
        pooled.append(item.std(dim=(2, 3), unbiased=False))
    return torch.cat(pooled, dim=1)


def build_factor_and_group_masks(class_to_idx: dict[str, int], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    factor_masks = rscd.build_factor_marginal_masks(class_to_idx, device)
    group_names = ("core_asphalt_concrete", "granular_mud_gravel", "winter_snow_ice", "wet_water_core")
    group_masks = torch.zeros((len(group_names), len(class_to_idx)), device=device)
    for class_name, class_idx in class_to_idx.items():
        parts = class_name.split("_")
        if class_name in {"fresh_snow", "melted_snow", "ice"}:
            group_masks[2, class_idx] = 1.0
            continue
        friction = parts[0] if len(parts) > 0 else ""
        material = parts[1] if len(parts) > 1 else ""
        if material in {"asphalt", "concrete"}:
            group_masks[0, class_idx] = 1.0
            if friction in {"wet", "water"}:
                group_masks[3, class_idx] = 1.0
        if material in {"mud", "gravel"}:
            group_masks[1, class_idx] = 1.0
    return factor_masks, group_masks


def coupling_features(
    logits: torch.Tensor,
    stats: torch.Tensor,
    factor_masks: dict[str, torch.Tensor],
    group_masks: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    factor_parts = []
    for name in ("friction", "material", "unevenness"):
        mask = factor_masks[name].to(device=logits.device, dtype=logits.dtype)
        marginal = torch.matmul(probs, mask.T)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(1e-8)
        factor_parts.append(marginal)
    factor_prob = torch.cat(factor_parts, dim=1)
    group_prob = torch.matmul(probs, group_masks.to(device=logits.device, dtype=logits.dtype).T)
    group_prob = group_prob / group_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
    factor_coupling = (stats.unsqueeze(2) * factor_prob.unsqueeze(1)).flatten(1)
    group_coupling = (stats.unsqueeze(2) * group_prob.unsqueeze(1)).flatten(1)
    return torch.cat([factor_prob, group_prob, factor_coupling, group_coupling], dim=1)


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    factor_masks: dict[str, torch.Tensor],
    group_masks: torch.Tensor,
    *,
    use_coupling_features: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features = []
    labels = []
    for batch in tqdm(loader, desc="extract", ascii=True):
        image = batch["image"].to(device)
        logits = model(image)
        stats = physics_stats(image)
        probs = torch.softmax(logits, dim=1)
        margin = torch.topk(probs, k=2, dim=1).values
        margin = (margin[:, 0] - margin[:, 1]).unsqueeze(1)
        parts = [logits, torch.log(probs.clamp_min(1e-8)), margin, stats]
        if use_coupling_features:
            parts.append(coupling_features(logits, stats, factor_masks, group_masks))
        feature = torch.cat(parts, dim=1)
        features.append(feature.detach().cpu().numpy().astype(np.float32))
        labels.append(batch["label"].numpy().astype(np.int64))
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, Any]:
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[idx] for idx in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "summary": {
            "top1": float(accuracy_score(y_true, y_pred)),
            "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "num_samples": int(y_true.shape[0]),
            "num_classes": int(len(labels)),
        },
        "classification_report": report,
    }


def class_relation_groups(idx_to_class: dict[int, str]) -> np.ndarray:
    """Map RSCD classes to relation-specific coupling families.

    The grouping is deliberately coarse and uses only label names. It encodes the
    hypothesis that wet/water core roads, dry core roughness, granular roads, and
    winter phase states need different correction rules.
    """

    groups = np.zeros(len(idx_to_class), dtype=np.int64)
    for idx, class_name in idx_to_class.items():
        parts = class_name.split("_")
        if class_name in {"fresh_snow", "melted_snow", "ice"}:
            groups[idx] = 3
            continue
        friction = parts[0] if len(parts) > 0 else ""
        material = parts[1] if len(parts) > 1 else ""
        if material in {"mud", "gravel"}:
            groups[idx] = 2
        elif material in {"asphalt", "concrete"} and friction in {"wet", "water"}:
            groups[idx] = 1
        else:
            groups[idx] = 0
    return groups


def make_logistic(c_value: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=1,
            random_state=int(seed),
        ),
    )


def predict_relation_experts(
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    *,
    num_classes: int,
    group_by_class: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    """Train expert calibrators on predicted relation families, then route test samples."""

    global_model = make_logistic(c_value, seed)
    global_model.fit(x_cal, y_cal)
    fallback = global_model.predict(x_test)
    cal_base_pred = x_cal[:, :num_classes].argmax(axis=1)
    test_base_pred = x_test[:, :num_classes].argmax(axis=1)
    cal_groups = group_by_class[cal_base_pred]
    test_groups = group_by_class[test_base_pred]
    pred = fallback.copy()
    for group_id in sorted(np.unique(group_by_class).tolist()):
        train_mask = cal_groups == group_id
        test_mask = test_groups == group_id
        if int(train_mask.sum()) < 80 or int(test_mask.sum()) == 0 or np.unique(y_cal[train_mask]).shape[0] < 2:
            continue
        expert = make_logistic(c_value, seed + 17 + int(group_id))
        expert.fit(x_cal[train_mask], y_cal[train_mask])
        pred[test_mask] = expert.predict(x_test[test_mask])
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:\NMI_SPWFM_datasets\model_zoo\RSPNet_L.pth"))
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_val.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_test.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-split", choices=("val", "train"), default="val")
    parser.add_argument("--max-calibration-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibrator-c", type=float, default=0.20)
    parser.add_argument(
        "--calibrator-mode",
        choices=("global", "relation_experts"),
        default="global",
        help="Use one posterior calibrator or separate calibrators routed by predicted RSCD relation family.",
    )
    parser.add_argument("--coupling-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = rscd.build_class_map([args.train_manifest, args.val_manifest, args.test_manifest])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    transform = build_transforms(224, train=False, aug_cfg={"resize_mode": "stretch"})
    calibration_manifest = args.val_manifest if args.calibration_split == "val" else args.train_manifest
    calibration_ds = rscd.RSCDSurfaceDataset(
        calibration_manifest,
        class_to_idx=class_to_idx,
        transform=transform,
        max_samples=args.max_calibration_samples,
        seed=args.seed,
    )
    test_ds = rscd.RSCDSurfaceDataset(
        args.test_manifest,
        class_to_idx=class_to_idx,
        transform=transform,
        max_samples=args.max_test_samples,
        seed=args.seed + 1,
    )
    calibration_loader = DataLoader(
        calibration_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=rscd.collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=rscd.collate,
    )
    model = rspnet_l(num_classes=len(class_to_idx))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    factor_masks, group_masks = build_factor_and_group_masks(class_to_idx, device)

    cache_dir = args.cache_dir
    cache_name = (
        f"features_{args.calibration_split}_cal{args.max_calibration_samples or 'full'}_"
        f"test{args.max_test_samples or 'full'}_coupling{int(bool(args.coupling_features))}.npz"
    )
    cache_path = cache_dir / cache_name if cache_dir is not None else None
    if cache_path is not None and bool(args.reuse_cache) and cache_path.exists():
        cache = np.load(cache_path)
        x_cal = cache["x_cal"]
        y_cal = cache["y_cal"]
        x_test = cache["x_test"]
        y_test = cache["y_test"]
    else:
        x_cal, y_cal = extract_features(
            model,
            calibration_loader,
            device,
            factor_masks,
            group_masks,
            use_coupling_features=bool(args.coupling_features),
        )
        x_test, y_test = extract_features(
            model,
            test_loader,
            device,
            factor_masks,
            group_masks,
            use_coupling_features=bool(args.coupling_features),
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, x_cal=x_cal, y_cal=y_cal, x_test=x_test, y_test=y_test)
    base_pred = x_test[:, : len(class_to_idx)].argmax(axis=1)
    base_metrics = summarize(y_test, base_pred, idx_to_class)
    if str(args.calibrator_mode) == "relation_experts":
        cal_pred = predict_relation_experts(
            x_cal,
            y_cal,
            x_test,
            num_classes=len(class_to_idx),
            group_by_class=class_relation_groups(idx_to_class),
            c_value=float(args.calibrator_c),
            seed=int(args.seed),
        )
    else:
        calibrator = make_logistic(float(args.calibrator_c), int(args.seed))
        calibrator.fit(x_cal, y_cal)
        cal_pred = calibrator.predict(x_test)
    cal_metrics = summarize(y_test, cal_pred, idx_to_class)
    out = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "calibration_split": args.calibration_split,
            "calibration_samples": int(y_cal.shape[0]),
            "test_samples": int(y_test.shape[0]),
            "feature": "RSPNet logits + log-probabilities + confidence margin + 26 physics statistics",
            "coupling_features": bool(args.coupling_features),
            "calibrator_c": float(args.calibrator_c),
            "calibrator_mode": str(args.calibrator_mode),
            "cache_path": str(cache_path) if cache_path is not None else None,
            "note": "Calibration labels are not test labels; this is a frozen-backbone physics-evidence posterior calibrator.",
        },
        "base": base_metrics,
        "physics_calibrator": cal_metrics,
    }
    (args.output_dir / "physics_calibrator_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"base": base_metrics["summary"], "physics_calibrator": cal_metrics["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

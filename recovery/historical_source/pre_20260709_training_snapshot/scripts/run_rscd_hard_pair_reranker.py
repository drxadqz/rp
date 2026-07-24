from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from run_rscd_surface_classification import (
    RSCDSurfaceDataset,
    _factor_text,
    build_class_map,
    confusion_rows,
    load_state_dict_allow_expanded_head,
)
from run_rscd_topology_logit_calibration import build_model_from_protocol, load_protocol
from friction_affordance.engine import dataloader_worker_settings
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed


DEFAULT_TRAIN = Path("data/manifests_full/rscd_prepared_train.csv")
DEFAULT_VAL = Path("data/manifests_full/rscd_prepared_val.csv")
DEFAULT_TEST = Path("data/manifests_full/rscd_prepared_test.csv")
DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "screen_physics_texture_hardboost025_lr1e5_s36k_e1_seed101_from_best" / "best.pt"
DEFAULT_OUT = DEFAULT_ROOT / "posthoc_hard_pair_reranker_current_best"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-fitted RSCD hard-pair top-2 reranker.")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=None,
        help="Optional run directory with protocol.json. When set, rebuild the exact model architecture from it.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backbone", default="convnext_tiny")
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--min-pair-samples", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=999)
    parser.add_argument("--no-physics-quality-region-cues", action="store_true")
    parser.add_argument(
        "--use-local-physics-stats",
        action="store_true",
        help=(
            "Append raw LocalPhysicsField evidence statistics to each hard-pair "
            "reranker. This tests relation-triggered local evidence without "
            "changing the neural checkpoint."
        ),
    )
    parser.add_argument(
        "--decision-mode",
        choices=("predict", "validation_threshold"),
        default="predict",
        help=(
            "predict uses the pair classifier's default decision. "
            "validation_threshold learns a per-pair threshold on validation to "
            "keep only beneficial top-2 flips."
        ),
    )
    parser.add_argument(
        "--min-pair-gain",
        type=float,
        default=0.0,
        help="Minimum validation pair-accuracy gain required before a pair reranker is kept.",
    )
    parser.add_argument(
        "--max-side-recall-drop",
        type=float,
        default=1.0,
        help=(
            "Maximum allowed validation recall drop for either side of a pair "
            "when using validation_threshold. Use a small value to protect "
            "Macro-F1-sensitive classes."
        ),
    )
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    args = parser.parse_args()

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    class_to_idx = build_class_map([args.train_manifest, args.val_manifest, args.test_manifest])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    if args.source_run is not None:
        protocol = load_protocol(args.source_run / "protocol.json")
        model = build_model_from_protocol(protocol["args"], class_to_idx).to(device)
        checkpoint = args.checkpoint if args.checkpoint != DEFAULT_CHECKPOINT else (args.source_run / "best.pt")
    else:
        from run_rscd_surface_classification import SurfaceClassifier

        model = SurfaceClassifier(
            backbone=str(args.backbone),
            embedding_dim=int(args.embedding_dim),
            num_classes=len(class_to_idx),
            pretrained=False,
            dropout=0.0,
            use_physics_branch=True,
            physics_dim=96,
            physics_quality_cues=True,
            physics_quality_region_cues=not bool(args.no_physics_quality_region_cues),
            class_to_idx=class_to_idx,
        ).to(device)
        checkpoint = args.checkpoint
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = state.get("model", state)
    missing, unexpected, partial = load_state_dict_allow_expanded_head(model, state_dict)
    if unexpected:
        print(f"WARNING unexpected checkpoint keys: {unexpected[:8]}", flush=True)
    if missing or partial:
        print(f"Checkpoint load missing={missing[:8]} partial={partial[:8]}", flush=True)

    transform = build_transforms(image_size=int(args.image_size), train=False, aug_cfg={"resize_mode": "letterbox"})
    num_workers, loader_kwargs = dataloader_worker_settings(
        {"num_workers": int(args.num_workers), "prefetch_factor": int(args.prefetch_factor)}
    )
    val_loader = DataLoader(
        RSCDSurfaceDataset(args.val_manifest, class_to_idx=class_to_idx, transform=transform),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        RSCDSurfaceDataset(args.test_manifest, class_to_idx=class_to_idx, transform=transform),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )

    val = collect_predictions(
        model,
        val_loader,
        device,
        idx_to_class,
        desc="val",
        use_local_physics_stats=bool(args.use_local_physics_stats),
    )
    test = collect_predictions(
        model,
        test_loader,
        device,
        idx_to_class,
        desc="test",
        use_local_physics_stats=bool(args.use_local_physics_stats),
    )
    hard_pairs = build_hard_pairs(class_to_idx)
    rerankers, pair_rows = fit_pair_rerankers(
        val,
        hard_pairs,
        idx_to_class,
        min_samples=int(args.min_pair_samples),
        decision_mode=str(args.decision_mode),
        min_pair_gain=float(args.min_pair_gain),
        max_side_recall_drop=float(args.max_side_recall_drop),
        threshold_grid_size=int(args.threshold_grid_size),
    )
    reranked_pred, applied_rows = apply_pair_rerankers(test, rerankers, idx_to_class)
    base_metrics = metrics_from_arrays(test["true"], test["pred"], idx_to_class)
    reranked_metrics = metrics_from_arrays(test["true"], reranked_pred, idx_to_class)
    comparison = summarize_slice_comparison(base_metrics, reranked_metrics)

    payload = {
        "protocol": {
            "role": "post-hoc validation-fitted hard-pair top-2 reranker",
            "claim_boundary": (
                "This is not a pure end-to-end single-model result. It tests whether "
                "audited heterophilic RSCD hard pairs contain recoverable boundary signal."
            ),
            "checkpoint": str(checkpoint),
            "source_run": str(args.source_run) if args.source_run is not None else None,
            "val_manifest": str(args.val_manifest),
            "test_manifest": str(args.test_manifest),
            "min_pair_samples": int(args.min_pair_samples),
            "use_local_physics_stats": bool(args.use_local_physics_stats),
            "decision_mode": str(args.decision_mode),
            "min_pair_gain": float(args.min_pair_gain),
            "max_side_recall_drop": float(args.max_side_recall_drop),
            "threshold_grid_size": int(args.threshold_grid_size),
            "num_rerankers": len(rerankers),
            "num_hard_pairs": len(hard_pairs),
            "applied_test_samples": len(applied_rows),
        },
        "base": base_metrics,
        "reranked": reranked_metrics,
        "slice_comparison": comparison,
        "trained_pair_rerankers": pair_rows,
        "applied_examples": applied_rows[:200],
    }
    (args.output_dir / "evaluate_test.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "evaluate_test.md").write_text(to_markdown(payload), encoding="utf-8")
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "pair_rerankers.csv", index=False, encoding="utf-8")
    pd.DataFrame(applied_rows).to_csv(args.output_dir / "applied_test_samples.csv", index=False, encoding="utf-8")
    print(json.dumps({"base": base_metrics["summary"], "reranked": reranked_metrics["summary"], "applied": len(applied_rows)}, indent=2), flush=True)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: dict[int, str],
    *,
    desc: str,
    use_local_physics_stats: bool,
) -> dict[str, Any]:
    model.eval()
    true_rows: list[int] = []
    pred_rows: list[int] = []
    logit_rows: list[np.ndarray] = []
    local_stats_rows: list[np.ndarray] = []
    path_rows: list[str] = []
    for batch in tqdm(loader, desc=f"reranker {desc}", leave=False, ascii=True):
        image = batch["image"].to(device)
        logits = model(image)
        pred = logits.argmax(dim=1)
        true_rows.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
        pred_rows.extend(pred.detach().cpu().numpy().astype(int).tolist())
        logit_rows.extend(logits.detach().cpu().numpy().astype(np.float32))
        if use_local_physics_stats:
            local_stats_rows.extend(local_physics_evidence_stats(image).detach().cpu().numpy().astype(np.float32))
        path_rows.extend(str(x) for x in batch["image_path"])
    logits_np = np.stack(logit_rows, axis=0)
    probs = softmax_np(logits_np)
    top2 = np.argsort(-probs, axis=1)[:, :2].astype(np.int64)
    payload = {
        "true": np.asarray(true_rows, dtype=np.int64),
        "pred": np.asarray(pred_rows, dtype=np.int64),
        "logits": logits_np,
        "probs": probs,
        "top2": top2,
        "paths": path_rows,
        "idx_to_class": idx_to_class,
    }
    if use_local_physics_stats:
        payload["local_physics_stats"] = np.stack(local_stats_rows, axis=0).astype(np.float32)
    return payload


def fit_pair_rerankers(
    val: dict[str, Any],
    hard_pairs: set[tuple[int, int]],
    idx_to_class: dict[int, str],
    *,
    min_samples: int,
    decision_mode: str,
    min_pair_gain: float,
    max_side_recall_drop: float,
    threshold_grid_size: int,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(val["top2"]):
        pair = tuple(sorted((int(a), int(b))))
        if pair in hard_pairs and int(val["true"][i]) in pair:
            buckets[pair].append(i)
    rerankers: dict[tuple[int, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for pair, indices in sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True):
        if len(indices) < int(min_samples):
            continue
        x = pair_features(val, indices, pair)
        y = np.asarray([1 if int(val["true"][i]) == pair[1] else 0 for i in indices], dtype=np.int64)
        if len(set(y.tolist())) < 2:
            continue
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, solver="liblinear", random_state=101)
        clf.fit(x, y)
        base_pair_pred = np.asarray([1 if int(val["pred"][i]) == pair[1] else 0 for i in indices], dtype=np.int64)
        base_acc = float((base_pair_pred == y).mean())
        threshold = 0.5
        side_recall_drop = 0.0
        if decision_mode == "validation_threshold":
            prob_b = clf.predict_proba(x)[:, 1]
            threshold, pred, fit_acc, side_recall_drop = select_validation_threshold(
                prob_b,
                y,
                base_pair_pred,
                grid_size=int(threshold_grid_size),
                max_side_recall_drop=float(max_side_recall_drop),
            )
        else:
            pred = clf.predict(x)
            fit_acc = float((pred == y).mean())
            side_recall_drop = pair_side_recall_drop(y, base_pair_pred, pred)
        if fit_acc + 1e-6 < base_acc + float(min_pair_gain):
            continue
        rerankers[pair] = {"clf": clf, "threshold": float(threshold), "decision_mode": decision_mode}
        rows.append(
            {
                "class_a": idx_to_class[pair[0]],
                "class_b": idx_to_class[pair[1]],
                "val_samples": len(indices),
                "base_pair_acc": base_acc,
                "fit_pair_acc": fit_acc,
                "fit_pair_gain": fit_acc - base_acc,
                "threshold": float(threshold),
                "side_recall_drop": float(side_recall_drop),
                "coef_norm": float(np.linalg.norm(clf.coef_)),
            }
        )
    return rerankers, rows


def apply_pair_rerankers(
    data: dict[str, Any],
    rerankers: dict[tuple[int, int], dict[str, Any]],
    idx_to_class: dict[int, str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pred = np.asarray(data["pred"], dtype=np.int64).copy()
    applied: list[dict[str, Any]] = []
    pair_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(data["top2"]):
        pair = tuple(sorted((int(a), int(b))))
        if pair in rerankers:
            pair_to_indices[pair].append(i)
    for pair, indices in pair_to_indices.items():
        x = pair_features(data, indices, pair)
        item = rerankers[pair]
        clf = item["clf"]
        if item.get("decision_mode") == "validation_threshold":
            threshold = float(item.get("threshold", 0.5))
            pair_pred = (clf.predict_proba(x)[:, 1] >= threshold).astype(np.int64)
        else:
            pair_pred = clf.predict(x)
        for row_idx, choose_b in zip(indices, pair_pred, strict=True):
            old = int(pred[row_idx])
            new = int(pair[1] if int(choose_b) == 1 else pair[0])
            pred[row_idx] = new
            if new != old:
                applied.append(
                    {
                        "image_path": data["paths"][row_idx],
                        "true_label": idx_to_class[int(data["true"][row_idx])],
                        "old_pred": idx_to_class[old],
                        "new_pred": idx_to_class[new],
                        "pair": f"{idx_to_class[pair[0]]} | {idx_to_class[pair[1]]}",
                    }
                )
    return pred, applied


def select_validation_threshold(
    prob_b: np.ndarray,
    y: np.ndarray,
    base_pair_pred: np.ndarray,
    *,
    grid_size: int,
    max_side_recall_drop: float,
) -> tuple[float, np.ndarray, float, float]:
    grid_size = max(int(grid_size), 3)
    candidates = np.linspace(0.05, 0.95, grid_size, dtype=np.float32)
    best_threshold = 0.5
    best_pred = (prob_b >= best_threshold).astype(np.int64)
    best_acc = float((best_pred == y).mean())
    best_drop = pair_side_recall_drop(y, base_pair_pred, best_pred)
    best_objective = -1.0
    for threshold in candidates:
        pred = (prob_b >= float(threshold)).astype(np.int64)
        acc = float((pred == y).mean())
        drop = pair_side_recall_drop(y, base_pair_pred, pred)
        if drop > float(max_side_recall_drop):
            objective = acc - 2.0 * (drop - float(max_side_recall_drop))
        else:
            objective = acc
        if objective > best_objective or (math.isclose(objective, best_objective) and drop < best_drop):
            best_objective = objective
            best_threshold = float(threshold)
            best_pred = pred
            best_acc = acc
            best_drop = drop
    return best_threshold, best_pred, best_acc, best_drop


def pair_side_recall_drop(y: np.ndarray, base_pair_pred: np.ndarray, new_pair_pred: np.ndarray) -> float:
    worst_drop = 0.0
    for side in (0, 1):
        keep = y == side
        if not bool(np.any(keep)):
            continue
        base_recall = float((base_pair_pred[keep] == side).mean())
        new_recall = float((new_pair_pred[keep] == side).mean())
        worst_drop = max(worst_drop, base_recall - new_recall)
    return worst_drop


def pair_features(data: dict[str, Any], indices: list[int], pair: tuple[int, int]) -> np.ndarray:
    logits = data["logits"][indices]
    probs = data["probs"][indices]
    a, b = pair
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1, keepdims=True) / math.log(probs.shape[1])
    feats = np.column_stack(
        [
            logits[:, a] - logits[:, b],
            probs[:, a] - probs[:, b],
            logits[:, a],
            logits[:, b],
            probs[:, a],
            probs[:, b],
            sorted_probs[:, 0],
            sorted_probs[:, 1],
            sorted_probs[:, 0] - sorted_probs[:, 1],
            entropy[:, 0],
        ]
    )
    if "local_physics_stats" in data:
        local_stats = np.asarray(data["local_physics_stats"][indices], dtype=np.float32)
        feats = np.concatenate([feats.astype(np.float32), local_stats], axis=1)
    return feats.astype(np.float32)


def local_physics_evidence_stats(image: torch.Tensor, grid_size: int = 3) -> torch.Tensor:
    """Raw LocalPhysicsField evidence statistics before the learned projection."""

    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3) / 8.0
    laplace = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    rgb = (image * std + mean).clamp(0.0, 1.0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    maxc = rgb.max(dim=1, keepdim=True).values
    minc = rgb.min(dim=1, keepdim=True).values
    value = maxc
    saturation = (maxc - minc) / maxc.clamp_min(1e-4)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
    lap = F.conv2d(gray, laplace, padding=1).abs()
    local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

    low_texture = torch.sigmoid((0.045 - grad) * 35.0)
    low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
    specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
    dark_water = torch.sigmoid((0.42 - value) * 10.0) * torch.sigmoid((0.30 - saturation) * 12.0) * low_texture
    thin_film = torch.clamp(specular + 0.6 * dark_water, 0.0, 1.0) * torch.sigmoid((0.08 - lap) * 22.0)
    texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
    rough_energy = torch.sigmoid((grad - 0.075) * 22.0)
    bright_marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
    wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)

    fields = [
        specular,
        dark_water,
        thin_film,
        texture_erasure,
        low_texture,
        rough_energy,
        low_contrast,
        bright_marking,
    ]
    stats: list[torch.Tensor] = []
    for idx, field in enumerate(fields):
        grid = F.adaptive_avg_pool2d(field, (grid_size, grid_size)).flatten(1)
        conn_source = wet_proxy if idx == 7 else field
        stats.extend(
            [
                grid,
                field.mean(dim=(2, 3)),
                field.std(dim=(2, 3)),
                field.amax(dim=(2, 3)),
                soft_connectedness(conn_source),
            ]
        )
    return torch.cat(stats, dim=1)


def soft_connectedness(mask: torch.Tensor) -> torch.Tensor:
    horizontal = (mask[:, :, :, :-1] * mask[:, :, :, 1:]).mean(dim=(2, 3))
    vertical = (mask[:, :, :-1, :] * mask[:, :, 1:, :]).mean(dim=(2, 3))
    return 0.5 * (horizontal + vertical)


def build_hard_pairs(class_to_idx: dict[str, int]) -> set[tuple[int, int]]:
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    pairs: set[tuple[int, int]] = set()
    for i, name_i in idx_to_class.items():
        factors_i = _factor_text(name_i)
        for j, name_j in idx_to_class.items():
            if i >= j:
                continue
            factors_j = _factor_text(name_j)
            relation = hard_relation(factors_i, factors_j)
            if relation:
                pairs.add(tuple(sorted((i, j))))
    audited_pairs = [
        ("dry_concrete_slight", "dry_concrete_severe"),
        ("water_concrete_smooth", "wet_concrete_smooth"),
        ("dry_mud", "dry_gravel"),
        ("wet_mud", "wet_gravel"),
        ("water_concrete_slight", "water_concrete_severe"),
        ("wet_concrete_slight", "wet_concrete_severe"),
        ("dry_asphalt_slight", "dry_asphalt_severe"),
        ("water_asphalt_slight", "water_asphalt_severe"),
        ("wet_asphalt_slight", "wet_asphalt_severe"),
    ]
    for a, b in audited_pairs:
        if a in class_to_idx and b in class_to_idx:
            pairs.add(tuple(sorted((class_to_idx[a], class_to_idx[b]))))
    return pairs


def hard_relation(a: dict[str, str], b: dict[str, str]) -> str | None:
    if not a or not b:
        return None
    keys = ("friction", "material", "unevenness")
    same = [a.get(k) == b.get(k) for k in keys]
    if sum(same) != 2:
        return None
    if a.get("material") == b.get("material") == "concrete" and a.get("friction") == b.get("friction"):
        return "concrete_roughness"
    if a.get("material") == b.get("material") and a.get("unevenness") == b.get("unevenness"):
        if {a.get("friction"), b.get("friction")} in [{"wet", "water"}, {"dry", "wet"}]:
            return "friction_neighbor"
    if a.get("friction") == b.get("friction") and a.get("unevenness") == b.get("unevenness"):
        if {a.get("material"), b.get("material")} in [{"mud", "gravel"}, {"asphalt", "concrete"}]:
            return "material_neighbor"
    return None


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, Any]:
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
    summary = {
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "num_samples": int(len(y_true)),
        "num_classes": int(len(labels)),
    }
    return {
        "summary": summary,
        "classification_report": report,
        "confusion": confusion_rows(y_true.tolist(), y_pred.tolist(), idx_to_class),
    }


def summarize_slice_comparison(base: dict[str, Any], reranked: dict[str, Any]) -> dict[str, Any]:
    base_rows = class_rows(base["classification_report"])
    new_rows = class_rows(reranked["classification_report"])
    base_slices = slice_rows(base_rows)
    new_slices = slice_rows(new_rows)
    return {
        "base_slices": base_slices,
        "reranked_slices": new_slices,
        "delta_slices": {
            name: {
                "delta_f1": float(new_slices.get(name, {}).get("f1", 0.0)) - float(value.get("f1", 0.0)),
                "delta_recall": float(new_slices.get(name, {}).get("recall", 0.0)) - float(value.get("recall", 0.0)),
            }
            for name, value in base_slices.items()
        },
    }


def class_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, item in report.items():
        if not isinstance(item, dict) or "f1-score" not in item:
            continue
        factors = _factor_text(label)
        rows.append(
            {
                "class_label": label,
                "friction": factors.get("friction", label),
                "material": factors.get("material"),
                "roughness": factors.get("unevenness"),
                "precision": float(item.get("precision") or 0.0),
                "recall": float(item.get("recall") or 0.0),
                "f1": float(item.get("f1-score") or 0.0),
                "support": int(item.get("support") or 0),
            }
        )
    return rows


def slice_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"friction:{row['friction']}"].append(row)
        if row["material"]:
            groups[f"material:{row['material']}"].append(row)
        if row["roughness"]:
            groups[f"roughness:{row['roughness']}"].append(row)
        if row["friction"] in {"wet", "water", "fresh_snow", "melted_snow", "ice"}:
            groups["safety:low_friction_visual"].append(row)
        if row["friction"] in {"wet", "water"}:
            groups["safety:wet_water"].append(row)
        if row["friction"] in {"fresh_snow", "melted_snow", "ice"}:
            groups["safety:winter"].append(row)
    return {name: aggregate(items) for name, items in sorted(groups.items())}


def aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
    support = sum(int(x["support"]) for x in items)
    if support <= 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    return {
        "precision": sum(x["precision"] * x["support"] for x in items) / support,
        "recall": sum(x["recall"] * x["support"] for x in items) / support,
        "f1": sum(x["f1"] * x["support"] for x in items) / support,
        "support": support,
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def pct(value: float, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def to_markdown(payload: dict[str, Any]) -> str:
    base = payload["base"]["summary"]
    new = payload["reranked"]["summary"]
    deltas = payload["slice_comparison"]["delta_slices"]
    lines = [
        "# RSCD Hard-Pair Reranker Result",
        "",
        payload["protocol"]["claim_boundary"],
        "",
        "## Summary",
        "",
        "| method | Top-1 | Mean-P | Mean-R | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| base PhysicsTexture | {pct(base['top1'])} | {pct(base['mean_precision'])} | {pct(base['mean_recall'])} | {pct(base['macro_f1'])} | {pct(base['weighted_f1'])} |",
        f"| hard-pair reranked | {pct(new['top1'])} | {pct(new['mean_precision'])} | {pct(new['mean_recall'])} | {pct(new['macro_f1'])} | {pct(new['weighted_f1'])} |",
        f"| delta | {pct(new['top1'] - base['top1'], True)} | {pct(new['mean_precision'] - base['mean_precision'], True)} | {pct(new['mean_recall'] - base['mean_recall'], True)} | {pct(new['macro_f1'] - base['macro_f1'], True)} | {pct(new['weighted_f1'] - base['weighted_f1'], True)} |",
        "",
        "## Safety Slices",
        "",
        "| slice | delta F1 | delta recall |",
        "|---|---:|---:|",
    ]
    for name in ["safety:wet_water", "friction:water", "safety:low_friction_visual", "roughness:slight", "roughness:severe", "material:concrete"]:
        item = deltas.get(name, {})
        lines.append(f"| `{name}` | {pct(float(item.get('delta_f1', 0.0)), True)} | {pct(float(item.get('delta_recall', 0.0)), True)} |")
    lines.extend(
        [
            "",
            "## Reranker Coverage",
            "",
            f"- hard-pair candidates: {payload['protocol']['num_hard_pairs']}",
            f"- trained validation pair rerankers: {payload['protocol']['num_rerankers']}",
            f"- changed test predictions: {payload['protocol']['applied_test_samples']}",
            "",
            "Decision rule: promote only if Top-1/Macro-F1 improve and wet/water or water slices do not regress.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

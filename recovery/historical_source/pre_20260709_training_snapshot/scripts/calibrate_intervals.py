from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.engine import build_model, dataloader_worker_settings, move_batch
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--min-group-calibration", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="Override config device, e.g. cpu for smoke checks.")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    data_cfg = cfg.setdefault("data", {})
    if args.max_val_samples is not None:
        data_cfg["max_val_samples"] = int(args.max_val_samples)
    if args.max_test_samples is not None:
        data_cfg["max_test_samples"] = int(args.max_test_samples)
    if args.num_workers is not None:
        data_cfg["num_workers"] = int(args.num_workers)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    calib = _collect(model, cfg, device, "val")
    test = _collect(model, cfg, device, "test")
    q = _conformal_radius(calib["scores"], args.target_coverage)

    result = {
        "checkpoint": str(args.checkpoint),
        "target_coverage": args.target_coverage,
        "calibration_split": _summarize(calib, q),
        "test_split": _summarize(test, q),
        "conformal_radius": q,
        "dataset_conditional_test": _summarize_group_conditional(
            calib, test, "dataset", args.target_coverage, args.min_group_calibration
        ),
        "dataset_core_conditional_test": _summarize_group_conditional(
            calib, test, "group_key", args.target_coverage, args.min_group_calibration
        ),
        "risk_conditional_test": _summarize_group_conditional(
            calib, test, "risk", args.target_coverage, args.min_group_calibration
        ),
        "hierarchical_conditional_test": _summarize_hierarchical_conditional(
            calib, test, args.target_coverage, args.min_group_calibration
        ),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


@torch.no_grad()
def _collect(model: torch.nn.Module, cfg: dict[str, Any], device: torch.device, split: str) -> dict[str, np.ndarray]:
    data_cfg = cfg["data"]
    manifests = data_cfg[f"{split}_manifests"]
    ds = ManifestDataset(
        manifests,
        transform=build_transforms(
            int(data_cfg.get("image_size", 224)),
            train=False,
            aug_cfg=data_cfg.get("augmentation"),
        ),
        max_samples=data_cfg.get(f"max_{split}_samples"),
        max_samples_per_dataset=data_cfg.get(f"max_{split}_samples_per_dataset"),
        max_samples_per_class=data_cfg.get(f"max_{split}_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)) + (1 if split == "val" else 2),
    )
    num_workers, loader_kwargs = dataloader_worker_settings(data_cfg)
    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )

    pred_parts = []
    target_parts = []
    mean_parts = []
    dataset_parts = []
    group_parts = []
    risk_parts = []
    progress_every = int(data_cfg.get("eval_progress_every", 250))
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        out = model(batch["image"], domain_idx=batch.get("domain_idx"))
        mask = batch["mu_mask"]
        if mask.any():
            pred_parts.append(out["mu_interval"][mask].detach().cpu())
            target_parts.append(batch["mu_interval"][mask].detach().cpu())
            mean_parts.append(out["mu_mean"][mask].detach().cpu())
            mask_cpu = mask.detach().cpu().numpy().astype(bool)
            dataset_parts.extend([name for name, keep in zip(batch["dataset"], mask_cpu) if keep])
            group_parts.extend([name for name, keep in zip(batch["group_key"], mask_cpu) if keep])
            risk_labels = batch["labels"]["risk"].detach().cpu().numpy()
            risk_parts.extend([str(int(label)) for label, keep in zip(risk_labels, mask_cpu) if keep])
        if progress_every > 0 and (
            batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches
        ):
            print(
                f"calibrate {split}: {batch_idx}/{total_batches} batches",
                file=sys.stderr,
                flush=True,
            )

    pred = torch.cat(pred_parts, dim=0).numpy()
    target = torch.cat(target_parts, dim=0).numpy()
    mean = torch.cat(mean_parts, dim=0).numpy()
    scores = np.maximum.reduce(
        [
            pred[:, 0] - target[:, 0],
            target[:, 1] - pred[:, 1],
            np.zeros(len(target), dtype=np.float32),
        ]
    )
    return {
        "pred": pred,
        "target": target,
        "mean": mean,
        "scores": scores,
        "dataset": np.asarray(dataset_parts),
        "group_key": np.asarray(group_parts),
        "risk": np.asarray(risk_parts),
    }


def _conformal_radius(scores: np.ndarray, target_coverage: float) -> float:
    if len(scores) == 0:
        return 0.0
    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
    sorted_scores = np.sort(scores)
    idx = min(math.ceil((len(sorted_scores) + 1) * target_coverage) - 1, len(sorted_scores) - 1)
    idx = max(idx, 0)
    return float(sorted_scores[idx])


def _summarize(items: dict[str, np.ndarray], radius: float) -> dict[str, float | int]:
    pred = items["pred"]
    target = items["target"]
    mean = items["mean"]
    raw_covers = (pred[:, 0] <= target[:, 0]) & (pred[:, 1] >= target[:, 1])
    calibrated = np.stack(
        [
            np.clip(pred[:, 0] - radius, 0.0, 1.2),
            np.clip(pred[:, 1] + radius, 0.0, 1.2),
        ],
        axis=1,
    )
    calibrated_covers = (calibrated[:, 0] <= target[:, 0]) & (calibrated[:, 1] >= target[:, 1])
    return {
        "num_samples": int(len(target)),
        "raw_coverage": float(raw_covers.mean()),
        "raw_width": float((pred[:, 1] - pred[:, 0]).mean()),
        "calibrated_coverage": float(calibrated_covers.mean()),
        "calibrated_width": float((calibrated[:, 1] - calibrated[:, 0]).mean()),
        "mean_mae_to_interval_mid": float(np.abs(mean - target.mean(axis=1)).mean()),
    }


def _summarize_group_conditional(
    calib: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    group_key: str,
    target_coverage: float,
    min_calibration_samples: int = 50,
) -> dict[str, dict[str, float | int]]:
    groups = sorted(set(calib[group_key].tolist()) | set(test[group_key].tolist()))
    out: dict[str, dict[str, float | int]] = {}
    calibrated_parts = []
    target_parts = []
    mean_parts = []
    global_radius = _conformal_radius(calib["scores"], target_coverage)
    for group in groups:
        calib_mask = calib[group_key] == group
        test_mask = test[group_key] == group
        if not test_mask.any():
            continue
        use_group_radius = int(calib_mask.sum()) >= int(min_calibration_samples)
        radius = (
            _conformal_radius(calib["scores"][calib_mask], target_coverage)
            if use_group_radius
            else global_radius
        )
        group_items = {
            "pred": test["pred"][test_mask],
            "target": test["target"][test_mask],
            "mean": test["mean"][test_mask],
        }
        summary = _summarize(group_items, radius)
        summary["conformal_radius"] = float(radius)
        summary["calibration_samples"] = int(calib_mask.sum())
        summary["used_group_radius"] = bool(use_group_radius)
        out[str(group)] = summary
        pred = test["pred"][test_mask]
        calibrated_parts.append(
            np.stack(
                [
                    np.clip(pred[:, 0] - radius, 0.0, 1.2),
                    np.clip(pred[:, 1] + radius, 0.0, 1.2),
                ],
                axis=1,
            )
        )
        target_parts.append(test["target"][test_mask])
        mean_parts.append(test["mean"][test_mask])
    if calibrated_parts:
        calibrated = np.concatenate(calibrated_parts, axis=0)
        target = np.concatenate(target_parts, axis=0)
        mean = np.concatenate(mean_parts, axis=0)
        covers = (calibrated[:, 0] <= target[:, 0]) & (calibrated[:, 1] >= target[:, 1])
        out["_pooled"] = {
            "num_samples": int(len(target)),
            "calibrated_coverage": float(covers.mean()),
            "calibrated_width": float((calibrated[:, 1] - calibrated[:, 0]).mean()),
            "mean_mae_to_interval_mid": float(np.abs(mean - target.mean(axis=1)).mean()),
        }
    return out


def _summarize_hierarchical_conditional(
    calib: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    target_coverage: float,
    min_calibration_samples: int = 50,
) -> dict[str, Any]:
    """Conservative per-sample calibration across nested road-state groups.

    The separate group-conditional summaries above can under-cover a small or
    visually ambiguous subgroup if its learned radius is smaller than a broader
    parent radius. For safety reporting we therefore use the maximum available
    radius among global, dataset, dataset::state, risk, and dataset::state+risk
    groups. This is deliberately conservative and must be judged by both
    coverage and width.
    """

    n = len(test["target"])
    if n == 0:
        return {}
    global_radius = _conformal_radius(calib["scores"], target_coverage)
    dataset_radii = _group_radii(calib, "dataset", target_coverage, min_calibration_samples)
    core_radii = _group_radii(calib, "group_key", target_coverage, min_calibration_samples)
    risk_radii = _group_radii(calib, "risk", target_coverage, min_calibration_samples)
    calib_core_risk = _combine_keys(calib["group_key"], calib["risk"])
    test_core_risk = _combine_keys(test["group_key"], test["risk"])
    core_risk_radii = _group_radii_from_keys(
        calib["scores"], calib_core_risk, target_coverage, min_calibration_samples
    )

    radii = np.full(n, global_radius, dtype=np.float32)
    sources: list[str] = []
    for idx in range(n):
        candidates: list[tuple[str, float]] = [("global", global_radius)]
        dataset = str(test["dataset"][idx])
        core = str(test["group_key"][idx])
        risk = str(test["risk"][idx])
        core_risk = str(test_core_risk[idx])
        if dataset in dataset_radii:
            candidates.append(("dataset", dataset_radii[dataset]))
        if core in core_radii:
            candidates.append(("dataset_core", core_radii[core]))
        if risk in risk_radii:
            candidates.append(("risk", risk_radii[risk]))
        if core_risk in core_risk_radii:
            candidates.append(("dataset_core_risk", core_risk_radii[core_risk]))
        source, radius = max(candidates, key=lambda item: item[1])
        radii[idx] = float(radius)
        sources.append(source)

    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "policy": {
            "target_coverage": float(target_coverage),
            "min_calibration_samples": int(min_calibration_samples),
            "radius_rule": (
                "per sample, use the maximum available conformal radius among "
                "global, dataset, dataset::state, risk, and dataset::state+risk"
            ),
            "claim_boundary": (
                "This is a conservative safety calibration policy. It can improve conditional "
                "coverage, but paper claims must report the resulting width."
            ),
        },
        "global_radius": float(global_radius),
        "radius_source_counts": source_counts,
        "pooled": _summarize_with_radii(test, radii),
        "dataset": _summarize_scope_with_radii(test, radii, "dataset"),
        "dataset_core": _summarize_scope_with_radii(test, radii, "group_key"),
        "risk": _summarize_scope_with_radii(test, radii, "risk"),
        "dataset_core_risk": _summarize_scope_with_radii(
            {**test, "dataset_core_risk": test_core_risk}, radii, "dataset_core_risk"
        ),
        "radius_tables": {
            "dataset": dataset_radii,
            "dataset_core": core_radii,
            "risk": risk_radii,
            "dataset_core_risk": core_risk_radii,
        },
    }


def _group_radii(
    calib: dict[str, np.ndarray],
    group_key: str,
    target_coverage: float,
    min_calibration_samples: int,
) -> dict[str, float]:
    return _group_radii_from_keys(
        calib["scores"], calib[group_key], target_coverage, min_calibration_samples
    )


def _group_radii_from_keys(
    scores: np.ndarray,
    keys: np.ndarray,
    target_coverage: float,
    min_calibration_samples: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in sorted(set(keys.tolist())):
        mask = keys == group
        if int(mask.sum()) < int(min_calibration_samples):
            continue
        out[str(group)] = _conformal_radius(scores[mask], target_coverage)
    return out


def _combine_keys(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray([f"{a}::risk={b}" for a, b in zip(left.tolist(), right.tolist())])


def _summarize_with_radii(items: dict[str, np.ndarray], radii: np.ndarray) -> dict[str, float | int]:
    pred = items["pred"]
    target = items["target"]
    mean = items["mean"]
    raw_covers = (pred[:, 0] <= target[:, 0]) & (pred[:, 1] >= target[:, 1])
    calibrated = np.stack(
        [
            np.clip(pred[:, 0] - radii, 0.0, 1.2),
            np.clip(pred[:, 1] + radii, 0.0, 1.2),
        ],
        axis=1,
    )
    calibrated_covers = (calibrated[:, 0] <= target[:, 0]) & (calibrated[:, 1] >= target[:, 1])
    return {
        "num_samples": int(len(target)),
        "raw_coverage": float(raw_covers.mean()),
        "raw_width": float((pred[:, 1] - pred[:, 0]).mean()),
        "calibrated_coverage": float(calibrated_covers.mean()),
        "calibrated_width": float((calibrated[:, 1] - calibrated[:, 0]).mean()),
        "mean_radius": float(np.mean(radii)),
        "max_radius": float(np.max(radii)),
        "mean_mae_to_interval_mid": float(np.abs(mean - target.mean(axis=1)).mean()),
    }


def _summarize_scope_with_radii(
    items: dict[str, np.ndarray], radii: np.ndarray, group_key: str
) -> dict[str, dict[str, float | int]]:
    groups = sorted(set(items[group_key].tolist()))
    out: dict[str, dict[str, float | int]] = {}
    calibrated_parts = []
    target_parts = []
    mean_parts = []
    radius_parts = []
    for group in groups:
        mask = items[group_key] == group
        if not mask.any():
            continue
        group_items = {
            "pred": items["pred"][mask],
            "target": items["target"][mask],
            "mean": items["mean"][mask],
        }
        group_radii = radii[mask]
        out[str(group)] = _summarize_with_radii(group_items, group_radii)
        pred = group_items["pred"]
        calibrated_parts.append(
            np.stack(
                [
                    np.clip(pred[:, 0] - group_radii, 0.0, 1.2),
                    np.clip(pred[:, 1] + group_radii, 0.0, 1.2),
                ],
                axis=1,
            )
        )
        target_parts.append(group_items["target"])
        mean_parts.append(group_items["mean"])
        radius_parts.append(group_radii)
    if calibrated_parts:
        calibrated = np.concatenate(calibrated_parts, axis=0)
        target = np.concatenate(target_parts, axis=0)
        mean = np.concatenate(mean_parts, axis=0)
        all_radii = np.concatenate(radius_parts, axis=0)
        covers = (calibrated[:, 0] <= target[:, 0]) & (calibrated[:, 1] >= target[:, 1])
        out["_pooled"] = {
            "num_samples": int(len(target)),
            "calibrated_coverage": float(covers.mean()),
            "calibrated_width": float((calibrated[:, 1] - calibrated[:, 0]).mean()),
            "mean_radius": float(np.mean(all_radii)),
            "max_radius": float(np.max(all_radii)),
            "mean_mae_to_interval_mid": float(np.abs(mean - target.mean(axis=1)).mean()),
        }
    return out


if __name__ == "__main__":
    main()

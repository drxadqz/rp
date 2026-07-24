from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.engine import build_model, dataloader_worker_settings, move_batch
from friction_affordance.ontology import RISK, TASKS
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml, resolve_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = _build_loader(cfg, args.split)
    collected = _collect(model, loader, device, progress_label=args.split)
    result = _summarize(collected, args.split)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


def _build_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    data_cfg = cfg["data"]
    manifests = data_cfg.get(f"{split}_manifests", data_cfg["val_manifests"])
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
    return DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )


def _collect(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    progress_label: str | None = None,
    progress_every: int = 250,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tasks": {task: defaultdict(list) for task in TASKS},
        "mu": defaultdict(list),
        "evidence": {"risk": defaultdict(list), "mu": defaultdict(list)},
    }
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        moved = move_batch(batch, device)
        pred = model(moved["image"], domain_idx=moved.get("domain_idx"))
        batch_size = moved["image"].size(0)
        dataset = np.asarray(batch["dataset"])
        group_key = np.asarray(batch["group_key"])
        evidence = pred.get("evidence_field")

        for task in TASKS:
            mask = moved["masks"][task].detach().cpu().numpy().astype(bool)
            if not mask.any():
                continue
            y_true = moved["labels"][task].detach().cpu().numpy()[mask]
            y_pred = pred["logits"][task].argmax(dim=1).detach().cpu().numpy()[mask]
            item = out["tasks"][task]
            item["y_true"].extend(y_true.tolist())
            item["y_pred"].extend(y_pred.tolist())
            item["dataset"].extend(dataset[mask].tolist())
            item["group_key"].extend(group_key[mask].tolist())

        mu_mask = moved["mu_mask"].detach().cpu().numpy().astype(bool)
        if mu_mask.any():
            pred_int = pred["mu_interval"].detach().cpu().numpy()[mu_mask]
            target_int = moved["mu_interval"].detach().cpu().numpy()[mu_mask]
            mu_mean = pred["mu_mean"].detach().cpu().numpy()[mu_mask]
            risk = moved["labels"]["risk"].detach().cpu().numpy()[mu_mask]
            mu = out["mu"]
            mu["pred_low"].extend(pred_int[:, 0].tolist())
            mu["pred_high"].extend(pred_int[:, 1].tolist())
            mu["target_low"].extend(target_int[:, 0].tolist())
            mu["target_high"].extend(target_int[:, 1].tolist())
            mu["mu_mean"].extend(mu_mean.tolist())
            mu["dataset"].extend(dataset[mu_mask].tolist())
            mu["group_key"].extend(group_key[mu_mask].tolist())
            mu["risk"].extend(risk.tolist())

            if evidence:
                ev_pred_int = evidence["mu_interval"].detach().cpu().numpy()[mu_mask]
                ev_mu_mean = evidence["mu_mean"].detach().cpu().numpy()[mu_mask]
                ev_mu = out["evidence"]["mu"]
                ev_mu["pred_low"].extend(ev_pred_int[:, 0].tolist())
                ev_mu["pred_high"].extend(ev_pred_int[:, 1].tolist())
                ev_mu["target_low"].extend(target_int[:, 0].tolist())
                ev_mu["target_high"].extend(target_int[:, 1].tolist())
                ev_mu["mu_mean"].extend(ev_mu_mean.tolist())
                ev_mu["dataset"].extend(dataset[mu_mask].tolist())
                ev_mu["group_key"].extend(group_key[mu_mask].tolist())
                ev_mu["risk"].extend(risk.tolist())

        if evidence and moved["masks"]["risk"].any():
            mask = moved["masks"]["risk"].detach().cpu().numpy().astype(bool)
            ev_risk = out["evidence"]["risk"]
            ev_risk["y_true"].extend(moved["labels"]["risk"].detach().cpu().numpy()[mask].tolist())
            ev_risk["y_pred"].extend(evidence["risk_logits"].argmax(dim=1).detach().cpu().numpy()[mask].tolist())
            ev_risk["dataset"].extend(dataset[mask].tolist())
            ev_risk["group_key"].extend(group_key[mask].tolist())

        out.setdefault("num_batches", 0)
        out["num_batches"] += 1
        out.setdefault("num_samples_seen", 0)
        out["num_samples_seen"] += batch_size
        if progress_label and progress_every > 0 and (
            batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches
        ):
            print(
                f"collect {progress_label}: {batch_idx}/{total_batches} batches",
                file=sys.stderr,
                flush=True,
            )
    return out


def _summarize(collected: dict[str, Any], split: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": split,
        "num_samples_seen": int(collected.get("num_samples_seen", 0)),
        "num_batches": int(collected.get("num_batches", 0)),
        "tasks": {},
    }
    for task, raw in collected["tasks"].items():
        y_true = np.asarray(raw["y_true"], dtype=int)
        y_pred = np.asarray(raw["y_pred"], dtype=int)
        if len(y_true) == 0:
            continue
        task_summary = _classification_summary(y_true, y_pred, labels=TASKS[task])
        task_summary["by_dataset"] = _grouped_classification_summary(
            y_true, y_pred, np.asarray(raw["dataset"]), labels=TASKS[task]
        )
        result["tasks"][task] = task_summary

    if "risk" in result["tasks"]:
        risk_raw = collected["tasks"]["risk"]
        y_true = np.asarray(risk_raw["y_true"], dtype=int)
        y_pred = np.asarray(risk_raw["y_pred"], dtype=int)
        low_friction_true = y_true >= RISK.index("high")
        low_friction_pred = y_pred >= RISK.index("high")
        num_positive = int(low_friction_true.sum())
        num_pred_positive = int(low_friction_pred.sum())
        result["low_friction_detection"] = {
            "positive_definition": "risk in {high, very_high}",
            "num_positive": num_positive,
            "num_pred_positive": num_pred_positive,
            "applicable": num_positive > 0,
            "recall": _safe_metric(recall_score, low_friction_true, low_friction_pred),
            "precision": _safe_metric(precision_score, low_friction_true, low_friction_pred),
            "f1": _safe_metric(f1_score, low_friction_true, low_friction_pred),
        }

    result["mu_interval"] = _mu_summary(collected["mu"])
    evidence = collected.get("evidence", {})
    if evidence and evidence.get("risk", {}).get("y_true"):
        ev_y_true = np.asarray(evidence["risk"]["y_true"], dtype=int)
        ev_y_pred = np.asarray(evidence["risk"]["y_pred"], dtype=int)
        result["evidence_field"] = {
            "risk": {
                **_classification_summary(ev_y_true, ev_y_pred, labels=RISK),
                "by_dataset": _grouped_classification_summary(
                    ev_y_true,
                    ev_y_pred,
                    np.asarray(evidence["risk"]["dataset"]),
                    labels=RISK,
                ),
            },
            "mu_interval": _mu_summary(evidence["mu"]),
        }
    return result


def _classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    label_indices = list(range(len(labels))) if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    supported_label_indices = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not supported_label_indices:
        supported_label_indices = label_indices
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=label_indices,
        average=None,
        zero_division=0,
    )
    class_names = labels or [str(i) for i in label_indices]
    supported_names = [
        str(class_names[idx]) if idx < len(class_names) else str(idx)
        for idx in supported_label_indices
    ]
    return {
        "num_samples": int(len(y_true)),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "macro_f1": _safe_metric(
            f1_score,
            y_true,
            y_pred,
            labels=supported_label_indices,
            average="macro",
        ),
        "macro_f1_label_policy": "labels present in y_true or y_pred for this evaluation scope",
        "macro_f1_supported_labels": supported_names,
        "per_class_f1": {
            str(name): float(score)
            for name, score in zip(class_names, per_class_f1.tolist())
        },
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=label_indices,
        ).astype(int).tolist(),
        "confusion_matrix_labels": [str(name) for name in class_names],
    }


def _grouped_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    out = {}
    for group in sorted(set(groups.tolist())):
        keep = groups == group
        if keep.any():
            out[str(group)] = _classification_summary(y_true[keep], y_pred[keep], labels=labels)
    if out:
        out["_worst_macro_f1"] = {
            "value": float(min(v["macro_f1"] for k, v in out.items() if not k.startswith("_")))
        }
    return out


def _mu_summary(raw: dict[str, list]) -> dict[str, Any]:
    if not raw["pred_low"]:
        return {}
    pred_low = np.asarray(raw["pred_low"], dtype=float)
    pred_high = np.asarray(raw["pred_high"], dtype=float)
    target_low = np.asarray(raw["target_low"], dtype=float)
    target_high = np.asarray(raw["target_high"], dtype=float)
    mean = np.asarray(raw["mu_mean"], dtype=float)
    dataset = np.asarray(raw["dataset"])
    risk = np.asarray(raw["risk"], dtype=int)

    base = _mu_arrays_summary(pred_low, pred_high, target_low, target_high, mean)
    base["by_dataset"] = {
        str(group): _mu_arrays_summary(
            pred_low[dataset == group],
            pred_high[dataset == group],
            target_low[dataset == group],
            target_high[dataset == group],
            mean[dataset == group],
        )
        for group in sorted(set(dataset.tolist()))
    }
    base["by_risk"] = {
        str(group): _mu_arrays_summary(
            pred_low[risk == group],
            pred_high[risk == group],
            target_low[risk == group],
            target_high[risk == group],
            mean[risk == group],
        )
        for group in sorted(set(risk.tolist()))
    }
    return base


def _mu_arrays_summary(
    pred_low: np.ndarray,
    pred_high: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
    mean: np.ndarray,
) -> dict[str, float | int]:
    if len(pred_low) == 0:
        return {"num_samples": 0}
    covers = (pred_low <= target_low) & (pred_high >= target_high)
    width = pred_high - pred_low
    target_mid = 0.5 * (target_low + target_high)
    return {
        "num_samples": int(len(pred_low)),
        "coverage": float(covers.mean()),
        "width_mean": float(width.mean()),
        "width_median": float(np.median(width)),
        "mean_mae_to_interval_mid": float(np.abs(mean - target_mid).mean()),
    }


def _safe_metric(fn, y_true, y_pred, **kwargs) -> float:
    try:
        return float(fn(y_true, y_pred, zero_division=0, **kwargs))
    except TypeError:
        return float(fn(y_true, y_pred, **kwargs))


if __name__ == "__main__":
    main()

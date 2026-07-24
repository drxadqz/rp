from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.engine import build_model, dataloader_worker_settings, move_batch
from friction_affordance.ontology import RISK, TASKS
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml, resolve_device


DEFAULT_QUALITY_CSV = Path("data/quality_flags/image_quality_flags.csv")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-sample predictions and evaluate performance slices by image quality flags."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        action="append",
        default=[],
        help="Override the checkpoint config manifests for this evaluation split.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-predictions", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or args.checkpoint.parent / "quality_slices"
    out_predictions = args.out_predictions or out_dir / f"predictions_{args.split}.csv"
    out_json = args.out_json or out_dir / f"quality_slices_{args.split}.json"
    out_md = args.out_md or out_dir / f"quality_slices_{args.split}.md"

    cfg = load_yaml(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    if args.eval_manifest:
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["data"][f"{args.split}_manifests"] = [str(path) for path in args.eval_manifest]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = build_loader(
        cfg,
        args.split,
        max_samples=int(args.max_samples),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    predictions = collect_predictions(model, loader, device)
    predictions = attach_quality_flags(predictions, args.quality_csv)
    radius = load_conformal_radius(args.calibration_json or args.checkpoint.parent / "interval_calibration_90.json")
    report = build_report(
        predictions,
        split=args.split,
        checkpoint=args.checkpoint,
        config=args.config,
        quality_csv=args.quality_csv,
        conformal_radius=radius,
    )

    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_predictions, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report, out_predictions), encoding="utf-8")
    print(render_markdown(report, out_predictions))
    print(f"wrote: {out_predictions}")


def build_loader(
    cfg: dict[str, Any],
    split: str,
    *,
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    data_cfg = dict(cfg["data"])
    if num_workers >= 0:
        data_cfg["num_workers"] = num_workers
    manifests = data_cfg.get(f"{split}_manifests", data_cfg["val_manifests"])
    ds = ManifestDataset(
        manifests,
        transform=build_transforms(
            int(data_cfg.get("image_size", 224)),
            train=False,
            aug_cfg=data_cfg.get("augmentation"),
        ),
        max_samples=max_samples if max_samples > 0 else data_cfg.get(f"max_{split}_samples"),
        max_samples_per_dataset=data_cfg.get(f"max_{split}_samples_per_dataset"),
        max_samples_per_class=data_cfg.get(f"max_{split}_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)) + (1 if split == "val" else 2),
    )
    num_workers_final, loader_kwargs = dataloader_worker_settings(data_cfg)
    return DataLoader(
        ds,
        batch_size=batch_size if batch_size > 0 else int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers_final,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )


def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for batch in loader:
        moved = move_batch(batch, device)
        out = model(moved["image"], grl_lambda=0.0, domain_idx=moved.get("domain_idx"))
        batch_size = int(moved["image"].size(0))
        pred_interval = out["mu_interval"].detach().cpu().numpy()
        mu_mean = out["mu_mean"].detach().cpu().numpy()
        target_interval = moved["mu_interval"].detach().cpu().numpy()
        mu_mask = moved["mu_mask"].detach().cpu().numpy().astype(bool)

        task_pred: dict[str, np.ndarray] = {}
        task_conf: dict[str, np.ndarray] = {}
        task_entropy: dict[str, np.ndarray] = {}
        for task in TASKS:
            logits = out["logits"][task]
            prob = torch.softmax(logits, dim=1)
            task_pred[task] = prob.argmax(dim=1).detach().cpu().numpy()
            task_conf[task] = prob.max(dim=1).values.detach().cpu().numpy()
            task_entropy[task] = (-(prob * prob.clamp_min(1e-8).log()).sum(dim=1)).detach().cpu().numpy()

        label_np = {
            task: moved["labels"][task].detach().cpu().numpy()
            for task in TASKS
        }
        mask_np = {
            task: moved["masks"][task].detach().cpu().numpy().astype(bool)
            for task in TASKS
        }

        for i in range(batch_size):
            rec: dict[str, Any] = {
                "image_path": str(batch["image_path"][i]),
                "image_path_norm": _norm_path(batch["image_path"][i]),
                "dataset": str(batch["dataset"][i]),
                "domain_id": str(batch["domain_id"][i]),
                "group_key": str(batch["group_key"][i]),
                "mu_known": bool(mu_mask[i]),
                "target_mu_low": float(target_interval[i, 0]) if mu_mask[i] else math.nan,
                "target_mu_high": float(target_interval[i, 1]) if mu_mask[i] else math.nan,
                "pred_mu_low": float(pred_interval[i, 0]),
                "pred_mu_high": float(pred_interval[i, 1]),
                "pred_mu_mean": float(mu_mean[i]),
                "raw_interval_width": float(pred_interval[i, 1] - pred_interval[i, 0]),
            }
            if mu_mask[i]:
                rec["raw_interval_covers"] = bool(
                    pred_interval[i, 0] <= target_interval[i, 0]
                    and pred_interval[i, 1] >= target_interval[i, 1]
                )
            else:
                rec["raw_interval_covers"] = None

            for task in TASKS:
                true_idx = int(label_np[task][i])
                pred_idx = int(task_pred[task][i])
                known = bool(mask_np[task][i])
                rec[f"true_{task}_idx"] = true_idx if known else -1
                rec[f"pred_{task}_idx"] = pred_idx
                rec[f"true_{task}"] = TASKS[task][true_idx] if known and 0 <= true_idx < len(TASKS[task]) else ""
                rec[f"pred_{task}"] = TASKS[task][pred_idx] if 0 <= pred_idx < len(TASKS[task]) else str(pred_idx)
                rec[f"{task}_known"] = known
                rec[f"{task}_correct"] = bool(known and true_idx == pred_idx)
                rec[f"{task}_confidence"] = float(task_conf[task][i])
                rec[f"{task}_entropy"] = float(task_entropy[task][i])
            high_idx = RISK.index("high")
            rec["low_friction_true"] = bool(rec["risk_known"] and int(rec["true_risk_idx"]) >= high_idx)
            rec["low_friction_pred"] = bool(int(rec["pred_risk_idx"]) >= high_idx)
            records.append(rec)
    return pd.DataFrame(records)


def attach_quality_flags(predictions: pd.DataFrame, quality_csv: Path) -> pd.DataFrame:
    if not quality_csv.exists():
        predictions["quality_joined"] = False
        return predictions
    quality = pd.read_csv(quality_csv, low_memory=False)
    quality = quality.copy()
    quality["image_path_norm"] = quality["image_path"].astype(str).map(_norm_path)
    keep_cols = [
        "image_path_norm",
        "decode_ok",
        "width",
        "height",
        "aspect",
        "brightness",
        "contrast",
        "saturation",
        "white_pixel_frac",
        "black_pixel_frac",
        "specular_highlight_frac",
        "edge_strength",
        "texture_energy",
        "near_white_score",
        "near_white_flag",
        "overexposed_flag",
        "low_contrast_flag",
        "low_texture_flag",
        "dark_flag",
        "suspicious_quality_flag",
    ]
    available = [col for col in keep_cols if col in quality.columns]
    joined = predictions.merge(
        quality[available].drop_duplicates("image_path_norm"),
        how="left",
        on="image_path_norm",
        suffixes=("", "_quality"),
    )
    joined["quality_joined"] = joined["decode_ok"].notna()
    for col in [
        "near_white_flag",
        "overexposed_flag",
        "low_contrast_flag",
        "low_texture_flag",
        "dark_flag",
        "suspicious_quality_flag",
    ]:
        if col in joined.columns:
            joined[col] = joined[col].map(_to_bool).fillna(False)
    return joined


def load_conformal_radius(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("conformal_radius", 0.0))
    except (OSError, ValueError, TypeError):
        return 0.0


def build_report(
    predictions: pd.DataFrame,
    *,
    split: str,
    checkpoint: Path,
    config: Path,
    quality_csv: Path,
    conformal_radius: float,
) -> dict[str, Any]:
    df = predictions.copy()
    if "raw_interval_covers" in df.columns:
        df["calibrated_interval_covers"] = (
            (df["pred_mu_low"].astype(float) - conformal_radius <= df["target_mu_low"].astype(float))
            & (df["pred_mu_high"].astype(float) + conformal_radius >= df["target_mu_high"].astype(float))
            & df["mu_known"].map(_to_bool)
        )
        df["calibrated_interval_width"] = df["raw_interval_width"].astype(float) + 2.0 * conformal_radius

    slices = {
        "all": df,
        "quality_joined": df[df.get("quality_joined", False).map(_to_bool)] if "quality_joined" in df else df.iloc[0:0],
        "normal_quality": _slice_flag(df, "suspicious_quality_flag", False),
        "suspicious_quality": _slice_flag(df, "suspicious_quality_flag", True),
        "not_near_white": _slice_flag(df, "near_white_flag", False),
        "near_white": _slice_flag(df, "near_white_flag", True),
        "overexposed": _slice_flag(df, "overexposed_flag", True),
        "low_contrast": _slice_flag(df, "low_contrast_flag", True),
        "low_texture": _slice_flag(df, "low_texture_flag", True),
    }
    for dataset in sorted(df["dataset"].dropna().astype(str).unique()):
        ddf = df[df["dataset"].astype(str) == dataset]
        slices[f"dataset::{dataset}"] = ddf
        slices[f"dataset::{dataset}::normal_quality"] = _slice_flag(ddf, "suspicious_quality_flag", False)
        slices[f"dataset::{dataset}::near_white"] = _slice_flag(ddf, "near_white_flag", True)
        if dataset == "roadsaw":
            for label in sorted(ddf["true_wetness"].dropna().astype(str).unique()):
                if label:
                    slices[f"roadsaw::wetness::{label}"] = ddf[ddf["true_wetness"].astype(str) == label]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": split,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "quality_csv": str(quality_csv),
        "conformal_radius": float(conformal_radius),
        "num_predictions": int(len(df)),
        "quality_join_rate": float(df["quality_joined"].mean()) if "quality_joined" in df and len(df) else 0.0,
        "slices": {name: summarize_slice(rows, conformal_radius=conformal_radius) for name, rows in slices.items()},
    }


def summarize_slice(rows: pd.DataFrame, *, conformal_radius: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "num_samples": int(len(rows)),
    }
    if len(rows) == 0:
        return out
    out["quality"] = {
        "near_white_rate": _mean_bool(rows, "near_white_flag"),
        "suspicious_quality_rate": _mean_bool(rows, "suspicious_quality_flag"),
        "median_brightness": _median(rows, "brightness"),
        "median_contrast": _median(rows, "contrast"),
        "median_texture_energy": _median(rows, "texture_energy"),
    }
    out["classification"] = {
        task: _task_metrics(rows, task)
        for task in ["friction", "risk", "wetness", "snow", "material"]
        if f"true_{task}_idx" in rows.columns
    }
    out["low_friction_detection"] = _low_friction_metrics(rows)
    out["mu_interval"] = _mu_metrics(rows, conformal_radius=conformal_radius)
    return out


def _task_metrics(rows: pd.DataFrame, task: str) -> dict[str, Any]:
    known = rows[f"{task}_known"].map(_to_bool)
    sub = rows[known]
    if sub.empty:
        return {"num_samples": 0}
    y_true = sub[f"true_{task}_idx"].astype(int).to_numpy()
    y_pred = sub[f"pred_{task}_idx"].astype(int).to_numpy()
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "num_samples": int(len(sub)),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "macro_f1": _safe_metric(f1_score, y_true, y_pred, labels=labels, average="macro"),
        "mean_confidence": float(sub[f"{task}_confidence"].astype(float).mean()),
        "mean_entropy": float(sub[f"{task}_entropy"].astype(float).mean()),
    }


def _low_friction_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    if "low_friction_true" not in rows.columns:
        return {"num_samples": 0}
    known = rows["risk_known"].map(_to_bool)
    sub = rows[known]
    if sub.empty:
        return {"num_samples": 0}
    y_true = sub["low_friction_true"].map(_to_bool).to_numpy()
    y_pred = sub["low_friction_pred"].map(_to_bool).to_numpy()
    return {
        "num_samples": int(len(sub)),
        "recall": _safe_metric(recall_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred),
        "f1": _safe_metric(f1_score, y_true, y_pred),
    }


def _mu_metrics(rows: pd.DataFrame, *, conformal_radius: float) -> dict[str, Any]:
    if "mu_known" not in rows.columns:
        return {"num_samples": 0}
    sub = rows[rows["mu_known"].map(_to_bool)]
    if sub.empty:
        return {"num_samples": 0}
    raw_covers = sub["raw_interval_covers"].map(_to_bool).to_numpy()
    calibrated = (
        (sub["pred_mu_low"].astype(float) - conformal_radius <= sub["target_mu_low"].astype(float))
        & (sub["pred_mu_high"].astype(float) + conformal_radius >= sub["target_mu_high"].astype(float))
    ).to_numpy()
    target_mid = 0.5 * (sub["target_mu_low"].astype(float) + sub["target_mu_high"].astype(float))
    return {
        "num_samples": int(len(sub)),
        "raw_coverage": float(raw_covers.mean()),
        "raw_width": float(sub["raw_interval_width"].astype(float).mean()),
        "calibrated_coverage": float(calibrated.mean()),
        "calibrated_width": float((sub["raw_interval_width"].astype(float) + 2.0 * conformal_radius).mean()),
        "mean_mae_to_interval_mid": float(np.abs(sub["pred_mu_mean"].astype(float) - target_mid).mean()),
    }


def render_markdown(report: dict[str, Any], predictions_csv: Path) -> str:
    lines = [
        "# Quality Slice Evaluation",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Split: `{report['split']}`",
        f"- Predictions CSV: `{predictions_csv}`",
        f"- Quality join rate: `{100.0 * report['quality_join_rate']:.2f}%`",
        f"- Global conformal radius: `{report['conformal_radius']:.6f}`",
        "",
        "## Main Slices",
        "",
        "| slice | n | friction F1 | risk F1 | low recall | raw cov | cal cov | near-white | suspicious |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    preferred = [
        "all",
        "normal_quality",
        "suspicious_quality",
        "not_near_white",
        "near_white",
        "dataset::roadsaw",
        "dataset::roadsaw::normal_quality",
        "dataset::roadsaw::near_white",
        "dataset::rscd",
        "dataset::roadsc",
    ]
    for name in preferred:
        if name in report["slices"]:
            lines.append(_slice_row(name, report["slices"][name]))
    lines += ["", "## RoadSaW Wetness Slices", ""]
    lines += ["| slice | n | friction F1 | risk F1 | low recall | raw cov | cal cov |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(key for key in report["slices"] if key.startswith("roadsaw::wetness::")):
        row = report["slices"][name]
        lines.append(
            "| {name} | {n} | {friction} | {risk} | {low} | {raw} | {cal} |".format(
                name=name,
                n=row.get("num_samples", 0),
                friction=_fmt_metric(row, "classification", "friction", "macro_f1"),
                risk=_fmt_metric(row, "classification", "risk", "macro_f1"),
                low=_fmt_metric(row, "low_friction_detection", "recall"),
                raw=_fmt_metric(row, "mu_interval", "raw_coverage"),
                cal=_fmt_metric(row, "mu_interval", "calibrated_coverage"),
            )
        )
    return "\n".join(lines) + "\n"


def _slice_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {row.get('num_samples', 0)} | "
        f"{_fmt_metric(row, 'classification', 'friction', 'macro_f1')} | "
        f"{_fmt_metric(row, 'classification', 'risk', 'macro_f1')} | "
        f"{_fmt_metric(row, 'low_friction_detection', 'recall')} | "
        f"{_fmt_metric(row, 'mu_interval', 'raw_coverage')} | "
        f"{_fmt_metric(row, 'mu_interval', 'calibrated_coverage')} | "
        f"{_fmt_metric(row, 'quality', 'near_white_rate')} | "
        f"{_fmt_metric(row, 'quality', 'suspicious_quality_rate')} |"
    )


def _fmt_metric(row: dict[str, Any], *keys: str) -> str:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return "-"
        cur = cur[key]
    if cur is None:
        return "-"
    return f"{100.0 * float(cur):.2f}%"


def _slice_flag(df: pd.DataFrame, flag: str, value: bool) -> pd.DataFrame:
    if flag not in df.columns:
        return df.iloc[0:0]
    return df[df[flag].map(_to_bool) == bool(value)]


def _mean_bool(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or df.empty:
        return None
    return float(df[col].map(_to_bool).mean())


def _median(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or df.empty:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _safe_metric(fn, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    try:
        return float(fn(y_true, y_pred, zero_division=0, **kwargs))
    except TypeError:
        return float(fn(y_true, y_pred, **kwargs))


def _norm_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").lower()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


if __name__ == "__main__":
    main()

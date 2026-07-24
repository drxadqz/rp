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

from evaluate_quality_slices import attach_quality_flags, build_loader
from friction_affordance.engine import _weak_style_perturb_normalized, build_model, move_batch
from friction_affordance.ontology import TASKS
from friction_affordance.utils import load_yaml, resolve_device


DEFAULT_QUALITY_CSV = Path("data/quality_flags/image_quality_flags.csv")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast post-hoc probe for TTA-consistency uncertainty calibration. "
            "It borrows test-time augmentation uncertainty from semi-supervised/CV robustness "
            "and tests whether high-consistency-error images need wider weak friction intervals."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--tta-views", type=int, default=2)
    parser.add_argument("--tta-strength", type=float, default=0.08)
    parser.add_argument("--tta-noise-std", type=float, default=0.01)
    parser.add_argument("--min-calibration-samples", type=int, default=50)
    parser.add_argument("--min-slice-samples", type=int, default=30)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--max-test-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-predictions", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    calib = _collect_tta_records(
        model,
        cfg,
        device,
        split="val",
        max_samples=int(args.max_val_samples),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        tta_views=int(args.tta_views),
        tta_strength=float(args.tta_strength),
        tta_noise_std=float(args.tta_noise_std),
        quality_csv=args.quality_csv,
    )
    test = _collect_tta_records(
        model,
        cfg,
        device,
        split="test",
        max_samples=int(args.max_test_samples),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        tta_views=int(args.tta_views),
        tta_strength=float(args.tta_strength),
        tta_noise_std=float(args.tta_noise_std),
        quality_csv=args.quality_csv,
    )

    report, enriched = evaluate_tta_policies(
        calib,
        test,
        checkpoint=args.checkpoint,
        config=args.config,
        quality_csv=args.quality_csv,
        target_coverage=float(args.target_coverage),
        min_calibration_samples=int(args.min_calibration_samples),
        min_slice_samples=int(args.min_slice_samples),
        tta_views=int(args.tta_views),
        tta_strength=float(args.tta_strength),
        tta_noise_std=float(args.tta_noise_std),
    )

    out_dir = args.out_dir or args.checkpoint.parent / "tta_uncertainty_conformal"
    out_json = args.out_json or out_dir / "tta_uncertainty_conformal.json"
    out_md = args.out_md or out_dir / "tta_uncertainty_conformal.md"
    out_predictions = args.out_predictions or out_dir / "predictions_test_with_tta_uncertainty.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    enriched.to_csv(out_predictions, index=False, encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote: {out_json}")
    print(f"wrote: {out_predictions}")


def _collect_tta_records(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    split: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    tta_views: int,
    tta_strength: float,
    tta_noise_std: float,
    quality_csv: Path,
) -> pd.DataFrame:
    print(f"collecting {split} TTA records on {device}...", file=sys.stderr, flush=True)
    loader = build_loader(
        cfg,
        split,
        max_samples=max_samples,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    records: list[dict[str, Any]] = []
    for batch in loader:
        moved = move_batch(batch, device)
        clean = model(moved["image"], grl_lambda=0.0, domain_idx=moved.get("domain_idx"))
        views = [clean]
        for _ in range(max(int(tta_views), 0)):
            perturbed = _weak_style_perturb_normalized(
                moved["image"],
                strength=tta_strength,
                noise_std=tta_noise_std,
            )
            views.append(model(perturbed, grl_lambda=0.0, domain_idx=moved.get("domain_idx")))
        records.extend(_batch_records(batch, moved, views))
    df = pd.DataFrame(records)
    df = attach_quality_flags(df, quality_csv)
    df["quality_bin"] = [_quality_bin(row) for _, row in df.iterrows()]
    df["tta_uncertainty_bin"] = _uncertainty_bins(df["tta_uncertainty_score"])
    df["dataset_key"] = df["dataset"].astype(str)
    df["group_key_safe"] = df["group_key"].astype(str)
    df["dataset_uncertainty_key"] = df["dataset_key"] + "::tta=" + df["tta_uncertainty_bin"].astype(str)
    df["dataset_core_uncertainty_key"] = df["group_key_safe"] + "::tta=" + df["tta_uncertainty_bin"].astype(str)
    quality_join = float(df["quality_joined"].map(_to_bool).mean()) if "quality_joined" in df else 0.0
    print(
        f"collected {split}: rows={len(df)} mu={int(df['mu_known_bool'].sum())} "
        f"quality_join={100.0 * quality_join:.2f}%",
        file=sys.stderr,
        flush=True,
    )
    return df


def _batch_records(
    batch: dict[str, Any],
    moved: dict[str, Any],
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clean = views[0]
    intervals = torch.stack([view["mu_interval"].detach() for view in views], dim=0)
    mu_means = torch.stack([view["mu_mean"].detach() for view in views], dim=0)
    interval_std = intervals.std(dim=0).mean(dim=1).cpu().numpy()
    mu_std = mu_means.std(dim=0).cpu().numpy()
    clean_interval = clean["mu_interval"].detach().cpu().numpy()
    clean_mean = clean["mu_mean"].detach().cpu().numpy()
    target_interval = moved["mu_interval"].detach().cpu().numpy()
    mu_mask = moved["mu_mask"].detach().cpu().numpy().astype(bool)

    task_pred: dict[str, np.ndarray] = {}
    task_conf: dict[str, np.ndarray] = {}
    task_entropy: dict[str, np.ndarray] = {}
    task_disagreement: dict[str, np.ndarray] = {}
    for task in TASKS:
        probs = [torch.softmax(view["logits"][task].detach(), dim=1) for view in views]
        stacked = torch.stack(probs, dim=0)
        clean_prob = probs[0]
        preds = stacked.argmax(dim=2)
        clean_pred = preds[0]
        task_pred[task] = clean_pred.cpu().numpy()
        task_conf[task] = clean_prob.max(dim=1).values.cpu().numpy()
        task_entropy[task] = (-(clean_prob * clean_prob.clamp_min(1e-8).log()).sum(dim=1)).cpu().numpy()
        task_disagreement[task] = (preds != clean_pred.unsqueeze(0)).float().mean(dim=0).cpu().numpy()

    label_np = {task: moved["labels"][task].detach().cpu().numpy() for task in TASKS}
    mask_np = {task: moved["masks"][task].detach().cpu().numpy().astype(bool) for task in TASKS}
    out: list[dict[str, Any]] = []
    for i in range(int(moved["image"].size(0))):
        score = float(
            interval_std[i]
            + mu_std[i]
            + 0.05 * task_disagreement["friction"][i]
            + 0.05 * task_disagreement["risk"][i]
        )
        rec: dict[str, Any] = {
            "image_path": str(batch["image_path"][i]),
            "image_path_norm": _norm_path(batch["image_path"][i]),
            "dataset": str(batch["dataset"][i]),
            "domain_id": str(batch["domain_id"][i]),
            "group_key": str(batch["group_key"][i]),
            "mu_known": bool(mu_mask[i]),
            "mu_known_bool": bool(mu_mask[i]),
            "target_mu_low": float(target_interval[i, 0]) if mu_mask[i] else math.nan,
            "target_mu_high": float(target_interval[i, 1]) if mu_mask[i] else math.nan,
            "pred_mu_low": float(clean_interval[i, 0]),
            "pred_mu_high": float(clean_interval[i, 1]),
            "pred_mu_mean": float(clean_mean[i]),
            "raw_interval_width": float(clean_interval[i, 1] - clean_interval[i, 0]),
            "tta_interval_std": float(interval_std[i]),
            "tta_mu_std": float(mu_std[i]),
            "tta_uncertainty_score": score,
        }
        if mu_mask[i]:
            rec["raw_interval_covers"] = bool(
                clean_interval[i, 0] <= target_interval[i, 0]
                and clean_interval[i, 1] >= target_interval[i, 1]
            )
            rec["conformal_score"] = float(
                max(
                    clean_interval[i, 0] - target_interval[i, 0],
                    target_interval[i, 1] - clean_interval[i, 1],
                    0.0,
                )
            )
        else:
            rec["raw_interval_covers"] = None
            rec["conformal_score"] = 0.0
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
            rec[f"{task}_tta_disagreement"] = float(task_disagreement[task][i])
        out.append(rec)
    return out


def evaluate_tta_policies(
    calib: pd.DataFrame,
    test: pd.DataFrame,
    *,
    checkpoint: Path,
    config: Path,
    quality_csv: Path,
    target_coverage: float,
    min_calibration_samples: int,
    min_slice_samples: int,
    tta_views: int,
    tta_strength: float,
    tta_noise_std: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    calib_known = calib[calib["mu_known_bool"]].copy()
    global_radius = _conformal_radius(calib_known["conformal_score"].to_numpy(dtype=float), target_coverage)
    tables = {
        "tta": _fit_group_radii(calib_known, "tta_uncertainty_bin", target_coverage, min_calibration_samples),
        "dataset_tta": _fit_group_radii(calib_known, "dataset_uncertainty_key", target_coverage, min_calibration_samples),
        "state_tta": _fit_group_radii(calib_known, "dataset_core_uncertainty_key", target_coverage, min_calibration_samples),
        "quality": _fit_group_radii(calib_known, "quality_bin", target_coverage, min_calibration_samples),
    }
    policy_defs = {
        "global": ["global"],
        "tta_uncertainty": ["global", "tta"],
        "dataset_tta_uncertainty": ["global", "tta", "dataset_tta"],
        "state_tta_uncertainty": ["global", "tta", "dataset_tta", "state_tta"],
        "quality_tta_uncertainty": ["global", "tta", "quality", "dataset_tta", "state_tta"],
    }
    enriched = test.copy()
    policies: dict[str, Any] = {}
    for name, scopes in policy_defs.items():
        radii = _assign_radii(test, global_radius, tables, scopes)
        enriched[f"{name}_radius"] = radii
        enriched[f"{name}_calibrated_low"] = np.clip(test["pred_mu_low"].to_numpy(dtype=float) - radii, 0.0, 1.2)
        enriched[f"{name}_calibrated_high"] = np.clip(test["pred_mu_high"].to_numpy(dtype=float) + radii, 0.0, 1.2)
        policies[name] = _policy_summary(test, radii, min_slice_samples=min_slice_samples, name=name)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "quality_csv": str(quality_csv),
        "target_coverage": float(target_coverage),
        "tta_views": int(tta_views),
        "tta_strength": float(tta_strength),
        "tta_noise_std": float(tta_noise_std),
        "calibration": {
            "num_samples": int(len(calib)),
            "num_mu_samples": int(len(calib_known)),
            "global_radius": float(global_radius),
            "num_radii": {name: int(len(table)) for name, table in tables.items()},
        },
        "test": {
            "num_samples": int(len(test)),
            "num_mu_samples": int(test["mu_known_bool"].sum()),
        },
        "policies": policies,
    }
    report["decision"] = _decision(report)
    return report, enriched


def _fit_group_radii(
    rows: pd.DataFrame,
    group_col: str,
    target_coverage: float,
    min_calibration_samples: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if group_col not in rows.columns:
        return out
    for group, part in rows.groupby(group_col, dropna=False, sort=True):
        if len(part) < int(min_calibration_samples):
            continue
        out[str(group)] = _conformal_radius(part["conformal_score"].to_numpy(dtype=float), target_coverage)
    return out


def _assign_radii(
    rows: pd.DataFrame,
    global_radius: float,
    tables: dict[str, dict[str, float]],
    scopes: list[str],
) -> np.ndarray:
    radii = np.full(len(rows), float(global_radius), dtype=np.float32)
    scope_cols = {
        "tta": "tta_uncertainty_bin",
        "dataset_tta": "dataset_uncertainty_key",
        "state_tta": "dataset_core_uncertainty_key",
        "quality": "quality_bin",
    }
    for scope in scopes:
        if scope == "global":
            continue
        table = tables.get(scope, {})
        col = scope_cols.get(scope)
        if not table or col not in rows.columns:
            continue
        mapped = rows[col].astype(str).map(table).astype(float)
        mask = mapped.notna().to_numpy()
        if mask.any():
            radii[mask] = np.maximum(radii[mask], mapped[mask].to_numpy(dtype=np.float32))
    return radii


def _policy_summary(rows: pd.DataFrame, radii: np.ndarray, *, min_slice_samples: int, name: str) -> dict[str, Any]:
    slices = {
        "dataset": _slice_table(rows, radii, "dataset_key", min_slice_samples),
        "state": _slice_table(rows, radii, "group_key_safe", min_slice_samples),
        "tta": _slice_table(rows, radii, "tta_uncertainty_bin", min_slice_samples),
        "quality": _slice_table(rows, radii, "quality_bin", min_slice_samples),
    }
    return {
        "policy": name,
        "pooled": _interval_summary(rows, radii),
        "radius": {
            "mean_radius": float(np.mean(radii[rows["mu_known_bool"].to_numpy(dtype=bool)]))
            if rows["mu_known_bool"].any()
            else 0.0,
            "max_radius": float(np.max(radii[rows["mu_known_bool"].to_numpy(dtype=bool)]))
            if rows["mu_known_bool"].any()
            else 0.0,
        },
        "slices": slices,
        "worst_slice": _worst_slice(slices),
    }


def _interval_summary(rows: pd.DataFrame, radii: np.ndarray) -> dict[str, Any]:
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    if not known.any():
        return {"num_samples": 0}
    sub = rows.loc[known]
    r = radii[known]
    raw_low = sub["pred_mu_low"].to_numpy(dtype=float)
    raw_high = sub["pred_mu_high"].to_numpy(dtype=float)
    target_low = sub["target_mu_low"].to_numpy(dtype=float)
    target_high = sub["target_mu_high"].to_numpy(dtype=float)
    covers = (raw_low - r <= target_low) & (raw_high + r >= target_high)
    raw_width = sub["raw_interval_width"].to_numpy(dtype=float)
    return {
        "num_samples": int(len(sub)),
        "raw_coverage": float(np.mean(sub.get("raw_interval_covers", False).map(_to_bool).to_numpy())),
        "raw_width": float(np.mean(raw_width)),
        "calibrated_coverage": float(np.mean(covers)),
        "calibrated_width": float(np.mean(np.clip(raw_width + 2.0 * r, 0.0, 1.2))),
        "mean_radius": float(np.mean(r)),
        "mean_tta_uncertainty": float(sub["tta_uncertainty_score"].mean()),
    }


def _slice_table(rows: pd.DataFrame, radii: np.ndarray, group_col: str, min_slice_samples: int) -> dict[str, Any]:
    if group_col not in rows.columns:
        return {}
    out: dict[str, Any] = {}
    values = rows[group_col].astype(str).to_numpy()
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    for group in sorted(set(values.tolist())):
        mask = values == group
        if int((mask & known).sum()) < int(min_slice_samples):
            continue
        out[str(group)] = _interval_summary(rows.loc[mask], radii[mask])
    return out


def _worst_slice(slices: dict[str, dict[str, Any]]) -> dict[str, Any]:
    worst: dict[str, Any] | None = None
    for scope, table in slices.items():
        for name, row in table.items():
            cov = row.get("calibrated_coverage")
            if cov is None:
                continue
            item = {
                "scope": scope,
                "name": name,
                "num_samples": row.get("num_samples"),
                "calibrated_coverage": float(cov),
                "calibrated_width": row.get("calibrated_width"),
            }
            if worst is None or item["calibrated_coverage"] < worst["calibrated_coverage"]:
                worst = item
    return worst or {}


def _decision(report: dict[str, Any]) -> dict[str, Any]:
    policies = report["policies"]
    base = policies["global"]["pooled"]
    base_worst = policies["global"]["worst_slice"]
    candidates = []
    for name, row in policies.items():
        if name == "global":
            continue
        pooled = row["pooled"]
        worst = row["worst_slice"]
        candidates.append(
            {
                "name": name,
                "coverage": _num(pooled.get("calibrated_coverage")),
                "width": _num(pooled.get("calibrated_width")),
                "worst_coverage": _num(worst.get("calibrated_coverage")),
                "coverage_delta_vs_global": _num(pooled.get("calibrated_coverage")) - _num(base.get("calibrated_coverage")),
                "width_delta_vs_global": _num(pooled.get("calibrated_width")) - _num(base.get("calibrated_width")),
                "worst_delta_vs_global": _num(worst.get("calibrated_coverage")) - _num(base_worst.get("calibrated_coverage")),
            }
        )
    feasible = [
        row
        for row in candidates
        if row["coverage_delta_vs_global"] >= 0.02
        and row["worst_delta_vs_global"] >= 0.02
        and row["width_delta_vs_global"] <= 0.08
    ]
    feasible.sort(key=lambda row: (-row["worst_delta_vs_global"], row["width_delta_vs_global"]))
    if feasible:
        return {
            "status": "keep_for_more_eval",
            "best": feasible[0],
            "reason": "TTA uncertainty bins improve coverage and worst-slice coverage with bounded width cost",
        }
    return {
        "status": "discard_or_hold",
        "best": candidates[0] if candidates else None,
        "reason": "TTA uncertainty did not clearly improve coverage-width-worst-slice tradeoff in this fast probe",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TTA Uncertainty Conformal Evaluation",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Target coverage: `{100.0 * report['target_coverage']:.1f}%`",
        f"- TTA views: `{report['tta_views']}`",
        f"- Calibration mu samples: `{report['calibration']['num_mu_samples']}`",
        f"- Test mu samples: `{report['test']['num_mu_samples']}`",
        "",
        "| policy | raw cov | cal cov | cal width | mean radius | worst slice | worst cov |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for name, row in report["policies"].items():
        pooled = row["pooled"]
        radius = row["radius"]
        worst = row["worst_slice"]
        lines.append(
            "| {name} | {raw} | {cov} | {width} | {radius} | {worst} | {worst_cov} |".format(
                name=name,
                raw=_fmt_pct(pooled.get("raw_coverage")),
                cov=_fmt_pct(pooled.get("calibrated_coverage")),
                width=_fmt_abs(pooled.get("calibrated_width")),
                radius=_fmt_abs(radius.get("mean_radius")),
                worst=_worst_name(worst),
                worst_cov=_fmt_pct(worst.get("calibrated_coverage")),
            )
        )
    decision = report["decision"]
    lines.extend(["", "## Fast Decision", "", f"- Status: `{decision['status']}`", f"- Reason: {decision['reason']}"])
    if decision.get("best"):
        best = decision["best"]
        lines.append(
            "- Best candidate: `{name}` coverage `{cov}`, width `{width}`, worst `{worst}`.".format(
                name=best.get("name"),
                cov=_fmt_pct(best.get("coverage")),
                width=_fmt_abs(best.get("width")),
                worst=_fmt_pct(best.get("worst_coverage")),
            )
        )
    return "\n".join(lines) + "\n"


def _conformal_radius(scores: np.ndarray, target_coverage: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
    sorted_scores = np.sort(scores)
    idx = min(math.ceil((len(sorted_scores) + 1) * target_coverage) - 1, len(sorted_scores) - 1)
    idx = max(idx, 0)
    return float(sorted_scores[idx])


def _uncertainty_bins(values: pd.Series) -> list[str]:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if len(values) < 3 or float(values.max()) <= float(values.min()):
        return ["stable"] * len(values)
    q50 = float(values.quantile(0.50))
    q80 = float(values.quantile(0.80))
    out = []
    for value in values:
        x = float(value)
        if x >= q80:
            out.append("high")
        elif x >= q50:
            out.append("mid")
        else:
            out.append("low")
    return out


def _quality_bin(row: pd.Series) -> str:
    if not _to_bool(row.get("quality_joined", False)):
        return "quality_unknown"
    for name in ["near_white", "overexposed", "low_texture", "low_contrast", "dark", "suspicious"]:
        if _to_bool(row.get(f"{name}_flag", False)):
            return name
    try:
        if float(row.get("specular_highlight_frac") or 0.0) >= 0.02:
            return "specular"
    except (TypeError, ValueError):
        pass
    return "normal"


def _norm_path(value: Any) -> str:
    return str(value).replace("\\", "/").lower()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _num(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        number = float(value)
        if math.isnan(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _worst_name(worst: dict[str, Any]) -> str:
    if not worst:
        return "-"
    return f"{worst.get('scope')}::{worst.get('name')}"


if __name__ == "__main__":
    main()

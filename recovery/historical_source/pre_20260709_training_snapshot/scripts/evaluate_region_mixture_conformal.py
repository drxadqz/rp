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

import cv2
import numpy as np
import pandas as pd
import torch

from evaluate_quality_mondrian_conformal import _collect_split, _conformal_radius
from friction_affordance.engine import build_model
from friction_affordance.utils import load_yaml, resolve_device


DEFAULT_QUALITY_CSV = Path("data/quality_flags/image_quality_flags.csv")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast post-hoc probe for segmentation-style region mixture calibration. "
            "It uses unsupervised road-surface regions as a cheap substitute for semantic "
            "segmentation masks and tests whether mixed/glossy/snow-like regions need "
            "more conservative weak friction intervals."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--min-calibration-samples", type=int, default=50)
    parser.add_argument("--min-slice-samples", type=int, default=30)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--max-test-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
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

    calib = _collect_split(
        model,
        cfg,
        device,
        split="val",
        quality_csv=args.quality_csv,
        max_samples=int(args.max_val_samples),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    test = _collect_split(
        model,
        cfg,
        device,
        split="test",
        quality_csv=args.quality_csv,
        max_samples=int(args.max_test_samples),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    calib = attach_region_mixture(calib, clusters=int(args.clusters), image_size=int(args.image_size))
    thresholds = _mixture_thresholds(calib["region_mixture_score"])
    calib = _apply_region_bins(calib, thresholds)
    test = attach_region_mixture(test, clusters=int(args.clusters), image_size=int(args.image_size))
    test = _apply_region_bins(test, thresholds)

    report, enriched = evaluate_region_policies(
        calib,
        test,
        checkpoint=args.checkpoint,
        config=args.config,
        quality_csv=args.quality_csv,
        target_coverage=float(args.target_coverage),
        min_calibration_samples=int(args.min_calibration_samples),
        min_slice_samples=int(args.min_slice_samples),
        thresholds=thresholds,
        clusters=int(args.clusters),
        image_size=int(args.image_size),
    )

    out_dir = args.out_dir or args.checkpoint.parent / "region_mixture_conformal"
    out_json = args.out_json or out_dir / "region_mixture_conformal.json"
    out_md = args.out_md or out_dir / "region_mixture_conformal.md"
    out_predictions = args.out_predictions or out_dir / "predictions_test_with_region_mixture.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    enriched.to_csv(out_predictions, index=False, encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote: {out_json}")
    print(f"wrote: {out_predictions}")


def attach_region_mixture(rows: pd.DataFrame, *, clusters: int, image_size: int) -> pd.DataFrame:
    out = rows.copy()
    cache: dict[str, dict[str, float | str]] = {}
    records = []
    for path in out["image_path"].astype(str).tolist():
        if path not in cache:
            cache[path] = _region_features(Path(path), clusters=clusters, image_size=image_size)
        records.append(cache[path])
    feats = pd.DataFrame(records)
    return pd.concat([out.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


def _region_features(path: Path, *, clusters: int, image_size: int) -> dict[str, float | str]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "region_decode_ok": False,
            "region_mixture_score": 0.0,
            "region_entropy": 0.0,
            "region_color_span": 0.0,
            "region_texture_span": 0.0,
            "region_snow_frac": 0.0,
            "region_specular_frac": 0.0,
            "region_dark_water_frac": 0.0,
        }
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = image.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)

    labels = _kmeans_regions(rgb, clusters=clusters)
    areas = []
    cluster_lab = []
    cluster_edge = []
    snow_frac = 0.0
    specular_frac = 0.0
    dark_water_frac = 0.0
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        area = float(mask.mean())
        if area <= 0:
            continue
        areas.append(area)
        cluster_lab.append(lab[mask].mean(axis=0))
        cluster_edge.append(float(edge[mask].mean()))
        v = float(value[mask].mean())
        s = float(saturation[mask].mean())
        e = float(edge[mask].mean())
        if v > 0.72 and s < 0.28:
            snow_frac += area
        if v > 0.82 and s < 0.24:
            specular_frac += area
        if v < 0.38 and e < 0.10:
            dark_water_frac += area
    p = np.asarray(areas, dtype=np.float32)
    entropy = float(-(p * np.log(np.clip(p, 1e-8, 1.0))).sum() / math.log(max(len(p), 2))) if len(p) else 0.0
    color_span = 0.0
    if len(cluster_lab) >= 2:
        centers = np.asarray(cluster_lab, dtype=np.float32)
        color_span = float(np.mean(np.std(centers / np.asarray([100.0, 255.0, 255.0], dtype=np.float32), axis=0)))
    texture_span = float(np.std(cluster_edge)) if len(cluster_edge) >= 2 else 0.0
    optical_risk = min(1.0, snow_frac + specular_frac + 0.5 * dark_water_frac)
    mixture_score = float(
        np.clip(0.40 * entropy + 1.10 * color_span + 0.80 * texture_span + 0.35 * optical_risk, 0.0, 1.0)
    )
    return {
        "region_decode_ok": True,
        "region_mixture_score": mixture_score,
        "region_entropy": entropy,
        "region_color_span": color_span,
        "region_texture_span": texture_span,
        "region_snow_frac": float(snow_frac),
        "region_specular_frac": float(specular_frac),
        "region_dark_water_frac": float(dark_water_frac),
    }


def _kmeans_regions(rgb: np.ndarray, *, clusters: int) -> np.ndarray:
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xy = np.stack([xx / max(w - 1, 1), yy / max(h - 1, 1)], axis=2)
    lab = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    features = np.concatenate([lab, 0.35 * xy], axis=2).reshape(-1, 5).astype(np.float32)
    k = max(2, min(int(clusters), features.shape[0]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
    _compactness, labels, _centers = cv2.kmeans(features, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(h, w)


def _mixture_thresholds(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return {
        "q50": float(values.quantile(0.50)),
        "q80": float(values.quantile(0.80)),
    }


def _apply_region_bins(rows: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = rows.copy()
    q50 = float(thresholds.get("q50", 0.0))
    q80 = float(thresholds.get("q80", q50))
    bins = []
    for value in pd.to_numeric(out["region_mixture_score"], errors="coerce").fillna(0.0):
        x = float(value)
        if x >= q80:
            bins.append("high")
        elif x >= q50:
            bins.append("mid")
        else:
            bins.append("low")
    out["region_mixture_bin"] = bins
    out["dataset_key"] = out["dataset"].astype(str)
    out["group_key_safe"] = out["group_key"].astype(str)
    out["dataset_region_key"] = out["dataset_key"] + "::region=" + out["region_mixture_bin"].astype(str)
    out["state_region_key"] = out["group_key_safe"] + "::region=" + out["region_mixture_bin"].astype(str)
    return out


def evaluate_region_policies(
    calib: pd.DataFrame,
    test: pd.DataFrame,
    *,
    checkpoint: Path,
    config: Path,
    quality_csv: Path,
    target_coverage: float,
    min_calibration_samples: int,
    min_slice_samples: int,
    thresholds: dict[str, float],
    clusters: int,
    image_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    calib_known = calib[calib["mu_known_bool"]].copy()
    global_radius = _conformal_radius(calib_known["conformal_score"].to_numpy(dtype=float), target_coverage)
    tables = {
        "region": _fit_group_radii(calib_known, "region_mixture_bin", target_coverage, min_calibration_samples),
        "dataset_region": _fit_group_radii(calib_known, "dataset_region_key", target_coverage, min_calibration_samples),
        "state_region": _fit_group_radii(calib_known, "state_region_key", target_coverage, min_calibration_samples),
    }
    policy_defs = {
        "global": ["global"],
        "region_mixture": ["global", "region"],
        "dataset_region_mixture": ["global", "region", "dataset_region"],
        "state_region_mixture": ["global", "region", "dataset_region", "state_region"],
    }
    enriched = test.copy()
    policies: dict[str, Any] = {}
    for name, scopes in policy_defs.items():
        radii = _assign_radii(test, global_radius, tables, scopes)
        enriched[f"{name}_radius"] = radii
        policies[name] = _policy_summary(test, radii, name=name, min_slice_samples=min_slice_samples)
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "quality_csv": str(quality_csv),
        "target_coverage": float(target_coverage),
        "clusters": int(clusters),
        "image_size": int(image_size),
        "thresholds": thresholds,
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


def _fit_group_radii(rows: pd.DataFrame, group_col: str, target_coverage: float, min_calibration_samples: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, part in rows.groupby(group_col, dropna=False, sort=True):
        if len(part) < int(min_calibration_samples):
            continue
        out[str(group)] = _conformal_radius(part["conformal_score"].to_numpy(dtype=float), target_coverage)
    return out


def _assign_radii(rows: pd.DataFrame, global_radius: float, tables: dict[str, dict[str, float]], scopes: list[str]) -> np.ndarray:
    radii = np.full(len(rows), float(global_radius), dtype=np.float32)
    scope_cols = {
        "region": "region_mixture_bin",
        "dataset_region": "dataset_region_key",
        "state_region": "state_region_key",
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


def _policy_summary(rows: pd.DataFrame, radii: np.ndarray, *, name: str, min_slice_samples: int) -> dict[str, Any]:
    slices = {
        "dataset": _slice_table(rows, radii, "dataset_key", min_slice_samples),
        "state": _slice_table(rows, radii, "group_key_safe", min_slice_samples),
        "region": _slice_table(rows, radii, "region_mixture_bin", min_slice_samples),
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
        "mean_region_mixture_score": float(sub["region_mixture_score"].mean()),
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
            "reason": "segmentation-style region mixture improves coverage/worst-slice coverage with bounded width cost",
        }
    return {
        "status": "discard_or_hold",
        "best": candidates[0] if candidates else None,
        "reason": "region mixture did not clearly improve coverage-width-worst-slice tradeoff in this fast probe",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Region Mixture Conformal Evaluation",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Target coverage: `{100.0 * report['target_coverage']:.1f}%`",
        f"- Clusters: `{report['clusters']}`",
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

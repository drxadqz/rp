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

from evaluate_quality_mondrian_conformal import (
    _collect_split,
    _conformal_radius,
    _fit_group_radii,
    _scope_column,
    _to_bool,
    prepare_records,
)
from friction_affordance.engine import build_model
from friction_affordance.utils import load_yaml, resolve_device


DEFAULT_QUALITY_CSV = Path("data/quality_flags/image_quality_flags.csv")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast post-hoc probe for asymmetric Mondrian conformal intervals. "
            "It tests whether separate lower/upper calibration can improve coverage-width "
            "without retraining."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--min-calibration-samples", type=int, default=50)
    parser.add_argument("--min-slice-samples", type=int, default=50)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
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

    report, enriched = evaluate_asymmetric_policies(
        calib,
        test,
        checkpoint=args.checkpoint,
        config=args.config,
        quality_csv=args.quality_csv,
        target_coverage=float(args.target_coverage),
        min_calibration_samples=int(args.min_calibration_samples),
        min_slice_samples=int(args.min_slice_samples),
    )

    out_dir = args.out_dir or args.checkpoint.parent / "asymmetric_mondrian_conformal"
    out_json = args.out_json or out_dir / "asymmetric_mondrian_conformal.json"
    out_md = args.out_md or out_dir / "asymmetric_mondrian_conformal.md"
    out_predictions = args.out_predictions or out_dir / "predictions_test_with_asymmetric_radii.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    enriched.to_csv(out_predictions, index=False, encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote: {out_json}")
    print(f"wrote: {out_predictions}")


def evaluate_asymmetric_policies(
    calib: pd.DataFrame,
    test: pd.DataFrame,
    *,
    checkpoint: Path,
    config: Path,
    quality_csv: Path,
    target_coverage: float,
    min_calibration_samples: int,
    min_slice_samples: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    calib = _add_asymmetric_scores(calib)
    test = _add_asymmetric_scores(test)
    calib_known = calib[calib["mu_known_bool"]].copy()

    symmetric_radius = _conformal_radius(
        calib_known["conformal_score"].to_numpy(dtype=float),
        target_coverage,
    )
    symmetric_tables = {
        "dataset": _fit_group_radii(calib_known, "dataset_key", target_coverage, min_calibration_samples),
        "state": _fit_group_radii(calib_known, "group_key_safe", target_coverage, min_calibration_samples),
        "risk": _fit_group_radii(calib_known, "risk_key", target_coverage, min_calibration_samples),
        "quality": _fit_group_radii(calib_known, "quality_bin", target_coverage, min_calibration_samples),
        "dataset_quality": _fit_group_radii(
            calib_known, "dataset_quality_key", target_coverage, min_calibration_samples
        ),
        "dataset_core_risk": _fit_group_radii(
            calib_known, "dataset_core_risk_key", target_coverage, min_calibration_samples
        ),
        "dataset_core_quality": _fit_group_radii(
            calib_known, "dataset_core_quality_key", target_coverage, min_calibration_samples
        ),
    }

    side_targets = {
        "marginal_q90": target_coverage,
        "bonferroni_q95": min(0.995, 0.5 + 0.5 * target_coverage),
    }
    scopes = {
        "global": ["global"],
        "hierarchical_safety": ["global", "dataset", "state", "risk", "dataset_core_risk"],
        "hierarchical_quality_safety": [
            "global",
            "dataset",
            "state",
            "risk",
            "quality",
            "dataset_quality",
            "dataset_core_risk",
            "dataset_core_quality",
        ],
    }

    enriched = test.copy()
    policies: dict[str, Any] = {}
    sym_low, sym_high = _assign_symmetric_radii(test, symmetric_radius, symmetric_tables, scopes["hierarchical_quality_safety"])
    policies["symmetric_hierarchical_quality"] = _policy_summary(
        test,
        sym_low,
        sym_high,
        name="symmetric_hierarchical_quality",
        min_slice_samples=min_slice_samples,
    )
    _store_policy_columns(enriched, "symmetric_hierarchical_quality", sym_low, sym_high)

    asym_tables = {
        label: _fit_asymmetric_tables(calib_known, side_target, min_calibration_samples)
        for label, side_target in side_targets.items()
    }
    for target_label, tables in asym_tables.items():
        for scope_name, scope_list in scopes.items():
            name = f"asymmetric_{target_label}_{scope_name}"
            lower, upper = _assign_asymmetric_radii(test, tables, scope_list)
            policies[name] = _policy_summary(
                test,
                lower,
                upper,
                name=name,
                min_slice_samples=min_slice_samples,
            )
            _store_policy_columns(enriched, name, lower, upper)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "quality_csv": str(quality_csv),
        "target_coverage": float(target_coverage),
        "min_calibration_samples": int(min_calibration_samples),
        "min_slice_samples": int(min_slice_samples),
        "calibration": {
            "num_samples": int(len(calib)),
            "num_mu_samples": int(len(calib_known)),
            "symmetric_global_radius": float(symmetric_radius),
            "side_targets": side_targets,
        },
        "test": {
            "num_samples": int(len(test)),
            "num_mu_samples": int(test["mu_known_bool"].sum()),
        },
        "policies": policies,
    }
    report["decision"] = _decision(report)
    return report, enriched


def _add_asymmetric_scores(rows: pd.DataFrame) -> pd.DataFrame:
    out = prepare_records(rows)
    known = out["mu_known_bool"].to_numpy(dtype=bool)
    lower = np.zeros(len(out), dtype=np.float32)
    upper = np.zeros(len(out), dtype=np.float32)
    if known.any():
        pred_low = out["pred_mu_low"].to_numpy(dtype=float)
        pred_high = out["pred_mu_high"].to_numpy(dtype=float)
        target_low = out["target_mu_low"].to_numpy(dtype=float)
        target_high = out["target_mu_high"].to_numpy(dtype=float)
        lower[known] = np.maximum(pred_low[known] - target_low[known], 0.0).astype(np.float32)
        upper[known] = np.maximum(target_high[known] - pred_high[known], 0.0).astype(np.float32)
    out["lower_conformal_score"] = lower
    out["upper_conformal_score"] = upper
    return out


def _fit_asymmetric_tables(
    rows: pd.DataFrame,
    side_target: float,
    min_calibration_samples: int,
) -> dict[str, Any]:
    tables: dict[str, Any] = {
        "global": {
            "lower": _conformal_radius(rows["lower_conformal_score"].to_numpy(dtype=float), side_target),
            "upper": _conformal_radius(rows["upper_conformal_score"].to_numpy(dtype=float), side_target),
        }
    }
    for scope, col in {
        "dataset": "dataset_key",
        "state": "group_key_safe",
        "risk": "risk_key",
        "quality": "quality_bin",
        "dataset_quality": "dataset_quality_key",
        "dataset_core_risk": "dataset_core_risk_key",
        "dataset_core_quality": "dataset_core_quality_key",
    }.items():
        tables[scope] = _fit_group_side_radii(rows, col, side_target, min_calibration_samples)
    return tables


def _fit_group_side_radii(
    rows: pd.DataFrame,
    group_col: str,
    side_target: float,
    min_calibration_samples: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if group_col not in rows.columns:
        return out
    for group, part in rows.groupby(group_col, dropna=False, sort=True):
        if len(part) < int(min_calibration_samples):
            continue
        out[str(group)] = {
            "lower": _conformal_radius(part["lower_conformal_score"].to_numpy(dtype=float), side_target),
            "upper": _conformal_radius(part["upper_conformal_score"].to_numpy(dtype=float), side_target),
        }
    return out


def _assign_symmetric_radii(
    rows: pd.DataFrame,
    global_radius: float,
    tables: dict[str, dict[str, float]],
    scopes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.full(len(rows), float(global_radius), dtype=np.float32)
    for scope in scopes:
        if scope == "global":
            continue
        table = tables.get(scope, {})
        col = _scope_column(scope)
        if not table or col not in rows.columns:
            continue
        mapped = rows[col].astype(str).map(table).astype(float)
        mask = mapped.notna().to_numpy()
        if mask.any():
            radii[mask] = np.maximum(radii[mask], mapped[mask].to_numpy(dtype=np.float32))
    return radii.copy(), radii.copy()


def _assign_asymmetric_radii(
    rows: pd.DataFrame,
    tables: dict[str, Any],
    scopes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    global_row = tables["global"]
    lower = np.full(len(rows), float(global_row["lower"]), dtype=np.float32)
    upper = np.full(len(rows), float(global_row["upper"]), dtype=np.float32)
    for scope in scopes:
        if scope == "global":
            continue
        table = tables.get(scope, {})
        col = _scope_column(scope)
        if not table or col not in rows.columns:
            continue
        for idx, key in enumerate(rows[col].astype(str).tolist()):
            item = table.get(key)
            if not item:
                continue
            lower[idx] = max(lower[idx], float(item["lower"]))
            upper[idx] = max(upper[idx], float(item["upper"]))
    return lower, upper


def _store_policy_columns(rows: pd.DataFrame, name: str, lower: np.ndarray, upper: np.ndarray) -> None:
    rows[f"{name}_lower_radius"] = lower
    rows[f"{name}_upper_radius"] = upper
    rows[f"{name}_calibrated_low"] = np.clip(rows["pred_mu_low"].to_numpy(dtype=float) - lower, 0.0, 1.2)
    rows[f"{name}_calibrated_high"] = np.clip(rows["pred_mu_high"].to_numpy(dtype=float) + upper, 0.0, 1.2)
    rows[f"{name}_calibrated_covers"] = _covers(rows, lower, upper)


def _policy_summary(
    rows: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    name: str,
    min_slice_samples: int,
) -> dict[str, Any]:
    pooled = _interval_summary(rows, lower, upper)
    slices = {
        "dataset": _slice_table(rows, lower, upper, "dataset_key", min_slice_samples),
        "state": _slice_table(rows, lower, upper, "group_key_safe", min_slice_samples),
        "quality": _slice_table(rows, lower, upper, "quality_bin", min_slice_samples),
        "dataset_quality": _slice_table(rows, lower, upper, "dataset_quality_key", min_slice_samples),
    }
    return {
        "policy": name,
        "pooled": pooled,
        "radius": {
            "mean_lower_radius": _mean_known(rows, lower),
            "mean_upper_radius": _mean_known(rows, upper),
            "max_lower_radius": _max_known(rows, lower),
            "max_upper_radius": _max_known(rows, upper),
        },
        "slices": slices,
        "worst_slice": _worst_slice(slices),
    }


def _interval_summary(rows: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    if not known.any():
        return {"num_samples": 0}
    sub = rows.loc[known]
    lo = lower[known]
    hi = upper[known]
    raw_low = sub["pred_mu_low"].to_numpy(dtype=float)
    raw_high = sub["pred_mu_high"].to_numpy(dtype=float)
    cal_low = np.clip(raw_low - lo, 0.0, 1.2)
    cal_high = np.clip(raw_high + hi, 0.0, 1.2)
    target_mid = 0.5 * (
        sub["target_mu_low"].to_numpy(dtype=float) + sub["target_mu_high"].to_numpy(dtype=float)
    )
    return {
        "num_samples": int(len(sub)),
        "raw_coverage": float(np.mean(sub.get("raw_interval_covers", False).map(_to_bool).to_numpy())),
        "raw_width": float(np.mean(sub["raw_interval_width"].to_numpy(dtype=float))),
        "calibrated_coverage": float(np.mean(_covers(sub, lo, hi))),
        "calibrated_width": float(np.mean(np.maximum(cal_high - cal_low, 0.0))),
        "mean_lower_radius": float(np.mean(lo)),
        "mean_upper_radius": float(np.mean(hi)),
        "mean_mae_to_interval_mid": float(
            np.abs(sub["pred_mu_mean"].to_numpy(dtype=float) - target_mid).mean()
        ),
    }


def _slice_table(
    rows: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    group_col: str,
    min_slice_samples: int,
) -> dict[str, Any]:
    if group_col not in rows.columns:
        return {}
    out: dict[str, Any] = {}
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    values = rows[group_col].astype(str).to_numpy()
    for group in sorted(set(values.tolist())):
        mask = values == group
        if int((mask & known).sum()) < int(min_slice_samples):
            continue
        out[str(group)] = _interval_summary(rows.loc[mask], lower[mask], upper[mask])
    return out


def _covers(rows: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    covers = np.zeros(len(rows), dtype=bool)
    if not known.any():
        return covers
    pred_low = rows["pred_mu_low"].to_numpy(dtype=float)
    pred_high = rows["pred_mu_high"].to_numpy(dtype=float)
    target_low = rows["target_mu_low"].to_numpy(dtype=float)
    target_high = rows["target_mu_high"].to_numpy(dtype=float)
    covers[known] = (pred_low[known] - lower[known] <= target_low[known]) & (
        pred_high[known] + upper[known] >= target_high[known]
    )
    return covers


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
    baseline = policies["symmetric_hierarchical_quality"]["pooled"]
    baseline_worst = policies["symmetric_hierarchical_quality"]["worst_slice"]
    candidates = []
    for name, row in policies.items():
        if name == "symmetric_hierarchical_quality":
            continue
        pooled = row["pooled"]
        worst = row["worst_slice"]
        cov = _num(pooled.get("calibrated_coverage"))
        width = _num(pooled.get("calibrated_width"))
        worst_cov = _num(worst.get("calibrated_coverage"))
        candidates.append(
            {
                "name": name,
                "coverage": cov,
                "width": width,
                "worst_coverage": worst_cov,
                "width_delta_vs_symmetric": width - _num(baseline.get("calibrated_width")),
                "coverage_delta_vs_symmetric": cov - _num(baseline.get("calibrated_coverage")),
                "worst_delta_vs_symmetric": worst_cov - _num(baseline_worst.get("calibrated_coverage")),
            }
        )
    feasible = [
        row
        for row in candidates
        if row["coverage"] >= report["target_coverage"]
        and row["worst_coverage"] >= max(0.88, _num(baseline_worst.get("calibrated_coverage")) - 0.01)
        and row["width_delta_vs_symmetric"] <= -0.01
    ]
    feasible.sort(key=lambda row: (row["width"], -row["coverage"], -row["worst_coverage"]))
    best = feasible[0] if feasible else None
    if best:
        return {
            "status": "keep_for_more_eval",
            "best": best,
            "reason": "asymmetric calibration preserves coverage while reducing width versus symmetric hierarchical quality",
        }
    return {
        "status": "discard_or_hold",
        "best": _best_by_width_at_coverage(candidates, report["target_coverage"]),
        "reason": "no asymmetric policy clearly beats symmetric hierarchical quality on coverage-width-worst-slice tradeoff",
    }


def _best_by_width_at_coverage(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    feasible = [row for row in rows if row["coverage"] >= target]
    feasible.sort(key=lambda row: (row["width"], -row["worst_coverage"]))
    return feasible[0] if feasible else None


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Asymmetric Mondrian Conformal Evaluation",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Target coverage: `{100.0 * report['target_coverage']:.1f}%`",
        f"- Calibration mu samples: `{report['calibration']['num_mu_samples']}`",
        f"- Test mu samples: `{report['test']['num_mu_samples']}`",
        "",
        "## Policy Ranking",
        "",
        "| policy | raw cov | cal cov | cal width | lower r | upper r | worst slice | worst cov |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    ranked = sorted(
        report["policies"].items(),
        key=lambda item: (
            -_num(item[1]["pooled"].get("calibrated_coverage")),
            _num(item[1]["pooled"].get("calibrated_width")),
        ),
    )
    for name, row in ranked:
        pooled = row["pooled"]
        radius = row["radius"]
        worst = row["worst_slice"]
        lines.append(
            "| {name} | {raw} | {cov} | {width} | {lower} | {upper} | {worst} | {worst_cov} |".format(
                name=name,
                raw=_fmt_pct(pooled.get("raw_coverage")),
                cov=_fmt_pct(pooled.get("calibrated_coverage")),
                width=_fmt_abs(pooled.get("calibrated_width")),
                lower=_fmt_abs(radius.get("mean_lower_radius")),
                upper=_fmt_abs(radius.get("mean_upper_radius")),
                worst=_worst_name(worst),
                worst_cov=_fmt_pct(worst.get("calibrated_coverage")),
            )
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Fast Decision",
            "",
            f"- Status: `{decision['status']}`",
            f"- Reason: {decision['reason']}",
        ]
    )
    if decision.get("best"):
        best = decision["best"]
        lines.append(
            "- Best candidate: `{name}` cov `{cov}` width `{width}` worst `{worst}` width delta `{delta}`".format(
                name=best.get("name"),
                cov=_fmt_pct(best.get("coverage")),
                width=_fmt_abs(best.get("width")),
                worst=_fmt_pct(best.get("worst_coverage")),
                delta=_fmt_signed_abs(best.get("width_delta_vs_symmetric")),
            )
        )
    return "\n".join(lines) + "\n"


def _mean_known(rows: pd.DataFrame, values: np.ndarray) -> float:
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    return float(np.mean(values[known])) if known.any() else 0.0


def _max_known(rows: pd.DataFrame, values: np.ndarray) -> float:
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    return float(np.max(values[known])) if known.any() else 0.0


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


def _fmt_signed_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


def _worst_name(worst: dict[str, Any]) -> str:
    if not worst:
        return "-"
    return f"{worst.get('scope')}::{worst.get('name')}"


if __name__ == "__main__":
    main()

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

from evaluate_quality_slices import attach_quality_flags, build_loader, collect_predictions
from friction_affordance.engine import build_model
from friction_affordance.utils import load_yaml, resolve_device


DEFAULT_QUALITY_CSV = Path("data/quality_flags/image_quality_flags.csv")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast post-hoc test for quality-aware Mondrian conformal calibration. "
            "This does not retrain a model; it checks whether dataset/state/quality "
            "conditional radii can fix weak-friction interval undercoverage."
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

    report, enriched_test = evaluate_policies(
        calib,
        test,
        checkpoint=args.checkpoint,
        config=args.config,
        quality_csv=args.quality_csv,
        target_coverage=float(args.target_coverage),
        min_calibration_samples=int(args.min_calibration_samples),
        min_slice_samples=int(args.min_slice_samples),
    )

    out_dir = args.out_dir or args.checkpoint.parent / "quality_mondrian_conformal"
    out_json = args.out_json or out_dir / "quality_mondrian_conformal.json"
    out_md = args.out_md or out_dir / "quality_mondrian_conformal.md"
    out_predictions = args.out_predictions or out_dir / "predictions_test_with_radii.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    enriched_test.to_csv(out_predictions, index=False, encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote: {out_json}")
    print(f"wrote: {out_predictions}")


def _collect_split(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    split: str,
    quality_csv: Path,
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    print(f"collecting {split} predictions on {device}...", file=sys.stderr, flush=True)
    loader = build_loader(
        cfg,
        split,
        max_samples=max_samples,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    df = collect_predictions(model, loader, device)
    df = attach_quality_flags(df, quality_csv)
    prepared = prepare_records(df)
    quality_join = _mean_bool(prepared, "quality_joined")
    print(
        f"collected {split}: rows={len(prepared)} mu={int(prepared['mu_known_bool'].sum())} "
        f"quality_join={(100.0 * (quality_join or 0.0)):.2f}%",
        file=sys.stderr,
        flush=True,
    )
    return prepared


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mu_known_bool"] = out.get("mu_known", False).map(_to_bool)
    for col in [
        "target_mu_low",
        "target_mu_high",
        "pred_mu_low",
        "pred_mu_high",
        "pred_mu_mean",
        "raw_interval_width",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    known = out["mu_known_bool"]
    score = np.zeros(len(out), dtype=np.float32)
    if known.any():
        pred_low = out.loc[known, "pred_mu_low"].to_numpy(dtype=float)
        pred_high = out.loc[known, "pred_mu_high"].to_numpy(dtype=float)
        target_low = out.loc[known, "target_mu_low"].to_numpy(dtype=float)
        target_high = out.loc[known, "target_mu_high"].to_numpy(dtype=float)
        score[known.to_numpy()] = np.maximum.reduce(
            [pred_low - target_low, target_high - pred_high, np.zeros(len(pred_low))]
        ).astype(np.float32)
    out["conformal_score"] = score
    out["quality_bin"] = [_quality_bin(row) for _, row in out.iterrows()]
    out["risk_key"] = out.get("true_risk_idx", -1).fillna(-1).astype(int).astype(str)
    out["dataset_key"] = out.get("dataset", "unknown").astype(str)
    out["group_key_safe"] = out.get("group_key", "unknown").astype(str)
    out["dataset_quality_key"] = out["dataset_key"] + "::quality=" + out["quality_bin"].astype(str)
    out["dataset_core_risk_key"] = out["group_key_safe"] + "::risk=" + out["risk_key"]
    out["dataset_core_quality_key"] = out["group_key_safe"] + "::quality=" + out["quality_bin"].astype(str)
    return out


def evaluate_policies(
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
    calib_known = calib[calib["mu_known_bool"]].copy()
    test_known = test[test["mu_known_bool"]].copy()
    global_radius = _conformal_radius(calib_known["conformal_score"].to_numpy(dtype=float), target_coverage)
    tables = {
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

    enriched = test.copy()
    policy_defs = {
        "global": ["global"],
        "dataset_mondrian": ["global", "dataset"],
        "state_mondrian": ["global", "state"],
        "hierarchical_safety": ["global", "dataset", "state", "risk", "dataset_core_risk"],
        "quality_mondrian": ["global", "quality", "dataset_quality"],
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
    policies: dict[str, Any] = {}
    for name, scopes in policy_defs.items():
        radii = _assign_radii(test, global_radius, tables, scopes)
        enriched[f"{name}_radius"] = radii
        enriched[f"{name}_calibrated_low"] = np.clip(enriched["pred_mu_low"].to_numpy(dtype=float) - radii, 0.0, 1.2)
        enriched[f"{name}_calibrated_high"] = np.clip(enriched["pred_mu_high"].to_numpy(dtype=float) + radii, 0.0, 1.2)
        enriched[f"{name}_calibrated_covers"] = _covers(enriched, radii)
        policies[name] = _policy_summary(
            test,
            radii,
            name=name,
            min_slice_samples=min_slice_samples,
        )

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
            "quality_join_rate": _mean_bool(calib, "quality_joined"),
            "global_radius": float(global_radius),
            "num_radii": {name: int(len(table)) for name, table in tables.items()},
        },
        "test": {
            "num_samples": int(len(test)),
            "num_mu_samples": int(len(test_known)),
            "quality_join_rate": _mean_bool(test, "quality_joined"),
        },
        "policies": policies,
        "decision": _decision(policies, quality_join_rate=_mean_bool(test, "quality_joined")),
    }
    return report, enriched


def _assign_radii(
    rows: pd.DataFrame,
    global_radius: float,
    tables: dict[str, dict[str, float]],
    scopes: list[str],
) -> np.ndarray:
    radii = np.full(len(rows), float(global_radius), dtype=np.float32)
    for scope in scopes:
        if scope == "global":
            continue
        table = tables.get(scope, {})
        key_col = _scope_column(scope)
        if not table or key_col not in rows.columns:
            continue
        group_values = rows[key_col].astype(str)
        scope_radii = group_values.map(table).astype(float)
        mask = scope_radii.notna().to_numpy()
        if mask.any():
            radii[mask] = np.maximum(radii[mask], scope_radii[mask].to_numpy(dtype=np.float32))
    return radii


def _scope_column(scope: str) -> str:
    return {
        "dataset": "dataset_key",
        "state": "group_key_safe",
        "risk": "risk_key",
        "quality": "quality_bin",
        "dataset_quality": "dataset_quality_key",
        "dataset_core_risk": "dataset_core_risk_key",
        "dataset_core_quality": "dataset_core_quality_key",
    }[scope]


def _policy_summary(
    rows: pd.DataFrame,
    radii: np.ndarray,
    *,
    name: str,
    min_slice_samples: int,
) -> dict[str, Any]:
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    pooled = _interval_summary(rows, radii)
    slices = {
        "dataset": _slice_table(rows, radii, "dataset_key", min_slice_samples),
        "state": _slice_table(rows, radii, "group_key_safe", min_slice_samples),
        "quality": _slice_table(rows, radii, "quality_bin", min_slice_samples),
        "dataset_quality": _slice_table(rows, radii, "dataset_quality_key", min_slice_samples),
    }
    worst = _worst_slice(slices)
    radius_source = {
        "mean_radius": float(np.nanmean(radii[known])) if known.any() else 0.0,
        "max_radius": float(np.nanmax(radii[known])) if known.any() else 0.0,
    }
    return {
        "policy": name,
        "pooled": pooled,
        "radius": radius_source,
        "slices": slices,
        "worst_slice": worst,
    }


def _interval_summary(rows: pd.DataFrame, radii: np.ndarray) -> dict[str, Any]:
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    if not known.any():
        return {"num_samples": 0}
    sub = rows.loc[known]
    r = radii[known]
    raw_covers = sub.get("raw_interval_covers", False).map(_to_bool).to_numpy()
    calibrated_covers = _covers(sub, r)
    raw_width = sub["raw_interval_width"].to_numpy(dtype=float)
    target_mid = 0.5 * (
        sub["target_mu_low"].to_numpy(dtype=float) + sub["target_mu_high"].to_numpy(dtype=float)
    )
    return {
        "num_samples": int(len(sub)),
        "raw_coverage": float(np.mean(raw_covers)),
        "raw_width": float(np.mean(raw_width)),
        "calibrated_coverage": float(np.mean(calibrated_covers)),
        "calibrated_width": float(np.mean(np.clip(raw_width + 2.0 * r, 0.0, 1.2))),
        "mean_radius": float(np.mean(r)),
        "mean_mae_to_interval_mid": float(
            np.abs(sub["pred_mu_mean"].to_numpy(dtype=float) - target_mid).mean()
        ),
    }


def _slice_table(
    rows: pd.DataFrame,
    radii: np.ndarray,
    group_col: str,
    min_slice_samples: int,
) -> dict[str, Any]:
    if group_col not in rows.columns:
        return {}
    out: dict[str, Any] = {}
    for group in sorted(rows[group_col].dropna().astype(str).unique().tolist()):
        mask = rows[group_col].astype(str).to_numpy() == group
        known_count = int((mask & rows["mu_known_bool"].to_numpy(dtype=bool)).sum())
        if known_count < int(min_slice_samples):
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


def _covers(rows: pd.DataFrame, radii: np.ndarray) -> np.ndarray:
    known = rows["mu_known_bool"].to_numpy(dtype=bool)
    covers = np.zeros(len(rows), dtype=bool)
    if not known.any():
        return covers
    pred_low = rows["pred_mu_low"].to_numpy(dtype=float)
    pred_high = rows["pred_mu_high"].to_numpy(dtype=float)
    target_low = rows["target_mu_low"].to_numpy(dtype=float)
    target_high = rows["target_mu_high"].to_numpy(dtype=float)
    covers[known] = (pred_low[known] - radii[known] <= target_low[known]) & (
        pred_high[known] + radii[known] >= target_high[known]
    )
    return covers


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


def _decision(policies: dict[str, Any], *, quality_join_rate: float | None) -> dict[str, Any]:
    global_row = policies.get("global", {})
    hierarchical = policies.get("hierarchical_safety", {})
    quality = policies.get("hierarchical_quality_safety", {})
    global_pooled = global_row.get("pooled", {})
    hierarchical_pooled = hierarchical.get("pooled", {})
    quality_pooled = quality.get("pooled", {})
    global_worst = global_row.get("worst_slice", {})
    hierarchical_worst = hierarchical.get("worst_slice", {})
    quality_worst = quality.get("worst_slice", {})
    reasons: list[str] = []
    if (quality_join_rate or 0.0) < 0.80:
        reasons.append("quality flags do not cover enough evaluation images")
    worst_gain_vs_hier = _num(quality_worst.get("calibrated_coverage")) - _num(
        hierarchical_worst.get("calibrated_coverage")
    )
    pooled_gain_vs_global = _num(quality_pooled.get("calibrated_coverage")) - _num(
        global_pooled.get("calibrated_coverage")
    )
    width_delta_vs_hier = _num(quality_pooled.get("calibrated_width")) - _num(
        hierarchical_pooled.get("calibrated_width")
    )
    if worst_gain_vs_hier < 0.02 and pooled_gain_vs_global < 0.01:
        reasons.append("quality-aware radii do not materially improve coverage")
    if width_delta_vs_hier > 0.08:
        reasons.append("quality-aware radii widen intervals too much")
    status = "keep_for_full_eval" if not reasons else "discard_or_hold"
    return {
        "status": status,
        "rule": (
            "keep only if hierarchical_quality_safety improves worst-slice coverage "
            "or pooled coverage with <=0.08 extra calibrated width versus hierarchical_safety"
        ),
        "pooled_gain_vs_global": pooled_gain_vs_global,
        "worst_gain_vs_hierarchical": worst_gain_vs_hier,
        "width_delta_vs_hierarchical": width_delta_vs_hier,
        "global_worst_slice": global_worst,
        "hierarchical_worst_slice": hierarchical_worst,
        "quality_worst_slice": quality_worst,
        "reasons": reasons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Quality-Mondrian Conformal Evaluation",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Target coverage: `{100.0 * report['target_coverage']:.1f}%`",
        f"- Calibration mu samples: `{report['calibration']['num_mu_samples']}`",
        f"- Test mu samples: `{report['test']['num_mu_samples']}`",
        f"- Quality join rate: `{100.0 * (report['test'].get('quality_join_rate') or 0.0):.2f}%`",
        "",
        "## Policy Ranking",
        "",
        "| policy | raw cov | cal cov | cal width | mean radius | worst slice | worst cov | decision note |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for name, row in report["policies"].items():
        pooled = row.get("pooled", {})
        radius = row.get("radius", {})
        worst = row.get("worst_slice", {})
        lines.append(
            "| {name} | {raw} | {cov} | {width} | {radius} | {worst_name} | {worst_cov} | {note} |".format(
                name=name,
                raw=_fmt_pct(pooled.get("raw_coverage")),
                cov=_fmt_pct(pooled.get("calibrated_coverage")),
                width=_fmt_abs(pooled.get("calibrated_width")),
                radius=_fmt_abs(radius.get("mean_radius")),
                worst_name=_worst_name(worst),
                worst_cov=_fmt_pct(worst.get("calibrated_coverage")),
                note=_policy_note(report, name),
            )
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Fast Decision",
            "",
            f"- Status: `{decision['status']}`",
            f"- Pooled gain vs global: `{_fmt_signed_pct(decision.get('pooled_gain_vs_global'))}`",
            f"- Worst-slice gain vs hierarchical: `{_fmt_signed_pct(decision.get('worst_gain_vs_hierarchical'))}`",
            f"- Width delta vs hierarchical: `{_fmt_signed_abs(decision.get('width_delta_vs_hierarchical'))}`",
        ]
    )
    if decision.get("reasons"):
        lines.append("- Reasons: " + "; ".join(str(item) for item in decision["reasons"]))
    else:
        lines.append("- Reasons: passes the post-hoc promotion gate; run full evaluation when GPU is idle.")
    return "\n".join(lines) + "\n"


def _policy_note(report: dict[str, Any], name: str) -> str:
    if name == "hierarchical_quality_safety":
        return report["decision"]["status"]
    return ""


def _worst_name(worst: dict[str, Any]) -> str:
    if not worst:
        return "-"
    return f"{worst.get('scope')}::{worst.get('name')}"


def _quality_bin(row: pd.Series) -> str:
    if not _to_bool(row.get("quality_joined", False)):
        return "quality_unknown"
    checks = [
        ("near_white", row.get("near_white_flag", False)),
        ("overexposed", row.get("overexposed_flag", False)),
        ("low_texture", row.get("low_texture_flag", False)),
        ("low_contrast", row.get("low_contrast_flag", False)),
        ("dark", row.get("dark_flag", False)),
        ("suspicious", row.get("suspicious_quality_flag", False)),
    ]
    for name, value in checks:
        if _to_bool(value):
            return name
    specular = _safe_float(row.get("specular_highlight_frac"))
    if specular is not None and specular >= 0.02:
        return "specular"
    return "normal"


def _mean_bool(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or df.empty:
        return None
    return float(df[col].map(_to_bool).mean())


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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _fmt_signed_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):+.2f}%"


def _fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_signed_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


if __name__ == "__main__":
    main()

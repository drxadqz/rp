from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")

RISK_NAMES = {
    "0": "very_low",
    "1": "low",
    "2": "medium",
    "3": "high",
    "4": "very_high",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--raw-watch-threshold", type=float, default=0.70)
    parser.add_argument("--calibrated-watch-threshold", type=float, default=0.88)
    parser.add_argument("--min-group-samples", type=int, default=50)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "interval_quality_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "interval_quality_report.json")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY_DIR / "interval_quality_cells.csv")
    args = parser.parse_args()

    report = build_report(
        args.root,
        target_coverage=float(args.target_coverage),
        raw_watch_threshold=float(args.raw_watch_threshold),
        calibrated_watch_threshold=float(args.calibrated_watch_threshold),
        min_group_samples=int(args.min_group_samples),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out_csv, report["cells"])
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(
    root: Path,
    *,
    target_coverage: float,
    raw_watch_threshold: float,
    calibrated_watch_threshold: float,
    min_group_samples: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        calib_path = run_dir / "interval_calibration_90.json"
        if not calib_path.exists():
            continue
        payload = load_json(calib_path)
        if not isinstance(payload, dict):
            continue
        row = summarize_run(run_dir.name, run_dir, payload)
        rows.append(row)
        run_cells = collect_cells(run_dir.name, payload)
        cells.extend(run_cells)
        for cell in run_cells:
            if int(cell.get("num_samples") or 0) < min_group_samples:
                continue
            raw_cov = as_float(cell.get("raw_coverage"))
            cal_cov = as_float(cell.get("calibrated_coverage"))
            if raw_cov is not None and raw_cov < raw_watch_threshold:
                watchlist.append(
                    {
                        **cell,
                        "reason": "raw_undercoverage",
                        "threshold": raw_watch_threshold,
                        "gap": raw_watch_threshold - raw_cov,
                    }
                )
            if cal_cov is not None and cal_cov < calibrated_watch_threshold:
                watchlist.append(
                    {
                        **cell,
                        "reason": "calibrated_undercoverage",
                        "threshold": calibrated_watch_threshold,
                        "gap": calibrated_watch_threshold - cal_cov,
                    }
                )

    rows.sort(key=lambda item: item["run"])
    cells.sort(key=lambda item: (item["run"], item["scope"], item["group"]))
    watchlist.sort(key=lambda item: (float(item.get("gap") or 0.0), int(item.get("num_samples") or 0)), reverse=True)
    return {
        "root": str(root),
        "target_coverage": target_coverage,
        "raw_watch_threshold": raw_watch_threshold,
        "calibrated_watch_threshold": calibrated_watch_threshold,
        "min_group_samples": min_group_samples,
        "num_runs": len(rows),
        "num_cells": len(cells),
        "num_watchlist_items": len(watchlist),
        "rows": rows,
        "cells": cells,
        "watchlist": watchlist,
    }


def summarize_run(run: str, run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    test = payload.get("test_split", {})
    dataset = payload.get("dataset_conditional_test", {}).get("_pooled", {})
    core = payload.get("dataset_core_conditional_test", {}).get("_pooled", {})
    risk = payload.get("risk_conditional_test", {}).get("_pooled", {})
    hierarchy = payload.get("hierarchical_conditional_test", {}).get("pooled", {})
    cells = collect_cells(run, payload)
    calibrated_cells = [cell for cell in cells if cell.get("calibrated_coverage") is not None]
    raw_cells = [cell for cell in cells if cell.get("raw_coverage") is not None]
    return {
        "run": run,
        "output_dir": str(run_dir),
        "num_samples": test.get("num_samples"),
        "raw_coverage": test.get("raw_coverage"),
        "raw_width": test.get("raw_width"),
        "pooled_calibrated_coverage": test.get("calibrated_coverage"),
        "pooled_calibrated_width": test.get("calibrated_width"),
        "dataset_calibrated_coverage": dataset.get("calibrated_coverage"),
        "dataset_calibrated_width": dataset.get("calibrated_width"),
        "dataset_core_calibrated_coverage": core.get("calibrated_coverage"),
        "dataset_core_calibrated_width": core.get("calibrated_width"),
        "risk_calibrated_coverage": risk.get("calibrated_coverage"),
        "risk_calibrated_width": risk.get("calibrated_width"),
        "hierarchical_calibrated_coverage": hierarchy.get("calibrated_coverage"),
        "hierarchical_calibrated_width": hierarchy.get("calibrated_width"),
        "hierarchical_mean_radius": hierarchy.get("mean_radius"),
        "worst_raw_cell": min_cell(raw_cells, "raw_coverage"),
        "worst_calibrated_cell": min_cell(calibrated_cells, "calibrated_coverage"),
        "widest_calibrated_cell": max_cell(calibrated_cells, "calibrated_width"),
    }


def collect_cells(run: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope, key in [
        ("dataset", "dataset_conditional_test"),
        ("dataset_core", "dataset_core_conditional_test"),
        ("risk", "risk_conditional_test"),
    ]:
        groups = payload.get(key, {})
        if not isinstance(groups, dict):
            continue
        for group, stats in groups.items():
            if str(group).startswith("_") or not isinstance(stats, dict):
                continue
            group_label = RISK_NAMES.get(str(group), str(group)) if scope == "risk" else str(group)
            out.append(
                {
                    "run": run,
                    "scope": scope,
                    "group": str(group),
                    "group_label": group_label,
                    "num_samples": stats.get("num_samples"),
                    "raw_coverage": stats.get("raw_coverage"),
                    "raw_width": stats.get("raw_width"),
                    "calibrated_coverage": stats.get("calibrated_coverage"),
                    "calibrated_width": stats.get("calibrated_width"),
                    "conformal_radius": stats.get("conformal_radius"),
                    "calibration_samples": stats.get("calibration_samples"),
                    "used_group_radius": stats.get("used_group_radius"),
                    "mean_mae_to_interval_mid": stats.get("mean_mae_to_interval_mid"),
                }
            )
    hierarchy = payload.get("hierarchical_conditional_test", {})
    if isinstance(hierarchy, dict):
        for scope, key in [
            ("hier_dataset", "dataset"),
            ("hier_dataset_core", "dataset_core"),
            ("hier_risk", "risk"),
            ("hier_dataset_core_risk", "dataset_core_risk"),
        ]:
            groups = hierarchy.get(key, {})
            if not isinstance(groups, dict):
                continue
            for group, stats in groups.items():
                if str(group).startswith("_") or not isinstance(stats, dict):
                    continue
                group_label = RISK_NAMES.get(str(group), str(group)) if scope == "hier_risk" else str(group)
                out.append(
                    {
                        "run": run,
                        "scope": scope,
                        "group": str(group),
                        "group_label": group_label,
                        "num_samples": stats.get("num_samples"),
                        "raw_coverage": stats.get("raw_coverage"),
                        "raw_width": stats.get("raw_width"),
                        "calibrated_coverage": stats.get("calibrated_coverage"),
                        "calibrated_width": stats.get("calibrated_width"),
                        "conformal_radius": stats.get("mean_radius"),
                        "calibration_samples": None,
                        "used_group_radius": "hierarchical_safety",
                        "mean_mae_to_interval_mid": stats.get("mean_mae_to_interval_mid"),
                    }
                )
    return out


def min_cell(cells: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    present = [cell for cell in cells if as_float(cell.get(key)) is not None]
    if not present:
        return None
    cell = min(present, key=lambda item: float(item[key]))
    return compact_cell(cell, key)


def max_cell(cells: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    present = [cell for cell in cells if as_float(cell.get(key)) is not None]
    if not present:
        return None
    cell = max(present, key=lambda item: float(item[key]))
    return compact_cell(cell, key)


def compact_cell(cell: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "scope": cell.get("scope"),
        "group": cell.get("group"),
        "group_label": cell.get("group_label"),
        key: cell.get(key),
        "num_samples": cell.get("num_samples"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Interval Quality Report",
        "",
        f"Root: `{report['root']}`",
        f"Runs with interval calibration: {report['num_runs']}",
        f"Target coverage: `{report['target_coverage']:.2f}`",
        "",
        "This report separates raw interval quality from conformal calibrated interval quality. It is intended to catch cases where coverage improves only because intervals become too wide.",
        "",
        "## Run Summary",
        "",
        "| Run | raw cov/width | pooled cov/width | dataset cov/width | dataset::state cov/width | risk cov/width | hierarchy cov/width | worst raw cell | worst calibrated cell | widest calibrated cell |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
                "| {run} | {raw} | {pooled} | {dataset} | {core} | {risk} | {hierarchy} | {worst_raw} | {worst_cal} | {widest} |".format(
                run=row["run"],
                raw=cov_width(row.get("raw_coverage"), row.get("raw_width")),
                pooled=cov_width(row.get("pooled_calibrated_coverage"), row.get("pooled_calibrated_width")),
                dataset=cov_width(row.get("dataset_calibrated_coverage"), row.get("dataset_calibrated_width")),
                core=cov_width(row.get("dataset_core_calibrated_coverage"), row.get("dataset_core_calibrated_width")),
                risk=cov_width(row.get("risk_calibrated_coverage"), row.get("risk_calibrated_width")),
                hierarchy=cov_width(row.get("hierarchical_calibrated_coverage"), row.get("hierarchical_calibrated_width")),
                worst_raw=cell_label(row.get("worst_raw_cell"), "raw_coverage"),
                worst_cal=cell_label(row.get("worst_calibrated_cell"), "calibrated_coverage"),
                widest=cell_label(row.get("widest_calibrated_cell"), "calibrated_width", percent=False),
            )
        )
    lines.extend(["", "## Undercoverage Watchlist", ""])
    watch = report.get("watchlist", [])[:40]
    if not watch:
        lines.append("- No interval cells crossed the configured undercoverage watch thresholds.")
    else:
        lines.append("| Run | Scope | Group | Reason | samples | raw cov | calibrated cov | calibrated width | gap |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for item in watch:
            lines.append(
                "| {run} | {scope} | {group} | {reason} | {n} | {raw_cov} | {cal_cov} | {cal_width} | {gap} |".format(
                    run=item.get("run"),
                    scope=item.get("scope"),
                    group=item.get("group_label"),
                    reason=item.get("reason"),
                    n=item.get("num_samples"),
                    raw_cov=fmt_pct(item.get("raw_coverage")),
                    cal_cov=fmt_pct(item.get("calibrated_coverage")),
                    cal_width=fmt_abs(item.get("calibrated_width")),
                    gap=fmt_pct(item.get("gap")),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- Prefer methods that improve raw coverage without a large calibrated-width penalty.",
            "- Use hierarchy calibration as a conservative safety policy only when its conditional coverage gain justifies the width cost.",
            "- Treat low RoadSaW or wet-state raw coverage as a P3 target, because it is the likely deployment-sensitive failure mode.",
            "- Treat calibrated coverage below the target band as a reliability problem, and calibrated width above competing rows as an informativeness problem.",
            "",
            "Full cell data are written to `interval_quality_cells.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def cell_label(cell: Any, key: str, *, percent: bool = True) -> str:
    if not isinstance(cell, dict):
        return "-"
    value = cell.get(key)
    value_text = fmt_pct(value) if percent else fmt_abs(value)
    return f"{cell.get('scope')}:{cell.get('group_label')} ({value_text}, n={cell.get('num_samples')})"


def cov_width(coverage: Any, width: Any) -> str:
    if coverage is None or width is None:
        return "-"
    return f"{fmt_pct(coverage)} / {fmt_abs(width)}"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

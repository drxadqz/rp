from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from friction_affordance.ontology import WETNESS
from paper_protocol_progress import ROWS


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY / "wetness_state_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY / "wetness_state_report.json")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY / "wetness_state_report.csv")
    args = parser.parse_args()

    report = build_report(args.root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    write_csv(report, args.out_csv)
    print(render_markdown(report))


def build_report(root: Path) -> dict[str, Any]:
    rows = []
    for run in ROWS:
        row = inspect_run(root / run)
        if row:
            rows.append(row)
    watchlist = [
        row
        for row in rows
        if row.get("status") == "complete"
        and (
            _num(row.get("roadsaw_wetness_macro_f1"), 1.0) < 0.70
            or _num(row.get("roadsaw_ordinal_mae"), 0.0) > 0.40
            or _num(row.get("roadsaw_severe_misorder_rate"), 0.0) > 0.08
        )
    ]
    watchlist.sort(
        key=lambda item: (
            _num(item.get("roadsaw_wetness_macro_f1"), 1.0),
            -_num(item.get("roadsaw_ordinal_mae"), 0.0),
        )
    )
    return {
        "root": str(root),
        "wetness_order": WETNESS,
        "rows": rows,
        "watchlist": watchlist,
        "num_complete": sum(1 for row in rows if row.get("status") == "complete"),
        "num_watchlist": len(watchlist),
    }


def inspect_run(run_dir: Path) -> dict[str, Any] | None:
    detailed = run_dir / "detailed_test.json"
    if not detailed.exists():
        return {"run": run_dir.name, "status": "missing_detailed_test"}
    payload = json.loads(detailed.read_text(encoding="utf-8"))
    wetness = (payload.get("tasks") or {}).get("wetness")
    if not isinstance(wetness, dict):
        return {"run": run_dir.name, "status": "missing_wetness_task"}
    overall_ord = ordinal_confusion_stats(wetness)
    roadsaw = (wetness.get("by_dataset") or {}).get("roadsaw")
    roadsaw_ord = ordinal_confusion_stats(roadsaw) if isinstance(roadsaw, dict) else {}
    row = {
        "run": run_dir.name,
        "status": "complete",
        "wetness_macro_f1": wetness.get("macro_f1"),
        "wetness_accuracy": wetness.get("accuracy"),
        "overall_ordinal_mae": overall_ord.get("ordinal_mae"),
        "overall_severe_misorder_rate": overall_ord.get("severe_misorder_rate"),
        "overall_worst_pair": overall_ord.get("worst_pair"),
    }
    if isinstance(roadsaw, dict):
        row.update(
            {
                "roadsaw_wetness_macro_f1": roadsaw.get("macro_f1"),
                "roadsaw_wetness_accuracy": roadsaw.get("accuracy"),
                "roadsaw_ordinal_mae": roadsaw_ord.get("ordinal_mae"),
                "roadsaw_severe_misorder_rate": roadsaw_ord.get("severe_misorder_rate"),
                "roadsaw_worst_pair": roadsaw_ord.get("worst_pair"),
            }
        )
        per_class = roadsaw.get("per_class_f1") or {}
        for label in WETNESS:
            row[f"roadsaw_f1_{label}"] = per_class.get(label)
    else:
        row.update(
            {
                "roadsaw_wetness_macro_f1": None,
                "roadsaw_wetness_accuracy": None,
                "roadsaw_ordinal_mae": None,
                "roadsaw_severe_misorder_rate": None,
                "roadsaw_worst_pair": None,
            }
        )
    return row


def ordinal_confusion_stats(task: dict[str, Any] | None) -> dict[str, Any]:
    if not task:
        return {}
    matrix = task.get("confusion_matrix")
    labels = task.get("confusion_matrix_labels")
    if not matrix or not labels:
        return {}
    order = {label: idx for idx, label in enumerate(WETNESS)}
    total = 0
    weighted_distance = 0.0
    severe = 0
    worst_pair: dict[str, Any] | None = None
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(matrix[i][j])
            if count <= 0:
                continue
            total += count
            distance = abs(order.get(str(true_label), i) - order.get(str(pred_label), j))
            weighted_distance += float(distance * count)
            if distance >= 2:
                severe += count
            if i != j and (worst_pair is None or count > int(worst_pair["count"])):
                worst_pair = {
                    "true": str(true_label),
                    "pred": str(pred_label),
                    "count": count,
                    "ordinal_distance": int(distance),
                }
    if total <= 0:
        return {}
    return {
        "ordinal_mae": weighted_distance / float(total),
        "severe_misorder_rate": severe / float(total),
        "worst_pair": worst_pair,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Wetness State Report",
        "",
        f"Root: `{report['root']}`",
        f"Wetness order: `{' -> '.join(report['wetness_order'])}`",
        f"Completed rows: `{report['num_complete']}`; watchlist rows: `{report['num_watchlist']}`.",
        "",
        "This report tracks the RoadSaW-sensitive wetness scale separately from the mixed friction task.",
        "",
        "## Run Summary",
        "",
        "| Run | Status | RoadSaW wet F1 | dry | damp | wet | very_wet | water | ordinal MAE | severe misorder | worst pair |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows", []):
        if row.get("status") != "complete":
            lines.append(f"| {row.get('run')} | {row.get('status')} | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| {run} | complete | {macro} | {dry} | {damp} | {wet} | {very_wet} | {water} | {mae} | {severe} | {pair} |".format(
                run=row.get("run"),
                macro=_fmt_pct(row.get("roadsaw_wetness_macro_f1")),
                dry=_fmt_pct(row.get("roadsaw_f1_dry")),
                damp=_fmt_pct(row.get("roadsaw_f1_damp")),
                wet=_fmt_pct(row.get("roadsaw_f1_wet")),
                very_wet=_fmt_pct(row.get("roadsaw_f1_very_wet")),
                water=_fmt_pct(row.get("roadsaw_f1_water")),
                mae=_fmt_abs(row.get("roadsaw_ordinal_mae")),
                severe=_fmt_pct(row.get("roadsaw_severe_misorder_rate")),
                pair=_pair(row.get("roadsaw_worst_pair")),
            )
        )
    lines.extend(
        [
            "",
            "## Watchlist",
            "",
        ]
    )
    if not report.get("watchlist"):
        lines.append("- No completed run crosses the wetness watch thresholds.")
    else:
        for row in report.get("watchlist", [])[:12]:
            lines.append(
                "- `{run}`: RoadSaW wetness F1 `{macro}`, ordinal MAE `{mae}`, severe misorder `{severe}`, worst pair `{pair}`.".format(
                    run=row.get("run"),
                    macro=_fmt_pct(row.get("roadsaw_wetness_macro_f1")),
                    mae=_fmt_abs(row.get("roadsaw_ordinal_mae")),
                    severe=_fmt_pct(row.get("roadsaw_severe_misorder_rate")),
                    pair=_pair(row.get("roadsaw_worst_pair")),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- Prefer candidates that improve RoadSaW damp/wet/very_wet F1 and reduce ordinal MAE.",
            "- Do not keep a wetness-specific module if it only improves dry accuracy while hurting wet or very_wet states.",
            "- Use this report together with low-friction recall, interval quality, LODO, and dataset-ID diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = report.get("rows", [])
    fields = [
        "run",
        "status",
        "wetness_macro_f1",
        "wetness_accuracy",
        "roadsaw_wetness_macro_f1",
        "roadsaw_wetness_accuracy",
        "roadsaw_f1_dry",
        "roadsaw_f1_damp",
        "roadsaw_f1_wet",
        "roadsaw_f1_very_wet",
        "roadsaw_f1_water",
        "roadsaw_ordinal_mae",
        "roadsaw_severe_misorder_rate",
        "roadsaw_worst_pair",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            out["roadsaw_worst_pair"] = json.dumps(out.get("roadsaw_worst_pair"), ensure_ascii=False)
            writer.writerow(out)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _pair(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    return f"{value.get('true')}->{value.get('pred')} (n={value.get('count')})"


def _num(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()

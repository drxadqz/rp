from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
CORE_ORDER = [
    "Global-only",
    "+ PhysicsTexture",
    "+ FrictionSet",
    "+ DG losses",
    "+ EvidenceField aux",
    "Full model",
]
METRIC_COLUMNS = [
    ("friction_macro_f1", "friction F1", True),
    ("risk_macro_f1", "risk F1", True),
    ("low_friction_recall", "low-friction recall", True),
    ("calibrated_coverage", "calibrated coverage", True),
    ("worst_dataset_f1", "worst dataset F1", True),
]
EXTRA_COLUMNS = [
    ("raw_interval_coverage", "raw coverage", True),
    ("calibrated_width", "calibrated width", False),
    ("dataset_id_balanced_accuracy", "dataset-ID bal acc", True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "paper_p0_ablation_table.md")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY_DIR / "paper_p0_ablation_table.csv")
    parser.add_argument("--out-tex", type=Path, default=DEFAULT_SUMMARY_DIR / "paper_p0_ablation_table.tex")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "paper_p0_ablation_table.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out_csv, report["rows"])
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_tex.write_text(render_latex(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    ablation = {row.get("method"): row for row in summary.get("ablation", []) if isinstance(row, dict)}
    rows = []
    for method in CORE_ORDER:
        source = dict(ablation.get(method, {"method": method, "status": "missing"}))
        row = {
            "method": method,
            "status": source.get("status"),
            "epoch": source.get("epoch"),
        }
        for key, label, percent in METRIC_COLUMNS + EXTRA_COLUMNS:
            row[key] = _num(source.get(key))
            row[f"{key}_ci_low"] = _num(source.get(f"{key}_ci_low"))
            row[f"{key}_ci_high"] = _num(source.get(f"{key}_ci_high"))
            row[f"{key}_paper"] = _fmt_metric(source, key, percent)
            row[f"{key}_label"] = label
        rows.append(row)

    complete_rows = [row for row in rows if row.get("status") == "complete"]
    best_by = {}
    for key, label, _ in METRIC_COLUMNS:
        valid = [row for row in complete_rows if isinstance(row.get(key), (int, float))]
        if valid:
            best = max(valid, key=lambda row: float(row[key]))
            best_by[key] = {
                "label": label,
                "method": best.get("method"),
                "value": best.get(key),
            }
    return {
        "summary_dir": str(summary_dir),
        "status": "complete" if len(complete_rows) == len(CORE_ORDER) else "incomplete",
        "claim_boundary": (
            "This table is the local paper-protocol P0 ablation on the public weak-label friction-affordance "
            "setup. It is not a direct measured tire-road friction benchmark."
        ),
        "rows": rows,
        "best_by_metric": best_by,
        "columns": [{"key": key, "label": label, "percent": percent} for key, label, percent in METRIC_COLUMNS],
        "extra_columns": [{"key": key, "label": label, "percent": percent} for key, label, percent in EXTRA_COLUMNS],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper P0 Ablation Table",
        "",
        f"Status: `{report['status']}`",
        "",
        report["claim_boundary"],
        "",
        "## Main Table",
        "",
        "| Method | friction F1 | risk F1 | low-friction recall | calibrated coverage | worst dataset F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {method} | {friction} | {risk} | {low} | {coverage} | {worst} |".format(
                method=row.get("method"),
                friction=row.get("friction_macro_f1_paper"),
                risk=row.get("risk_macro_f1_paper"),
                low=row.get("low_friction_recall_paper"),
                coverage=row.get("calibrated_coverage_paper"),
                worst=row.get("worst_dataset_f1_paper"),
            )
        )
    lines.extend(
        [
            "",
            "## Reviewer Diagnostics",
            "",
            "| Method | raw coverage | calibrated width | dataset-ID bal acc | epoch |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| {method} | {raw} | {width} | {domain} | {epoch} |".format(
                method=row.get("method"),
                raw=row.get("raw_interval_coverage_paper"),
                width=row.get("calibrated_width_paper"),
                domain=row.get("dataset_id_balanced_accuracy_paper"),
                epoch=row.get("epoch") or "-",
            )
        )
    lines.extend(["", "## Best Metric Owners", ""])
    for key, item in report["best_by_metric"].items():
        value = _fmt_value(item.get("value"), percent=True)
        lines.append(f"- {item.get('label')}: `{item.get('method')}` ({value}).")
    lines.append("")
    return "\n".join(lines)


def render_latex(report: dict[str, Any]) -> str:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & friction F1 & risk F1 & low-friction recall & calibrated coverage & worst dataset F1 \\\\",
        "\\midrule",
    ]
    for row in report["rows"]:
        cells = [
            _latex_escape(str(row.get("method"))),
            _latex_metric(row.get("friction_macro_f1_paper")),
            _latex_metric(row.get("risk_macro_f1_paper")),
            _latex_metric(row.get("low_friction_recall_paper")),
            _latex_metric(row.get("calibrated_coverage_paper")),
            _latex_metric(row.get("worst_dataset_f1_paper")),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "method",
        "status",
        "epoch",
        "friction_macro_f1",
        "risk_macro_f1",
        "low_friction_recall",
        "calibrated_coverage",
        "worst_dataset_f1",
        "raw_interval_coverage",
        "calibrated_width",
        "dataset_id_balanced_accuracy",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _fmt_metric(row: dict[str, Any], key: str, percent: bool) -> str:
    value = _num(row.get(key))
    if value is None:
        return "-"
    low = _num(row.get(f"{key}_ci_low"))
    high = _num(row.get(f"{key}_ci_high"))
    point = _fmt_value(value, percent)
    if low is None or high is None:
        return point
    return f"{point} [{_fmt_value(low, percent)}, {_fmt_value(high, percent)}]"


def _fmt_value(value: Any, percent: bool) -> str:
    number = _num(value)
    if number is None:
        return "-"
    if percent:
        return f"{100.0 * number:.2f}"
    return f"{number:.4f}"


def _latex_metric(value: Any) -> str:
    text = str(value or "-")
    return text.replace("[", "\\scriptsize{[").replace("]", "]}")


def _latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")

P0_COLUMNS = [
    "Method",
    "Status",
    "friction F1",
    "risk F1",
    "low-friction recall",
    "calibrated coverage",
    "worst dataset F1",
]
LODO_COLUMNS = [
    "Held-out dataset",
    "Status",
    "friction F1",
    "risk F1",
    "low-friction recall",
    "calibrated coverage",
    "calibrated width",
]
FAIR_COLUMNS = [
    "Dataset",
    "Status",
    "FAF risk F1",
    "ConvNeXt risk F1",
    "delta risk F1",
    "FAF friction F1",
    "ConvNeXt friction F1",
    "delta friction F1",
    "delta calibrated coverage",
]
FINAL_COLUMNS = [
    "Dataset",
    "Status",
    "final risk F1",
    "ConvNeXt risk F1",
    "delta risk F1",
    "final friction F1",
    "ConvNeXt friction F1",
    "delta friction F1",
    "delta calibrated coverage",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    args = parser.parse_args()

    summary = _load_json(args.summary_dir / "paper_protocol_summary.json") or {}
    payload = build_tables(summary)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    _write_table(args.summary_dir / "required_p0_ablation_table", P0_COLUMNS, payload["p0"])
    _write_table(args.summary_dir / "required_lodo_table", LODO_COLUMNS, payload["lodo"])
    _write_table(args.summary_dir / "required_fair_comparison_table", FAIR_COLUMNS, payload["fair"])
    _write_table(args.summary_dir / "required_final_method_table", FINAL_COLUMNS, payload["final"])
    _write_index(args.summary_dir / "required_result_tables.md", payload)
    print((args.summary_dir / "required_result_tables.md").read_text(encoding="utf-8"))


def build_tables(summary: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    return {
        "p0": [_p0_row(row) for row in summary.get("core_ablation", [])],
        "lodo": [_lodo_row(row) for row in summary.get("lodo", [])],
        "fair": _fair_rows(
            summary.get("single_dataset", []),
            summary.get("fair_baselines", []),
            summary.get("fair_single_dataset_deltas", []),
            final=False,
        ),
        "final": _fair_rows(
            summary.get("final_single_dataset", []),
            summary.get("fair_baselines", []),
            summary.get("final_fair_single_dataset_deltas", []),
            final=True,
        ),
    }


def _p0_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "Method": str(row.get("method", "-")),
        "Status": str(row.get("status", "-")),
        "friction F1": _pct(row.get("friction_f1") or row.get("friction_macro_f1")),
        "risk F1": _pct(row.get("risk_f1") or row.get("risk_macro_f1")),
        "low-friction recall": _pct(row.get("low_friction_recall")),
        "calibrated coverage": _pct(row.get("calibrated_coverage")),
        "worst dataset F1": _pct(row.get("worst_dataset_f1")),
    }


def _lodo_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "Held-out dataset": str(row.get("method", "-")),
        "Status": str(row.get("status", "-")),
        "friction F1": _pct(row.get("friction_macro_f1") or row.get("friction_f1")),
        "risk F1": _pct(row.get("risk_macro_f1") or row.get("risk_f1")),
        "low-friction recall": _pct(row.get("low_friction_recall")),
        "calibrated coverage": _pct(row.get("calibrated_coverage")),
        "calibrated width": _num(row.get("calibrated_width")),
    }


def _fair_rows(
    faf_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    *,
    final: bool,
) -> list[dict[str, str]]:
    faf_by_dataset = {_dataset_from_method(row.get("method")): row for row in faf_rows}
    baseline_by_dataset = {_dataset_from_method(row.get("method")): row for row in baseline_rows}
    deltas_by_dataset = {str(row.get("dataset", "")).lower(): row for row in delta_rows}
    datasets = ["roadsaw", "rscd", "roadsc"]
    out = []
    for dataset in datasets:
        faf = faf_by_dataset.get(dataset, {})
        baseline = baseline_by_dataset.get(dataset, {})
        delta = deltas_by_dataset.get(dataset, {})
        status = delta.get("status")
        if not status:
            status = "complete" if faf.get("status") == baseline.get("status") == "complete" else "pending"
        if final:
            out.append(
                {
                    "Dataset": _label_dataset(dataset),
                    "Status": str(status),
                    "final risk F1": _pct(faf.get("risk_macro_f1")),
                    "ConvNeXt risk F1": _pct(baseline.get("risk_macro_f1")),
                    "delta risk F1": _signed_pct(delta.get("delta_risk_macro_f1")),
                    "final friction F1": _pct(faf.get("friction_macro_f1")),
                    "ConvNeXt friction F1": _pct(baseline.get("friction_macro_f1")),
                    "delta friction F1": _signed_pct(delta.get("delta_friction_macro_f1")),
                    "delta calibrated coverage": _signed_pct(delta.get("delta_calibrated_coverage")),
                }
            )
        else:
            out.append(
                {
                    "Dataset": _label_dataset(dataset),
                    "Status": str(status),
                    "FAF risk F1": _pct(faf.get("risk_macro_f1")),
                    "ConvNeXt risk F1": _pct(baseline.get("risk_macro_f1")),
                    "delta risk F1": _signed_pct(delta.get("delta_risk_macro_f1")),
                    "FAF friction F1": _pct(faf.get("friction_macro_f1")),
                    "ConvNeXt friction F1": _pct(baseline.get("friction_macro_f1")),
                    "delta friction F1": _signed_pct(delta.get("delta_friction_macro_f1")),
                    "delta calibrated coverage": _signed_pct(delta.get("delta_calibrated_coverage")),
                }
            )
    return out


def _write_table(stem: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    md_path.write_text(_markdown_table(columns, rows), encoding="utf-8")


def _write_index(path: Path, payload: dict[str, list[dict[str, str]]]) -> None:
    lines = [
        "# Required Result Tables",
        "",
        "These tables mirror the requested paper-facing evidence layout. Missing or running rows are left blank instead of reusing stale metrics.",
        "",
        "## P0 Core Ablation",
        "",
        _markdown_table(P0_COLUMNS, payload["p0"]),
        "## Leave-One-Dataset-Out",
        "",
        _markdown_table(LODO_COLUMNS, payload["lodo"]),
        "## Single-Dataset FAF vs ConvNeXt",
        "",
        _markdown_table(FAIR_COLUMNS, payload["fair"]),
        "## Final Method vs ConvNeXt",
        "",
        _markdown_table(FINAL_COLUMNS, payload["final"]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(columns: list[str], rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "-") or "-") for column in columns) + " |")
    if not rows:
        lines.append("| " + " | ".join("-" for _ in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def _dataset_from_method(method: Any) -> str:
    text = str(method or "").lower()
    if "roadsaw" in text:
        return "roadsaw"
    if "rscd" in text:
        return "rscd"
    if "roadsc" in text:
        return "roadsc"
    return text


def _label_dataset(dataset: str) -> str:
    return {"roadsaw": "RoadSaW", "rscd": "RSCD", "roadsc": "RoadSC"}.get(dataset, dataset)


def _pct(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{100.0 * float(value):.2f}"


def _signed_pct(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{100.0 * float(value):+.2f}"


def _num(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "safety_selection_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "safety_selection_report.json")
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir)
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def build_report(root: Path, summary_dir: Path) -> dict[str, Any]:
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    rows = []
    for group in ["ablation", "lodo", "single_dataset", "fair_baselines"]:
        for row in summary.get(group, []):
            out_dir = Path(str(row.get("output_dir", "")))
            if not out_dir.exists():
                continue
            if not (out_dir / "best_safety.pt").exists() and not (out_dir / "safety_selected").exists():
                continue
            rows.append(_row(row, out_dir, group))

    verdict = "pending"
    complete = [row for row in rows if row["status"] == "complete"]
    if complete:
        helpful = [
            row
            for row in complete
            if (_num(row.get("delta_low_friction_recall")) or 0.0) > 0
            or (_num(row.get("delta_raw_interval_coverage")) or 0.0) > 0
        ]
        verdict = "safety_selected_available"
        if helpful:
            verdict = "safety_selected_has_safety_gain"
    return {
        "root": str(root),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "rows": rows,
        "rule": (
            "Safety-selected checkpoints are supplemental. The main ablation table "
            "remains loss-selected unless the protocol is explicitly changed before a new full rerun."
        ),
    }


def _row(main: dict[str, Any], out_dir: Path, group: str) -> dict[str, Any]:
    safety = _load_json(out_dir / "safety_selected" / "safety_selected_summary.json")
    status = "complete" if isinstance(safety, dict) else "pending"
    row = {
        "method": main.get("method"),
        "group": group,
        "output_dir": str(out_dir),
        "status": status,
        "main_friction_f1": main.get("friction_macro_f1"),
        "main_risk_f1": main.get("risk_macro_f1"),
        "main_low_friction_recall": main.get("low_friction_recall"),
        "main_raw_interval_coverage": main.get("raw_interval_coverage"),
        "main_calibrated_coverage": main.get("calibrated_coverage"),
        "main_calibrated_width": main.get("calibrated_width"),
    }
    if isinstance(safety, dict):
        row.update(
            {
                "safety_checkpoint": safety.get("checkpoint"),
                "safety_friction_f1": safety.get("friction_macro_f1"),
                "safety_risk_f1": safety.get("risk_macro_f1"),
                "safety_low_friction_recall": safety.get("low_friction_recall"),
                "safety_raw_interval_coverage": safety.get("raw_interval_coverage"),
                "safety_calibrated_coverage": safety.get("calibrated_coverage"),
                "safety_calibrated_width": safety.get("calibrated_width"),
            }
        )
        for key in [
            "friction_f1",
            "risk_f1",
            "low_friction_recall",
            "raw_interval_coverage",
            "calibrated_coverage",
            "calibrated_width",
        ]:
            row[f"delta_{key}"] = _delta(row.get(f"safety_{key}"), row.get(f"main_{key}"))
    return row


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Safety Selection Report",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        report["rule"],
        "",
        "| Method | Status | d risk F1 | d low recall | d raw cov | d calib cov | d calib width | safety risk F1 | main risk F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not report["rows"]:
        lines.append("| - | pending | - | - | - | - | - | - | - |")
    for row in report["rows"]:
        lines.append(
            "| {method} | {status} | {drisk} | {dlow} | {draw} | {dcov} | {dwidth} | {srisk} | {mrisk} |".format(
                method=row.get("method"),
                status=row.get("status"),
                drisk=_fmt_percent(row.get("delta_risk_f1"), signed=True),
                dlow=_fmt_percent(row.get("delta_low_friction_recall"), signed=True),
                draw=_fmt_percent(row.get("delta_raw_interval_coverage"), signed=True),
                dcov=_fmt_percent(row.get("delta_calibrated_coverage"), signed=True),
                dwidth=_fmt_abs(row.get("delta_calibrated_width"), signed=True),
                srisk=_fmt_percent(row.get("safety_risk_f1")),
                mrisk=_fmt_percent(row.get("main_risk_f1")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _delta(cur: Any, prev: Any) -> float | None:
    cur_num = _num(cur)
    prev_num = _num(prev)
    if cur_num is None or prev_num is None:
        return None
    return cur_num - prev_num


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_percent(value: Any, *, signed: bool = False) -> str:
    num = _num(value)
    if num is None:
        return "-"
    return f"{100.0 * num:+.2f}" if signed else f"{100.0 * num:.2f}"


def _fmt_abs(value: Any, *, signed: bool = False) -> str:
    num = _num(value)
    if num is None:
        return "-"
    return f"{num:+.4f}" if signed else f"{num:.4f}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

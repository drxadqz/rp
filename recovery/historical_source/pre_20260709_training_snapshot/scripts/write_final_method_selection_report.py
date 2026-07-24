from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "final_method_selection_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "final_method_selection_report.json")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY_DIR / "final_method_selection_scores.csv")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    write_scores_csv(args.out_csv, report["scored_rows"])
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    module_decisions = _load_csv(summary_dir / "module_decisions.csv")
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}

    ablation_rows = summary.get("ablation", []) or []
    rows = []
    rows.extend(_tag_rows(ablation_rows[:6], "p0_core"))
    rows.extend(_tag_rows(ablation_rows[6:], "p1_candidate"))
    rows.extend(_tag_rows(summary.get("lodo", []), "lodo"))
    rows.extend(_tag_rows(summary.get("single_dataset", []), "single_dataset_faf"))
    rows.extend(_tag_rows(summary.get("fair_baselines", []), "single_dataset_baseline"))
    rows.extend(_tag_rows(summary.get("final_lodo", []), "final_lodo"))
    rows.extend(_tag_rows(summary.get("final_single_dataset", []), "final_single_dataset"))

    scored_rows = [_score_row(row) for row in rows]
    complete_rows = [row for row in scored_rows if row["status"] == "complete"]
    complete_rows.sort(key=lambda row: row.get("selection_score") if row.get("selection_score") is not None else -1e9, reverse=True)

    requirements = {item.get("name"): item for item in completeness.get("requirements", [])}
    verdict = _verdict(requirements, complete_rows, summary)
    return {
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "selection_rule": selection_rule_text(),
        "requirements": requirements,
        "provisional_top_completed": complete_rows[:8],
        "scored_rows": scored_rows,
        "module_decisions": module_decisions,
        "risk_register": risk_register(requirements, module_decisions, wetness, shortcut, interval),
        "recommended_action": recommended_action(verdict, requirements, module_decisions),
    }


def _tag_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copied = dict(row)
        copied["group"] = group
        out.append(copied)
    return out


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    risk = _num(row.get("risk_macro_f1") or row.get("risk_f1"))
    low = _num(row.get("low_friction_recall"))
    worst = _num(row.get("worst_dataset_f1"))
    raw_cov = _num(row.get("raw_interval_coverage") or row.get("raw_coverage"))
    cal_cov = _num(row.get("calibrated_coverage"))
    cal_width = _num(row.get("calibrated_width"))
    dataset_id = _num(row.get("dataset_id_balanced_accuracy"))

    if row.get("status") != "complete":
        score = None
    else:
        coverage_score = 1.0 - abs((cal_cov if cal_cov is not None else 0.0) - 0.90)
        width_penalty = 0.30 * (cal_width if cal_width is not None else 0.60)
        shortcut_penalty = 0.10 * max(0.0, (dataset_id if dataset_id is not None else 0.85) - 0.85)
        score = (
            0.30 * (risk if risk is not None else 0.0)
            + 0.22 * (low if low is not None else 0.0)
            + 0.18 * (worst if worst is not None else risk if risk is not None else 0.0)
            + 0.12 * (raw_cov if raw_cov is not None else 0.0)
            + 0.12 * coverage_score
            - width_penalty
            - shortcut_penalty
        )

    return {
        "group": row.get("group"),
        "method": row.get("method"),
        "status": row.get("status"),
        "risk_f1": risk,
        "low_friction_recall": low,
        "worst_dataset_f1": worst,
        "raw_interval_coverage": raw_cov,
        "calibrated_coverage": cal_cov,
        "calibrated_width": cal_width,
        "dataset_id_balanced_accuracy": dataset_id,
        "selection_score": score,
    }


def selection_rule_text() -> str:
    return (
        "Select the final method only after P0, LODO, fair single-dataset baselines, "
        "and final-method evidence are complete. Rank candidates by safety and "
        "generalization first: risk F1, low-friction recall, worst-dataset F1, raw "
        "interval coverage, calibrated coverage near 90%, narrower calibrated width, "
        "and lower dataset-ID shortcut. Do not keep a module solely because it improves "
        "pooled accuracy if it hurts held-out RoadSaW, low-friction recall, or interval quality."
    )


def _verdict(
    requirements: dict[str, dict[str, Any]],
    complete_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    if not complete_rows:
        return "pending_no_complete_rows"
    hard = [
        "p0_ablation_complete",
        "lodo_complete",
        "fair_single_dataset_complete",
        "final_method_complete",
    ]
    if any(requirements.get(name, {}).get("status") != "complete" for name in hard):
        return "pending_hard_evidence"
    final_rows = [row for row in summary.get("final_lodo", []) if row.get("status") == "complete"]
    roadsaw = next((row for row in final_rows if "RoadSaW" in str(row.get("method"))), None)
    if not roadsaw:
        return "pending_final_roadsaw"
    if (_num(roadsaw.get("risk_macro_f1")) or 0.0) < 0.55:
        return "reject_current_final_route_roadsaw_weak"
    return "ready_to_select_final_method"


def risk_register(
    requirements: dict[str, dict[str, Any]],
    module_decisions: list[dict[str, Any]],
    wetness: dict[str, Any],
    shortcut: dict[str, Any],
    interval: dict[str, Any],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for name, item in requirements.items():
        if item.get("status") != "complete":
            risks.append(
                {
                    "risk": name,
                    "level": "block",
                    "evidence": "missing: " + ", ".join(item.get("missing", []) or []),
                    "action": "Run the queued experiment group and refresh postprocess reports.",
                }
            )
    for row in module_decisions:
        if row.get("decision") in {"rework_or_remove", "remove_or_rework"}:
            risks.append(
                {
                    "risk": f"module_{row.get('module')}",
                    "level": "warn",
                    "evidence": row.get("reason", ""),
                    "action": "Do not include this module in the final route unless later LODO/fair evidence reverses the decision.",
                }
            )
    if shortcut.get("verdict") == "warn":
        risks.append(
            {
                "risk": "dataset_shortcut",
                "level": "warn",
                "evidence": f"{shortcut.get('num_high_shortcut')} of {shortcut.get('num_complete')} completed rows exceed shortcut threshold.",
                "action": "Prefer candidates that reduce risk/core-state-conditioned dataset-ID balanced accuracy.",
            }
        )
    if int(wetness.get("num_watchlist", 0) or 0) > 0:
        risks.append(
            {
                "risk": "roadsaw_wetness",
                "level": "warn",
                "evidence": f"{wetness.get('num_watchlist')} completed rows are on the RoadSaW wetness watchlist.",
                "action": "Use wetness ordinal loss, wet-state hard sampling, and held-out RoadSaW LODO before claiming robustness.",
            }
        )
    if int(interval.get("num_watchlist_items", 0) or 0) > 0:
        risks.append(
            {
                "risk": "conditional_interval_undercoverage",
                "level": "warn",
                "evidence": f"{interval.get('num_watchlist_items')} conditional cells are undercovered.",
                "action": "Use P3 safety-weighted coverage and conditional calibration; report width together with coverage.",
            }
        )
    return risks


def recommended_action(
    verdict: str,
    requirements: dict[str, dict[str, Any]],
    module_decisions: list[dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    if requirements.get("p0_ablation_complete", {}).get("status") != "complete":
        actions.append("Finish `v5_full_faf` and rerun the full postprocess pipeline to close the P0 table.")
    if requirements.get("lodo_complete", {}).get("status") != "complete":
        actions.append("Run LODO next; inspect held-out RoadSaW before making any OOD claim.")
    if requirements.get("fair_single_dataset_complete", {}).get("status") != "complete":
        actions.append("Run matched single-dataset FAF and ConvNeXt baselines for direct public-dataset comparison.")
    if requirements.get("candidate_path_complete", {}).get("status") != "complete":
        actions.append("Run v6-v25 candidates and compare them with the same safety/generalization score.")
    if requirements.get("final_method_complete", {}).get("status") != "complete":
        actions.append("Run final lean road-ROI safety LODO and single-dataset comparisons after candidate selection.")
    risky_modules = [row.get("module") for row in module_decisions if row.get("decision") in {"rework_or_remove", "remove_or_rework"}]
    if risky_modules:
        actions.append("Treat these modules as provisional removals unless later evidence rescues them: " + ", ".join(str(item) for item in risky_modules) + ".")
    if verdict == "ready_to_select_final_method":
        actions.append("Freeze the highest-ranked final route, regenerate LaTeX tables, and prepare paper figures/failure cases.")
    return actions


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final Method Selection Report",
        "",
        f"Summary dir: `{report['summary_dir']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Selection Rule",
        "",
        report["selection_rule"],
        "",
        "## Provisional Top Completed Rows",
        "",
        "| Rank | Group | Method | Status | score | risk F1 | low recall | worst F1 | raw cov | calib cov | calib width | dataset-ID |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    top = report.get("provisional_top_completed", [])
    if not top:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - |")
    for idx, row in enumerate(top, start=1):
        lines.append(
            "| {idx} | {group} | {method} | {status} | {score} | {risk} | {low} | {worst} | {raw} | {cal} | {width} | {domain} |".format(
                idx=idx,
                group=row.get("group"),
                method=row.get("method"),
                status=row.get("status"),
                score=_fmt_abs(row.get("selection_score")),
                risk=_fmt_pct(row.get("risk_f1")),
                low=_fmt_pct(row.get("low_friction_recall")),
                worst=_fmt_pct(row.get("worst_dataset_f1")),
                raw=_fmt_pct(row.get("raw_interval_coverage")),
                cal=_fmt_pct(row.get("calibrated_coverage")),
                width=_fmt_abs(row.get("calibrated_width")),
                domain=_fmt_pct(row.get("dataset_id_balanced_accuracy")),
            )
        )
    lines.extend(["", "## Module Decisions", ""])
    lines.append("| Module | Decision | Reason |")
    lines.append("|---|---|---|")
    for row in report.get("module_decisions", []):
        lines.append(f"| {row.get('module')} | `{row.get('decision')}` | {row.get('reason', '-')} |")
    lines.extend(["", "## Risk Register", ""])
    lines.append("| Risk | Level | Evidence | Action |")
    lines.append("|---|---|---|---|")
    for row in report.get("risk_register", []):
        lines.append(f"| `{row.get('risk')}` | `{row.get('level')}` | {row.get('evidence')} | {row.get('action')} |")
    lines.extend(["", "## Recommended Actions", ""])
    for item in report.get("recommended_action", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: Any) -> str:
    num = _num(value)
    if num is None:
        return "-"
    return f"{100.0 * num:.2f}"


def _fmt_abs(value: Any) -> str:
    num = _num(value)
    if num is None:
        return "-"
    return f"{num:.4f}"


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()

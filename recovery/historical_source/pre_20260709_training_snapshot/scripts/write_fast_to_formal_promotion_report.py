from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")

ANCHOR = "v1_physics_texture"
FALLBACK_LEAN = [
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v15_lean_bottom_square_style_safety",
    "v14_lean_road_roi_safety",
    "v13_lean_physics_evidence",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-promotions", type=int, default=2)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "fast_to_formal_promotion_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "fast_to_formal_promotion_report.json")
    args = parser.parse_args()

    report = build_report(
        args.summary_dir,
        python=args.python,
        root=args.root,
        log_dir=args.log_dir,
        max_promotions=max(1, int(args.max_promotions)),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(
    summary_dir: Path,
    *,
    python: Path,
    root: Path,
    log_dir: Path,
    max_promotions: int,
) -> dict[str, Any]:
    fast = _load_json(summary_dir / "fast_screen_status_report.json") or {}
    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    rows = fast.get("rows", []) if isinstance(fast.get("rows"), list) else []
    completed = [row for row in rows if row.get("status") == "complete"]
    anchor = _find_source(completed, ANCHOR)
    candidates = [
        _candidate_delta(row, anchor)
        for row in completed
        if _is_candidate_source(str(row.get("source_run") or ""))
    ]
    candidates.sort(key=lambda row: row["promotion_score"], reverse=True)

    promoted = [
        row for row in candidates
        if row["promote"]
    ][:max_promotions]

    verdict = "waiting_for_fast_screen"
    if completed and anchor is None:
        verdict = "waiting_for_anchor"
    elif anchor is not None and promoted:
        verdict = "promotion_ready"
    elif anchor is not None and not promoted:
        verdict = "no_candidate_clearly_promoted"

    fallback = []
    if verdict == "no_candidate_clearly_promoted":
        fallback = FALLBACK_LEAN

    selected_sources = [row["source_run"] for row in promoted]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "claim_boundary": (
            "Fast-screen promotion only chooses which full formal experiments to run next. "
            "It is not a final result and cannot support publication claims by itself."
        ),
        "current_gpu_state": _gpu_state(dashboard),
        "counts": fast.get("counts", {}),
        "anchor": anchor,
        "candidates": candidates,
        "promoted": promoted,
        "fallback_sources": fallback,
        "formal_command": _formal_command(
            python=python,
            root=root,
            log_dir=log_dir,
            sources=selected_sources,
        ) if selected_sources else None,
        "fast_screen_command": (
            f"{python} scripts\\run_fast_screen_protocol.py --scope candidates --lean-first-wave"
        ),
        "rules": _rules(max_promotions),
        "next_actions": _next_actions(verdict, selected_sources),
    }


def _candidate_delta(row: dict[str, Any], anchor: dict[str, Any] | None) -> dict[str, Any]:
    source = str(row.get("source_run") or "")
    risk = _num(row.get("risk_f1"))
    friction = _num(row.get("friction_f1"))
    low = _num(row.get("low_friction_recall"), default=None)
    worst = _num(row.get("worst_dataset_f1"))
    raw_coverage = _num(row.get("raw_coverage"))
    coverage = _num(row.get("calibrated_coverage"))
    width = _num(row.get("calibrated_width"))
    shortcut = _num(row.get("dataset_id_bal_acc"), default=0.85)
    score = _num(row.get("screen_score"))

    if anchor is None:
        deltas = {}
        promote = False
    else:
        anchor_low = _num(anchor.get("low_friction_recall"), default=None)
        low_applicable = _low_recall_applicable(row) and _low_recall_applicable(anchor)
        low_delta = None if not low_applicable or low is None or anchor_low is None else low - anchor_low
        deltas = {
            "risk_f1": risk - _num(anchor.get("risk_f1")),
            "friction_f1": friction - _num(anchor.get("friction_f1")),
            "low_friction_recall": low_delta,
            "worst_dataset_f1": worst - _num(anchor.get("worst_dataset_f1")),
            "raw_coverage": raw_coverage - _num(anchor.get("raw_coverage")),
            "calibrated_coverage": coverage - _num(anchor.get("calibrated_coverage")),
            "calibrated_width": width - _num(anchor.get("calibrated_width")),
            "dataset_id_bal_acc": shortcut - _num(anchor.get("dataset_id_bal_acc"), default=0.85),
            "screen_score": score - _num(anchor.get("screen_score")),
        }
        helps_task = (
            deltas["risk_f1"] >= 0.010
            or (low_delta is not None and low_delta >= 0.015)
            or deltas["worst_dataset_f1"] >= 0.020
        )
        safety_not_hurt = (
            deltas["friction_f1"] >= -0.010
            and deltas["risk_f1"] >= -0.005
            and (low_delta is None or low_delta >= -0.010)
            and deltas["worst_dataset_f1"] >= -0.020
        )
        interval_ok = coverage >= 0.89 and deltas["calibrated_width"] <= 0.03 and width <= 0.65
        raw_coverage_ok = raw_coverage >= 0.45 and deltas["raw_coverage"] >= -0.03
        shortcut_ok = shortcut <= 0.88 or deltas["dataset_id_bal_acc"] <= -0.02
        promote = helps_task and safety_not_hurt and interval_ok and raw_coverage_ok and shortcut_ok

    promotion_score = (
        score
        + 0.35 * max(deltas.get("risk_f1", 0.0), 0.0)
        + 0.30 * max(_zero_if_none(deltas.get("low_friction_recall")), 0.0)
        + 0.35 * max(deltas.get("worst_dataset_f1", 0.0), 0.0)
        + 0.08 * max(deltas.get("raw_coverage", 0.0), 0.0)
        - 0.08 * max(-deltas.get("raw_coverage", 0.0), 0.0)
        - 0.10 * max(deltas.get("calibrated_width", 0.0), 0.0)
        - 0.10 * max(deltas.get("dataset_id_bal_acc", 0.0), 0.0)
    )
    return {
        "run": row.get("run"),
        "source_run": source,
        "screen_score": score,
        "promotion_score": promotion_score,
        "promote": promote,
        "metrics": {
            "friction_f1": friction,
            "risk_f1": risk,
            "low_friction_recall": low,
            "low_friction_recall_applicable": _low_recall_applicable(row),
            "low_friction_positive_count": row.get("low_friction_positive_count"),
            "raw_coverage": raw_coverage,
            "calibrated_coverage": coverage,
            "calibrated_width": width,
            "worst_dataset_f1": worst,
            "dataset_id_bal_acc": shortcut,
        },
        "delta_vs_anchor": deltas,
    }


def _formal_command(*, python: Path, root: Path, log_dir: Path, sources: list[str]) -> str:
    selected = " ".join(sources)
    return (
        f"{python} scripts\\run_paper_protocol_direct.py --phase candidates "
        f"--only {selected} --python {python} --root {root} --log-dir {log_dir} --postprocess-each"
    )


def _rules(max_promotions: int) -> list[str]:
    return [
        f"Promote at most {max_promotions} fast-screen candidates before running expensive full candidate sweeps.",
        "A candidate must improve risk F1 by at least 1 point, low-friction recall by at least 1.5 points, or worst-dataset F1 by at least 2 points against screen_v1_physics_texture.",
        "Promotion is blocked if friction F1, risk F1, low-friction recall, or worst-dataset F1 regresses beyond the small safety tolerance.",
        "Promotion is blocked if calibrated coverage falls below 89%, interval width exceeds 0.65, or interval width expands by more than 0.03.",
        "Promotion is blocked if raw interval coverage falls below 45% or drops by more than three points against the screen anchor.",
        "Promotion is blocked if dataset-ID shortcut is above 88% and does not drop by at least two points.",
        "If no candidate is clearly promoted, do not launch a formal fallback; freeze the weak branch and design a new fast-screen candidate from the failure mode.",
    ]


def _next_actions(verdict: str, selected_sources: list[str]) -> list[str]:
    if verdict == "waiting_for_fast_screen":
        return [
            "Wait for the current official GPU run to finish or for a safe idle window.",
            "Run the fast-screen candidate queue before launching expensive v6-v24 full runs.",
        ]
    if verdict == "waiting_for_anchor":
        return ["Run screen_v1_physics_texture so all candidate deltas have a common protected lean-core anchor."]
    if verdict == "promotion_ready":
        return [
            "Run the formal command for promoted candidates: " + ", ".join(selected_sources) + ".",
            "Refresh postprocess reports and compare promoted rows against P0 and LODO evidence.",
        ]
    return [
        "No candidate clearly beats the screen anchor; do not launch a formal fallback candidate.",
        "Use the failure as evidence to prune FrictionSet/DG/full-fusion and design the next fast-screen candidate before any formal run.",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fast-To-Formal Promotion Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        f"Fast-screen counts: `{json.dumps(report.get('counts', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    if report.get("formal_command"):
        lines.extend(["## Formal Command", "", f"`{report['formal_command']}`", ""])
    lines.extend(["## Candidate Ranking", ""])
    candidates = report.get("candidates", [])
    if candidates:
        lines.append("| Candidate | Promote | score | d risk | d low recall | d worst | d raw cov | d width | d shortcut |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in candidates:
            d = row.get("delta_vs_anchor") or {}
            lines.append(
                "| {name} | {promote} | {score} | {risk} | {low} | {worst} | {raw_cov} | {width} | {shortcut} |".format(
                    name=row.get("source_run"),
                    promote="yes" if row.get("promote") else "no",
                    score=_fmt(row.get("promotion_score")),
                    risk=_fmt_pct(d.get("risk_f1")),
                    low=_fmt_pct(d.get("low_friction_recall")),
                    worst=_fmt_pct(d.get("worst_dataset_f1")),
                    raw_cov=_fmt_pct(d.get("raw_coverage")),
                    width=_fmt(d.get("calibrated_width")),
                    shortcut=_fmt_pct(d.get("dataset_id_bal_acc")),
                )
            )
    else:
        lines.append("No completed fast-screen candidates yet.")
    lines.extend(["", "## Rules", ""])
    for item in report["rules"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _find_source(rows: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("source_run") == source:
            return row
    return None


def _is_candidate_source(source: str) -> bool:
    return source.startswith("v") and source != ANCHOR and not source.startswith(("v0_", "v1_", "v2_", "v3_", "v4_"))


def _low_recall_applicable(row: dict[str, Any]) -> bool:
    if row.get("low_friction_recall_applicable") is False:
        return False
    positive = _num(row.get("low_friction_positive_count"), default=None)
    if positive == 0:
        return False
    return _num(row.get("low_friction_recall"), default=None) is not None


def _zero_if_none(value: float | None) -> float:
    return 0.0 if value is None else value


def _gpu_state(dashboard: dict[str, Any]) -> dict[str, Any]:
    return ((dashboard.get("system") or {}).get("gpu") or {}) if isinstance(dashboard, dict) else {}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except json.JSONDecodeError:
        return None


def _num(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{100.0 * float(value):+.2f}%"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    main()

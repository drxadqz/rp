from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
LODO_RUNS = {
    "RoadSaW": "lodo_roadsaw_full_faf",
    "RSCD": "lodo_rscd_full_faf",
    "RoadSC": "lodo_roadsc_full_faf",
}
ROADSAW_STRONG_RISK_F1 = 0.70
ROADSAW_USABLE_RISK_F1 = 0.60
ROADSAW_USABLE_FRICTION_F1 = 0.55
ROADSAW_USABLE_CALIBRATED_COVERAGE = 0.85


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "lodo_generalization_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "lodo_generalization_report.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir, args.root)
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def build_report(summary_dir: Path, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    rows = [_row(item, root) for item in summary.get("lodo", [])]
    rows = _merge_artifact_rows(rows, root)
    roadsaw_readout = _roadsaw_readout(summary, rows)
    complete = [row for row in rows if row["status"] == "complete"]
    roadsaw = next((row for row in rows if row["held_out"] == "RoadSaW"), None)
    verdict = "pending"
    if len(complete) == 3:
        worst = min(complete, key=lambda row: _num(row.get("risk_f1")) or -1.0)
        roadsaw_ok = roadsaw is not None and roadsaw["status"] == "complete" and (_num(roadsaw.get("risk_f1")) or 0.0) >= 0.55
        if roadsaw_ok and (_num(worst.get("risk_f1")) or 0.0) >= 0.50:
            verdict = "supports_cautious_domain_generalization"
        elif roadsaw_ok:
            verdict = "roadsaw_ok_but_other_domain_weak"
        else:
            verdict = "generalization_failure_needs_algorithm_update"
    return {
        "summary_dir": str(summary_dir),
        "root": str(root),
        "verdict": verdict,
        "rows": rows,
        "roadsaw_readout": roadsaw_readout,
        "interpretation_rules": [
            "Held-out RoadSaW is the decisive row because it is the known weak wetness/water-film domain.",
            "If held-out RoadSaW risk F1 is low, prioritize Fourier style jitter, condition-aware wet-state sampling, condition-aware alignment, and EvidenceField ROI constraints.",
            "LODO results should be reported separately from mixed-test P0 results because they answer a stronger OOD question.",
            "A failed LODO is still useful if it identifies which domain and class states drive the failure.",
        ],
    }


def _roadsaw_readout(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    roadsaw = next((row for row in rows if row["held_out"] == "RoadSaW"), {})
    mixed_rows = [
        row
        for row in summary.get("dataset_breakdown", [])
        if row.get("dataset") == "roadsaw"
        and row.get("status") == "complete"
        and _num(row.get("risk_macro_f1")) is not None
    ]
    mixed_best = max(mixed_rows, key=lambda row: _num(row.get("risk_macro_f1")) or -1.0) if mixed_rows else {}
    status = roadsaw.get("status", "missing")
    risk = _num(roadsaw.get("risk_f1"))
    friction = _num(roadsaw.get("friction_f1"))
    low = _num(roadsaw.get("low_friction_recall"))
    low_applicable = roadsaw.get("low_friction_recall_applicable")
    low_positive_count = _num(roadsaw.get("low_friction_positive_count"))
    coverage = _num(roadsaw.get("calibrated_coverage"))
    width = _num(roadsaw.get("calibrated_width"))
    mixed_risk = _num(mixed_best.get("risk_macro_f1"))
    mixed_friction = _num(mixed_best.get("friction_macro_f1"))
    mixed_coverage = _num(mixed_best.get("calibrated_coverage"))

    if status != "complete":
        verdict = "pending"
        claim = "Held-out RoadSaW evidence is not available yet; do not make an OOD claim."
    elif (
        (risk or 0.0) >= ROADSAW_STRONG_RISK_F1
        and (friction or 0.0) >= ROADSAW_USABLE_FRICTION_F1
        and (coverage or 0.0) >= ROADSAW_USABLE_CALIBRATED_COVERAGE
    ):
        verdict = "strong_roadsaw_ood_signal"
        claim = "This supports a cautious RoadSaW held-out generalization claim, pending the other LODO rows and fair ConvNeXt baselines."
    elif (
        (risk or 0.0) >= ROADSAW_USABLE_RISK_F1
        and (friction or 0.0) >= ROADSAW_USABLE_FRICTION_F1
        and (coverage or 0.0) >= ROADSAW_USABLE_CALIBRATED_COVERAGE
    ):
        verdict = "usable_but_needs_robustness_candidates"
        claim = "RoadSaW transfer is usable but not strong enough to carry the paper without P1/P2/P3 robustness evidence."
    else:
        verdict = "roadsaw_ood_failure"
        claim = "RoadSaW held-out transfer is weak; use this as the main failure analysis and prioritize the configured robustness candidates."

    actions = _roadsaw_actions(verdict, risk, friction, coverage, width)
    return {
        "status": status,
        "verdict": verdict,
        "claim_boundary": claim,
        "heldout_metrics": {
            "friction_f1": friction,
            "risk_f1": risk,
            "low_friction_recall": low,
            "low_friction_recall_applicable": low_applicable,
            "low_friction_positive_count": low_positive_count,
            "calibrated_coverage": coverage,
            "calibrated_width": width,
        },
        "mixed_test_reference": {
            "method": mixed_best.get("method"),
            "friction_f1": mixed_friction,
            "risk_f1": mixed_risk,
            "calibrated_coverage": mixed_coverage,
        },
        "deltas_vs_mixed_reference": {
            "friction_f1": _delta(friction, mixed_friction),
            "risk_f1": _delta(risk, mixed_risk),
            "calibrated_coverage": _delta(coverage, mixed_coverage),
        },
        "decision_thresholds": {
            "strong_risk_f1": ROADSAW_STRONG_RISK_F1,
            "usable_risk_f1": ROADSAW_USABLE_RISK_F1,
            "usable_friction_f1": ROADSAW_USABLE_FRICTION_F1,
            "usable_calibrated_coverage": ROADSAW_USABLE_CALIBRATED_COVERAGE,
        },
        "next_actions": actions,
    }


def _roadsaw_actions(
    verdict: str,
    risk: float | None,
    friction: float | None,
    coverage: float | None,
    width: float | None,
) -> list[str]:
    if verdict == "pending":
        return [
            "Wait for lodo_roadsaw_full_faf to finish before making any cross-dataset generalization claim.",
            "When complete, compare it against mixed-test RoadSaW rows and the single-dataset ConvNeXt baseline.",
        ]
    actions = [
        "Always report held-out RoadSaW separately from mixed-test P0 results.",
        "Use paired/bootstrap uncertainty and class or wetness-state breakdown before deciding the final claim.",
    ]
    if (risk or 0.0) < ROADSAW_USABLE_RISK_F1:
        actions.append("Prioritize v6/v7/v11/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 to reduce dataset style shortcut: Fourier style jitter, input/color canonicalization, quality-aware wet-road cues, Feature MixStyle, state contrast, weak interval-order consistency, visual-quality uncertainty weighting, ambiguity-aware interval ordering, segmentation-style region-mixture/multi-query evidence, DANN, and domain adapters.")
    if (friction or 0.0) < ROADSAW_USABLE_FRICTION_F1:
        actions.append("Prioritize v9/v10/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 for damp/wet/very_wet discrimination, multi-query local evidence, and condition-aware alignment.")
    if (coverage or 0.0) < ROADSAW_USABLE_CALIBRATED_COVERAGE:
        actions.append("Prioritize v12/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 and final road-ROI safety configs to improve RoadSaW interval coverage-width tradeoff.")
    if width is not None and width > 0.65:
        actions.append("Do not claim uncertainty quality from coverage alone; width is large and needs interval sharpening.")
    if len(actions) == 2:
        actions.append("Proceed to the remaining LODO rows and matched single-dataset ConvNeXt comparison before final selection.")
    return actions


def _row(item: dict[str, Any], root: Path) -> dict[str, Any]:
    method = str(item.get("method", ""))
    held_out = method.replace("held-out ", "") if method.startswith("held-out ") else method
    preliminary = _preliminary_evaluate_test(root, held_out, item.get("status"))
    low_applicable = item.get("low_friction_recall_applicable")
    low_positive_count = item.get("low_friction_positive_count")
    low_recall = item.get("low_friction_recall")
    if low_applicable is False or _num(low_positive_count) == 0:
        low_recall = None
    return {
        "held_out": held_out,
        "status": item.get("status", "missing"),
        "friction_f1": item.get("friction_macro_f1"),
        "risk_f1": item.get("risk_macro_f1"),
        "low_friction_recall": low_recall,
        "low_friction_recall_applicable": low_applicable,
        "low_friction_positive_count": low_positive_count,
        "raw_coverage": item.get("raw_interval_coverage"),
        "calibrated_coverage": item.get("calibrated_coverage"),
        "calibrated_width": item.get("calibrated_width"),
        "worst_dataset_f1": item.get("worst_dataset_f1"),
        "audit_verdict": item.get("audit_verdict"),
        "preliminary_evaluate_test": preliminary,
    }


def _merge_artifact_rows(rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    by_held = {str(row.get("held_out")): row for row in rows}
    for held_out, run in LODO_RUNS.items():
        artifact_row = _artifact_row(root / run, held_out)
        if artifact_row is None:
            if held_out not in by_held:
                by_held[held_out] = {
                    "held_out": held_out,
                    "status": "missing",
                    "friction_f1": None,
                    "risk_f1": None,
                    "low_friction_recall": None,
                    "low_friction_recall_applicable": None,
                    "low_friction_positive_count": None,
                    "raw_coverage": None,
                    "calibrated_coverage": None,
                    "calibrated_width": None,
                    "worst_dataset_f1": None,
                    "audit_verdict": None,
                    "preliminary_evaluate_test": _preliminary_evaluate_test(root, held_out, "missing"),
                }
            continue
        current = by_held.get(held_out)
        if current is None or current.get("status") != "complete":
            by_held[held_out] = artifact_row
    return [by_held[name] for name in LODO_RUNS if name in by_held]


def _artifact_row(run_dir: Path, held_out: str) -> dict[str, Any] | None:
    detailed = _load_json(run_dir / "detailed_test.json")
    calibration = _load_json(run_dir / "interval_calibration_90.json")
    audit = _load_json(run_dir / "topvenue_result_audit.json")
    required = (
        run_dir / "best.pt",
        run_dir / "detailed_test.json",
        run_dir / "interval_calibration_90.json",
        run_dir / "bootstrap_metrics.json",
        run_dir / "topvenue_result_audit.json",
    )
    if not all(path.exists() for path in required):
        return None
    tasks = detailed.get("tasks", {}) if isinstance(detailed, dict) else {}
    friction = tasks.get("friction", {}) if isinstance(tasks, dict) else {}
    risk = tasks.get("risk", {}) if isinstance(tasks, dict) else {}
    interval = detailed.get("mu_interval", {}) if isinstance(detailed, dict) else {}
    low = detailed.get("low_friction_detection", {}) if isinstance(detailed, dict) else {}
    test_split = calibration.get("test_split", {}) if isinstance(calibration, dict) else {}
    low_applicable = low.get("applicable")
    low_positive_count = low.get("num_positive")
    low_recall = low.get("recall")
    if low_applicable is False or _num(low_positive_count) == 0:
        low_recall = None
    return {
        "held_out": held_out,
        "status": "complete",
        "friction_f1": friction.get("macro_f1"),
        "risk_f1": risk.get("macro_f1"),
        "low_friction_recall": low_recall,
        "low_friction_recall_applicable": low_applicable,
        "low_friction_positive_count": low_positive_count,
        "raw_coverage": interval.get("coverage"),
        "calibrated_coverage": test_split.get("calibrated_coverage"),
        "calibrated_width": test_split.get("calibrated_width"),
        "worst_dataset_f1": (risk.get("by_dataset", {}).get("_worst_macro_f1", {}) or {}).get("value"),
        "audit_verdict": audit.get("verdict") if isinstance(audit, dict) else None,
        "preliminary_evaluate_test": None,
    }


def _preliminary_evaluate_test(root: Path, held_out: str, status: Any) -> dict[str, Any] | None:
    if status == "complete":
        return None
    run = LODO_RUNS.get(held_out)
    if not run:
        return None
    path = root / run / "evaluate_test.json"
    metrics = _load_json(path)
    if not isinstance(metrics, dict):
        return None
    return {
        "path": str(path),
        "acc_friction": metrics.get("acc_friction"),
        "acc_risk": metrics.get("acc_risk"),
        "acc_snow": metrics.get("acc_snow"),
        "raw_coverage": metrics.get("mu_interval_coverage"),
        "raw_width": metrics.get("mu_interval_width"),
        "loss": metrics.get("loss"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LODO Generalization Report",
        "",
        f"Summary dir: `{report['summary_dir']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "| Held-out dataset | Status | friction F1 | risk F1 | low-friction recall | raw coverage | calibrated coverage | calibrated width | audit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        preliminary = row.get("preliminary_evaluate_test") or {}
        status = row["status"]
        if preliminary:
            status = f"{status}; prelim eval"
        lines.append(
            "| {held} | {status} | {friction} | {risk} | {low} | {raw} | {cov} | {width} | {audit} |".format(
                held=row["held_out"],
                status=status,
                friction=_fmt_percent(row.get("friction_f1")),
                risk=_fmt_percent(row.get("risk_f1")),
                low=_fmt_percent(row.get("low_friction_recall")),
                raw=_fmt_percent(row.get("raw_coverage")),
                cov=_fmt_percent(row.get("calibrated_coverage")),
                width=_fmt_abs(row.get("calibrated_width")),
                audit=row.get("audit_verdict") or "-",
            )
        )
    prelim_rows = [row for row in report["rows"] if row.get("preliminary_evaluate_test")]
    if prelim_rows:
        lines.extend(["", "## Preliminary LODO Evaluate-Test Readout", ""])
        lines.append(
            "These values come from `evaluate_test.json` before detailed F1, calibration, bootstrap, and audit artifacts are complete."
        )
        lines.append("")
        lines.append("| Held-out dataset | friction acc | risk acc | snow acc | raw coverage | raw width | loss |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in prelim_rows:
            metrics = row.get("preliminary_evaluate_test") or {}
            lines.append(
                "| {held} | {friction} | {risk} | {snow} | {cov} | {width} | {loss} |".format(
                    held=row.get("held_out"),
                    friction=_fmt_percent(metrics.get("acc_friction")),
                    risk=_fmt_percent(metrics.get("acc_risk")),
                    snow=_fmt_percent(metrics.get("acc_snow")),
                    cov=_fmt_percent(metrics.get("raw_coverage")),
                    width=_fmt_abs(metrics.get("raw_width")),
                    loss=_fmt_abs(metrics.get("loss")),
                )
            )
    readout = report.get("roadsaw_readout") or {}
    if readout:
        metrics = readout.get("heldout_metrics") or {}
        ref = readout.get("mixed_test_reference") or {}
        deltas = readout.get("deltas_vs_mixed_reference") or {}
        lines.extend(["", "## Held-Out RoadSaW Readout", ""])
        lines.append(f"- Verdict: `{readout.get('verdict')}`.")
        lines.append(f"- Claim boundary: {readout.get('claim_boundary')}")
        lines.append(
            "- Held-out metrics: friction F1 `{friction}`, risk F1 `{risk}`, low recall `{low}` (positives: `{pos}`), calibrated coverage `{cov}`, width `{width}`.".format(
                friction=_fmt_percent(metrics.get("friction_f1")),
                risk=_fmt_percent(metrics.get("risk_f1")),
                low=_fmt_percent(metrics.get("low_friction_recall")),
                pos=_fmt_count(metrics.get("low_friction_positive_count")),
                cov=_fmt_percent(metrics.get("calibrated_coverage")),
                width=_fmt_abs(metrics.get("calibrated_width")),
            )
        )
        lines.append(
            "- Mixed-test reference: `{method}`, friction F1 `{friction}`, risk F1 `{risk}`, calibrated coverage `{cov}`.".format(
                method=ref.get("method") or "-",
                friction=_fmt_percent(ref.get("friction_f1")),
                risk=_fmt_percent(ref.get("risk_f1")),
                cov=_fmt_percent(ref.get("calibrated_coverage")),
            )
        )
        lines.append(
            "- Delta vs mixed-test reference: friction F1 `{friction}`, risk F1 `{risk}`, calibrated coverage `{cov}`.".format(
                friction=_fmt_signed_percent(deltas.get("friction_f1")),
                risk=_fmt_signed_percent(deltas.get("risk_f1")),
                cov=_fmt_signed_percent(deltas.get("calibrated_coverage")),
            )
        )
        lines.extend(["", "## RoadSaW Next Actions", ""])
        for idx, action in enumerate(readout.get("next_actions", []), start=1):
            lines.append(f"{idx}. {action}")
    lines.extend(["", "## Interpretation Rules", ""])
    for idx, rule in enumerate(report["interpretation_rules"], start=1):
        lines.append(f"{idx}. {rule}")
    lines.append("")
    return "\n".join(lines)


def _fmt_percent(value: Any) -> str:
    num = _num(value)
    return "-" if num is None else f"{100.0 * num:.2f}"


def _fmt_signed_percent(value: Any) -> str:
    num = _num(value)
    return "-" if num is None else f"{100.0 * num:+.2f}"


def _fmt_abs(value: Any) -> str:
    num = _num(value)
    return "-" if num is None else f"{num:.4f}"


def _fmt_count(value: Any) -> str:
    num = _num(value)
    return "-" if num is None else str(int(num))


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


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


if __name__ == "__main__":
    main()

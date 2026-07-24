from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY = Path("reports/paper_protocol_summary")
RUN = "v14_lean_road_roi_safety"
ANCHOR_METHOD = "+ PhysicsTexture"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY / "v14_first_epoch_gate_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY / "v14_first_epoch_gate_report.json")
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path, summary_dir: Path) -> dict[str, Any]:
    run_dir = root / RUN
    state = _load_json(run_dir / "training_state.json") or {}
    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    watch = _load_json(summary_dir / "active_training_watch_report.json") or {}
    p0_anchor = _read_p0_anchor(summary_dir / "paper_p0_ablation_table.csv")

    val = state.get("val_metrics") or {}
    active = _active(queue, watch)
    has_val = bool(val)
    decision = _decision(val, p0_anchor, has_val)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run": RUN,
        "run_dir": str(run_dir),
        "has_training_state": bool(state),
        "active": active,
        "has_validation_epoch": has_val,
        "epoch": state.get("epoch"),
        "planned_epochs": state.get("epochs"),
        "val_metrics": _compact_val(val),
        "anchor_method": ANCHOR_METHOD,
        "anchor": p0_anchor,
        "decision": decision,
        "claim_boundary": (
            "This is an early training gate for a candidate route. It is not a final "
            "paper result and cannot replace full test evaluation, calibration, "
            "bootstrap confidence intervals, LODO, or matched ConvNeXt comparisons."
        ),
    }


def _active(queue: dict[str, Any], watch: dict[str, Any]) -> dict[str, Any]:
    active = watch.get("active") or {}
    if active:
        return {
            "name": active.get("name"),
            "phase": active.get("phase"),
            "epoch": active.get("epoch"),
            "epochs": active.get("epochs"),
            "step": active.get("step"),
            "steps": active.get("steps"),
            "eta": active.get("eta"),
            "rate": active.get("rate"),
        }
    for row in queue.get("runs", []) or queue.get("queue", []) or []:
        if row.get("run") == RUN or row.get("name") == RUN:
            return row
    return {}


def _compact_val(val: dict[str, Any]) -> dict[str, float]:
    keys = [
        "loss",
        "acc_friction",
        "acc_risk",
        "mu_interval_coverage",
        "mu_interval_width",
        "mu_mean_mae_to_interval_mid",
    ]
    out: dict[str, float] = {}
    for key in keys:
        value = val.get(key)
        if value is not None:
            out[key] = float(value)
    return out


def _read_p0_anchor(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == ANCHOR_METHOD:
                return {
                    "method": row.get("method", ANCHOR_METHOD),
                    "friction_macro_f1": _float(row.get("friction_macro_f1")),
                    "risk_macro_f1": _float(row.get("risk_macro_f1")),
                    "low_friction_recall": _float(row.get("low_friction_recall")),
                    "calibrated_coverage": _float(row.get("calibrated_coverage")),
                    "worst_dataset_f1": _float(row.get("worst_dataset_f1")),
                    "raw_interval_coverage": _float(row.get("raw_interval_coverage")),
                    "calibrated_width": _float(row.get("calibrated_width")),
                }
    return {}


def _decision(val: dict[str, Any], anchor: dict[str, Any], has_val: bool) -> dict[str, Any]:
    if not has_val:
        return {
            "status": "wait_for_first_validation",
            "action": "Keep monitoring; do not promote or kill v14 before the first validation epoch.",
        }
    risk = _float(val.get("acc_risk"))
    friction = _float(val.get("acc_friction"))
    cov = _float(val.get("mu_interval_coverage"))
    width = _float(val.get("mu_interval_width"))
    anchor_risk = _float(anchor.get("risk_macro_f1"))
    anchor_friction = _float(anchor.get("friction_macro_f1"))
    problems: list[str] = []
    if risk is not None and anchor_risk is not None and risk < anchor_risk - 0.08:
        problems.append("risk accuracy is far below the PhysicsTexture test-F1 anchor")
    if friction is not None and anchor_friction is not None and friction < anchor_friction - 0.08:
        problems.append("friction accuracy is far below the PhysicsTexture test-F1 anchor")
    if cov is not None and cov < 0.45:
        problems.append("raw interval coverage is very low in early validation")
    if width is not None and width > 0.65:
        problems.append("interval width is already very large")
    if problems:
        return {
            "status": "early_warning_continue_until_policy_stop",
            "problems": problems,
            "action": (
                "Do not promote v14 yet. Continue only under the existing early-stop policy; "
                "if the same failures persist after several validation epochs, prune this route."
            ),
        }
    return {
        "status": "healthy_first_validation",
        "action": (
            "Keep v14 in the candidate queue and wait for full postprocessing before any "
            "paper-level claim."
        ),
    }


def _float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) <= 1.5:
            return f"{value:.2%}"
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    val = report.get("val_metrics") or {}
    anchor = report.get("anchor") or {}
    decision = report.get("decision") or {}
    active = report.get("active") or {}
    lines = [
        "# v14 First-Epoch Gate Report",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        "",
        f"Boundary: {report.get('claim_boundary')}",
        "",
        "## Active State",
        "",
        "- Run: `{run}`.".format(run=report.get("run")),
        "- Active: `{name}` phase `{phase}` epoch `{epoch}/{epochs}` step `{step}/{steps}` ETA `{eta}`.".format(
            name=active.get("name", "-"),
            phase=active.get("phase", "-"),
            epoch=active.get("epoch", "-"),
            epochs=active.get("epochs", "-"),
            step=active.get("step", "-"),
            steps=active.get("steps", "-"),
            eta=active.get("eta", "-"),
        ),
        "- Validation available: `{}`.".format(report.get("has_validation_epoch")),
        "",
        "## Current Validation Metrics",
        "",
        "| metric | v14 current | PhysicsTexture anchor |",
        "|---|---:|---:|",
        f"| risk | {_fmt(val.get('acc_risk'))} | {_fmt(anchor.get('risk_macro_f1'))} |",
        f"| friction | {_fmt(val.get('acc_friction'))} | {_fmt(anchor.get('friction_macro_f1'))} |",
        f"| raw coverage | {_fmt(val.get('mu_interval_coverage'))} | {_fmt(anchor.get('raw_interval_coverage'))} |",
        f"| interval width | {_fmt(val.get('mu_interval_width'))} | {_fmt(anchor.get('calibrated_width'))} |",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.get('status')}`.",
        f"- Action: {decision.get('action')}",
    ]
    problems = decision.get("problems") or []
    if problems:
        lines.extend(["", "Problems:"])
        lines.extend(f"- {item}" for item in problems)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

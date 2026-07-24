from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("reports/paper_protocol_summary/paper_protocol_summary.json")
DEFAULT_OUT = Path("reports/paper_protocol_summary/module_decisions.md")


ADJACENT_STEPS = [
    ("PhysicsTexture", "Global-only", "+ PhysicsTexture"),
    ("FrictionSet", "+ PhysicsTexture", "+ FrictionSet"),
    ("DG losses", "+ FrictionSet", "+ DG losses"),
    ("EvidenceField aux", "+ DG losses", "+ EvidenceField aux"),
    ("Full fusion", "+ EvidenceField aux", "Full model"),
    ("Fourier style jitter", "Full model", "Full + Fourier candidate"),
    ("Domain-adversarial training", "Full + Fourier candidate", "Full + Fourier + DANN candidate"),
    ("Road prior", "Full + Fourier candidate", "Full + Fourier + road prior candidate"),
    ("Wet-state hard sampling", "Full + Fourier + road prior candidate", "Full + wet-state hard-sampling candidate"),
    ("Weak-view consistency", "Full + wet-state hard-sampling candidate", "Full + consistency candidate"),
    ("Domain-specific adapter", "Full + consistency candidate", "Full + domain adapter candidate"),
    ("ROI interval safety", "Full + consistency candidate", "Full + ROI interval-safety candidate"),
]

PRIMARY_KEYS = [
    "risk_macro_f1",
    "low_friction_recall",
    "worst_dataset_risk_f1",
    "friction_macro_f1",
]

INTERVAL_KEYS = [
    "raw_interval_coverage",
    "calibrated_coverage",
    "calibrated_width",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=Path("reports/paper_protocol_summary/module_decisions.csv"))
    args = parser.parse_args()

    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    rows = decide(payload.get("ablation", []))
    md = render(rows)
    print(md)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def decide(ablation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row.get("method"): row for row in ablation}
    rows = []
    for module, prev_name, cur_name in ADJACENT_STEPS:
        prev = by_method.get(prev_name)
        cur = by_method.get(cur_name)
        if not prev or not cur or prev.get("status") != "complete" or cur.get("status") != "complete":
            rows.append({"module": module, "decision": "pending", "reason": "Required adjacent rows are incomplete."})
            continue
        deltas = {f"delta_{key}": delta(cur, prev, key) for key in PRIMARY_KEYS}
        deltas.update({f"delta_{key}": delta(cur, prev, key) for key in INTERVAL_KEYS})
        deltas["delta_dataset_id_balanced_accuracy"] = delta(cur, prev, "dataset_id_balanced_accuracy")
        improved = sum(
            1
            for key, value in deltas.items()
            if key in {f"delta_{item}" for item in PRIMARY_KEYS} and value is not None and value >= 0.005
        )
        harmed = sum(
            1
            for key, value in deltas.items()
            if key in {f"delta_{item}" for item in PRIMARY_KEYS} and value is not None and value <= -0.02
        )
        interval_help = (
            deltas.get("delta_raw_interval_coverage") is not None
            and deltas["delta_raw_interval_coverage"] >= 0.03
        ) or (
            deltas.get("delta_calibrated_width") is not None
            and deltas["delta_calibrated_width"] <= -0.03
            and (
                deltas.get("delta_calibrated_coverage") is None
                or deltas["delta_calibrated_coverage"] >= -0.01
            )
        )
        shortcut_help = (
            deltas.get("delta_dataset_id_balanced_accuracy") is not None
            and deltas["delta_dataset_id_balanced_accuracy"] <= -0.02
        )
        width_hurt = (
            deltas.get("delta_calibrated_width") is not None
            and deltas["delta_calibrated_width"] >= 0.05
            and not interval_help
        )
        if harmed > 0:
            decision = "rework_or_remove"
            reason = "At least one primary metric drops by more than two points."
        elif width_hurt:
            decision = "rework_or_remove"
            reason = "Calibrated interval width grows without enough coverage benefit."
        elif improved >= 2 or interval_help or shortcut_help:
            decision = "keep"
            reason = "Primary metrics, interval quality, or shortcut resistance improves."
        else:
            decision = "merge_or_simplify"
            reason = "Adjacent gain is small; keep only if interpretability or LODO supports it."
        rows.append({"module": module, "decision": decision, "reason": reason, **deltas})
    return rows


def delta(cur: dict[str, Any], prev: dict[str, Any], key: str) -> float | None:
    if cur.get(key) is None or prev.get(key) is None:
        return None
    return float(cur[key]) - float(prev[key])


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["# Module Decisions", ""]
    lines.append("| Module | Decision | d risk F1 | d low recall | d worst risk F1 | d raw cov | d calib width | d dataset-ID bal acc | Reason |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {module} | {decision} | {dr} | {dl} | {dw} | {dc} | {dwidth} | {ddomain} | {reason} |".format(
                module=row["module"],
                decision=row["decision"],
                dr=fmt(row.get("delta_risk_macro_f1")),
                dl=fmt(row.get("delta_low_friction_recall")),
                dw=fmt(row.get("delta_worst_dataset_risk_f1")),
                dc=fmt(row.get("delta_raw_interval_coverage")),
                dwidth=fmt_abs_delta(row.get("delta_calibrated_width")),
                ddomain=fmt(row.get("delta_dataset_id_balanced_accuracy")),
                reason=row["reason"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):+.2f}"


def fmt_abs_delta(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


if __name__ == "__main__":
    main()

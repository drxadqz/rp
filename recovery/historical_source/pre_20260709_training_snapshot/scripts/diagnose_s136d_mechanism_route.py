from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SOTA_TOP1 = 0.9286
SOTA_MACRO_F1 = 0.8949
KEY_CLASSES = [
    "water_concrete_slight",
    "wet_concrete_slight",
    "water_concrete_severe",
    "wet_concrete_severe",
    "dry_concrete_slight",
    "dry_concrete_severe",
    "water_asphalt_slight",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    payload = read_json(run_dir / "test_metrics.json")
    if payload is None:
        return None
    return dict(payload.get("summary", payload))


def read_per_class(run_dir: Path | None) -> dict[str, dict[str, float]]:
    if run_dir is None:
        return {}
    path = run_dir / "per_class_metrics.csv"
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("class") or row.get("\ufeffclass") or "")
            if not name:
                continue
            rows[name] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return rows


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.3f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:+.3f} pp"


def metric_delta(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, float] | None:
    if candidate is None or baseline is None:
        return None
    keys = ["top1", "macro_f1", "mean_precision", "mean_recall", "weighted_f1"]
    return {key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in keys}


def class_delta(
    candidate: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in sorted(set(candidate) | set(baseline)):
        c = candidate.get(name, {})
        b = baseline.get(name, {})
        out.append(
            {
                "class": name,
                "candidate_f1": float(c.get("f1", 0.0)),
                "baseline_f1": float(b.get("f1", 0.0)),
                "delta_f1": float(c.get("f1", 0.0)) - float(b.get("f1", 0.0)),
                "candidate_precision": float(c.get("precision", 0.0)),
                "baseline_precision": float(b.get("precision", 0.0)),
                "delta_precision": float(c.get("precision", 0.0)) - float(b.get("precision", 0.0)),
                "candidate_recall": float(c.get("recall", 0.0)),
                "baseline_recall": float(b.get("recall", 0.0)),
                "delta_recall": float(c.get("recall", 0.0)) - float(b.get("recall", 0.0)),
                "support": float(c.get("support", b.get("support", 0.0))),
                "key_class": name in KEY_CLASSES,
            }
        )
    return out


def summarize_run(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "top1": float(summary.get("top1", 0.0)),
        "macro_f1": float(summary.get("macro_f1", 0.0)),
        "mean_precision": float(summary.get("mean_precision", 0.0)),
        "mean_recall": float(summary.get("mean_recall", 0.0)),
        "weighted_f1": float(summary.get("weighted_f1", 0.0)),
        "num_samples": int(float(summary.get("num_samples", 0) or 0)),
        "param_count": int(float(summary.get("param_count", 0) or 0)),
        "hard_class_mean_f1": float(summary.get("hard_class_mean_f1", 0.0)),
        "water_concrete_slight_f1": float(summary.get("water_concrete_slight_f1", 0.0)),
        "friction_acc": float(summary.get("friction_acc", 0.0)),
        "material_acc": float(summary.get("material_acc", 0.0)),
        "roughness_acc": float(summary.get("roughness_acc", 0.0)),
    }


def key_class_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["class"] in KEY_CLASSES]


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    s96 = payload["runs"].get("S96_cap250")
    s136 = payload["runs"].get("S136_no_distill_screen")
    s136d = payload["runs"].get("S136d_safe_distill_screen")
    s136d_full = payload["runs"].get("S136d_full")
    s7 = payload["runs"].get("S7_full")

    if s136d_full is not None:
        if s136d_full["top1"] >= SOTA_TOP1 and s136d_full["macro_f1"] >= SOTA_MACRO_F1:
            return {
                "route_action": "candidate_complete_verify_and_write",
                "reason": "S136d full meets both public SOTA thresholds; run final audit and prepare paper-level evidence.",
            }
        delta_s7 = metric_delta(s136d_full, s7)
        if delta_s7 and delta_s7["macro_f1"] >= 0.0 and delta_s7["top1"] < 0.0:
            return {
                "route_action": "improve_top1_calibration_after_full",
                "reason": "Full run preserves or improves macro-F1 but still misses Top-1; next mechanism should target confidence calibration and fine-grained decision margins without hurting balanced F1.",
            }
        return {
            "route_action": "revise_full_custom_backbone_or_distill",
            "reason": "S136d full does not beat public SOTA; use per-class drops to decide whether to tune safe-distill weights or redesign early evidence experts.",
        }

    if s136d is None:
        if s136 is None:
            return {
                "route_action": "wait_for_s136_screen",
                "reason": "Neither S136 nor S136d screen metrics exist yet; keep current queued experiments.",
            }
        delta = metric_delta(s136, s96)
        if delta and delta["macro_f1"] >= 0.0 and delta["top1"] >= -0.005:
            return {
                "route_action": "consider_s136_full_before_s136d",
                "reason": "No-distill S136 already looks promising versus S96; let the existing S136 watcher decide full promotion before invoking S136d.",
            }
        return {
            "route_action": "run_s136d_screen_if_upstream_fails",
            "reason": "S136 no-distill is missing or weak; S136d safe-distill is the queued fallback to stabilize the custom backbone.",
        }

    delta_s96 = metric_delta(s136d, s96)
    delta_s136 = metric_delta(s136d, s136)
    if delta_s96 and delta_s96["macro_f1"] >= 0.0 and delta_s96["top1"] >= 0.0:
        if delta_s136 is None:
            return {
                "route_action": "screen_promising_run_full_but_collect_nodistill_control",
                "reason": "S136d improves over S96, but S136 no-distill control is unavailable; full can proceed only as route evidence, while mechanism attribution remains pending.",
            }
        if delta_s136["macro_f1"] >= 0.0 and delta_s136["top1"] >= -0.002:
            return {
                "route_action": "promote_s136d_full",
                "reason": "S136d beats S96 and does not regress versus S136 no-distill; safe-distill guard is plausible and full protocol is justified.",
            }
        return {
            "route_action": "promote_backbone_not_distill",
            "reason": "S136d beats S96 but not S136 no-distill; keep the custom backbone evidence, but do not claim safe-distill as the main mechanism.",
        }

    if delta_s136 and delta_s136["macro_f1"] < -0.005:
        return {
            "route_action": "weaken_or_remove_safe_distill",
            "reason": "S136d harms macro-F1 relative to S136; safe-distill is over-constraining factor/coupled boundary learning.",
        }
    if delta_s96 and delta_s96["top1"] >= 0.0 and delta_s96["macro_f1"] < 0.0:
        return {
            "route_action": "rebalance_focus_and_classwise_noharm",
            "reason": "S136d improves Top-1 but hurts Macro-F1, implying majority/easy-class protection is suppressing weak coupled classes.",
        }
    return {
        "route_action": "revise_early_evidence_experts",
        "reason": "S136d screen does not clear the route gate; next custom backbone should change the early evidence experts rather than add later heads.",
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    runs = payload["runs"]
    lines = [
        "# S136/S136d Mechanism Diagnosis",
        "",
        f"- Decision: `{payload['decision']['route_action']}`",
        f"- Reason: {payload['decision']['reason']}",
        "",
        "## Available Runs",
        "",
        "| Run | Samples | Top-1 | Macro-F1 | Hard F1 | WCS F1 | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["S96_cap250", "S136_no_distill_screen", "S136d_safe_distill_screen", "S7_full", "S136d_full"]:
        summary = runs.get(name)
        if summary is None:
            lines.append(f"| {name} | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {summary['num_samples']} | {pct(summary['top1'])} | {pct(summary['macro_f1'])} | "
            f"{pct(summary['hard_class_mean_f1'])} | {pct(summary['water_concrete_slight_f1'])} | {summary['param_count']} |"
        )

    lines.extend(["", "## Main Deltas", "", "| Comparison | Top-1 | Macro-F1 | Mean-P | Mean-R |", "|---|---:|---:|---:|---:|"])
    for item in payload["comparisons"]:
        delta = item.get("metric_delta")
        if delta is None:
            lines.append(f"| {item['name']} | - | - | - | - |")
        else:
            lines.append(
                f"| {item['name']} | {pp(delta['top1'])} | {pp(delta['macro_f1'])} | "
                f"{pp(delta['mean_precision'])} | {pp(delta['mean_recall'])} |"
            )

    lines.extend(["", "## Key Class Deltas", ""])
    for item in payload["comparisons"]:
        rows = item.get("key_class_delta") or []
        if not rows:
            continue
        lines.extend([f"### {item['name']}", "", "| Class | Candidate F1 | Baseline F1 | Delta |", "|---|---:|---:|---:|"])
        for row in rows:
            lines.append(
                f"| {row['class']} | {pct(row['candidate_f1'])} | {pct(row['baseline_f1'])} | {pp(row['delta_f1'])} |"
            )
        lines.append("")

    lines.extend(["## Worst Drops", ""])
    for item in payload["comparisons"]:
        rows = item.get("largest_drops") or []
        if not rows:
            continue
        lines.extend([f"### {item['name']}", "", "| Class | Candidate F1 | Baseline F1 | Delta |", "|---|---:|---:|---:|"])
        for row in rows[:8]:
            lines.append(
                f"| {row['class']} | {pct(row['candidate_f1'])} | {pct(row['baseline_f1'])} | {pp(row['delta_f1'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Rule",
            "",
            "- S136 versus S96 isolates the custom early-coupling backbone route.",
            "- S136d versus S136 isolates the safe-distillation/no-harm training guard.",
            "- Full complete-manifest metrics are the only paper-level SOTA evidence.",
            "- Screen metrics only decide whether a route deserves full training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose S136/S136d mechanism attribution and next action.")
    parser.add_argument("--s96-dir", required=True, type=Path)
    parser.add_argument("--s7-dir", required=True, type=Path)
    parser.add_argument("--s136-dir", type=Path)
    parser.add_argument("--s136d-dir", type=Path)
    parser.add_argument("--s136d-full-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "S96_cap250": summarize_run(read_summary(args.s96_dir)),
        "S136_no_distill_screen": summarize_run(read_summary(args.s136_dir)),
        "S136d_safe_distill_screen": summarize_run(read_summary(args.s136d_dir)),
        "S7_full": summarize_run(read_summary(args.s7_dir)),
        "S136d_full": summarize_run(read_summary(args.s136d_full_dir)),
    }
    per_class = {
        "S96_cap250": read_per_class(args.s96_dir),
        "S136_no_distill_screen": read_per_class(args.s136_dir),
        "S136d_safe_distill_screen": read_per_class(args.s136d_dir),
        "S7_full": read_per_class(args.s7_dir),
        "S136d_full": read_per_class(args.s136d_full_dir),
    }
    comparison_specs = [
        ("S136 vs S96", "S136_no_distill_screen", "S96_cap250"),
        ("S136d vs S96", "S136d_safe_distill_screen", "S96_cap250"),
        ("S136d vs S136", "S136d_safe_distill_screen", "S136_no_distill_screen"),
        ("S136d_full vs S7_full", "S136d_full", "S7_full"),
    ]
    comparisons: list[dict[str, Any]] = []
    for label, cand, base in comparison_specs:
        delta = metric_delta(summaries.get(cand), summaries.get(base))
        rows = class_delta(per_class.get(cand, {}), per_class.get(base, {})) if delta is not None else []
        comparisons.append(
            {
                "name": label,
                "candidate": cand,
                "baseline": base,
                "metric_delta": delta,
                "key_class_delta": key_class_table(rows),
                "largest_drops": sorted(rows, key=lambda row: row["delta_f1"])[:12],
            }
        )
    payload = {
        "ok": True,
        "runs": summaries,
        "comparisons": comparisons,
        "decision": decide({"runs": summaries, "comparisons": comparisons}),
    }
    (args.output_dir / "s136d_mechanism_diagnosis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "s136d_mechanism_diagnosis.md")
    print(json.dumps({"ok": True, "decision": payload["decision"], "report": str(args.output_dir / "s136d_mechanism_diagnosis.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

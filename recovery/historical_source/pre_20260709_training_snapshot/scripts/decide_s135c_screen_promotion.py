from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(metrics: dict[str, Any]) -> dict[str, float]:
    summary = metrics.get("summary", metrics)
    return {str(k): float(v) for k, v in summary.items() if isinstance(v, (int, float))}


def _read_per_class(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(name).lstrip("\ufeff") for name in (reader.fieldnames or [])]
        class_field = "class" if "class" in fieldnames else (fieldnames[0] if fieldnames else "class")
        for row in reader:
            if class_field not in row and f"\ufeff{class_field}" in row:
                class_field = f"\ufeff{class_field}"
            name = str(row[class_field])
            out[name] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide whether S135c cap250 screen should be promoted to full RSCD.")
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-name", default="S135c")
    parser.add_argument("--baseline-name", default="S96")
    parser.add_argument("--macro-margin", type=float, default=0.0010)
    parser.add_argument("--top1-drop-tolerance", type=float, default=0.0005)
    parser.add_argument("--wcs-margin", type=float, default=0.0030)
    parser.add_argument("--wc-severe-drop-tolerance", type=float, default=0.0050)
    parser.add_argument("--hard-mean-drop-tolerance", type=float, default=0.0020)
    args = parser.parse_args()

    output_dir = args.output_dir or args.candidate_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_metrics = _read_json(args.candidate_dir / "test_metrics.json")
    baseline_metrics = _read_json(args.baseline_dir / "test_metrics.json")
    candidate_summary = _summary(candidate_metrics)
    baseline_summary = _summary(baseline_metrics)
    candidate_class = _read_per_class(args.candidate_dir / "per_class_metrics.csv")
    baseline_class = _read_per_class(args.baseline_dir / "per_class_metrics.csv")

    def cls_f1(name: str, table: dict[str, dict[str, float]]) -> float:
        return float(table.get(name, {}).get("f1", 0.0))

    checks = [
        {
            "name": "macro_f1_gain",
            "candidate": candidate_summary.get("macro_f1", 0.0),
            "baseline": baseline_summary.get("macro_f1", 0.0),
            "threshold": args.macro_margin,
            "pass": candidate_summary.get("macro_f1", 0.0) >= baseline_summary.get("macro_f1", 0.0) + args.macro_margin,
            "direction": "candidate_minus_baseline_at_least_threshold",
        },
        {
            "name": "top1_no_material_drop",
            "candidate": candidate_summary.get("top1", 0.0),
            "baseline": baseline_summary.get("top1", 0.0),
            "threshold": -args.top1_drop_tolerance,
            "pass": candidate_summary.get("top1", 0.0) >= baseline_summary.get("top1", 0.0) - args.top1_drop_tolerance,
            "direction": "candidate_minus_baseline_not_below_threshold",
        },
        {
            "name": "water_concrete_slight_f1_gain",
            "candidate": cls_f1("water_concrete_slight", candidate_class),
            "baseline": cls_f1("water_concrete_slight", baseline_class),
            "threshold": args.wcs_margin,
            "pass": cls_f1("water_concrete_slight", candidate_class)
            >= cls_f1("water_concrete_slight", baseline_class) + args.wcs_margin,
            "direction": "candidate_minus_baseline_at_least_threshold",
        },
        {
            "name": "water_concrete_severe_no_spill",
            "candidate": cls_f1("water_concrete_severe", candidate_class),
            "baseline": cls_f1("water_concrete_severe", baseline_class),
            "threshold": -args.wc_severe_drop_tolerance,
            "pass": cls_f1("water_concrete_severe", candidate_class)
            >= cls_f1("water_concrete_severe", baseline_class) - args.wc_severe_drop_tolerance,
            "direction": "candidate_minus_baseline_not_below_threshold",
        },
        {
            "name": "hard_class_mean_no_spill",
            "candidate": candidate_summary.get("hard_class_mean_f1", 0.0),
            "baseline": baseline_summary.get("hard_class_mean_f1", 0.0),
            "threshold": -args.hard_mean_drop_tolerance,
            "pass": candidate_summary.get("hard_class_mean_f1", 0.0)
            >= baseline_summary.get("hard_class_mean_f1", 0.0) - args.hard_mean_drop_tolerance,
            "direction": "candidate_minus_baseline_not_below_threshold",
        },
    ]
    for item in checks:
        item["delta"] = float(item["candidate"] - item["baseline"])

    promote = all(bool(item["pass"]) for item in checks)
    decision = {
        "candidate_name": args.candidate_name,
        "baseline_name": args.baseline_name,
        "candidate_dir": str(args.candidate_dir),
        "baseline_dir": str(args.baseline_dir),
        "promote_to_full": promote,
        "checks": checks,
        "candidate_summary": {
            key: candidate_summary.get(key)
            for key in ("top1", "macro_f1", "mean_precision", "mean_recall", "hard_class_mean_f1", "num_errors")
        },
        "baseline_summary": {
            key: baseline_summary.get(key)
            for key in ("top1", "macro_f1", "mean_precision", "mean_recall", "hard_class_mean_f1", "num_errors")
        },
    }
    (output_dir / "screen_promotion_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"# {args.candidate_name} screen promotion decision",
        "",
        f"- Candidate: `{args.candidate_dir}`",
        f"- Baseline: `{args.baseline_dir}`",
        f"- Promote to full: **{promote}**",
        "",
        "| Check | Candidate | Baseline | Delta | Threshold | Pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in checks:
        lines.append(
            f"| {item['name']} | {item['candidate']:.6f} | {item['baseline']:.6f} | "
            f"{item['delta']:+.6f} | {item['threshold']:+.6f} | {item['pass']} |"
        )
    (output_dir / "screen_promotion_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"promote_to_full": promote, "decision": str(output_dir / "screen_promotion_decision.json")}))
    return 0 if promote else 2


if __name__ == "__main__":
    raise SystemExit(main())

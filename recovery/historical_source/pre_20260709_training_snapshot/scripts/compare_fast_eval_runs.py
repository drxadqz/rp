from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.4f} pp"


def summary_from_metrics(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    summary = payload.get("summary", {})
    report = payload.get("classification_report", {})
    per_class = []
    for name, item in report.items():
        if not isinstance(item, dict) or "f1-score" not in item:
            continue
        per_class.append((name, float(item.get("f1-score") or 0.0)))
    weakest = min(per_class, key=lambda x: x[1]) if per_class else ("", 0.0)
    return {
        "top1": float(summary.get("top1") or 0.0),
        "macro_f1": float(summary.get("macro_f1") or summary.get("mean_f1") or 0.0),
        "mean_precision": float(summary.get("mean_precision") or 0.0),
        "mean_recall": float(summary.get("mean_recall") or 0.0),
        "num_samples": int(summary.get("num_samples") or 0),
        "num_errors": int(summary.get("num_errors") or 0),
        "weakest_class": weakest[0],
        "weakest_f1": weakest[1],
    }


def f1_from_row(row: dict[str, str]) -> float:
    for key in ("f1", "f1-score", "f1_score"):
        if key in row and row[key] != "":
            return float(row[key])
    return 0.0


def compare_per_class(base_rows: list[dict[str, str]], cand_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    base = {row["class"]: row for row in base_rows}
    cand = {row["class"]: row for row in cand_rows}
    rows = []
    for cls in sorted(set(base) | set(cand)):
        b = base.get(cls, {})
        c = cand.get(cls, {})
        b_f1 = f1_from_row(b)
        c_f1 = f1_from_row(c)
        rows.append(
            {
                "class": cls,
                "baseline_f1": b_f1,
                "candidate_f1": c_f1,
                "delta_f1_pp": (c_f1 - b_f1) * 100.0,
                "baseline_precision": float(b.get("precision") or 0.0),
                "candidate_precision": float(c.get("precision") or 0.0),
                "baseline_recall": float(b.get("recall") or 0.0),
                "candidate_recall": float(c.get("recall") or 0.0),
                "support": float(c.get("support") or b.get("support") or 0.0),
            }
        )
    return sorted(rows, key=lambda row: float(row["delta_f1_pp"]))


def compare_predictions(base_rows: list[dict[str, str]], cand_rows: list[dict[str, str]]) -> dict[str, Any]:
    base = {row["image_path"]: row for row in base_rows}
    cand = {row["image_path"]: row for row in cand_rows}
    changed = []
    fixed = []
    regressed = []
    neutral = []
    for path in sorted(set(base) & set(cand)):
        b = base[path]
        c = cand[path]
        if b["pred_label"] == c["pred_label"]:
            continue
        true_label = b["true_label"]
        was_ok = b["pred_label"] == true_label
        now_ok = c["pred_label"] == true_label
        row = {
            "image_path": path,
            "true_label": true_label,
            "baseline_pred": b["pred_label"],
            "candidate_pred": c["pred_label"],
            "baseline_confidence": b.get("confidence", ""),
            "candidate_confidence": c.get("confidence", ""),
        }
        changed.append(row)
        if now_ok and not was_ok:
            fixed.append(row)
        elif was_ok and not now_ok:
            regressed.append(row)
        else:
            neutral.append(row)
    return {
        "changed": changed,
        "fixed": fixed,
        "regressed": regressed,
        "neutral": neutral,
        "fixed_by_true": Counter(row["true_label"] for row in fixed),
        "regressed_by_true": Counter(row["true_label"] for row in regressed),
        "changed_by_pair": Counter(f"{row['baseline_pred']}->{row['candidate_pred']}" for row in changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two RSCD fast/full eval outputs.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics = summary_from_metrics(args.baseline_dir / "metrics.json")
    candidate_metrics = summary_from_metrics(args.candidate_dir / "metrics.json")
    per_class = compare_per_class(
        read_csv(args.baseline_dir / "per_class_metrics.csv"),
        read_csv(args.candidate_dir / "per_class_metrics.csv"),
    )
    pred_cmp = compare_predictions(
        read_csv(args.baseline_dir / "predictions_test.csv"),
        read_csv(args.candidate_dir / "predictions_test.csv"),
    )

    write_rows(args.out_dir / "per_class_delta.csv", per_class)
    write_rows(args.out_dir / "changed_predictions.csv", pred_cmp["changed"])
    write_rows(args.out_dir / "fixed_predictions.csv", pred_cmp["fixed"])
    write_rows(args.out_dir / "regressed_predictions.csv", pred_cmp["regressed"])

    summary = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_top1_pp": (candidate_metrics["top1"] - baseline_metrics["top1"]) * 100.0,
        "delta_macro_f1_pp": (candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]) * 100.0,
        "delta_errors": candidate_metrics["num_errors"] - baseline_metrics["num_errors"],
        "changed": len(pred_cmp["changed"]),
        "fixed": len(pred_cmp["fixed"]),
        "regressed": len(pred_cmp["regressed"]),
        "neutral_changed": len(pred_cmp["neutral"]),
        "fixed_by_true": dict(pred_cmp["fixed_by_true"].most_common()),
        "regressed_by_true": dict(pred_cmp["regressed_by_true"].most_common()),
        "changed_by_pair": dict(pred_cmp["changed_by_pair"].most_common(30)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# RSCD Eval Comparison",
        "",
        f"- {args.baseline_name}: Top1={pct(baseline_metrics['top1'])}, Macro-F1={pct(baseline_metrics['macro_f1'])}, errors={baseline_metrics['num_errors']}, weakest={baseline_metrics['weakest_class']} ({pct(baseline_metrics['weakest_f1'])})",
        f"- {args.candidate_name}: Top1={pct(candidate_metrics['top1'])}, Macro-F1={pct(candidate_metrics['macro_f1'])}, errors={candidate_metrics['num_errors']}, weakest={candidate_metrics['weakest_class']} ({pct(candidate_metrics['weakest_f1'])})",
        f"- Delta: Top1={pp(candidate_metrics['top1'] - baseline_metrics['top1'])}, Macro-F1={pp(candidate_metrics['macro_f1'] - baseline_metrics['macro_f1'])}, errors={summary['delta_errors']:+d}",
        "",
        "## Prediction Changes",
        "",
        f"- changed={summary['changed']}, fixed={summary['fixed']}, regressed={summary['regressed']}, neutral_changed={summary['neutral_changed']}",
        f"- fixed_by_true={summary['fixed_by_true']}",
        f"- regressed_by_true={summary['regressed_by_true']}",
        "",
        "## Largest F1 Gains",
        "",
    ]
    for row in sorted(per_class, key=lambda item: float(item["delta_f1_pp"]), reverse=True)[:10]:
        lines.append(
            f"- {row['class']}: {float(row['delta_f1_pp']):+.3f} pp "
            f"({float(row['baseline_f1']) * 100:.2f}->{float(row['candidate_f1']) * 100:.2f})"
        )
    lines.extend(["", "## Largest F1 Drops", ""])
    for row in per_class[:10]:
        lines.append(
            f"- {row['class']}: {float(row['delta_f1_pp']):+.3f} pp "
            f"({float(row['baseline_f1']) * 100:.2f}->{float(row['candidate_f1']) * 100:.2f})"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

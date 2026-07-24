from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path(
    "E:/perception_outputs/rscd_surface_classification/"
    "c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709"
)
DEFAULT_ANCHOR_DIR = Path(
    "E:/perception_outputs/rscd_surface_classification/"
    "c3_farnet_official_anchor_source_reliable_router_s7_fulltest_20260708/fast_test"
)
DEFAULT_SOTA_CSV = Path("reports/paper_protocol_summary/rscd_literature_sota_protocol_audit_20260703.csv")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_per_class(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("class") or row.get("class_label")
            if not name:
                continue
            f1_key = "f1" if "f1" in row else "f1-score"
            rows[name] = {
                "precision": float(row.get("precision", 0.0) or 0.0),
                "recall": float(row.get("recall", 0.0) or 0.0),
                "f1": float(row.get(f1_key, 0.0) or 0.0),
                "support": float(row.get("support", 0.0) or 0.0),
            }
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{100.0 * delta:.2f} pp"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_read_sota(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def best_public_thresholds(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Return best aggregate public baselines from the local SOTA audit CSV.

    The CSV stores percentages, not fractions. Rows for the current project are
    excluded so that the threshold represents external papers only.
    """

    best: dict[str, dict[str, Any]] = {
        "top1": {"value": None, "method": ""},
        "macro_f1": {"value": None, "method": ""},
    }
    for row in rows:
        method = str(row.get("method", ""))
        if "Current" in method or "Ours" in method or "FAF" in method:
            continue
        fair_note = str(row.get("use_for_fair_claim", "")).lower()
        if "not comparable" in fair_note or "extra data" in fair_note or "expanded" in fair_note:
            continue
        top1 = parse_float(row.get("top1_%"))
        macro = parse_float(row.get("mean_f1_%") or row.get("macro_f1_%"))
        if top1 is not None and (best["top1"]["value"] is None or top1 > float(best["top1"]["value"])):
            best["top1"] = {"value": top1, "method": method}
        if macro is not None and (best["macro_f1"]["value"] is None or macro > float(best["macro_f1"]["value"])):
            best["macro_f1"] = {"value": macro, "method": method}
    return best


def summarize(run_dir: Path, anchor_dir: Path, out_dir: Path, sota_csv: Path, run_name: str) -> int:
    metrics_path = run_dir / "metrics.json"
    per_class_path = run_dir / "per_class_metrics.csv"
    history_path = run_dir / "history.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists() or not per_class_path.exists():
        pending = {
            "status": "pending",
            "run_dir": str(run_dir),
            "metrics_exists": metrics_path.exists(),
            "per_class_exists": per_class_path.exists(),
            "history_exists": history_path.exists(),
        }
        (out_dir / "formal_fullmanifest_status.json").write_text(
            json.dumps(pending, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(pending, ensure_ascii=False))
        return 2

    anchor_metrics_path = anchor_dir / "metrics.json"
    anchor_per_class_path = anchor_dir / "per_class_metrics.csv"
    if not anchor_metrics_path.exists() or not anchor_per_class_path.exists():
        raise FileNotFoundError(f"Missing anchor metrics under {anchor_dir}")

    run_metrics = read_json(metrics_path)
    anchor_metrics = read_json(anchor_metrics_path)
    run_summary = run_metrics["summary"]
    anchor_summary = anchor_metrics["summary"]
    run_pc = read_per_class(per_class_path)
    anchor_pc = read_per_class(anchor_per_class_path)

    labels = sorted(set(run_pc) & set(anchor_pc))
    delta_rows: list[dict[str, Any]] = []
    for label in labels:
        r = run_pc[label]
        a = anchor_pc[label]
        delta_rows.append(
            {
                "class": label,
                "run_precision_%": 100.0 * r["precision"],
                "run_recall_%": 100.0 * r["recall"],
                "run_f1_%": 100.0 * r["f1"],
                "anchor_f1_%": 100.0 * a["f1"],
                "delta_f1_pp": 100.0 * (r["f1"] - a["f1"]),
                "support": r["support"],
            }
        )
    delta_rows.sort(key=lambda row: row["delta_f1_pp"])
    write_csv(
        out_dir / "formal_fullmanifest_per_class_delta_vs_s7_anchor.csv",
        delta_rows,
        ["class", "run_precision_%", "run_recall_%", "run_f1_%", "anchor_f1_%", "delta_f1_pp", "support"],
    )

    bottom_rows = sorted(delta_rows, key=lambda row: row["run_f1_%"])[:10]
    write_csv(
        out_dir / "formal_fullmanifest_bottom_classes.csv",
        bottom_rows,
        ["class", "run_precision_%", "run_recall_%", "run_f1_%", "anchor_f1_%", "delta_f1_pp", "support"],
    )

    sota_rows = maybe_read_sota(sota_csv)
    public_thresholds = best_public_thresholds(sota_rows)
    aggregate_rows: list[dict[str, Any]] = [
        {
            "method": f"Current {run_name}",
            "top1_%": 100.0 * float(run_summary["top1"]),
            "macro_f1_%": 100.0 * float(run_summary["macro_f1"]),
            "mean_p_%": 100.0 * float(run_summary.get("mean_precision", 0.0)),
            "mean_r_%": 100.0 * float(run_summary.get("mean_recall", 0.0)),
            "num_samples": int(run_summary.get("num_samples", 0)),
            "note": "local full-test result",
        },
        {
            "method": "Previous S7 full-test anchor",
            "top1_%": 100.0 * float(anchor_summary["top1"]),
            "macro_f1_%": 100.0 * float(anchor_summary["macro_f1"]),
            "mean_p_%": 100.0 * float(anchor_summary.get("mean_precision", 0.0)),
            "mean_r_%": 100.0 * float(anchor_summary.get("mean_recall", 0.0)),
            "num_samples": int(anchor_summary.get("num_samples", 0)),
            "note": "local full-test result",
        },
    ]
    for row in sota_rows:
        aggregate_rows.append(
            {
                "method": row.get("method", ""),
                "top1_%": row.get("top1_%", ""),
                "macro_f1_%": row.get("mean_f1_%", row.get("macro_f1_%", "")),
                "mean_p_%": row.get("mean_p_%", ""),
                "mean_r_%": row.get("mean_r_%", ""),
                "num_samples": "",
                "note": row.get("use_for_fair_claim", row.get("source", "")),
            }
        )
    write_csv(
        out_dir / "formal_fullmanifest_sota_aggregate_comparison.csv",
        aggregate_rows,
        ["method", "top1_%", "macro_f1_%", "mean_p_%", "mean_r_%", "num_samples", "note"],
    )

    weakest = min(run_pc.items(), key=lambda item: item[1]["f1"])
    best_delta = max(delta_rows, key=lambda row: row["delta_f1_pp"])
    worst_delta = min(delta_rows, key=lambda row: row["delta_f1_pp"])
    run_top1_pct = 100.0 * float(run_summary["top1"])
    run_macro_pct = 100.0 * float(run_summary["macro_f1"])
    top1_threshold = public_thresholds["top1"]["value"]
    macro_threshold = public_thresholds["macro_f1"]["value"]
    top1_gap = None if top1_threshold is None else run_top1_pct - float(top1_threshold)
    macro_gap = None if macro_threshold is None else run_macro_pct - float(macro_threshold)
    md = [
        f"# {run_name} Result Summary",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Anchor dir: `{anchor_dir}`",
        f"- Test N: {int(run_summary.get('num_samples', 0))}",
        f"- Top-1: {pct(float(run_summary['top1']))} ({pp(float(run_summary['top1']) - float(anchor_summary['top1']))} vs previous S7 anchor)",
        f"- Macro-F1: {pct(float(run_summary['macro_f1']))} ({pp(float(run_summary['macro_f1']) - float(anchor_summary['macro_f1']))} vs previous S7 anchor)",
        f"- Mean-P / Mean-R: {pct(float(run_summary.get('mean_precision', 0.0)))} / {pct(float(run_summary.get('mean_recall', 0.0)))}",
        f"- Weakest class: `{weakest[0]}` F1={pct(float(weakest[1]['f1']))}, precision={pct(float(weakest[1]['precision']))}, recall={pct(float(weakest[1]['recall']))}",
        f"- Largest class gain vs anchor: `{best_delta['class']}` {best_delta['delta_f1_pp']:.2f} pp",
        f"- Largest class drop vs anchor: `{worst_delta['class']}` {worst_delta['delta_f1_pp']:.2f} pp",
        "",
        "## External SOTA Gate",
        "",
        "- Top-1 gate: "
        + (
            f"{run_top1_pct:.2f}% vs {float(top1_threshold):.2f}% ({public_thresholds['top1']['method']}), gap {top1_gap:+.2f} pp"
            if top1_threshold is not None
            else "no external Top-1 threshold found"
        ),
        "- Macro/Mean-F1 gate: "
        + (
            f"{run_macro_pct:.2f}% vs {float(macro_threshold):.2f}% ({public_thresholds['macro_f1']['method']}), gap {macro_gap:+.2f} pp"
            if macro_threshold is not None
            else "no external Macro/Mean-F1 threshold found"
        ),
        "- Interpretation: RoadFormer/RoadMamba are aggregate-only comparisons; RSPNet-L is the primary local per-class comparison when its reproduced artifacts are used.",
        "",
        "## Generated Files",
        "",
        "- `formal_fullmanifest_per_class_delta_vs_s7_anchor.csv`",
        "- `formal_fullmanifest_bottom_classes.csv`",
        "- `formal_fullmanifest_sota_aggregate_comparison.csv`",
        "",
        "## Immediate Decision Rule",
        "",
        "- If Macro-F1 and weak-class F1 improve without a large Top-1 drop, keep this as the formal main result.",
        "- If Top-1 improves but Macro-F1 drops, inspect bottom-class CSV before claiming progress.",
        "- If both regress, keep previous S7 anchor as current best and use this run only as full-training evidence.",
    ]
    (out_dir / "formal_fullmanifest_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    status = {"status": "complete", "summary_md": str(out_dir / "formal_fullmanifest_summary.md")}
    (out_dir / "formal_fullmanifest_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a formal full-manifest C3-FaRNet result.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--sota-csv", type=Path, default=DEFAULT_SOTA_CSV)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/paper_protocol_summary/formal_fullmanifest_s7_20260709"))
    parser.add_argument("--run-name", default="Formal Full-Manifest S7")
    args = parser.parse_args()
    raise SystemExit(summarize(args.run_dir, args.anchor_dir, args.out_dir, args.sota_csv, args.run_name))


if __name__ == "__main__":
    main()

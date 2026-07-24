from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


DEFAULT_OUTPUT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_REPORT_DIR = Path("reports/paper_protocol_summary")


def canonical_label(label: str) -> str:
    return str(label).strip().replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local RSCD protocol and metric definitions.")
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_val.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_test.csv"))
    parser.add_argument("--run-name", default="eval_semantic_attention_line_fourier_directed_conf001_fulltest")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def manifest_summary(path: Path) -> dict[str, object]:
    df = pd.read_csv(path, usecols=["image_path", "class_label"])
    labels_raw = sorted(df["class_label"].astype(str).unique())
    labels_canon = sorted({canonical_label(x) for x in labels_raw})
    missing = 0
    for image_path in df["image_path"].sample(min(2000, len(df)), random_state=2026):
        if not Path(str(image_path)).exists():
            missing += 1
    counts = df["class_label"].map(canonical_label).value_counts().sort_index()
    return {
        "path": str(path),
        "rows": int(len(df)),
        "raw_classes": int(len(labels_raw)),
        "canonical_classes": int(len(labels_canon)),
        "sampled_missing_2000": int(missing),
        "raw_label_examples": labels_raw[:8],
        "canonical_label_examples": labels_canon[:8],
        "class_counts": counts.to_dict(),
    }


def prediction_metrics(run_dir: Path) -> dict[str, object]:
    pred_path = run_dir / "predictions_test.csv"
    eval_path = run_dir / "evaluate_test.json"
    pred = pd.read_csv(pred_path)
    y_true = pred["true_label"].map(canonical_label)
    y_pred = pred["pred_label"].map(canonical_label)
    labels = sorted(set(y_true).union(set(y_pred)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    top1 = float((y_true == y_pred).mean())
    with eval_path.open("r", encoding="utf-8") as f:
        stored = json.load(f)
    return {
        "run_dir": str(run_dir),
        "samples": int(len(pred)),
        "classes": int(len(labels)),
        "top1": top1,
        "mean_precision": float(precision.mean()),
        "mean_recall": float(recall.mean()),
        "mean_f1": float(f1.mean()),
        "min_support": int(min(support)),
        "max_support": int(max(support)),
        "stored_summary": stored.get("summary", {}),
    }


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {
        "train": manifest_summary(args.train_manifest),
        "val": manifest_summary(args.val_manifest),
        "test": manifest_summary(args.test_manifest),
    }
    metric = prediction_metrics(args.output_root / args.run_name)
    split_rows = [
        {
            "split": split,
            "rows": summary["rows"],
            "raw_classes": summary["raw_classes"],
            "canonical_classes": summary["canonical_classes"],
            "sample_missing": summary["sampled_missing_2000"],
        }
        for split, summary in split_summaries.items()
    ]
    metric_rows = [
        {"metric": "Top-1", "value": metric["top1"] * 100.0},
        {"metric": "Mean-P", "value": metric["mean_precision"] * 100.0},
        {"metric": "Mean-R", "value": metric["mean_recall"] * 100.0},
        {"metric": "Mean-F1", "value": metric["mean_f1"] * 100.0},
    ]

    report = f"""# RSCD Protocol and Metric Audit

## Splits

{markdown_table(split_rows, ["split", "rows", "raw_classes", "canonical_classes", "sample_missing"])}

## Current Run Metrics

- Run: `{args.run_name}`
- Test samples: {metric["samples"]}
- Classes after canonicalization: {metric["classes"]}
- Support range: {metric["min_support"]} to {metric["max_support"]}

{markdown_table(metric_rows, ["metric", "value"])}

## Metric Definition

This audit computes:

- Top-1 = fraction of samples whose canonical predicted label equals the canonical true label.
- Mean-P/R/F1 = unweighted arithmetic mean of per-class precision, recall, and F1 over the 27 canonical classes.

This is the metric family reported by RSCD benchmark papers, but a formal SOTA claim still requires matching image preprocessing, split files, label canonicalization, and test set construction.

## Protocol Risks

1. Raw manifests mix hyphen and underscore label spelling; canonicalization is mandatory.
2. RSCD test support is imbalanced: some water/wet/severe classes have 800 samples, many classes have 2350.
3. Our Top-1 trails the newest RSCD papers found so far, while Mean-F1 is higher; this unusual pattern must be explained before claiming superiority.
4. RoadSaW/RoadSC are domain-shift tests, not fair RSCD SOTA training data unless a domain-generalization protocol is explicitly defined.
"""
    out = args.report_dir / "rscd_protocol_metric_audit_20260628.md"
    out.write_text(report, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

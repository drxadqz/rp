from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


DEFAULT_RUN = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\eval_local_physics_field_fulltest"
)
DEFAULT_REPORT = Path("reports/paper_protocol_summary/rscd_factor_failure_audit_latest.md")


def parse_label(label: str) -> dict[str, str]:
    text = str(label).strip().replace("-", "_")
    if text in {"ice", "fresh_snow", "melted_snow"}:
        return {"friction": text, "material": "winter", "roughness": "winter"}
    parts = text.split("_")
    if len(parts) == 2 and parts[1] in {"mud", "gravel"}:
        return {"friction": parts[0], "material": parts[1], "roughness": "granular"}
    if len(parts) >= 3:
        return {"friction": parts[0], "material": parts[1], "roughness": parts[2]}
    return {"friction": "unknown", "material": "unknown", "roughness": "unknown"}


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def factor_metrics(df: pd.DataFrame, factor: str) -> dict[str, object]:
    y_true = df[f"true_{factor}"].astype(str)
    y_pred = df[f"pred_{factor}"].astype(str)
    labels = sorted(set(y_true).union(set(y_pred)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    rows = []
    for label, pp, rr, ff, ss in zip(labels, p, r, f1, support):
        rows.append(
            {
                "label": label,
                "precision": float(pp),
                "recall": float(rr),
                "f1": float(ff),
                "support": int(ss),
            }
        )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),
        "rows": rows,
    }


def confusion_counter(df: pd.DataFrame, factor: str, limit: int = 20) -> list[dict[str, object]]:
    wrong = df[df[f"true_{factor}"] != df[f"pred_{factor}"]]
    counts = Counter(zip(wrong[f"true_{factor}"], wrong[f"pred_{factor}"]))
    return [
        {"true": str(true), "pred": str(pred), "count": int(count)}
        for (true, pred), count in counts.most_common(limit)
    ]


def hard_class_rows(df: pd.DataFrame, limit: int = 12) -> list[dict[str, object]]:
    rows = []
    grouped = df.groupby("true_label", sort=True)
    for label, part in grouped:
        rows.append(
            {
                "class": str(label),
                "support": int(len(part)),
                "accuracy": float((part["true_label"] == part["pred_label"]).mean()),
                "friction_acc": float((part["true_friction"] == part["pred_friction"]).mean()),
                "material_acc": float((part["true_material"] == part["pred_material"]).mean()),
                "roughness_acc": float((part["true_roughness"] == part["pred_roughness"]).mean()),
            }
        )
    rows.sort(key=lambda item: (item["accuracy"], item["support"]))
    return rows[:limit]


def markdown_table(rows: list[dict[str, object]], columns: list[str], percent_cols: set[str] | None = None) -> str:
    percent_cols = percent_cols or set()
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells: list[str] = []
        for col in columns:
            value = row[col]
            if col in percent_cols:
                cells.append(pct(float(value)))
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RSCD factor-level failure modes.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    pred_path = args.run_dir / "predictions_test.csv"
    df = pd.read_csv(pred_path)
    df["true_label"] = df["true_label"].astype(str).str.replace("-", "_", regex=False)
    df["pred_label"] = df["pred_label"].astype(str).str.replace("-", "_", regex=False)
    for prefix in ["true", "pred"]:
        parsed = df[f"{prefix}_label"].map(parse_label)
        for factor in ["friction", "material", "roughness"]:
            df[f"{prefix}_{factor}"] = parsed.map(lambda item, f=factor: item[f])

    metrics = {factor: factor_metrics(df, factor) for factor in ["friction", "material", "roughness"]}
    exact = float((df["true_label"] == df["pred_label"]).mean())
    factor_exact = {
        "friction_material": float(
            ((df["true_friction"] == df["pred_friction"]) & (df["true_material"] == df["pred_material"])).mean()
        ),
        "friction_roughness": float(
            ((df["true_friction"] == df["pred_friction"]) & (df["true_roughness"] == df["pred_roughness"])).mean()
        ),
        "material_roughness": float(
            ((df["true_material"] == df["pred_material"]) & (df["true_roughness"] == df["pred_roughness"])).mean()
        ),
        "all_three": exact,
    }
    wrong = df[df["true_label"] != df["pred_label"]]
    wrong_pattern = Counter()
    for _, row in wrong.iterrows():
        missed = [
            factor
            for factor in ["friction", "material", "roughness"]
            if row[f"true_{factor}"] != row[f"pred_{factor}"]
        ]
        wrong_pattern["+".join(missed)] += 1
    pattern_rows = [
        {"wrong_factors": key, "count": int(value), "share_of_errors": value / max(len(wrong), 1)}
        for key, value in wrong_pattern.most_common()
    ]

    summary_rows = [
        {
            "factor": factor,
            "accuracy": metrics[factor]["accuracy"],
            "macro_f1": metrics[factor]["macro_f1"],
        }
        for factor in ["friction", "material", "roughness"]
    ]
    pair_rows = [{"factor_pair": key, "accuracy": value} for key, value in factor_exact.items()]
    hard_rows = hard_class_rows(df)

    confusion_sections = []
    for factor in ["friction", "material", "roughness"]:
        confusion_sections.extend(
            [
                f"### {factor} top confusions",
                "",
                markdown_table(confusion_counter(df, factor), ["true", "pred", "count"]),
                "",
            ]
        )

    report = "\n".join(
        [
            "# RSCD Factor Failure Audit",
            "",
            f"- Run: `{args.run_dir.name}`",
            f"- Samples: {len(df)}",
            f"- Exact 27-class Top-1: {pct(exact)}",
            "",
            "## Factor-Level Metrics",
            "",
            markdown_table(summary_rows, ["factor", "accuracy", "macro_f1"], {"accuracy", "macro_f1"}),
            "",
            "## Coupled Factor Accuracy",
            "",
            markdown_table(pair_rows, ["factor_pair", "accuracy"], {"accuracy"}),
            "",
            "## Error Factor Patterns",
            "",
            markdown_table(pattern_rows, ["wrong_factors", "count", "share_of_errors"], {"share_of_errors"}),
            "",
            "## Lowest Exact Classes",
            "",
            markdown_table(
                hard_rows,
                ["class", "support", "accuracy", "friction_acc", "material_acc", "roughness_acc"],
                {"accuracy", "friction_acc", "material_acc", "roughness_acc"},
            ),
            "",
            *confusion_sections,
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    payload = {
        "run_dir": str(args.run_dir),
        "samples": int(len(df)),
        "exact_top1": exact,
        "factor_metrics": metrics,
        "factor_pair_accuracy": factor_exact,
        "error_patterns": pattern_rows,
        "lowest_classes": hard_rows,
    }
    args.report.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()

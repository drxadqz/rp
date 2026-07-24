from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from friction_affordance.ontology import RISK, TASKS, infer_record, label_to_index, risk_from_mu_interval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(path, dtype=str, low_memory=False) for path in args.test_manifest], ignore_index=True)
    result = evaluate(df)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


def evaluate(df: pd.DataFrame) -> dict[str, Any]:
    predictions: dict[str, list[int]] = {task: [] for task in TASKS}
    targets: dict[str, list[int]] = {task: [] for task in TASKS}
    datasets: list[str] = []
    pred_low = []
    pred_high = []
    target_low = []
    target_high = []
    risk_true = []
    risk_pred = []

    for _, row in df.iterrows():
        dataset = str(row.get("dataset", ""))
        record = infer_record(dataset, str(row.get("class_label", "")))
        fields = {
            "friction": record.friction,
            "material": record.material,
            "unevenness": record.unevenness,
            "wetness": record.wetness,
            "snow": record.snow,
            "risk": record.risk,
        }
        for task, pred_label in fields.items():
            true_idx = label_to_index(task, row.get(f"{task}_label"))
            pred_idx = label_to_index(task, pred_label)
            if true_idx >= 0 and pred_idx >= 0:
                targets[task].append(true_idx)
                predictions[task].append(pred_idx)

        mu_low = maybe_float(row.get("mu_low"))
        mu_high = maybe_float(row.get("mu_high"))
        if mu_low is not None and mu_high is not None and record.mu_low is not None and record.mu_high is not None:
            target_low.append(mu_low)
            target_high.append(mu_high)
            pred_low.append(float(record.mu_low))
            pred_high.append(float(record.mu_high))
            datasets.append(dataset)
            true_risk_idx = label_to_index("risk", row.get("risk_label"))
            pred_risk_label = risk_from_mu_interval(record.mu_low, record.mu_high)
            pred_risk_idx = label_to_index("risk", pred_risk_label)
            if true_risk_idx >= 0 and pred_risk_idx >= 0:
                risk_true.append(true_risk_idx)
                risk_pred.append(pred_risk_idx)

    out: dict[str, Any] = {"num_samples": int(len(df)), "tasks": {}}
    for task in TASKS:
        if targets[task]:
            out["tasks"][task] = classification_summary(targets[task], predictions[task])
    out["low_friction_detection"] = low_friction_summary(risk_true, risk_pred)
    out["mu_interval"] = interval_summary(pred_low, pred_high, target_low, target_high)
    out["note"] = (
        "Ontology-oracle baseline uses dataset/class_label at test time. It is not a fair visual classifier baseline; "
        "it is a sanity-check upper bound for the weak label-to-interval ontology."
    )
    return out


def classification_summary(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    return {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def low_friction_summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if not y_true:
        return {}
    true = [idx >= RISK.index("high") for idx in y_true]
    pred = [idx >= RISK.index("high") for idx in y_pred]
    return {
        "positive_definition": "risk in {high, very_high}",
        "recall": float(recall_score(true, pred, zero_division=0)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
    }


def interval_summary(
    pred_low: list[float],
    pred_high: list[float],
    target_low: list[float],
    target_high: list[float],
) -> dict[str, float | int]:
    if not pred_low:
        return {}
    covers = [
        pl <= tl and ph >= th
        for pl, ph, tl, th in zip(pred_low, pred_high, target_low, target_high)
    ]
    widths = [ph - pl for pl, ph in zip(pred_low, pred_high)]
    mid_mae = [
        abs(0.5 * (pl + ph) - 0.5 * (tl + th))
        for pl, ph, tl, th in zip(pred_low, pred_high, target_low, target_high)
    ]
    return {
        "num_samples": int(len(pred_low)),
        "coverage": float(sum(covers) / len(covers)),
        "width_mean": float(sum(widths) / len(widths)),
        "mean_mae_to_interval_mid": float(sum(mid_mae) / len(mid_mae)),
    }


def maybe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()

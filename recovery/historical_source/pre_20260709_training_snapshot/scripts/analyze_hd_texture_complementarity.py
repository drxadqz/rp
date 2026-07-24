"""Analyze whether HD texture factor probes complement the current RSCD model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


WINTER = {"fresh_snow", "melted_snow", "ice"}
PAVED_MATERIALS = {"asphalt", "concrete"}
PAVED_ROUGHNESS = {"smooth", "slight", "severe"}
GRANULAR_MATERIALS = {"mud", "gravel"}


def factor_text(label: str) -> dict[str, str]:
    text = str(label).replace("-", "_")
    if text in WINTER:
        return {"friction": text, "material": "winter", "roughness": "winter"}
    parts = text.split("_")
    if len(parts) == 2:
        return {"friction": parts[0], "material": parts[1], "roughness": "granular"}
    if len(parts) >= 3:
        return {"friction": parts[0], "material": parts[1], "roughness": parts[2]}
    return {"friction": "unknown", "material": "unknown", "roughness": "unknown"}


def compose_label(friction: str, material: str, roughness: str, valid: set[str], fallback: str) -> str:
    if friction in WINTER:
        label = friction
    elif material in GRANULAR_MATERIALS and friction in {"dry", "wet", "water"}:
        label = f"{friction}_{material}"
    elif material in PAVED_MATERIALS and roughness in PAVED_ROUGHNESS and friction in {"dry", "wet", "water"}:
        label = f"{friction}_{material}_{roughness}"
    else:
        return fallback
    return label if label in valid else fallback


def factor_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    parsed = df[source].map(factor_text)
    for name in ("friction", "material", "roughness"):
        df[f"{source}_{name}"] = parsed.map(lambda item, n=name: item[n])
    return df


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    labels = sorted(set(y_true.astype(str)) | set(y_pred.astype(str)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "top1": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
    }


def train_factor_model(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> ExtraTreesClassifier:
    clf = ExtraTreesClassifier(
        n_estimators=260,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument(
        "--current-predictions",
        type=Path,
        default=Path(
            "D:/NMI_SPWFM_datasets/friction_affordance_outputs/rscd_surface_classification/"
            "eval_dry_concrete_vor_residual_scale012_fulltest/predictions_test.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.probe_dir / "train_sample.csv")
    test_df = pd.read_csv(args.probe_dir / "test_sample.csv")
    x_train = np.load(args.probe_dir / "x_train.npy")
    x_test = np.load(args.probe_dir / "x_test.npy")
    current = pd.read_csv(args.current_predictions)

    current["image_path_norm"] = current["image_path"].astype(str)
    test_df["image_path_norm"] = test_df["image_path"].astype(str)
    merged = test_df.merge(
        current[["image_path_norm", "pred_label", "confidence"]],
        on="image_path_norm",
        how="left",
        validate="one_to_one",
    )
    if merged["pred_label"].isna().any():
        missing = int(merged["pred_label"].isna().sum())
        raise ValueError(f"current predictions missing {missing} sampled test images")

    for frame in (train_df, merged):
        frame["true_label"] = frame["class_label"].astype(str).str.replace("-", "_", regex=False)
        factor_columns(frame, "true_label")
    factor_columns(merged, "pred_label")

    hd_pred_cols: dict[str, str] = {}
    hd_conf_cols: dict[str, str] = {}
    for factor in ("friction", "material", "roughness"):
        y_train = train_df[f"true_label_{factor}"].astype(str).to_numpy()
        clf = train_factor_model(x_train, y_train, int(args.seed))
        probs = clf.predict_proba(x_test)
        pred_idx = probs.argmax(axis=1)
        pred = clf.classes_[pred_idx]
        conf = probs[np.arange(len(probs)), pred_idx]
        pred_col = f"hd_{factor}"
        conf_col = f"hd_{factor}_confidence"
        merged[pred_col] = pred
        merged[conf_col] = conf
        hd_pred_cols[factor] = pred_col
        hd_conf_cols[factor] = conf_col

    valid_labels = set(merged["true_label"].astype(str).unique())
    baseline = metrics(merged["true_label"], merged["pred_label"])
    payload: dict[str, object] = {
        "probe_dir": str(args.probe_dir),
        "current_predictions": str(args.current_predictions),
        "samples": int(len(merged)),
        "baseline": baseline,
        "factor_complementarity": {},
        "selective_factor_overwrite": [],
    }

    rows = []
    for factor in ("friction", "material", "roughness"):
        true = merged[f"true_label_{factor}"].astype(str)
        current_factor = merged[f"pred_label_{factor}"].astype(str)
        hd_factor = merged[hd_pred_cols[factor]].astype(str)
        cur_correct = current_factor.eq(true)
        hd_correct = hd_factor.eq(true)
        comp = {
            "current_factor_accuracy": float(cur_correct.mean()),
            "hd_factor_accuracy": float(hd_correct.mean()),
            "both_correct": float((cur_correct & hd_correct).mean()),
            "current_wrong_hd_correct": float((~cur_correct & hd_correct).mean()),
            "current_correct_hd_wrong": float((cur_correct & ~hd_correct).mean()),
            "both_wrong": float((~cur_correct & ~hd_correct).mean()),
            "hd_recovery_share_of_current_errors": float(((~cur_correct & hd_correct).sum()) / max((~cur_correct).sum(), 1)),
            "hd_damage_share_of_current_correct": float(((cur_correct & ~hd_correct).sum()) / max(cur_correct.sum(), 1)),
        }
        payload["factor_complementarity"][factor] = comp
        row = {"factor": factor, **comp}
        rows.append(row)

    thresholds_current = [0.50, 0.60, 0.70, 0.80, 0.90]
    thresholds_hd = [0.35, 0.45, 0.55, 0.65, 0.75]
    for factor in ("friction", "material", "roughness"):
        for t_cur in thresholds_current:
            for t_hd in thresholds_hd:
                trial = merged.copy()
                apply_mask = (trial["confidence"].astype(float) <= t_cur) & (
                    trial[hd_conf_cols[factor]].astype(float) >= t_hd
                )
                labels = []
                for _, row in trial.iterrows():
                    parts = factor_text(str(row["pred_label"]))
                    parts[factor] = str(row[hd_pred_cols[factor]])
                    labels.append(
                        compose_label(
                            parts["friction"],
                            parts["material"],
                            parts["roughness"],
                            valid_labels,
                            fallback=str(row["pred_label"]),
                        )
                    )
                trial_pred = pd.Series(np.where(apply_mask.to_numpy(), labels, trial["pred_label"].astype(str)), index=trial.index)
                m = metrics(trial["true_label"], trial_pred)
                payload["selective_factor_overwrite"].append(
                    {
                        "factor": factor,
                        "current_conf_le": float(t_cur),
                        "hd_conf_ge": float(t_hd),
                        "coverage": float(apply_mask.mean()),
                        "changed": int(apply_mask.sum()),
                        "top1": m["top1"],
                        "macro_f1": m["macro_f1"],
                        "delta_top1": m["top1"] - baseline["top1"],
                        "delta_macro_f1": m["macro_f1"] - baseline["macro_f1"],
                    }
                )

    merged.to_csv(args.output_dir / "hd_texture_factor_predictions_with_current.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "hd_texture_complementarity_rows.csv", index=False)
    selective = pd.DataFrame(payload["selective_factor_overwrite"]).sort_values(
        ["delta_macro_f1", "delta_top1"], ascending=False
    )
    selective.to_csv(args.output_dir / "hd_texture_selective_overwrite_grid.csv", index=False)
    with open(args.output_dir / "hd_texture_complementarity.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    top_rows = selective.head(12)
    md = [
        "# HD Texture Complementarity Audit",
        "",
        f"- samples: {len(merged)}",
        f"- baseline Top-1: {baseline['top1'] * 100:.4f}%",
        f"- baseline Macro-F1: {baseline['macro_f1'] * 100:.4f}%",
        "",
        "## Factor Complementarity",
        "",
        pd.DataFrame(rows).to_markdown(index=False) if False else _simple_md_table(pd.DataFrame(rows)),
        "",
        "## Best Selective Factor Overwrite Trials",
        "",
        _simple_md_table(top_rows),
    ]
    (args.output_dir / "hd_texture_complementarity.md").write_text("\n".join(md), encoding="utf-8")


def _simple_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import PureWindowsPath
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse=False)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _safe_numeric_columns(df: pd.DataFrame, skip: set[str], mask: pd.Series | None = None) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col in skip:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        completeness = converted.loc[mask].notna().mean() if mask is not None else converted.notna().mean()
        if completeness > 0.95:
            df[col] = converted.astype(np.float32)
            out.append(col)
    return out


def _factor_tuple(label: str) -> tuple[str, str, str]:
    label = str(label).replace("-", "_")
    parts = label.split("_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return (label, "none", "none")
    friction = parts[0] if parts else "none"
    if len(parts) == 3:
        return (friction, parts[1], parts[2])
    if len(parts) == 2:
        return (friction, parts[1], "none")
    return (friction, "none", "none")


def _add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    strict_factors = df["strict_pred"].map(_factor_tuple)
    wet_factors = df["wet_pred"].map(_factor_tuple)
    for idx, axis in enumerate(["friction", "material", "roughness"]):
        df[f"pred_diff_{axis}"] = [
            int(a[idx] != b[idx]) for a, b in zip(strict_factors, wet_factors, strict=False)
        ]
    df["strict_pred_factor"] = ["|".join(x) for x in strict_factors]
    df["wet_pred_factor"] = ["|".join(x) for x in wet_factors]
    df["pred_pair"] = df["strict_pred"].astype(str) + "=>" + df["wet_pred"].astype(str)
    return df


def _make_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    preprocess_scaled = ColumnTransformer(
        [
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            ("cat", _one_hot_encoder(), categorical),
        ],
        remainder="drop",
    )
    preprocess_raw = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", _one_hot_encoder(), categorical),
        ],
        remainder="drop",
    )
    return {
        "logistic_l2": Pipeline(
            [
                ("prep", preprocess_scaled),
                ("clf", LogisticRegression(max_iter=2000, C=0.8, class_weight="balanced", solver="lbfgs")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", preprocess_raw),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=6,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=1701,
                    ),
                ),
            ]
        ),
        "hist_gradient": Pipeline(
            [
                ("prep", preprocess_scaled),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.04,
                        max_leaf_nodes=12,
                        l2_regularization=0.08,
                        random_state=1701,
                    ),
                ),
            ]
        ),
    }


def _score_predictions(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def _classification_metrics(labels: list[str], pred: list[str], class_order: list[str]) -> dict[str, Any]:
    label_to_idx = {name: idx for idx, name in enumerate(class_order)}
    y_true = np.asarray([label_to_idx[x] for x in labels], dtype=np.int64)
    y_pred = np.asarray([label_to_idx[x] for x in pred], dtype=np.int64)
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_order)),
        zero_division=0,
    )
    return {
        "top1": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),
        "num_errors": int((y_true != y_pred).sum()),
        "per_class": {
            cls: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(class_order)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--feature-values", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--feature-mode",
        choices=["all", "values_only", "categorical_only"],
        default="all",
        help="all uses image values plus candidate prediction metadata; values_only removes prediction categories/confidences.",
    )
    parser.add_argument(
        "--group-prefix-len",
        type=int,
        default=0,
        help="If positive, use GroupKFold by the first N characters of the image filename to reduce near-duplicate leakage.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comp = _read_csv(args.comparison)
    feats = _read_csv(args.feature_values)
    comp["strict_ok"] = _bool_series(comp["strict_ok"])
    comp["wet_ok"] = _bool_series(comp["wet_ok"])
    comp["image_key"] = comp["image_path"].astype(str).str.lower()
    feats["image_key"] = feats["image_path"].astype(str).str.lower()

    merged = comp.merge(feats, on="image_key", how="left", suffixes=("", "_feat"))
    merged = _add_pair_features(merged)
    merged["disagree"] = merged["strict_pred"].astype(str) != merged["wet_pred"].astype(str)
    merged["one_correct"] = merged["strict_ok"] ^ merged["wet_ok"]
    merged["wet_better"] = (merged["wet_ok"] & ~merged["strict_ok"]).astype(np.int64)

    skip = {
        "image_path",
        "image_path_feat",
        "image_key",
        "true_label",
        "label",
        "pred_label",
        "strict_pred",
        "wet_pred",
        "case",
        "strict_ok",
        "wet_ok",
        "strict_ok_feat",
        "wet_ok_feat",
        "candidate_ok",
        "candidate_ok_feat",
        "disagree",
        "one_correct",
        "wet_better",
    }
    train_mask = merged["disagree"] & merged["one_correct"]
    numeric = _safe_numeric_columns(merged, skip, train_mask)
    leakage_patterns = ("true_match", "_ok", "one_correct", "wet_better")
    numeric = [col for col in numeric if not any(pattern in col for pattern in leakage_patterns)]
    categorical = ["strict_pred", "wet_pred", "strict_pred_factor", "wet_pred_factor", "pred_pair"]
    prediction_numeric = {"strict_conf", "wet_conf", "conf_delta", "base_confidence"}
    prediction_numeric.update({f"pred_diff_{axis}" for axis in ["friction", "material", "roughness"]})
    if args.feature_mode == "values_only":
        numeric = [col for col in numeric if col not in prediction_numeric]
        categorical = []
    elif args.feature_mode == "categorical_only":
        numeric = []

    train_df = merged[train_mask & merged[numeric].notna().all(axis=1)].copy()
    y = train_df["wet_better"].to_numpy(dtype=np.int64)
    x = train_df[numeric + categorical].copy()
    groups = None
    num_groups = 0
    if args.group_prefix_len > 0:
        groups = (
            train_df["image_path"]
            .astype(str)
            .map(lambda value: PureWindowsPath(value).name[: args.group_prefix_len])
            .to_numpy()
        )
        num_groups = int(len(set(groups.tolist())))
        folds = min(int(args.folds), num_groups) if len(y) else 0
    else:
        folds = min(int(args.folds), int(np.bincount(y).min())) if len(y) else 0
    if folds < 2:
        raise RuntimeError("Not enough one-correct disagreement samples with features for cross-validation.")

    cv = GroupKFold(n_splits=folds) if groups is not None else StratifiedKFold(n_splits=folds, shuffle=True, random_state=1701)
    models = _make_models(numeric, categorical)
    model_rows: list[dict[str, Any]] = []
    best_name = ""
    best_score = -1.0
    oof_by_model: dict[str, np.ndarray] = {}
    for name, model in models.items():
        oof = cross_val_predict(model, x, y, cv=cv, groups=groups, method="predict")
        oof_by_model[name] = oof.astype(np.int64)
        scores = _score_predictions(y, oof)
        row = {"model": name, "n": int(len(y)), "folds": int(folds), **scores}
        model_rows.append(row)
        if scores["macro_f1"] > best_score:
            best_name = name
            best_score = scores["macro_f1"]

    best_oof = oof_by_model[best_name]
    train_df["arbiter_choose_wet_oof"] = best_oof
    train_df["arbiter_correct_oof"] = (best_oof == y).astype(np.int64)

    routed = merged.copy()
    routed["arbiter_pred"] = routed["strict_pred"].astype(str)
    routed["arbiter_source"] = "strict_default"
    oof_map = dict(zip(train_df["image_key"], train_df["arbiter_choose_wet_oof"], strict=False))
    for idx, row in routed.iterrows():
        key = row["image_key"]
        if key in oof_map and int(oof_map[key]) == 1:
            routed.at[idx, "arbiter_pred"] = str(row["wet_pred"])
            routed.at[idx, "arbiter_source"] = f"{best_name}_oof_wet"
        elif key in oof_map:
            routed.at[idx, "arbiter_source"] = f"{best_name}_oof_strict"

    class_order = sorted(set(routed["true_label"].astype(str)) | set(routed["strict_pred"].astype(str)) | set(routed["wet_pred"].astype(str)))
    strict_metrics = _classification_metrics(
        routed["true_label"].astype(str).tolist(),
        routed["strict_pred"].astype(str).tolist(),
        class_order,
    )
    wet_metrics = _classification_metrics(
        routed["true_label"].astype(str).tolist(),
        routed["wet_pred"].astype(str).tolist(),
        class_order,
    )
    arbiter_metrics = _classification_metrics(
        routed["true_label"].astype(str).tolist(),
        routed["arbiter_pred"].astype(str).tolist(),
        class_order,
    )

    # Fit the best linear model once on all one-correct disagreements for coefficient inspection.
    top_features: list[dict[str, Any]] = []
    if best_name == "logistic_l2":
        final_model = models[best_name]
        final_model.fit(x, y)
        prep = final_model.named_steps["prep"]
        clf = final_model.named_steps["clf"]
        names = list(prep.get_feature_names_out())
        coefs = clf.coef_.reshape(-1)
        order = np.argsort(np.abs(coefs))[::-1][:40]
        for pos in order:
            top_features.append({"feature": names[int(pos)], "coef": float(coefs[int(pos)])})

    summary = {
        "note": "Diagnostic only: this uses full-test predictions/features with cross-validation to detect an arbitration signal; it is not an official publishable test metric. Input features exclude true-label-derived match indicators.",
        "num_rows": int(len(merged)),
        "num_disagreements": int(merged["disagree"].sum()),
        "num_one_correct_disagreements": int(merged["one_correct"].sum()),
        "num_one_correct_with_features": int(len(train_df)),
        "one_correct_wet_better": int(y.sum()),
        "one_correct_strict_better": int((1 - y).sum()),
        "best_model": best_name,
        "selector_cv": model_rows,
        "strict": {k: strict_metrics[k] for k in ["top1", "macro_f1", "num_errors"]},
        "wetfamily": {k: wet_metrics[k] for k in ["top1", "macro_f1", "num_errors"]},
        "arbiter_oof": {k: arbiter_metrics[k] for k in ["top1", "macro_f1", "num_errors"]},
        "num_numeric_features_used": int(len(numeric)),
        "categorical_features_used": categorical,
        "feature_mode": args.feature_mode,
        "group_prefix_len": int(args.group_prefix_len),
        "num_groups": num_groups,
        "top_logistic_features": top_features,
    }

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(model_rows).to_csv(args.out_dir / "selector_cv_metrics.csv", index=False, encoding="utf-8")
    train_df.to_csv(args.out_dir / "one_correct_disagreement_oof.csv", index=False, encoding="utf-8")
    routed[
        [
            "image_path",
            "true_label",
            "strict_pred",
            "wet_pred",
            "strict_ok",
            "wet_ok",
            "arbiter_pred",
            "arbiter_source",
            "case",
            "conf_delta",
        ]
    ].to_csv(args.out_dir / "arbiter_predictions.csv", index=False, encoding="utf-8")

    report = [
        "# Strict-Best vs Wet/Water-Family Arbiter Diagnostic",
        "",
        "This is diagnostic only. It uses full-test predictions with cross-validation to ask whether image-derived values contain an expert-selection signal.",
        "",
        f"- Disagreements: {summary['num_disagreements']}",
        f"- One-correct disagreements: {summary['num_one_correct_disagreements']}",
        f"- One-correct disagreements with feature rows: {summary['num_one_correct_with_features']}",
        f"- Best selector: {best_name}",
        "",
        "## Selector CV",
        "",
        "| model | Acc | Balanced Acc | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in model_rows:
        report.append(
            f"| {row['model']} | {row['acc'] * 100:.2f} | {row['balanced_acc'] * 100:.2f} | {row['macro_f1'] * 100:.2f} |"
        )
    report += [
        "",
        "## Routed Full Prediction Diagnostic",
        "",
        "| prediction source | Top-1 | Macro-F1 | errors |",
        "|---|---:|---:|---:|",
        f"| strict best | {strict_metrics['top1'] * 100:.4f} | {strict_metrics['macro_f1'] * 100:.4f} | {strict_metrics['num_errors']} |",
        f"| wet/water-family candidate | {wet_metrics['top1'] * 100:.4f} | {wet_metrics['macro_f1'] * 100:.4f} | {wet_metrics['num_errors']} |",
        f"| OOF arbiter diagnostic | {arbiter_metrics['top1'] * 100:.4f} | {arbiter_metrics['macro_f1'] * 100:.4f} | {arbiter_metrics['num_errors']} |",
        "",
        "## Decision Hint",
        "",
    ]
    if arbiter_metrics["num_errors"] < strict_metrics["num_errors"]:
        report.append("- A learnable arbitration signal may exist. Next step: distill the selector into a single-model mechanism route and verify on validation/full test without test-set leakage.")
    else:
        report.append("- The available value/categorical features do not produce a reliable arbitration gain. Do not build a two-expert selector from this evidence alone.")
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

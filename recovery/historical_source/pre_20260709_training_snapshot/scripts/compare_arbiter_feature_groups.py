from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _auc_abs(a: np.ndarray, b: np.ndarray) -> float:
    values = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(len(a), dtype=np.int8), np.ones(len(b), dtype=np.int8)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    n0 = float((labels == 0).sum())
    n1 = float((labels == 1).sum())
    auc = (float(ranks[labels == 1].sum()) - n1 * (n1 + 1.0) / 2.0) / max(n0 * n1, EPS)
    return max(float(auc), 1.0 - float(auc))


def _cohen_abs(a: np.ndarray, b: np.ndarray) -> float:
    va = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    vb = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    pooled = math.sqrt(max(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1), EPS))
    return abs(float((b.mean() - a.mean()) / pooled))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--feature-values", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    comp = pd.read_csv(args.comparison, encoding="utf-8-sig")
    feats = pd.read_csv(args.feature_values, encoding="utf-8-sig")
    comp["strict_ok"] = _as_bool(comp["strict_ok"])
    comp["wet_ok"] = _as_bool(comp["wet_ok"])
    comp["image_key"] = comp["image_path"].astype(str).str.lower()
    feats["image_key"] = feats["image_path"].astype(str).str.lower()
    merged = comp.merge(feats, on="image_key", how="inner", suffixes=("", "_feat"))

    mask = (merged["strict_pred"].astype(str) != merged["wet_pred"].astype(str)) & (
        merged["strict_ok"] ^ merged["wet_ok"]
    )
    rows = merged[mask].copy()
    rows["wet_candidate_better"] = (rows["wet_ok"] & ~rows["strict_ok"]).astype(np.int64)
    y = rows["wet_candidate_better"].to_numpy(dtype=np.int64)

    skip = {
        "image_path",
        "image_path_feat",
        "image_key",
        "true_label",
        "label",
        "label_feat",
        "pred_label",
        "pred_label_feat",
        "strict_pred",
        "wet_pred",
        "candidate_label",
        "strict_ok",
        "wet_ok",
        "strict_ok_feat",
        "candidate_ok",
        "case",
        "case_feat",
        "wet_candidate_better",
    }
    out_rows: list[dict[str, object]] = []
    for col in rows.columns:
        if col in skip:
            continue
        values = pd.to_numeric(rows[col], errors="coerce")
        if values.notna().mean() < 0.95:
            continue
        strict_better_values = values[y == 0].to_numpy(dtype=np.float64)
        wet_better_values = values[y == 1].to_numpy(dtype=np.float64)
        if len(strict_better_values) < 2 or len(wet_better_values) < 2:
            continue
        out_rows.append(
            {
                "feature": col,
                "strict_better_mean": float(strict_better_values.mean()),
                "wet_candidate_better_mean": float(wet_better_values.mean()),
                "larger_when": "wet_candidate_better"
                if wet_better_values.mean() > strict_better_values.mean()
                else "strict_better",
                "cohen_d_abs": _cohen_abs(strict_better_values, wet_better_values),
                "auc_abs": _auc_abs(strict_better_values, wet_better_values),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["cohen_d_abs", "auc_abs"], ascending=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(out.head(25).to_string(index=False))
    print(f"wrote {len(out)} feature comparisons to {args.out_csv}")


if __name__ == "__main__":
    main()

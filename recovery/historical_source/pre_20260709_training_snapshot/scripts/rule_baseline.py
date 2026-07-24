from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, action="append", default=[])
    parser.add_argument("--eval-manifest", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.train_manifest or args.eval_manifest:
        if not args.train_manifest or not args.eval_manifest:
            raise SystemExit("Use both --train-manifest and --eval-manifest.")
        train = _read_manifests(args.train_manifest)
        eval_df = _read_manifests(args.eval_manifest)
    elif args.manifest:
        # Backward-compatible mode. Prefer train/eval manifests for paper tables.
        train = _read_manifests([args.manifest])
        eval_df = train.copy()
    else:
        raise SystemExit("Provide --manifest or --train-manifest/--eval-manifest.")

    class_stats = (
        train.groupby("class_label")[["mu_low", "mu_high"]]
        .median()
        .rename(columns={"mu_low": "pred_low", "mu_high": "pred_high"})
    )
    global_interval = train[["mu_low", "mu_high"]].median().rename({"mu_low": "pred_low", "mu_high": "pred_high"})
    pred = eval_df.join(class_stats, on="class_label")
    pred["pred_low"] = pred["pred_low"].fillna(float(global_interval["pred_low"]))
    pred["pred_high"] = pred["pred_high"].fillna(float(global_interval["pred_high"]))

    metrics = {
        "fit_manifests": [str(path) for path in (args.train_manifest or ([args.manifest] if args.manifest else []))],
        "eval_manifests": [str(path) for path in (args.eval_manifest or ([args.manifest] if args.manifest else []))],
        "fit_num_samples": int(len(train)),
        "eval": _summarize(pred),
        "by_dataset": {
            str(dataset): _summarize(group)
            for dataset, group in pred.groupby("dataset", dropna=False)
        },
        "note": "Rule baseline fits class_label median weak-friction intervals on train manifests and evaluates on held-out manifests. It is not a visual model.",
    }
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


def _read_manifests(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype=str, low_memory=False) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    has_mu = df["mu_low"].notna() & df["mu_high"].notna()
    df = df[has_mu].copy()
    if df.empty:
        raise SystemExit("No mu intervals in manifests.")
    df["mu_low"] = df["mu_low"].astype(float)
    df["mu_high"] = df["mu_high"].astype(float)
    return df


def _summarize(pred: pd.DataFrame) -> dict[str, float | int]:
    covers = (pred["pred_low"] <= pred["mu_low"]) & (pred["pred_high"] >= pred["mu_high"])
    widths = pred["pred_high"] - pred["pred_low"]
    pred_mid = 0.5 * (pred["pred_low"] + pred["pred_high"])
    target_mid = 0.5 * (pred["mu_low"] + pred["mu_high"])
    return {
        "num_samples": int(len(pred)),
        "coverage": float(covers.mean()),
        "avg_width": float(widths.mean()),
        "median_width": float(widths.median()),
        "mid_mae": float((pred_mid - target_mid).abs().mean()),
    }


if __name__ == "__main__":
    main()

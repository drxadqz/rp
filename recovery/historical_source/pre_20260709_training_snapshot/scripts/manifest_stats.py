from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifests = [*args.manifests, *args.manifest]
    if not manifests:
        raise SystemExit("At least one manifest path is required.")

    frames = []
    for path in manifests:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df["manifest"] = str(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    summary = {
        "num_samples": int(len(df)),
        "by_dataset": df["dataset"].value_counts(dropna=False).to_dict(),
        "by_split": df["split"].value_counts(dropna=False).to_dict(),
        "by_class": df["class_label"].value_counts(dropna=False).head(100).to_dict(),
        "missing_ratio": {
            col: float(df[col].isna().mean())
            for col in [
                "friction_label",
                "material_label",
                "unevenness_label",
                "wetness_label",
                "snow_label",
                "risk_label",
                "mu_low",
                "mu_high",
            ]
            if col in df.columns
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

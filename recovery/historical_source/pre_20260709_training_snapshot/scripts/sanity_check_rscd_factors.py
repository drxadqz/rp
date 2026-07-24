from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.rscd_factors import canonical_class_label, sanity_summary  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data" / "manifests_full" / "rscd_prepared_train.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check RSCD factor parsing and hard-pair construction.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    df = pd.read_csv(args.manifest, usecols=["class_label"], dtype=str, low_memory=False)
    labels = sorted({canonical_class_label(v) for v in df["class_label"].dropna().astype(str)})
    class_to_idx = {name: idx for idx, name in enumerate(labels)}
    print(sanity_summary(class_to_idx))


if __name__ == "__main__":
    main()

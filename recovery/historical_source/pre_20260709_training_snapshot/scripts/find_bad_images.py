from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageFile
from tqdm import tqdm

from friction_affordance.datasets import ManifestDataset
from friction_affordance.utils import load_yaml


ImageFile.LOAD_TRUNCATED_IMAGES = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    manifests = data_cfg[f"{args.split}_manifests"]
    ds = ManifestDataset(
        manifests,
        transform=None,
        max_samples=data_cfg.get(f"max_{args.split}_samples"),
        max_samples_per_dataset=data_cfg.get(f"max_{args.split}_samples_per_dataset"),
        max_samples_per_class=data_cfg.get(f"max_{args.split}_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)) + (0 if args.split == "train" else 1),
    )

    bad_rows = []
    for i, row in tqdm(ds.df.iterrows(), total=len(ds.df), desc=f"scan-{args.split}", ascii=True):
        path = Path(str(row["image_path"]))
        try:
            with Image.open(path) as img:
                img.convert("RGB").load()
        except (OSError, SyntaxError, ValueError) as exc:
            bad_rows.append(
                {
                    "index": i,
                    "image_path": str(path),
                    "dataset": row.get("dataset", ""),
                    "class_label": row.get("class_label", ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    print(f"scanned: {len(ds.df)}")
    print(f"bad_images: {len(bad_rows)}")
    for row in bad_rows[:20]:
        print(f"{row['dataset']} {row['class_label']} {row['image_path']} ({row['error_type']}: {row['error']})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "image_path", "dataset", "class_label", "error_type", "error"],
            )
            writer.writeheader()
            writer.writerows(bad_rows)
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()

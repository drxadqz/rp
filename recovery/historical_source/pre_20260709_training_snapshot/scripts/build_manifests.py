from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.datasets.scanners import scan_imagefolder, scan_rscd_prepared, split_and_write_manifests
from friction_affordance.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/local_paths.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, ds in cfg.get("datasets", {}).items():
        dtype = ds.get("type")
        print(f"Building manifest for {name} ({dtype})")
        if dtype == "rscd_prepared":
            df = scan_rscd_prepared(ds["labels_csv"], max_per_class=args.max_per_class)
        elif dtype == "imagefolder":
            df = scan_imagefolder(ds["root"], ds.get("dataset_name", name), max_per_class=args.max_per_class)
        elif dtype == "rscd_raw":
            print("  Skipping rscd_raw because rscd_prepared labels are available and richer.")
            continue
        else:
            print(f"  Unknown dataset type {dtype}; skipped.")
            continue
        written = split_and_write_manifests(df, args.out_dir / name)
        for path in written:
            print(f"  wrote {path} ({sum(1 for _ in open(path, encoding='utf-8')) - 1} rows)")


if __name__ == "__main__":
    main()


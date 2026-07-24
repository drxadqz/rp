from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml


DEFAULT_CANDIDATES = {
    "rscd_prepared": [
        r"D:\NMI_SPWFM_datasets\friction_affordance_data\RSCD_prepared\official_friction\labels.csv",
        r"E:\datasets\RSCD_prepared\official_friction\labels.csv",
        r"E:\datasets\RSCD_prepared\hf_friction_ready\labels.csv",
    ],
    "rscd_raw": [
        r"D:\NMI_SPWFM_datasets\friction_affordance_data\RSCD_raw\RSCD dataset-1million",
        r"E:\datasets\RSCD_raw\RSCD dataset-1million",
    ],
    "roadsaw": [
        r"D:\NMI_SPWFM_datasets\friction_affordance_data\RoadSaW-150_s",
        r"E:\datasets\WheelCorridor\raw\RoadSaW-150_s\RoadSaW-150_s",
    ],
    "roadsc": [
        r"D:\NMI_SPWFM_datasets\friction_affordance_data\RoadSC-balanced_to_RoadSaW12-150_l",
        r"E:\datasets\WheelCorridor\raw\RoadSC-balanced_to_RoadSaW12-150_l\RoadSC-balanced_to_RoadSaW12-150_l",
    ],
    "data_root": [
        r"D:\NMI_SPWFM_datasets\friction_affordance_data",
        r"G:\friction_affordance_data",
    ],
}


def first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, default=None, help="Write a local_paths.yaml file")
    args = parser.parse_args()

    found = {key: first_existing(paths) for key, paths in DEFAULT_CANDIDATES.items()}
    print("Detected dataset paths:")
    for key, path in found.items():
        print(f"  {key}: {path or 'MISSING'}")

    if args.write:
        cfg = {
            "data_root": found["data_root"] or r"D:\NMI_SPWFM_datasets\friction_affordance_data",
            "datasets": {},
        }
        if found["rscd_prepared"]:
            cfg["datasets"]["rscd_prepared"] = {
                "type": "rscd_prepared",
                "labels_csv": found["rscd_prepared"],
            }
        elif found["rscd_raw"]:
            cfg["datasets"]["rscd_prepared"] = {
                "type": "imagefolder",
                "root": found["rscd_raw"],
                "dataset_name": "rscd",
            }
        if found["roadsaw"]:
            cfg["datasets"]["roadsaw"] = {
                "type": "imagefolder",
                "root": found["roadsaw"],
                "dataset_name": "roadsaw",
            }
        if found["roadsc"]:
            cfg["datasets"]["roadsc"] = {
                "type": "imagefolder",
                "root": found["roadsc"],
                "dataset_name": "roadsc",
            }
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"Wrote {args.write}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml


DATASET_KEYS = ("rscd", "roadsaw", "roadsc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("configs/experiments/leave_one_dataset_out"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    with args.base.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for held_out in DATASET_KEYS:
        cfg = copy.deepcopy(base)
        output_root = args.output_root or Path(base.get("output_dir", "outputs/run")).parent
        cfg["output_dir"] = str(output_root / f"lodo_{held_out}")
        data = cfg["data"]
        train = [p for p in data["train_manifests"] if not _is_dataset_manifest(p, held_out)]
        val = [p for p in data["val_manifests"] if not _is_dataset_manifest(p, held_out)]
        test = [
            p
            for p in data.get("test_manifests", data["val_manifests"])
            if _is_dataset_manifest(p, held_out)
        ]
        if not train or not val or not test:
            raise ValueError(
                f"Cannot build leave-one-dataset config for {held_out}: "
                f"train={len(train)} val={len(val)} test={len(test)}"
            )
        data["train_manifests"] = train
        data["val_manifests"] = val
        data["test_manifests"] = test
        cfg["experiment_note"] = (
            f"Leave-one-dataset-out: train/val exclude {held_out}; "
            f"test uses only {held_out}. This probes cross-dataset generalization."
        )
        out = args.out_dir / f"lodo_{held_out}.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(out)


def _is_dataset_manifest(path: str | Path, dataset: str) -> bool:
    name = Path(path).name.lower()
    if dataset == "rscd":
        return "rscd" in name
    return dataset in name


if __name__ == "__main__":
    main()

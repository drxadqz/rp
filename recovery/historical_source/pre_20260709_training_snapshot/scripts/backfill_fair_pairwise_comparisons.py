from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")
DEFAULT_OUT = Path("reports/paper_protocol_summary/fair_pairwise")

PAIRS = [
    ("rscd", "single_rscd_full_faf", "baseline_single_rscd_global_convnext"),
    ("roadsaw", "single_roadsaw_full_faf", "baseline_single_roadsaw_global_convnext"),
    ("roadsc", "single_roadsc_full_faf", "baseline_single_roadsc_global_convnext"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num-bootstrap", type=int, default=500)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ran = 0
    for dataset, faf_name, base_name in PAIRS:
        faf_dir = args.root / faf_name
        base_dir = args.root / base_name
        out_json = args.out_dir / f"{dataset}_faf_vs_global_convnext_paired_bootstrap.json"
        out_md = args.out_dir / f"{dataset}_faf_vs_global_convnext_paired_bootstrap.md"
        if out_json.exists() and out_md.exists():
            continue
        if not _ready(faf_dir) or not _ready(base_dir):
            print(f"skip {dataset}: pair is not complete")
            continue
        subprocess.run(
            [
                sys.executable,
                "scripts/paired_model_bootstrap_compare.py",
                "--config-a",
                str(args.config_dir / f"{faf_name}.yaml"),
                "--checkpoint-a",
                str(faf_dir / "best.pt"),
                "--name-a",
                faf_name,
                "--config-b",
                str(args.config_dir / f"{base_name}.yaml"),
                "--checkpoint-b",
                str(base_dir / "best.pt"),
                "--name-b",
                base_name,
                "--split",
                "test",
                "--num-bootstrap",
                str(args.num_bootstrap),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
            check=True,
        )
        ran += 1
    print(f"paired comparisons generated: {ran}")


def _ready(path: Path) -> bool:
    return all((path / name).exists() for name in ["best.pt", "detailed_test.json", "interval_calibration_90.json"])


if __name__ == "__main__":
    main()

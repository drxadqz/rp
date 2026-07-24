from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--num-bootstrap", type=int, default=500)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    runs = []
    for run_dir in sorted(path for path in args.root.glob("*") if path.is_dir()):
        if not _is_ready(run_dir):
            continue
        if (run_dir / "bootstrap_metrics.json").exists():
            continue
        config = args.config_dir / f"{run_dir.name}.yaml"
        if not config.exists():
            print(f"skip {run_dir.name}: config not found: {config}")
            continue
        runs.append((run_dir, config))

    if args.max_runs is not None:
        runs = runs[: max(0, int(args.max_runs))]

    if not runs:
        print("No completed runs need bootstrap backfill.")
        return

    for run_dir, config in runs:
        checkpoint = run_dir / "best.pt"
        print(f"Backfilling bootstrap metrics for {run_dir.name}")
        subprocess.run(
            [
                sys.executable,
                "scripts/bootstrap_metrics.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--target-coverage",
                str(args.target_coverage),
                "--num-bootstrap",
                str(args.num_bootstrap),
                "--out-json",
                str(run_dir / "bootstrap_metrics.json"),
                "--out-md",
                str(run_dir / "bootstrap_metrics.md"),
            ],
            check=True,
        )


def _is_ready(run_dir: Path) -> bool:
    required = [
        "best.pt",
        "detailed_test.json",
        "interval_calibration_90.json",
    ]
    return all((run_dir / name).exists() for name in required)


if __name__ == "__main__":
    main()

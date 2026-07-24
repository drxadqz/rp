from __future__ import annotations

import argparse
import os
from pathlib import Path

from run_paper_protocol_direct import _prepare_env, _run_config, _wait_for_pid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "rscd_per_day_queue"

RUNS = [
    "configs/experiments/rscd_per_day/rscd_per_day_full_faf.yaml",
    "configs/experiments/rscd_per_day/baseline_rscd_per_day_global_convnext.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the strict RSCD per-day benchmark using the same per-run "
            "training and evaluation contract as the paper protocol."
        )
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--keep-last-checkpoint", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional run stems to execute: rscd_per_day_full_faf and/or baseline_rscd_per_day_global_convnext.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    _prepare_env()

    if args.wait_pid:
        _wait_for_pid(args.wait_pid, args.log_dir)

    selected = _select_runs(args.only)
    for config in selected:
        _run_config(
            Path(config),
            python=args.python,
            log_dir=args.log_dir,
            force_train=args.force_train,
            keep_last_checkpoint=args.keep_last_checkpoint,
        )


def _select_runs(only: list[str] | None) -> list[str]:
    if not only:
        return RUNS
    requested = {Path(item).stem for item in only}
    selected = [config for config in RUNS if Path(config).stem in requested]
    missing = requested - {Path(config).stem for config in selected}
    if missing:
        raise ValueError(f"Unknown RSCD per-day run(s): {', '.join(sorted(missing))}")
    return selected


if __name__ == "__main__":
    main()

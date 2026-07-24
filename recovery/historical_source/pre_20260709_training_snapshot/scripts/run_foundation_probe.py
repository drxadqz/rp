from __future__ import annotations

import argparse
import os
from pathlib import Path

from run_paper_protocol_direct import _prepare_env, _run_config, _wait_for_pid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "foundation_probe_queue"

RUNS = [
    "configs/experiments/foundation_probe/foundation_dinov2_global_probe.yaml",
    "configs/experiments/foundation_probe/foundation_dinov2_quality_faf_probe.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DINOv2 foundation-feature feasibility probes using the same "
            "per-run artifact contract as the paper protocol."
        )
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--wait-pid", type=int, action="append", default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--keep-last-checkpoint", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "Optional run stems: foundation_dinov2_global_probe and/or "
            "foundation_dinov2_quality_faf_probe."
        ),
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    _prepare_env()

    for pid in args.wait_pid or []:
        _wait_for_pid(int(pid), args.log_dir)

    for config in _select_runs(args.only):
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
        raise ValueError(f"Unknown foundation probe run(s): {', '.join(sorted(missing))}")
    return selected


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
REQUIRED_FOR_COMPLETION = [
    "best.pt",
    "config.json",
    "detailed_test.json",
    "interval_calibration_90.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    candidates = []
    for run_dir in sorted(path for path in args.root.glob("*") if path.is_dir()):
        if not all((run_dir / name).exists() for name in REQUIRED_FOR_COMPLETION):
            continue
        last = run_dir / "last.pt"
        best = run_dir / "best.pt"
        if last.exists() and best.exists() and last.resolve() != best.resolve():
            candidates.append(last)

    total = sum(path.stat().st_size for path in candidates if path.exists())
    print(f"completed runs with removable last.pt: {len(candidates)}")
    print(f"potential free space GB: {total / 1024**3:.3f}")
    for path in candidates:
        print(path)
    if args.apply:
        for path in candidates:
            path.unlink()
        print("removed candidates")
    else:
        print("dry run only; pass --apply to remove listed files")


if __name__ == "__main__":
    main()

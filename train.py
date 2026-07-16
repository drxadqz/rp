from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import run_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C3-FaRNet on RSCD.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()

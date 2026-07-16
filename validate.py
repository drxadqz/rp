from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import run_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate C3-FaRNet on RSCD.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--logit-patch-rules", type=Path, default=None)
    args = parser.parse_args()
    run_eval(
        args.config,
        args.checkpoint,
        split="val",
        seed_override=args.seed,
        output_dir_override=args.output_dir,
        logit_patch_rules_path=args.logit_patch_rules,
    )


if __name__ == "__main__":
    main()

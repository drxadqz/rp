from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(path for path in args.root.glob("*") if path.is_dir()):
        complete = all((run_dir / name).exists() for name in ["detailed_test.json", "interval_calibration_90.json"])
        for checkpoint_name in ["best.pt", "best_safety.pt"]:
            checkpoint = run_dir / checkpoint_name
            if not checkpoint.exists():
                continue
            row = slim_checkpoint(checkpoint, apply=args.apply and complete)
            row["run"] = run_dir.name
            row["checkpoint"] = checkpoint_name
            row["complete"] = complete
            rows.append(row)

    for row in rows:
        status = "slimmed" if row["changed"] else "unchanged"
        print(
            "{run}: {status} complete={complete} before={before_mb:.1f}MB after={after_mb:.1f}MB".format(
                run=f"{row['run']}/{row['checkpoint']}",
                status=status,
                complete=row["complete"],
                before_mb=row["before_bytes"] / (1024 * 1024),
                after_mb=row["after_bytes"] / (1024 * 1024),
            )
        )


def slim_checkpoint(path: Path, apply: bool) -> dict:
    before = path.stat().st_size
    ckpt = torch.load(path, map_location="cpu")
    has_optimizer = bool(ckpt.get("optimizer"))
    if has_optimizer and apply:
        slim = {
            "model": ckpt["model"],
            "optimizer": None,
            "epoch": ckpt.get("epoch"),
            "metrics": ckpt.get("metrics", {}),
            "config": ckpt.get("config", {}),
        }
        tmp = path.with_suffix(path.suffix + ".slim_tmp")
        torch.save(slim, tmp)
        tmp.replace(path)
    after = path.stat().st_size
    return {
        "path": str(path),
        "changed": bool(has_optimizer and apply),
        "before_bytes": before,
        "after_bytes": after,
    }


if __name__ == "__main__":
    main()

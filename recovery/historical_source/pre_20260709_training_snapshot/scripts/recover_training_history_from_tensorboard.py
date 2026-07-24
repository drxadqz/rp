from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    history = recover_history(args.tb_dir, args.run_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_csv:
        write_csv(history, args.out_csv)
    print(json.dumps({"epochs": len(history), "out_json": str(args.out_json)}, indent=2))


def recover_history(tb_dir: Path, run_dir: Path | None = None) -> list[dict[str, Any]]:
    accumulator = EventAccumulator(str(tb_dir))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    by_epoch: dict[int, dict[str, Any]] = {}
    for tag in tags:
        if not (tag.startswith("train/") or tag.startswith("val/")):
            continue
        split, key = tag.split("/", 1)
        metric_key = f"{split}_metrics"
        for event in accumulator.Scalars(tag):
            epoch = int(event.step)
            row = by_epoch.setdefault(
                epoch,
                {
                    "epoch": epoch,
                    "epochs": None,
                    "train_metrics": {},
                    "val_metrics": {},
                    "saved_best": False,
                    "saved_best_safety": False,
                },
            )
            row[metric_key][key] = float(event.value)

    planned_epochs = _load_planned_epochs(run_dir) if run_dir else None
    rows = [by_epoch[key] for key in sorted(by_epoch)]
    best_loss = float("inf")
    best_safety = float("-inf")
    for row in rows:
        row["epochs"] = planned_epochs or rows[-1]["epoch"]
        val_metrics = row.get("val_metrics", {})
        val_loss = float(val_metrics.get("loss", float("inf")))
        if val_loss < best_loss:
            best_loss = val_loss
            row["saved_best"] = True
        safety = _safety_proxy(val_metrics)
        row["safety_proxy"] = safety
        if safety > best_safety:
            best_safety = safety
            row["saved_best_safety"] = True
        row["best_metric"] = best_loss if best_loss < float("inf") else None
        row["best_safety_metric"] = best_safety if best_safety > float("-inf") else None
    return [row for row in rows if row.get("train_metrics") or row.get("val_metrics")]


def _load_planned_epochs(run_dir: Path | None) -> int | None:
    if run_dir is None:
        return None
    for name, keys in [
        ("training_state.json", ["epochs"]),
        ("config.json", ["optim", "epochs"]),
    ]:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = payload
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _safety_proxy(metrics: dict[str, Any]) -> float:
    risk = float(metrics.get("acc_risk", 0.0) or 0.0)
    friction = float(metrics.get("acc_friction", 0.0) or 0.0)
    coverage = float(metrics.get("mu_interval_coverage", 0.0) or 0.0)
    width = float(metrics.get("mu_interval_width", 0.0) or 0.0)
    return risk + 0.5 * friction + 0.5 * coverage - 0.1 * width


def write_csv(history: list[dict[str, Any]], path: Path) -> None:
    keys = set()
    for row in history:
        for prefix in ("train", "val"):
            for key in row.get(f"{prefix}_metrics", {}):
                keys.add(f"{prefix}_{key}")
    ordered = [
        "epoch",
        "epochs",
        "saved_best",
        "saved_best_safety",
        "safety_proxy",
        "best_metric",
        "best_safety_metric",
    ] + sorted(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in history:
            out: dict[str, Any] = {
                "epoch": row.get("epoch"),
                "epochs": row.get("epochs"),
                "saved_best": row.get("saved_best", False),
                "saved_best_safety": row.get("saved_best_safety", False),
                "safety_proxy": row.get("safety_proxy"),
                "best_metric": row.get("best_metric"),
                "best_safety_metric": row.get("best_safety_metric"),
            }
            for prefix in ("train", "val"):
                for key, value in row.get(f"{prefix}_metrics", {}).items():
                    out[f"{prefix}_{key}"] = value
            writer.writerow(out)


if __name__ == "__main__":
    main()

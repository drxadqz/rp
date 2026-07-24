from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


EPOCH_RE = re.compile(r"^Epoch\s+(\d+)/(\d+)")
METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    history = parse_log(args.log)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_csv:
        write_csv(history, args.out_csv)
    print(json.dumps({"epochs": len(history), "out_json": str(args.out_json)}, indent=2))


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        epoch_match = EPOCH_RE.match(line)
        if epoch_match:
            if cur is not None:
                rows.append(cur)
            cur = {
                "epoch": int(epoch_match.group(1)),
                "epochs": int(epoch_match.group(2)),
                "train_metrics": {},
                "val_metrics": {},
                "saved_best": False,
            }
            continue
        if cur is None:
            continue
        if line.startswith("train:"):
            cur["train_metrics"] = parse_metrics(line)
        elif line.startswith("val"):
            cur["val_metrics"] = parse_metrics(line)
        elif "saved best checkpoint" in line:
            cur["saved_best"] = True
    if cur is not None:
        rows.append(cur)
    return [row for row in rows if row.get("train_metrics") or row.get("val_metrics")]


def parse_metrics(line: str) -> dict[str, float]:
    return {key: float(value) for key, value in METRIC_RE.findall(line)}


def write_csv(history: list[dict[str, Any]], path: Path) -> None:
    keys = set()
    for row in history:
        for prefix in ("train", "val"):
            for key in row.get(f"{prefix}_metrics", {}):
                keys.add(f"{prefix}_{key}")
    ordered = ["epoch", "epochs", "saved_best"] + sorted(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in history:
            out: dict[str, Any] = {
                "epoch": row.get("epoch"),
                "epochs": row.get("epochs"),
                "saved_best": row.get("saved_best", False),
            }
            for prefix in ("train", "val"):
                for key, value in row.get(f"{prefix}_metrics", {}).items():
                    out[f"{prefix}_{key}"] = value
            writer.writerow(out)


if __name__ == "__main__":
    main()

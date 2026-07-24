from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from friction_affordance.ontology import IGNORE_INDEX, TASKS, label_to_index


TASK_COLUMNS = {
    "friction": "friction_label",
    "material": "material_label",
    "unevenness": "unevenness_label",
    "wetness": "wetness_label",
    "snow": "snow_label",
    "risk": "risk_label",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    result = {"manifests": [], "overall": {"errors": [], "warnings": []}}
    for item in args.manifest:
        result["manifests"].append(_audit_manifest(Path(item)))

    invalid_total = sum(m["invalid_label_count"] for m in result["manifests"])
    bad_interval_total = sum(m["bad_interval_count"] for m in result["manifests"])
    missing_image_total = sum(m["missing_image_count"] for m in result["manifests"])
    if invalid_total:
        result["overall"]["errors"].append(f"invalid labels: {invalid_total}")
    if bad_interval_total:
        result["overall"]["errors"].append(f"bad mu intervals: {bad_interval_total}")
    if missing_image_total:
        result["overall"]["warnings"].append(f"missing image paths: {missing_image_total}")

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_to_markdown(result), encoding="utf-8")


def _audit_manifest(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    item: dict[str, Any] = {
        "path": str(path),
        "num_rows": int(len(df)),
        "invalid_label_count": 0,
        "bad_interval_count": 0,
        "missing_image_count": 0,
        "tasks": {},
    }
    if "image_path" in df.columns:
        missing = df["image_path"].isna() | (df["image_path"].astype(str).str.strip() == "")
        item["missing_image_count"] = int(missing.sum())

    for task, column in TASK_COLUMNS.items():
        if column not in df.columns:
            item["tasks"][task] = {"present": False}
            continue
        values = df[column]
        known = values.notna() & ~values.astype(str).str.lower().isin(["", "nan", "none", "null", "unknown", "-1"])
        invalid_values: dict[str, int] = {}
        for value, count in values[known].value_counts(dropna=False).items():
            if label_to_index(task, value) == IGNORE_INDEX:
                invalid_values[str(value)] = int(count)
        invalid_count = int(sum(invalid_values.values()))
        item["invalid_label_count"] += invalid_count
        item["tasks"][task] = {
            "present": True,
            "known_count": int(known.sum()),
            "known_ratio": float(known.mean()) if len(values) else 0.0,
            "invalid_count": invalid_count,
            "invalid_values": invalid_values,
        }

    if {"mu_low", "mu_high"}.issubset(df.columns):
        low = pd.to_numeric(df["mu_low"], errors="coerce")
        high = pd.to_numeric(df["mu_high"], errors="coerce")
        known = low.notna() & high.notna()
        bad = known & ((low < 0.0) | (high > 1.2) | (low > high))
        item["mu_interval"] = {
            "known_count": int(known.sum()),
            "known_ratio": float(known.mean()) if len(df) else 0.0,
            "bad_count": int(bad.sum()),
            "width_mean": float((high[known] - low[known]).mean()) if known.any() else None,
        }
        item["bad_interval_count"] = int(bad.sum())
    return item


def _to_markdown(result: dict[str, Any]) -> str:
    lines = ["# Manifest Label Audit", ""]
    errors = result["overall"]["errors"]
    warnings = result["overall"]["warnings"]
    lines.append(f"- Errors: {', '.join(errors) if errors else 'none'}")
    lines.append(f"- Warnings: {', '.join(warnings) if warnings else 'none'}")
    lines.append("")
    lines.append("| manifest | rows | invalid labels | bad intervals | missing image paths |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in result["manifests"]:
        lines.append(
            f"| `{Path(item['path']).name}` | {item['num_rows']} | "
            f"{item['invalid_label_count']} | {item['bad_interval_count']} | {item['missing_image_count']} |"
        )
    lines.append("")
    lines.append("## Task Details")
    for item in result["manifests"]:
        lines.append("")
        lines.append(f"### `{Path(item['path']).name}`")
        lines.append("")
        lines.append("| task | known ratio | invalid count | invalid values |")
        lines.append("|---|---:|---:|---|")
        for task, task_info in item["tasks"].items():
            if not task_info.get("present"):
                lines.append(f"| {task} | - | - | missing column |")
                continue
            invalid_values = task_info.get("invalid_values") or {}
            invalid_text = ", ".join(f"{k}:{v}" for k, v in invalid_values.items()) if invalid_values else "-"
            lines.append(
                f"| {task} | {task_info['known_ratio']:.4f} | "
                f"{task_info['invalid_count']} | {invalid_text} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

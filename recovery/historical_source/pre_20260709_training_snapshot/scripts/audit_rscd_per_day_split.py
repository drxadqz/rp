"""Audit official RSCD per-day split files against local manifests.

The MiviaLab RSCD per-day split is useful only if its CSV rows can be
resolved to the local RSCD image paths and if acquisition-day groups do not
cross train/validation/test. This script validates those assumptions and
optionally writes mapped local manifests for valid split files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DATE_RE = re.compile(r"^(\d{8})")


def normalize_class(value: str) -> str:
    return value.strip().replace("-", "_")


def read_local_manifests(paths: Iterable[Path]) -> Tuple[Dict[str, dict], Counter]:
    by_name: Dict[str, dict] = {}
    name_counts: Counter = Counter()
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                image_path = row.get("image_path", "")
                name = Path(image_path).name
                if not name:
                    continue
                name_counts[name] += 1
                by_name.setdefault(name, row)
    return by_name, name_counts


def parse_day(filename: str) -> str:
    match = DATE_RE.match(filename)
    return match.group(1) if match else ""


def audit_split(split_path: Path, local_by_name: Dict[str, dict], duplicate_names: set) -> dict:
    rows = 0
    malformed = 0
    missing = 0
    duplicate_name_rows = 0
    class_mismatch = 0
    split_dir_counts: Counter = Counter()
    class_counts: Counter = Counter()
    days: Counter = Counter()
    examples = defaultdict(list)
    mapped_rows: List[dict] = []

    with split_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        expected_fields = {"image", "label", "class"}
        if set(reader.fieldnames or []) < expected_fields:
            return {
                "split_file": str(split_path),
                "valid_csv_header": False,
                "rows": 0,
                "valid_for_protocol": False,
                "reason": f"missing expected fields: {sorted(expected_fields)}",
            }

        for row in reader:
            rows += 1
            image = (row.get("image") or "").strip()
            official_class = (row.get("class") or "").strip()
            parts = image.replace("\\", "/").split("/")
            if len(parts) < 2 or not official_class:
                malformed += 1
                if len(examples["malformed"]) < 5:
                    examples["malformed"].append(row)
                continue
            split_dir_counts[parts[0]] += 1
            class_counts[normalize_class(official_class)] += 1
            filename = parts[-1]
            day = parse_day(filename)
            if day:
                days[day] += 1
            local = local_by_name.get(filename)
            if local is None:
                missing += 1
                if len(examples["missing"]) < 5:
                    examples["missing"].append(row)
                continue
            if filename in duplicate_names:
                duplicate_name_rows += 1
                if len(examples["duplicate_name"]) < 5:
                    examples["duplicate_name"].append(row)
            local_class = normalize_class(local.get("class_label", ""))
            if normalize_class(official_class) != local_class:
                class_mismatch += 1
                if len(examples["class_mismatch"]) < 5:
                    examples["class_mismatch"].append(
                        {
                            "official": row,
                            "local_class_label": local.get("class_label", ""),
                            "local_image_path": local.get("image_path", ""),
                        }
                    )
                continue
            mapped = dict(local)
            mapped["split"] = split_path.stem.replace("vali_20k", "val").replace("test_50k", "test")
            mapped["class_label"] = normalize_class(official_class)
            mapped["per_day_source_image"] = image
            mapped_rows.append(mapped)

    valid_for_protocol = (
        rows > 0
        and malformed == 0
        and missing == 0
        and duplicate_name_rows == 0
        and class_mismatch == 0
        and len(mapped_rows) == rows
    )
    return {
        "split_file": str(split_path),
        "valid_csv_header": True,
        "rows": rows,
        "mapped_rows": len(mapped_rows),
        "malformed_rows": malformed,
        "missing_local_rows": missing,
        "duplicate_name_rows": duplicate_name_rows,
        "class_mismatch_rows": class_mismatch,
        "unique_days": len(days),
        "split_dir_counts": dict(split_dir_counts),
        "top_classes": class_counts.most_common(10),
        "top_days": days.most_common(10),
        "examples": dict(examples),
        "valid_for_protocol": valid_for_protocol,
        "mapped_rows_data": mapped_rows,
    }


def write_mapped_manifest(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    base_fields = [
        "image_path",
        "split",
        "dataset",
        "class_label",
        "domain_id",
        "friction_label",
        "material_label",
        "unevenness_label",
        "wetness_label",
        "snow_label",
        "risk_label",
        "mu_low",
        "mu_high",
        "per_day_source_image",
    ]
    extra = [k for k in rows[0].keys() if k not in base_fields]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra)
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(report: dict) -> str:
    lines = [
        "# RSCD Per-Day Split Audit",
        "",
        f"External split dir: `{report['external_dir']}`",
        f"Local rows indexed: `{report['local_index_rows']}`",
        f"Duplicate local basenames: `{report['duplicate_local_basenames']}`",
        "",
        "## Split Files",
        "",
        "| File | Rows | Mapped | Days | Missing | Class mismatch | Malformed | Valid |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in report["splits"]:
        lines.append(
            "| {file} | {rows} | {mapped} | {days} | {missing} | {mismatch} | {malformed} | `{valid}` |".format(
                file=Path(split["split_file"]).name,
                rows=split.get("rows", 0),
                mapped=split.get("mapped_rows", 0),
                days=split.get("unique_days", 0),
                missing=split.get("missing_local_rows", 0),
                mismatch=split.get("class_mismatch_rows", 0),
                malformed=split.get("malformed_rows", 0),
                valid=split.get("valid_for_protocol", False),
            )
        )
    lines.extend(["", "## Day Leakage Check", ""])
    if report["day_overlaps"]:
        lines.append("Day overlaps were found and must be fixed before using this protocol:")
        for key, values in report["day_overlaps"].items():
            lines.append(f"- `{key}`: {', '.join(values[:20])}")
    else:
        lines.append("No acquisition-day overlap was found among the valid mapped split files.")
    lines.extend(["", "## Protocol Decision", ""])
    lines.append(report["protocol_decision"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_splits/rscd_per_day"))
    parser.add_argument("--local-manifest", type=Path, action="append")
    parser.add_argument("--out-json", type=Path, default=Path("reports/paper_protocol_summary/rscd_per_day_split_audit.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/paper_protocol_summary/rscd_per_day_split_audit.md"))
    parser.add_argument("--mapped-dir", type=Path, default=None)
    args = parser.parse_args()

    local_manifests = args.local_manifest or [
        Path("data/manifests_full/rscd_prepared_train.csv"),
        Path("data/manifests_full/rscd_prepared_val.csv"),
        Path("data/manifests_full/rscd_prepared_test.csv"),
    ]
    local_by_name, name_counts = read_local_manifests(local_manifests)
    duplicate_names = {name for name, count in name_counts.items() if count > 1}

    split_files = [
        args.external_dir / "train.csv",
        args.external_dir / "vali_20k.csv",
        args.external_dir / "test_50k.csv",
    ]
    splits = [audit_split(path, local_by_name, duplicate_names) for path in split_files if path.exists()]

    day_sets = {}
    for split in splits:
        if not split.get("valid_for_protocol"):
            continue
        split_name = Path(split["split_file"]).stem.replace("vali_20k", "val").replace("test_50k", "test")
        day_sets[split_name] = {day for day, _ in split.get("top_days", [])}
        # Use all days, not just top days.
        day_sets[split_name] = {
            parse_day(Path(row["image_path"]).name)
            for row in split["mapped_rows_data"]
            if parse_day(Path(row["image_path"]).name)
        }

    day_overlaps = {}
    names = sorted(day_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(day_sets[left] & day_sets[right])
            if overlap:
                day_overlaps[f"{left}_vs_{right}"] = overlap

    all_three_valid = {Path(s["split_file"]).name: s.get("valid_for_protocol") for s in splits}
    protocol_ready = (
        all_three_valid.get("train.csv", False)
        and all_three_valid.get("vali_20k.csv", False)
        and all_three_valid.get("test_50k.csv", False)
        and not day_overlaps
    )
    if protocol_ready:
        decision = (
            "The official RSCD per-day split is locally usable. It can be promoted "
            "to a stricter RSCD benchmark after matching configs are generated."
        )
    else:
        decision = (
            "The RSCD per-day protocol is not ready for training yet. Use it as a "
            "planned benchmark until every official split CSV is complete, all rows "
            "map to local images/classes, and day-overlap remains zero."
        )

    if args.mapped_dir is not None:
        for split in splits:
            if not split.get("valid_for_protocol"):
                continue
            split_name = Path(split["split_file"]).stem.replace("vali_20k", "val").replace("test_50k", "test")
            write_mapped_manifest(args.mapped_dir / f"rscd_per_day_{split_name}.csv", split["mapped_rows_data"])

    json_splits = []
    for split in splits:
        clean = {k: v for k, v in split.items() if k != "mapped_rows_data"}
        json_splits.append(clean)

    report = {
        "external_dir": str(args.external_dir),
        "local_manifest_paths": [str(p) for p in local_manifests],
        "local_index_rows": sum(name_counts.values()),
        "local_unique_basenames": len(local_by_name),
        "duplicate_local_basenames": len(duplicate_names),
        "splits": json_splits,
        "day_overlaps": day_overlaps,
        "protocol_ready": protocol_ready,
        "protocol_decision": decision,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(make_markdown(report), encoding="utf-8")
    print(make_markdown(report))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_DIR = Path("data/manifests_full")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "dataset_path_completeness_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "dataset_path_completeness_report.json",
    )
    args = parser.parse_args()

    report = build_report(args.manifest_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(manifest_dir: Path) -> dict[str, Any]:
    rows_by_dataset: Counter[str] = Counter()
    splits_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    classes_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    paths_by_dataset: dict[str, set[str]] = defaultdict(set)
    manifest_rows: list[dict[str, Any]] = []

    for manifest in sorted(manifest_dir.glob("*.csv")):
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row_count = 0
            for row in reader:
                dataset = (row.get("dataset") or "").strip() or "unknown"
                split = (row.get("split") or manifest.stem.rsplit("_", 1)[-1]).strip()
                label = (row.get("class_label") or row.get("label") or "").strip()
                image_path = (row.get("image_path") or "").strip()
                row_count += 1
                rows_by_dataset[dataset] += 1
                splits_by_dataset[dataset][split] += 1
                if label:
                    classes_by_dataset[dataset][label] += 1
                if image_path:
                    paths_by_dataset[dataset].add(image_path)
        manifest_rows.append({"manifest": str(manifest), "rows": row_count})

    all_paths = sorted({path for paths in paths_by_dataset.values() for path in paths})
    existence = _batched_exists(all_paths)
    missing_paths = [path for path in all_paths if not existence.get(path, False)]
    dataset_rows: list[dict[str, Any]] = []
    for dataset in sorted(rows_by_dataset):
        paths = sorted(paths_by_dataset[dataset])
        missing = [path for path in paths if not existence.get(path, False)]
        dataset_rows.append(
            {
                "dataset": dataset,
                "rows": int(rows_by_dataset[dataset]),
                "unique_paths": len(paths),
                "existing_unique_paths": len(paths) - len(missing),
                "missing_unique_paths": len(missing),
                "splits": dict(sorted(splits_by_dataset[dataset].items())),
                "num_classes": len(classes_by_dataset[dataset]),
                "top_classes": dict(classes_by_dataset[dataset].most_common(12)),
                "missing_examples": missing[:10],
            }
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_dir": str(manifest_dir),
        "claim_boundary": (
            "This is a full manifest path-existence check. It proves files referenced by "
            "the current manifests are present or missing; it does not validate label truth "
            "or measured tire-road friction."
        ),
        "manifests": manifest_rows,
        "totals": {
            "rows": int(sum(rows_by_dataset.values())),
            "unique_paths": len(all_paths),
            "existing_unique_paths": len(all_paths) - len(missing_paths),
            "missing_unique_paths": len(missing_paths),
        },
        "dataset_rows": dataset_rows,
        "missing_examples": missing_paths[:20],
    }


def _batched_exists(paths: list[str]) -> dict[str, bool]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for raw in paths:
        path = Path(raw)
        grouped[str(path.parent)].append((raw, path.name))

    out: dict[str, bool] = {}
    for parent, items in grouped.items():
        try:
            names = {entry.name for entry in os.scandir(parent)}
        except OSError:
            for raw, _ in items:
                out[raw] = False
            continue
        for raw, name in items:
            out[raw] = name in names
    return out


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "# Dataset Path Completeness Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Full Manifest Path Check",
        "",
        f"- Rows: `{totals['rows']}`.",
        f"- Unique image paths: `{totals['unique_paths']}`.",
        f"- Existing unique image paths: `{totals['existing_unique_paths']}`.",
        f"- Missing unique image paths: `{totals['missing_unique_paths']}`.",
        "",
        "| Dataset | rows | unique paths | existing | missing | splits |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["dataset_rows"]:
        lines.append(
            "| {dataset} | {rows} | {unique} | {existing} | {missing} | `{splits}` |".format(
                dataset=row["dataset"],
                rows=row["rows"],
                unique=row["unique_paths"],
                existing=row["existing_unique_paths"],
                missing=row["missing_unique_paths"],
                splits=json.dumps(row["splits"], ensure_ascii=False, sort_keys=True),
            )
        )
    if report.get("missing_examples"):
        lines.extend(["", "## Missing Examples", ""])
        for item in report["missing_examples"]:
            lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

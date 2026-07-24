from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_data\ExtremeRoad_raw\extracted")
DEFAULT_OUT_DIR = Path("data/manifests_extreme")
DEFAULT_REPORT_DIR = Path("reports/paper_protocol_summary")


CLASS_MAP = {
    "1-Ice Surface": {
        "class_label": "ice_surface",
        "friction_label": "ice",
        "material_label": "",
        "unevenness_label": "",
        "wetness_label": "",
        "snow_label": "ice",
        "risk_label": "very_high",
        "mu_low": 0.03,
        "mu_high": 0.25,
        "mu_source": "Zhao2025 smooth/rough ice anchors plus Liu2025/TRB conservative ice envelope",
    },
    "2-Rough Ice Surface": {
        "class_label": "rough_ice_surface",
        "friction_label": "ice",
        "material_label": "",
        "unevenness_label": "severe",
        "wetness_label": "",
        "snow_label": "ice",
        "risk_label": "high",
        "mu_low": 0.15,
        "mu_high": 0.35,
        "mu_source": "Zhao2025 rough-ice anchor [0.21,0.23] widened for visual-only uncertainty",
    },
    "3-Loose snow surface": {
        "class_label": "loose_snow_surface",
        "friction_label": "partial_snow",
        "material_label": "",
        "unevenness_label": "",
        "wetness_label": "",
        "snow_label": "partial_snow",
        "risk_label": "high",
        "mu_low": 0.20,
        "mu_high": 0.55,
        "mu_source": "Liu2025 loose-snow review anchor and TRB snow/treated-snow conservative envelope",
    },
    "4-Muddy Road After Snow": {
        "class_label": "muddy_road_after_snow",
        "friction_label": "melted_snow",
        "material_label": "dirt_mud",
        "unevenness_label": "",
        "wetness_label": "water",
        "snow_label": "melted_snow",
        "risk_label": "high",
        "mu_low": 0.15,
        "mu_high": 0.45,
        "mu_source": "TRB refrozen/melting snow anchors plus visual-only muddy-water uncertainty",
    },
    "5-Waterlogged Pavement": {
        "class_label": "waterlogged_pavement",
        "friction_label": "water",
        "material_label": "asphalt",
        "unevenness_label": "",
        "wetness_label": "water",
        "snow_label": "none",
        "risk_label": "high",
        "mu_low": 0.20,
        "mu_high": 0.60,
        "mu_source": "Zhao2025 waterlogged equation and Liu2025 wet/asphalt envelope widened for visual-only data",
    },
    "6-Semi-impregnated Asphalt Pavement": {
        "class_label": "semi_impregnated_asphalt_pavement",
        "friction_label": "damp",
        "material_label": "asphalt",
        "unevenness_label": "",
        "wetness_label": "damp",
        "snow_label": "none",
        "risk_label": "low",
        "mu_low": 0.55,
        "mu_high": 0.90,
        "mu_source": "Zhao2025 semi-wet asphalt anchor [0.72,0.77] inside conservative damp-asphalt envelope",
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_COLUMNS = [
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
    "source_class",
    "mu_source",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    report, rows = build_dataset(args.root, seed=int(args.seed), train_ratio=float(args.train_ratio), val_ratio=float(args.val_ratio))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        write_csv(args.out_dir / f"extreme_road_{split}.csv", [row for row in rows if row["split"] == split])
    write_csv(args.out_dir / "extreme_road_all.csv", rows)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report["manifests"] = {
        "train": str(args.out_dir / "extreme_road_train.csv"),
        "val": str(args.out_dir / "extreme_road_val.csv"),
        "test": str(args.out_dir / "extreme_road_test.csv"),
        "all": str(args.out_dir / "extreme_road_all.csv"),
    }
    (args.report_dir / "extreme_road_dataset_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.report_dir / "extreme_road_dataset_audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_dataset(root: Path, *, seed: int, train_ratio: float, val_ratio: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(f"ExtremeRoad extracted root not found: {root}")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "root": str(root),
        "dataset": "ExtremeRoad",
        "role": "future direct_visual_friction protocol candidate, not part of the RSCD/RoadSaW/RoadSC main paper_protocol tables",
        "license": "BSD-3-Clause repository license; README requests citation and acknowledgment of Zhao et al. 2025",
        "split_policy": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1.0 - train_ratio - val_ratio,
            "stratified_by": "source class directory",
        },
        "class_mapping": CLASS_MAP,
        "invalid_images": [],
        "classes": {},
    }
    for class_dir_name, mapping in CLASS_MAP.items():
        class_dir = root / class_dir_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class directory missing: {class_dir}")
        image_paths = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS and "__MACOSX" not in str(path)
        )
        rng.shuffle(image_paths)
        n = len(image_paths)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        split_for_index = {}
        for idx, path in enumerate(image_paths):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_for_index[path] = split

        stats = []
        valid_paths = []
        for path in image_paths:
            try:
                stats.append(image_stats(path))
                valid_paths.append(path)
            except Exception as exc:  # noqa: BLE001
                audit["invalid_images"].append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        for path in valid_paths:
            rows.append(make_row(path, class_dir_name, mapping, split_for_index[path]))
        audit["classes"][mapping["class_label"]] = summarize_class(class_dir_name, mapping, valid_paths, stats, split_for_index)

    rows.sort(key=lambda item: (item["split"], item["class_label"], item["image_path"]))
    audit["num_rows"] = len(rows)
    audit["split_counts"] = dict(Counter(row["split"] for row in rows))
    audit["class_counts"] = dict(Counter(row["class_label"] for row in rows))
    audit["mu_policy"] = (
        "Intervals are conservative class-level anchors for visual-only direct-friction validation. "
        "They are not per-image measured friction coefficients; vehicle, tire, speed, load, temperature, and water depth are unobserved."
    )
    return audit, rows


def make_row(path: Path, source_class: str, mapping: dict[str, Any], split: str) -> dict[str, Any]:
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "image_path": str(path),
            "split": split,
            "dataset": "extreme_road",
            "domain_id": "extreme_road",
            "source_class": source_class,
        }
    )
    for key, value in mapping.items():
        row[key] = value
    row["mu_low"] = f"{float(mapping['mu_low']):.4f}"
    row["mu_high"] = f"{float(mapping['mu_high']):.4f}"
    return row


def image_stats(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        arr = np.asarray(image, dtype=np.float32) / 255.0
    brightness = float(arr.mean())
    contrast = float(arr.std())
    max_ch = arr.max(axis=2)
    min_ch = arr.min(axis=2)
    saturation = float(((max_ch - min_ch) / np.maximum(max_ch, 1e-6)).mean())
    near_white = bool(brightness > 0.86 and contrast < 0.08 and saturation < 0.12)
    low_contrast = bool(contrast < 0.035)
    return {
        "width": int(width),
        "height": int(height),
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "near_white": near_white,
        "low_contrast": low_contrast,
    }


def summarize_class(
    source_class: str,
    mapping: dict[str, Any],
    paths: list[Path],
    stats: list[dict[str, Any]],
    split_for_index: dict[Path, str],
) -> dict[str, Any]:
    dims = Counter(f"{item['width']}x{item['height']}" for item in stats)
    split_counts = Counter(split_for_index[path] for path in paths)
    return {
        "source_class": source_class,
        "class_label": mapping["class_label"],
        "num_images": len(paths),
        "split_counts": dict(split_counts),
        "top_dimensions": dict(dims.most_common(8)),
        "brightness_mean": mean([item["brightness"] for item in stats]),
        "contrast_mean": mean([item["contrast"] for item in stats]),
        "saturation_mean": mean([item["saturation"] for item in stats]),
        "near_white_count": int(sum(1 for item in stats if item["near_white"])),
        "low_contrast_count": int(sum(1 for item in stats if item["low_contrast"])),
        "mu_interval": [mapping["mu_low"], mapping["mu_high"]],
        "mu_source": mapping["mu_source"],
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ExtremeRoad Dataset Audit",
        "",
        f"Root: `{report['root']}`",
        "",
        f"Rows: `{report['num_rows']}`",
        "",
        "Claim boundary: this dataset is a future `direct_visual_friction` candidate. It is not included in the current RSCD/RoadSaW/RoadSC main paper-protocol tables.",
        "",
        "## Split Counts",
        "",
        "| split | rows |",
        "|---|---:|",
    ]
    for split, count in sorted(report["split_counts"].items()):
        lines.append(f"| {split} | {count} |")
    lines.extend(["", "## Classes", "", "| class | source class | images | train | val | test | mu interval | near-white | low-contrast | top dimensions |"])
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for class_label, item in sorted(report["classes"].items()):
        splits = item["split_counts"]
        dims = ", ".join(f"{k}:{v}" for k, v in item["top_dimensions"].items())
        lines.append(
            "| {cls} | {src} | {n} | {tr} | {va} | {te} | [{lo:.2f}, {hi:.2f}] | {nw} | {lc} | {dims} |".format(
                cls=class_label,
                src=item["source_class"],
                n=item["num_images"],
                tr=splits.get("train", 0),
                va=splits.get("val", 0),
                te=splits.get("test", 0),
                lo=float(item["mu_interval"][0]),
                hi=float(item["mu_interval"][1]),
                nw=item["near_white_count"],
                lc=item["low_contrast_count"],
                dims=dims,
            )
        )
    lines.extend(
        [
            "",
            "## Interval Policy",
            "",
            report["mu_policy"],
            "",
            "## Generated Manifests",
            "",
        ]
    )
    for key, value in report.get("manifests", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    if report["invalid_images"]:
        lines.extend(["", "## Invalid Images", ""])
        for item in report["invalid_images"][:20]:
            lines.append(f"- `{item['path']}`: {item['error']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

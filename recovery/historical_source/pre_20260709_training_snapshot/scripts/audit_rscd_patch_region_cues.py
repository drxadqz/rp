from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_data\RSCD_raw\RSCD dataset-1million")
OUT = Path("reports/paper_protocol_summary/rscd_patch_region_cue_statistics")
CLASSES = [
    "dry_asphalt_smooth",
    "dry_asphalt_slight",
    "dry_asphalt_severe",
    "wet_asphalt_smooth",
    "wet_asphalt_slight",
    "wet_asphalt_severe",
    "water_asphalt_smooth",
    "water_asphalt_slight",
    "water_asphalt_severe",
    "dry_concrete_smooth",
    "wet_concrete_smooth",
    "water_concrete_smooth",
    "dry_gravel",
    "wet_gravel",
    "water_gravel",
    "dry_mud",
    "wet_mud",
    "water_mud",
    "fresh_snow",
    "melted_snow",
    "ice",
]


def main() -> None:
    rng = random.Random(29)
    records = []
    for cls in CLASSES:
        paths = collect_images(ROOT / "train" / cls)
        if not paths:
            continue
        for path in rng.sample(paths, min(80, len(paths))):
            records.append({"class": cls, "path": str(path), **extract_features(path)})

    feature_names = [key for key in records[0] if key not in {"class", "path"}] if records else []
    fisher_rows = []
    for name in feature_names:
        fisher_rows.append({"feature": name, "fisher_ratio": fisher_ratio(records, name), "kind": feature_kind(name)})
    fisher_rows.sort(key=lambda item: item["fisher_ratio"], reverse=True)

    summary = {
        "claim_boundary": (
            "This audit samples real RSCD close road patches and compares vertical "
            "bottom-vs-top cues with position-invariant patch statistics. It is a "
            "feature diagnostic, not model-performance evidence."
        ),
        "dataset_root": str(ROOT),
        "num_samples": len(records),
        "num_classes": len(sorted({row["class"] for row in records})),
        "image_sizes": sorted({tuple(Image.open(row["path"]).size) for row in records[: min(100, len(records))]}),
        "top_features": fisher_rows[:16],
        "bottom_top_features": [row for row in fisher_rows if row["kind"] == "vertical_region_delta"],
        "patch_distribution_features": [row for row in fisher_rows if row["kind"] == "patch_distribution"][:12],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "feature_rank": fisher_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    OUT.with_suffix(".md").write_text(to_markdown(summary, fisher_rows), encoding="utf-8")
    print(OUT.with_suffix(".md"))


def collect_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(folder.glob(ext))
    return sorted(paths)


def extract_features(path: Path) -> dict[str, float]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    sat = (maxc - minc) / np.maximum(maxc, 1e-4)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    grad = np.sqrt(gx * gx + gy * gy + 1e-6)
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    lap = np.abs(lap)

    snow = sigmoid((maxc - 0.72) * 12.0) * sigmoid((0.28 - sat) * 12.0)
    spec = sigmoid((maxc - 0.82) * 14.0) * sigmoid((0.24 - sat) * 12.0)
    dark = sigmoid((0.38 - maxc) * 10.0) * sigmoid((0.45 - grad) * 12.0)
    wet = np.clip(spec + 0.5 * dark, 0.0, 1.0)
    low_texture = sigmoid((0.045 - grad) * 35.0)
    thin_water = wet * sigmoid((0.08 - lap) * 22.0)

    features = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "saturation_mean": float(sat.mean()),
        "saturation_std": float(sat.std()),
        "value_mean": float(maxc.mean()),
        "value_std": float(maxc.std()),
        "grad_mean": float(grad.mean()),
        "grad_std": float(grad.std()),
        "snow_mean": float(snow.mean()),
        "specular_mean": float(spec.mean()),
        "dark_water_mean": float(dark.mean()),
        "wet_proxy_mean": float(wet.mean()),
        "thin_water_mean": float(thin_water.mean()),
        "low_texture_mean": float(low_texture.mean()),
        "wet_top10": top_fraction_mean(wet, 0.10),
        "specular_top10": top_fraction_mean(spec, 0.10),
        "snow_top10": top_fraction_mean(snow, 0.10),
        "grad_top10": top_fraction_mean(grad, 0.10),
        "thin_water_top10": top_fraction_mean(thin_water, 0.10),
        "wet_bottom_minus_top": region_delta(wet),
        "specular_bottom_minus_top": region_delta(spec),
        "snow_bottom_minus_top": region_delta(snow),
        "grad_bottom_minus_top": region_delta(grad),
        "gray_bottom_minus_top": region_delta(gray),
        "thin_water_bottom_minus_top": region_delta(thin_water),
    }
    return features


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def top_fraction_mean(x: np.ndarray, fraction: float) -> float:
    flat = x.reshape(-1)
    k = max(1, int(flat.size * fraction))
    return float(np.partition(flat, flat.size - k)[flat.size - k :].mean())


def region_delta(x: np.ndarray) -> float:
    h = x.shape[0]
    return float(x[h // 2 :, :].mean() - x[: h // 2, :].mean())


def fisher_ratio(records: list[dict[str, float | str]], feature: str) -> float:
    values = np.asarray([float(row[feature]) for row in records], dtype=np.float64)
    labels = [str(row["class"]) for row in records]
    grand = float(values.mean())
    between = 0.0
    within = 0.0
    for label in sorted(set(labels)):
        group = np.asarray([float(row[feature]) for row in records if row["class"] == label], dtype=np.float64)
        if group.size == 0:
            continue
        between += group.size * float((group.mean() - grand) ** 2)
        within += float(((group - group.mean()) ** 2).sum())
    return float(between / max(within, 1e-12))


def feature_kind(name: str) -> str:
    if "bottom_minus_top" in name:
        return "vertical_region_delta"
    if "top10" in name:
        return "patch_distribution"
    return "global_patch_stat"


def pct_rank(rows: list[dict[str, float | str]], name: str) -> str:
    for idx, row in enumerate(rows, start=1):
        if row["feature"] == name:
            return f"#{idx}"
    return "-"


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.4f}"


def to_markdown(summary: dict, rows: list[dict[str, float | str]]) -> str:
    lines = [
        "# RSCD Patch Region-Cue Statistics",
        "",
        summary["claim_boundary"],
        "",
        "## Sample",
        "",
        f"- Samples: `{summary['num_samples']}`",
        f"- Classes: `{summary['num_classes']}`",
        f"- Observed sampled sizes: `{summary['image_sizes']}`",
        "",
        "## Top Feature Separability",
        "",
        "| rank | feature | kind | Fisher ratio |",
        "|---:|---|---|---:|",
    ]
    for idx, row in enumerate(rows[:16], start=1):
        lines.append(f"| {idx} | `{row['feature']}` | `{row['kind']}` | {fmt(float(row['fisher_ratio']))} |")
    lines.extend(
        [
            "",
            "## Vertical Region Delta Features",
            "",
            "| feature | rank | Fisher ratio |",
            "|---|---:|---:|",
        ]
    )
    for row in [item for item in rows if item["kind"] == "vertical_region_delta"]:
        lines.append(f"| `{row['feature']}` | {pct_rank(rows, str(row['feature']))} | {fmt(float(row['fisher_ratio']))} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If vertical bottom-vs-top features rank below global or top-fraction patch statistics, RSCD should not be described as using a tire-contact-region prior. The correct RSCD wording is patch-level texture/reflectance/low-saturation wetness evidence.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

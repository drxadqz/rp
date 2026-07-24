from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image, ImageStat

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_MANIFESTS = [
    Path("data/manifests_full/rscd_prepared_train.csv"),
    Path("data/manifests_full/roadsaw_train.csv"),
    Path("data/manifests_full/roadsc_train.csv"),
]
DEFAULT_CONFIG = Path("configs/experiments/paper_protocol/lodo_roadsaw_full_faf.yaml")
DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/dataset_image_style_audit.md")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/dataset_image_style_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-samples-per-dataset", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    manifests = args.manifest or DEFAULT_MANIFESTS
    report = build_report(
        manifests=manifests,
        config=args.config,
        max_samples_per_dataset=max(1, int(args.max_samples_per_dataset)),
        seed=int(args.seed),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(
    *,
    manifests: list[Path],
    config: Path,
    max_samples_per_dataset: int,
    seed: int,
) -> dict[str, Any]:
    rows_by_dataset = sample_manifest_rows(manifests, max_samples_per_dataset, seed)
    samples = {}
    for dataset, rows in sorted(rows_by_dataset.items()):
        samples[dataset] = inspect_images(rows)
    config_info = load_config_info(config)
    cross = cross_dataset_signals(samples)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claim_boundary": (
            "This audit measures image-format and low-level style differences only. "
            "It does not prove tire-road friction accuracy or replace LODO evidence."
        ),
        "manifests": [str(path) for path in manifests],
        "config": str(config),
        "max_samples_per_dataset": max_samples_per_dataset,
        "config_image_pipeline": config_info,
        "datasets": samples,
        "cross_dataset_signals": cross,
        "recommendations": recommendations(samples, cross, config_info),
    }


def sample_manifest_rows(
    manifests: list[Path],
    max_samples_per_dataset: int,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict[str, str]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for manifest in manifests:
        if not manifest.exists():
            continue
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dataset = str(row.get("dataset") or row.get("domain_id") or manifest.stem)
                counts[dataset] += 1
                bucket = reservoirs[dataset]
                item = dict(row)
                item["_manifest"] = str(manifest)
                if len(bucket) < max_samples_per_dataset:
                    bucket.append(item)
                else:
                    idx = rng.randrange(counts[dataset])
                    if idx < max_samples_per_dataset:
                        bucket[idx] = item
    return dict(reservoirs)


def inspect_images(rows: list[dict[str, str]]) -> dict[str, Any]:
    records = []
    missing = []
    for row in rows:
        path = Path(str(row.get("image_path") or ""))
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                records.append(image_record(image, path, row))
        except (OSError, ValueError) as exc:
            missing.append(f"{path} ({exc})")
    return summarize_records(records, missing)


def image_record(image: Image.Image, path: Path, row: dict[str, str]) -> dict[str, Any]:
    width, height = image.size
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    hsv = rgb.convert("HSV")
    gray_stat = ImageStat.Stat(gray)
    hsv_stat = ImageStat.Stat(hsv)
    return {
        "path": str(path),
        "dataset": row.get("dataset"),
        "class_label": row.get("class_label"),
        "width": width,
        "height": height,
        "aspect": width / height if height else None,
        "mode": image.mode,
        "format": image.format,
        "suffix": path.suffix.lower(),
        "brightness": gray_stat.mean[0] / 255.0,
        "contrast": gray_stat.stddev[0] / 255.0,
        "saturation": hsv_stat.mean[1] / 255.0,
    }


def summarize_records(records: list[dict[str, Any]], missing: list[str]) -> dict[str, Any]:
    if not records:
        return {"num_samples": 0, "num_missing": len(missing), "missing_examples": missing[:10]}
    widths = [float(row["width"]) for row in records]
    heights = [float(row["height"]) for row in records]
    aspects = [float(row["aspect"]) for row in records if row.get("aspect") is not None]
    brightness = [float(row["brightness"]) for row in records]
    contrast = [float(row["contrast"]) for row in records]
    saturation = [float(row["saturation"]) for row in records]
    return {
        "num_samples": len(records),
        "num_missing": len(missing),
        "missing_examples": missing[:10],
        "width": stats(widths),
        "height": stats(heights),
        "aspect": stats(aspects),
        "brightness": stats(brightness),
        "contrast": stats(contrast),
        "saturation": stats(saturation),
        "formats": dict(Counter(str(row.get("format") or "-") for row in records)),
        "modes": dict(Counter(str(row.get("mode") or "-") for row in records)),
        "suffixes": dict(Counter(str(row.get("suffix") or "-") for row in records)),
        "class_counts": dict(Counter(str(row.get("class_label") or "-") for row in records)),
    }


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    mu = mean(values)
    var = mean([(item - mu) ** 2 for item in values])
    return {
        "mean": mu,
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "std": math.sqrt(var),
    }


def load_config_info(config: Path) -> dict[str, Any]:
    if yaml is None or not config.exists():
        return {}
    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    data = cfg.get("data") or {}
    aug = data.get("augmentation") or {}
    return {
        "image_size": data.get("image_size"),
        "random_resized_crop": aug.get("random_resized_crop"),
        "crop_scale": aug.get("crop_scale"),
        "crop_ratio": aug.get("crop_ratio"),
        "color_jitter": aug.get("color_jitter"),
        "random_grayscale_p": aug.get("random_grayscale_p"),
        "gaussian_blur_p": aug.get("gaussian_blur_p"),
        "fourier_low_freq_jitter_p": aug.get("fourier_low_freq_jitter_p"),
        "resize_mode": aug.get("resize_mode", "stretch"),
    }


def cross_dataset_signals(samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    present = {name: row for name, row in samples.items() if row.get("num_samples", 0) > 0}
    brightness = {name: _dig(row, "brightness", "mean") for name, row in present.items()}
    contrast = {name: _dig(row, "contrast", "mean") for name, row in present.items()}
    saturation = {name: _dig(row, "saturation", "mean") for name, row in present.items()}
    aspect = {name: _dig(row, "aspect", "median") for name, row in present.items()}
    width = {name: _dig(row, "width", "median") for name, row in present.items()}
    height = {name: _dig(row, "height", "median") for name, row in present.items()}
    return {
        "brightness_range": value_range(brightness),
        "contrast_range": value_range(contrast),
        "saturation_range": value_range(saturation),
        "aspect_median_range": value_range(aspect),
        "width_median_range": value_range(width),
        "height_median_range": value_range(height),
        "mixed_file_suffixes": {
            name: row.get("suffixes", {})
            for name, row in present.items()
            if len(row.get("suffixes", {})) > 1
        },
        "non_rgb_modes": {
            name: row.get("modes", {})
            for name, row in present.items()
            if any(mode not in {"RGB", "-"} for mode in row.get("modes", {}))
        },
    }


def value_range(values: dict[str, float | None]) -> dict[str, Any]:
    clean = {key: value for key, value in values.items() if value is not None}
    if not clean:
        return {"values": values, "span": None}
    return {"values": clean, "span": max(clean.values()) - min(clean.values())}


def recommendations(
    samples: dict[str, dict[str, Any]],
    cross: dict[str, Any],
    config_info: dict[str, Any],
) -> list[str]:
    out = []
    image_size = config_info.get("image_size")
    if image_size:
        out.append(
            f"Keep a fixed model input size of {image_size} and report native size/aspect differences as a shortcut risk."
        )
    if _span(cross, "width_median_range") or _span(cross, "height_median_range"):
        out.append(
            "Run or retain aspect-ratio-robust resize/crop policy; compare against a letterbox resize candidate if RoadSaW/RSCD native aspect ratios diverge."
        )
    if (_span(cross, "brightness_range") or 0.0) > 0.08 or (_span(cross, "saturation_range") or 0.0) > 0.08:
        out.append(
            "Treat color and illumination as dataset shortcuts; prioritize Fourier amplitude jitter, color jitter, grayscale probability, and possible CLAHE/Retinex candidates."
        )
    if cross.get("mixed_file_suffixes") or cross.get("non_rgb_modes"):
        out.append(
            "Normalize all samples to RGB tensors and document file-format/mode heterogeneity; do not let suffix or alpha-channel artifacts leak dataset identity."
        )
    out.append(
        "Judge every normalization candidate by dataset-ID probe, held-out RoadSaW LODO, low-friction recall, and coverage-width tradeoff rather than pooled accuracy alone."
    )
    return out


def _span(cross: dict[str, Any], key: str) -> float | None:
    value = cross.get(key) or {}
    span = value.get("span")
    return float(span) if span is not None else None


def _dig(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Image Style Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Image Pipeline",
        "",
        f"- Config: `{report['config']}`.",
        f"- Image size: `{report['config_image_pipeline'].get('image_size')}`.",
        f"- Augmentation: `{json.dumps(report['config_image_pipeline'], ensure_ascii=False)}`.",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | samples | size median | aspect median | brightness | contrast | saturation | suffixes | modes |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, row in sorted(report["datasets"].items()):
        lines.append(
            "| {name} | {n} | {w}x{h} | {aspect} | {b} | {c} | {s} | {suffixes} | {modes} |".format(
                name=name,
                n=row.get("num_samples"),
                w=_fmt_num(_dig(row, "width", "median"), digits=0),
                h=_fmt_num(_dig(row, "height", "median"), digits=0),
                aspect=_fmt_num(_dig(row, "aspect", "median")),
                b=_fmt_num(_dig(row, "brightness", "mean")),
                c=_fmt_num(_dig(row, "contrast", "mean")),
                s=_fmt_num(_dig(row, "saturation", "mean")),
                suffixes=_dict_short(row.get("suffixes", {})),
                modes=_dict_short(row.get("modes", {})),
            )
        )
    lines.extend(["", "## Cross-Dataset Signals", ""])
    for key, value in report["cross_dataset_signals"].items():
        lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
    lines.extend(["", "## Recommendations", ""])
    for item in report["recommendations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _dict_short(value: dict[str, Any]) -> str:
    if not value:
        return "-"
    return ", ".join(f"{key}:{val}" for key, val in sorted(value.items())[:4])


if __name__ == "__main__":
    main()

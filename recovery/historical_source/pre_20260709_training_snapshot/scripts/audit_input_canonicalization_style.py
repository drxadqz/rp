from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
import yaml
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.transforms import BottomSquareCropResize, GrayWorldColorConstancy, LetterboxResize


DEFAULT_MANIFESTS = [
    Path("data/manifests_full/rscd_prepared_train.csv"),
    Path("data/manifests_full/roadsaw_train.csv"),
    Path("data/manifests_full/roadsc_train.csv"),
]
DEFAULT_CONFIGS = [
    Path("configs/experiments/paper_protocol/v5_full_faf.yaml"),
    Path("configs/experiments/paper_protocol/v15_lean_bottom_square_style_safety.yaml"),
    Path("configs/experiments/paper_protocol/v16_lean_bottom_square_color_constancy_safety.yaml"),
]
DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/input_canonicalization_style_audit.md")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/input_canonicalization_style_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, default=None)
    parser.add_argument("--config", action="append", type=Path, default=None)
    parser.add_argument("--max-samples-per-dataset", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(
        manifests=args.manifest or DEFAULT_MANIFESTS,
        configs=args.config or DEFAULT_CONFIGS,
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
    configs: list[Path],
    max_samples_per_dataset: int,
    seed: int,
) -> dict[str, Any]:
    rows_by_dataset = sample_manifest_rows(manifests, max_samples_per_dataset, seed)
    config_rows = []
    for config in configs:
        cfg = yaml.safe_load(config.read_text(encoding="utf-8")) if config.exists() else {}
        data = (cfg or {}).get("data", {})
        aug = data.get("augmentation", {}) or {}
        transform_info = {
            "image_size": int(data.get("image_size", 224)),
            "resize_mode": str(aug.get("resize_mode", "stretch")).lower(),
            "gray_world_alpha": float(aug.get("gray_world_alpha", 0.0) or 0.0),
            "fourier_low_freq_jitter_p": float(aug.get("fourier_low_freq_jitter_p", 0.0) or 0.0),
            "random_resized_crop": bool(aug.get("random_resized_crop", False)),
        }
        datasets = {
            dataset: inspect_dataset(rows, transform_info)
            for dataset, rows in sorted(rows_by_dataset.items())
        }
        cross = cross_dataset_signals(datasets)
        config_rows.append(
            {
                "run": config.stem,
                "config": str(config),
                "transform": transform_info,
                "datasets": datasets,
                "cross_dataset_signals": cross,
            }
        )

    baseline = config_rows[0]["cross_dataset_signals"] if config_rows else {}
    for row in config_rows:
        row["relative_to_first_config"] = relative_change(row["cross_dataset_signals"], baseline)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claim_boundary": (
            "This audit measures deterministic input-style canonicalization only. "
            "It does not prove friction accuracy, but it checks whether candidate "
            "preprocessing can reduce dataset-visible low-level style gaps."
        ),
        "manifests": [str(path) for path in manifests],
        "max_samples_per_dataset": int(max_samples_per_dataset),
        "configs": config_rows,
        "recommendations": recommendations(config_rows),
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


def inspect_dataset(rows: list[dict[str, str]], transform_info: dict[str, Any]) -> dict[str, Any]:
    records = []
    missing = []
    for row in rows:
        path = Path(str(row.get("image_path") or ""))
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                records.append(image_record(image, transform_info))
        except (OSError, ValueError) as exc:
            missing.append(f"{path} ({exc})")
    return summarize(records, missing)


def image_record(image: Image.Image, transform_info: dict[str, Any]) -> dict[str, Any]:
    tensor = apply_canonical_transform(image.convert("RGB"), transform_info)
    rgb_image = tensor_to_image(tensor)
    width, height = rgb_image.size
    gray = rgb_image.convert("L")
    hsv = rgb_image.convert("HSV")
    gray_stat = ImageStat.Stat(gray)
    hsv_stat = ImageStat.Stat(hsv)
    channel_means = [float(item) / 255.0 for item in ImageStat.Stat(rgb_image).mean]
    return {
        "width": width,
        "height": height,
        "aspect": width / height if height else None,
        "brightness": gray_stat.mean[0] / 255.0,
        "contrast": gray_stat.stddev[0] / 255.0,
        "saturation": hsv_stat.mean[1] / 255.0,
        "red_mean": channel_means[0],
        "green_mean": channel_means[1],
        "blue_mean": channel_means[2],
        "channel_mean_spread": max(channel_means) - min(channel_means),
    }


def apply_canonical_transform(image: Image.Image, transform_info: dict[str, Any]) -> torch.Tensor:
    image_size = int(transform_info.get("image_size", 224))
    resize_mode = str(transform_info.get("resize_mode", "stretch")).lower()
    if resize_mode in {"letterbox", "pad", "aspect_pad"}:
        image = LetterboxResize(image_size)(image)
    elif resize_mode in {"bottom_square", "bottom_center_square", "road_bottom_square"}:
        image = BottomSquareCropResize(image_size)(image)
    else:
        image = transforms.functional.resize(image, [image_size, image_size])
    tensor = transforms.functional.to_tensor(image)
    alpha = float(transform_info.get("gray_world_alpha", 0.0) or 0.0)
    if alpha > 0:
        tensor = GrayWorldColorConstancy(alpha=alpha)(tensor)
    return tensor.clamp(0.0, 1.0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return transforms.functional.to_pil_image(array)


def summarize(records: list[dict[str, Any]], missing: list[str]) -> dict[str, Any]:
    if not records:
        return {"num_samples": 0, "num_missing": len(missing), "missing_examples": missing[:10]}
    keys = [
        "width",
        "height",
        "aspect",
        "brightness",
        "contrast",
        "saturation",
        "red_mean",
        "green_mean",
        "blue_mean",
        "channel_mean_spread",
    ]
    return {
        "num_samples": len(records),
        "num_missing": len(missing),
        "missing_examples": missing[:10],
        **{key: stats([float(row[key]) for row in records if row.get(key) is not None]) for key in keys},
    }


def cross_dataset_signals(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    present = {name: row for name, row in datasets.items() if row.get("num_samples", 0) > 0}
    signals = {
        "brightness_span": dataset_stat_span(present, "brightness"),
        "contrast_span": dataset_stat_span(present, "contrast"),
        "saturation_span": dataset_stat_span(present, "saturation"),
        "red_mean_span": dataset_stat_span(present, "red_mean"),
        "green_mean_span": dataset_stat_span(present, "green_mean"),
        "blue_mean_span": dataset_stat_span(present, "blue_mean"),
        "channel_mean_spread_span": dataset_stat_span(present, "channel_mean_spread"),
        "aspect_median_span": dataset_stat_span(present, "aspect", field="median"),
        "width_median_span": dataset_stat_span(present, "width", field="median"),
        "height_median_span": dataset_stat_span(present, "height", field="median"),
    }
    color_span = max(
        _span_value(signals["red_mean_span"]),
        _span_value(signals["green_mean_span"]),
        _span_value(signals["blue_mean_span"]),
    )
    style_score = (
        _span_value(signals["brightness_span"])
        + _span_value(signals["saturation_span"])
        + color_span
        + 0.5 * _span_value(signals["channel_mean_spread_span"])
        + 0.25 * _span_value(signals["contrast_span"])
    )
    signals["style_gap_score"] = style_score
    return signals


def dataset_stat_span(
    datasets: dict[str, dict[str, Any]],
    key: str,
    *,
    field: str = "mean",
) -> dict[str, Any]:
    values = {
        name: _dig(row, key, field)
        for name, row in datasets.items()
        if _dig(row, key, field) is not None
    }
    if not values:
        return {"values": values, "span": None}
    return {"values": values, "span": max(values.values()) - min(values.values())}


def relative_change(cur: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in cur.items():
        if isinstance(value, dict) and "span" in value:
            cur_span = value.get("span")
            base_span = (base.get(key) or {}).get("span") if isinstance(base.get(key), dict) else None
            out[key] = _relative(cur_span, base_span)
        elif key == "style_gap_score":
            out[key] = _relative(value, base.get(key))
    return out


def _relative(cur: Any, base: Any) -> dict[str, float | None]:
    if cur is None or base in {None, 0}:
        return {"delta": None, "relative": None}
    cur_f = float(cur)
    base_f = float(base)
    return {"delta": cur_f - base_f, "relative": cur_f / base_f}


def recommendations(config_rows: list[dict[str, Any]]) -> list[str]:
    if not config_rows:
        return ["Run the audit after configs are generated."]
    ranked = sorted(
        config_rows,
        key=lambda row: float(row.get("cross_dataset_signals", {}).get("style_gap_score", 1e9)),
    )
    best = ranked[0]
    out = [
        f"Lowest deterministic style-gap score: `{best['run']}`.",
        "Treat this as preprocessing evidence only; final retention still requires dataset-ID probes, LODO, and task metrics.",
    ]
    first = config_rows[0]
    if best["run"] != first["run"]:
        rel = best.get("relative_to_first_config", {}).get("style_gap_score", {})
        value = rel.get("relative")
        if value is not None:
            out.append(
                f"`{best['run']}` reduces the style-gap score to {100.0 * float(value):.1f}% of `{first['run']}` in this diagnostic."
            )
    out.append(
        "If color canonicalization reduces style score but later hurts RoadSaW wetness F1, keep v16 as a negative ablation and prefer v15/v14."
    )
    return out


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


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Input Canonicalization Style Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Cross-Dataset Style Signals",
        "",
        "| Run | resize | GrayWorld | style score | brightness span | saturation span | RGB span max | channel-spread span | aspect span | relative score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("configs", []):
        cross = row.get("cross_dataset_signals", {})
        transform = row.get("transform", {})
        rgb_span = max(
            _span_value(cross.get("red_mean_span")),
            _span_value(cross.get("green_mean_span")),
            _span_value(cross.get("blue_mean_span")),
        )
        rel = (row.get("relative_to_first_config") or {}).get("style_gap_score", {}).get("relative")
        lines.append(
            "| {run} | {resize} | {gray} | {score} | {bright} | {sat} | {rgb} | {spread} | {aspect} | {rel} |".format(
                run=row.get("run"),
                resize=transform.get("resize_mode"),
                gray=_fmt_abs(transform.get("gray_world_alpha")),
                score=_fmt_abs(cross.get("style_gap_score")),
                bright=_fmt_abs(_span_value(cross.get("brightness_span"))),
                sat=_fmt_abs(_span_value(cross.get("saturation_span"))),
                rgb=_fmt_abs(rgb_span),
                spread=_fmt_abs(_span_value(cross.get("channel_mean_spread_span"))),
                aspect=_fmt_abs(_span_value(cross.get("aspect_median_span"))),
                rel=_fmt_abs(rel),
            )
        )

    lines.extend(["", "## Dataset Means", ""])
    for row in report.get("configs", []):
        lines.extend(["", f"### {row.get('run')}", ""])
        lines.append("| Dataset | samples | brightness | contrast | saturation | R | G | B | channel spread |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for dataset, stats_row in sorted((row.get("datasets") or {}).items()):
            lines.append(
                "| {dataset} | {n} | {b} | {c} | {s} | {r} | {g} | {blue} | {spread} |".format(
                    dataset=dataset,
                    n=stats_row.get("num_samples"),
                    b=_fmt_abs(_dig(stats_row, "brightness", "mean")),
                    c=_fmt_abs(_dig(stats_row, "contrast", "mean")),
                    s=_fmt_abs(_dig(stats_row, "saturation", "mean")),
                    r=_fmt_abs(_dig(stats_row, "red_mean", "mean")),
                    g=_fmt_abs(_dig(stats_row, "green_mean", "mean")),
                    blue=_fmt_abs(_dig(stats_row, "blue_mean", "mean")),
                    spread=_fmt_abs(_dig(stats_row, "channel_mean_spread", "mean")),
                )
            )

    lines.extend(["", "## Recommendations", ""])
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _span_value(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("span")
    return 0.0 if value is None else float(value)


def _dig(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    main()

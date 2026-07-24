from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


DEFAULT_MANIFEST_DIR = Path("data/manifests_full")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_FIG_DIR = DEFAULT_SUMMARY_DIR / "figures" / "dataset_integrity_view"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "dataset_integrity_view_audit.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "dataset_integrity_view_audit.json")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rscd-max-images", type=int, default=12000)
    parser.add_argument("--max-images-per-dataset", type=int, default=20000)
    parser.add_argument("--max-path-check-per-dataset", type=int, default=8000)
    parser.add_argument("--max-white-examples", type=int, default=40)
    parser.add_argument("--grid-samples-per-dataset", type=int, default=24)
    args = parser.parse_args()

    report = build_report(
        manifest_dir=args.manifest_dir,
        fig_dir=args.fig_dir,
        seed=args.seed,
        rscd_max_images=args.rscd_max_images,
        max_images_per_dataset=args.max_images_per_dataset,
        max_path_check_per_dataset=args.max_path_check_per_dataset,
        max_white_examples=args.max_white_examples,
        grid_samples_per_dataset=args.grid_samples_per_dataset,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(
    *,
    manifest_dir: Path,
    fig_dir: Path,
    seed: int,
    rscd_max_images: int,
    max_images_per_dataset: int,
    max_path_check_per_dataset: int,
    max_white_examples: int,
    grid_samples_per_dataset: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    fig_dir.mkdir(parents=True, exist_ok=True)

    manifests = _load_manifests(manifest_dir)
    all_df = pd.concat(manifests.values(), ignore_index=True)
    all_df["image_path_norm"] = all_df["image_path"].astype(str).str.strip()

    path_checks = _check_paths(all_df, max_per_dataset=max_path_check_per_dataset)
    dataset_rows: dict[str, Any] = {}
    sampled_records: list[dict[str, Any]] = []
    white_records: list[dict[str, Any]] = []
    grid_paths: dict[str, str] = {}
    white_grid_paths: dict[str, str] = {}

    for dataset, df in all_df.groupby("dataset", sort=True):
        sample_df = _image_sample(
            df,
            dataset=str(dataset),
            rscd_max_images=rscd_max_images,
            max_images_per_dataset=max_images_per_dataset,
        )
        records, errors = _image_stats(sample_df)
        sampled_records.extend(records)
        white = sorted(
            [row for row in records if row["near_white_flag"]],
            key=lambda row: (row["near_white_score"], row["white_pixel_frac"], row["brightness"]),
            reverse=True,
        )
        white_records.extend(white[:max_white_examples])
        dataset_rows[str(dataset)] = _dataset_summary(df, records, errors)
        grid = fig_dir / f"{dataset}_random_grid.jpg"
        white_grid = fig_dir / f"{dataset}_whitest_grid.jpg"
        _make_grid(_balanced_grid_records(records, grid_samples_per_dataset), grid, title=f"{dataset} random/balanced samples")
        _make_grid(white[:grid_samples_per_dataset], white_grid, title=f"{dataset} whitest/low-contrast samples")
        grid_paths[str(dataset)] = str(grid)
        white_grid_paths[str(dataset)] = str(white_grid)

    cross = _cross_dataset_summary(dataset_rows)
    recommendation = _recommendation(dataset_rows, cross)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_dir": str(manifest_dir),
        "claim_boundary": (
            "This audit checks local file integrity, image geometry, and low-level appearance. "
            "It does not prove measured tire-road friction accuracy."
        ),
        "path_checks": path_checks,
        "dataset_rows": dataset_rows,
        "cross_dataset": cross,
        "white_records_top": white_records[: max_white_examples * max(1, len(dataset_rows))],
        "grid_paths": grid_paths,
        "white_grid_paths": white_grid_paths,
        "recommendation": recommendation,
        "protocol_implications": _protocol_implications(recommendation),
    }


def _load_manifests(manifest_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(manifest_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df["manifest_file"] = str(path)
        frames[path.stem] = df
    if not frames:
        raise FileNotFoundError(f"No CSV manifests found under {manifest_dir}")
    return frames


def _check_paths(df: pd.DataFrame, *, max_per_dataset: int) -> dict[str, Any]:
    rows = []
    all_unique_paths = sorted(set(df["image_path_norm"].astype(str)))
    checked_unique_paths: list[str] = []
    existing = 0
    missing_examples: list[str] = []
    by_dataset_check: dict[str, dict[str, Any]] = {}
    for dataset, sub in df.groupby("dataset", sort=True):
        paths = sub["image_path_norm"].astype(str)
        unique = paths.nunique()
        unique_list = sorted(set(paths))
        if len(unique_list) > max_per_dataset:
            rng = random.Random(31 + len(str(dataset)))
            check_list = rng.sample(unique_list, max_per_dataset)
            check_mode = "sampled"
        else:
            check_list = unique_list
            check_mode = "full"
        checked_unique_paths.extend(check_list)
        dataset_existing = 0
        dataset_missing = 0
        existence = _batched_path_exists(check_list)
        for path_str in check_list:
            ok = existence.get(path_str, False)
            if ok:
                existing += 1
                dataset_existing += 1
            else:
                dataset_missing += 1
                if len(missing_examples) < 20:
                    missing_examples.append(path_str)
        by_dataset_check[str(dataset)] = {
            "check_mode": check_mode,
            "checked_unique_paths": int(len(check_list)),
            "existing_checked_paths": int(dataset_existing),
            "missing_checked_paths": int(dataset_missing),
            "sample_missing_rate": _safe_div(dataset_missing, len(check_list)),
        }
        duplicate_rows = int(len(paths) - unique)
        rows.append(
            {
                "dataset": str(dataset),
                "rows": int(len(sub)),
                "unique_paths": int(unique),
                "duplicate_rows": duplicate_rows,
                "splits": {str(k): int(v) for k, v in sub["split"].astype(str).value_counts().sort_index().items()},
                "classes": {str(k): int(v) for k, v in sub["class_label"].astype(str).value_counts().sort_index().items()},
                **by_dataset_check[str(dataset)],
            }
        )
    return {
        "total_rows": int(len(df)),
        "total_unique_paths": int(len(all_unique_paths)),
        "check_mode": "full_for_small_datasets_sampled_for_large_datasets",
        "checked_unique_paths": int(len(checked_unique_paths)),
        "existing_checked_paths": int(existing),
        "missing_checked_paths": int(len(checked_unique_paths) - existing),
        "missing_examples": missing_examples,
        "by_dataset": rows,
    }


def _batched_path_exists(paths: list[str]) -> dict[str, bool]:
    """Check many sibling files by scanning each parent directory once."""
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
        lower_names = {name.lower() for name in names}
        for raw, name in items:
            out[raw] = name in names or name.lower() in lower_names
    return out


def _image_sample(
    df: pd.DataFrame,
    *,
    dataset: str,
    rscd_max_images: int,
    max_images_per_dataset: int,
) -> pd.DataFrame:
    df = df.copy()
    df["image_path_norm"] = df["image_path"].astype(str).str.strip()
    unique = df.drop_duplicates("image_path_norm")
    cap = min(max_images_per_dataset, rscd_max_images if dataset == "rscd" else max_images_per_dataset)
    if len(unique) > cap:
        per_class = max(1, cap // max(1, unique["class_label"].nunique()))
        parts = []
        for _, sub in unique.groupby("class_label", sort=True):
            n = min(len(sub), per_class)
            parts.append(sub.sample(n=n, random_state=13))
        sample = pd.concat(parts, ignore_index=True)
        if len(sample) < cap:
            rest = unique.drop(sample.index, errors="ignore")
            if len(rest) > 0:
                sample = pd.concat(
                    [sample, rest.sample(n=min(len(rest), cap - len(sample)), random_state=17)],
                    ignore_index=True,
                )
        return sample
    return unique


def _image_stats(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        path_str = str(getattr(row, "image_path_norm", getattr(row, "image_path"))).strip()
        path = Path(path_str)
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                arr = np.asarray(img, dtype=np.float32) / 255.0
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            errors.append(
                {
                    "image_path": path_str,
                    "dataset": str(getattr(row, "dataset")),
                    "class_label": str(getattr(row, "class_label")),
                    "error": repr(exc),
                }
            )
            continue
        gray = arr.mean(axis=2)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        saturation = _mean_saturation(arr)
        white_pixel_frac = float(((arr[:, :, 0] > 0.95) & (arr[:, :, 1] > 0.95) & (arr[:, :, 2] > 0.95)).mean())
        near_white_score = brightness + white_pixel_frac - contrast - 0.5 * saturation
        near_white_flag = bool(brightness >= 0.82 and white_pixel_frac >= 0.25 and contrast <= 0.16 and saturation <= 0.20)
        h, w = gray.shape
        records.append(
            {
                "image_path": path_str,
                "dataset": str(getattr(row, "dataset")),
                "split": str(getattr(row, "split")),
                "class_label": str(getattr(row, "class_label")),
                "friction_label": str(getattr(row, "friction_label")),
                "wetness_label": str(getattr(row, "wetness_label")),
                "snow_label": str(getattr(row, "snow_label")),
                "risk_label": str(getattr(row, "risk_label")),
                "width": int(w),
                "height": int(h),
                "aspect": float(w / max(h, 1)),
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "white_pixel_frac": white_pixel_frac,
                "near_white_score": near_white_score,
                "near_white_flag": near_white_flag,
            }
        )
    return records, errors


def _mean_saturation(arr: np.ndarray) -> float:
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx <= 1e-6, 0.0, (mx - mn) / np.maximum(mx, 1e-6))
    return float(sat.mean())


def _dataset_summary(df: pd.DataFrame, records: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    by_class_white = Counter(row["class_label"] for row in records if row["near_white_flag"])
    by_class_total = Counter(row["class_label"] for row in records)
    dims = Counter(f"{row['width']}x{row['height']}" for row in records)
    aspect_values = [row["aspect"] for row in records]
    out = {
        "rows": int(len(df)),
        "unique_paths": int(df["image_path"].astype(str).str.strip().nunique()),
        "sampled_images": int(len(records)),
        "decode_errors": int(len(errors)),
        "decode_error_examples": errors[:10],
        "splits": {str(k): int(v) for k, v in df["split"].astype(str).value_counts().sort_index().items()},
        "classes": {str(k): int(v) for k, v in df["class_label"].astype(str).value_counts().sort_index().items()},
        "dimension_top": {str(k): int(v) for k, v in dims.most_common(8)},
        "width": _num_summary([row["width"] for row in records]),
        "height": _num_summary([row["height"] for row in records]),
        "aspect": _num_summary(aspect_values),
        "brightness": _num_summary([row["brightness"] for row in records]),
        "contrast": _num_summary([row["contrast"] for row in records]),
        "saturation": _num_summary([row["saturation"] for row in records]),
        "white_pixel_frac": _num_summary([row["white_pixel_frac"] for row in records]),
        "near_white": {
            "count": int(sum(1 for row in records if row["near_white_flag"])),
            "rate": _safe_div(sum(1 for row in records if row["near_white_flag"]), len(records)),
            "by_class": {
                cls: {
                    "count": int(count),
                    "sample_total": int(by_class_total[cls]),
                    "rate": _safe_div(count, by_class_total[cls]),
                }
                for cls, count in by_class_white.most_common()
            },
        },
    }
    out["view_inference"] = _view_inference(str(df["dataset"].iloc[0]), out)
    return out


def _num_summary(values: list[float | int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p05": None, "median": None, "mean": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _view_inference(dataset: str, row: dict[str, Any]) -> dict[str, str]:
    aspect = row.get("aspect", {}).get("median")
    dims = row.get("dimension_top", {})
    if dataset == "rscd":
        return {
            "inference": "local_patch_or_narrow_forward_crop",
            "evidence": f"dominant dimensions {dims}; median aspect {aspect}. This is visually/geometry-wise unlike square RoadSaW/RoadSC prepared crops.",
            "caution": "Confirm with the original RSCD paper/source before describing it as a left/right wheel camera; local geometry alone cannot prove sensor placement.",
        }
    if dataset in {"roadsaw", "roadsc"}:
        return {
            "inference": "square_prepared_vehicle_scene_crop",
            "evidence": f"dominant dimensions {dims}; median aspect {aspect}. Filenames and directory names indicate LKW video-derived prepared crops.",
            "caution": "The prepared square crops are not the same visual domain as RSCD narrow patches; cross-dataset training must be framed as domain generalization, not a single homogeneous benchmark.",
        }
    return {"inference": "unknown", "evidence": "", "caution": ""}


def _cross_dataset_summary(dataset_rows: dict[str, Any]) -> dict[str, Any]:
    med_brightness = {k: _dig(v, "brightness", "median") for k, v in dataset_rows.items()}
    med_aspect = {k: _dig(v, "aspect", "median") for k, v in dataset_rows.items()}
    med_width = {k: _dig(v, "width", "median") for k, v in dataset_rows.items()}
    med_height = {k: _dig(v, "height", "median") for k, v in dataset_rows.items()}
    white_rates = {k: _dig(v, "near_white", "rate") for k, v in dataset_rows.items()}
    return {
        "median_brightness": med_brightness,
        "median_aspect": med_aspect,
        "median_width": med_width,
        "median_height": med_height,
        "near_white_rate": white_rates,
        "brightness_span": _span(med_brightness),
        "aspect_span": _span(med_aspect),
        "width_span": _span(med_width),
        "height_span": _span(med_height),
        "white_rate_span": _span(white_rates),
    }


def _dig(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _span(values: dict[str, Any]) -> float | None:
    nums = [float(v) for v in values.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(max(nums) - min(nums)) if nums else None


def _recommendation(dataset_rows: dict[str, Any], cross: dict[str, Any]) -> dict[str, Any]:
    aspect_span = cross.get("aspect_span") or 0.0
    brightness_span = cross.get("brightness_span") or 0.0
    white_rates = cross.get("near_white_rate") or {}
    reasons = []
    if aspect_span > 0.25:
        reasons.append("large_view_geometry_gap")
    if brightness_span > 0.08:
        reasons.append("large_photometric_gap")
    if any((v or 0.0) > 0.10 for v in white_rates.values()):
        reasons.append("substantial_near_white_subset")

    if reasons:
        route = "hierarchical_protocol_not_naive_pooling"
        decision = (
            "Do not treat RSCD, RoadSaW, and RoadSC as one homogeneous image distribution. "
            "Use single-dataset FAF-vs-ConvNeXt as the primary fair benchmark, and use multi-dataset/LODO as a domain-generalization stress test."
        )
    else:
        route = "pooled_training_feasible_with_lodo"
        decision = (
            "Pooled training is visually less risky, but still needs LODO and dataset-ID probes before any generalization claim."
        )
    return {
        "route": route,
        "reasons": reasons,
        "decision": decision,
        "preferred_next_experiments": [
            "single_rscd_full_faf vs baseline_single_rscd_global_convnext",
            "single_roadsaw_full_faf vs baseline_single_roadsaw_global_convnext",
            "single_roadsc_full_faf vs baseline_single_roadsc_global_convnext",
            "lodo_roadsaw_full_faf / lodo_rscd_full_faf / lodo_roadsc_full_faf",
            "v15/v16 input canonicalization only if they reduce shortcut without erasing wetness cues",
        ],
    }


def _protocol_implications(recommendation: dict[str, Any]) -> list[str]:
    if recommendation.get("route") == "hierarchical_protocol_not_naive_pooling":
        return [
            "Primary numeric claims should be single-dataset same-split FAF vs ConvNeXt, not pooled multi-dataset accuracy.",
            "LODO should be reported as a harder cross-sensor/cross-view stress test; a failure is valid evidence of domain mismatch.",
            "Use bottom-square/ROI/canonicalization candidates to test whether a common road-contact view can be constructed.",
            "Do not overclaim measured friction; call the output a weak visual friction-affordance interval.",
        ]
    return [
        "Pooled training may be used as the main method after LODO and shortcut probes pass.",
        "Still report per-dataset breakdowns and conditional calibration.",
    ]


def _balanced_grid_records(records: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_class[row["class_label"]].append(row)
    out: list[dict[str, Any]] = []
    for cls in sorted(by_class):
        candidates = by_class[cls]
        if candidates:
            out.append(random.choice(candidates))
        if len(out) >= n:
            break
    if len(out) < n:
        rest = [row for row in records if row not in out]
        random.shuffle(rest)
        out.extend(rest[: n - len(out)])
    return out[:n]


def _make_grid(records: list[dict[str, Any]], out_path: Path, *, title: str, cell: int = 180) -> None:
    if not records:
        return
    cols = 4
    rows = math.ceil(len(records) / cols)
    label_h = 44
    title_h = 34
    canvas = Image.new("RGB", (cols * cell, rows * (cell + label_h) + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for i, row in enumerate(records):
        x = (i % cols) * cell
        y = title_h + (i // cols) * (cell + label_h)
        try:
            with Image.open(row["image_path"]) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                thumb = ImageOps.contain(img, (cell, cell))
        except Exception:
            thumb = Image.new("RGB", (cell, cell), (240, 80, 80))
        px = x + (cell - thumb.width) // 2
        py = y + (cell - thumb.height) // 2
        canvas.paste(thumb, (px, py))
        label = "{cls} b={b:.2f} w={w:.2f}".format(
            cls=str(row.get("class_label", ""))[:22],
            b=float(row.get("brightness", 0.0)),
            w=float(row.get("white_pixel_frac", 0.0)),
        )
        draw.text((x + 4, y + cell + 2), label, fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + cell + 16), Path(row["image_path"]).name[:30], fill=(70, 70, 70), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Integrity And View Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## File Integrity",
        "",
        f"- Rows: `{report['path_checks']['total_rows']}`.",
        f"- Unique image paths: `{report['path_checks']['total_unique_paths']}`.",
        f"- Path-check mode: `{report['path_checks']['check_mode']}`.",
        f"- Checked unique image paths: `{report['path_checks']['checked_unique_paths']}`.",
        f"- Existing checked paths: `{report['path_checks']['existing_checked_paths']}`.",
        f"- Missing checked paths: `{report['path_checks']['missing_checked_paths']}`.",
        "",
        "| Dataset | rows | unique paths | check mode | checked | missing checked | splits |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in report["path_checks"]["by_dataset"]:
        lines.append(
            "| {dataset} | {rows} | {unique} | {mode} | {checked} | {missing} | {splits} |".format(
                dataset=row["dataset"],
                rows=row["rows"],
                unique=row["unique_paths"],
                mode=row.get("check_mode"),
                checked=row.get("checked_unique_paths"),
                missing=row.get("missing_checked_paths"),
                splits=json.dumps(row["splits"], ensure_ascii=False, sort_keys=True),
            )
        )

    lines.extend(["", "## Image Geometry And Whiteness", ""])
    lines.append("| Dataset | sampled | top dimensions | median size | aspect | brightness | contrast | saturation | near-white |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for name, row in report["dataset_rows"].items():
        dims = ", ".join([f"{k}:{v}" for k, v in list(row["dimension_top"].items())[:3]])
        lines.append(
            "| {name} | {n} | {dims} | {w}x{h} | {aspect} | {b} | {c} | {s} | {white} |".format(
                name=name,
                n=row["sampled_images"],
                dims=dims,
                w=_fmt(_dig(row, "width", "median"), 1),
                h=_fmt(_dig(row, "height", "median"), 1),
                aspect=_fmt(_dig(row, "aspect", "median")),
                b=_fmt(_dig(row, "brightness", "median")),
                c=_fmt(_dig(row, "contrast", "median")),
                s=_fmt(_dig(row, "saturation", "median")),
                white=_fmt_pct(_dig(row, "near_white", "rate")),
            )
        )
    lines.extend(["", "## View Inference", ""])
    for name, row in report["dataset_rows"].items():
        view = row.get("view_inference", {})
        lines.append(f"- `{name}`: `{view.get('inference')}`. {view.get('evidence')} {view.get('caution')}")
    lines.extend(["", "## Near-White Classes", ""])
    for name, row in report["dataset_rows"].items():
        near = row.get("near_white", {}).get("by_class") or {}
        if not near:
            lines.append(f"- `{name}`: no near-white images under the conservative threshold in the sampled/opened set.")
            continue
        top = list(near.items())[:8]
        text = "; ".join(
            f"{cls}: {item['count']}/{item['sample_total']} ({_fmt_pct(item['rate'])})"
            for cls, item in top
        )
        lines.append(f"- `{name}`: {text}.")
    lines.extend(["", "## Sample Figures", ""])
    for name, path in report.get("grid_paths", {}).items():
        lines.append(f"- `{name}` balanced/random samples: `{path}`")
    for name, path in report.get("white_grid_paths", {}).items():
        lines.append(f"- `{name}` whitest samples: `{path}`")
    rec = report["recommendation"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Route: `{rec['route']}`.",
            f"- Reasons: `{json.dumps(rec['reasons'], ensure_ascii=False)}`.",
            f"- Decision: {rec['decision']}",
            "",
            "Protocol implications:",
        ]
    )
    for item in report["protocol_implications"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

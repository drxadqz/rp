from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_MANIFESTS = [
    Path("data/manifests_full/roadsaw_val.csv"),
    Path("data/manifests_full/roadsc_val.csv"),
    Path("data/manifests_full/rscd_prepared_val.csv"),
]
DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary/pseudo_segmentation_masks")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit lightweight pseudo road/contact masks and local material-region "
            "masks on RSCD/RoadSaW/RoadSC. This is a CPU-side decision tool for "
            "whether heavier SAM/Mask2Former-style pseudo-labeling is worth adding."
        )
    )
    parser.add_argument("--manifests", type=Path, nargs="*", default=DEFAULT_MANIFESTS)
    parser.add_argument("--samples-per-dataset", type=int, default=180)
    parser.add_argument("--overlays-per-dataset", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_DIR / "pseudo_segmentation_mask_audit.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_DIR / "pseudo_segmentation_mask_audit.md")
    args = parser.parse_args()

    df = _load_manifests(args.manifests)
    selected = _sample_rows(df, samples_per_dataset=int(args.samples_per_dataset), seed=int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    overlay_counts: dict[str, int] = {}
    for idx, item in selected.reset_index(drop=True).iterrows():
        row = _audit_one(
            item,
            index=int(idx),
            image_size=int(args.image_size),
            clusters=int(args.clusters),
        )
        if row is None:
            continue
        dataset = str(row.get("dataset", "unknown"))
        if overlay_counts.get(dataset, 0) < int(args.overlays_per_dataset):
            overlay = row.pop("_overlay")
            overlay_path = overlay_dir / _overlay_name(idx, row)
            cv2.imwrite(str(overlay_path), overlay)
            row["overlay_path"] = str(overlay_path)
            overlay_counts[dataset] = overlay_counts.get(dataset, 0) + 1
        else:
            row.pop("_overlay", None)
            row["overlay_path"] = None
        rows.append(row)

    report = _build_report(rows, args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def _load_manifests(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["manifest_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No readable manifests were provided.")
    return pd.concat(frames, ignore_index=True)


def _sample_rows(df: pd.DataFrame, *, samples_per_dataset: int, seed: int) -> pd.DataFrame:
    parts = []
    rng_seed = int(seed)
    for dataset, dataset_df in df.groupby("dataset", dropna=False):
        class_count = max(int(dataset_df["class_label"].nunique()), 1) if "class_label" in dataset_df else 1
        per_class = max(1, math.ceil(samples_per_dataset / class_count))
        class_parts = []
        for _, class_df in dataset_df.groupby("class_label", dropna=False):
            take = min(len(class_df), per_class)
            class_parts.append(class_df.sample(n=take, random_state=rng_seed))
            rng_seed += 1
        sampled = pd.concat(class_parts, ignore_index=True)
        if len(sampled) > samples_per_dataset:
            sampled = sampled.sample(n=samples_per_dataset, random_state=rng_seed)
            rng_seed += 1
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


def _audit_one(item: pd.Series, *, index: int, image_size: int, clusters: int) -> dict[str, Any] | None:
    path = Path(str(item["image_path"]))
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {
            "decode_ok": False,
            "image_path": str(path),
            "dataset": item.get("dataset"),
            "class_label": item.get("class_label"),
        }

    native_h, native_w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edge = _edge_magnitude(gray)
    texture = _local_std(gray, kernel=11)
    labels = _kmeans_regions(rgb_f, clusters=clusters)

    bottom_mask, center_bottom_mask, top_mask = _priors(image_size)
    road_mask, raw_road_mask = _pseudo_contact_mask(
        labels, value, saturation, edge, bottom_mask, center_bottom_mask, top_mask
    )
    mixture_map = _material_mixture_map(rgb_f, value, saturation, texture, edge, labels)
    mixture_mask = mixture_map >= float(np.quantile(mixture_map, 0.75))

    near_white = (value > 0.88) & (saturation < 0.18)
    low_texture = texture < 0.025
    specular = (value > 0.82) & (saturation < 0.24)
    dark_water = (value < 0.36) & (edge < 0.08)

    raw_road_area = float(raw_road_mask.mean())
    raw_center_bottom_coverage = _safe_ratio(
        np.logical_and(raw_road_mask, center_bottom_mask).sum(), center_bottom_mask.sum()
    )
    raw_top_mass = _safe_ratio(np.logical_and(raw_road_mask, top_mask).sum(), raw_road_mask.sum())
    road_area = float(road_mask.mean())
    center_bottom_coverage = _safe_ratio(np.logical_and(road_mask, center_bottom_mask).sum(), center_bottom_mask.sum())
    bottom_coverage = _safe_ratio(np.logical_and(road_mask, bottom_mask).sum(), bottom_mask.sum())
    top_mass = _safe_ratio(np.logical_and(road_mask, top_mask).sum(), road_mask.sum())
    mixture_area = float(mixture_mask.mean())
    mixture_on_road = _safe_ratio(np.logical_and(mixture_mask, road_mask).sum(), mixture_mask.sum())
    external_value = _external_mask_value(
        raw_road_area,
        raw_center_bottom_coverage,
        raw_top_mass,
        center_bottom_coverage,
    )
    overlay = _overlay(resized, road_mask, raw_road_mask, mixture_mask, bottom_mask)

    return {
        "decode_ok": True,
        "image_path": str(path),
        "dataset": item.get("dataset"),
        "class_label": item.get("class_label"),
        "friction_label": item.get("friction_label"),
        "wetness_label": item.get("wetness_label"),
        "snow_label": item.get("snow_label"),
        "risk_label": item.get("risk_label"),
        "native_width": int(native_w),
        "native_height": int(native_h),
        "brightness": float(value.mean()),
        "contrast": float(gray.std()),
        "saturation": float(saturation.mean()),
        "near_white_frac": float(near_white.mean()),
        "low_texture_frac": float(low_texture.mean()),
        "specular_frac": float(specular.mean()),
        "dark_water_frac": float(dark_water.mean()),
        "raw_pseudo_road_area": raw_road_area,
        "raw_pseudo_road_center_bottom_coverage": float(raw_center_bottom_coverage),
        "raw_pseudo_road_top_mass": float(raw_top_mass),
        "pseudo_road_area": road_area,
        "pseudo_road_bottom_coverage": float(bottom_coverage),
        "pseudo_road_center_bottom_coverage": float(center_bottom_coverage),
        "pseudo_road_top_mass": float(top_mass),
        "region_mixture_area": mixture_area,
        "region_mixture_on_road": float(mixture_on_road),
        "region_mixture_mean": float(mixture_map.mean()),
        "region_mixture_p90": float(np.quantile(mixture_map, 0.90)),
        "cluster_entropy": _cluster_entropy(labels),
        "external_mask_value": external_value,
        "_overlay": overlay,
    }


def _priors(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    y = yy / max(size - 1, 1)
    x = xx / max(size - 1, 1)
    bottom = y >= 0.45
    center_bottom = (y >= 0.50) & (np.abs(x - 0.5) <= 0.34)
    top = y <= 0.25
    return bottom, center_bottom, top


def _pseudo_contact_mask(
    labels: np.ndarray,
    value: np.ndarray,
    saturation: np.ndarray,
    edge: np.ndarray,
    bottom_mask: np.ndarray,
    center_bottom_mask: np.ndarray,
    top_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        area = float(mask.mean())
        if area <= 0:
            continue
        bottom_density = _safe_ratio(np.logical_and(mask, bottom_mask).sum(), mask.sum())
        center_density = _safe_ratio(np.logical_and(mask, center_bottom_mask).sum(), mask.sum())
        top_density = _safe_ratio(np.logical_and(mask, top_mask).sum(), mask.sum())
        specular_density = float(((value > 0.82) & (saturation < 0.24) & mask).sum()) / float(mask.sum())
        texture_density = float(edge[mask].mean())
        score = (
            1.45 * bottom_density
            + 0.75 * center_density
            + 0.20 * texture_density
            + 0.10 * specular_density
            - 0.65 * top_density
            - 0.12 * abs(area - 0.45)
        )
        scores.append((score, label, area))
    if not scores:
        return center_bottom_mask.copy(), np.zeros_like(center_bottom_mask, dtype=bool)
    scores.sort(reverse=True)
    selected = [scores[0][1]]
    if len(scores) > 1 and scores[1][0] >= scores[0][0] - 0.20:
        selected.append(scores[1][1])
    raw_mask = _morph(np.isin(labels, selected))
    mask = _morph(np.logical_or(raw_mask, center_bottom_mask))
    return mask, raw_mask


def _material_mixture_map(
    rgb: np.ndarray,
    value: np.ndarray,
    saturation: np.ndarray,
    texture: np.ndarray,
    edge: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    rgb_std = np.mean([_local_std(rgb[:, :, channel], kernel=11) for channel in range(3)], axis=0)
    value_std = _local_std(value, kernel=11)
    sat_std = _local_std(saturation, kernel=11)
    boundary = _boundaries(labels).astype(np.float32)
    boundary_density = cv2.blur(boundary, (9, 9))
    optical = ((value > 0.82) & (saturation < 0.24)).astype(np.float32)
    snow_like = ((value > 0.72) & (saturation < 0.28)).astype(np.float32)
    raw = (
        0.22 * _norm(rgb_std)
        + 0.18 * _norm(value_std)
        + 0.15 * _norm(sat_std)
        + 0.18 * _norm(texture)
        + 0.12 * _norm(edge)
        + 0.10 * boundary_density
        + 0.03 * optical
        + 0.02 * snow_like
    )
    return np.clip(raw, 0.0, 1.0)


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _local_std(channel: np.ndarray, *, kernel: int) -> np.ndarray:
    mean = cv2.blur(channel.astype(np.float32), (kernel, kernel))
    mean_sq = cv2.blur((channel.astype(np.float32) ** 2), (kernel, kernel))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _kmeans_regions(rgb: np.ndarray, *, clusters: int) -> np.ndarray:
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xy = np.stack([xx / max(w - 1, 1), yy / max(h - 1, 1)], axis=2)
    lab = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    texture = _local_std(gray, kernel=9)[..., None]
    features = np.concatenate([lab, 0.35 * xy, 0.6 * texture], axis=2).reshape(-1, 6).astype(np.float32)
    k = max(2, min(int(clusters), features.shape[0]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.01)
    _compactness, labels, _centers = cv2.kmeans(features, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(h, w)


def _cluster_entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts.astype(np.float32) / float(counts.sum())
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(np.clip(p, 1e-8, 1.0))).sum() / math.log(len(p)))


def _external_mask_value(
    area: float,
    center_bottom_coverage: float,
    top_mass: float,
    roi_center_bottom_coverage: float,
) -> str:
    if area >= 0.84 and center_bottom_coverage >= 0.78:
        return "low_increment_road_dominant"
    if 0.25 <= area <= 0.84 and center_bottom_coverage >= 0.55 and top_mass <= 0.35:
        return "useful_candidate"
    if roi_center_bottom_coverage >= 0.90:
        return "roi_required_mask_unstable"
    return "unstable_or_low_confidence"


def _overlay(
    image_bgr: np.ndarray,
    road_mask: np.ndarray,
    raw_road_mask: np.ndarray,
    mixture_mask: np.ndarray,
    bottom_mask: np.ndarray,
) -> np.ndarray:
    out = image_bgr.copy()
    green = np.zeros_like(out)
    green[:, :] = (40, 210, 80)
    blue = np.zeros_like(out)
    blue[:, :] = (255, 120, 20)
    out[road_mask] = (0.55 * out[road_mask].astype(np.float32) + 0.45 * green[road_mask].astype(np.float32)).astype(np.uint8)
    out[mixture_mask] = (0.62 * out[mixture_mask].astype(np.float32) + 0.38 * blue[mixture_mask].astype(np.float32)).astype(np.uint8)
    raw_boundaries = _boundaries(raw_road_mask.astype(np.uint8))
    boundaries = np.logical_or(_boundaries(road_mask.astype(np.uint8)), _boundaries(mixture_mask.astype(np.uint8)))
    out[boundaries] = (0, 0, 255)
    out[raw_boundaries] = (255, 255, 255)
    bottom_line = np.where(np.diff(bottom_mask.astype(np.int8), axis=0, prepend=0) == 1)
    out[bottom_line] = (0, 255, 255)
    return out


def _boundaries(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    return boundary


def _morph(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((7, 7), np.uint8)
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return out.astype(bool)


def _norm(value: np.ndarray) -> np.ndarray:
    lo = float(np.quantile(value, 0.02))
    hi = float(np.quantile(value, 0.98))
    if hi <= lo + 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _safe_ratio(num: Any, den: Any) -> float:
    den_f = float(den)
    if den_f <= 0:
        return 0.0
    return float(num) / den_f


def _overlay_name(index: int, row: dict[str, Any]) -> str:
    bits = [
        f"{index:04d}",
        _safe(row.get("dataset")),
        _safe(row.get("class_label")),
        _safe(row.get("external_mask_value")),
    ]
    return "_".join(bits) + ".jpg"


def _safe(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    keep = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in text]
    return "".join(keep).strip("-")[:60] or "unknown"


def _build_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    dataset_rows: list[dict[str, Any]] = []
    if not frame.empty:
        for dataset, part in frame.groupby("dataset", dropna=False):
            value_counts = part["external_mask_value"].value_counts(normalize=True).to_dict()
            dataset_rows.append(
                {
                    "dataset": str(dataset),
                    "samples": int(len(part)),
                    "raw_pseudo_road_area_mean": _mean(part, "raw_pseudo_road_area"),
                    "raw_center_bottom_coverage_mean": _mean(part, "raw_pseudo_road_center_bottom_coverage"),
                    "raw_top_mass_mean": _mean(part, "raw_pseudo_road_top_mass"),
                    "pseudo_road_area_mean": _mean(part, "pseudo_road_area"),
                    "pseudo_road_area_p10": _quantile(part, "pseudo_road_area", 0.10),
                    "pseudo_road_area_p90": _quantile(part, "pseudo_road_area", 0.90),
                    "center_bottom_coverage_mean": _mean(part, "pseudo_road_center_bottom_coverage"),
                    "top_mass_mean": _mean(part, "pseudo_road_top_mass"),
                    "region_mixture_area_mean": _mean(part, "region_mixture_area"),
                    "region_mixture_on_road_mean": _mean(part, "region_mixture_on_road"),
                    "near_white_frac_mean": _mean(part, "near_white_frac"),
                    "specular_frac_mean": _mean(part, "specular_frac"),
                    "low_texture_frac_mean": _mean(part, "low_texture_frac"),
                    "cluster_entropy_mean": _mean(part, "cluster_entropy"),
                    "external_mask_value_distribution": {
                        str(key): float(value) for key, value in value_counts.items()
                    },
                    "route_recommendation": _dataset_recommendation(value_counts),
                }
            )
    verdict = _overall_verdict(dataset_rows)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": verdict,
        "claim_boundary": (
            "This is a pseudo-mask feasibility audit, not pixel-level ground truth. "
            "It estimates whether heavier SAM/Mask2Former road masks are likely to "
            "add value on the current public road-condition images."
        ),
        "manifests": [str(path) for path in args.manifests],
        "image_size": int(args.image_size),
        "clusters": int(args.clusters),
        "samples_per_dataset_requested": int(args.samples_per_dataset),
        "samples_total": len(rows),
        "dataset_rows": dataset_rows,
        "overlay_dir": str(args.out_dir / "overlays"),
        "next_actions": _next_actions(verdict, dataset_rows),
        "rows": rows,
    }


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame[column].empty:
        return None
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _quantile(frame: pd.DataFrame, column: str, q: float) -> float | None:
    if column not in frame or frame[column].empty:
        return None
    return float(pd.to_numeric(frame[column], errors="coerce").quantile(q))


def _dataset_recommendation(value_counts: dict[Any, float]) -> str:
    low = float(value_counts.get("low_increment_road_dominant", 0.0))
    useful = float(value_counts.get("useful_candidate", 0.0))
    unstable = float(value_counts.get("unstable_or_low_confidence", 0.0))
    roi_required = float(value_counts.get("roi_required_mask_unstable", 0.0))
    if useful >= 0.45 and unstable <= 0.25 and roi_required <= 0.35:
        return "external_mask_audit_worthwhile"
    if roi_required >= 0.45:
        return "roi_helps_but_external_mask_unstable"
    if low >= 0.55:
        return "road_patch_dominant_use_roi_or_region_mixture"
    if unstable >= 0.35:
        return "do_not_use_heavy_masks_without_manual_review"
    return "keep_lightweight_masks_only"


def _overall_verdict(dataset_rows: list[dict[str, Any]]) -> str:
    if not dataset_rows:
        return "no_samples"
    recommendations = [row.get("route_recommendation") for row in dataset_rows]
    if "external_mask_audit_worthwhile" in recommendations:
        return "small_external_mask_audit_worthwhile"
    if "roi_helps_but_external_mask_unstable" in recommendations:
        return "lightweight_roi_preferred_over_heavy_masks"
    if all(rec == "road_patch_dominant_use_roi_or_region_mixture" for rec in recommendations):
        return "heavy_segmentation_low_priority"
    if "do_not_use_heavy_masks_without_manual_review" in recommendations:
        return "manual_review_before_heavy_segmentation"
    return "lightweight_roi_and_region_mixture_preferred"


def _next_actions(verdict: str, dataset_rows: list[dict[str, Any]]) -> list[str]:
    actions = [
        "Use this audit as feasibility evidence only; do not claim pixel-level road masks.",
        "Keep bottom/center ROI and region-mixture cues in the formal candidate queue.",
    ]
    if verdict == "small_external_mask_audit_worthwhile":
        actions.append(
            "Run a 100-image SAM/Mask2Former or CLIPSeg pseudo-mask audit before any full-dataset preprocessing."
        )
    else:
        actions.append(
            "Do not run full-dataset SAM/Mask2Former preprocessing before current v14-v24 candidates finish."
        )
    if any(row.get("route_recommendation") == "do_not_use_heavy_masks_without_manual_review" for row in dataset_rows):
        actions.append("Manually inspect unstable overlays before adding pseudo-mask supervision.")
    return actions


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pseudo-Segmentation Mask Feasibility Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Samples | Raw road area | Raw center-bottom | ROI road area | ROI center-bottom | Raw top mass | Region-mixture area | Mixture on road | Near-white | External-mask distribution | Recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report.get("dataset_rows", []):
        lines.append(
            "| {dataset} | {samples} | {raw_area} | {raw_center} | {area} | {center} | {top} | {mix} | {mixroad} | {white} | {dist} | `{rec}` |".format(
                dataset=row.get("dataset", "-"),
                samples=row.get("samples", "-"),
                raw_area=_fmt(row.get("raw_pseudo_road_area_mean")),
                raw_center=_fmt(row.get("raw_center_bottom_coverage_mean")),
                area=_fmt(row.get("pseudo_road_area_mean")),
                center=_fmt(row.get("center_bottom_coverage_mean")),
                top=_fmt(row.get("raw_top_mass_mean")),
                mix=_fmt(row.get("region_mixture_area_mean")),
                mixroad=_fmt(row.get("region_mixture_on_road_mean")),
                white=_fmt(row.get("near_white_frac_mean")),
                dist=_fmt_dist(row.get("external_mask_value_distribution")),
                rec=row.get("route_recommendation", "-"),
            )
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(report.get("next_actions", []), start=1))
    lines.extend(["", "## Overlay Examples", ""])
    lines.extend(["| Dataset | Class | Decision | Road area | Mixture area | Overlay |", "|---|---|---|---:|---:|---|"])
    overlay_rows = [row for row in report.get("rows", []) if row.get("overlay_path")]
    for row in overlay_rows[:36]:
        lines.append(
            "| {dataset} | {cls} | `{decision}` | {area} | {mix} | {path} |".format(
                dataset=row.get("dataset", "-"),
                cls=row.get("class_label", "-"),
                decision=row.get("external_mask_value", "-"),
                area=_fmt(row.get("pseudo_road_area")),
                mix=_fmt(row.get("region_mixture_area")),
                path=row.get("overlay_path", "-"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_dist(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ", ".join(f"{key}:{float(val):.2f}" for key, val in sorted(value.items()))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_pseudo_segmentation_masks import (  # noqa: E402
    DEFAULT_MANIFESTS,
    _boundaries,
    _load_manifests,
    _material_mixture_map,
    _overlay_name,
    _priors,
    _pseudo_contact_mask,
    _sample_rows,
    _safe_ratio,
)


DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary/external_segmentation_masks")
DEFAULT_CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
CLIPSEG_PROMPTS = [
    "road surface",
    "asphalt road",
    "wet road surface",
    "snow covered road",
    "ice on road",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Small external segmentation audit for road/contact pseudo masks. "
            "Backends are optional: check/opencv always work; clipseg, segformer, "
            "and sam require their own dependencies and model weights."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["check", "opencv", "clipseg", "segformer", "sam"],
        default="check",
    )
    parser.add_argument("--manifests", type=Path, nargs="*", default=DEFAULT_MANIFESTS)
    parser.add_argument("--samples-per-dataset", type=int, default=30)
    parser.add_argument("--overlays-per-dataset", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--clipseg-model", type=str, default=DEFAULT_CLIPSEG_MODEL)
    parser.add_argument("--segformer-model", type=str, default=DEFAULT_SEGFORMER_MODEL)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_DIR / "external_segmentation_mask_audit.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_DIR / "external_segmentation_mask_audit.md")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    deps = _dependency_status()
    if args.backend == "check":
        report = _build_check_report(args, deps)
        _write(report, args)
        print(render_markdown(report))
        return

    blocker = _backend_blocker(args, deps)
    if blocker:
        report = _build_check_report(args, deps, blocker=blocker)
        _write(report, args)
        print(render_markdown(report))
        return

    selected = _sample_rows(
        _load_manifests(args.manifests),
        samples_per_dataset=int(args.samples_per_dataset),
        seed=int(args.seed),
    )
    segmenter = _build_segmenter(args)
    rows = _run_audit(selected, segmenter, args)
    report = _build_result_report(args, deps, rows)
    _write(report, args)
    print(render_markdown(report))


def _dependency_status() -> dict[str, bool]:
    modules = {
        "torch": "torch",
        "torchvision": "torchvision",
        "cv2": "cv2",
        "timm": "timm",
        "transformers": "transformers",
        "segment_anything": "segment_anything",
        "detectron2": "detectron2",
        "mmseg": "mmseg",
        "mmcv": "mmcv",
    }
    return {name: importlib.util.find_spec(module) is not None for name, module in modules.items()}


def _backend_blocker(args: argparse.Namespace, deps: dict[str, bool]) -> str | None:
    if args.backend in {"clipseg", "segformer"} and not deps.get("transformers", False):
        return "transformers_missing"
    if args.backend == "sam":
        if not deps.get("segment_anything", False):
            return "segment_anything_missing"
        if args.sam_checkpoint is None or not args.sam_checkpoint.exists():
            return "sam_checkpoint_missing"
    return None


def _build_segmenter(args: argparse.Namespace) -> Any:
    if args.backend == "opencv":
        return OpenCVSegmenter(clusters=int(args.clusters))
    if args.backend == "clipseg":
        return ClipSegSegmenter(
            model_name=str(args.clipseg_model),
            device=str(args.device),
            local_files_only=bool(args.local_files_only),
        )
    if args.backend == "segformer":
        return SegFormerSegmenter(
            model_name=str(args.segformer_model),
            device=str(args.device),
            local_files_only=bool(args.local_files_only),
        )
    if args.backend == "sam":
        return SamSegmenter(
            checkpoint=Path(args.sam_checkpoint),
            model_type=str(args.sam_model_type),
            device=str(args.device),
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


class OpenCVSegmenter:
    def __init__(self, *, clusters: int) -> None:
        self.clusters = clusters

    def __call__(self, image_bgr: np.ndarray, *, image_size: int) -> np.ndarray:
        resized = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb_f = rgb.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        value = hsv[:, :, 2] / 255.0
        saturation = hsv[:, :, 1] / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        bottom, center, top = _priors(image_size)
        labels = _kmeans_regions(rgb_f, clusters=self.clusters)
        _roi_mask, raw_mask = _pseudo_contact_mask(labels, value, saturation, edge, bottom, center, top)
        return raw_mask


class ClipSegSegmenter:
    def __init__(self, *, model_name: str, device: str, local_files_only: bool) -> None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = torch.device(device)
        self.processor = CLIPSegProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray, *, image_size: int) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((image_size, image_size))
        images = [pil] * len(CLIPSEG_PROMPTS)
        inputs = self.processor(text=CLIPSEG_PROMPTS, images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        mask = probs.max(axis=0)
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        return mask >= max(0.32, float(np.quantile(mask, 0.72)))


class SegFormerSegmenter:
    def __init__(self, *, model_name: str, device: str, local_files_only: bool) -> None:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        labels = getattr(self.model.config, "id2label", {}) or {}
        target_terms = ("road", "sidewalk", "runway", "path", "earth", "field", "snow")
        self.target_ids = [
            int(index)
            for index, name in labels.items()
            if any(term in str(name).lower() for term in target_terms)
        ]
        if not self.target_ids:
            self.target_ids = [int(index) for index, name in labels.items() if "road" in str(name).lower()]

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray, *, image_size: int) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        labels = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy()
        if not self.target_ids:
            return np.zeros((image_size, image_size), dtype=bool)
        return np.isin(labels, self.target_ids)


class SamSegmenter:
    def __init__(self, *, checkpoint: Path, model_type: str, device: str) -> None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=12,
            pred_iou_thresh=0.82,
            stability_score_thresh=0.88,
            crop_n_layers=0,
            min_mask_region_area=128,
        )

    def __call__(self, image_bgr: np.ndarray, *, image_size: int) -> np.ndarray:
        resized = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        masks = self.generator.generate(rgb)
        if not masks:
            return np.zeros((image_size, image_size), dtype=bool)
        _bottom, center, top = _priors(image_size)
        scored = []
        for item in masks:
            mask = item.get("segmentation", np.zeros((image_size, image_size), dtype=bool)).astype(bool)
            area = float(mask.mean())
            center_cover = _safe_ratio(np.logical_and(mask, center).sum(), center.sum())
            top_mass = _safe_ratio(np.logical_and(mask, top).sum(), mask.sum())
            score = center_cover - 0.45 * top_mass - 0.10 * abs(area - 0.45)
            scored.append((score, mask))
        scored.sort(key=lambda x: x[0], reverse=True)
        mask = scored[0][1]
        if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.12:
            mask = np.logical_or(mask, scored[1][1])
        return mask


def _run_audit(selected: pd.DataFrame, segmenter: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    overlay_dir = args.out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for index, item in selected.reset_index(drop=True).iterrows():
        path = Path(str(item["image_path"]))
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            rows.append(
                {
                    "decode_ok": False,
                    "image_path": str(path),
                    "dataset": item.get("dataset"),
                    "class_label": item.get("class_label"),
                }
            )
            continue
        mask = segmenter(image_bgr, image_size=int(args.image_size))
        row, overlay = _score_mask(item, image_bgr, mask, args)
        dataset = str(row.get("dataset", "unknown"))
        if overlay_counts.get(dataset, 0) < int(args.overlays_per_dataset):
            name_row = {**row, "external_mask_value": row.get("decision")}
            overlay_path = overlay_dir / _overlay_name(int(index), name_row)
            cv2.imwrite(str(overlay_path), overlay)
            row["overlay_path"] = str(overlay_path)
            overlay_counts[dataset] = overlay_counts.get(dataset, 0) + 1
        else:
            row["overlay_path"] = None
        rows.append(row)
    return rows


def _score_mask(
    item: pd.Series,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    image_size = int(args.image_size)
    resized = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    texture = _local_std(gray, kernel=11)
    edge = _edge_magnitude(gray)
    labels = _kmeans_regions(rgb_f, clusters=int(args.clusters))
    mixture_map = _material_mixture_map(rgb_f, value, saturation, texture, edge, labels)
    mixture_mask = mixture_map >= float(np.quantile(mixture_map, 0.75))
    bottom, center, top = _priors(image_size)
    mask = cv2.resize(mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)
    near_white = (value > 0.88) & (saturation < 0.18)
    area = float(mask.mean())
    center_cover = _safe_ratio(np.logical_and(mask, center).sum(), center.sum())
    bottom_cover = _safe_ratio(np.logical_and(mask, bottom).sum(), bottom.sum())
    top_mass = _safe_ratio(np.logical_and(mask, top).sum(), mask.sum())
    mixture_on_mask = _safe_ratio(np.logical_and(mask, mixture_mask).sum(), mixture_mask.sum())
    decision = _decision(area, center_cover, top_mass)
    overlay = _overlay(resized, mask, mixture_mask, bottom)
    return (
        {
            "decode_ok": True,
            "image_path": str(item["image_path"]),
            "dataset": item.get("dataset"),
            "class_label": item.get("class_label"),
            "friction_label": item.get("friction_label"),
            "wetness_label": item.get("wetness_label"),
            "snow_label": item.get("snow_label"),
            "risk_label": item.get("risk_label"),
            "backend": args.backend,
            "mask_area": area,
            "center_bottom_coverage": float(center_cover),
            "bottom_coverage": float(bottom_cover),
            "top_mass": float(top_mass),
            "region_mixture_area": float(mixture_mask.mean()),
            "region_mixture_on_mask": float(mixture_on_mask),
            "near_white_frac": float(near_white.mean()),
            "decision": decision,
        },
        overlay,
    )


def _decision(area: float, center_cover: float, top_mass: float) -> str:
    if area >= 0.84 and center_cover >= 0.78:
        return "road_patch_dominant_low_increment"
    if 0.20 <= area <= 0.84 and center_cover >= 0.55 and top_mass <= 0.35:
        return "mask_candidate_useful"
    if center_cover >= 0.35 and top_mass <= 0.55:
        return "mask_candidate_needs_roi_fusion"
    return "mask_unstable"


def _overlay(image_bgr: np.ndarray, mask: np.ndarray, mixture_mask: np.ndarray, bottom_mask: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    green = np.zeros_like(out)
    green[:, :] = (40, 210, 80)
    blue = np.zeros_like(out)
    blue[:, :] = (255, 120, 20)
    out[mask] = (0.55 * out[mask].astype(np.float32) + 0.45 * green[mask].astype(np.float32)).astype(np.uint8)
    out[mixture_mask] = (0.62 * out[mixture_mask].astype(np.float32) + 0.38 * blue[mixture_mask].astype(np.float32)).astype(np.uint8)
    out[_boundaries(mask.astype(np.uint8))] = (255, 255, 255)
    out[_boundaries(mixture_mask.astype(np.uint8))] = (0, 0, 255)
    bottom_line = np.where(np.diff(bottom_mask.astype(np.int8), axis=0, prepend=0) == 1)
    out[bottom_line] = (0, 255, 255)
    return out


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


def _local_std(channel: np.ndarray, *, kernel: int) -> np.ndarray:
    mean = cv2.blur(channel.astype(np.float32), (kernel, kernel))
    mean_sq = cv2.blur((channel.astype(np.float32) ** 2), (kernel, kernel))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _build_check_report(
    args: argparse.Namespace,
    deps: dict[str, bool],
    *,
    blocker: str | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "verdict": "blocked_missing_dependency" if blocker else "dependency_check_only",
        "blocker": blocker,
        "claim_boundary": (
            "This report only audits whether external segmentation backends are available. "
            "No pixel-level road-mask claim is made until a backend produces sample masks "
            "and overlays that pass manual/reviewer checks."
        ),
        "dependency_status": deps,
        "recommended_install_options": _install_options(deps),
        "next_actions": _check_next_actions(args.backend, blocker, deps),
        "rows": [],
        "dataset_rows": [],
    }


def _build_result_report(
    args: argparse.Namespace,
    deps: dict[str, bool],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    dataset_rows = []
    if not frame.empty and "dataset" in frame:
        for dataset, part in frame.groupby("dataset", dropna=False):
            decisions = part["decision"].value_counts(normalize=True).to_dict()
            dataset_rows.append(
                {
                    "dataset": str(dataset),
                    "samples": int(len(part)),
                    "mask_area_mean": _mean(part, "mask_area"),
                    "center_bottom_coverage_mean": _mean(part, "center_bottom_coverage"),
                    "top_mass_mean": _mean(part, "top_mass"),
                    "region_mixture_on_mask_mean": _mean(part, "region_mixture_on_mask"),
                    "near_white_frac_mean": _mean(part, "near_white_frac"),
                    "decision_distribution": {str(k): float(v) for k, v in decisions.items()},
                }
            )
    verdict = _result_verdict(dataset_rows)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "verdict": verdict,
        "claim_boundary": (
            "This is a sample-level external segmentation audit. It does not create "
            "ground-truth road masks; it only decides whether a pseudo-mask source is "
            "stable enough to consider for EvidenceField supervision."
        ),
        "dependency_status": deps,
        "samples_total": len(rows),
        "dataset_rows": dataset_rows,
        "overlay_dir": str(args.out_dir / "overlays"),
        "rows": rows,
        "next_actions": _result_next_actions(verdict),
    }


def _result_verdict(dataset_rows: list[dict[str, Any]]) -> str:
    if not dataset_rows:
        return "no_samples"
    useful_rates = []
    unstable_rates = []
    for row in dataset_rows:
        dist = row.get("decision_distribution") or {}
        useful_rates.append(float(dist.get("mask_candidate_useful", 0.0)))
        unstable_rates.append(float(dist.get("mask_unstable", 0.0)))
    if min(useful_rates or [0.0]) >= 0.45 and max(unstable_rates or [1.0]) <= 0.25:
        return "external_masks_promising_for_small_ablation"
    if max(unstable_rates or [0.0]) >= 0.35:
        return "external_masks_unstable_do_not_full_preprocess"
    return "external_masks_maybe_use_with_roi_fusion"


def _result_next_actions(verdict: str) -> list[str]:
    if verdict == "external_masks_promising_for_small_ablation":
        return [
            "Run a small ablation that distills these masks into EvidenceField attention.",
            "Keep raw overlays and dataset-slice metrics as reviewer evidence.",
            "Do not claim pixel-level ground truth.",
        ]
    if verdict == "external_masks_unstable_do_not_full_preprocess":
        return [
            "Do not preprocess the full dataset with this backend.",
            "Prefer lightweight bottom ROI and region-mixture cues.",
            "Inspect failure overlays before trying another backend.",
        ]
    return [
        "Use masks only fused with bottom/center ROI constraints.",
        "Compare against v23 region-mixture cues before adding training complexity.",
    ]


def _check_next_actions(backend: str, blocker: str | None, deps: dict[str, bool]) -> list[str]:
    if blocker == "transformers_missing":
        return [
            "Install transformers only after the active formal GPU queue is idle.",
            "Prefer CLIPSeg first because it can use road/wet/snow text prompts.",
            "Run no more than 100 images before deciding whether to full-preprocess.",
        ]
    if blocker == "segment_anything_missing":
        return [
            "Install segment-anything and download an explicit SAM checkpoint only for a small audit.",
            "Use CPU or an idle GPU guard; do not run SAM during formal training.",
        ]
    if blocker == "sam_checkpoint_missing":
        return [
            "Provide a local SAM checkpoint path before running the SAM backend.",
            "Record checkpoint URL/hash in the report before using masks as pseudo labels.",
        ]
    if backend == "check":
        return [
            "Current environment can run OpenCV fallback immediately.",
            "CLIPSeg/SegFormer require transformers; SAM requires segment-anything plus a checkpoint.",
            "Keep external mask audit separate from formal training until dependency and overlay evidence pass.",
        ]
    return [
        "Run the selected backend only on a small sample first.",
        "Promote only if masks are useful across RSCD, RoadSaW, and RoadSC.",
    ]


def _install_options(deps: dict[str, bool]) -> list[str]:
    options = []
    if not deps.get("transformers", False):
        options.append("pip install transformers accelerate safetensors")
    if not deps.get("segment_anything", False):
        options.append("pip install git+https://github.com/facebookresearch/segment-anything.git")
    return options


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame[column].empty:
        return None
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External Segmentation Mask Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Backend: `{report.get('backend', '-')}`",
        f"Verdict: `{report.get('verdict', '-')}`",
        "",
        f"Claim boundary: {report.get('claim_boundary', '-')}",
        "",
        "## Dependencies",
        "",
        "| Dependency | Status |",
        "|---|---|",
    ]
    for name, ok in (report.get("dependency_status") or {}).items():
        lines.append(f"| {name} | {'ok' if ok else 'missing'} |")
    install = report.get("recommended_install_options") or []
    if install:
        lines.extend(["", "Recommended install options after GPU queue is idle:"])
        lines.extend(f"- `{cmd}`" for cmd in install)
    rows = report.get("dataset_rows") or []
    if rows:
        lines.extend(["", "## Dataset Summary", ""])
        lines.extend(
            [
                "| Dataset | Samples | Mask area | Center-bottom coverage | Top mass | Mixture on mask | Near-white | Decisions |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                "| {dataset} | {samples} | {area} | {center} | {top} | {mix} | {white} | {decisions} |".format(
                    dataset=row.get("dataset", "-"),
                    samples=row.get("samples", "-"),
                    area=_fmt(row.get("mask_area_mean")),
                    center=_fmt(row.get("center_bottom_coverage_mean")),
                    top=_fmt(row.get("top_mass_mean")),
                    mix=_fmt(row.get("region_mixture_on_mask_mean")),
                    white=_fmt(row.get("near_white_frac_mean")),
                    decisions=_fmt_dist(row.get("decision_distribution")),
                )
            )
    overlay_rows = [row for row in report.get("rows", []) if row.get("overlay_path")]
    if overlay_rows:
        lines.extend(["", "## Overlay Examples", ""])
        lines.extend(["| Dataset | Class | Decision | Overlay |", "|---|---|---|---|"])
        for row in overlay_rows[:36]:
            lines.append(
                "| {dataset} | {cls} | `{decision}` | {path} |".format(
                    dataset=row.get("dataset", "-"),
                    cls=row.get("class_label", "-"),
                    decision=row.get("decision", "-"),
                    path=row.get("overlay_path", "-"),
                )
            )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(report.get("next_actions", []), start=1))
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


def _write(report: dict[str, Any], args: argparse.Namespace) -> None:
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()

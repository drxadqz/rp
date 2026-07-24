from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_external_segmentation_masks import (  # noqa: E402
    DEFAULT_CLIPSEG_MODEL,
    DEFAULT_SEGFORMER_MODEL,
    OpenCVSegmenter,
    ClipSegSegmenter,
    SegFormerSegmenter,
    SamSegmenter,
    _dependency_status,
    _backend_blocker,
    _build_result_report,
    _score_mask,
    render_markdown,
)
from friction_affordance.transforms import BottomSquareCropResize, LetterboxResize  # noqa: E402


DEFAULT_MANIFESTS = [
    Path("data/manifests_full/roadsaw_train.csv"),
    Path("data/manifests_full/roadsaw_val.csv"),
    Path("data/manifests_full/roadsaw_test.csv"),
    Path("data/manifests_full/roadsc_train.csv"),
    Path("data/manifests_full/roadsc_val.csv"),
    Path("data/manifests_full/roadsc_test.csv"),
    Path("data/manifests_full/rscd_prepared_train.csv"),
    Path("data/manifests_full/rscd_prepared_val.csv"),
    Path("data/manifests_full/rscd_prepared_test.csv"),
]
DEFAULT_OUT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\road_mask_cache")
DEFAULT_REPORT_DIR = Path("reports/paper_protocol_summary/external_road_mask_cache")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible offline road/contact pseudo-mask caches and "
            "manifest copies with a road_mask_path column. Run a small sample "
            "first; promote to full preprocessing only after the audit report "
            "and overlays are acceptable."
        )
    )
    parser.add_argument("--backend", choices=["opencv", "clipseg", "segformer", "sam"], default="opencv")
    parser.add_argument("--manifests", type=Path, nargs="*", default=DEFAULT_MANIFESTS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-manifest-dir", type=Path, default=DEFAULT_OUT_ROOT / "manifests")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--samples-per-manifest", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--resize-mode", choices=["bottom_square", "letterbox", "stretch"], default="bottom_square")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--clipseg-model", type=str, default=DEFAULT_CLIPSEG_MODEL)
    parser.add_argument("--segformer-model", type=str, default=DEFAULT_SEGFORMER_MODEL)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.report_dir

    deps = _dependency_status()
    blocker = _backend_blocker(args, deps)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.out_manifest_dir.mkdir(parents=True, exist_ok=True)
    if blocker:
        report = _blocked_report(args, deps, blocker)
        _write_report(report, args)
        print(render_markdown(report))
        return

    segmenter = _build_segmenter(args)
    image_transform = _canonical_image_transform(args)
    rows_for_audit: list[dict[str, Any]] = []
    manifest_reports: list[dict[str, Any]] = []

    for manifest in args.manifests:
        if not manifest.exists():
            manifest_reports.append({"manifest": str(manifest), "status": "missing", "rows": 0})
            continue
        frame = pd.read_csv(manifest, dtype=str, low_memory=False)
        if args.samples_per_manifest > 0 and len(frame) > args.samples_per_manifest:
            frame = frame.sample(n=args.samples_per_manifest, random_state=int(args.seed)).reset_index(drop=True)
        out_frame, stats, scored_rows = _process_manifest(frame, manifest, segmenter, image_transform, args)
        out_manifest = args.out_manifest_dir / f"{manifest.stem}__{args.backend}_{args.resize_mode}_{args.image_size}.csv"
        out_frame.to_csv(out_manifest, index=False)
        stats["output_manifest"] = str(out_manifest)
        manifest_reports.append(stats)
        rows_for_audit.extend(scored_rows)

    valid_audit_rows = [row for row in rows_for_audit if row.get("decode_ok") and row.get("decision")]
    report = _build_result_report(args, deps, valid_audit_rows)
    report["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report["cache_root"] = str(args.out_root)
    report["manifest_reports"] = manifest_reports
    report["decode_failures_in_cache"] = sum(int(row.get("failed", 0)) for row in manifest_reports)
    report["claim_boundary"] = (
        "Cached masks are pseudo-labels generated from public images by an "
        "explicit backend and deterministic image canonicalization. They are "
        "not ground-truth road masks and must be ablated against the no-mask "
        "EvidenceField and lightweight ROI/region-mixture variants."
    )
    report["training_contract"] = {
        "manifest_column": "road_mask_path",
        "data_config_required": {
            "load_road_masks": True,
            "road_mask_pretransformed": True,
            "augmentation.horizontal_flip_p": 0.0,
            "augmentation.random_resized_crop": False,
            "augmentation.resize_mode": args.resize_mode,
        },
    }
    _write_report(report, args)
    print(render_markdown(report))


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


def _canonical_image_transform(args: argparse.Namespace):
    image_size = int(args.image_size)
    if args.resize_mode == "bottom_square":
        return BottomSquareCropResize(image_size)
    if args.resize_mode == "letterbox":
        return LetterboxResize(image_size)
    return lambda image: image.resize((image_size, image_size))


def _process_manifest(
    frame: pd.DataFrame,
    manifest: Path,
    segmenter: Any,
    image_transform,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    out_frame = frame.copy()
    mask_paths: list[str] = []
    scored_rows: list[dict[str, Any]] = []
    decoded = 0
    reused = 0
    written = 0
    failed = 0

    for idx, item in out_frame.iterrows():
        source = Path(str(item.get("image_path", "")))
        mask_path = _mask_path_for(source, manifest, item, args)
        mask_paths.append(str(mask_path))
        if mask_path.exists() and not args.overwrite:
            reused += 1
            continue
        try:
            canonical_bgr = _load_canonical_bgr(source, image_transform)
        except (OSError, ValueError) as exc:
            failed += 1
            scored_rows.append(
                {
                    "decode_ok": False,
                    "image_path": str(source),
                    "dataset": item.get("dataset"),
                    "class_label": item.get("class_label"),
                    "backend": args.backend,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        decoded += 1
        mask = segmenter(canonical_bgr, image_size=int(args.image_size))
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
        written += 1
        if len(scored_rows) < 360:
            score_item = item.copy()
            score_item["image_path"] = str(source)
            row, _overlay = _score_mask(score_item, canonical_bgr, mask, args)
            row["road_mask_path"] = str(mask_path)
            scored_rows.append(row)

    out_frame["road_mask_path"] = mask_paths
    return (
        out_frame,
        {
            "manifest": str(manifest),
            "status": "ok",
            "rows": int(len(out_frame)),
            "decoded": int(decoded),
            "reused": int(reused),
            "written": int(written),
            "failed": int(failed),
        },
        scored_rows,
    )


def _load_canonical_bgr(path: Path, image_transform) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image_transform(image)
        rgb = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _mask_path_for(source: Path, manifest: Path, item: pd.Series, args: argparse.Namespace) -> Path:
    dataset = str(item.get("dataset") or "unknown").lower().replace("\\", "_").replace("/", "_")
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:20]
    return (
        args.out_root
        / "masks"
        / args.backend
        / f"{args.resize_mode}_{int(args.image_size)}"
        / manifest.stem
        / dataset
        / f"{digest}.png"
    )


def _blocked_report(args: argparse.Namespace, deps: dict[str, bool], blocker: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "verdict": "blocked_missing_dependency",
        "blocker": blocker,
        "dependency_status": deps,
        "samples_total": 0,
        "dataset_rows": [],
        "rows": [],
        "manifest_reports": [],
        "claim_boundary": "No masks were generated because the requested backend is unavailable.",
        "next_actions": [
            "Install the missing backend only after the formal GPU queue is idle.",
            "Rerun this script with a small samples-per-manifest value before full preprocessing.",
        ],
    }


def _write_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / f"road_mask_cache_{args.backend}_{args.resize_mode}_{int(args.image_size)}.json"
    md_path = args.report_dir / f"road_mask_cache_{args.backend}_{args.resize_mode}_{int(args.image_size)}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()

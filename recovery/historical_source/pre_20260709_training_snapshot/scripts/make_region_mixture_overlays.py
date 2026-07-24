from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from evaluate_region_mixture_conformal import _kmeans_regions


DEFAULT_PREDICTIONS = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol"
    r"\single_roadsaw_full_faf\region_mixture_full_cpu\predictions_test_with_region_mixture.csv"
)
DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary/region_mixture_overlays")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual overlays for region-mixture audit examples.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_DIR / "region_mixture_overlay_audit.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_DIR / "region_mixture_overlay_audit.md")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions, low_memory=False)
    selected = _select_examples(df, samples=int(args.samples), seed=int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in selected.reset_index(drop=True).iterrows():
        path = Path(str(row["image_path"]))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        overlay, label_viz = _overlay_image(image, clusters=int(args.clusters), image_size=int(args.image_size))
        stem = f"{index:03d}_{_safe(row.get('dataset'))}_{_safe(row.get('true_friction'))}_{_safe(row.get('region_mixture_bin'))}"
        overlay_path = args.out_dir / f"{stem}_overlay.jpg"
        labels_path = args.out_dir / f"{stem}_regions.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(labels_path), label_viz)
        rows.append(
            {
                "image_path": str(path),
                "overlay_path": str(overlay_path),
                "regions_path": str(labels_path),
                "dataset": row.get("dataset"),
                "true_friction": row.get("true_friction"),
                "pred_friction": row.get("pred_friction"),
                "true_risk": row.get("true_risk"),
                "pred_risk": row.get("pred_risk"),
                "region_mixture_bin": row.get("region_mixture_bin"),
                "region_mixture_score": _num(row.get("region_mixture_score")),
                "raw_interval_covers": bool(row.get("raw_interval_covers")),
                "state_region_radius": _num(row.get("state_region_mixture_radius")),
            }
        )
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": str(args.predictions),
        "out_dir": str(args.out_dir),
        "samples_requested": int(args.samples),
        "samples_written": len(rows),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def _select_examples(df: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    parts = []
    for group_col in ["true_friction", "region_mixture_bin"]:
        if group_col not in df.columns:
            continue
        per_group = max(1, samples // max(int(df[group_col].nunique()), 1))
        for _, part in df.groupby(group_col, dropna=False):
            parts.append(part.sort_values("region_mixture_score", ascending=False).head(per_group))
    hard = df.sort_values(["region_mixture_score"], ascending=False).head(max(1, samples // 3))
    random = df.sample(n=min(len(df), max(1, samples // 3)), random_state=seed)
    selected = pd.concat([*parts, hard, random], ignore_index=True).drop_duplicates("image_path")
    return selected.head(samples)


def _overlay_image(image_bgr: np.ndarray, *, clusters: int, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    labels = _kmeans_regions(rgb, clusters=clusters)
    colors = _label_colors(labels)
    label_viz = colors.copy()
    boundaries = _boundaries(labels)
    overlay = (0.58 * image_bgr.astype(np.float32) + 0.42 * label_viz.astype(np.float32)).astype(np.uint8)
    overlay[boundaries] = (0, 0, 255)
    return overlay, label_viz


def _label_colors(labels: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
        ],
        dtype=np.uint8,
    )
    out = palette[np.mod(labels, len(palette))]
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _boundaries(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    return boundary


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Region Mixture Overlay Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Samples written: `{report['samples_written']}`",
        f"Output dir: `{report['out_dir']}`",
        "",
        "| idx | true | pred | bin | score | overlay |",
        "|---:|---|---|---|---:|---|",
    ]
    for idx, row in enumerate(report.get("rows", [])[:30]):
        lines.append(
            "| {idx} | {true} | {pred} | {bin} | {score} | {overlay} |".format(
                idx=idx,
                true=row.get("true_friction"),
                pred=row.get("pred_friction"),
                bin=row.get("region_mixture_bin"),
                score=_fmt(row.get("region_mixture_score")),
                overlay=row.get("overlay_path"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _safe(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    keep = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in text]
    return "".join(keep).strip("-")[:40] or "unknown"


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "-"
    return f"{number:.4f}"


if __name__ == "__main__":
    main()

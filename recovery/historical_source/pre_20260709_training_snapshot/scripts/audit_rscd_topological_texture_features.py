from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy import ndimage


ImageFile.LOAD_TRUNCATED_IMAGES = True

MANIFEST = Path("data/manifests_full/rscd_prepared_train.csv")
OUT_JSON = Path("reports/paper_protocol_summary/rscd_topological_texture_audit.json")
OUT_MD = Path("reports/paper_protocol_summary/rscd_topological_texture_audit.md")
SAMPLES_PER_CLASS = 45
IMAGE_SIZE = 96
THRESHOLDS = np.linspace(0.15, 0.85, 8, dtype=np.float32)


def main() -> None:
    df = pd.read_csv(MANIFEST, dtype=str, low_memory=False)
    df = df[df["class_label"].notna() & df["image_path"].notna()].copy()
    sampled_parts = []
    for label, group in df.groupby("class_label"):
        part = group.sample(n=min(SAMPLES_PER_CLASS, len(group)), random_state=20260626).copy()
        part["class_label"] = label
        sampled_parts.append(part)
    sampled = pd.concat(sampled_parts, ignore_index=True)
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, float]] = []
    for _, item in sampled.iterrows():
        label = canonical_label(str(item["class_label"]))
        path = Path(str(item["image_path"]))
        try:
            image = load_image(path)
        except OSError:
            continue
        feats = extract_features(image)
        rows.append(
            {
                "class_label": label,
                "friction": friction_state(label),
                "material": material_state(label),
                "image_path": str(path),
            }
        )
        feature_rows.append(feats)

    meta = pd.DataFrame(rows)
    feats = pd.DataFrame(feature_rows)
    joined = pd.concat([meta, feats], axis=1)
    feature_names = list(feats.columns)
    class_scores = fisher_table(joined, feature_names, "class_label")
    friction_scores = fisher_table(joined, feature_names, "friction")
    material_scores = fisher_table(joined[joined["material"].notna()].copy(), feature_names, "material")

    result = {
        "claim_boundary": (
            "This is a feature diagnostic for RSCD patch images. It tests whether simple "
            "topological summaries of binary filtrations contain label-separating signal; "
            "it is not model-performance evidence."
        ),
        "sample": {
            "manifest": str(MANIFEST),
            "rows": int(len(joined)),
            "samples_per_class": SAMPLES_PER_CLASS,
            "image_size": IMAGE_SIZE,
            "thresholds": [float(x) for x in THRESHOLDS],
            "classes": int(joined["class_label"].nunique()),
        },
        "top_class_features": class_scores[:20],
        "top_friction_features": friction_scores[:20],
        "top_material_features": material_scores[:20],
        "group_means": group_means(joined),
        "interpretation": interpretation(class_scores, friction_scores, material_scores),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(to_markdown(result), encoding="utf-8")
    print(OUT_MD)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def extract_features(rgb: np.ndarray) -> dict[str, float]:
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = (maxc - minc) / np.maximum(maxc, 1e-4)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    snow = sigmoid((maxc - 0.72) * 12.0) * sigmoid((0.28 - saturation) * 12.0)
    specular = sigmoid((maxc - 0.82) * 14.0) * sigmoid((0.24 - saturation) * 12.0)
    dark_water = sigmoid((0.38 - maxc) * 10.0) * sigmoid((0.12 - grad) * 30.0)
    wet = np.clip(specular + 0.5 * dark_water, 0.0, 1.0)
    low_texture = sigmoid((0.045 - grad) * 35.0)

    fields = {
        "gray_high": gray,
        "grad_high": normalize01(grad),
        "snow_high": snow,
        "wet_high": wet,
        "low_texture_high": low_texture,
    }
    out: dict[str, float] = {}
    for name, field in fields.items():
        ecs = []
        comps = []
        holes = []
        for threshold in THRESHOLDS:
            mask = field >= float(threshold)
            comp, hole = component_hole_count(mask)
            ec = comp - hole
            comps.append(comp)
            holes.append(hole)
            ecs.append(ec)
        arr_ec = np.asarray(ecs, dtype=np.float32)
        arr_comp = np.asarray(comps, dtype=np.float32)
        arr_hole = np.asarray(holes, dtype=np.float32)
        out[f"{name}_ec_mean"] = float(arr_ec.mean())
        out[f"{name}_ec_std"] = float(arr_ec.std())
        out[f"{name}_ec_range"] = float(arr_ec.max() - arr_ec.min())
        out[f"{name}_components_mean"] = float(arr_comp.mean())
        out[f"{name}_components_max"] = float(arr_comp.max())
        out[f"{name}_holes_mean"] = float(arr_hole.mean())
        out[f"{name}_holes_max"] = float(arr_hole.max())
        out[f"{name}_holes_auc"] = float(arr_hole.sum() / len(arr_hole))
    return out


def component_hole_count(mask: np.ndarray) -> tuple[int, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    _, num_components = ndimage.label(mask, structure=structure)
    inv = ~mask
    labeled_bg, num_bg = ndimage.label(inv, structure=structure)
    if num_bg <= 0:
        return int(num_components), 0
    border_labels = set(np.unique(labeled_bg[0, :]).tolist())
    border_labels.update(np.unique(labeled_bg[-1, :]).tolist())
    border_labels.update(np.unique(labeled_bg[:, 0]).tolist())
    border_labels.update(np.unique(labeled_bg[:, -1]).tolist())
    border_labels.discard(0)
    hole_labels = set(range(1, int(num_bg) + 1)) - border_labels
    return int(num_components), int(len(hole_labels))


def fisher_table(df: pd.DataFrame, feature_names: list[str], target: str) -> list[dict[str, Any]]:
    labels = sorted(str(x) for x in df[target].dropna().unique())
    rows = []
    for feature in feature_names:
        global_mean = float(df[feature].mean())
        between = 0.0
        within = 0.0
        for label in labels:
            part = df[df[target].astype(str) == label][feature].astype(float)
            if part.empty:
                continue
            between += float(len(part)) * (float(part.mean()) - global_mean) ** 2
            within += float(((part - float(part.mean())) ** 2).sum())
        score = between / max(within, 1e-8)
        rows.append({"feature": feature, "target": target, "fisher_ratio": score})
    rows.sort(key=lambda x: float(x["fisher_ratio"]), reverse=True)
    return rows


def group_means(df: pd.DataFrame) -> dict[str, Any]:
    chosen = [
        "wet_high_holes_mean",
        "wet_high_components_mean",
        "snow_high_components_mean",
        "low_texture_high_ec_range",
        "grad_high_components_mean",
    ]
    out: dict[str, Any] = {}
    for group in ["friction", "material"]:
        values = {}
        for label, part in df.groupby(group):
            if label is None or (isinstance(label, float) and math.isnan(label)):
                continue
            values[str(label)] = {feature: float(part[feature].mean()) for feature in chosen if feature in part}
        out[group] = values
    return out


def interpretation(
    class_scores: list[dict[str, Any]],
    friction_scores: list[dict[str, Any]],
    material_scores: list[dict[str, Any]],
) -> list[str]:
    top_names = {row["feature"] for row in class_scores[:10] + friction_scores[:10] + material_scores[:10]}
    messages = []
    if any("holes" in name for name in top_names):
        messages.append(
            "Hole-count features appear among the strongest separators, suggesting that puddles, snow granularity, "
            "and rough aggregate can be represented as topology of thresholded texture evidence."
        )
    if any("components" in name for name in top_names):
        messages.append(
            "Connected-component counts are label-separating, which supports a topology-aware version of PhysicsTexture: "
            "not just how much wet/snow evidence exists, but how fragmented or connected it is."
        )
    if not messages:
        messages.append(
            "The topological curves are weaker than expected in this sample; keep them as analysis unless a fast model ablation improves metrics."
        )
    messages.append(
        "A feasible next model is a lightweight TopologicalTexture branch using Euler/component/hole curves on wet, snow, low-texture, and gradient maps."
    )
    return messages


def canonical_label(label: str) -> str:
    return label.strip().lower().replace("-", "_")


def friction_state(label: str) -> str:
    label = canonical_label(label)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return label
    return label.split("_")[0] if label else "unknown"


def material_state(label: str) -> str | None:
    label = canonical_label(label)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return None
    parts = label.split("_")
    return parts[1] if len(parts) >= 2 else None


def normalize01(x: np.ndarray) -> np.ndarray:
    return (x - float(x.min())) / max(float(x.max() - x.min()), 1e-6)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# RSCD Topological Texture Audit",
        "",
        result["claim_boundary"],
        "",
        "## Sample",
        "",
        f"- Rows: {result['sample']['rows']}",
        f"- Classes: {result['sample']['classes']}",
        f"- Image size: {result['sample']['image_size']}",
        "",
        "## Top Class-Separating Features",
        "",
        "| rank | feature | Fisher ratio |",
        "|---:|---|---:|",
    ]
    for idx, row in enumerate(result["top_class_features"][:12], start=1):
        lines.append(f"| {idx} | `{row['feature']}` | {float(row['fisher_ratio']):.4f} |")
    lines.extend(["", "## Top Friction-Separating Features", "", "| rank | feature | Fisher ratio |", "|---:|---|---:|"])
    for idx, row in enumerate(result["top_friction_features"][:12], start=1):
        lines.append(f"| {idx} | `{row['feature']}` | {float(row['fisher_ratio']):.4f} |")
    lines.extend(["", "## Top Material-Separating Features", "", "| rank | feature | Fisher ratio |", "|---:|---|---:|"])
    for idx, row in enumerate(result["top_material_features"][:12], start=1):
        lines.append(f"| {idx} | `{row['feature']}` | {float(row['fisher_ratio']):.4f} |")
    lines.extend(["", "## Interpretation", ""])
    for item in result["interpretation"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

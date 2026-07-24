from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from friction_affordance.models.coupled_factor_backbone import FixedRoadEvidenceMaps  # noqa: E402


EVIDENCE_NAMES = [
    "gray",
    "saturation",
    "gradient",
    "laplacian",
    "local_contrast",
    "dark_film",
    "specular",
    "wet_proxy",
    "rough_proxy",
    "concrete_proxy",
    "asphalt_proxy",
    "visible_texture_under_film",
]

DEFAULT_FOCUS_CLASSES = [
    "water_concrete_slight",
    "water_concrete_severe",
    "water_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "wet_concrete_smooth",
    "dry_concrete_slight",
    "dry_concrete_severe",
    "dry_concrete_smooth",
    "water_asphalt_slight",
    "water_asphalt_severe",
    "water_asphalt_smooth",
    "wet_asphalt_slight",
    "wet_asphalt_severe",
    "wet_asphalt_smooth",
]

DEFAULT_PAIRS = [
    "water_concrete_severe::water_concrete_slight",
    "wet_concrete_severe::wet_concrete_slight",
    "dry_concrete_severe::dry_concrete_slight",
    "water_concrete_smooth::wet_concrete_smooth",
    "water_asphalt_severe::water_asphalt_slight",
]


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image_path": str(row.get("image_path", "")),
                    "true_label": str(row.get("true_label", "")),
                    "pred_label": str(row.get("pred_label", "")),
                    "confidence": float(row.get("confidence") or 0.0),
                }
            )
    return rows


def _load_image(path: str, image_size: int) -> torch.Tensor | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor
    except Exception:
        return None


def _feature_family(feature: str) -> str:
    base = feature.split("_", 1)[0]
    if "local_contrast" in feature or "visible_texture" in feature:
        return "contrast_visibility"
    if "dark_film" in feature:
        return "dark_film_quantile"
    if "wet_proxy" in feature or "specular" in feature:
        return "film_reflectance"
    if "rough_proxy" in feature or "gradient" in feature or "laplacian" in feature:
        return "roughness_texture"
    if "concrete_proxy" in feature or "asphalt_proxy" in feature:
        return "material_proxy"
    if "saturation" in feature:
        return "chromatic_micro_variation"
    if "gray" in feature:
        return "illumination"
    return base


def _stats_from_maps(maps: torch.Tensor) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    h = maps.shape[-2]
    lower = maps[:, :, h // 2 :, :]
    upper = maps[:, :, : h // 2, :]
    for item_idx in range(maps.shape[0]):
        row: dict[str, float] = {}
        item = maps[item_idx]
        item_lower = lower[item_idx]
        item_upper = upper[item_idx]
        for channel_idx, name in enumerate(EVIDENCE_NAMES):
            channel = item[channel_idx]
            row[f"{name}_mean"] = float(channel.mean().item())
            row[f"{name}_std"] = float(channel.std(unbiased=False).item())
            row[f"{name}_q10"] = float(torch.quantile(channel.flatten(), 0.10).item())
            row[f"{name}_q90"] = float(torch.quantile(channel.flatten(), 0.90).item())
            row[f"{name}_lower_minus_upper"] = float(
                item_lower[channel_idx].mean().item() - item_upper[channel_idx].mean().item()
            )
        rows.append(row)
    return rows


def _cohen_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    pooled = math.sqrt((float(np.var(arr_a)) + float(np.var(arr_b))) / 2.0 + 1e-12)
    return float((float(np.mean(arr_a)) - float(np.mean(arr_b))) / pooled)


def _rank_auc(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.5
    values = [(v, 1) for v in a] + [(v, 0) for v in b]
    values.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(values):
        j = idx + 1
        while j < len(values) and values[j][0] == values[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            if values[k][1] == 1:
                rank_sum += avg_rank
        idx = j
    n1 = len(a)
    n0 = len(b)
    u = rank_sum - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pair_rows(feature_rows: list[dict[str, Any]], pairs: list[str]) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_class[str(row["true_label"])].append(row)
    feature_names = [name for name in feature_rows[0].keys() if name not in {"image_path", "true_label", "pred_label", "confidence"}]
    out: list[dict[str, Any]] = []
    for pair in pairs:
        class_a, class_b = pair.split("::", 1)
        rows_a = by_class.get(class_a, [])
        rows_b = by_class.get(class_b, [])
        if not rows_a or not rows_b:
            continue
        for feature in feature_names:
            vals_a = [float(row[feature]) for row in rows_a]
            vals_b = [float(row[feature]) for row in rows_b]
            d = _cohen_d(vals_a, vals_b)
            out.append(
                {
                    "pair": pair,
                    "class_a": class_a,
                    "class_b": class_b,
                    "feature": feature,
                    "family": _feature_family(feature),
                    "n_a": len(vals_a),
                    "n_b": len(vals_b),
                    "mean_a": float(np.mean(vals_a)),
                    "mean_b": float(np.mean(vals_b)),
                    "cohen_d_a_minus_b": d,
                    "abs_cohen_d": abs(d),
                    "auc_a_greater_b": _rank_auc(vals_a, vals_b),
                }
            )
    out.sort(key=lambda row: (row["pair"], -float(row["abs_cohen_d"])))
    return out


def _class_summary(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_class[str(row["true_label"])].append(row)
    feature_names = [name for name in feature_rows[0].keys() if name not in {"image_path", "true_label", "pred_label", "confidence"}]
    out: list[dict[str, Any]] = []
    for label, rows in sorted(by_class.items()):
        item: dict[str, Any] = {"class": label, "n": len(rows)}
        for feature in feature_names:
            vals = [float(row[feature]) for row in rows]
            item[f"{feature}_mean"] = float(np.mean(vals))
        out.append(item)
    return out


def _write_markdown(payload: dict[str, Any], pair_rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    lines.append("# S136 Evidence Preflight")
    lines.append("")
    lines.append(f"- Predictions: `{payload['predictions']}`")
    lines.append(f"- Processed images: `{payload['processed_images']}`")
    lines.append(f"- Failures: `{payload['failures']}`")
    lines.append(f"- Image size: `{payload['image_size']}`")
    lines.append("")
    lines.append("## Route Relevance")
    lines.append("")
    lines.append(
        "This preflight checks whether the S136 fixed evidence maps already separate RSCD hard pairs before any training. "
        "It does not validate model accuracy; it validates whether the proposed custom backbone has task-relevant physical signals to route."
    )
    lines.append("")
    for pair in payload["pairs"]:
        rows = [row for row in pair_rows if row["pair"] == pair][:8]
        lines.append(f"## {pair.replace('::', ' vs ')}")
        lines.append("")
        lines.append("| Feature | Family | |d| | signed d | AUC |")
        lines.append("|---|---|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['feature']} | {row['family']} | {row['abs_cohen_d']:.4f} | "
                f"{row['cohen_d_a_minus_b']:.4f} | {row['auc_a_greater_b']:.4f} |"
            )
        if rows:
            top = rows[0]
            lines.append("")
            lines.append(
                f"Mechanism note: strongest S136 evidence is `{top['feature']}` ({top['family']}), "
                f"|d|={top['abs_cohen_d']:.3f}."
            )
        lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        "If S135c fails promotion, S136 remains a plausible next route only if the target hard pairs above show useful pre-training separability "
        "in contrast, film, material, or roughness evidence. It should still be tested first under the same capped screen budget."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit S136 fixed evidence maps on real RSCD images.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--focus-class", action="append", default=[])
    parser.add_argument("--pair", action="append", default=[])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    focus = args.focus_class or DEFAULT_FOCUS_CLASSES
    pairs = args.pair or DEFAULT_PAIRS
    rows = [row for row in _read_predictions(args.predictions) if row["true_label"] in focus]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["true_label"])].append(row)
    selected: list[dict[str, Any]] = []
    for label, label_rows in sorted(by_class.items()):
        rng.shuffle(label_rows)
        selected.extend(label_rows[: args.max_per_class])

    model = FixedRoadEvidenceMaps().cpu().eval()
    feature_rows: list[dict[str, Any]] = []
    failures = 0
    batch_tensors: list[torch.Tensor] = []
    batch_rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal batch_tensors, batch_rows, feature_rows
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors, dim=0).cpu()
        with torch.no_grad():
            maps = model(batch)
        stats_rows = _stats_from_maps(maps)
        for meta, stats in zip(batch_rows, stats_rows):
            feature_rows.append({**meta, **stats})
        batch_tensors = []
        batch_rows = []

    for row in selected:
        tensor = _load_image(str(row["image_path"]), args.image_size)
        if tensor is None:
            failures += 1
            continue
        batch_tensors.append(tensor)
        batch_rows.append(row)
        if len(batch_tensors) >= args.batch_size:
            flush()
    flush()

    if not feature_rows:
        payload = {
            "ok": False,
            "predictions": str(args.predictions),
            "processed_images": 0,
            "failures": failures,
        }
        (args.output_dir / "s136_evidence_preflight.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    pair_summary = _pair_rows(feature_rows, pairs)
    class_summary = _class_summary(feature_rows)
    _write_csv(args.output_dir / "s136_evidence_image_features.csv", feature_rows)
    _write_csv(args.output_dir / "s136_evidence_pair_separability.csv", pair_summary)
    _write_csv(args.output_dir / "s136_evidence_class_summary.csv", class_summary)
    payload = {
        "ok": True,
        "predictions": str(args.predictions),
        "selected_images": len(selected),
        "processed_images": len(feature_rows),
        "failures": failures,
        "image_size": args.image_size,
        "max_per_class": args.max_per_class,
        "focus_classes": focus,
        "pairs": pairs,
    }
    (args.output_dir / "s136_evidence_preflight.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(payload, pair_summary, args.output_dir / "s136_evidence_preflight.md")
    print(json.dumps({"ok": True, "report": str(args.output_dir / "s136_evidence_preflight.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

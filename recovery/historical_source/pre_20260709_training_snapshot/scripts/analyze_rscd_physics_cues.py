from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_FOCUS_CLASSES = [
    "water_concrete_slight",
    "water_concrete_severe",
    "water_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "wet_concrete_smooth",
    "dry_concrete_slight",
    "dry_concrete_severe",
    "water_asphalt_slight",
    "water_asphalt_severe",
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


def _safe_open_image(path: str, image_size: int) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _features(arr: np.ndarray) -> dict[str, float]:
    r = arr[:, :, 0]
    g_ch = arr[:, :, 1]
    b = arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = (maxc - minc) / (maxc + 1e-6)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    grad = np.sqrt(gx * gx + gy * gy)
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    abs_lap = np.abs(lap)

    bright = (gray > 0.72).astype(np.float32)
    dark = (gray < 0.28).astype(np.float32)
    low_sat = (sat < 0.18).astype(np.float32)
    specular = bright * low_sat
    dark_water = dark * low_sat
    wet_proxy = np.clip(specular + 0.6 * dark_water, 0.0, 1.0)
    rough_proxy = _sigmoid((grad * 8.0 + abs_lap * 5.0 - 0.22) * 8.0)

    h = gray.shape[0]
    upper = slice(0, h // 2)
    lower = slice(h // 2, h)

    def mean(x: np.ndarray) -> float:
        return float(np.mean(x))

    def std(x: np.ndarray) -> float:
        return float(np.std(x))

    def q90(x: np.ndarray) -> float:
        return float(np.quantile(x, 0.90))

    def q10(x: np.ndarray) -> float:
        return float(np.quantile(x, 0.10))

    return {
        "gray_mean": mean(gray),
        "gray_std": std(gray),
        "gray_q10": q10(gray),
        "gray_q90": q90(gray),
        "sat_mean": mean(sat),
        "sat_std": std(sat),
        "grad_mean": mean(grad),
        "grad_q90": q90(grad),
        "lap_abs_mean": mean(abs_lap),
        "lap_abs_q90": q90(abs_lap),
        "specular_ratio": mean(specular),
        "dark_water_ratio": mean(dark_water),
        "wet_proxy_mean": mean(wet_proxy),
        "rough_proxy_mean": mean(rough_proxy),
        "lower_gray_mean": mean(gray[lower, :]),
        "upper_gray_mean": mean(gray[upper, :]),
        "lower_minus_upper_gray": mean(gray[lower, :]) - mean(gray[upper, :]),
        "lower_wet_proxy": mean(wet_proxy[lower, :]),
        "upper_wet_proxy": mean(wet_proxy[upper, :]),
        "lower_minus_upper_wet": mean(wet_proxy[lower, :]) - mean(wet_proxy[upper, :]),
        "lower_rough_proxy": mean(rough_proxy[lower, :]),
        "upper_rough_proxy": mean(rough_proxy[upper, :]),
        "lower_minus_upper_rough": mean(rough_proxy[lower, :]) - mean(rough_proxy[upper, :]),
        "texture_to_wet_ratio": mean(rough_proxy) / (mean(wet_proxy) + 1e-4),
        "wet_to_texture_ratio": mean(wet_proxy) / (mean(rough_proxy) + 1e-4),
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def _cohen_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    pooled = math.sqrt((float(np.var(a_arr)) + float(np.var(b_arr))) / 2.0 + 1e-12)
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled)


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


def _select_rows(
    predictions: list[dict[str, Any]],
    focus_classes: set[str],
    max_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        label = str(row["true_label"])
        if label in focus_classes:
            by_class[label].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in sorted(by_class):
        rows = by_class[label]
        rng.shuffle(rows)
        selected.extend(rows[:max_per_class])
    return selected


def _top_confusion_classes(predictions: list[dict[str, Any]], top_k: int) -> set[str]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in predictions:
        true_label = str(row["true_label"])
        pred_label = str(row["pred_label"])
        if true_label != pred_label:
            counter[(true_label, pred_label)] += 1
    classes: set[str] = set()
    for (true_label, pred_label), _count in counter.most_common(top_k):
        classes.add(true_label)
        classes.add(pred_label)
    return classes


def _class_feature_summary(feature_rows: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[str(row["true_label"])].append(row)
    rows = []
    for label, items in grouped.items():
        out: dict[str, Any] = {"class": label, "n": len(items)}
        for feature in feature_names:
            values = [float(item[feature]) for item in items]
            out[f"{feature}_mean"] = float(np.mean(values))
            out[f"{feature}_std"] = float(np.std(values))
        rows.append(out)
    return sorted(rows, key=lambda row: str(row["class"]))


def _pair_rows(
    feature_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    feature_names: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_class[str(row["true_label"])].append(row)

    confusion_counter: Counter[tuple[str, str]] = Counter()
    for row in predictions:
        true_label = str(row["true_label"])
        pred_label = str(row["pred_label"])
        if true_label != pred_label:
            confusion_counter[(true_label, pred_label)] += 1

    pair_set: set[tuple[str, str]] = set()
    for (a, b), _count in confusion_counter.most_common(top_k):
        if a in by_class and b in by_class:
            pair_set.add(tuple(sorted((a, b))))
    for pair in [
        ("water_concrete_slight", "water_concrete_severe"),
        ("wet_concrete_slight", "wet_concrete_severe"),
        ("dry_concrete_slight", "dry_concrete_severe"),
        ("water_concrete_smooth", "wet_concrete_smooth"),
        ("water_asphalt_slight", "water_asphalt_severe"),
    ]:
        if pair[0] in by_class and pair[1] in by_class:
            pair_set.add(tuple(sorted(pair)))

    rows: list[dict[str, Any]] = []
    for a, b in sorted(pair_set):
        a_items = by_class[a]
        b_items = by_class[b]
        for feature in feature_names:
            a_values = [float(item[feature]) for item in a_items]
            b_values = [float(item[feature]) for item in b_items]
            d = _cohen_d(a_values, b_values)
            auc = _rank_auc(a_values, b_values)
            rows.append(
                {
                    "class_a": a,
                    "class_b": b,
                    "feature": feature,
                    "n_a": len(a_items),
                    "n_b": len(b_items),
                    "mean_a": float(np.mean(a_values)) if a_values else 0.0,
                    "mean_b": float(np.mean(b_values)) if b_values else 0.0,
                    "delta_a_minus_b": float(np.mean(a_values) - np.mean(b_values)) if a_values and b_values else 0.0,
                    "cohen_d_a_minus_b": d,
                    "abs_cohen_d": abs(d),
                    "auc_a_greater_b": auc,
                    "confusion_count_a_to_b": int(confusion_counter[(a, b)]),
                    "confusion_count_b_to_a": int(confusion_counter[(b, a)]),
                }
            )
    return sorted(rows, key=lambda row: (str(row["class_a"]), str(row["class_b"]), -float(row["abs_cohen_d"])))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RSCD physics-cue separability for hard classes and confusion pairs.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--focus-class", action="append", default=[])
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--top-confusion-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13579)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _read_predictions(args.predictions)
    focus_classes = set(DEFAULT_FOCUS_CLASSES)
    focus_classes.update(args.focus_class)
    focus_classes.update(_top_confusion_classes(predictions, args.top_confusion_pairs))

    selected = _select_rows(predictions, focus_classes, args.max_per_class, args.seed)
    feature_rows: list[dict[str, Any]] = []
    failures = 0
    for idx, row in enumerate(selected, start=1):
        arr = _safe_open_image(str(row["image_path"]), args.image_size)
        if arr is None:
            failures += 1
            continue
        feature_rows.append({**row, **_features(arr)})
        if idx % 1000 == 0:
            print(f"processed {idx}/{len(selected)} images")

    if not feature_rows:
        payload = {"ok": False, "reason": "no readable images", "selected": len(selected), "failures": failures}
        (args.output_dir / "physics_cue_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 1

    skip = {"image_path", "true_label", "pred_label", "confidence"}
    feature_names = [name for name in feature_rows[0].keys() if name not in skip]
    _write_csv(args.output_dir / "image_physics_features.csv", feature_rows)
    class_summary = _class_feature_summary(feature_rows, feature_names)
    _write_csv(args.output_dir / "class_physics_feature_summary.csv", class_summary)
    pair_summary = _pair_rows(feature_rows, predictions, feature_names, args.top_confusion_pairs)
    _write_csv(args.output_dir / "pair_physics_separability.csv", pair_summary)

    best_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_summary:
        best_by_pair[(str(row["class_a"]), str(row["class_b"]))].append(row)

    payload = {
        "ok": True,
        "predictions": str(args.predictions),
        "selected_images": len(selected),
        "processed_images": len(feature_rows),
        "failures": failures,
        "focus_classes": sorted(focus_classes),
        "feature_names": feature_names,
        "top_pair_features": {
            f"{a} vs {b}": rows[:5]
            for (a, b), rows in sorted(best_by_pair.items())
        },
    }
    (args.output_dir / "physics_cue_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# RSCD physics-cue separability analysis",
        "",
        f"- Predictions: `{args.predictions}`",
        f"- Processed images: {len(feature_rows)} / selected {len(selected)}; failures: {failures}",
        f"- Classes analyzed: {len(focus_classes)}",
        "",
        "## Why This Matters",
        "",
        "This read-only audit measures which hand-crafted physics cues separate the hard RSCD classes. "
        "The next network mechanism should use cues with large signed separation early in the backbone, "
        "especially for water/wet + concrete + slight/severe coupling.",
        "",
        "## Most Separable Hard Pairs",
        "",
    ]
    for (a, b), rows in sorted(best_by_pair.items()):
        md.extend([f"### {a} vs {b}", "", "| Feature | Mean A | Mean B | Delta A-B | Cohen d | AUC(A>B) | Confusions A->B/B->A |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in rows[:5]:
            md.append(
                f"| {row['feature']} | {_fmt(row['mean_a'])} | {_fmt(row['mean_b'])} | "
                f"{_fmt(row['delta_a_minus_b'])} | {_fmt(row['cohen_d_a_minus_b'])} | "
                f"{_fmt(row['auc_a_greater_b'])} | {row['confusion_count_a_to_b']}/{row['confusion_count_b_to_a']} |"
            )
        md.append("")

    wc_key = tuple(sorted(("water_concrete_slight", "water_concrete_severe")))
    wc_rows = best_by_pair.get(wc_key, [])
    if wc_rows:
        best = wc_rows[0]
        md.extend(
            [
                "## Mechanism Hint",
                "",
                f"- For `water_concrete_slight` vs `water_concrete_severe`, the strongest measured cue is "
                f"`{best['feature']}` with Cohen d `{best['cohen_d_a_minus_b']:.3f}`.",
                "- If S135c does not pass the screen, the next single route should convert this cue into an early conditioner or task-specific stem, not a late classifier head.",
            ]
        )

    (args.output_dir / "physics_cue_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "physics_cue_analysis.md"), "ok": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _label(name: str) -> str:
    return name.replace("_", "-")


def _open_thumb(path: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def make_edge_montage(
    predictions: pd.DataFrame,
    edges: pd.DataFrame,
    out_path: Path,
    *,
    top_edges: int,
    samples_per_edge: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = seed
    thumb_size = (160, 106)
    label_width = 310
    row_height = 145
    width = label_width + samples_per_edge * (thumb_size[0] + 8) + 18
    height = top_edges * row_height + 12
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    chosen: list[dict[str, object]] = []

    for row_idx, edge in enumerate(edges.head(top_edges).itertuples(index=False)):
        y0 = 10 + row_idx * row_height
        true_label = str(edge.true_label)
        pred_label = str(edge.pred_label)
        subset = predictions[
            (predictions["true_label"] == true_label)
            & (predictions["pred_label"] == pred_label)
        ].copy()
        if subset.empty:
            continue
        sample = subset.sample(n=min(samples_per_edge, len(subset)), random_state=rng + row_idx)
        text = (
            f"{row_idx + 1}. {_label(true_label)} -> {_label(pred_label)}\n"
            f"count={int(edge.count)}, err={float(edge.error_rate_in_true_class) * 100:.2f}%\n"
            f"{edge.relation}"
        )
        _draw_text(draw, (10, y0 + 12), text)
        for col_idx, item in enumerate(sample.itertuples(index=False)):
            try:
                thumb = _open_thumb(str(item.image_path), thumb_size)
            except Exception:
                thumb = Image.new("RGB", thumb_size, (230, 230, 230))
            x0 = label_width + col_idx * (thumb_size[0] + 8)
            canvas.paste(thumb, (x0, y0 + 8))
            _draw_text(draw, (x0, y0 + thumb_size[1] + 13), f"conf={float(item.confidence):.2f}")
        chosen.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": int(edge.count),
                "error_rate": float(edge.error_rate_in_true_class),
                "relation": str(edge.relation),
                "sampled": len(sample),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return chosen


def make_low_f1_montage(
    predictions: pd.DataFrame,
    report: pd.DataFrame,
    out_path: Path,
    *,
    classes: int,
    samples_per_class: int,
    seed: int,
) -> list[dict[str, object]]:
    thumb_size = (150, 100)
    label_width = 290
    row_height = 132
    width = label_width + samples_per_class * (thumb_size[0] + 8) + 18
    height = classes * row_height + 12
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    rows: list[dict[str, object]] = []

    for row_idx, item in enumerate(report.sort_values("f1").head(classes).itertuples(index=False)):
        y0 = 10 + row_idx * row_height
        cls = str(item.class_label)
        subset = predictions[predictions["true_label"] == cls].copy()
        sample = subset.sample(n=min(samples_per_class, len(subset)), random_state=seed + 100 + row_idx)
        _draw_text(
            draw,
            (10, y0 + 10),
            f"{row_idx + 1}. {_label(cls)}\nF1={float(item.f1) * 100:.2f}% R={float(item.recall) * 100:.2f}% P={float(item.precision) * 100:.2f}%",
        )
        for col_idx, pred in enumerate(sample.itertuples(index=False)):
            try:
                thumb = _open_thumb(str(pred.image_path), thumb_size)
            except Exception:
                thumb = Image.new("RGB", thumb_size, (230, 230, 230))
            x0 = label_width + col_idx * (thumb_size[0] + 8)
            canvas.paste(thumb, (x0, y0 + 6))
            pred_tag = "ok" if pred.true_label == pred.pred_label else _label(str(pred.pred_label))[:18]
            _draw_text(draw, (x0, y0 + thumb_size[1] + 10), pred_tag)
        rows.append(
            {
                "class_label": cls,
                "f1": float(item.f1),
                "recall": float(item.recall),
                "precision": float(item.precision),
                "sampled": len(sample),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-edges", type=int, default=12)
    parser.add_argument("--samples-per-edge", type=int, default=5)
    parser.add_argument("--low-f1-classes", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    edges = pd.read_csv(args.edges)
    nodes = pd.read_csv(args.nodes).rename(columns={"class_label": "class_label"})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    edge_png = args.out_dir / "rscd_hard_confusion_edge_montage.png"
    low_f1_png = args.out_dir / "rscd_low_f1_class_visual_montage.png"
    chosen_edges = make_edge_montage(
        predictions,
        edges,
        edge_png,
        top_edges=int(args.top_edges),
        samples_per_edge=int(args.samples_per_edge),
        seed=int(args.seed),
    )
    low_rows = make_low_f1_montage(
        predictions,
        nodes,
        low_f1_png,
        classes=int(args.low_f1_classes),
        samples_per_class=int(args.samples_per_class),
        seed=int(args.seed),
    )

    report_path = args.out_dir / "rscd_visual_hard_edge_observations.md"
    lines = [
        "# RSCD Visual Hard-Edge Observations",
        "",
        "This diagnostic uses the current best full-test predictions. It visualizes audited confusion edges and low-F1 classes without changing the training protocol.",
        "",
        f"- Hard-edge montage: `{edge_png.name}`",
        f"- Low-F1 class montage: `{low_f1_png.name}`",
        "",
        "## Strong Confusion Edges",
        "",
        "| edge | mistakes | error rate | relation |",
        "|---|---:|---:|---|",
    ]
    for row in chosen_edges:
        lines.append(
            f"| `{_label(str(row['true_label']))}` -> `{_label(str(row['pred_label']))}` "
            f"| {row['count']} | {float(row['error_rate']) * 100:.2f}% | {row['relation']} |"
        )
    lines.extend(["", "## Lowest-F1 Classes", "", "| class | F1 | recall | precision |", "|---|---:|---:|---:|"])
    for row in low_rows:
        lines.append(
            f"| `{_label(str(row['class_label']))}` | {float(row['f1']) * 100:.2f}% "
            f"| {float(row['recall']) * 100:.2f}% | {float(row['precision']) * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Working Interpretation",
            "",
            "- Most errors stay inside factor-neighbor edges: roughness, friction wetness, or material changes while the other factors remain fixed.",
            "- The visual challenge is therefore boundary evidence, not coarse road-scene recognition.",
            "- Future modules should learn pair-conditioned evidence for roughness and water-film boundaries instead of smoothing neighboring labels.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from friction_affordance.c3_experiment import build_model, load_config  # noqa: E402


FOCUS_CLASSES = [
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
    "wet_asphalt_slight",
    "wet_asphalt_severe",
]

MECHANISM_NAMES = [
    "concrete_like",
    "film",
    "rough",
    "hidden_rough",
    "rough_ring",
    "persistence",
    "rough_island",
    "film_boundary",
    "signed_severe_minus_slight",
    "contrast_visibility",
    "dark_film_quantile",
    "chroma_micro_variation",
]


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image_path": str(row.get("image_path", "")),
                    "true_label": str(row.get("true_label", "")),
                    "pred_label": str(row.get("pred_label", "")),
                }
            )
    return rows


def _class_map_from_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    labels = sorted({row["true_label"] for row in rows if row.get("true_label")})
    return {name: idx for idx, name in enumerate(labels)}


def _sample_rows(rows: list[dict[str, str]], classes: list[str], max_per_class: int, seed: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    wanted = set(classes)
    for row in rows:
        if row["true_label"] in wanted and row.get("image_path"):
            grouped[row["true_label"]].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label in sorted(grouped):
        group = grouped[label]
        rng.shuffle(group)
        selected.extend(group[:max_per_class])
    return selected


def _load_batch(paths: list[str], image_size: int) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    tensors = []
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            data = data.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
            tensors.append((data - mean) / std)
    return torch.stack(tensors, dim=0)


def _top_fraction_mean(x: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    flat = x.flatten(1)
    k = max(1, int(flat.size(1) * float(fraction)))
    return flat.topk(k, dim=1).values.mean(dim=1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _class_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["true_label"])].append(row)
    out: list[dict[str, Any]] = []
    for label, items in grouped.items():
        item: dict[str, Any] = {"class": label, "n": len(items)}
        for key in keys:
            vals = [float(row[key]) for row in items]
            item[f"{key}_mean"] = _mean(vals)
            item[f"{key}_max"] = max(vals) if vals else 0.0
        out.append(item)
    return sorted(out, key=lambda item: str(item["class"]))


def _pair_delta(summary: list[dict[str, Any]], a: str, b: str, keys: list[str]) -> dict[str, Any] | None:
    by_class = {str(row["class"]): row for row in summary}
    if a not in by_class or b not in by_class:
        return None
    row: dict[str, Any] = {"class_a": a, "class_b": b, "n_a": by_class[a]["n"], "n_b": by_class[b]["n"]}
    for key in keys:
        av = float(by_class[a].get(f"{key}_mean", 0.0))
        bv = float(by_class[b].get(f"{key}_mean", 0.0))
        row[f"{key}_mean_a"] = av
        row[f"{key}_mean_b"] = bv
        row[f"{key}_delta_a_minus_b"] = av - bv
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit S135c early stem activation on real RSCD images.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13531)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    rows = _read_prediction_rows(args.predictions)
    class_to_idx = _class_map_from_rows(rows)
    model = build_model(cfg, class_to_idx).cpu().eval()
    conditioner = getattr(model, "water_concrete_topology_texture_stem_conditioner", None)
    if conditioner is None:
        raise RuntimeError("S135c conditioner is not enabled")
    image_size = int(args.image_size or cfg.get("data", {}).get("image_size", 192))
    selected = _sample_rows(rows, FOCUS_CLASSES, int(args.max_per_class), int(args.seed))

    image_rows: list[dict[str, Any]] = []
    failures = 0
    with torch.no_grad():
        for start in range(0, len(selected), int(args.batch_size)):
            batch_rows = selected[start : start + int(args.batch_size)]
            try:
                x = _load_batch([row["image_path"] for row in batch_rows], image_size)
            except Exception:
                failures += len(batch_rows)
                continue
            _y, aux = conditioner(x)
            mech = aux["mechanism"].detach().float()
            spatial_gate = aux["spatial_gate"].detach().float()
            learned_gate = aux["gate"].detach().float().flatten()
            delta = aux["delta"].detach().float()
            for idx, row in enumerate(batch_rows):
                out: dict[str, Any] = {
                    "image_path": row["image_path"],
                    "true_label": row["true_label"],
                    "pred_label": row["pred_label"],
                    "learned_gate": float(learned_gate[idx]),
                    "spatial_gate_mean": float(spatial_gate[idx : idx + 1].mean()),
                    "spatial_gate_top10": float(_top_fraction_mean(spatial_gate[idx : idx + 1], 0.10)[0]),
                    "delta_abs_mean": float(delta[idx : idx + 1].abs().mean()),
                }
                for ch, name in enumerate(MECHANISM_NAMES):
                    channel = mech[idx : idx + 1, ch : ch + 1]
                    out[f"{name}_mean"] = float(channel.mean())
                    out[f"{name}_top10"] = float(_top_fraction_mean(channel, 0.10)[0])
                image_rows.append(out)

    keys = [
        "spatial_gate_mean",
        "spatial_gate_top10",
        "learned_gate",
        "contrast_visibility_mean",
        "dark_film_quantile_mean",
        "chroma_micro_variation_mean",
        "signed_severe_minus_slight_mean",
        "film_mean",
        "rough_mean",
        "film_boundary_mean",
        "delta_abs_mean",
    ]
    summary_rows = _class_summary(image_rows, keys)
    pair_specs = [
        ("water_concrete_severe", "water_concrete_slight"),
        ("wet_concrete_severe", "wet_concrete_slight"),
        ("dry_concrete_severe", "dry_concrete_slight"),
        ("water_concrete_smooth", "wet_concrete_smooth"),
        ("water_asphalt_severe", "water_asphalt_slight"),
        ("wet_asphalt_severe", "wet_asphalt_slight"),
    ]
    pair_rows = [row for row in (_pair_delta(summary_rows, a, b, keys) for a, b in pair_specs) if row is not None]

    _write_csv(args.output_dir / "s135c_stem_image_activation.csv", image_rows)
    _write_csv(args.output_dir / "s135c_stem_class_activation_summary.csv", summary_rows)
    _write_csv(args.output_dir / "s135c_stem_pair_activation_delta.csv", pair_rows)

    by_class = {str(row["class"]): row for row in summary_rows}
    wc_delta = next((row for row in pair_rows if row["class_a"] == "water_concrete_severe" and row["class_b"] == "water_concrete_slight"), None)
    payload = {
        "ok": True,
        "config": str(args.config),
        "predictions": str(args.predictions),
        "processed": len(image_rows),
        "selected": len(selected),
        "failures": failures,
        "image_size": image_size,
        "mechanism_channels": getattr(conditioner, "mechanism_channels", None),
        "water_concrete_delta": wc_delta,
    }
    (args.output_dir / "s135c_stem_activation_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# S135c Real-Image Stem Activation Audit",
        "",
        f"- Config: `{args.config}`",
        f"- Predictions source: `{args.predictions}`",
        f"- Processed images: {len(image_rows)} / selected {len(selected)}; failures: {failures}",
        f"- Mechanism channels: {getattr(conditioner, 'mechanism_channels', None)}",
        "",
        "## Class Means",
        "",
        "| Class | n | spatial gate | contrast visibility | dark film quantile | chroma micro-var | signed severe | delta abs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in FOCUS_CLASSES:
        row = by_class.get(label)
        if not row:
            continue
        md.append(
            f"| {label} | {row['n']} | {row['spatial_gate_mean_mean']:.6f} | "
            f"{row['contrast_visibility_mean_mean']:.6f} | {row['dark_film_quantile_mean_mean']:.6f} | "
            f"{row['chroma_micro_variation_mean_mean']:.6f} | {row['signed_severe_minus_slight_mean_mean']:.6f} | "
            f"{row['delta_abs_mean_mean']:.6f} |"
        )

    md.extend(
        [
            "",
            "## Hard Pair Deltas",
            "",
            "Delta is `class_a - class_b`; for `water_concrete_severe - water_concrete_slight`, positive contrast visibility means the new mechanism follows the stable cue evidence.",
            "",
            "| Pair | spatial gate delta | contrast visibility delta | dark film delta | chroma var delta | signed severe delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        pair = f"{row['class_a']} - {row['class_b']}"
        md.append(
            f"| {pair} | {row['spatial_gate_mean_delta_a_minus_b']:.6f} | "
            f"{row['contrast_visibility_mean_delta_a_minus_b']:.6f} | "
            f"{row['dark_film_quantile_mean_delta_a_minus_b']:.6f} | "
            f"{row['chroma_micro_variation_mean_delta_a_minus_b']:.6f} | "
            f"{row['signed_severe_minus_slight_mean_delta_a_minus_b']:.6f} |"
        )

    md.extend(["", "## Mechanism Diagnosis", ""])
    if wc_delta is None:
        md.append("- `water_concrete_severe` vs `water_concrete_slight` was not available in the sampled rows.")
    else:
        cv_delta = float(wc_delta["contrast_visibility_mean_delta_a_minus_b"])
        gate_delta = float(wc_delta["spatial_gate_mean_delta_a_minus_b"])
        signed_delta = float(wc_delta["signed_severe_minus_slight_mean_delta_a_minus_b"])
        md.append(f"- Water-concrete severe minus slight contrast visibility delta: `{cv_delta:.6f}`.")
        md.append(f"- Water-concrete severe minus slight spatial gate delta: `{gate_delta:.6f}`.")
        md.append(f"- Water-concrete severe minus slight signed mechanism delta: `{signed_delta:.6f}`.")
        if cv_delta > 0:
            md.append("- The new contrast-visibility channel follows the stable cue direction found in S7/S96 audits.")
        else:
            md.append("- The contrast-visibility channel does not follow the expected direction; if S135c fails, revise this gate before another run.")
    md.append("- `delta_abs_mean` is expected to be zero before training because the final adapter layer is zero-initialized.")

    (args.output_dir / "s135c_stem_activation_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(args.output_dir / "s135c_stem_activation_audit.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

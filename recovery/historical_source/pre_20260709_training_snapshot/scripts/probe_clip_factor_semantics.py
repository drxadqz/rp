"""Probe whether CLIP can serve as an offline RSCD factor-semantic teacher.

This is not a promoted model.  It checks whether a multimodal model can
separate RSCD's friction, material, and unevenness factors on public images
well enough to justify later distillation or expert routing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm


PROMPTS: dict[str, dict[str, list[str]]] = {
    "friction": {
        "dry": [
            "a close-up road surface patch that is dry",
            "dry pavement texture photographed from a vehicle",
            "a dry road ground patch with no water film",
        ],
        "wet": [
            "a close-up road surface patch that is wet",
            "wet pavement with a thin water film",
            "a damp road surface patch without standing water",
        ],
        "water": [
            "a close-up road surface patch covered by water",
            "road pavement with standing water or a puddle",
            "a flooded wet road surface patch",
        ],
        "fresh_snow": [
            "a road surface patch covered by fresh snow",
            "white fresh snow on a road",
            "a snowy road surface patch with loose snow",
        ],
        "melted_snow": [
            "a road surface patch with melting snow or slush",
            "slushy melted snow on the road",
            "a partly melted snowy road surface patch",
        ],
        "ice": [
            "an icy road surface patch",
            "a road covered by ice",
            "glassy ice on a road surface",
        ],
    },
    "material": {
        "asphalt": [
            "a close-up asphalt road surface patch",
            "black asphalt pavement texture",
            "vehicle camera patch of asphalt pavement",
        ],
        "concrete": [
            "a close-up concrete road surface patch",
            "gray concrete pavement texture",
            "vehicle camera patch of concrete pavement",
        ],
        "dirt_mud": [
            "a muddy road surface patch",
            "brown mud on a road",
            "soft muddy ground texture on a road",
        ],
        "gravel": [
            "a gravel road surface patch",
            "coarse gravel stones on a road",
            "granular gravel ground texture",
        ],
        "winter": [
            "a winter road surface covered by snow or ice",
            "snow or ice hiding the road material",
            "a snowy icy road surface patch",
        ],
    },
    "unevenness": {
        "smooth": [
            "a smooth flat road surface patch",
            "smooth pavement with little texture",
            "a flat even road ground patch",
        ],
        "slight": [
            "a slightly rough road surface patch",
            "pavement with moderate small texture",
            "a road surface with mild unevenness",
        ],
        "severe": [
            "a very rough uneven road surface patch",
            "pavement with severe cracks bumps or rough texture",
            "a road surface with strong unevenness",
        ],
        "granular": [
            "a granular gravel or mud road surface patch",
            "loose particles and coarse grains on a road",
            "granular off-road ground texture",
        ],
        "winter": [
            "snow or ice covering road texture",
            "a winter road patch where roughness is hidden by snow or ice",
            "snowy icy road surface texture",
        ],
    },
}


def normalize_factor(row: pd.Series, factor: str) -> str:
    value = row[f"{factor}_label"]
    label = str(row["class_label"]).replace("-", "_")
    if factor == "material" and (not isinstance(value, str) or value == "nan"):
        return "winter"
    if factor == "unevenness" and (not isinstance(value, str) or value == "nan"):
        return "winter" if label in {"fresh_snow", "melted_snow", "ice"} else "granular"
    if factor == "unevenness" and value in {"mud", "gravel"}:
        return "granular"
    if factor == "material" and value == "mud":
        return "dirt_mud"
    return str(value)


def sample_manifest(manifest: Path, samples_per_class: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    chunks = []
    for _, group in df.groupby("class_label", sort=True):
        chunks.append(group.sample(n=min(samples_per_class, len(group)), random_state=seed))
    sampled = pd.concat(chunks, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def frame_to_markdown(df: pd.DataFrame) -> str:
    headers = [""] + [str(col) for col in df.columns]
    rows = [[str(idx)] + [str(value) for value in row] for idx, row in df.iterrows()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def text_features(model, processor, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for factor, label_prompts in PROMPTS.items():
            out[factor] = {}
            for label, prompts in label_prompts.items():
                inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
                feat = _as_feature_tensor(model.get_text_features(**inputs))
                feat = F.normalize(feat, dim=1).mean(dim=0, keepdim=True)
                out[factor][label] = F.normalize(feat, dim=1)
    return out


def _as_feature_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "image_embeds"):
        return value.image_embeds
    if hasattr(value, "text_embeds"):
        return value.text_embeds
    if hasattr(value, "pooler_output"):
        return value.pooler_output
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported CLIP feature output type: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_test.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    sampled = sample_manifest(args.manifest, int(args.samples_per_class), int(args.seed))
    sampled.to_csv(args.output_dir / "sampled_images.csv", index=False)

    txt = text_features(model, processor, device)
    labels_by_factor = {factor: list(label_prompts.keys()) for factor, label_prompts in PROMPTS.items()}
    text_mats = {
        factor: torch.cat([txt[factor][label] for label in labels], dim=0)
        for factor, labels in labels_by_factor.items()
    }

    records = []
    paths = sampled["image_path"].astype(str).tolist()
    with torch.no_grad():
        for batch_paths in tqdm(list(batched(paths, int(args.batch_size))), desc="clip-probe"):
            images = [load_image(p) for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            image_feat = _as_feature_tensor(model.get_image_features(**inputs))
            image_feat = F.normalize(image_feat, dim=1)
            for row_offset, path in enumerate(batch_paths):
                rec = {"image_path": path}
                for factor, labels in labels_by_factor.items():
                    logits = (image_feat[row_offset : row_offset + 1] @ text_mats[factor].T).squeeze(0)
                    probs = logits.softmax(dim=0).detach().cpu()
                    idx = int(probs.argmax().item())
                    rec[f"clip_{factor}"] = labels[idx]
                    rec[f"clip_{factor}_confidence"] = float(probs[idx])
                    for label, prob in zip(labels, probs.tolist()):
                        rec[f"p_{factor}_{label}"] = float(prob)
                records.append(rec)

    pred = pd.DataFrame(records)
    merged = sampled.merge(pred, on="image_path", how="left")
    for factor in ("friction", "material", "unevenness"):
        merged[f"true_{factor}"] = merged.apply(lambda row, f=factor: normalize_factor(row, f), axis=1)
    merged.to_csv(args.output_dir / "clip_factor_predictions.csv", index=False)

    metrics: dict[str, object] = {
        "model": args.model,
        "manifest": str(args.manifest),
        "samples": int(len(merged)),
        "samples_per_class": int(args.samples_per_class),
    }
    md = [
        "# CLIP RSCD Factor-Semantic Probe",
        "",
        f"- model: `{args.model}`",
        f"- samples: {len(merged)} ({args.samples_per_class} per RSCD class)",
        f"- manifest: `{args.manifest}`",
        "",
        "This probe is only for judging whether a multimodal model is usable as an offline factor teacher.",
        "",
    ]
    for factor in ("friction", "material", "unevenness"):
        true = merged[f"true_{factor}"].astype(str)
        pred_factor = merged[f"clip_{factor}"].astype(str)
        labels = sorted(set(true.tolist()) | set(pred_factor.tolist()))
        acc = accuracy_score(true, pred_factor)
        metrics[f"{factor}_accuracy"] = float(acc)
        report = classification_report(true, pred_factor, labels=labels, zero_division=0, output_dict=True)
        metrics[f"{factor}_report"] = report
        cm = confusion_matrix(true, pred_factor, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true:{x}" for x in labels], columns=[f"pred:{x}" for x in labels])
        cm_df.to_csv(args.output_dir / f"clip_{factor}_confusion.csv")
        md.extend(
            [
                f"## {factor}",
                "",
                f"- accuracy: {acc * 100:.2f}%",
                f"- macro F1: {report['macro avg']['f1-score'] * 100:.2f}%",
                "",
                frame_to_markdown(cm_df),
                "",
            ]
        )

    with open(args.output_dir / "clip_factor_probe.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    (args.output_dir / "clip_factor_probe.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

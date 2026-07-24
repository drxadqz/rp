from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.engine import build_model, move_batch
from friction_affordance.ontology import FRICTION_STATES, RISK
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml, resolve_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evidence_maps"))
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument(
        "--selection",
        choices=[
            "first",
            "mixed",
            "risk_success",
            "risk_failure",
            "low_friction_success",
            "low_friction_failure",
            "roadsaw_failure",
        ],
        default="mixed",
    )
    parser.add_argument("--scan-multiplier", type=int, default=20)
    parser.add_argument("--clean", action="store_true", help="Remove old jpg/json outputs in out-dir before exporting.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = _build_loader(cfg, args.split, max(args.max_samples * args.scan_multiplier, args.max_samples))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in list(args.out_dir.glob("*.jpg")) + list(args.out_dir.glob("*.json")):
            path.unlink()
    metadata: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for batch in loader:
        moved = move_batch(batch, device)
        out = model(moved["image"], grl_lambda=0.0, domain_idx=moved.get("domain_idx"))
        evidence = out.get("evidence_field")
        if not evidence:
            raise RuntimeError("Checkpoint/config does not enable model.use_evidence_field.")
        attn = evidence["attention"]
        attn = F.interpolate(attn, size=moved["image"].shape[-2:], mode="bilinear", align_corners=False)
        risk_pred = out["logits"]["risk"].argmax(dim=1).detach().cpu().tolist()
        friction_pred = out["logits"]["friction"].argmax(dim=1).detach().cpu().tolist()
        risk_true = moved["labels"]["risk"].detach().cpu().tolist()
        friction_true = moved["labels"]["friction"].detach().cpu().tolist()
        risk_mask = moved["masks"]["risk"].detach().cpu().tolist()
        friction_mask = moved["masks"]["friction"].detach().cpu().tolist()
        mu_interval = out["mu_interval"].detach().cpu().numpy()
        for i in range(moved["image"].size(0)):
            if len(selected) >= args.max_samples:
                break
            record = _record_for_sample(
                batch=batch,
                i=i,
                risk_pred=risk_pred[i],
                risk_true=risk_true[i],
                risk_mask=bool(risk_mask[i]),
                friction_pred=friction_pred[i],
                friction_true=friction_true[i],
                friction_mask=bool(friction_mask[i]),
                mu_interval=mu_interval[i],
            )
            record.update(_attention_diagnostics(evidence, i))
            if not _keep_record(record, args.selection, selected):
                continue
            base = _tensor_to_uint8(moved["image"][i].detach().cpu())
            heat = _attention_to_heat(attn[i, 0].detach().cpu().numpy())
            overlay = _overlay(base, heat)
            name = "{idx:04d}_{tag}_{dataset}_true-{true}_pred-{pred}.jpg".format(
                idx=len(selected),
                tag=record["selection_tag"],
                dataset=record["dataset"],
                true=record.get("true_risk", "na"),
                pred=record["pred_risk"],
            )
            Image.fromarray(overlay).save(args.out_dir / name, quality=92)
            record["file"] = name
            metadata.append(record)
            selected.append(record)
        if len(selected) >= args.max_samples:
            break

    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(selected)} evidence overlays to {args.out_dir}")


def _build_loader(cfg: dict[str, Any], split: str, max_samples: int) -> DataLoader:
    data_cfg = cfg["data"]
    manifests = data_cfg.get(f"{split}_manifests", data_cfg["val_manifests"])
    ds = ManifestDataset(
        manifests,
        transform=build_transforms(
            int(data_cfg.get("image_size", 224)),
            train=False,
            aug_cfg=data_cfg.get("augmentation"),
        ),
        max_samples=max_samples,
        sample_seed=int(data_cfg.get("sample_seed", 17)) + 11,
    )
    return DataLoader(
        ds,
        batch_size=max(1, min(int(data_cfg.get("batch_size", 16)), max_samples)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_manifest_batch,
    )


def _record_for_sample(
    batch: dict[str, Any],
    i: int,
    risk_pred: int,
    risk_true: int,
    risk_mask: bool,
    friction_pred: int,
    friction_true: int,
    friction_mask: bool,
    mu_interval: np.ndarray,
) -> dict[str, Any]:
    risk_correct = bool(risk_mask and risk_pred == risk_true)
    friction_correct = bool(friction_mask and friction_pred == friction_true)
    low_true = bool(risk_mask and risk_true >= RISK.index("high"))
    low_pred = bool(risk_pred >= RISK.index("high"))
    if not risk_mask:
        tag = "unlabeled"
    elif risk_correct and low_true:
        tag = "low_success"
    elif (not risk_correct) and low_true:
        tag = "low_failure"
    elif risk_correct:
        tag = "risk_success"
    else:
        tag = "risk_failure"
    return {
        "image_path": batch["image_path"][i],
        "dataset": batch["dataset"][i],
        "group_key": batch["group_key"][i],
        "selection_tag": tag,
        "risk_correct": risk_correct,
        "friction_correct": friction_correct,
        "true_risk": RISK[risk_true] if risk_mask else None,
        "pred_risk": RISK[risk_pred],
        "true_friction": FRICTION_STATES[friction_true] if friction_mask else None,
        "pred_friction": FRICTION_STATES[friction_pred],
        "low_friction_true": low_true,
        "low_friction_pred": low_pred,
        "pred_mu_low": float(mu_interval[0]),
        "pred_mu_high": float(mu_interval[1]),
    }


def _attention_diagnostics(evidence: dict[str, torch.Tensor], i: int) -> dict[str, float]:
    attention = evidence["attention"][i, 0].detach().cpu()
    h, w = attention.shape
    yy = torch.linspace(0.0, 1.0, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0.0, 1.0, w).view(1, w).expand(h, w)
    bottom_half = yy >= 0.5
    bottom_third = yy >= (2.0 / 3.0)
    center = (xx >= 0.25) & (xx <= 0.75)
    center_bottom = center & bottom_half
    top_half = yy < 0.5

    attn = attention / attention.sum().clamp_min(1e-8)
    out = {
        "attention_bottom_half_mass": float(attn[bottom_half].sum()),
        "attention_bottom_third_mass": float(attn[bottom_third].sum()),
        "attention_center_bottom_mass": float(attn[center_bottom].sum()),
        "attention_top_half_mass": float(attn[top_half].sum()),
    }
    entropy = evidence.get("attention_entropy")
    if entropy is not None:
        out["attention_entropy"] = float(entropy[i].detach().cpu())
    for key in ["road_likelihood", "contact_prior"]:
        value = evidence.get(key)
        if value is None:
            continue
        value_map = value[i, 0].detach().cpu()
        if value_map.shape != attn.shape:
            value_map = F.interpolate(
                value[i : i + 1],
                size=attn.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0].detach().cpu()
        out[f"attention_weighted_{key}_mean"] = float((attn * value_map).sum())
    return out


def _keep_record(record: dict[str, Any], selection: str, selected: list[dict[str, Any]]) -> bool:
    if selection == "first":
        return True
    if selection == "risk_success":
        return bool(record["risk_correct"])
    if selection == "risk_failure":
        return record.get("true_risk") is not None and not bool(record["risk_correct"])
    if selection == "low_friction_success":
        return bool(record["low_friction_true"]) and bool(record["risk_correct"])
    if selection == "low_friction_failure":
        return bool(record["low_friction_true"]) and not bool(record["risk_correct"])
    if selection == "roadsaw_failure":
        return record["dataset"] == "roadsaw" and record.get("true_risk") is not None and not bool(record["risk_correct"])

    quotas = {
        "low_failure": 4,
        "risk_failure": 6,
        "low_success": 4,
        "risk_success": 10,
    }
    counts: dict[str, int] = {}
    for item in selected:
        counts[item["selection_tag"]] = counts.get(item["selection_tag"], 0) + 1
    tag = str(record["selection_tag"])
    return counts.get(tag, 0) < quotas.get(tag, 4)


def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _attention_to_heat(attention: np.ndarray) -> np.ndarray:
    attn = attention.astype(np.float32)
    attn = attn - float(attn.min())
    attn = attn / max(float(attn.max()), 1e-6)
    red = attn
    green = np.clip(1.0 - np.abs(attn - 0.45) / 0.45, 0.0, 1.0)
    blue = 1.0 - attn
    return np.stack([red, green, blue], axis=-1)


def _overlay(base: np.ndarray, heat: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    heat_uint8 = (heat * 255.0).round().astype(np.uint8)
    out = (1.0 - alpha) * base.astype(np.float32) + alpha * heat_uint8.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()

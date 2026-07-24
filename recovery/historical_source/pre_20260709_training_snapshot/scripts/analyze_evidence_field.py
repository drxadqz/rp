from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.engine import build_model, dataloader_worker_settings, move_batch
from friction_affordance.ontology import FRICTION_STATES, RISK
from friction_affordance.transforms import build_mask_transforms, build_transforms
from friction_affordance.utils import load_yaml, resolve_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = _build_loader(cfg, args.split, args.max_samples)
    records = _collect(model, loader, device)
    report = _summarize(records)
    report["checkpoint"] = str(args.checkpoint)
    report["config"] = str(args.config)
    report["split"] = args.split
    report["max_samples"] = int(args.max_samples)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(_render_markdown(report))


def _build_loader(cfg: dict[str, Any], split: str, max_samples: int) -> DataLoader:
    data_cfg = dict(cfg["data"])
    data_cfg["num_workers"] = 0
    ds = ManifestDataset(
        data_cfg.get(f"{split}_manifests", data_cfg["val_manifests"]),
        transform=build_transforms(
            int(data_cfg.get("image_size", 224)),
            train=False,
            aug_cfg=data_cfg.get("augmentation"),
        ),
        mask_transform=build_mask_transforms(
            int(data_cfg.get("image_size", 224)),
            data_cfg.get("augmentation"),
            pretransformed=bool(data_cfg.get("road_mask_pretransformed", False)),
        )
        if bool(data_cfg.get("load_road_masks", False))
        else None,
        load_road_masks=bool(data_cfg.get("load_road_masks", False)),
        max_samples=max_samples if max_samples > 0 else None,
        max_samples_per_dataset=data_cfg.get(f"max_{split}_samples_per_dataset"),
        max_samples_per_class=data_cfg.get(f"max_{split}_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)) + 31,
    )
    num_workers, loader_kwargs = dataloader_worker_settings(data_cfg)
    return DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )


def _collect(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch in loader:
        moved = move_batch(batch, device)
        out = model(moved["image"], grl_lambda=0.0, domain_idx=moved.get("domain_idx"))
        evidence = out.get("evidence_field")
        if not evidence:
            raise RuntimeError("Checkpoint/config does not enable model.use_evidence_field.")

        risk_pred = out["logits"]["risk"].argmax(dim=1).detach().cpu().numpy()
        friction_pred = out["logits"]["friction"].argmax(dim=1).detach().cpu().numpy()
        risk_true = moved["labels"]["risk"].detach().cpu().numpy()
        friction_true = moved["labels"]["friction"].detach().cpu().numpy()
        risk_mask = moved["masks"]["risk"].detach().cpu().numpy().astype(bool)
        friction_mask = moved["masks"]["friction"].detach().cpu().numpy().astype(bool)
        pred_interval = out["mu_interval"].detach().cpu().numpy()
        target_interval = moved["mu_interval"].detach().cpu().numpy()
        mu_mask = moved["mu_mask"].detach().cpu().numpy().astype(bool)

        for i in range(moved["image"].size(0)):
            diag = _attention_diagnostics(evidence, i)
            if "road_mask" in moved:
                diag.update(_attention_mask_diagnostics(evidence["attention"], moved["road_mask"], i))
            risk_known = bool(risk_mask[i])
            friction_known = bool(friction_mask[i])
            risk_correct = bool(risk_known and int(risk_pred[i]) == int(risk_true[i]))
            friction_correct = bool(friction_known and int(friction_pred[i]) == int(friction_true[i]))
            low_true = bool(risk_known and int(risk_true[i]) >= RISK.index("high"))
            low_pred = bool(int(risk_pred[i]) >= RISK.index("high"))
            interval_covers = None
            interval_width = None
            if bool(mu_mask[i]):
                interval_covers = bool(
                    pred_interval[i, 0] <= target_interval[i, 0]
                    and pred_interval[i, 1] >= target_interval[i, 1]
                )
                interval_width = float(pred_interval[i, 1] - pred_interval[i, 0])
            records.append(
                {
                    "image_path": batch["image_path"][i],
                    "dataset": batch["dataset"][i],
                    "group_key": batch["group_key"][i],
                    "risk_known": risk_known,
                    "risk_correct": risk_correct,
                    "friction_known": friction_known,
                    "friction_correct": friction_correct,
                    "true_risk": RISK[int(risk_true[i])] if risk_known else None,
                    "pred_risk": RISK[int(risk_pred[i])],
                    "true_friction": FRICTION_STATES[int(friction_true[i])] if friction_known else None,
                    "pred_friction": FRICTION_STATES[int(friction_pred[i])],
                    "low_friction_true": low_true,
                    "low_friction_pred": low_pred,
                    "mu_known": bool(mu_mask[i]),
                    "interval_covers": interval_covers,
                    "interval_width": interval_width,
                    **diag,
                }
            )
    return records


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
    result = {
        "attention_bottom_half_mass": float(attn[bottom_half].sum()),
        "attention_bottom_third_mass": float(attn[bottom_third].sum()),
        "attention_center_bottom_mass": float(attn[center_bottom].sum()),
        "attention_top_half_mass": float(attn[top_half].sum()),
    }
    entropy = evidence.get("attention_entropy")
    if entropy is not None:
        result["attention_entropy"] = float(entropy[i].detach().cpu())
    for key in ["road_likelihood", "contact_prior"]:
        value = evidence.get(key)
        if value is None:
            continue
        value_map = value[i, 0].detach().cpu()
        if value_map.shape != attn.shape:
            value_map = F.interpolate(value[i : i + 1], size=attn.shape, mode="bilinear", align_corners=False)[0, 0]
        result[f"attention_weighted_{key}_mean"] = float((attn * value_map.detach().cpu()).sum())
    return result


def _attention_mask_diagnostics(
    attention: torch.Tensor,
    road_mask: torch.Tensor,
    i: int,
) -> dict[str, float]:
    attn = attention[i : i + 1]
    mask = road_mask[i : i + 1].to(device=attn.device, dtype=attn.dtype)
    if mask.shape[-2:] != attn.shape[-2:]:
        mask = F.interpolate(mask, size=attn.shape[-2:], mode="bilinear", align_corners=False)
    if mask.size(1) != 1:
        mask = mask.mean(dim=1, keepdim=True)
    mask = mask.clamp(0.0, 1.0)
    attn_norm = attn / attn.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    hard = (mask >= 0.5).to(dtype=attn_norm.dtype)
    return {
        "attention_weighted_external_road_mask_mean": float((attn_norm * mask).sum().detach().cpu()),
        "attention_external_road_mask_hard_mass": float((attn_norm * hard).sum().detach().cpu()),
        "external_road_mask_area": float(hard.mean().detach().cpu()),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "all": records,
        "risk_success": [r for r in records if r["risk_known"] and r["risk_correct"]],
        "risk_failure": [r for r in records if r["risk_known"] and not r["risk_correct"]],
        "low_friction_success": [r for r in records if r["low_friction_true"] and r["risk_correct"]],
        "low_friction_failure": [r for r in records if r["low_friction_true"] and not r["risk_correct"]],
    }
    for dataset in sorted({str(r["dataset"]) for r in records}):
        groups[f"dataset::{dataset}"] = [r for r in records if str(r["dataset"]) == dataset]
        groups[f"dataset::{dataset}::risk_failure"] = [
            r for r in records if str(r["dataset"]) == dataset and r["risk_known"] and not r["risk_correct"]
        ]

    summary = {name: _group_summary(items) for name, items in groups.items()}
    return {
        "num_records": len(records),
        "summary": summary,
        "examples": {
            "lowest_bottom_mass_failures": _examples(
                [r for r in records if r["risk_known"] and not r["risk_correct"]],
                key="attention_bottom_half_mass",
                reverse=False,
            ),
            "highest_top_mass_failures": _examples(
                [r for r in records if r["risk_known"] and not r["risk_correct"]],
                key="attention_top_half_mass",
                reverse=True,
            ),
            "roadsaw_failures": _examples(
                [r for r in records if r["dataset"] == "roadsaw" and r["risk_known"] and not r["risk_correct"]],
                key="attention_bottom_half_mass",
                reverse=False,
            ),
        },
    }


def _group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "attention_bottom_half_mass",
        "attention_bottom_third_mass",
        "attention_center_bottom_mass",
        "attention_top_half_mass",
        "attention_entropy",
        "attention_weighted_road_likelihood_mean",
        "attention_weighted_contact_prior_mean",
        "attention_weighted_external_road_mask_mean",
        "attention_external_road_mask_hard_mass",
        "external_road_mask_area",
        "interval_width",
    ]
    out: dict[str, Any] = {"num_samples": len(items)}
    if not items:
        return out
    risk_known = [r for r in items if r["risk_known"]]
    friction_known = [r for r in items if r["friction_known"]]
    mu_known = [r for r in items if r["mu_known"]]
    if risk_known:
        out["risk_accuracy"] = float(np.mean([r["risk_correct"] for r in risk_known]))
        out["low_friction_recall_proxy"] = _low_recall(risk_known)
    if friction_known:
        out["friction_accuracy"] = float(np.mean([r["friction_correct"] for r in friction_known]))
    if mu_known:
        covers = [r["interval_covers"] for r in mu_known if r["interval_covers"] is not None]
        if covers:
            out["raw_interval_coverage"] = float(np.mean(covers))
    for field in fields:
        vals = [float(r[field]) for r in items if r.get(field) is not None]
        if vals:
            out[field] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            }
    return out


def _low_recall(items: list[dict[str, Any]]) -> float | None:
    lows = [r for r in items if r["low_friction_true"]]
    if not lows:
        return None
    return float(np.mean([r["low_friction_pred"] for r in lows]))


def _examples(items: list[dict[str, Any]], *, key: str, reverse: bool, n: int = 12) -> list[dict[str, Any]]:
    picked = sorted(items, key=lambda r: float(r.get(key, 0.0)), reverse=reverse)[:n]
    fields = [
        "image_path",
        "dataset",
        "group_key",
        "true_risk",
        "pred_risk",
        "true_friction",
        "pred_friction",
        "attention_bottom_half_mass",
        "attention_center_bottom_mass",
        "attention_top_half_mass",
        "attention_entropy",
        "attention_weighted_external_road_mask_mean",
        "attention_external_road_mask_hard_mass",
    ]
    return [{field: item.get(field) for field in fields} for item in picked]


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EvidenceField Audit",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Split: `{report['split']}`",
        f"- Samples scanned: {report['num_records']}",
        "",
        "## Attention Summary",
        "",
        "| Group | N | risk acc | bottom half | center bottom | top half | entropy | road-likelihood | external road mask |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in report["summary"].items():
        lines.append(
                "| {group} | {n} | {risk} | {bottom} | {center} | {top} | {entropy} | {road} | {external} |".format(
                group=group,
                n=row.get("num_samples", 0),
                risk=_fmt(row.get("risk_accuracy")),
                bottom=_fmt_nested(row, "attention_bottom_half_mass"),
                center=_fmt_nested(row, "attention_center_bottom_mass"),
                top=_fmt_nested(row, "attention_top_half_mass"),
                entropy=_fmt_nested(row, "attention_entropy"),
                road=_fmt_nested(row, "attention_weighted_road_likelihood_mean"),
                external=_fmt_nested(row, "attention_weighted_external_road_mask_mean"),
            )
        )
    lines.extend(["", "## Failure Examples", ""])
    for name, examples in report["examples"].items():
        lines.extend([f"### {name}", ""])
        if not examples:
            lines.append("No examples.")
            lines.append("")
            continue
        lines.append("| Dataset | true risk | pred risk | bottom | top | image |")
        lines.append("|---|---|---|---:|---:|---|")
        for item in examples:
            lines.append(
                "| {dataset} | {true} | {pred} | {bottom} | {top} | `{image}` |".format(
                    dataset=item.get("dataset"),
                    true=item.get("true_risk"),
                    pred=item.get("pred_risk"),
                    bottom=_fmt(item.get("attention_bottom_half_mass")),
                    top=_fmt(item.get("attention_top_half_mass")),
                    image=item.get("image_path"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _fmt_nested(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if isinstance(value, dict):
        return _fmt(value.get("mean"))
    return _fmt(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()

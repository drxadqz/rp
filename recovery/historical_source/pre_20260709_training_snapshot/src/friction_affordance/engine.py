from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.losses import compute_total_loss, prediction_consistency_loss
from friction_affordance.metrics import average_dicts, batch_metrics
from friction_affordance.models import FrictionAffordanceModel
from friction_affordance.transforms import build_mask_transforms, build_transforms
from friction_affordance.utils import AverageMeter


def build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 224))
    aug_cfg = data_cfg.get("augmentation") or {}
    load_road_masks = bool(data_cfg.get("load_road_masks", False))
    if load_road_masks and float(aug_cfg.get("horizontal_flip_p", 0.5)) > 0:
        raise ValueError(
            "External road_mask supervision requires aligned image/mask geometry. "
            "Set data.augmentation.horizontal_flip_p: 0.0 for road-mask runs."
        )
    mask_transform = None
    if load_road_masks:
        mask_transform = build_mask_transforms(
            image_size,
            aug_cfg,
            pretransformed=bool(data_cfg.get("road_mask_pretransformed", False)),
        )
    train_ds = ManifestDataset(
        data_cfg["train_manifests"],
        transform=build_transforms(image_size, train=True, aug_cfg=aug_cfg),
        max_samples=data_cfg.get("max_train_samples"),
        max_samples_per_dataset=data_cfg.get("max_train_samples_per_dataset"),
        max_samples_per_class=data_cfg.get("max_train_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)),
        mask_transform=mask_transform,
        load_road_masks=load_road_masks,
    )
    val_ds = ManifestDataset(
        data_cfg["val_manifests"],
        transform=build_transforms(image_size, train=False, aug_cfg=aug_cfg),
        max_samples=data_cfg.get("max_val_samples"),
        max_samples_per_dataset=data_cfg.get("max_val_samples_per_dataset"),
        max_samples_per_class=data_cfg.get("max_val_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)) + 1,
        mask_transform=mask_transform,
        load_road_masks=load_road_masks,
    )
    num_workers, loader_kwargs = dataloader_worker_settings(data_cfg)

    sampler = None
    shuffle = True
    if bool(data_cfg.get("balanced_sampling", False)):
        weights = _balanced_sampling_weights(
            train_ds.df,
            data_cfg.get("balanced_group_columns", ["dataset", "class_label"]),
            dataset_first=bool(data_cfg.get("balanced_dataset_first", True)),
            overrides=data_cfg.get("balanced_weight_overrides"),
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(data_cfg.get("balanced_num_samples_per_epoch", len(train_ds))),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )
    return train_loader, val_loader


def dataloader_worker_settings(data_cfg: dict[str, Any]) -> tuple[int, dict[str, int | bool]]:
    num_workers = int(data_cfg.get("num_workers", 0))
    allow_windows_workers = os.environ.get("FAF_ALLOW_WINDOWS_DATALOADER_WORKERS", "").lower() in {"1", "true", "yes"}
    # Windows shared-memory backed DataLoader workers can fail with WinError 1455 on long evaluation passes.
    if sys.platform.startswith("win") and num_workers > 0 and not allow_windows_workers:
        num_workers = 0
    loader_kwargs: dict[str, int | bool] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return num_workers, loader_kwargs


def _balanced_sampling_weights(
    df,
    group_columns: list[str],
    dataset_first: bool = True,
    overrides: list[dict[str, Any]] | None = None,
) -> list[float]:
    columns = [col for col in group_columns if col in df.columns]
    if not columns:
        weights = [1.0] * len(df)
        return _apply_weight_overrides(df, weights, overrides)
    if dataset_first and "dataset" in df.columns:
        dataset_group = df["dataset"].astype(str)
        group_key = df[columns].astype(str).agg("||".join, axis=1)
        group_counts = group_key.groupby([dataset_group, group_key]).transform("size").astype(float)
        groups_per_dataset = group_key.groupby(dataset_group).transform("nunique").astype(float)
        dataset_count = max(int(dataset_group.nunique()), 1)
        weights = 1.0 / (float(dataset_count) * groups_per_dataset.clip(lower=1.0) * group_counts.clip(lower=1.0))
        return _apply_weight_overrides(df, weights.tolist(), overrides)
    sizes = df.groupby(columns, dropna=False)[columns[0]].transform("size").astype(float)
    weights = 1.0 / sizes.clip(lower=1.0)
    return _apply_weight_overrides(df, weights.tolist(), overrides)


def _apply_weight_overrides(df, weights: list[float], overrides: list[dict[str, Any]] | None) -> list[float]:
    if not overrides:
        return weights
    out = list(weights)
    for item in overrides:
        if not isinstance(item, dict):
            continue
        where = item.get("where", {})
        if not isinstance(where, dict):
            continue
        multiplier = float(item.get("multiplier", 1.0))
        if multiplier <= 0:
            continue
        keep = _match_rows(df, where)
        if not keep.any():
            continue
        indices = keep[keep].index.tolist()
        for idx in indices:
            out[int(idx)] *= multiplier
    return out


def _match_rows(df, where: dict[str, Any]):
    keep = None
    for column, expected in where.items():
        if column not in df.columns:
            continue
        values = df[column].astype(str).str.lower()
        if isinstance(expected, list):
            options = {str(item).lower() for item in expected}
            cur = values.isin(options)
        else:
            cur = values == str(expected).lower()
        keep = cur if keep is None else (keep & cur)
    if keep is None:
        import pandas as pd

        return pd.Series(False, index=df.index)
    return keep


def build_model(cfg: dict[str, Any]) -> FrictionAffordanceModel:
    model_cfg = cfg.get("model", {})
    return FrictionAffordanceModel(
        backbone=model_cfg.get("backbone", "simple_cnn"),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained=bool(model_cfg.get("pretrained", False)),
        use_physics_branch=bool(model_cfg.get("use_physics_branch", False)),
        physics_dim=int(model_cfg.get("physics_dim", 64)),
        physics_quality_cues=bool(model_cfg.get("physics_quality_cues", False)),
        physics_quality_region_cues=bool(model_cfg.get("physics_quality_region_cues", True)),
        num_domains=int(model_cfg.get("num_domains", 0)),
        use_friction_set=bool(model_cfg.get("use_friction_set", False)),
        friction_set_entropy_expansion=float(model_cfg.get("friction_set_entropy_expansion", 0.10)),
        friction_set_interval_mix=float(model_cfg.get("friction_set_interval_mix", 1.0)),
        use_evidence_field=bool(model_cfg.get("use_evidence_field", False)),
        evidence_dim=int(model_cfg.get("evidence_dim", 64)),
        evidence_hidden_dim=int(model_cfg.get("evidence_hidden_dim", 48)),
        evidence_patch_stride=int(model_cfg.get("evidence_patch_stride", 8)),
        evidence_contact_prior_strength=float(model_cfg.get("evidence_contact_prior_strength", 1.0)),
        evidence_road_likelihood_prior_strength=float(model_cfg.get("evidence_road_likelihood_prior_strength", 0.0)),
        evidence_entropy_expansion=float(model_cfg.get("evidence_entropy_expansion", 0.08)),
        evidence_interval_mix=float(model_cfg.get("evidence_interval_mix", 0.0)),
        evidence_risk_logit_mix=float(model_cfg.get("evidence_risk_logit_mix", 0.0)),
        evidence_region_mixture_cues=bool(model_cfg.get("evidence_region_mixture_cues", False)),
        evidence_region_mixture_expansion=float(model_cfg.get("evidence_region_mixture_expansion", 0.0)),
        evidence_region_mixture_kernel_size=int(model_cfg.get("evidence_region_mixture_kernel_size", 9)),
        evidence_num_queries=int(model_cfg.get("evidence_num_queries", 1)),
        evidence_query_disagreement_expansion=float(model_cfg.get("evidence_query_disagreement_expansion", 0.0)),
        use_domain_adapters=bool(model_cfg.get("use_domain_adapters", False)),
        domain_adapter_scale=float(model_cfg.get("domain_adapter_scale", 0.15)),
        use_feature_mixstyle=bool(model_cfg.get("use_feature_mixstyle", False)),
        feature_mixstyle_p=float(model_cfg.get("feature_mixstyle_p", 0.5)),
        feature_mixstyle_alpha=float(model_cfg.get("feature_mixstyle_alpha", 0.1)),
        feature_mixstyle_groups=int(model_cfg.get("feature_mixstyle_groups", 8)),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    out["image"] = batch["image"].to(device)
    out["labels"] = {k: v.to(device) for k, v in batch["labels"].items()}
    out["masks"] = {k: v.to(device) for k, v in batch["masks"].items()}
    out["mu_interval"] = batch["mu_interval"].to(device)
    out["mu_mask"] = batch["mu_mask"].to(device)
    if "road_mask" in batch:
        out["road_mask"] = batch["road_mask"].to(device)
    if "domain_idx" in batch:
        out["domain_idx"] = batch["domain_idx"].to(device)
    if "group_idx" in batch:
        out["group_idx"] = batch["group_idx"].to(device)
    return out


def train_one_epoch(model, loader, optimizer, device, loss_cfg, scaler=None, use_amp: bool = False) -> dict[str, float]:
    model.train()
    meter = AverageMeter()
    logs = []
    grl_lambda = float(loss_cfg.get("domain_grl_lambda", loss_cfg.get("domain_weight", 0.0)))
    grad_accum_steps = max(int(loss_cfg.get("grad_accum_steps", 1)), 1)
    log_every = int(loss_cfg.get("log_every_steps", 250))
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False, ascii=True), start=1):
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch["image"], grl_lambda=grl_lambda, domain_idx=batch.get("domain_idx"))
            loss, loss_logs = compute_total_loss(outputs, batch, loss_cfg)
            aug_weight = float(loss_cfg.get("aug_consistency_weight", 0.0))
            if aug_weight > 0:
                max_samples = int(loss_cfg.get("aug_consistency_max_samples", batch["image"].size(0)))
                keep = min(max(max_samples, 1), batch["image"].size(0))
                perturbed = _weak_style_perturb_normalized(
                    batch["image"][:keep],
                    strength=float(loss_cfg.get("aug_consistency_strength", 0.08)),
                    noise_std=float(loss_cfg.get("aug_consistency_noise_std", 0.01)),
                    mask_ratio=float(loss_cfg.get("aug_consistency_mask_ratio", 0.0)),
                    mask_block_frac=float(loss_cfg.get("aug_consistency_mask_block_frac", 0.18)),
                    mask_max_blocks=int(loss_cfg.get("aug_consistency_mask_max_blocks", 4)),
                    mask_value=str(loss_cfg.get("aug_consistency_mask_value", "mean")),
                )
                aug_domain_idx = batch.get("domain_idx")
                if aug_domain_idx is not None:
                    aug_domain_idx = aug_domain_idx[:keep]
                aug_outputs = model(perturbed, grl_lambda=grl_lambda, domain_idx=aug_domain_idx)
                attention_mask_mode = str(loss_cfg.get("aug_consistency_attention_mask", "none")).lower()
                attention_mask = None
                if attention_mask_mode in {"batch_road_mask", "external_road_mask"} and "road_mask" in batch:
                    attention_mask = batch["road_mask"][:keep]
                aug_loss, aug_logs = prediction_consistency_loss(
                    aug_outputs,
                    outputs,
                    interval_weight=float(loss_cfg.get("aug_consistency_interval_weight", 1.0)),
                    attention_weight=float(loss_cfg.get("aug_consistency_attention_weight", 0.0)),
                    attention_mask=attention_mask,
                    attention_mask_mode=attention_mask_mode,
                    attention_mask_threshold=float(loss_cfg.get("aug_consistency_attention_mask_threshold", 0.0)),
                    attention_mask_sharpness=float(loss_cfg.get("aug_consistency_attention_mask_sharpness", 12.0)),
                )
                loss = loss + aug_weight * aug_loss
                loss_logs.update(aug_logs)
                loss_logs["loss_total"] = float(loss.detach().cpu())
            backward_loss = loss / float(grad_accum_steps)
        if scaler is not None and use_amp:
            scaler.scale(backward_loss).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), loss_cfg.get("grad_clip_norm", 5.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            backward_loss.backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), loss_cfg.get("grad_clip_norm", 5.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        meter.update(float(loss.detach().cpu()), batch["image"].size(0))
        logs.append({**loss_logs, **batch_metrics(outputs, batch)})
        if log_every > 0 and (step % log_every == 0 or step == len(loader)):
            print(
                f"  train step {step}/{len(loader)} loss={meter.avg:.4f}",
                flush=True,
            )
    return {"loss": meter.avg, **average_dicts(logs)}


def _weak_style_perturb_normalized(
    image: torch.Tensor,
    *,
    strength: float,
    noise_std: float,
    mask_ratio: float = 0.0,
    mask_block_frac: float = 0.18,
    mask_max_blocks: int = 4,
    mask_value: str = "mean",
) -> torch.Tensor:
    """Small camera-style perturbation in ImageNet-normalized space.

    The optional random block mask is a lightweight MIC-style consistency
    perturbation. It masks a few image regions in the weak view so the model
    must keep friction/risk evidence stable without relying on a single brittle
    texture patch or dataset-specific background cue.
    """
    if image.numel() == 0:
        return image
    strength = max(float(strength), 0.0)
    noise_std = max(float(noise_std), 0.0)
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = (image * std + mean).clamp(0.0, 1.0)
    b = rgb.size(0)
    brightness = torch.empty((b, 1, 1, 1), device=rgb.device, dtype=rgb.dtype).uniform_(
        1.0 - strength,
        1.0 + strength,
    )
    contrast = torch.empty((b, 1, 1, 1), device=rgb.device, dtype=rgb.dtype).uniform_(
        1.0 - strength,
        1.0 + strength,
    )
    rgb_mean = rgb.mean(dim=(2, 3), keepdim=True)
    rgb = (rgb - rgb_mean) * contrast + rgb_mean
    rgb = rgb * brightness
    if noise_std > 0:
        rgb = rgb + torch.randn_like(rgb) * noise_std
    rgb = rgb.clamp(0.0, 1.0)
    if mask_ratio > 0:
        rgb = _random_block_mask_rgb(
            rgb,
            mask_ratio=mask_ratio,
            block_frac=mask_block_frac,
            max_blocks=mask_max_blocks,
            mask_value=mask_value,
        )
    return (rgb - mean) / std


def _random_block_mask_rgb(
    rgb: torch.Tensor,
    *,
    mask_ratio: float,
    block_frac: float,
    max_blocks: int,
    mask_value: str,
) -> torch.Tensor:
    """Mask random rectangular regions in RGB space for consistency training."""
    if rgb.ndim != 4 or rgb.numel() == 0:
        return rgb
    ratio = max(0.0, min(float(mask_ratio), 0.75))
    if ratio <= 0:
        return rgb
    b, _, h, w = rgb.shape
    block_frac = max(0.04, min(float(block_frac), 0.75))
    max_blocks = max(int(max_blocks), 1)
    block_h = max(1, min(h, int(round(h * block_frac))))
    block_w = max(1, min(w, int(round(w * block_frac))))
    target_area = max(1, int(round(ratio * h * w)))
    out = rgb.clone()
    mode = str(mask_value or "mean").lower()
    for i in range(b):
        if mode == "zero":
            fill = out.new_zeros((3, 1, 1))
        elif mode == "random":
            fill = torch.rand((3, 1, 1), device=out.device, dtype=out.dtype)
        else:
            fill = out[i].mean(dim=(1, 2), keepdim=True)
        masked = 0
        for _ in range(max_blocks):
            y0 = int(torch.randint(0, max(h - block_h + 1, 1), (1,), device=out.device).item())
            x0 = int(torch.randint(0, max(w - block_w + 1, 1), (1,), device=out.device).item())
            out[i, :, y0 : y0 + block_h, x0 : x0 + block_w] = fill
            masked += block_h * block_w
            if masked >= target_area:
                break
    return out.clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(model, loader, device, loss_cfg) -> dict[str, float]:
    model.eval()
    meter = AverageMeter()
    logs = []
    for batch in tqdm(loader, desc="eval", leave=False, ascii=True):
        batch = move_batch(batch, device)
        outputs = model(batch["image"], grl_lambda=0.0, domain_idx=batch.get("domain_idx"))
        loss, loss_logs = compute_total_loss(outputs, batch, loss_cfg)
        meter.update(float(loss.detach().cpu()), batch["image"].size(0))
        logs.append({**loss_logs, **batch_metrics(outputs, batch)})
    return {"loss": meter.avg, **average_dicts(logs)}


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: dict[str, float], cfg: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )

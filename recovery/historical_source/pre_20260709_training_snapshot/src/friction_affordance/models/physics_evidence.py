from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PhysicsEvidenceTarget(nn.Module):
    """Analytic dense physical evidence fields from normalized road images.

    These targets are weak, fixed, public-data supervision signals. They are not
    treated as ground-truth friction labels. Their role is to nudge early/mid
    features toward evidence that is useful for RSCD's coupled factors:
    obstruction, visible roughness, hidden roughness, thin film, texture erasure,
    dry roughness, masked concrete roughness, film-rough coupling, and wet
    granular evidence.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0,
        )
        self.register_buffer(
            "laplace",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
        )

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        lo = flat.amin(dim=2).view(x.shape[0], x.shape[1], 1, 1)
        hi = flat.amax(dim=2).view(x.shape[0], x.shape[1], 1, 1)
        return (x - lo) / (hi - lo).clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        rgb = (x * std + mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x.to(device=x.device, dtype=x.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(device=x.device, dtype=x.dtype), padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace.to(device=x.device, dtype=x.dtype), padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        grad_norm = self._normalize_map(grad)
        lap_norm = self._normalize_map(lap)
        contrast_norm = self._normalize_map(local_contrast)
        rough_base = torch.clamp(0.42 * grad_norm + 0.34 * lap_norm + 0.24 * contrast_norm, 0.0, 1.0)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = torch.sigmoid((0.42 - value) * 10.0) * torch.sigmoid((0.30 - saturation) * 12.0) * low_texture
        thin_film = torch.clamp(specular + 0.6 * dark_water, 0.0, 1.0) * torch.sigmoid((0.08 - lap) * 22.0)
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        snow_phase = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)

        obstruction = torch.clamp(
            0.40 * thin_film + 0.30 * dark_water + 0.20 * specular + 0.35 * texture_erasure,
            0.0,
            1.0,
        )
        visibility = 1.0 - obstruction
        visible_rough = rough_base * visibility * (1.0 - snow_phase) * (1.0 - marking)
        hidden_rough = rough_base * obstruction * (1.0 - snow_phase)
        concrete_like = (
            torch.sigmoid((value - 0.38) * 8.0)
            * torch.sigmoid((0.82 - value) * 8.0)
            * torch.sigmoid((0.28 - saturation) * 10.0)
            * (1.0 - snow_phase)
            * (1.0 - marking)
        )
        dry_rough = rough_base * (1.0 - obstruction) * concrete_like
        masked_concrete_rough = hidden_rough * concrete_like
        film_rough_coupling = thin_film * rough_base
        granular = torch.sigmoid((local_contrast - 0.040) * 35.0) * torch.sigmoid((saturation - 0.045) * 8.0)
        granular = granular * (1.0 - marking)
        granular_wet = granular * torch.clamp(thin_film + dark_water, 0.0, 1.0)

        return torch.cat(
            [
                obstruction,
                visible_rough,
                hidden_rough,
                thin_film,
                texture_erasure,
                dry_rough,
                masked_concrete_rough,
                film_rough_coupling,
                granular_wet,
            ],
            dim=1,
        ).clamp(0.0, 1.0)


class PhysicsEvidenceMapHeads(nn.Module):
    """Small map-prediction heads for ConvNeXt stage features."""

    def __init__(self, channels_by_stage: dict[str, int], num_fields: int = 9) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {stage: self._make_head(channels, int(num_fields)) for stage, channels in channels_by_stage.items()}
        )

    @staticmethod
    def _make_head(channels: int, num_fields: int) -> nn.Sequential:
        hidden = max(32, min(128, int(channels) // 2))
        return nn.Sequential(
            nn.Conv2d(int(channels), hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, int(num_fields), kernel_size=1),
        )

    def forward(self, stage_maps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            feat = stage_maps.get(name)
            if feat is not None:
                out[name] = head(feat)
        return out


def physics_evidence_loss(
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    stage_weights: dict[str, float] | None = None,
    field_weights: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if stage_weights is None:
        stage_weights = {"early": 1.0, "mid": 0.75, "late": 0.45, "final": 0.30}
    total = target.new_tensor(0.0)
    active = 0.0
    weight_map = None
    weight_norm = None
    if sample_weight is not None:
        weight_map = sample_weight.to(device=target.device, dtype=target.dtype).view(-1, 1, 1, 1)
        weight_norm = weight_map.mean().clamp_min(1e-6)
    field_weight_map = None
    field_weight_norm = None
    if field_weights is not None:
        field_weight_map = field_weights.to(device=target.device, dtype=target.dtype).view(1, -1, 1, 1)
        field_weight_norm = field_weight_map.mean().clamp_min(1e-6)
    for name, pred in predictions.items():
        weight = float(stage_weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        tgt = F.interpolate(target, size=pred.shape[-2:], mode="area")
        per_pixel = F.smooth_l1_loss(torch.sigmoid(pred), tgt, reduction="none")
        if field_weight_map is not None and field_weight_norm is not None:
            per_pixel = per_pixel * field_weight_map / field_weight_norm
        if weight_map is not None and weight_norm is not None:
            per_pixel = per_pixel * weight_map
            loss = per_pixel.mean() / weight_norm
        else:
            loss = per_pixel.mean()
        total = total + weight * loss
        active += weight
    if active <= 0.0:
        return target.new_tensor(0.0)
    return total / active

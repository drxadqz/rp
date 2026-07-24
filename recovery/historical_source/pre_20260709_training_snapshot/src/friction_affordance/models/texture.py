from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class PhysicsTextureBranch(nn.Module):
    """Fixed visual road-state descriptors followed by a small projection.

    The branch keeps the model grounded in low-level evidence that is strongly
    tied to road friction affordance: brightness/saturation, snow-like whiteness,
    specular wet highlights, dark water-like regions, edge roughness, and soft
    connectedness of wet/snow regions.
    """

    def __init__(
        self,
        out_dim: int = 64,
        quality_cues: bool = False,
        quality_region_cues: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.quality_cues = bool(quality_cues)
        self.quality_region_cues = bool(quality_region_cues)
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
        self.num_stats = 42 if self.quality_cues else 18
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(18, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        if self.quality_cues:
            self.proj[1] = nn.Linear(self.num_stats, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()

        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = torch.sigmoid((0.38 - value) * 10.0) * torch.sigmoid((0.45 - grad) * 12.0)
        edge_soft = torch.sigmoid((grad - 0.08) * 20.0)

        wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)
        snow_conn = _soft_connectedness(snow_like)
        wet_conn = _soft_connectedness(wet_proxy)

        stats = [
            gray.mean(dim=(2, 3)),
            gray.std(dim=(2, 3)),
            saturation.mean(dim=(2, 3)),
            saturation.std(dim=(2, 3)),
            value.mean(dim=(2, 3)),
            value.std(dim=(2, 3)),
            grad.mean(dim=(2, 3)),
            grad.std(dim=(2, 3)),
            lap.mean(dim=(2, 3)),
            lap.std(dim=(2, 3)),
            snow_like.mean(dim=(2, 3)),
            specular.mean(dim=(2, 3)),
            dark_water.mean(dim=(2, 3)),
            edge_soft.mean(dim=(2, 3)),
            snow_conn,
            wet_conn,
            wet_proxy.mean(dim=(2, 3)),
            rgb[:, 2:3].mean(dim=(2, 3)) - rgb[:, 0:1].mean(dim=(2, 3)),
        ]
        if self.quality_cues:
            stats.extend(
                _quality_and_region_cues(
                    gray=gray,
                    value=value,
                    saturation=saturation,
                    grad=grad,
                    lap=lap,
                    snow_like=snow_like,
                    specular=specular,
                    dark_water=dark_water,
                    wet_proxy=wet_proxy,
                    region_cues=self.quality_region_cues,
                )
            )
        return self.proj(torch.cat(stats, dim=1))


class LocalPhysicsFieldBranch(nn.Module):
    """Weak segmentation-style local physics evidence for road patches.

    Instead of predicting a pixel mask, this branch builds differentiable soft
    evidence maps for specular water, dark smooth water, thin film, texture loss,
    rough gradients, and local contrast. A small grid summary preserves where
    evidence is concentrated inside the patch while avoiding brittle bottom-road
    priors that do not hold for RSCD crops.
    """

    def __init__(self, out_dim: int = 64, grid_size: int = 3) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.grid_size = int(grid_size)
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
        self.num_fields = 8
        self.num_stats = self.num_fields * (self.grid_size * self.grid_size + 4)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * low_texture
        )
        thin_film = torch.clamp(specular + 0.6 * dark_water, 0.0, 1.0) * torch.sigmoid((0.08 - lap) * 22.0)
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        rough_energy = torch.sigmoid((grad - 0.075) * 22.0)
        bright_marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)

        fields = [
            specular,
            dark_water,
            thin_film,
            texture_erasure,
            low_texture,
            rough_energy,
            low_contrast,
            bright_marking,
        ]
        stats = []
        for field in fields:
            grid = F.adaptive_avg_pool2d(field, (self.grid_size, self.grid_size)).flatten(1)
            stats.extend(
                [
                    grid,
                    field.mean(dim=(2, 3)),
                    field.std(dim=(2, 3)),
                    field.amax(dim=(2, 3)),
                    _soft_connectedness(field if field is not bright_marking else wet_proxy),
                ]
            )
        return self.proj(torch.cat(stats, dim=1))


class VisibilityObservedRoughnessBranch(nn.Module):
    """Optical visibility and roughness coupling descriptors for RSCD.

    Water film, specular highlights, and dark smooth puddles can hide concrete
    and asphalt roughness. This branch separates latent roughness evidence from
    the part that remains visible under optical obstruction, then summarizes the
    resulting fields with patch-invariant distribution and topology statistics.
    """

    def __init__(self, out_dim: int = 64, grid_size: int = 3) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.grid_size = int(grid_size)
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
        self.register_buffer("topology_thresholds", torch.linspace(0.18, 0.82, 6).view(1, 1, 6, 1, 1))
        self.num_fields = 9
        self.num_field_stats = self.grid_size * self.grid_size + 8 + 6
        self.num_global_stats = 14
        self.num_stats = self.num_fields * self.num_field_stats + self.num_global_stats
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def _field_stats(self, field: torch.Tensor) -> torch.Tensor:
        grid = F.adaptive_avg_pool2d(field, (self.grid_size, self.grid_size)).flatten(1)
        return torch.cat(
            [
                grid,
                field.mean(dim=(2, 3)),
                field.std(dim=(2, 3)),
                field.amax(dim=(2, 3)),
                _top_fraction_mean(field, fraction=0.05),
                _top_fraction_mean(field, fraction=0.15),
                _soft_connectedness(field),
                _mask_entropy(field.clamp(0.0, 1.0)),
                (field.flatten(1) > 0.50).to(dtype=field.dtype).mean(dim=1, keepdim=True),
                _soft_euler_curve_stats(_normalize_map(field), self.topology_thresholds),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 128:
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        blue_red = rgb[:, 2:3] - rgb[:, 0:1]

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        grad_norm = _normalize_map(grad)
        lap_norm = _normalize_map(lap)
        contrast_norm = _normalize_map(local_contrast)
        rough_base = torch.clamp(0.42 * grad_norm + 0.34 * lap_norm + 0.24 * contrast_norm, 0.0, 1.0)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * low_texture
        )
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
        granular = (
            torch.sigmoid((local_contrast - 0.040) * 35.0)
            * torch.sigmoid((saturation - 0.045) * 8.0)
            * (1.0 - marking)
        )
        granular_wet = granular * torch.clamp(thin_film + dark_water, 0.0, 1.0)

        fields = [
            obstruction,
            visible_rough,
            hidden_rough,
            thin_film,
            texture_erasure,
            dry_rough,
            masked_concrete_rough,
            film_rough_coupling,
            granular_wet,
        ]
        field_stats = [self._field_stats(field) for field in fields]
        global_stats = torch.cat(
            [
                gray.mean(dim=(2, 3)),
                gray.std(dim=(2, 3)),
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                blue_red.mean(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                rough_base.mean(dim=(2, 3)),
                visibility.mean(dim=(2, 3)),
                _top_fraction_mean(obstruction, fraction=0.10),
                _top_fraction_mean(hidden_rough, fraction=0.10),
            ],
            dim=1,
        )
        return self.proj(torch.cat([*field_stats, global_stats], dim=1))


class RelationConditionedPhysicsExpertBranch(nn.Module):
    """Relation-specific physical evidence experts for compositional RSCD labels.

    RSCD hard labels are not independent symbols. Most failures are boundaries
    that share two factors and differ in one factor: wet versus water film,
    smooth/slight/severe roughness, or mud versus gravel material. This branch
    keeps those relations separate before classification. Each expert sees the
    same image, but receives different physical fields and topology summaries.
    """

    def __init__(self, out_dim: int = 72, expert_dim: int = 24, grid_size: int = 3) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.expert_dim = int(expert_dim)
        self.grid_size = int(grid_size)
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
        self.register_buffer("topology_thresholds", torch.linspace(0.15, 0.85, 8).view(1, 1, 8, 1, 1))
        self.num_field_stats = self.grid_size * self.grid_size + 4 + 6
        self.num_global_stats = 8
        self.num_expert_stats = 3 * self.num_field_stats + self.num_global_stats

        def make_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(self.num_expert_stats),
                nn.Linear(self.num_expert_stats, self.expert_dim),
                nn.GELU(),
                nn.Linear(self.expert_dim, self.expert_dim),
            )

        self.friction_expert = make_expert()
        self.roughness_expert = make_expert()
        self.material_expert = make_expert()
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * self.num_expert_stats),
            nn.Linear(3 * self.num_expert_stats, max(16, self.expert_dim)),
            nn.GELU(),
            nn.Linear(max(16, self.expert_dim), 3),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(3 * self.expert_dim + 3),
            nn.Linear(3 * self.expert_dim + 3, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def _field_stats(self, field: torch.Tensor) -> torch.Tensor:
        grid = F.adaptive_avg_pool2d(field, (self.grid_size, self.grid_size)).flatten(1)
        topo = _soft_euler_curve_stats(_normalize_map(field), self.topology_thresholds)
        return torch.cat(
            [
                grid,
                field.mean(dim=(2, 3)),
                field.std(dim=(2, 3)),
                field.amax(dim=(2, 3)),
                _soft_connectedness(field),
                topo,
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 128:
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        blue_red = rgb[:, 2:3] - rgb[:, 0:1]

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        grad_norm = _normalize_map(grad)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        lap_norm = _normalize_map(lap)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)
        contrast_norm = _normalize_map(local_contrast)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * low_texture
        )
        thin_film = torch.clamp(specular + 0.6 * dark_water, 0.0, 1.0) * torch.sigmoid((0.08 - lap) * 22.0)
        wet_proxy = torch.clamp(specular + 0.5 * dark_water + 0.35 * thin_film, 0.0, 1.0)
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        rough_energy = torch.sigmoid((grad - 0.075) * 22.0)
        rough_fragment = torch.clamp(0.60 * grad_norm + 0.40 * contrast_norm, 0.0, 1.0)
        granular = rough_energy * torch.sigmoid((local_contrast - 0.035) * 35.0)
        mud_smooth = low_texture * low_contrast * torch.sigmoid((0.48 - value) * 6.0)

        global_stats = torch.cat(
            [
                gray.mean(dim=(2, 3)),
                gray.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                blue_red.mean(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
            ],
            dim=1,
        )

        friction_stats = torch.cat(
            [self._field_stats(wet_proxy), self._field_stats(thin_film), self._field_stats(texture_erasure), global_stats],
            dim=1,
        )
        roughness_stats = torch.cat(
            [self._field_stats(rough_energy), self._field_stats(rough_fragment), self._field_stats(lap_norm), global_stats],
            dim=1,
        )
        material_stats = torch.cat(
            [self._field_stats(granular), self._field_stats(mud_smooth), self._field_stats(contrast_norm), global_stats],
            dim=1,
        )

        gate_logits = self.gate(torch.cat([friction_stats, roughness_stats, material_stats], dim=1))
        gate = torch.softmax(gate_logits, dim=1)
        experts = torch.cat(
            [
                self.friction_expert(friction_stats) * gate[:, 0:1],
                self.roughness_expert(roughness_stats) * gate[:, 1:2],
                self.material_expert(material_stats) * gate[:, 2:3],
                gate,
            ],
            dim=1,
        )
        return self.out_proj(experts)


class TopologicalTextureBranch(nn.Module):
    """Euler-curve summaries of friction-relevant texture evidence.

    The branch turns wet, snow, low-texture, and gradient maps into soft binary
    filtrations. For each threshold it estimates the Euler characteristic from
    differentiable 2x2 cell counts, giving the classifier a compact description
    of whether evidence appears as connected films, fragmented granules, or
    hole-rich rough texture.
    """

    def __init__(self, out_dim: int = 48, image_size: int = 96) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.image_size = int(image_size)
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
        self.register_buffer("thresholds", torch.linspace(0.15, 0.85, 8).view(1, 1, 8, 1, 1))
        self.num_stats = 5 * 6
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) != self.image_size:
            rgb = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        grad_norm = _normalize_map(grad)

        snow_like = torch.sigmoid((maxc - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        specular = torch.sigmoid((maxc - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = torch.sigmoid((0.38 - maxc) * 10.0) * torch.sigmoid((0.12 - grad) * 30.0)
        wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)
        low_texture = torch.sigmoid((0.045 - grad) * 35.0)

        stats = []
        for field in [gray, grad_norm, snow_like, wet_proxy, low_texture]:
            stats.append(_soft_euler_curve_stats(field, self.thresholds))
        return self.proj(torch.cat(stats, dim=1))


class AntiHumanTextureBranch(nn.Module):
    """Patch-level tail cues for visual evidence humans often underweight.

    Human observers are strongly affected by brightness adaptation, shadows, and
    global scene context. RSCD is a cropped road patch dataset, so this branch
    ignores vertical contact priors and summarizes only distribution tails of
    low-saturation glare, dark smooth water, thin films, and texture loss.
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
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
        self.num_stats = 10 * 7
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        chroma = rgb / rgb.sum(dim=1, keepdim=True).clamp_min(1e-4)
        blue_red = (chroma[:, 2:3] - chroma[:, 0:1]).abs()

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        near_white = torch.sigmoid((maxc - 0.88) * 16.0) * torch.sigmoid((0.22 - saturation) * 14.0)
        specular = torch.sigmoid((maxc - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_smooth = (
            torch.sigmoid((0.42 - maxc) * 10.0)
            * torch.sigmoid((0.26 - saturation) * 12.0)
            * low_texture
        )
        thin_film = torch.clamp(specular + 0.6 * dark_smooth, 0.0, 1.0) * torch.sigmoid((0.08 - lap) * 22.0)
        mirror_smooth = specular * low_contrast
        hidden_water = dark_smooth * low_contrast
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        chroma_ambiguity = blue_red * torch.sigmoid((0.18 - saturation) * 12.0)

        fields = [
            near_white,
            specular,
            dark_smooth,
            thin_film,
            mirror_smooth,
            hidden_water,
            texture_erasure,
            low_texture,
            low_contrast,
            chroma_ambiguity,
        ]
        return self.proj(torch.cat([_tail_stats(field) for field in fields], dim=1))


class RetinexTextureBranch(nn.Module):
    """Illumination-invariant texture cues for wet and low-contrast road states.

    Human vision is strong at global scene understanding, but it can miss
    friction-relevant cues when illumination dominates the appearance: wet films
    can look like dark shadows, specular water can look like dry bright concrete,
    and smooth low-texture asphalt can hide under color cast. This branch uses
    Retinex-style local reflectance, normalized chromaticity, and optional
    vertical region cues to expose evidence that is deliberately less tied to
    global brightness and camera style.
    """

    def __init__(self, out_dim: int = 48, region_cues: bool = True) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.region_cues = bool(region_cues)
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
        self.num_stats = 48
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        local_rgb = F.avg_pool2d(rgb, kernel_size=31, stride=1, padding=15).clamp_min(1e-4)
        reflectance = torch.log(rgb.clamp_min(1e-4)) - torch.log(local_rgb)
        refl_luma = 0.299 * reflectance[:, 0:1] + 0.587 * reflectance[:, 1:2] + 0.114 * reflectance[:, 2:3]
        illumination = torch.log(local_rgb.mean(dim=1, keepdim=True))

        chroma = rgb / rgb.sum(dim=1, keepdim=True).clamp_min(1e-4)
        chroma_r = chroma[:, 0:1]
        chroma_g = chroma[:, 1:2]
        chroma_b = chroma[:, 2:3]
        chroma_bg = chroma_b - chroma_r

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        rgx = F.conv2d(refl_luma, self.sobel_x, padding=1)
        rgy = F.conv2d(refl_luma, self.sobel_y, padding=1)
        refl_grad = torch.sqrt(rgx.square() + rgy.square() + 1e-6)
        refl_lap = F.conv2d(refl_luma, self.laplace, padding=1).abs()
        igx = F.conv2d(illumination, self.sobel_x, padding=1)
        igy = F.conv2d(illumination, self.sobel_y, padding=1)
        illum_grad = torch.sqrt(igx.square() + igy.square() + 1e-6)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_reflectance_texture = torch.sigmoid((0.040 - refl_grad) * 38.0)
        specular_invariant = (
            torch.sigmoid((value - 0.80) * 14.0)
            * torch.sigmoid((0.24 - saturation) * 12.0)
            * low_reflectance_texture
        )
        shadow_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * low_texture
        )
        film_smooth = (
            torch.sigmoid((0.10 - lap) * 18.0)
            * torch.sigmoid((0.055 - grad) * 30.0)
            * torch.sigmoid((0.24 - saturation) * 10.0)
        )
        matte_dry = torch.sigmoid((refl_grad - 0.035) * 28.0) * torch.sigmoid((saturation - 0.08) * 10.0)
        texture_loss = low_texture * low_reflectance_texture
        retinex_edges = refl_grad / (grad + refl_grad + illum_grad + 1e-4)

        global_stats = [
            refl_luma.mean(dim=(2, 3)),
            refl_luma.std(dim=(2, 3)),
            refl_grad.mean(dim=(2, 3)),
            refl_grad.std(dim=(2, 3)),
            refl_lap.mean(dim=(2, 3)),
            refl_lap.std(dim=(2, 3)),
            illumination.mean(dim=(2, 3)),
            illumination.std(dim=(2, 3)),
            illum_grad.mean(dim=(2, 3)),
            illum_grad.std(dim=(2, 3)),
            chroma_r.mean(dim=(2, 3)),
            chroma_r.std(dim=(2, 3)),
            chroma_g.mean(dim=(2, 3)),
            chroma_g.std(dim=(2, 3)),
            chroma_b.mean(dim=(2, 3)),
            chroma_b.std(dim=(2, 3)),
            chroma_bg.mean(dim=(2, 3)),
            chroma_bg.std(dim=(2, 3)),
            saturation.mean(dim=(2, 3)),
            saturation.std(dim=(2, 3)),
            value.mean(dim=(2, 3)),
            value.std(dim=(2, 3)),
            grad.mean(dim=(2, 3)),
            grad.std(dim=(2, 3)),
            specular_invariant.mean(dim=(2, 3)),
            specular_invariant.std(dim=(2, 3)),
            shadow_water.mean(dim=(2, 3)),
            shadow_water.std(dim=(2, 3)),
            film_smooth.mean(dim=(2, 3)),
            film_smooth.std(dim=(2, 3)),
            matte_dry.mean(dim=(2, 3)),
            matte_dry.std(dim=(2, 3)),
            texture_loss.mean(dim=(2, 3)),
            texture_loss.std(dim=(2, 3)),
            retinex_edges.mean(dim=(2, 3)),
            retinex_edges.std(dim=(2, 3)),
        ]
        if self.region_cues:
            spatial_stats = [
                _region_mean(specular_invariant, y0=0.50, y1=1.00)
                - _region_mean(specular_invariant, y0=0.00, y1=0.50),
                _region_mean(shadow_water, y0=0.50, y1=1.00) - _region_mean(shadow_water, y0=0.00, y1=0.50),
                _region_mean(film_smooth, y0=0.50, y1=1.00) - _region_mean(film_smooth, y0=0.00, y1=0.50),
                _region_mean(texture_loss, y0=0.50, y1=1.00) - _region_mean(texture_loss, y0=0.00, y1=0.50),
                _region_mean(refl_grad, y0=0.50, y1=1.00) - _region_mean(refl_grad, y0=0.00, y1=0.50),
                _region_mean(illum_grad, y0=0.50, y1=1.00) - _region_mean(illum_grad, y0=0.00, y1=0.50),
            ]
        else:
            spatial_stats = [
                _top_fraction_mean(specular_invariant, fraction=0.25),
                _top_fraction_mean(shadow_water, fraction=0.25),
                _top_fraction_mean(film_smooth, fraction=0.25),
                _top_fraction_mean(texture_loss, fraction=0.25),
                _top_fraction_mean(refl_grad, fraction=0.25),
                _top_fraction_mean(illum_grad, fraction=0.25),
            ]
        tail_stats = [
            _top_fraction_mean(specular_invariant, fraction=0.10),
            _top_fraction_mean(shadow_water, fraction=0.10),
            _top_fraction_mean(film_smooth, fraction=0.10),
            _top_fraction_mean(texture_loss, fraction=0.10),
            _soft_connectedness(film_smooth),
            _soft_connectedness(shadow_water),
        ]
        stats = global_stats + spatial_stats + tail_stats
        return self.proj(torch.cat(stats, dim=1))


def _soft_connectedness(mask: torch.Tensor) -> torch.Tensor:
    pooled = F.avg_pool2d(mask, kernel_size=9, stride=1, padding=4)
    mass = mask.mean(dim=(2, 3)).clamp_min(1e-4)
    clustered = (pooled.square() * mask).mean(dim=(2, 3))
    return clustered / mass


def _quality_and_region_cues(
    *,
    gray: torch.Tensor,
    value: torch.Tensor,
    saturation: torch.Tensor,
    grad: torch.Tensor,
    lap: torch.Tensor,
    snow_like: torch.Tensor,
    specular: torch.Tensor,
    dark_water: torch.Tensor,
    wet_proxy: torch.Tensor,
    region_cues: bool = True,
) -> list[torch.Tensor]:
    """Differentiable diagnostics for ambiguous wet/overexposed road images.

    These cues are intentionally low-level and sensor-agnostic. For front-view
    driving frames, region ratios expose whether wet/snow evidence lives near
    the lower contact-relevant road area. For RSCD-style close road patches,
    those vertical ratios can be disabled and replaced with position-invariant
    patch-distribution statistics.
    """

    white_hi = torch.sigmoid((value - 0.93) * 18.0) * torch.sigmoid((0.20 - saturation) * 14.0)
    low_texture = torch.sigmoid((0.045 - grad) * 35.0)
    matte_bright = torch.sigmoid((value - 0.78) * 12.0) * torch.sigmoid((0.08 - lap.abs()) * 22.0)
    smooth_bright = (
        torch.sigmoid((value - 0.62) * 10.0)
        * torch.sigmoid((0.18 - saturation) * 12.0)
        * torch.sigmoid((0.055 - grad) * 30.0)
    )
    smooth_dark = (
        torch.sigmoid((0.45 - value) * 10.0)
        * torch.sigmoid((0.25 - saturation) * 12.0)
        * torch.sigmoid((0.045 - grad) * 35.0)
    )
    mirror_candidate = specular * torch.sigmoid((0.07 - grad) * 24.0)
    thin_water = wet_proxy * torch.sigmoid((0.08 - lap.abs()) * 22.0)

    wet_bottom = _region_mean(wet_proxy, y0=0.50, y1=1.00)
    wet_top = _region_mean(wet_proxy, y0=0.00, y1=0.50)
    spec_bottom = _region_mean(specular, y0=0.50, y1=1.00)
    spec_top = _region_mean(specular, y0=0.00, y1=0.50)
    snow_bottom = _region_mean(snow_like, y0=0.50, y1=1.00)
    snow_top = _region_mean(snow_like, y0=0.00, y1=0.50)
    texture_bottom = _region_mean(grad, y0=0.50, y1=1.00)
    texture_top = _region_mean(grad, y0=0.00, y1=0.50)
    brightness_bottom = _region_mean(gray, y0=0.50, y1=1.00)
    brightness_top = _region_mean(gray, y0=0.00, y1=0.50)
    smooth_wet_bottom = _region_mean(smooth_bright + smooth_dark, y0=0.50, y1=1.00)
    smooth_wet_top = _region_mean(smooth_bright + smooth_dark, y0=0.00, y1=0.50)
    mirror_bottom = _region_mean(mirror_candidate, y0=0.50, y1=1.00)
    mirror_top = _region_mean(mirror_candidate, y0=0.00, y1=0.50)
    smooth_dark_bottom = _region_mean(smooth_dark, y0=0.50, y1=1.00)
    smooth_dark_top = _region_mean(smooth_dark, y0=0.00, y1=0.50)

    global_quality = [
        white_hi.mean(dim=(2, 3)),
        white_hi.std(dim=(2, 3)),
        _top_fraction_mean(white_hi, fraction=0.10),
        low_texture.mean(dim=(2, 3)),
        matte_bright.mean(dim=(2, 3)),
        wet_proxy.std(dim=(2, 3)),
        specular.std(dim=(2, 3)),
        dark_water.std(dim=(2, 3)),
    ]
    if region_cues:
        spatial_quality = [
            wet_bottom - wet_top,
            spec_bottom - spec_top,
            snow_bottom - snow_top,
            texture_bottom - texture_top,
            brightness_bottom - brightness_top,
            wet_bottom,
            spec_bottom,
            snow_bottom,
        ]
    else:
        spatial_quality = [
            _top_fraction_mean(wet_proxy, fraction=0.10),
            _top_fraction_mean(specular, fraction=0.10),
            _top_fraction_mean(snow_like, fraction=0.10),
            _top_fraction_mean(grad, fraction=0.10),
            _top_fraction_mean(gray, fraction=0.10),
            _top_fraction_mean(wet_proxy, fraction=0.25),
            _top_fraction_mean(specular, fraction=0.25),
            _top_fraction_mean(snow_like, fraction=0.25),
        ]

    tail_quality = [
        wet_bottom - wet_top,
        smooth_bright.mean(dim=(2, 3)),
        smooth_dark.mean(dim=(2, 3)),
        mirror_candidate.mean(dim=(2, 3)),
        thin_water.mean(dim=(2, 3)),
        thin_water.std(dim=(2, 3)),
    ]
    if region_cues:
        tail_quality = [
            smooth_bright.mean(dim=(2, 3)),
            smooth_dark.mean(dim=(2, 3)),
            mirror_candidate.mean(dim=(2, 3)),
            thin_water.mean(dim=(2, 3)),
            smooth_wet_bottom - smooth_wet_top,
            mirror_bottom - mirror_top,
            smooth_dark_bottom - smooth_dark_top,
            thin_water.std(dim=(2, 3)),
        ]
    else:
        tail_quality = [
            smooth_bright.mean(dim=(2, 3)),
            smooth_dark.mean(dim=(2, 3)),
            mirror_candidate.mean(dim=(2, 3)),
            thin_water.mean(dim=(2, 3)),
            _top_fraction_mean(smooth_bright + smooth_dark, fraction=0.10),
            _top_fraction_mean(mirror_candidate, fraction=0.10),
            _top_fraction_mean(smooth_dark, fraction=0.10),
            thin_water.std(dim=(2, 3)),
        ]
    return global_quality + spatial_quality + tail_quality


def _region_mean(x: torch.Tensor, *, y0: float, y1: float) -> torch.Tensor:
    h = x.size(-2)
    start = int(max(0, min(h - 1, round(float(y0) * h))))
    end = int(max(start + 1, min(h, round(float(y1) * h))))
    return x[:, :, start:end, :].mean(dim=(2, 3))


def _top_fraction_mean(x: torch.Tensor, *, fraction: float) -> torch.Tensor:
    flat = x.flatten(1)
    k = max(1, int(flat.size(1) * float(fraction)))
    return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)


def _tail_stats(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    return torch.cat(
        [
            flat.mean(dim=1, keepdim=True),
            flat.std(dim=1, keepdim=True),
            _top_fraction_mean(x, fraction=0.03),
            _top_fraction_mean(x, fraction=0.10),
            _top_fraction_mean(x, fraction=0.25),
            (flat > 0.50).to(dtype=x.dtype).mean(dim=1, keepdim=True),
            (flat > 0.75).to(dtype=x.dtype).mean(dim=1, keepdim=True),
        ],
        dim=1,
    )


class DirectionalTextureBranch(nn.Module):
    """Fixed multi-scale directional texture cues for road-state classification.

    RSCD-style classes differ not only by color but also by asphalt aggregate,
    cracks, snow streaks, water films, and over-smooth reflections. This branch
    measures those local cues with frozen filters and lets a small projection
    learn how much they should matter for the final classifier.
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
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
            "diag_pos",
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]).view(1, 1, 3, 3) / 2.0,
        )
        self.register_buffer(
            "diag_neg",
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]).view(1, 1, 3, 3) / 2.0,
        )
        self.register_buffer(
            "laplace",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
        )
        self.num_stats = 36
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        gd1 = F.conv2d(gray, self.diag_pos, padding=1)
        gd2 = F.conv2d(gray, self.diag_neg, padding=1)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()

        ax = gx.abs()
        ay = gy.abs()
        ad1 = gd1.abs()
        ad2 = gd2.abs()
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        directional_energy = torch.cat([ax, ay, ad1, ad2], dim=1)
        energy_sum = directional_energy.sum(dim=1, keepdim=True).clamp_min(1e-5)
        dominant = directional_energy.max(dim=1, keepdim=True).values / energy_sum
        anisotropy = (ax - ay).abs() / (ax + ay + 1e-5)

        local_mean_7 = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        local_mean_17 = F.avg_pool2d(gray, kernel_size=17, stride=1, padding=8)
        fine_residual = (gray - local_mean_7).abs()
        coarse_residual = (local_mean_7 - local_mean_17).abs()
        low_texture = torch.sigmoid((0.04 - grad) * 35.0)
        smooth_bright = torch.sigmoid((gray - 0.72) * 12.0) * low_texture * torch.sigmoid((0.22 - saturation) * 12.0)
        dark_smooth = torch.sigmoid((0.42 - gray) * 12.0) * low_texture * torch.sigmoid((0.28 - saturation) * 12.0)
        crack_like = torch.sigmoid((lap - 0.08) * 18.0) * torch.sigmoid((saturation - 0.05) * 8.0)

        stats = [
            grad.mean(dim=(2, 3)),
            grad.std(dim=(2, 3)),
            lap.mean(dim=(2, 3)),
            lap.std(dim=(2, 3)),
            ax.mean(dim=(2, 3)),
            ay.mean(dim=(2, 3)),
            ad1.mean(dim=(2, 3)),
            ad2.mean(dim=(2, 3)),
            dominant.mean(dim=(2, 3)),
            dominant.std(dim=(2, 3)),
            anisotropy.mean(dim=(2, 3)),
            anisotropy.std(dim=(2, 3)),
            fine_residual.mean(dim=(2, 3)),
            fine_residual.std(dim=(2, 3)),
            coarse_residual.mean(dim=(2, 3)),
            coarse_residual.std(dim=(2, 3)),
            low_texture.mean(dim=(2, 3)),
            smooth_bright.mean(dim=(2, 3)),
            dark_smooth.mean(dim=(2, 3)),
            crack_like.mean(dim=(2, 3)),
            _region_mean(grad, y0=0.50, y1=1.00) - _region_mean(grad, y0=0.00, y1=0.50),
            _region_mean(lap, y0=0.50, y1=1.00) - _region_mean(lap, y0=0.00, y1=0.50),
            _region_mean(smooth_bright, y0=0.50, y1=1.00) - _region_mean(smooth_bright, y0=0.00, y1=0.50),
            _region_mean(dark_smooth, y0=0.50, y1=1.00) - _region_mean(dark_smooth, y0=0.00, y1=0.50),
            _region_mean(crack_like, y0=0.50, y1=1.00) - _region_mean(crack_like, y0=0.00, y1=0.50),
            _top_fraction_mean(grad, fraction=0.10),
            _top_fraction_mean(lap, fraction=0.10),
            _top_fraction_mean(fine_residual, fraction=0.10),
            _top_fraction_mean(smooth_bright, fraction=0.10),
            _top_fraction_mean(dark_smooth, fraction=0.10),
            _scale_mean(grad, kernel=3),
            _scale_mean(grad, kernel=9),
            _scale_mean(grad, kernel=21),
            _scale_mean(lap, kernel=9),
            _scale_mean(fine_residual, kernel=9),
            _scale_mean(coarse_residual, kernel=17),
        ]
        return self.proj(torch.cat(stats, dim=1))


def _scale_mean(x: torch.Tensor, *, kernel: int) -> torch.Tensor:
    pooled = F.avg_pool2d(x, kernel_size=kernel, stride=kernel, ceil_mode=True)
    return pooled.std(dim=(2, 3))


class WaveletTextureBranch(nn.Module):
    """Haar-wavelet road texture cues inspired by lightweight RSCD models.

    Wetness, snow, gravel, and unevenness often live in the balance between
    low-frequency smooth films and high-frequency aggregate/edge detail. This
    branch keeps fixed Haar-band statistics and lets the classifier learn a
    compact, interpretable projection from those frequency cues.
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.num_stats = 40
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        ll1, lh1, hl1, hh1 = _haar_bands(gray)
        ll2, lh2, hl2, hh2 = _haar_bands(ll1)
        high1 = torch.sqrt(lh1.square() + hl1.square() + hh1.square() + 1e-6)
        high2 = torch.sqrt(lh2.square() + hl2.square() + hh2.square() + 1e-6)
        low_texture = torch.sigmoid((0.035 - high1) * 45.0)
        smooth_bright = torch.sigmoid((ll1 - 0.70) * 12.0) * low_texture
        smooth_dark = torch.sigmoid((0.42 - ll1) * 12.0) * low_texture
        high_ratio = high1 / (ll1.abs() + high1 + 1e-4)
        band_balance = (lh1.abs() - hl1.abs()).abs() / (lh1.abs() + hl1.abs() + 1e-5)

        sat_ll, sat_lh, sat_hl, sat_hh = _haar_bands(saturation)
        val_ll, val_lh, val_hl, val_hh = _haar_bands(value)
        color_smooth = torch.sigmoid((0.18 - sat_ll) * 12.0) * low_texture
        specular_smooth = torch.sigmoid((val_ll - 0.82) * 14.0) * color_smooth
        snow_smooth = torch.sigmoid((val_ll - 0.72) * 12.0) * torch.sigmoid((0.25 - sat_ll) * 12.0)

        stats = [
            ll1.mean(dim=(2, 3)),
            ll1.std(dim=(2, 3)),
            ll2.mean(dim=(2, 3)),
            ll2.std(dim=(2, 3)),
            high1.mean(dim=(2, 3)),
            high1.std(dim=(2, 3)),
            high2.mean(dim=(2, 3)),
            high2.std(dim=(2, 3)),
            lh1.abs().mean(dim=(2, 3)),
            hl1.abs().mean(dim=(2, 3)),
            hh1.abs().mean(dim=(2, 3)),
            lh2.abs().mean(dim=(2, 3)),
            hl2.abs().mean(dim=(2, 3)),
            hh2.abs().mean(dim=(2, 3)),
            high_ratio.mean(dim=(2, 3)),
            high_ratio.std(dim=(2, 3)),
            band_balance.mean(dim=(2, 3)),
            band_balance.std(dim=(2, 3)),
            low_texture.mean(dim=(2, 3)),
            smooth_bright.mean(dim=(2, 3)),
            smooth_dark.mean(dim=(2, 3)),
            specular_smooth.mean(dim=(2, 3)),
            snow_smooth.mean(dim=(2, 3)),
            color_smooth.mean(dim=(2, 3)),
            sat_lh.abs().mean(dim=(2, 3)),
            sat_hl.abs().mean(dim=(2, 3)),
            sat_hh.abs().mean(dim=(2, 3)),
            val_lh.abs().mean(dim=(2, 3)),
            val_hl.abs().mean(dim=(2, 3)),
            val_hh.abs().mean(dim=(2, 3)),
            _region_mean(high1, y0=0.50, y1=1.00) - _region_mean(high1, y0=0.00, y1=0.50),
            _region_mean(low_texture, y0=0.50, y1=1.00) - _region_mean(low_texture, y0=0.00, y1=0.50),
            _region_mean(specular_smooth, y0=0.50, y1=1.00) - _region_mean(specular_smooth, y0=0.00, y1=0.50),
            _region_mean(snow_smooth, y0=0.50, y1=1.00) - _region_mean(snow_smooth, y0=0.00, y1=0.50),
            _top_fraction_mean(high1, fraction=0.10),
            _top_fraction_mean(high_ratio, fraction=0.10),
            _top_fraction_mean(low_texture, fraction=0.10),
            _top_fraction_mean(specular_smooth, fraction=0.10),
            _scale_mean(high1, kernel=3),
            _scale_mean(low_texture, kernel=3),
        ]
        return self.proj(torch.cat(stats, dim=1))


def _haar_bands(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h = x.size(-2) - (x.size(-2) % 2)
    w = x.size(-1) - (x.size(-1) % 2)
    x = x[..., :h, :w]
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.25
    lh = (x00 - x01 + x10 - x11) * 0.25
    hl = (x00 + x01 - x10 - x11) * 0.25
    hh = (x00 - x01 - x10 + x11) * 0.25
    return ll, lh, hl, hh


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(2)
    low = flat.amin(dim=2).view(x.size(0), x.size(1), 1, 1)
    high = flat.amax(dim=2).view(x.size(0), x.size(1), 1, 1)
    return (x - low) / (high - low).clamp_min(1e-6)


def _soft_euler_curve_stats(field: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    masks = torch.sigmoid((field.unsqueeze(2) - thresholds.to(device=field.device, dtype=field.dtype)) * 18.0)
    a = masks[..., :-1, :-1]
    b = masks[..., :-1, 1:]
    c = masks[..., 1:, :-1]
    d = masks[..., 1:, 1:]
    q1 = (
        a * (1.0 - b) * (1.0 - c) * (1.0 - d)
        + (1.0 - a) * b * (1.0 - c) * (1.0 - d)
        + (1.0 - a) * (1.0 - b) * c * (1.0 - d)
        + (1.0 - a) * (1.0 - b) * (1.0 - c) * d
    )
    q3 = (
        (1.0 - a) * b * c * d
        + a * (1.0 - b) * c * d
        + a * b * (1.0 - c) * d
        + a * b * c * (1.0 - d)
    )
    qd = a * d * (1.0 - b) * (1.0 - c) + b * c * (1.0 - a) * (1.0 - d)
    euler = (q1.sum(dim=(-1, -2)) - q3.sum(dim=(-1, -2)) + 2.0 * qd.sum(dim=(-1, -2))) * 0.25
    area = masks.mean(dim=(-1, -2)).squeeze(1)
    euler = euler.squeeze(1) / max(float(field.size(-1) * field.size(-2)), 1.0)
    return torch.cat(
        [
            euler.mean(dim=1, keepdim=True),
            euler.std(dim=1, keepdim=True),
            (euler.amax(dim=1) - euler.amin(dim=1)).unsqueeze(1),
            area.mean(dim=1, keepdim=True),
            area.std(dim=1, keepdim=True),
            (area.amax(dim=1) - area.amin(dim=1)).unsqueeze(1),
        ],
        dim=1,
    )


class PhysicsAttentionBranch(nn.Module):
    """Weak segmentation-style physics attention for road-surface evidence.

    Public RSCD labels do not include pixel masks, so this branch builds soft
    pseudo-masks from physical optics and texture cues. The masks act like
    segmentation queries for snow, wet highlights, dark water, smooth films,
    and rough aggregate; each query pools local evidence instead of treating
    the whole crop as one undifferentiated image.
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
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
        self.num_stats = 54
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()

        snow_mask = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.26 - saturation) * 12.0)
        wet_highlight = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = torch.sigmoid((0.40 - value) * 10.0) * torch.sigmoid((0.055 - grad) * 30.0)
        smooth_film = torch.sigmoid((0.055 - grad) * 30.0) * torch.sigmoid((0.10 - lap) * 18.0)
        rough_aggregate = torch.sigmoid((grad - 0.075) * 18.0) * torch.sigmoid((lap - 0.050) * 16.0)
        contact_prior = _vertical_prior_like(gray, bottom_weight=1.0, top_weight=0.35)
        wet_contact = torch.clamp((wet_highlight + dark_water + smooth_film) * contact_prior, 0.0, 1.0)

        stats = [
            gray.mean(dim=(2, 3)),
            gray.std(dim=(2, 3)),
            saturation.mean(dim=(2, 3)),
            value.mean(dim=(2, 3)),
            grad.mean(dim=(2, 3)),
            lap.mean(dim=(2, 3)),
        ]
        for mask in [snow_mask, wet_highlight, dark_water, smooth_film, rough_aggregate, wet_contact]:
            stats.extend(
                [
                    mask.mean(dim=(2, 3)),
                    _mask_entropy(mask),
                    _masked_mean(gray, mask),
                    _masked_mean(saturation, mask),
                    _masked_mean(value, mask),
                    _masked_mean(grad, mask),
                    _masked_mean(lap, mask),
                ]
            )
        stats.extend(
            [
                _region_mean(snow_mask, y0=0.50, y1=1.00) - _region_mean(snow_mask, y0=0.00, y1=0.50),
                _region_mean(wet_contact, y0=0.50, y1=1.00) - _region_mean(wet_contact, y0=0.00, y1=0.50),
                _region_mean(rough_aggregate, y0=0.50, y1=1.00)
                - _region_mean(rough_aggregate, y0=0.00, y1=0.50),
                _top_fraction_mean(wet_contact, fraction=0.10),
                _top_fraction_mean(snow_mask, fraction=0.10),
                _top_fraction_mean(rough_aggregate, fraction=0.10),
            ]
        )
        return self.proj(torch.cat(stats, dim=1))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = (x * mask).sum(dim=(2, 3))
    mass = mask.sum(dim=(2, 3)).clamp_min(1e-4)
    return weighted / mass


def _mask_entropy(mask: torch.Tensor) -> torch.Tensor:
    p = mask.clamp(1e-5, 1.0 - 1e-5)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return ent.mean(dim=(2, 3))


def _vertical_prior_like(x: torch.Tensor, *, bottom_weight: float, top_weight: float) -> torch.Tensor:
    h = x.size(-2)
    weights = torch.linspace(float(top_weight), float(bottom_weight), h, device=x.device, dtype=x.dtype)
    return weights.view(1, 1, h, 1).expand_as(x)


class SemanticPhysicsAttentionBranch(nn.Module):
    """Patch-invariant weak semantic attention for friction-relevant evidence.

    This is a safer RSCD variant of segmentation-style attention. It avoids
    bottom/contact-zone assumptions and builds soft region proposals from
    optical physics: snow-like whiteness, mirror-like water, dark smooth water,
    thin low-texture films, rough aggregate, granular material, and bright
    marking-like artifacts. Each region pools the same low-level measurements,
    giving the classifier weak semantic evidence without pixel-level masks.
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
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
        self.num_stats = 68
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        snow = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        mirror_water = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * low_texture
        )
        thin_film = torch.clamp(mirror_water + 0.6 * dark_water, 0.0, 1.0) * low_contrast
        rough_aggregate = torch.sigmoid((grad - 0.075) * 22.0) * torch.sigmoid((lap - 0.050) * 18.0)
        granular = torch.sigmoid((local_contrast - 0.045) * 35.0) * torch.sigmoid((saturation - 0.05) * 8.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)

        masks = [snow, mirror_water, dark_water, thin_film, rough_aggregate, granular, marking]
        base_stats = [
            gray.mean(dim=(2, 3)),
            gray.std(dim=(2, 3)),
            saturation.mean(dim=(2, 3)),
            saturation.std(dim=(2, 3)),
            grad.mean(dim=(2, 3)),
            grad.std(dim=(2, 3)),
            lap.mean(dim=(2, 3)),
            local_contrast.mean(dim=(2, 3)),
        ]
        region_stats = []
        for mask in masks:
            region_stats.extend(
                [
                    mask.mean(dim=(2, 3)),
                    _mask_entropy(mask),
                    _masked_mean(gray, mask),
                    _masked_mean(saturation, mask),
                    _masked_mean(value, mask),
                    _masked_mean(grad, mask),
                    _soft_connectedness(mask),
                ]
            )
        interaction_stats = [
            torch.clamp(mirror_water + dark_water + thin_film, 0.0, 1.0).mean(dim=(2, 3)),
            _top_fraction_mean(mirror_water + dark_water + thin_film, fraction=0.05),
            _top_fraction_mean(rough_aggregate, fraction=0.05),
            _top_fraction_mean(granular, fraction=0.05),
            (thin_film * (1.0 - marking)).mean(dim=(2, 3)),
            (rough_aggregate * (1.0 - marking)).mean(dim=(2, 3)),
            (granular * (1.0 - marking)).mean(dim=(2, 3)),
            (marking * grad).mean(dim=(2, 3)),
            (low_texture * low_contrast).mean(dim=(2, 3)),
            torch.clamp(snow + mirror_water + dark_water, 0.0, 1.0).std(dim=(2, 3)),
            (rough_aggregate * low_contrast).mean(dim=(2, 3)),
        ]
        return self.proj(torch.cat(base_stats + region_stats + interaction_stats, dim=1))


class FactorConditionedPhysicsTokenBranch(nn.Module):
    """Factor-query attention over physics evidence tokens.

    Human visual inspection tends to collapse a road patch into one impression:
    bright, dark, rough, wet, or snowy. RSCD labels are more structured than that:
    friction state, material, and unevenness each need different evidence. This
    branch builds weak physics tokens, then lets three learned factor queries
    attend to them separately. It is patch-invariant and uses no pixel labels.
    """

    def __init__(self, out_dim: int = 48, token_dim: int = 16) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.token_dim = int(token_dim)
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
        self.num_token_stats = 10
        self.num_global_stats = 8
        self.num_factors = 3
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.num_token_stats),
            nn.Linear(self.num_token_stats, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.factor_queries = nn.Parameter(torch.empty(self.num_factors, self.token_dim))
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.num_factors * self.token_dim + self.num_factors + self.num_global_stats),
            nn.Linear(self.num_factors * self.token_dim + self.num_factors + self.num_global_stats, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        nn.init.trunc_normal_(self.factor_queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        snow = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        mirror_water = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * low_texture
        )
        thin_film = torch.clamp(mirror_water + 0.6 * dark_water, 0.0, 1.0) * low_contrast
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        rough_aggregate = torch.sigmoid((grad - 0.075) * 22.0) * torch.sigmoid((lap - 0.050) * 18.0)
        granular = torch.sigmoid((local_contrast - 0.045) * 35.0) * torch.sigmoid((saturation - 0.05) * 8.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        dark_rough = rough_aggregate * torch.sigmoid((0.55 - value) * 8.0)

        masks = [
            snow,
            mirror_water,
            dark_water,
            thin_film,
            texture_erasure,
            rough_aggregate,
            granular,
            marking,
            dark_rough,
        ]
        token_rows = []
        for mask in masks:
            token_rows.append(
                torch.cat(
                    [
                        mask.mean(dim=(2, 3)),
                        _mask_entropy(mask),
                        _masked_mean(gray, mask),
                        _masked_mean(saturation, mask),
                        _masked_mean(value, mask),
                        _masked_mean(grad, mask),
                        _masked_mean(lap, mask),
                        _masked_mean(local_contrast, mask),
                        _soft_connectedness(mask),
                        _top_fraction_mean(mask, fraction=0.05),
                    ],
                    dim=1,
                )
            )
        token_stats = torch.stack(token_rows, dim=1)
        token = self.token_proj(token_stats)
        query = F.normalize(self.factor_queries.to(dtype=token.dtype), dim=1)
        key = F.normalize(token, dim=2)
        attention = torch.softmax(torch.einsum("fd,btd->bft", query, key) / math.sqrt(float(self.token_dim)), dim=2)
        factor_token = torch.einsum("bft,btd->bfd", attention, token)
        attn_entropy = -(attention.clamp_min(1e-6) * attention.clamp_min(1e-6).log()).sum(dim=2)
        global_stats = torch.cat(
            [
                gray.mean(dim=(2, 3)),
                gray.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                value.mean(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        return self.out_proj(torch.cat([factor_token.flatten(1), attn_entropy, global_stats], dim=1))


class FactorCoupledPhysicsTokenBranch(nn.Module):
    """Factor-coupled physical evidence tokens for compositional road labels.

    RSCD classes are annotated as friction state, material, and unevenness, but
    the visual evidence is not additive. Wet asphalt, wet concrete, and wet mud
    change reflectance and texture in different ways. This branch first builds
    physically meaningful evidence tokens, then learns single-factor, pairwise,
    and three-way coupling queries over those tokens before the classifier.
    """

    def __init__(self, out_dim: int = 64, token_dim: int = 16) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.token_dim = int(token_dim)
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
        self.num_token_stats = 12
        self.num_global_stats = 10
        self.num_factors = 3
        self.num_pairs = 3
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.num_token_stats),
            nn.Linear(self.num_token_stats, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.factor_queries = nn.Parameter(torch.empty(self.num_factors, self.token_dim))
        self.pair_queries = nn.Parameter(torch.empty(self.num_pairs, self.token_dim))
        self.triple_query = nn.Parameter(torch.empty(1, self.token_dim))
        out_in_dim = (
            self.num_factors * self.token_dim
            + self.num_pairs * self.token_dim
            + self.num_pairs * self.token_dim
            + self.token_dim
            + self.token_dim
            + self.num_factors
            + self.num_pairs
            + 1
            + self.num_global_stats
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(out_in_dim),
            nn.Linear(out_in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        nn.init.trunc_normal_(self.factor_queries, std=0.02)
        nn.init.trunc_normal_(self.pair_queries, std=0.02)
        nn.init.trunc_normal_(self.triple_query, std=0.02)
        last = self.out_proj[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        snow = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        mirror_water = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * low_texture
        )
        thin_film = torch.clamp(mirror_water + 0.6 * dark_water, 0.0, 1.0) * low_contrast
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        rough_aggregate = torch.sigmoid((grad - 0.075) * 22.0) * torch.sigmoid((lap - 0.050) * 18.0)
        granular = torch.sigmoid((local_contrast - 0.045) * 35.0) * torch.sigmoid((saturation - 0.05) * 8.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        dark_rough = rough_aggregate * torch.sigmoid((0.55 - value) * 8.0)
        porous_film = thin_film * rough_aggregate
        wet_granular = torch.clamp(mirror_water + dark_water, 0.0, 1.0) * granular
        dry_microtexture = rough_aggregate * (1.0 - torch.clamp(mirror_water + dark_water + snow, 0.0, 1.0))

        masks = [
            snow,
            mirror_water,
            dark_water,
            thin_film,
            texture_erasure,
            rough_aggregate,
            granular,
            marking,
            dark_rough,
            porous_film,
            wet_granular,
            dry_microtexture,
        ]
        token_rows = []
        for mask in masks:
            token_rows.append(
                torch.cat(
                    [
                        mask.mean(dim=(2, 3)),
                        mask.std(dim=(2, 3)),
                        _mask_entropy(mask),
                        _masked_mean(gray, mask),
                        _masked_mean(saturation, mask),
                        _masked_mean(value, mask),
                        _masked_mean(grad, mask),
                        _masked_mean(lap, mask),
                        _masked_mean(local_contrast, mask),
                        _soft_connectedness(mask),
                        _top_fraction_mean(mask, fraction=0.05),
                        _top_fraction_mean(mask, fraction=0.15),
                    ],
                    dim=1,
                )
            )
        token_stats = torch.stack(token_rows, dim=1)
        token = self.token_proj(token_stats)
        key = F.normalize(token, dim=2)

        factor_query = F.normalize(self.factor_queries.to(dtype=token.dtype), dim=1)
        factor_attn = torch.softmax(
            torch.einsum("fd,btd->bft", factor_query, key) / math.sqrt(float(self.token_dim)),
            dim=2,
        )
        factor_token = torch.einsum("bft,btd->bfd", factor_attn, token)

        pair_query = F.normalize(self.pair_queries.to(dtype=token.dtype), dim=1)
        pair_attn = torch.softmax(
            torch.einsum("pd,btd->bpt", pair_query, key) / math.sqrt(float(self.token_dim)),
            dim=2,
        )
        pair_token = torch.einsum("bpt,btd->bpd", pair_attn, token)

        triple_query = F.normalize(self.triple_query.to(dtype=token.dtype), dim=1)
        triple_attn = torch.softmax(
            torch.einsum("qd,btd->bqt", triple_query, key) / math.sqrt(float(self.token_dim)),
            dim=2,
        )
        triple_token = torch.einsum("bqt,btd->bqd", triple_attn, token).squeeze(1)

        fm = factor_token[:, 0] * factor_token[:, 1]
        fu = factor_token[:, 0] * factor_token[:, 2]
        mu = factor_token[:, 1] * factor_token[:, 2]
        pair_product = torch.stack([fm, fu, mu], dim=1)
        triple_product = factor_token[:, 0] * factor_token[:, 1] * factor_token[:, 2]
        factor_entropy = -(factor_attn.clamp_min(1e-6) * factor_attn.clamp_min(1e-6).log()).sum(dim=2)
        pair_entropy = -(pair_attn.clamp_min(1e-6) * pair_attn.clamp_min(1e-6).log()).sum(dim=2)
        triple_entropy = -(triple_attn.clamp_min(1e-6) * triple_attn.clamp_min(1e-6).log()).sum(dim=2)

        global_stats = torch.cat(
            [
                gray.mean(dim=(2, 3)),
                gray.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                value.mean(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                torch.clamp(mirror_water + dark_water + thin_film, 0.0, 1.0).mean(dim=(2, 3)),
                (rough_aggregate * (1.0 - marking)).mean(dim=(2, 3)),
            ],
            dim=1,
        )
        return self.out_proj(
            torch.cat(
                [
                    factor_token.flatten(1),
                    pair_token.flatten(1),
                    pair_product.flatten(1),
                    triple_token,
                    triple_product,
                    factor_entropy,
                    pair_entropy,
                    triple_entropy,
                    global_stats,
                ],
                dim=1,
            )
        )

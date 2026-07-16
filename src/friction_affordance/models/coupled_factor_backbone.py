from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from friction_affordance.rscd_factors import FACTOR_LABELS, build_rscd_factor_spec


@dataclass(frozen=True)
class CoupledFactorBackboneConfig:
    in_channels: int = 3
    stem_dim: int = 48
    stage_dims: tuple[int, int, int] = (96, 192, 384)
    evidence_channels: int = 12
    num_coupling_experts: int = 5
    dropout: float = 0.15
    coupling_gate_mode: str = "learned"
    use_concrete_roughness_scale_space: bool = False
    concrete_roughness_scale_space_mode: str = "learned"
    concrete_roughness_scale_space_scale: float = 0.18
    use_dual_film_texture_coupling: bool = False
    dual_film_texture_coupling_mode: str = "learned"
    dual_film_texture_coupling_scale: float = 0.16


class FixedRoadEvidenceMaps(nn.Module):
    """Differentiable RSCD evidence maps for film, material, and roughness cues."""

    out_channels = 12

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("laplace", laplace, persistent=False)

    def rgb(self, image: torch.Tensor) -> torch.Tensor:
        return (image * self.std + self.mean).clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.rgb(image)
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        max_rgb = image.amax(dim=1, keepdim=True)
        min_rgb = image.amin(dim=1, keepdim=True)
        sat = (max_rgb - min_rgb) / max_rgb.clamp_min(1e-4)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_sq_mean = F.avg_pool2d(gray.square(), kernel_size=9, stride=1, padding=4)
        local_std = (local_sq_mean - local_mean.square()).clamp_min(0.0).sqrt()
        dark_film = torch.sigmoid((0.34 - gray) * 12.0)
        specular = torch.sigmoid((max_rgb - 0.72) * 12.0) * torch.sigmoid((0.22 - sat) * 10.0)
        wet_proxy = (0.55 * specular + 0.45 * dark_film).clamp(0.0, 1.0)
        rough_proxy = torch.sigmoid((grad + 0.55 * lap - 0.12) * 12.0)
        concrete_proxy = torch.sigmoid((0.16 - sat) * 10.0) * torch.sigmoid((local_std - 0.035) * 14.0)
        asphalt_proxy = torch.sigmoid((sat - 0.035) * 10.0) * torch.sigmoid((0.56 - gray) * 8.0)
        visible_texture_under_film = (local_std * (0.35 + 0.65 * wet_proxy)).clamp(0.0, 1.0)
        texture_to_wet = (local_std / (wet_proxy + 0.08)).clamp(0.0, 4.0) / 4.0
        return torch.cat(
            [
                gray,
                sat,
                grad,
                lap,
                local_std,
                dark_film,
                specular,
                wet_proxy,
                rough_proxy,
                concrete_proxy,
                asphalt_proxy,
                visible_texture_under_film + texture_to_wet,
            ],
            dim=1,
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FactorCouplingGate(nn.Module):
    """Image-level gate for RSCD-specific coupling experts."""

    expert_names = ("water_concrete", "wet_concrete", "dry_concrete", "asphalt_family", "identity")

    def __init__(self, evidence_channels: int, num_experts: int) -> None:
        super().__init__()
        stats_dim = evidence_channels * 2
        self.gate = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_experts),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        mean = evidence.mean(dim=(2, 3))
        std = evidence.std(dim=(2, 3), unbiased=False)
        stats = torch.cat([mean, std], dim=1)
        return torch.softmax(self.gate(stats), dim=1)


class ConcreteRoughnessScaleSpaceConditioner(nn.Module):
    """Early concrete-conditioned roughness route for RSCD slight/severe classes."""

    route_names = ("dry_concrete_roughness", "film_concrete_roughness", "non_concrete_guard")

    def __init__(self, channels: int, *, scale: float = 0.18, mode: str = "learned") -> None:
        super().__init__()
        self.scale = float(scale)
        self.mode = str(mode).lower()
        self.stats_dim = 15
        self.route_gate = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, 48),
            nn.GELU(),
            nn.Linear(48, len(self.route_names)),
        )
        self.route_experts = nn.ModuleList(
            [
                nn.Sequential(
                    DepthwiseSeparableBlock(channels, dilation=1),
                    DepthwiseSeparableBlock(channels, dilation=2),
                ),
                nn.Sequential(
                    DepthwiseSeparableBlock(channels, dilation=2),
                    DepthwiseSeparableBlock(channels, dilation=4),
                ),
                nn.Identity(),
            ]
        )

    @staticmethod
    def _norm_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        lo = flat.amin(dim=1, keepdim=True).view(-1, 1, 1, 1)
        hi = flat.amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo).clamp_min(1e-6)

    @staticmethod
    def _top_mean(x: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(1, int(flat.shape[1] * float(fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def _route_evidence(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gray = evidence[:, 0:1]
        sat = evidence[:, 1:2]
        grad = evidence[:, 2:3]
        lap = evidence[:, 3:4]
        local_std = evidence[:, 4:5]
        wet_proxy = evidence[:, 7:8]
        concrete_proxy = evidence[:, 9:10]
        asphalt_proxy = evidence[:, 10:11]
        texture_under_film = evidence[:, 11:12]

        rough_fine = self._norm_map(0.55 * grad + 0.45 * lap)
        rough_mid = self._norm_map(F.avg_pool2d(rough_fine, kernel_size=5, stride=1, padding=2))
        rough_coarse = self._norm_map(F.avg_pool2d(rough_fine, kernel_size=13, stride=1, padding=6))
        scale_disagreement = (rough_fine - rough_coarse).abs()
        visible_rough = (0.50 * rough_fine + 0.30 * rough_mid + 0.20 * scale_disagreement).clamp(0.0, 1.0)
        hidden_film_rough = (0.35 * rough_mid + 0.35 * rough_coarse + 0.30 * texture_under_film).clamp(0.0, 1.0)

        dry_concrete = concrete_proxy * (1.0 - wet_proxy) * visible_rough
        film_concrete = concrete_proxy * wet_proxy * hidden_film_rough
        non_concrete_guard = (1.0 - concrete_proxy).clamp(0.0, 1.0) * (0.35 + 0.65 * asphalt_proxy)
        route_maps = torch.cat([dry_concrete, film_concrete, non_concrete_guard], dim=1).clamp(0.0, 1.0)
        route_maps = route_maps / route_maps.sum(dim=1, keepdim=True).clamp_min(1e-4)

        stats = torch.cat(
            [
                gray.mean(dim=(2, 3)),
                sat.mean(dim=(2, 3)),
                local_std.mean(dim=(2, 3)),
                wet_proxy.mean(dim=(2, 3)),
                concrete_proxy.mean(dim=(2, 3)),
                dry_concrete.mean(dim=(2, 3)),
                film_concrete.mean(dim=(2, 3)),
                non_concrete_guard.mean(dim=(2, 3)),
                rough_fine.mean(dim=(2, 3)),
                rough_mid.mean(dim=(2, 3)),
                rough_coarse.mean(dim=(2, 3)),
                scale_disagreement.mean(dim=(2, 3)),
                self._top_mean(visible_rough),
                self._top_mean(hidden_film_rough),
                texture_under_film.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        return route_maps, stats

    def forward(
        self,
        x: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        route_maps, stats = self._route_evidence(evidence)
        if self.mode in {"off", "false", "none", "disabled"}:
            route_weights = x.new_zeros((x.shape[0], len(self.route_names)))
            return x, {"route_weights": route_weights, "route_maps": route_maps, "stats": stats}
        if self.mode == "learned":
            route_weights = torch.softmax(self.route_gate(stats), dim=1)
        elif self.mode == "fixed_uniform":
            route_weights = x.new_full((x.shape[0], len(self.route_names)), 1.0 / float(len(self.route_names)))
        elif self.mode == "fixed_physics":
            route_weights = route_maps.mean(dim=(2, 3))
            route_weights = route_weights / route_weights.sum(dim=1, keepdim=True).clamp_min(1e-4)
        else:
            raise ValueError(f"unknown concrete_roughness_scale_space_mode: {self.mode}")
        route_maps_small = F.interpolate(route_maps, size=x.shape[-2:], mode="bilinear", align_corners=False)
        expert_out = torch.stack([expert(x) for expert in self.route_experts], dim=1)
        combined_gate = route_weights[:, :, None, None, None] * route_maps_small[:, :, None, :, :]
        combined_gate = combined_gate / combined_gate.sum(dim=1, keepdim=True).clamp_min(1e-4)
        mixed = (expert_out * combined_gate).sum(dim=1)
        return x + float(self.scale) * (mixed - x), {
            "route_weights": route_weights,
            "route_maps": route_maps,
            "stats": stats,
        }


class DualFilmTextureRoughnessCouplingConditioner(nn.Module):
    """Early dual-law conditioner for dry concrete and wet/water concrete."""

    route_names = ("dry_concrete_visible", "film_concrete_hidden", "asphalt_guard", "identity")

    def __init__(self, channels: int, *, scale: float = 0.16, mode: str = "learned") -> None:
        super().__init__()
        self.scale = float(scale)
        self.mode = str(mode).lower()
        self.stats_dim = 18
        self.route_gate = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, 56),
            nn.GELU(),
            nn.Linear(56, len(self.route_names)),
        )
        self.route_experts = nn.ModuleList(
            [
                nn.Sequential(
                    DepthwiseSeparableBlock(channels, dilation=1),
                    DepthwiseSeparableBlock(channels, dilation=2),
                ),
                nn.Sequential(
                    DepthwiseSeparableBlock(channels, dilation=2),
                    DepthwiseSeparableBlock(channels, dilation=4),
                ),
                nn.Sequential(
                    DepthwiseSeparableBlock(channels, dilation=1),
                ),
                nn.Identity(),
            ]
        )

    @staticmethod
    def _norm_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        lo = flat.amin(dim=1, keepdim=True).view(-1, 1, 1, 1)
        hi = flat.amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo).clamp_min(1e-6)

    @staticmethod
    def _top_mean(x: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(1, int(flat.shape[1] * float(fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def _route_evidence(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gray = evidence[:, 0:1]
        sat = evidence[:, 1:2]
        grad = evidence[:, 2:3]
        lap = evidence[:, 3:4]
        local_std = evidence[:, 4:5]
        dark_film = evidence[:, 5:6]
        specular = evidence[:, 6:7]
        wet_proxy = evidence[:, 7:8]
        concrete_proxy = evidence[:, 9:10]
        asphalt_proxy = evidence[:, 10:11]
        texture_under_film = evidence[:, 11:12]

        fine_texture = self._norm_map(0.45 * grad + 0.35 * lap + 0.20 * local_std)
        meso_texture = self._norm_map(F.avg_pool2d(fine_texture, kernel_size=7, stride=1, padding=3))
        coarse_texture = self._norm_map(F.avg_pool2d(fine_texture, kernel_size=17, stride=1, padding=8))
        rough_dispersion = (fine_texture - meso_texture).abs()
        film_erasure = (wet_proxy * (1.0 - local_std.clamp(0.0, 1.0)) * (0.55 * specular + 0.45 * dark_film)).clamp(0.0, 1.0)
        wet_connected = F.avg_pool2d(wet_proxy, kernel_size=9, stride=1, padding=4)
        hidden_texture = (0.35 * texture_under_film + 0.30 * meso_texture + 0.20 * coarse_texture + 0.15 * film_erasure).clamp(0.0, 1.0)
        visible_roughness = (0.45 * fine_texture + 0.30 * rough_dispersion + 0.25 * meso_texture).clamp(0.0, 1.0)

        dry_concrete_visible = concrete_proxy * (1.0 - wet_proxy) * visible_roughness
        film_concrete_hidden = concrete_proxy * wet_proxy * (0.55 * hidden_texture + 0.45 * wet_connected).clamp(0.0, 1.0)
        asphalt_guard = asphalt_proxy * (0.45 + 0.55 * (1.0 - concrete_proxy).clamp(0.0, 1.0))
        identity = (1.0 - torch.maximum(concrete_proxy, asphalt_proxy)).clamp(0.0, 1.0) * 0.50 + 0.05
        route_maps = torch.cat([dry_concrete_visible, film_concrete_hidden, asphalt_guard, identity], dim=1).clamp(0.0, 1.0)
        route_maps = route_maps / route_maps.sum(dim=1, keepdim=True).clamp_min(1e-4)

        stats = torch.cat(
            [
                gray.mean(dim=(2, 3)),
                sat.mean(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                local_std.mean(dim=(2, 3)),
                wet_proxy.mean(dim=(2, 3)),
                concrete_proxy.mean(dim=(2, 3)),
                asphalt_proxy.mean(dim=(2, 3)),
                dry_concrete_visible.mean(dim=(2, 3)),
                film_concrete_hidden.mean(dim=(2, 3)),
                asphalt_guard.mean(dim=(2, 3)),
                fine_texture.mean(dim=(2, 3)),
                meso_texture.mean(dim=(2, 3)),
                coarse_texture.mean(dim=(2, 3)),
                rough_dispersion.mean(dim=(2, 3)),
                film_erasure.mean(dim=(2, 3)),
                self._top_mean(visible_roughness),
                self._top_mean(hidden_texture),
            ],
            dim=1,
        )
        return route_maps, stats

    def forward(self, x: torch.Tensor, evidence: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        route_maps, stats = self._route_evidence(evidence)
        if self.mode in {"off", "false", "none", "disabled"}:
            route_weights = x.new_zeros((x.shape[0], len(self.route_names)))
            return x, {"route_weights": route_weights, "route_maps": route_maps, "stats": stats}
        if self.mode == "learned":
            route_weights = torch.softmax(self.route_gate(stats), dim=1)
        elif self.mode == "fixed_uniform":
            route_weights = x.new_full((x.shape[0], len(self.route_names)), 1.0 / float(len(self.route_names)))
        elif self.mode == "fixed_physics":
            route_weights = route_maps.mean(dim=(2, 3))
            route_weights = route_weights / route_weights.sum(dim=1, keepdim=True).clamp_min(1e-4)
        else:
            raise ValueError(f"unknown dual_film_texture_coupling_mode: {self.mode}")
        route_maps_small = F.interpolate(route_maps, size=x.shape[-2:], mode="bilinear", align_corners=False)
        expert_out = torch.stack([expert(x) for expert in self.route_experts], dim=1)
        combined_gate = route_weights[:, :, None, None, None] * route_maps_small[:, :, None, :, :]
        combined_gate = combined_gate / combined_gate.sum(dim=1, keepdim=True).clamp_min(1e-4)
        mixed = (expert_out * combined_gate).sum(dim=1)
        return x + float(self.scale) * (mixed - x), {
            "route_weights": route_weights,
            "route_maps": route_maps,
            "stats": stats,
        }


class CoupledFactorStem(nn.Module):
    """Early factor-conditioned stem for RSCD film/material/roughness coupling."""

    def __init__(self, cfg: CoupledFactorBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.evidence = FixedRoadEvidenceMaps()
        in_channels = cfg.in_channels + cfg.evidence_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, cfg.stem_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg.stem_dim),
            nn.GELU(),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(DepthwiseSeparableBlock(cfg.stem_dim, dilation=1), DepthwiseSeparableBlock(cfg.stem_dim, dilation=2)),
                nn.Sequential(DepthwiseSeparableBlock(cfg.stem_dim, dilation=1), DepthwiseSeparableBlock(cfg.stem_dim, dilation=3)),
                nn.Sequential(DepthwiseSeparableBlock(cfg.stem_dim, dilation=2), DepthwiseSeparableBlock(cfg.stem_dim, dilation=4)),
                nn.Sequential(DepthwiseSeparableBlock(cfg.stem_dim, dilation=1), DepthwiseSeparableBlock(cfg.stem_dim, dilation=1)),
                nn.Identity(),
            ]
        )
        self.gate = FactorCouplingGate(cfg.evidence_channels, len(self.experts))
        self.concrete_roughness_conditioner = (
            ConcreteRoughnessScaleSpaceConditioner(
                cfg.stem_dim,
                scale=float(cfg.concrete_roughness_scale_space_scale),
                mode=str(cfg.concrete_roughness_scale_space_mode),
            )
            if bool(cfg.use_concrete_roughness_scale_space)
            else None
        )
        self.dual_film_texture_conditioner = (
            DualFilmTextureRoughnessCouplingConditioner(
                cfg.stem_dim,
                scale=float(cfg.dual_film_texture_coupling_scale),
                mode=str(cfg.dual_film_texture_coupling_mode),
            )
            if bool(cfg.use_dual_film_texture_coupling)
            else None
        )
        self.mix_norm = nn.BatchNorm2d(cfg.stem_dim)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        evidence = self.evidence(image)
        x = self.proj(torch.cat([image, evidence], dim=1))
        aux: dict[str, torch.Tensor] = {"evidence": evidence}
        if self.concrete_roughness_conditioner is not None:
            x, cr_aux = self.concrete_roughness_conditioner(x, evidence)
            aux["concrete_roughness_route_weights"] = cr_aux["route_weights"]
            aux["concrete_roughness_route_maps"] = cr_aux["route_maps"]
            aux["concrete_roughness_stats"] = cr_aux["stats"]
        if self.dual_film_texture_conditioner is not None:
            x, dual_aux = self.dual_film_texture_conditioner(x, evidence)
            aux["dual_film_texture_route_weights"] = dual_aux["route_weights"]
            aux["dual_film_texture_route_maps"] = dual_aux["route_maps"]
            aux["dual_film_texture_stats"] = dual_aux["stats"]
        if self.cfg.coupling_gate_mode == "learned":
            weights = self.gate(evidence)
        elif self.cfg.coupling_gate_mode == "fixed_uniform":
            weights = x.new_full((x.shape[0], len(self.experts)), 1.0 / float(len(self.experts)))
        elif self.cfg.coupling_gate_mode == "fixed_identity":
            weights = x.new_zeros((x.shape[0], len(self.experts)))
            weights[:, -1] = 1.0
        else:
            raise ValueError(f"unknown coupling_gate_mode: {self.cfg.coupling_gate_mode}")
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (expert_out * weights[:, :, None, None, None]).sum(dim=1)
        aux["coupling_weights"] = weights
        return self.mix_norm(mixed), aux


class DownsampleStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, depth: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[DepthwiseSeparableBlock(out_dim, dilation=1 + (idx % 2)) for idx in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class CoupledFactorBackbone(nn.Module):
    """Prototype custom RSCD backbone with explicit early coupling experts.

    The backbone targets the known RSCD bottleneck: low-friction film evidence,
    concrete/asphalt material cues, and slight/severe roughness are coupled
    differently across classes. It therefore routes early texture extraction
    through coupling experts instead of adding a late correction head.
    """

    def __init__(self, cfg: CoupledFactorBackboneConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CoupledFactorBackboneConfig()
        c1, c2, c3 = self.cfg.stage_dims
        self.stem = CoupledFactorStem(self.cfg)
        self.stage1 = DownsampleStage(self.cfg.stem_dim, c1, depth=2)
        self.stage2 = DownsampleStage(c1, c2, depth=3)
        self.stage3 = DownsampleStage(c2, c3, depth=3)
        self.out_dim = c3
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(c3)

    def forward_features(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x, aux = self.stem(image)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        pooled = self.pool(x).flatten(1)
        return self.norm(pooled), aux

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features, _ = self.forward_features(image)
        return features


class RSCDCoupledFactorClassifier(nn.Module):
    """Prototype classifier wrapper for same-budget RSCD screens."""

    def __init__(self, num_classes: int = 27, cfg: CoupledFactorBackboneConfig | None = None) -> None:
        super().__init__()
        self.backbone = CoupledFactorBackbone(cfg)
        self.head = nn.Sequential(
            nn.Dropout((cfg or self.backbone.cfg).dropout),
            nn.Linear(self.backbone.out_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, *, return_aux: bool = False):
        features, aux = self.backbone.forward_features(image)
        logits = self.head(features)
        if return_aux:
            return logits, aux
        return logits


class RSCDCoupledFactorFactorizedClassifier(nn.Module):
    """Trainable S136 classifier with class, factor, and coupling-gate outputs."""

    def __init__(
        self,
        class_to_idx: dict[str, int],
        cfg: CoupledFactorBackboneConfig | None = None,
    ) -> None:
        super().__init__()
        self.spec = build_rscd_factor_spec(class_to_idx)
        self.backbone = CoupledFactorBackbone(cfg)
        active_cfg = cfg or self.backbone.cfg
        self.dropout = nn.Dropout(active_cfg.dropout)
        self.class_head = nn.Linear(self.backbone.out_dim, self.spec.num_classes)
        self.factor_heads = nn.ModuleDict(
            {
                axis: nn.Linear(self.backbone.out_dim, len(labels))
                for axis, labels in FACTOR_LABELS.items()
            }
        )

    def forward(self, image: torch.Tensor, *, return_aux: bool = False):
        features, aux = self.backbone.forward_features(image)
        dropped = self.dropout(features)
        logits = self.class_head(dropped)
        if not return_aux:
            return logits
        factor_logits = {
            axis: head(dropped)
            for axis, head in self.factor_heads.items()
        }
        return {
            "logits": logits,
            "features": features,
            "factor_logits": factor_logits,
            "coupling_weights": aux["coupling_weights"],
            "evidence": aux["evidence"],
            "concrete_roughness_route_weights": aux.get("concrete_roughness_route_weights"),
            "concrete_roughness_stats": aux.get("concrete_roughness_stats"),
            "dual_film_texture_route_weights": aux.get("dual_film_texture_route_weights"),
            "dual_film_texture_stats": aux.get("dual_film_texture_stats"),
            "boundary_logits": {},
        }


def count_parameters(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))

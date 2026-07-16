from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from friction_affordance.models.backbone import build_backbone
from friction_affordance.models.texture import (
    LocalPhysicsFieldBranch,
    PhysicsTextureBranch,
    SemanticPhysicsAttentionBranch,
)
from friction_affordance.rscd_factors import (
    FACTOR_LABELS,
    RSCDFactorSpec,
    build_rscd_factor_spec,
    canonical_class_label,
)


class C3PhysicsEvidenceStats(nn.Module):
    """Differentiable image evidence used by C3-FaRNet losses and gates.

    The vector is deliberately low-level: wet-film, dark-water, specular,
    roughness, snow/ice-like whiteness, texture erasure, and soft wet
    connectedness. It supplies mechanism evidence without adding new labels.
    """

    out_dim = 16

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
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3) / 4.0,
        )

    @staticmethod
    def _soft_connectedness(mask: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(mask, kernel_size=9, stride=1, padding=4)
        return (mask * pooled).mean(dim=(2, 3))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        r, g_ch, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
        value = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = ((value - minc) / value.clamp_min(1e-4)).clamp(0.0, 1.0)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6).clamp(0.0, 1.0)
        lap = F.conv2d(gray, self.laplace, padding=1).abs().clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        specular = torch.sigmoid((value - 0.80) * 14.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.34 - saturation) * 10.0)
            * torch.sigmoid((0.055 - grad) * 30.0)
        )
        wet = torch.clamp(specular + 0.65 * dark_water, 0.0, 1.0)
        rough = (0.42 * grad + 0.30 * lap + 0.28 * contrast).clamp(0.0, 1.0)
        low_texture = torch.sigmoid((0.050 - grad) * 32.0)
        low_contrast = torch.sigmoid((0.035 - contrast) * 40.0)
        texture_erasure = low_texture * low_contrast
        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.30 - saturation) * 12.0)
        ice_like = specular * low_texture * torch.sigmoid((0.22 - saturation) * 12.0)
        marking_like = torch.sigmoid((value - 0.76) * 15.0) * torch.sigmoid((grad - 0.08) * 18.0)

        return torch.cat(
            [
                gray.mean(dim=(2, 3)),
                gray.std(dim=(2, 3), unbiased=False),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3), unbiased=False),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3), unbiased=False),
                lap.mean(dim=(2, 3)),
                contrast.mean(dim=(2, 3)),
                specular.mean(dim=(2, 3)),
                dark_water.mean(dim=(2, 3)),
                wet.mean(dim=(2, 3)),
                rough.mean(dim=(2, 3)),
                texture_erasure.mean(dim=(2, 3)),
                snow_like.mean(dim=(2, 3)),
                ice_like.mean(dim=(2, 3)),
                self._soft_connectedness(torch.clamp(wet + snow_like, 0.0, 1.0)),
            ],
            dim=1,
        )

    @staticmethod
    def roughness_reliability_target(stats: torch.Tensor) -> torch.Tensor:
        rough = stats[:, 11:12]
        specular = stats[:, 8:9]
        dark_water = stats[:, 9:10]
        wet = stats[:, 10:11]
        erasure = stats[:, 12:13]
        snow = stats[:, 13:14]
        ice = stats[:, 14:15]
        return torch.sigmoid(4.0 * rough - 2.2 * wet - 1.7 * dark_water - 1.4 * specular - 1.2 * erasure - snow - ice)


def _normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(1)
    lo = flat.amin(dim=1, keepdim=True).view(-1, 1, 1, 1)
    hi = flat.amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo).clamp_min(float(eps))


class DryConcreteRoughnessVORResidual(nn.Module):
    """Narrow dry-concrete roughness chart carried forward from the best anchor."""

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 48,
        scale: float = 0.12,
        gate_threshold: float = 0.12,
        gate_temperature: float = 14.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        required = ["dry_concrete_smooth", "dry_concrete_slight", "dry_concrete_severe"]
        missing = [name for name in required if name not in class_to_idx]
        if missing:
            raise ValueError(f"DryConcreteRoughnessVORResidual missing RSCD classes: {missing}")
        self.register_buffer("dry_concrete_idx", torch.as_tensor([class_to_idx[name] for name in required], dtype=torch.long))
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
        stat_dim = 40
        logit_dim = 10
        self.head = nn.Sequential(
            nn.LayerNorm(stat_dim + logit_dim),
            nn.Linear(stat_dim + logit_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3),
        )
        if zero_init:
            last = self.head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    @staticmethod
    def _top_fraction_mean(x: torch.Tensor, fraction: float) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(1, int(flat.size(1) * float(fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def _field_stats(self, field: torch.Tensor) -> torch.Tensor:
        flat = field.flatten(1)
        return torch.cat(
            [
                field.mean(dim=(2, 3)),
                field.std(dim=(2, 3), unbiased=False),
                self._top_fraction_mean(field, 0.05),
                self._top_fraction_mean(field, 0.15),
                C3PhysicsEvidenceStats._soft_connectedness(field),
                (flat > 0.50).to(dtype=field.dtype).mean(dim=1, keepdim=True),
            ],
            dim=1,
        )

    def _stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 128:
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
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
        obstruction = torch.clamp(
            0.40 * thin_film + 0.30 * dark_water + 0.20 * specular + 0.35 * texture_erasure,
            0.0,
            1.0,
        )
        snow_phase = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        concrete_like = (
            torch.sigmoid((value - 0.38) * 8.0)
            * torch.sigmoid((0.82 - value) * 8.0)
            * torch.sigmoid((0.30 - saturation) * 10.0)
            * (1.0 - snow_phase)
            * (1.0 - marking)
        )
        visible_rough = rough_base * (1.0 - obstruction) * (1.0 - snow_phase) * (1.0 - marking)
        dry_rough = visible_rough * concrete_like
        anti_glare_rough = dry_rough * (1.0 - specular) * (1.0 - texture_erasure)
        global_stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3), unbiased=False),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3), unbiased=False),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3), unbiased=False),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3), unbiased=False),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3), unbiased=False),
            ],
            dim=1,
        )
        return torch.cat(
            [
                global_stats,
                self._field_stats(rough_base),
                self._field_stats(visible_rough),
                self._field_stats(dry_rough),
                self._field_stats(anti_glare_rough),
                self._field_stats(concrete_like),
            ],
            dim=1,
        )

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        dry_idx = self.dry_concrete_idx.to(device=base_logits.device)
        dry_logits = base_logits.index_select(1, dry_idx)
        probs = F.softmax(base_logits, dim=1)
        dry_probs = probs.index_select(1, dry_idx)
        dry_mass = dry_probs.sum(dim=1, keepdim=True)
        sorted_logits = dry_logits.sort(dim=1, descending=True).values
        sorted_probs = dry_probs.sort(dim=1, descending=True).values
        logit_features = torch.cat(
            [
                dry_logits,
                dry_probs,
                dry_mass,
                dry_probs.amax(dim=1, keepdim=True),
                sorted_logits[:, 0:1] - sorted_logits[:, 1:2],
                sorted_probs[:, 0:1] - sorted_probs[:, 1:2],
            ],
            dim=1,
        )
        stats = self._stats(image).to(device=base_logits.device, dtype=base_logits.dtype)
        raw_delta = torch.tanh(self.head(torch.cat([stats, logit_features.to(dtype=stats.dtype)], dim=1)))
        centered_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
        gate = torch.sigmoid((dry_mass - self.gate_threshold) * self.gate_temperature)
        ambiguity = torch.sigmoid((0.42 - (sorted_probs[:, 0:1] - sorted_probs[:, 1:2])) * 10.0)
        delta = centered_delta * gate.to(dtype=centered_delta.dtype) * ambiguity.to(dtype=centered_delta.dtype) * self.scale
        residual = torch.zeros_like(base_logits)
        residual.scatter_add_(1, dry_idx.view(1, -1).expand(base_logits.size(0), -1), delta.to(dtype=residual.dtype))
        return residual


class DryConcreteOrdinalChartResidual(DryConcreteRoughnessVORResidual):
    """Protected dry-concrete ordinal chart for smooth/slight/severe.

    Unlike pair-local residuals, this module writes one shared three-class
    correction through two ordered bases:

    - a roughness slope basis for smooth -> severe;
    - a middle-state basis that lets `slight` be a protected intermediate state.

    The final layer is zero-initialized, so enabling the module does not change
    existing checkpoints until the chart is explicitly trained.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 48,
        scale: float = 0.06,
        gate_threshold: float = 0.12,
        gate_temperature: float = 14.0,
        protect_confidence: float = 0.72,
        protect_temperature: float = 18.0,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            zero_init=False,
        )
        self.protect_confidence = float(protect_confidence)
        self.protect_temperature = float(protect_temperature)
        stat_dim = 40
        logit_dim = 10
        self.head = nn.Sequential(
            nn.LayerNorm(stat_dim + logit_dim),
            nn.Linear(stat_dim + logit_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.register_buffer(
            "ordinal_basis",
            torch.tensor(
                [
                    [-1.0, 0.0, 1.0],
                    [-0.5, 1.0, -0.5],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        dry_idx = self.dry_concrete_idx.to(device=base_logits.device)
        dry_logits = base_logits.index_select(1, dry_idx)
        probs = F.softmax(base_logits, dim=1)
        dry_probs = probs.index_select(1, dry_idx)
        dry_mass = dry_probs.sum(dim=1, keepdim=True)
        dry_max = dry_probs.amax(dim=1, keepdim=True)
        sorted_logits = dry_logits.sort(dim=1, descending=True).values
        sorted_probs = dry_probs.sort(dim=1, descending=True).values
        logit_features = torch.cat(
            [
                dry_logits,
                dry_probs,
                dry_mass,
                dry_max,
                sorted_logits[:, 0:1] - sorted_logits[:, 1:2],
                sorted_probs[:, 0:1] - sorted_probs[:, 1:2],
            ],
            dim=1,
        )
        stats = self._stats(image).to(device=base_logits.device, dtype=base_logits.dtype)
        coeff = torch.tanh(self.head(torch.cat([stats, logit_features.to(dtype=stats.dtype)], dim=1)))
        basis = self.ordinal_basis.to(device=base_logits.device, dtype=base_logits.dtype)
        chart_delta = coeff.to(dtype=base_logits.dtype) @ basis
        gate = torch.sigmoid((dry_mass - self.gate_threshold) * self.gate_temperature)
        ambiguity = torch.sigmoid((0.42 - (sorted_probs[:, 0:1] - sorted_probs[:, 1:2])) * 10.0)
        protect = torch.sigmoid((self.protect_confidence - dry_max) * self.protect_temperature)
        delta = chart_delta * gate.to(dtype=chart_delta.dtype) * ambiguity.to(dtype=chart_delta.dtype)
        delta = delta * protect.to(dtype=chart_delta.dtype) * self.scale
        residual = torch.zeros_like(base_logits)
        residual.scatter_add_(1, dry_idx.view(1, -1).expand(base_logits.size(0), -1), delta.to(dtype=residual.dtype))
        return residual


class DryConcreteValidationSafeTransition(nn.Module):
    """Validation-selected dry-concrete slight/severe transition controller.

    This is the in-model form of the strict validation-safe rule that improved
    `dry_concrete_slight` without producing regressions in the cap120 screen.
    It is deliberately narrow: it only shifts mass from the source class to the
    target class when the target is already a top-k alternative and the margin
    is small.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        source: str = "dry_concrete_severe",
        target: str = "dry_concrete_slight",
        topk: int = 2,
        margin: float = 0.20,
        delta: float = 0.20,
    ) -> None:
        super().__init__()
        source_name = canonical_class_label(source)
        target_name = canonical_class_label(target)
        missing = [name for name in (source_name, target_name) if name not in class_to_idx]
        if missing:
            raise ValueError(f"DryConcreteValidationSafeTransition missing RSCD classes: {missing}")
        self.source_name = source_name
        self.target_name = target_name
        self.topk = max(1, int(topk))
        self.margin = float(margin)
        self.delta = float(delta)
        self.register_buffer("source_idx", torch.as_tensor(int(class_to_idx[source_name]), dtype=torch.long))
        self.register_buffer("target_idx", torch.as_tensor(int(class_to_idx[target_name]), dtype=torch.long))

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source = int(self.source_idx.item())
        target = int(self.target_idx.item())
        topk = max(1, min(int(self.topk), int(logits.shape[1])))
        pred = logits.argmax(dim=1)
        order = torch.argsort(logits, dim=1, descending=True)
        target_in_topk = order[:, :topk].eq(target).any(dim=1)
        close = (logits[:, source] - logits[:, target]) <= float(self.margin)
        mask = pred.eq(source) & target_in_topk & close
        residual = torch.zeros_like(logits)
        if bool(mask.any()):
            residual[mask, target] = residual[mask, target] + float(self.delta)
            residual[mask, source] = residual[mask, source] - 0.25 * float(self.delta)
        return logits + residual, {
            "residual": residual,
            "mask": mask,
        }


class BackboneIsolatedDryConcreteLogitAdapter(nn.Module):
    """Closed-set dry-concrete adapter driven by an isolated backbone branch."""

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        branch_dim: int = 96,
        hidden_dim: int = 64,
        scale: float = 0.18,
        gate_threshold: float = 0.10,
        gate_temperature: float = 14.0,
        dropout: float = 0.02,
        output_mode: str = "free",
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.output_mode = str(output_mode)
        if self.output_mode not in {"free", "ordinal"}:
            raise ValueError(f"Unsupported backbone isolated dry-concrete output_mode: {output_mode}")
        required = ["dry_concrete_smooth", "dry_concrete_slight", "dry_concrete_severe"]
        missing = [name for name in required if name not in class_to_idx]
        if missing:
            raise ValueError(f"BackboneIsolatedDryConcreteLogitAdapter missing RSCD classes: {missing}")
        self.register_buffer("dry_concrete_idx", torch.as_tensor([class_to_idx[name] for name in required], dtype=torch.long))
        logit_dim = 10
        input_dim = int(branch_dim) + C3PhysicsEvidenceStats.out_dim + logit_dim + 1
        head_out_dim = 2 if self.output_mode == "ordinal" else 3
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), head_out_dim),
        )
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.register_buffer(
            "ordinal_basis",
            torch.tensor(
                [
                    [-1.0, 0.0, 1.0],
                    [-0.5, 1.0, -0.5],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        logits: torch.Tensor,
        branch_feature: torch.Tensor,
        branch_gate: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        dry_idx = self.dry_concrete_idx.to(device=logits.device)
        dry_logits = logits.index_select(1, dry_idx)
        probs = F.softmax(logits, dim=1)
        dry_probs = probs.index_select(1, dry_idx)
        dry_mass = dry_probs.sum(dim=1, keepdim=True)
        sorted_probs = dry_probs.sort(dim=1, descending=True).values
        sorted_logits = dry_logits.sort(dim=1, descending=True).values
        prob_gap = sorted_probs[:, 0:1] - sorted_probs[:, 1:2]
        logit_features = torch.cat(
            [
                dry_logits,
                dry_probs,
                dry_mass,
                dry_probs.amax(dim=1, keepdim=True),
                sorted_logits[:, 0:1] - sorted_logits[:, 1:2],
                prob_gap,
            ],
            dim=1,
        )
        branch_feature = branch_feature.to(device=logits.device, dtype=logits.dtype)
        branch_gate = branch_gate.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
        evidence = evidence.to(device=logits.device, dtype=logits.dtype)
        head_raw = torch.tanh(self.head(torch.cat([branch_feature, evidence, logit_features, branch_gate], dim=1)))
        if self.output_mode == "ordinal":
            basis = self.ordinal_basis.to(device=logits.device, dtype=logits.dtype)
            raw = head_raw.to(dtype=logits.dtype) @ basis
        else:
            raw = head_raw
        centered = raw - raw.mean(dim=1, keepdim=True)
        dry_mass_gate = torch.sigmoid((dry_mass - self.gate_threshold) * self.gate_temperature)
        ambiguity_gate = torch.sigmoid((0.42 - prob_gap) * 10.0)
        gate = branch_gate * dry_mass_gate.to(dtype=logits.dtype) * ambiguity_gate.to(dtype=logits.dtype)
        delta = centered * gate * self.scale
        residual = torch.zeros_like(logits)
        residual.scatter_add_(1, dry_idx.view(1, -1).expand(logits.size(0), -1), delta.to(dtype=residual.dtype))
        return logits + residual, {
            "residual": residual,
            "raw": raw,
            "gate": gate,
            "dry_mass": dry_mass,
            "prob_gap": prob_gap,
        }


class BackboneFamilyOrdinalNoSpillAdapter(nn.Module):
    """Family-wise roughness ordinal adapter from intermediate backbone maps.

    Each family is one friction-material slice, e.g. wet-concrete. The adapter
    can only redistribute logits among smooth/slight/severe inside that slice,
    so it targets RSCD roughness errors without changing friction or material.
    """

    default_families: tuple[tuple[str, str], ...] = (
        ("dry", "asphalt"),
        ("dry", "concrete"),
        ("wet", "asphalt"),
        ("wet", "concrete"),
        ("water", "asphalt"),
        ("water", "concrete"),
    )

    @classmethod
    def _parse_families(cls, families: Any) -> tuple[tuple[str, str], ...]:
        if families is None:
            return cls.default_families
        if isinstance(families, str):
            items: list[Any] = [item.strip() for item in families.split(",") if item.strip()]
        else:
            items = list(families)
        parsed: list[tuple[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                friction = str(item.get("friction", "")).strip()
                material = str(item.get("material", "")).strip()
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                friction, material = str(item[0]).strip(), str(item[1]).strip()
            else:
                text = str(item).strip()
                if "|" in text:
                    friction, material = [part.strip() for part in text.split("|", 1)]
                elif "_" in text:
                    friction, material = [part.strip() for part in text.split("_", 1)]
                else:
                    raise ValueError(f"Invalid family spec for BackboneFamilyOrdinalNoSpillAdapter: {item!r}")
            pair = (friction, material)
            if pair not in cls.default_families:
                raise ValueError(f"Unsupported RSCD ordinal family: {pair!r}")
            parsed.append(pair)
        if not parsed:
            raise ValueError("BackboneFamilyOrdinalNoSpillAdapter requires at least one family")
        return tuple(parsed)

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 96,
        family_embed_dim: int = 12,
        scale: float = 0.18,
        gate_threshold: float = 0.055,
        gate_temperature: float = 10.0,
        dropout: float = 0.02,
        families: Any = None,
    ) -> None:
        super().__init__()
        self.families = self._parse_families(families)
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        family_indices: list[list[int]] = []
        missing: list[str] = []
        for friction, material in self.families:
            names = [
                f"{friction}_{material}_smooth",
                f"{friction}_{material}_slight",
                f"{friction}_{material}_severe",
            ]
            row = []
            for name in names:
                if name not in class_to_idx:
                    missing.append(name)
                else:
                    row.append(int(class_to_idx[name]))
            if len(row) == 3:
                family_indices.append(row)
        if missing:
            raise ValueError(f"BackboneFamilyOrdinalNoSpillAdapter missing RSCD classes: {missing}")
        self.register_buffer("family_indices", torch.as_tensor(family_indices, dtype=torch.long))
        self.family_embedding = nn.Embedding(len(self.families), int(family_embed_dim))
        stage_pool_dim = 4 * 96 + 4 * 192
        logit_dim = 10
        input_dim = stage_pool_dim + 28 + logit_dim + int(family_embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.register_buffer(
            "ordinal_basis",
            torch.tensor(
                [
                    [-1.0, 0.0, 1.0],
                    [-0.5, 1.0, -0.5],
                ],
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _masked_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.to(device=feat.device, dtype=feat.dtype).clamp(0.0, 1.0)
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-4)
        return (feat * mask).sum(dim=(2, 3)) / denom

    @staticmethod
    def _evidence_masks(evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        concrete = evidence[:, 6:7].clamp(0.0, 1.0)
        asphalt = evidence[:, 7:8].clamp(0.0, 1.0)
        specular = evidence[:, 8:9].clamp(0.0, 1.0)
        dark_water = evidence[:, 9:10].clamp(0.0, 1.0)
        erasure = evidence[:, 10:11].clamp(0.0, 1.0)
        rough = evidence[:, 11:12].clamp(0.0, 1.0)
        boundary = evidence[:, 12:13].clamp(0.0, 1.0)
        marking = evidence[:, 13:14].clamp(0.0, 1.0)
        film = torch.maximum(specular, erasure).clamp(0.0, 1.0)
        water = dark_water
        clean = (1.0 - 0.70 * marking).clamp(0.12, 1.0)
        dry = (1.0 - torch.maximum(film, water)).clamp(0.0, 1.0) * clean
        wet = film * (1.0 - 0.45 * water).clamp(0.0, 1.0) * clean
        water_state = water * clean
        smooth = (1.0 - rough).clamp(0.0, 1.0)
        slight = rough * (1.0 - boundary).clamp(0.0, 1.0)
        severe = torch.maximum(rough, boundary).clamp(0.0, 1.0)
        return {
            "dry": dry,
            "wet": wet,
            "water": water_state,
            "asphalt": asphalt,
            "concrete": concrete,
            "smooth": smooth,
            "slight": slight,
            "severe": severe,
        }

    def _family_vector(
        self,
        stage_maps: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        friction: str,
        material: str,
    ) -> torch.Tensor:
        family = (masks[friction] * masks[material]).clamp(0.0, 1.0)
        smooth = (family * masks["smooth"]).clamp(0.0, 1.0)
        slight = (family * masks["slight"]).clamp(0.0, 1.0)
        severe = (family * masks["severe"]).clamp(0.0, 1.0)
        pools: list[torch.Tensor] = []
        for key in ("1", "3"):
            feat = stage_maps[key]
            pools.extend(
                [
                    self._masked_pool(feat, family),
                    self._masked_pool(feat, smooth),
                    self._masked_pool(feat, slight),
                    self._masked_pool(feat, severe),
                ]
            )
        return torch.cat(pools, dim=1)

    def forward(
        self,
        logits: torch.Tensor,
        stage_maps: dict[str, torch.Tensor],
        evidence: torch.Tensor,
        stats: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if "1" not in stage_maps or "3" not in stage_maps:
            return logits, {
                "residual": torch.zeros_like(logits),
                "gates": logits.new_zeros((logits.shape[0], len(self.families))),
            }
        evidence = evidence.to(device=logits.device, dtype=logits.dtype)
        stats = stats.to(device=logits.device, dtype=logits.dtype)
        masks = self._evidence_masks(evidence)
        family_indices = self.family_indices.to(device=logits.device)
        basis = self.ordinal_basis.to(device=logits.device, dtype=logits.dtype)
        residual = torch.zeros_like(logits)
        gates: list[torch.Tensor] = []
        raw_terms: list[torch.Tensor] = []
        for family_id, (friction, material) in enumerate(self.families):
            idx = family_indices[family_id]
            family_logits = logits.index_select(1, idx)
            probs = F.softmax(logits, dim=1)
            family_probs = probs.index_select(1, idx)
            family_mass = family_probs.sum(dim=1, keepdim=True)
            sorted_probs = family_probs.sort(dim=1, descending=True).values
            sorted_logits = family_logits.sort(dim=1, descending=True).values
            prob_gap = sorted_probs[:, 0:1] - sorted_probs[:, 1:2]
            logit_features = torch.cat(
                [
                    family_logits,
                    family_probs,
                    family_mass,
                    family_probs.amax(dim=1, keepdim=True),
                    sorted_logits[:, 0:1] - sorted_logits[:, 1:2],
                    prob_gap,
                ],
                dim=1,
            )
            family_vec = self._family_vector(stage_maps, masks, friction, material).to(dtype=logits.dtype)
            family_id_tensor = torch.full((logits.shape[0],), family_id, device=logits.device, dtype=torch.long)
            family_embed = self.family_embedding(family_id_tensor).to(dtype=logits.dtype)
            coeff = torch.tanh(self.head(torch.cat([family_vec, stats, logit_features, family_embed], dim=1)))
            raw = coeff @ basis
            raw = raw - raw.mean(dim=1, keepdim=True)
            gate = torch.sigmoid((family_mass - self.gate_threshold) * self.gate_temperature)
            gate = gate * torch.sigmoid((0.42 - prob_gap) * 10.0)
            delta = raw * gate.to(dtype=raw.dtype) * self.scale
            residual.scatter_add_(1, idx.view(1, -1).expand(logits.size(0), -1), delta.to(dtype=residual.dtype))
            gates.append(gate)
            raw_terms.append(raw)
        return logits + residual, {
            "residual": residual,
            "gates": torch.cat(gates, dim=1),
            "raw": torch.stack(raw_terms, dim=1),
        }


class DryConcretePairSignedSelector(nn.Module):
    """Pair-local signed selector for dry-concrete roughness boundaries.

    The selector is intentionally narrow: it does not classify all RSCD
    classes. It only sees DryVOR-style roughness/concrete statistics plus the
    local hard-pair anchor state and learns when a signed severe/slight margin
    correction should be shifted or attenuated.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 48,
        shift_scale: float = 0.65,
        gain_scale: float = 0.50,
    ) -> None:
        super().__init__()
        self.shift_scale = float(shift_scale)
        self.gain_scale = float(gain_scale)
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
        input_dim = 40 + 5 + 1
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _top_fraction_mean(x: torch.Tensor, fraction: float) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(1, int(flat.size(1) * float(fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def _field_stats(self, field: torch.Tensor) -> torch.Tensor:
        flat = field.flatten(1)
        return torch.cat(
            [
                field.mean(dim=(2, 3)),
                field.std(dim=(2, 3), unbiased=False),
                self._top_fraction_mean(field, 0.05),
                self._top_fraction_mean(field, 0.15),
                C3PhysicsEvidenceStats._soft_connectedness(field),
                (flat > 0.50).to(dtype=field.dtype).mean(dim=1, keepdim=True),
            ],
            dim=1,
        )

    def _stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 128:
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
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
        obstruction = torch.clamp(
            0.40 * thin_film + 0.30 * dark_water + 0.20 * specular + 0.35 * texture_erasure,
            0.0,
            1.0,
        )
        snow_phase = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        concrete_like = (
            torch.sigmoid((value - 0.38) * 8.0)
            * torch.sigmoid((0.82 - value) * 8.0)
            * torch.sigmoid((0.30 - saturation) * 10.0)
            * (1.0 - snow_phase)
            * (1.0 - marking)
        )
        visible_rough = rough_base * (1.0 - obstruction) * (1.0 - snow_phase) * (1.0 - marking)
        dry_rough = visible_rough * concrete_like
        anti_glare_rough = dry_rough * (1.0 - specular) * (1.0 - texture_erasure)
        global_stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3), unbiased=False),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3), unbiased=False),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3), unbiased=False),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3), unbiased=False),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3), unbiased=False),
            ],
            dim=1,
        )
        return torch.cat(
            [
                global_stats,
                self._field_stats(rough_base),
                self._field_stats(visible_rough),
                self._field_stats(dry_rough),
                self._field_stats(anti_glare_rough),
                self._field_stats(concrete_like),
            ],
            dim=1,
        )

    def forward(
        self,
        image: torch.Tensor,
        pair_features: torch.Tensor,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return adjusted raw margin, gain, and signed shift with shape [B]."""

        stats = self._stats(image).to(device=pair_features.device, dtype=pair_features.dtype)
        raw_col = raw.to(dtype=pair_features.dtype).unsqueeze(1)
        out = self.net(torch.cat([stats, pair_features, raw_col], dim=1))
        shift = self.shift_scale * torch.tanh(out[:, 0])
        gain = (1.0 + self.gain_scale * torch.tanh(out[:, 1])).clamp(0.05, 2.0)
        return raw + shift.to(dtype=raw.dtype), gain.to(dtype=raw.dtype), shift.to(dtype=raw.dtype)


class FeatureValueBoundaryCorrector(nn.Module):
    """Pair-local logit correction from measured image-value evidence.

    This module is deliberately narrow. It does not reclassify all RSCD
    classes from hand-crafted values. Instead it reads image-derived
    wet-film, texture-erasure, visible-roughness, color and contrast values,
    then learns a signed correction only for diagnosed hard class pairs.
    """

    def __init__(
        self,
        class_to_idx: dict[str, int],
        pairs: list[str] | tuple[str, ...] | None,
        *,
        hidden_dim: int = 64,
        scale: float = 0.22,
        gate_margin: float = 1.05,
        gate_temperature: float = 4.5,
        gate_floor: float = 0.0,
        value_aug_std: float = 0.0,
        dropout: float = 0.0,
        severe_tail_protect: bool = False,
        severe_tail_protect_pairs: list[str] | tuple[str, ...] | None = None,
        severe_tail_protect_strength: float = 0.85,
        severe_tail_protect_prob: float = 0.34,
        severe_tail_protect_tail_threshold: float = 0.115,
        severe_tail_protect_temperature: float = 16.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_margin = float(gate_margin)
        self.gate_temperature = float(gate_temperature)
        self.gate_floor = float(gate_floor)
        self.value_aug_std = max(float(value_aug_std), 0.0)
        self.severe_tail_protect = bool(severe_tail_protect)
        self.severe_tail_protect_strength = min(max(float(severe_tail_protect_strength), 0.0), 1.0)
        self.severe_tail_protect_prob = float(severe_tail_protect_prob)
        self.severe_tail_protect_tail_threshold = float(severe_tail_protect_tail_threshold)
        self.severe_tail_protect_temperature = float(severe_tail_protect_temperature)
        self.idx_to_class = {int(idx): canonical_class_label(name) for name, idx in class_to_idx.items()}
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
        self.pair_names: dict[frozenset[str], tuple[str, str]] = {}
        for item in pairs or []:
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) != 2:
                raise ValueError(f"feature_value_boundary_pairs entry must contain two class names: {item}")
            left_name, right_name = canonical_class_label(parts[0]), canonical_class_label(parts[1])
            if left_name not in class_to_idx or right_name not in class_to_idx:
                raise ValueError(f"unknown feature_value_boundary_pairs class: {item}")
            self.pair_names[frozenset((left_name, right_name))] = (left_name, right_name)
        self.severe_tail_protect_pairs: set[frozenset[str]] = set()
        for item in severe_tail_protect_pairs or ["water_concrete_slight|water_concrete_severe"]:
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) == 2:
                self.severe_tail_protect_pairs.add(
                    frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))
                )
        self.stats_dim = 65
        input_dim = self.stats_dim + C3PhysicsEvidenceStats.out_dim + 5 + 1
        self.pair_nets = nn.ModuleDict()
        for left_name, right_name in self.pair_names.values():
            key = self._pair_key(class_to_idx[left_name], class_to_idx[right_name])
            net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 2),
            )
            last = net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            self.pair_nets[key] = net

    @staticmethod
    def _pair_key(left: int, right: int) -> str:
        return f"p{int(left)}_{int(right)}"

    @staticmethod
    def _top_fraction_mean(x: torch.Tensor, fraction: float) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(1, int(flat.size(1) * float(fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def _field_stats(self, field: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                field.mean(dim=(2, 3)),
                field.std(dim=(2, 3), unbiased=False),
                self._top_fraction_mean(field, 0.05),
                self._top_fraction_mean(field, 0.15),
            ],
            dim=1,
        )

    def _stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 128:
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
        r, g_ch, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
        value = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = ((value - minc) / value.clamp_min(1e-4)).clamp(0.0, 1.0)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6).clamp(0.0, 1.0)
        lap = F.conv2d(gray, self.laplace, padding=1).abs().clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4).clamp(0.0, 1.0)
        rough = torch.clamp(0.42 * grad + 0.30 * lap + 0.28 * contrast, 0.0, 1.0)
        specular = torch.sigmoid((value - 0.80) * 14.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.34 - saturation) * 10.0)
            * torch.sigmoid((0.055 - grad) * 30.0)
        )
        wet = torch.clamp(specular + 0.65 * dark_water, 0.0, 1.0)
        low_texture = torch.sigmoid((0.050 - grad) * 32.0)
        low_contrast = torch.sigmoid((0.035 - contrast) * 40.0)
        texture_erasure = low_texture * low_contrast
        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.30 - saturation) * 12.0)
        marking_like = torch.sigmoid((value - 0.76) * 15.0) * torch.sigmoid((grad - 0.08) * 18.0)
        concrete_like = (
            torch.sigmoid((value - 0.38) * 8.0)
            * torch.sigmoid((0.82 - value) * 8.0)
            * torch.sigmoid((0.30 - saturation) * 10.0)
            * (1.0 - snow_like)
            * (1.0 - marking_like)
        )
        asphalt_like = (
            torch.sigmoid((saturation - 0.04) * 16.0)
            * torch.sigmoid((0.70 - value) * 8.0)
            * (1.0 - snow_like)
        ).clamp(0.0, 1.0)
        visible_rough = rough * (1.0 - torch.clamp(wet + texture_erasure + snow_like, 0.0, 1.0))
        global_stats = torch.cat(
            [
                gray.mean(dim=(2, 3)),
                gray.std(dim=(2, 3), unbiased=False),
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3), unbiased=False),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3), unbiased=False),
                r.mean(dim=(2, 3)),
                g_ch.mean(dim=(2, 3)),
                b.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        fields = (
            grad,
            lap,
            contrast,
            rough,
            visible_rough,
            specular,
            dark_water,
            wet,
            texture_erasure,
            concrete_like,
            asphalt_like,
            snow_like,
            marking_like,
            C3PhysicsEvidenceStats._soft_connectedness(torch.clamp(wet + snow_like, 0.0, 1.0)).view(-1, 1, 1, 1),
        )
        field_stats = torch.cat([self._field_stats(field) for field in fields[:-1]], dim=1)
        connected = fields[-1].flatten(1)
        return torch.cat([global_stats, field_stats, connected, concrete_like.mean(dim=(2, 3)), asphalt_like.mean(dim=(2, 3)), marking_like.mean(dim=(2, 3))], dim=1)

    def _severe_tail_protect_gate(
        self,
        pair_names: frozenset[str],
        stats: torch.Tensor,
        probs: torch.Tensor,
        left: int,
        right: int,
        signed: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.severe_tail_protect
            or self.severe_tail_protect_strength <= 0.0
            or pair_names not in self.severe_tail_protect_pairs
        ):
            return signed.new_zeros(signed.shape)
        left_name = self.idx_to_class[int(left)]
        right_name = self.idx_to_class[int(right)]
        if "water_concrete_severe" not in pair_names or "water_concrete_slight" not in pair_names:
            return signed.new_zeros(signed.shape)
        severe = int(left) if left_name == "water_concrete_severe" else int(right)
        rough_top5 = stats[:, 23].clamp(0.0, 1.0)
        rough_top15 = stats[:, 24].clamp(0.0, 1.0)
        lap_top5 = stats[:, 15].clamp(0.0, 1.0)
        contrast_top15 = stats[:, 20].clamp(0.0, 1.0)
        grad_top15 = stats[:, 12].clamp(0.0, 1.0)
        dark_water_std = stats[:, 34].clamp(0.0, 1.0)
        dark_water_top15 = stats[:, 36].clamp(0.0, 1.0)
        wet_top15 = stats[:, 40].clamp(0.0, 1.0)
        erasure_top15 = stats[:, 44].clamp(0.0, 1.0)
        concrete_mean = stats[:, 62].clamp(0.0, 1.0)
        marking_mean = stats[:, 64].clamp(0.0, 1.0)
        snow_top15 = stats[:, 56].clamp(0.0, 1.0)
        hidden_tail = torch.clamp(
            0.22 * rough_top5
            + 0.18 * rough_top15
            + 0.17 * lap_top5
            + 0.15 * contrast_top15
            + 0.12 * grad_top15
            + 0.10 * dark_water_std
            + 0.06 * erasure_top15,
            0.0,
            1.0,
        )
        water_concrete_context = torch.clamp(
            concrete_mean
            * (0.42 * wet_top15 + 0.28 * dark_water_top15 + 0.20 * erasure_top15 + 0.10 * dark_water_std),
            0.0,
            1.0,
        )
        artifact_guard = (1.0 - torch.maximum(marking_mean, snow_top15)).clamp(0.10, 1.0)
        tail_gate = torch.sigmoid(
            (hidden_tail - float(self.severe_tail_protect_tail_threshold))
            * float(self.severe_tail_protect_temperature)
        )
        severe_prob = probs[:, severe].clamp(0.0, 1.0)
        severe_prob_gate = torch.sigmoid(
            (severe_prob - float(self.severe_tail_protect_prob))
            * float(self.severe_tail_protect_temperature)
        )
        harmful_raw = signed if severe != int(left) else -signed
        harmful_gate = torch.sigmoid(harmful_raw * float(self.severe_tail_protect_temperature))
        return (tail_gate * water_concrete_context * artifact_guard * severe_prob_gate * harmful_gate).clamp(0.0, 1.0)

    def forward(
        self,
        image: torch.Tensor,
        logits: torch.Tensor,
        evidence_stats: torch.Tensor,
        spec: RSCDFactorSpec,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if self.scale <= 0.0 or not self.pair_nets:
            return logits, {}, {}, {}
        stats = self._stats(image).to(device=logits.device, dtype=logits.dtype)
        if self.training and self.value_aug_std > 0.0:
            stats = (stats + torch.randn_like(stats) * float(self.value_aug_std)).clamp(-2.0, 2.0)
        evidence = evidence_stats.to(device=logits.device, dtype=logits.dtype)
        probs = F.softmax(logits, dim=1)
        residual = torch.zeros_like(logits)
        raw_logits: dict[str, torch.Tensor] = {}
        deltas: dict[str, torch.Tensor] = {}
        gates: dict[str, torch.Tensor] = {}
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            left_name = self.idx_to_class[left]
            right_name = self.idx_to_class[right]
            wanted = self.pair_names.get(frozenset((left_name, right_name)))
            if wanted is None:
                continue
            key = self._pair_key(left, right)
            reversed_orientation = wanted != (left_name, right_name)
            net_key = key if key in self.pair_nets else self._pair_key(right, left)
            if net_key not in self.pair_nets:
                continue
            pair_features = C3FaRNetSurfaceClassifier._hardpair_pair_features(logits, probs, left, right).to(dtype=logits.dtype)
            signed_gap = (logits[:, left : left + 1] - logits[:, right : right + 1]).to(dtype=logits.dtype)
            pair_input = torch.cat([stats, evidence, pair_features, torch.tanh(0.25 * signed_gap)], dim=1)
            raw = self.pair_nets[net_key](pair_input)
            signed_raw = raw[:, 0]
            if reversed_orientation:
                signed_raw = -signed_raw
            signed = torch.tanh(signed_raw)
            base_gap = signed_gap.abs().squeeze(1)
            pair_mass = (probs[:, left] + probs[:, right]).clamp(0.0, 1.0)
            boundary_gate = torch.sigmoid((self.gate_margin - base_gap) * self.gate_temperature) * pair_mass
            learned_gate = torch.sigmoid(raw[:, 1])
            if self.gate_floor > 0.0:
                learned_gate = self.gate_floor + (1.0 - self.gate_floor) * learned_gate
            delta = self.scale * boundary_gate * learned_gate * signed
            pair_names = frozenset((left_name, right_name))
            severe_tail_protect_gate = self._severe_tail_protect_gate(
                pair_names,
                stats,
                probs,
                left,
                right,
                signed,
            )
            if self.severe_tail_protect_strength > 0.0:
                delta = delta * (1.0 - self.severe_tail_protect_strength * severe_tail_protect_gate)
            residual[:, left] = residual[:, left] + delta
            residual[:, right] = residual[:, right] - delta
            raw_logits[key] = signed_raw
            deltas[key] = delta
            gates[key] = boundary_gate * learned_gate
            if self.severe_tail_protect:
                gates[f"{key}_severe_tail_protect"] = severe_tail_protect_gate
        return logits + residual, raw_logits, deltas, gates


class WaterConcreteOpponentFeatureConditioner(nn.Module):
    """Feature-level opponent axes for the wet/water-concrete hard subgraph.

    The S96 logit-level pair comparator showed a small but clean signal on the
    water-concrete-slight boundary. This module moves that idea earlier: it
    learns pair-local signed axes in the fused feature space before the factor
    decoder and calibrated heads. The axes are activated only by water/concrete
    value evidence and by near-boundary pair probability mass, so the update is
    tied to the RSCD coupling mechanism instead of becoming a generic residual.
    """

    value_dim = 20

    def __init__(
        self,
        class_to_idx: dict[str, int],
        pairs: list[str] | tuple[str, ...] | None,
        *,
        feature_dim: int,
        hidden_dim: int = 64,
        scale: float = 0.018,
        gate_margin: float = 1.08,
        gate_temperature: float = 4.5,
        gate_floor: float = 0.03,
        value_aug_std: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.gate_margin = float(gate_margin)
        self.gate_temperature = float(gate_temperature)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.value_aug_std = max(float(value_aug_std), 0.0)
        self.idx_to_class = {int(idx): canonical_class_label(name) for name, idx in class_to_idx.items()}
        self.pair_names: dict[frozenset[str], tuple[str, str]] = {}
        for item in pairs or []:
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) != 2:
                raise ValueError(f"water_concrete_opponent_pairs entry must contain two class names: {item}")
            left_name, right_name = canonical_class_label(parts[0]), canonical_class_label(parts[1])
            if left_name not in class_to_idx or right_name not in class_to_idx:
                raise ValueError(f"unknown water_concrete_opponent_pairs class: {item}")
            self.pair_names[frozenset((left_name, right_name))] = (left_name, right_name)
        input_dim = self.value_dim + 5 + 1
        self.pair_nets = nn.ModuleDict()
        self.axes = nn.ParameterDict()
        feature_dim = int(feature_dim)
        for left_name, right_name in self.pair_names.values():
            key = self._pair_key(class_to_idx[left_name], class_to_idx[right_name])
            net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 2),
            )
            last = net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            self.pair_nets[key] = net
            axis = torch.empty(feature_dim)
            nn.init.normal_(axis, mean=0.0, std=0.02)
            self.axes[key] = nn.Parameter(axis)

    @staticmethod
    def _pair_key(left: int, right: int) -> str:
        return f"p{int(left)}_{int(right)}"

    @staticmethod
    def _hand_gate(pair_names: frozenset[str], values: torch.Tensor) -> torch.Tensor:
        macro_rough = values[:, 0].clamp(0.0, 1.0)
        micro_rough = values[:, 1].clamp(0.0, 1.0)
        film = values[:, 2].clamp(0.0, 1.0)
        artifact = values[:, 3].clamp(0.0, 1.0)
        saturation = values[:, 4].clamp(0.0, 1.0)
        macro_mean = values[:, 5].clamp(0.0, 1.0)
        macro_std = values[:, 6].clamp(0.0, 1.0)
        meso_std = values[:, 7].clamp(0.0, 1.0)
        lap_std = values[:, 9].clamp(0.0, 1.0)
        grad_std = values[:, 10].clamp(0.0, 1.0)
        anisotropy = values[:, 11].clamp(0.0, 1.0)
        dark_water = values[:, 12].clamp(0.0, 1.0)
        dark_water_top = values[:, 13].clamp(0.0, 1.0)
        specular = values[:, 14].clamp(0.0, 1.0)
        specular_top = values[:, 15].clamp(0.0, 1.0)
        texture_erasure = values[:, 16].clamp(0.0, 1.0)
        texture_erasure_top = values[:, 17].clamp(0.0, 1.0)
        value_std = values[:, 19].clamp(0.0, 1.0)
        if pair_names == frozenset(("water_concrete_smooth", "water_concrete_slight")):
            score = 0.32 * dark_water_top + 0.24 * dark_water + 0.18 * film + 0.16 * texture_erasure + 0.10 * value_std
            threshold = 0.34
        elif pair_names == frozenset(("water_concrete_slight", "water_concrete_severe")):
            hidden_rough = 0.26 * macro_rough + 0.20 * macro_std + 0.17 * meso_std + 0.14 * lap_std + 0.11 * grad_std
            film_context = 0.18 * dark_water_top + 0.12 * film + 0.10 * texture_erasure_top
            score = (hidden_rough + film_context + 0.08 * anisotropy).clamp(0.0, 1.0)
            threshold = 0.40
        elif pair_names == frozenset(("water_concrete_slight", "wet_concrete_slight")):
            score = 0.30 * dark_water_top + 0.24 * dark_water + 0.18 * film + 0.14 * specular_top + 0.08 * specular + 0.06 * (1.0 - saturation)
            threshold = 0.36
        else:
            score = 0.24 * macro_rough + 0.22 * micro_rough + 0.20 * film + 0.18 * texture_erasure_top + 0.16 * macro_mean
            threshold = 0.38
        artifact_guard = (1.0 - 0.70 * artifact).clamp(0.12, 1.0)
        return torch.sigmoid((score.clamp(0.0, 1.0) - threshold) * 8.0) * artifact_guard

    def forward(
        self,
        feature: torch.Tensor,
        base_logits: torch.Tensor,
        value_vector: torch.Tensor,
        spec: RSCDFactorSpec,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.scale <= 0.0 or not self.pair_nets:
            return feature, {}, {}, {}
        values = value_vector[:, : self.value_dim].to(device=feature.device, dtype=feature.dtype).clamp(0.0, 1.0)
        if self.training and self.value_aug_std > 0.0:
            values = (values + torch.randn_like(values) * float(self.value_aug_std)).clamp(0.0, 1.0)
        probs = F.softmax(base_logits, dim=1)
        residual = torch.zeros_like(feature)
        raw_logits: dict[str, torch.Tensor] = {}
        deltas: dict[str, torch.Tensor] = {}
        gates: dict[str, torch.Tensor] = {}
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            left_name = self.idx_to_class[left]
            right_name = self.idx_to_class[right]
            wanted = self.pair_names.get(frozenset((left_name, right_name)))
            if wanted is None:
                continue
            key = self._pair_key(left, right)
            reversed_orientation = wanted != (left_name, right_name)
            net_key = key if key in self.pair_nets else self._pair_key(right, left)
            if net_key not in self.pair_nets:
                continue
            pair_features = C3FaRNetSurfaceClassifier._hardpair_pair_features(
                base_logits,
                probs,
                left,
                right,
            ).to(dtype=feature.dtype)
            signed_gap = (base_logits[:, left : left + 1] - base_logits[:, right : right + 1]).to(dtype=feature.dtype)
            pair_input = torch.cat([values, pair_features, torch.tanh(0.25 * signed_gap)], dim=1)
            raw = self.pair_nets[net_key](pair_input)
            signed_raw = raw[:, 0]
            if reversed_orientation:
                signed_raw = -signed_raw
            learned_gate = torch.sigmoid(raw[:, 1])
            if self.gate_floor > 0.0:
                learned_gate = self.gate_floor + (1.0 - self.gate_floor) * learned_gate
            base_gap = signed_gap.abs().squeeze(1)
            pair_mass = (probs[:, left] + probs[:, right]).clamp(0.0, 1.0)
            boundary_gate = torch.sigmoid((self.gate_margin - base_gap) * self.gate_temperature) * pair_mass
            pair_names = frozenset((left_name, right_name))
            hand_gate = self._hand_gate(pair_names, values).to(dtype=feature.dtype)
            gate = (boundary_gate * learned_gate * hand_gate).clamp(0.0, 1.0)
            axis = F.normalize(self.axes[net_key].to(device=feature.device, dtype=feature.dtype), dim=0)
            delta = self.scale * gate.unsqueeze(1) * torch.tanh(signed_raw).unsqueeze(1) * axis.view(1, -1)
            residual = residual + delta
            raw_logits[key] = signed_raw
            deltas[key] = delta
            gates[key] = gate
        return feature + residual, raw_logits, deltas, gates


class C3PhysicsEvidenceMaps(nn.Module):
    """Differentiable local evidence maps for spatial factor queries."""

    out_channels = 10

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
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3) / 4.0,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        r, g_ch, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
        value = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = ((value - minc) / value.clamp_min(1e-4)).clamp(0.0, 1.0)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6).clamp(0.0, 1.0)
        lap = F.conv2d(gray, self.laplace, padding=1).abs().clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4).clamp(0.0, 1.0)
        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        low_contrast = torch.sigmoid((0.030 - contrast) * 45.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = torch.sigmoid((0.42 - value) * 10.0) * torch.sigmoid((0.30 - saturation) * 12.0) * low_texture
        wet = torch.clamp(specular + 0.65 * dark_water, 0.0, 1.0)
        rough = torch.clamp(0.42 * grad + 0.30 * lap + 0.28 * contrast, 0.0, 1.0)
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.30 - saturation) * 12.0)
        marking_like = torch.sigmoid((value - 0.88) * 16.0) * torch.sigmoid((0.22 - saturation) * 14.0)
        wet_connected = wet * F.avg_pool2d(wet, kernel_size=9, stride=1, padding=4)
        return torch.cat(
            [
                gray,
                saturation,
                grad,
                lap,
                contrast,
                specular,
                dark_water,
                wet_connected.clamp(0.0, 1.0),
                rough * (1.0 - texture_erasure).clamp(0.0, 1.0),
                torch.clamp(texture_erasure + snow_like + marking_like, 0.0, 1.0),
            ],
            dim=1,
        )


class PhysicsTextureStemAdapter(nn.Module):
    """Early physics-conditioned image adapter before the ConvNeXt stem.

    The adapter is not a late classifier residual. It only writes a bounded
    input-level correction where low-level RSCD evidence indicates smooth film,
    dark water, visible roughness, or texture erasure. The final projection is
    zero-initialized so resumed calibrated checkpoints start from the exact
    original image stream.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        scale: float = 0.035,
        gate_floor: float = 0.18,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.scale = max(float(scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.evidence_maps = C3PhysicsEvidenceMaps()
        in_channels = 3 + 3 + 3 + C3PhysicsEvidenceMaps.out_channels + 1
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden_dim)), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=max(1, hidden_dim), bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )
        last = self.adapter[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _target_gate(evidence: torch.Tensor) -> torch.Tensor:
        grad = evidence[:, 2:3].clamp(0.0, 1.0)
        lap = evidence[:, 3:4].clamp(0.0, 1.0)
        contrast = evidence[:, 4:5].clamp(0.0, 1.0)
        specular = evidence[:, 5:6].clamp(0.0, 1.0)
        dark_water = evidence[:, 6:7].clamp(0.0, 1.0)
        wet_connected = evidence[:, 7:8].clamp(0.0, 1.0)
        visible_rough = evidence[:, 8:9].clamp(0.0, 1.0)
        artifact = evidence[:, 9:10].clamp(0.0, 1.0)
        rough_energy = (0.42 * grad + 0.30 * lap + 0.28 * contrast).clamp(0.0, 1.0)
        smooth_film = torch.sigmoid((0.20 - rough_energy) * 10.0) * torch.maximum(specular, wet_connected)
        hidden_rough = torch.maximum(dark_water, wet_connected) * visible_rough
        dry_texture = (1.0 - torch.maximum(specular, dark_water)) * rough_energy
        gate = torch.clamp(0.45 * smooth_film + 0.35 * hidden_rough + 0.25 * dry_texture, 0.0, 1.0)
        return (gate * (1.0 - 0.65 * artifact).clamp(0.20, 1.0)).clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.scale <= 0.0:
            return image
        evidence = self.evidence_maps(image).to(device=image.device, dtype=image.dtype)
        low = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        high = image - low
        gate = self._target_gate(evidence).to(device=image.device, dtype=image.dtype)
        fields = torch.cat([image, low, high, evidence, gate], dim=1)
        correction_gate = self.gate_floor + (1.0 - self.gate_floor) * gate
        delta = self.scale * correction_gate * torch.tanh(self.adapter(fields))
        return image + delta


class ScaleSpaceRoughnessStemAdapter(nn.Module):
    """Early scale-space roughness adapter for RSCD roughness boundaries.

    This module targets the repeated `slight`/`severe` failure mode. It reads
    multi-scale gradient and Laplacian tails before the ConvNeXt stem, suppresses
    snow/marking/specular artifacts, and writes a bounded image-level correction.
    The last projection is zero-initialized so resumed checkpoints have exactly
    the same predictions until this adapter is trained.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        scale: float = 0.020,
        gate_floor: float = 0.10,
        gate_mode: str = "concrete_tail",
        dry_tail_weight: float = 1.0,
        wet_hidden_tail_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.gate_mode = str(gate_mode).strip().lower()
        if self.gate_mode not in {"concrete_tail", "coupled_selective", "tristate_water_concrete_guarded"}:
            raise ValueError(f"unknown scale-space roughness stem gate_mode: {gate_mode}")
        self.use_tristate_water_concrete_fields = self.gate_mode == "tristate_water_concrete_guarded"
        self.dry_tail_weight = max(float(dry_tail_weight), 0.0)
        self.wet_hidden_tail_weight = max(float(wet_hidden_tail_weight), 0.0)
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
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3) / 4.0,
        )
        map_channels = 15 if self.use_tristate_water_concrete_fields else 12
        in_channels = 3 + 3 + 3 + map_channels + 1
        hidden_dim = int(hidden_dim)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden_dim)), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=max(1, hidden_dim), bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )
        last = self.adapter[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _scale_gradient(gray: torch.Tensor, kernel_size: int, sobel_x: torch.Tensor, sobel_y: torch.Tensor) -> torch.Tensor:
        if kernel_size > 1:
            gray = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    @staticmethod
    def _tail_map(x: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
        local = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        local_var = F.avg_pool2d((x - local).square(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        local_std = torch.sqrt(local_var + 1e-6)
        return torch.sigmoid((x - local - 0.35 * local_std) * 26.0)

    def evidence_maps(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return roughness maps [B, 12, H, W] and summary stats [B, 12].

        Channels 9-11 are task-specific gates:
        `concrete_tail`, `dry_concrete_tail`, and `wet_hidden_concrete_tail`.
        They separate visible dry roughness from roughness hidden by water/wet
        film, which is the main RSCD slight/severe coupling ambiguity.
        In `tristate_water_concrete_guarded` mode, channels 12-14 further
        split wet/water concrete evidence into smooth-film, slight-tail, and
        severe-tail fields before the ConvNeXt stem sees the image.
        """

        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        r, g_ch, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
        value = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = ((value - minc) / value.clamp_min(1e-4)).clamp(0.0, 1.0)

        grad_1 = self._scale_gradient(gray, 1, self.sobel_x, self.sobel_y).clamp(0.0, 1.0)
        grad_3 = self._scale_gradient(gray, 3, self.sobel_x, self.sobel_y).clamp(0.0, 1.0)
        grad_7 = self._scale_gradient(gray, 7, self.sobel_x, self.sobel_y).clamp(0.0, 1.0)
        grad_13 = self._scale_gradient(gray, 13, self.sobel_x, self.sobel_y).clamp(0.0, 1.0)
        lap = F.conv2d(gray, self.laplace, padding=1).abs().clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4).clamp(0.0, 1.0)

        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        low_texture = torch.sigmoid((0.055 - grad_1) * 30.0)
        low_contrast = torch.sigmoid((0.050 - contrast) * 30.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = torch.sigmoid((0.42 - value) * 10.0) * torch.sigmoid((0.34 - saturation) * 10.0) * low_texture
        film_erasure = low_texture * low_contrast
        concrete_like = (
            torch.sigmoid((value - 0.38) * 8.0)
            * torch.sigmoid((0.82 - value) * 8.0)
            * torch.sigmoid((0.32 - saturation) * 10.0)
            * (1.0 - snow_like)
            * (1.0 - marking)
        ).clamp(0.0, 1.0)
        artifact_suppression = (1.0 - torch.maximum(marking, snow_like)).clamp(0.0, 1.0)
        visible_tail = torch.maximum(self._tail_map(grad_1), self._tail_map(lap)) * artifact_suppression * (1.0 - 0.45 * specular)
        scale_tail = torch.maximum(self._tail_map(grad_3), torch.maximum(self._tail_map(grad_7), self._tail_map(grad_13)))
        concrete_tail = (0.55 * visible_tail + 0.45 * scale_tail) * concrete_like
        wet_proxy = torch.maximum(specular, dark_water)
        dry_concrete_tail = concrete_like * scale_tail * (1.0 - wet_proxy).clamp(0.0, 1.0) * artifact_suppression
        wet_hidden_concrete_tail = concrete_like * wet_proxy * torch.maximum(scale_tail, film_erasure) * artifact_suppression
        map_list = [
            grad_1,
            grad_3,
            grad_7,
            grad_13,
            lap,
            contrast,
            self._tail_map(grad_1),
            scale_tail,
            concrete_like,
            concrete_tail,
            dry_concrete_tail,
            wet_hidden_concrete_tail,
        ]
        if self.use_tristate_water_concrete_fields:
            tail_strength = (
                0.34 * grad_1
                + 0.30 * grad_7
                + 0.22 * lap
                + 0.14 * contrast
            ).clamp(0.0, 1.0)
            severe_tail = (scale_tail * torch.sigmoid((tail_strength - 0.105) * 20.0)).clamp(0.0, 1.0)
            smooth_film_field = (
                concrete_like
                * wet_proxy
                * film_erasure
                * (1.0 - scale_tail).clamp(0.0, 1.0)
                * artifact_suppression
            ).clamp(0.0, 1.0)
            slight_tail_field = (
                concrete_like
                * wet_proxy
                * scale_tail
                * (1.0 - severe_tail).clamp(0.0, 1.0)
                * artifact_suppression
            ).clamp(0.0, 1.0)
            severe_tail_field = (concrete_like * wet_proxy * severe_tail * artifact_suppression).clamp(0.0, 1.0)
            map_list.extend([smooth_film_field, slight_tail_field, severe_tail_field])
        maps = torch.cat(map_list, dim=1).clamp(0.0, 1.0)
        stats = torch.cat(
            [
                concrete_tail.mean(dim=(2, 3)),
                concrete_tail.std(dim=(2, 3), unbiased=False),
                dry_concrete_tail.mean(dim=(2, 3)),
                dry_concrete_tail.std(dim=(2, 3), unbiased=False),
                wet_hidden_concrete_tail.mean(dim=(2, 3)),
                wet_hidden_concrete_tail.std(dim=(2, 3), unbiased=False),
                maps[:, 7:8].mean(dim=(2, 3)),
                maps[:, 7:8].std(dim=(2, 3), unbiased=False),
                grad_1.mean(dim=(2, 3)),
                grad_7.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        return maps, stats

    def forward(
        self,
        image: torch.Tensor,
        *,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.scale <= 0.0:
            if return_stats:
                return image, image.new_zeros((image.size(0), 8))
            return image
        maps, stats = self.evidence_maps(image)
        low = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        high = image - low
        if self.gate_mode == "coupled_selective":
            dry_gate = self.dry_tail_weight * maps[:, 10:11]
            wet_gate = self.wet_hidden_tail_weight * maps[:, 11:12]
            gate_base = torch.maximum(dry_gate, wet_gate).clamp(0.0, 1.0)
        elif self.gate_mode == "tristate_water_concrete_guarded":
            dry_gate = self.dry_tail_weight * maps[:, 10:11]
            smooth_film = maps[:, 12:13]
            slight_tail = maps[:, 13:14]
            severe_tail = maps[:, 14:15]
            wet_gate = self.wet_hidden_tail_weight * (
                0.55 * smooth_film + 1.10 * slight_tail + 0.48 * severe_tail
            )
            gate_base = torch.maximum(dry_gate, wet_gate).clamp(0.0, 1.0)
        else:
            gate_base = maps[:, 9:10]
        gate = (self.gate_floor + (1.0 - self.gate_floor) * gate_base).to(dtype=image.dtype)
        fields = torch.cat([image, low, high, maps.to(dtype=image.dtype), gate], dim=1)
        delta = self.scale * gate * torch.tanh(self.adapter(fields))
        adapted = image + delta
        if return_stats:
            return adapted, stats.to(dtype=image.dtype)
        return adapted


class FactorQueryDecoder(nn.Module):
    """Decode friction/material/roughness/coupling tokens.

    The default path preserves the original global-token decoder. When spatial
    factor queries are enabled, the four factor tokens additionally attend to
    the backbone feature map plus local physics evidence maps. The spatial
    residual is zero-initialized so resumed calibrated checkpoints start from
    the same predictions and can learn only if local evidence helps.
    """

    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        evidence_dim: int = 16,
        hidden_dim: int = 512,
        *,
        spatial_map_dim: int | None = None,
        spatial_evidence_dim: int = 10,
        spatial_heads: int = 4,
        spatial_scale: float = 0.25,
    ) -> None:
        super().__init__()
        input_dim = int(in_dim) + int(evidence_dim)
        token_dim = int(token_dim)
        self.queries = nn.Parameter(torch.randn(4, int(token_dim)) * 0.02)
        self.use_spatial_queries = spatial_map_dim is not None
        self.spatial_scale = float(spatial_scale)

        def projector() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), token_dim),
            )

        self.projectors = nn.ModuleDict(
            {
                "friction": projector(),
                "material": projector(),
                "roughness": projector(),
                "coupling": projector(),
            }
        )
        if self.use_spatial_queries:
            map_input_dim = int(spatial_map_dim) + int(spatial_evidence_dim)
            self.spatial_key = nn.Sequential(
                nn.Conv2d(map_input_dim, token_dim, kernel_size=1, bias=False),
                nn.GELU(),
            )
            self.spatial_value = nn.Sequential(
                nn.Conv2d(map_input_dim, token_dim, kernel_size=1, bias=False),
                nn.GELU(),
            )
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=max(1, int(spatial_heads)),
                batch_first=True,
            )
            self.spatial_out = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, token_dim),
            )
            last = self.spatial_out[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        else:
            self.spatial_key = None
            self.spatial_value = None
            self.spatial_attn = None
            self.spatial_out = None

    def forward(
        self,
        feature: torch.Tensor,
        evidence_stats: torch.Tensor,
        *,
        spatial_map: torch.Tensor | None = None,
        spatial_evidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([feature, evidence_stats.to(dtype=feature.dtype)], dim=1)
        names = ("friction", "material", "roughness", "coupling")
        tokens = {
            name: self.projectors[name](x) + self.queries[idx].to(dtype=feature.dtype).unsqueeze(0)
            for idx, name in enumerate(names)
        }
        if (
            self.use_spatial_queries
            and self.spatial_key is not None
            and self.spatial_value is not None
            and self.spatial_attn is not None
            and self.spatial_out is not None
            and spatial_map is not None
            and spatial_evidence is not None
        ):
            spatial_evidence = F.interpolate(
                spatial_evidence,
                size=spatial_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(device=spatial_map.device, dtype=spatial_map.dtype)
            kv = torch.cat([spatial_map, spatial_evidence], dim=1)
            key = self.spatial_key(kv).flatten(2).transpose(1, 2)
            value = self.spatial_value(kv).flatten(2).transpose(1, 2)
            query = torch.stack([tokens[name] for name in names], dim=1)
            attended, _ = self.spatial_attn(query, key, value, need_weights=False)
            residual = self.spatial_scale * self.spatial_out(attended)
            tokens = {name: tokens[name] + residual[:, idx] for idx, name in enumerate(names)}
        return tokens


class RoughnessReliability(nn.Module):
    """Mix visible roughness and friction/material conditional prior.

    When `use_coupling_context` is enabled, the reliability scalar follows the
    C3-FaRNet design objective: rho_R is predicted from the coupling token plus
    physics evidence. The default keeps the historical evidence-only interface
    so old checkpoints and baseline configs remain compatible.
    """

    def __init__(
        self,
        token_dim: int,
        evidence_dim: int = 16,
        hidden_dim: int = 128,
        *,
        use_coupling_context: bool = False,
    ) -> None:
        super().__init__()
        self.use_coupling_context = bool(use_coupling_context)
        rho_dim = int(evidence_dim) + (int(token_dim) if self.use_coupling_context else 0)
        self.rho = nn.Sequential(
            nn.LayerNorm(rho_dim),
            nn.Linear(rho_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.prior = nn.Sequential(
            nn.LayerNorm(2 * int(token_dim)),
            nn.Linear(2 * int(token_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(token_dim)),
        )

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_r_visible: torch.Tensor,
        evidence_stats: torch.Tensor,
        z_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho_input = evidence_stats.to(dtype=z_r_visible.dtype)
        if self.use_coupling_context:
            if z_c is None:
                z_c = torch.zeros_like(z_r_visible)
            rho_input = torch.cat([z_c.to(dtype=z_r_visible.dtype), rho_input], dim=1)
        rho = torch.sigmoid(self.rho(rho_input))
        prior = self.prior(torch.cat([z_f, z_m], dim=1))
        z_r = rho * z_r_visible + (1.0 - rho) * prior
        return z_r, rho


class PseudoRoughnessAwareReliability(nn.Module):
    """Tri-state roughness reliability for wet/water concrete boundaries.

    RSCD roughness labels are visually coupled with water film and debris:
    rough-looking pixels may be true geometric roughness, hidden roughness under
    water film, or pseudo-roughness from debris/reflection. This module does not
    classify from handcrafted values. It uses those values to condition a small,
    zero-initialized residual on the roughness token and reliability scalar.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        evidence_dim: int = 16,
        value_dim: int = 20,
        hidden_dim: int = 128,
        scale: float = 0.06,
        rho_scale: float = 0.10,
        gate_floor: float = 0.0,
        dropout: float = 0.0,
        detach_context: bool = False,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.rho_scale = max(float(rho_scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.detach_context = bool(detach_context)
        context_dim = 4 * int(token_dim) + int(evidence_dim) + int(value_dim) + 7
        self.state_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 3),
        )
        self.token_residual = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(token_dim)),
        )
        self.rho_residual = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        for module in (self.token_residual[-1], self.rho_residual[-1]):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _hand_states(value_vector: torch.Tensor, evidence_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        macro_rough = value_vector[:, 0:1].clamp(0.0, 1.0)
        micro_rough = value_vector[:, 1:2].clamp(0.0, 1.0)
        film = value_vector[:, 2:3].clamp(0.0, 1.0)
        artifact = value_vector[:, 3:4].clamp(0.0, 1.0)
        dark_water = torch.maximum(value_vector[:, 12:13], value_vector[:, 13:14]).clamp(0.0, 1.0)
        specular = torch.maximum(value_vector[:, 14:15], value_vector[:, 15:16]).clamp(0.0, 1.0)
        erasure = torch.maximum(value_vector[:, 16:17], value_vector[:, 17:18]).clamp(0.0, 1.0)
        rough_visibility = C3PhysicsEvidenceStats.roughness_reliability_target(evidence_stats).clamp(0.0, 1.0)
        wetness = evidence_stats[:, 10:11].clamp(0.0, 1.0)

        geometric_rough = torch.maximum(macro_rough, 0.65 * micro_rough).clamp(0.0, 1.0)
        true_visible = (geometric_rough * (1.0 - 0.55 * artifact) * (0.45 + 0.55 * rough_visibility)).clamp(0.0, 1.0)
        hidden_by_film = (torch.maximum(film, dark_water) * erasure * (1.0 - 0.35 * rough_visibility)).clamp(0.0, 1.0)
        pseudo_rough = (
            torch.maximum(artifact, 0.65 * specular)
            * torch.maximum(micro_rough, erasure)
            * (1.0 - 0.55 * macro_rough)
        ).clamp(0.0, 1.0)
        film_conflict = (torch.maximum(film, dark_water) * geometric_rough).clamp(0.0, 1.0)
        hand_gate = torch.maximum(
            torch.maximum(hidden_by_film, pseudo_rough),
            film_conflict * torch.maximum(wetness, film),
        ).clamp(0.0, 1.0)
        states = torch.cat(
            [
                true_visible,
                hidden_by_film,
                pseudo_rough,
                film_conflict,
                rough_visibility,
                wetness,
                geometric_rough,
            ],
            dim=1,
        )
        return states, hand_gate

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_r_visible: torch.Tensor,
        z_r: torch.Tensor,
        rho: torch.Tensor,
        evidence_stats: torch.Tensor,
        value_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        value_vector = value_vector.to(device=z_r.device, dtype=z_r.dtype)
        evidence_stats = evidence_stats.to(device=z_r.device, dtype=z_r.dtype)
        states, hand_gate = self._hand_states(value_vector, evidence_stats)
        context = torch.cat(
            [
                z_f,
                z_m,
                z_r_visible,
                z_r,
                evidence_stats,
                value_vector,
                states.to(dtype=z_r.dtype),
            ],
            dim=1,
        )
        if self.detach_context:
            context = context.detach()
        state_probs = torch.softmax(self.state_head(context), dim=1)
        learned_gate = (state_probs[:, 1:2] + state_probs[:, 2:3] + 0.5 * state_probs[:, 0:1]).clamp(0.0, 1.0)
        gate = (hand_gate.to(dtype=z_r.dtype) * learned_gate).clamp(0.0, 1.0)
        if self.gate_floor > 0.0:
            gate = self.gate_floor + (1.0 - self.gate_floor) * gate
        token_delta = self.scale * gate * torch.tanh(self.token_residual(context))
        rho_delta = self.rho_scale * gate * torch.tanh(self.rho_residual(context))
        rho_refined = (rho + rho_delta).clamp(1e-4, 1.0 - 1e-4)
        return z_r + token_delta, rho_refined, {
            "states": states,
            "state_probs": state_probs,
            "gate": gate,
            "hand_gate": hand_gate.to(dtype=z_r.dtype),
            "learned_gate": learned_gate,
            "token_delta": token_delta,
            "rho_delta": rho_delta,
        }


class ScaleSpaceRoughnessTokenConditioner(nn.Module):
    """Condition roughness/coupling tokens with RSCD scale-space evidence.

    Unlike the input stem adapter, this module never changes pixels entering
    ConvNeXt. It extracts dry-concrete roughness tails and wet/water hidden-tail
    cues from the image, then writes a zero-initialized residual directly to the
    roughness and coupling tokens. The target is the RSCD coupling boundary:
    water/wet + concrete + slight/severe and dry-concrete slight/severe.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int = 64,
        scale: float = 0.10,
        gate_floor: float = 0.0,
        dry_tail_weight: float = 1.0,
        wet_hidden_tail_weight: float = 0.75,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.dry_tail_weight = max(float(dry_tail_weight), 0.0)
        self.wet_hidden_tail_weight = max(float(wet_hidden_tail_weight), 0.0)
        self.extractor = ScaleSpaceRoughnessStemAdapter(
            hidden_dim=1,
            scale=0.0,
            gate_floor=0.0,
            gate_mode="coupled_selective",
        )
        for param in self.extractor.adapter.parameters():
            param.requires_grad_(False)

        stats_dim = 12
        hidden_dim = int(hidden_dim)

        def projector() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(stats_dim),
                nn.Linear(stats_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, int(token_dim)),
            )

        self.roughness_projector = projector()
        self.coupling_projector = projector()
        for module in (self.roughness_projector[-1], self.coupling_projector[-1]):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        image: torch.Tensor,
        tokens: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.scale <= 0.0:
            batch = image.size(0)
            token = tokens["roughness"]
            zero_stats = token.new_zeros((batch, 12))
            return tokens, {
                "stats": zero_stats,
                "gate": token.new_zeros((batch, 1)),
                "roughness_delta": torch.zeros_like(token),
                "coupling_delta": torch.zeros_like(tokens["coupling"]),
            }

        _, stats = self.extractor.evidence_maps(image)
        stats = stats.to(device=tokens["roughness"].device, dtype=tokens["roughness"].dtype)
        dry_tail = stats[:, 2:3] + 0.5 * stats[:, 3:4]
        wet_hidden_tail = stats[:, 4:5] + 0.5 * stats[:, 5:6]
        gate_base = (self.dry_tail_weight * dry_tail + self.wet_hidden_tail_weight * wet_hidden_tail).clamp(0.0, 1.0)
        gate = self.gate_floor + (1.0 - self.gate_floor) * gate_base
        rough_delta = self.scale * gate * torch.tanh(self.roughness_projector(stats))
        coupling_delta = self.scale * gate * torch.tanh(self.coupling_projector(stats))
        updated = dict(tokens)
        updated["roughness"] = updated["roughness"] + rough_delta
        updated["coupling"] = updated["coupling"] + coupling_delta
        return updated, {
            "stats": stats,
            "gate": gate,
            "roughness_delta": rough_delta,
            "coupling_delta": coupling_delta,
        }


class LocalGlobalScaleTokenConditioner(nn.Module):
    """Fuse local, global, and scale evidence into RSCD factor tokens.

    This is a task-adapted mid-level mechanism for the current FAF network. It
    does not classify from handcrafted values and does not add a late logit
    patch. Instead it builds a context from:

    - global PhysicsTexture evidence: wet film, dark water, roughness, erasure;
    - LocalPhysicsField evidence: local weak mask/statistics already used by
      the backbone residual path;
    - scale-space roughness evidence: visible dry roughness tails and wet-hidden
      concrete tails.

    The context writes a bounded, zero-initialized residual only to roughness and
    coupling tokens, which targets the RSCD water/wet + concrete + roughness
    ambiguity without globally moving unrelated class logits at initialization.
    """

    stats_dim = 12
    mechanism_dim = 6

    def __init__(
        self,
        *,
        feature_dim: int = 0,
        token_dim: int,
        evidence_dim: int,
        local_dim: int,
        hidden_dim: int = 96,
        scale: float = 0.050,
        feature_scale: float = 0.010,
        gate_floor: float = 0.0,
        dropout: float = 0.0,
        detach_context: bool = False,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.feature_scale = max(float(feature_scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.local_dim = int(local_dim)
        self.detach_context = bool(detach_context)
        self.extractor = ScaleSpaceRoughnessStemAdapter(
            hidden_dim=1,
            scale=0.0,
            gate_floor=0.0,
            gate_mode="coupled_selective",
        )
        for param in self.extractor.adapter.parameters():
            param.requires_grad_(False)

        input_dim = self.stats_dim + int(evidence_dim) + self.local_dim + self.mechanism_dim
        hidden_dim = int(hidden_dim)

        def projector(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, int(out_dim)),
            )

        self.feature_projector = projector(int(feature_dim)) if int(feature_dim) > 0 else None
        self.roughness_projector = projector(int(token_dim))
        self.coupling_projector = projector(int(token_dim))
        self.learned_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        zero_init_layers = [self.roughness_projector[-1], self.coupling_projector[-1]]
        if self.feature_projector is not None:
            zero_init_layers.append(self.feature_projector[-1])
        for module in zero_init_layers:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        gate_last = self.learned_gate[-1]
        if isinstance(gate_last, nn.Linear):
            nn.init.zeros_(gate_last.weight)
            nn.init.constant_(gate_last.bias, -1.5)

    @staticmethod
    def _mechanism_features(scale_stats: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        concrete_tail = scale_stats[:, 0:1].clamp(0.0, 1.0)
        concrete_tail_std = scale_stats[:, 1:2].clamp(0.0, 1.0)
        dry_tail = scale_stats[:, 2:3].clamp(0.0, 1.0)
        dry_tail_std = scale_stats[:, 3:4].clamp(0.0, 1.0)
        wet_hidden_tail = scale_stats[:, 4:5].clamp(0.0, 1.0)
        wet_hidden_tail_std = scale_stats[:, 5:6].clamp(0.0, 1.0)
        scale_tail = scale_stats[:, 6:7].clamp(0.0, 1.0)
        scale_tail_std = scale_stats[:, 7:8].clamp(0.0, 1.0)
        lap_mean = scale_stats[:, 10:11].clamp(0.0, 1.0)

        specular = evidence[:, 8:9].clamp(0.0, 1.0)
        dark_water = evidence[:, 9:10].clamp(0.0, 1.0)
        wet = evidence[:, 10:11].clamp(0.0, 1.0)
        rough = evidence[:, 11:12].clamp(0.0, 1.0)
        erasure = evidence[:, 12:13].clamp(0.0, 1.0)
        snow = evidence[:, 13:14].clamp(0.0, 1.0)
        ice = evidence[:, 14:15].clamp(0.0, 1.0)

        visible_rough = (
            0.34 * dry_tail
            + 0.22 * dry_tail_std
            + 0.20 * scale_tail
            + 0.14 * scale_tail_std
            + 0.10 * rough
        ).clamp(0.0, 1.0)
        hidden_rough = (
            0.30 * wet_hidden_tail
            + 0.18 * wet_hidden_tail_std
            + 0.20 * wet
            + 0.14 * dark_water
            + 0.10 * erasure
            + 0.08 * specular
        ).clamp(0.0, 1.0)
        concrete_context = (0.46 * concrete_tail + 0.24 * concrete_tail_std + 0.16 * dry_tail + 0.14 * wet_hidden_tail).clamp(0.0, 1.0)
        film_context = (0.36 * wet + 0.22 * dark_water + 0.20 * specular + 0.22 * erasure).clamp(0.0, 1.0)
        rough_visibility = C3PhysicsEvidenceStats.roughness_reliability_target(evidence).clamp(0.0, 1.0)
        artifact_guard = (1.0 - 0.72 * torch.maximum(snow, ice)).clamp(0.08, 1.0)
        scale_consistency = (0.45 * scale_tail + 0.28 * scale_tail_std + 0.17 * lap_mean + 0.10 * rough).clamp(0.0, 1.0)
        water_concrete_coupling = (concrete_context * torch.maximum(hidden_rough, film_context)).clamp(0.0, 1.0)
        return torch.cat(
            [
                visible_rough,
                hidden_rough,
                concrete_context,
                film_context,
                rough_visibility,
                scale_consistency * artifact_guard,
            ],
            dim=1,
        ), water_concrete_coupling, artifact_guard

    def prepare(
        self,
        image: torch.Tensor,
        evidence: torch.Tensor,
        local_physics_feature: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        _, scale_stats = self.extractor.evidence_maps(image)
        scale_stats = scale_stats.to(device=device, dtype=dtype)
        evidence = evidence.to(device=device, dtype=dtype)
        if local_physics_feature is None:
            local = scale_stats.new_zeros((scale_stats.size(0), self.local_dim))
        else:
            local = local_physics_feature.to(device=device, dtype=dtype)
            if local.size(1) != self.local_dim:
                raise ValueError(
                    f"local physics feature dim mismatch: expected {self.local_dim}, got {local.size(1)}"
                )
        mechanism, water_concrete_coupling, artifact_guard = self._mechanism_features(scale_stats, evidence)
        context = torch.cat([scale_stats, evidence, local, mechanism], dim=1)
        if self.detach_context:
            context = context.detach()
        learned_gate = torch.sigmoid(self.learned_gate(context))
        hand_gate = torch.maximum(mechanism[:, 0:1], mechanism[:, 1:2])
        hand_gate = torch.maximum(hand_gate, water_concrete_coupling)
        hand_gate = (hand_gate * artifact_guard).clamp(0.0, 1.0)
        gate = self.gate_floor + (1.0 - self.gate_floor) * learned_gate * hand_gate
        return {
            "context": context,
            "scale_stats": scale_stats,
            "mechanism": mechanism,
            "gate": gate,
            "learned_gate": learned_gate,
            "hand_gate": hand_gate,
        }

    def condition_feature(
        self,
        feature: torch.Tensor,
        prepared: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_projector is None or self.feature_scale <= 0.0:
            return feature, torch.zeros_like(feature)
        gate = prepared["gate"].to(device=feature.device, dtype=feature.dtype)
        context = prepared["context"].to(device=feature.device, dtype=feature.dtype)
        delta = self.feature_scale * gate * torch.tanh(self.feature_projector(context))
        return feature + delta, delta

    def condition_tokens(
        self,
        tokens: dict[str, torch.Tensor],
        prepared: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        token = tokens["roughness"]
        if self.scale <= 0.0:
            batch = token.size(0)
            return tokens, {
                "scale_stats": token.new_zeros((batch, self.stats_dim)),
                "mechanism": token.new_zeros((batch, self.mechanism_dim)),
                "gate": token.new_zeros((batch, 1)),
                "learned_gate": token.new_zeros((batch, 1)),
                "hand_gate": token.new_zeros((batch, 1)),
                "roughness_delta": torch.zeros_like(token),
                "coupling_delta": torch.zeros_like(tokens["coupling"]),
            }
        context = prepared["context"].to(device=token.device, dtype=token.dtype)
        gate = prepared["gate"].to(device=token.device, dtype=token.dtype)
        rough_delta = self.scale * gate * torch.tanh(self.roughness_projector(context))
        coupling_delta = self.scale * gate * torch.tanh(self.coupling_projector(context))
        updated = dict(tokens)
        updated["roughness"] = updated["roughness"] + rough_delta
        updated["coupling"] = updated["coupling"] + coupling_delta
        return updated, {
            "scale_stats": prepared["scale_stats"],
            "mechanism": prepared["mechanism"],
            "gate": gate,
            "learned_gate": prepared["learned_gate"],
            "hand_gate": prepared["hand_gate"],
            "roughness_delta": rough_delta,
            "coupling_delta": coupling_delta,
        }

    def forward(
        self,
        image: torch.Tensor,
        tokens: dict[str, torch.Tensor],
        evidence: torch.Tensor,
        local_physics_feature: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        prepared = self.prepare(
            image,
            evidence,
            local_physics_feature,
            device=tokens["roughness"].device,
            dtype=tokens["roughness"].dtype,
        )
        return self.condition_tokens(tokens, prepared)


class WaterFilmRoughnessFeatureFiLM(nn.Module):
    """Feature-level FiLM for water-film and hidden-roughness coupling.

    Earlier value-only experiments showed that wet/water concrete errors are
    not solved by a standalone handcrafted classifier. This module instead
    turns the same physical evidence into an identity-initialized channel-wise
    modulation of the learned feature entering the classifier. The hand gate is
    active only when concrete context, water film/texture erasure, and visible
    or hidden roughness cues agree, so the mechanism is tied to RSCD's coupled
    water/wet + concrete + roughness boundary rather than a generic feature
    residual.
    """

    stats_dim = 12
    mechanism_dim = 6

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 128,
        scale: float = 0.080,
        gate_floor: float = 0.0,
        max_gamma: float = 0.18,
        dropout: float = 0.0,
        detach_context: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scale = max(float(scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.max_gamma = max(float(max_gamma), 0.0)
        self.detach_context = bool(detach_context)
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
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3) / 4.0,
        )
        input_dim = self.stats_dim + C3PhysicsEvidenceStats.out_dim + self.mechanism_dim + 5
        hidden_dim = int(hidden_dim)
        self.modulator = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 2 * self.feature_dim + 1),
        )
        last = self.modulator[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _scale_gradient(self, gray: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size > 1:
            gray = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6).clamp(0.0, 1.0)

    @staticmethod
    def _tail_map(x: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
        local = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        local_var = F.avg_pool2d((x - local).square(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        local_std = torch.sqrt(local_var + 1e-6)
        return torch.sigmoid((x - local - 0.35 * local_std) * 26.0)

    def _scale_stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 128:
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
        r, g_ch, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
        value = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = ((value - minc) / value.clamp_min(1e-4)).clamp(0.0, 1.0)
        grad_1 = self._scale_gradient(gray, 1)
        grad_3 = self._scale_gradient(gray, 3)
        grad_7 = self._scale_gradient(gray, 7)
        grad_13 = self._scale_gradient(gray, 13)
        lap = F.conv2d(gray, self.laplace, padding=1).abs().clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4).clamp(0.0, 1.0)

        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        low_texture = torch.sigmoid((0.055 - grad_1) * 30.0)
        low_contrast = torch.sigmoid((0.050 - contrast) * 30.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_water = torch.sigmoid((0.42 - value) * 10.0) * torch.sigmoid((0.34 - saturation) * 10.0) * low_texture
        film_erasure = low_texture * low_contrast
        concrete_like = (
            torch.sigmoid((value - 0.38) * 8.0)
            * torch.sigmoid((0.82 - value) * 8.0)
            * torch.sigmoid((0.32 - saturation) * 10.0)
            * (1.0 - snow_like)
            * (1.0 - marking)
        ).clamp(0.0, 1.0)
        artifact_suppression = (1.0 - torch.maximum(marking, snow_like)).clamp(0.0, 1.0)
        visible_tail = torch.maximum(self._tail_map(grad_1), self._tail_map(lap)) * artifact_suppression * (1.0 - 0.45 * specular)
        scale_tail = torch.maximum(self._tail_map(grad_3), torch.maximum(self._tail_map(grad_7), self._tail_map(grad_13)))
        concrete_tail = (0.55 * visible_tail + 0.45 * scale_tail) * concrete_like
        wet_proxy = torch.maximum(specular, dark_water)
        dry_concrete_tail = concrete_like * scale_tail * (1.0 - wet_proxy).clamp(0.0, 1.0) * artifact_suppression
        wet_hidden_concrete_tail = concrete_like * wet_proxy * torch.maximum(scale_tail, film_erasure) * artifact_suppression
        return torch.cat(
            [
                concrete_tail.mean(dim=(2, 3)),
                concrete_tail.std(dim=(2, 3), unbiased=False),
                dry_concrete_tail.mean(dim=(2, 3)),
                dry_concrete_tail.std(dim=(2, 3), unbiased=False),
                wet_hidden_concrete_tail.mean(dim=(2, 3)),
                wet_hidden_concrete_tail.std(dim=(2, 3), unbiased=False),
                scale_tail.mean(dim=(2, 3)),
                scale_tail.std(dim=(2, 3), unbiased=False),
                grad_1.mean(dim=(2, 3)),
                grad_7.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
            ],
            dim=1,
        )

    def forward(
        self,
        image: torch.Tensor,
        feature: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.scale <= 0.0:
            return feature, {
                "scale_stats": feature.new_zeros((feature.size(0), self.stats_dim)),
                "mechanism": feature.new_zeros((feature.size(0), self.mechanism_dim)),
                "gate": feature.new_zeros((feature.size(0), 1)),
                "hand_gate": feature.new_zeros((feature.size(0), 1)),
                "learned_gate": feature.new_zeros((feature.size(0), 1)),
                "feature_delta": torch.zeros_like(feature),
            }
        scale_stats = self._scale_stats(image).to(device=feature.device, dtype=feature.dtype)
        evidence = evidence.to(device=feature.device, dtype=feature.dtype)
        mechanism, water_concrete_coupling, artifact_guard = LocalGlobalScaleTokenConditioner._mechanism_features(
            scale_stats,
            evidence,
        )
        visible_rough = mechanism[:, 0:1]
        hidden_rough = mechanism[:, 1:2]
        concrete_context = mechanism[:, 2:3]
        film_context = mechanism[:, 3:4]
        rough_visibility = mechanism[:, 4:5]
        scale_consistency = mechanism[:, 5:6]
        dry_visible_gate = concrete_context * visible_rough * (1.0 - film_context).clamp(0.0, 1.0)
        wet_hidden_gate = water_concrete_coupling * torch.maximum(hidden_rough, film_context) * (1.0 - rough_visibility).clamp(0.0, 1.0)
        rough_conflict = (torch.maximum(hidden_rough, film_context) - visible_rough).clamp(0.0, 1.0)
        hand_gate = torch.maximum(0.70 * dry_visible_gate, wet_hidden_gate)
        hand_gate = torch.maximum(hand_gate, 0.55 * rough_conflict * concrete_context * scale_consistency)
        hand_gate = (hand_gate * artifact_guard).clamp(0.0, 1.0)
        context = torch.cat(
            [
                scale_stats,
                evidence,
                mechanism,
                water_concrete_coupling,
                artifact_guard,
                dry_visible_gate,
                wet_hidden_gate,
                rough_conflict,
            ],
            dim=1,
        )
        if self.detach_context:
            context = context.detach()
        raw = self.modulator(context)
        gamma_raw, beta_raw, gate_raw = torch.split(raw, [self.feature_dim, self.feature_dim, 1], dim=1)
        learned_gate = torch.sigmoid(gate_raw)
        gate = self.gate_floor + (1.0 - self.gate_floor) * hand_gate * learned_gate
        feature_norm = F.layer_norm(feature, (feature.size(1),))
        gamma = self.max_gamma * torch.tanh(gamma_raw)
        beta = torch.tanh(beta_raw)
        delta = self.scale * gate * (feature_norm * gamma + beta)
        return feature + delta, {
            "scale_stats": scale_stats,
            "mechanism": mechanism,
            "gate": gate,
            "hand_gate": hand_gate,
            "learned_gate": learned_gate,
            "feature_delta": delta,
        }


class PairValueMechanismConditioner(nn.Module):
    """Mid-level conditioning from diagnosed pair-value mechanisms.

    This module is not a standalone value classifier and does not write a final
    logit residual. It reads the value families found in the hard-error audit
    and applies a bounded, zero-initialized residual before the calibrated heads:
    a small global-feature update plus roughness/coupling token updates. The
    handcrafted gates keep the learned residual tied to RSCD mechanisms instead
    of generic feature concatenation.
    """

    value_dim = 20
    gate_dim = 6

    def __init__(
        self,
        *,
        feature_dim: int,
        token_dim: int,
        hidden_dim: int = 64,
        feature_scale: float = 0.010,
        token_scale: float = 0.060,
        gate_floor: float = 0.0,
        value_aug_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_scale = max(float(feature_scale), 0.0)
        self.token_scale = max(float(token_scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.value_aug_std = max(float(value_aug_std), 0.0)
        input_dim = self.value_dim + self.gate_dim
        hidden_dim = int(hidden_dim)

        def projector(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, int(out_dim)),
            )

        self.feature_projector = projector(int(feature_dim))
        self.roughness_projector = projector(int(token_dim))
        self.coupling_projector = projector(int(token_dim))
        for module in (self.feature_projector[-1], self.roughness_projector[-1], self.coupling_projector[-1]):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _mechanism_gates(value_vector: torch.Tensor) -> torch.Tensor:
        macro_rough = value_vector[:, 0:1].clamp(0.0, 1.0)
        micro_rough = value_vector[:, 1:2].clamp(0.0, 1.0)
        film = value_vector[:, 2:3].clamp(0.0, 1.0)
        artifact = value_vector[:, 3:4].clamp(0.0, 1.0)
        saturation = value_vector[:, 4:5].clamp(0.0, 1.0)
        macro_mean = value_vector[:, 5:6].clamp(0.0, 1.0)
        macro_std = value_vector[:, 6:7].clamp(0.0, 1.0)
        meso_std = value_vector[:, 7:8].clamp(0.0, 1.0)
        micro_std = value_vector[:, 8:9].clamp(0.0, 1.0)
        lap_std = value_vector[:, 9:10].clamp(0.0, 1.0)
        grad_std = value_vector[:, 10:11].clamp(0.0, 1.0)
        anisotropy = value_vector[:, 11:12].clamp(0.0, 1.0)
        dark_water = value_vector[:, 12:13].clamp(0.0, 1.0)
        dark_water_top = value_vector[:, 13:14].clamp(0.0, 1.0)
        texture_erasure = value_vector[:, 16:17].clamp(0.0, 1.0)
        texture_erasure_top = value_vector[:, 17:18].clamp(0.0, 1.0)
        value_mean = value_vector[:, 18:19].clamp(0.0, 1.0)

        dry_concrete_smooth_slight = (
            0.34 * meso_std + 0.24 * macro_mean + 0.22 * texture_erasure_top + 0.20 * grad_std
        ).clamp(0.0, 1.0)
        dry_concrete_slight_severe = (0.45 * macro_rough + 0.30 * macro_std + 0.15 * anisotropy + 0.10 * meso_std).clamp(0.0, 1.0)
        water_asphalt_smooth_slight = (0.42 * film + 0.26 * texture_erasure_top + 0.18 * micro_std + 0.14 * lap_std).clamp(0.0, 1.0)
        water_concrete_smooth_slight = (
            0.34 * dark_water_top + 0.28 * dark_water + 0.20 * value_mean + 0.18 * texture_erasure
        ).clamp(0.0, 1.0)
        wet_concrete_slight_severe = (0.34 * anisotropy + 0.26 * macro_rough + 0.22 * texture_erasure_top + 0.18 * saturation).clamp(0.0, 1.0)
        water_gravel_mud = (0.34 * micro_rough + 0.24 * micro_std + 0.22 * lap_std + 0.20 * grad_std).clamp(0.0, 1.0)
        gates = torch.cat(
            [
                dry_concrete_smooth_slight,
                dry_concrete_slight_severe,
                water_asphalt_smooth_slight,
                water_concrete_smooth_slight,
                wet_concrete_slight_severe,
                water_gravel_mud,
            ],
            dim=1,
        )
        artifact_guard = (1.0 - 0.65 * artifact).clamp(0.15, 1.0)
        return (gates * artifact_guard).clamp(0.0, 1.0)

    def prepare(self, value_vector: torch.Tensor) -> dict[str, torch.Tensor]:
        values = value_vector[:, : self.value_dim].clamp(0.0, 1.0)
        if self.training and self.value_aug_std > 0.0:
            values = (values + torch.randn_like(values) * float(self.value_aug_std)).clamp(0.0, 1.0)
        gates = self._mechanism_gates(values)
        gate = gates.max(dim=1, keepdim=True).values
        gate = self.gate_floor + (1.0 - self.gate_floor) * gate
        return {
            "input": torch.cat([values, gates], dim=1),
            "gates": gates,
            "gate": gate,
        }

    def condition_feature(
        self,
        feature: torch.Tensor,
        prepared: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_scale <= 0.0:
            return feature, torch.zeros_like(feature)
        x = prepared["input"].to(device=feature.device, dtype=feature.dtype)
        gate = prepared["gate"].to(device=feature.device, dtype=feature.dtype)
        delta = self.feature_scale * gate * torch.tanh(self.feature_projector(x))
        return feature + delta, delta

    def condition_tokens(
        self,
        tokens: dict[str, torch.Tensor],
        prepared: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        roughness = tokens["roughness"]
        coupling = tokens["coupling"]
        if self.token_scale <= 0.0:
            return tokens, {
                "roughness_delta": torch.zeros_like(roughness),
                "coupling_delta": torch.zeros_like(coupling),
            }
        x = prepared["input"].to(device=roughness.device, dtype=roughness.dtype)
        gate = prepared["gate"].to(device=roughness.device, dtype=roughness.dtype)
        rough_delta = self.token_scale * gate * torch.tanh(self.roughness_projector(x))
        coupling_delta = self.token_scale * gate * torch.tanh(self.coupling_projector(x))
        updated = dict(tokens)
        updated["roughness"] = updated["roughness"] + rough_delta
        updated["coupling"] = updated["coupling"] + coupling_delta
        return updated, {
            "roughness_delta": rough_delta,
            "coupling_delta": coupling_delta,
        }


class CoupledFormExpertConditioner(nn.Module):
    """Pair-value conditioner with separate experts for different RSCD couplings.

    Earlier pair-value experiments used one shared context and one max gate for
    all hard boundaries. That is too coarse for RSCD because the visual coupling
    form differs across boundaries: dry-concrete roughness is mainly geometric,
    water/wet concrete mixes true roughness with water-film occlusion, and
    asphalt water/smooth cues are dominated by dark water-film coverage. This
    module keeps those mechanisms separated and writes only zero-initialized
    feature/token residuals before the calibrated decision path.
    """

    value_dim = 20
    form_names = (
        "dry_concrete_smooth_slight",
        "dry_concrete_slight_severe",
        "water_asphalt_smooth_slight",
        "water_concrete_smooth_slight",
        "wet_concrete_slight_severe",
        "water_gravel_mud",
    )

    def __init__(
        self,
        *,
        feature_dim: int,
        token_dim: int,
        hidden_dim: int = 64,
        feature_scale: float = 0.010,
        token_scale: float = 0.060,
        gate_floor: float = 0.0,
        value_aug_std: float = 0.0,
        learned_gate_bias: float = -1.5,
        detach_context: bool = False,
    ) -> None:
        super().__init__()
        self.feature_scale = max(float(feature_scale), 0.0)
        self.token_scale = max(float(token_scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.value_aug_std = max(float(value_aug_std), 0.0)
        self.detach_context = bool(detach_context)
        self.num_forms = len(self.form_names)
        input_dim = self.value_dim + 3
        hidden_dim = int(hidden_dim)

        masks = torch.zeros(self.num_forms, self.value_dim)
        # value vector indices:
        # 0 macro rough, 1 micro rough, 2 film, 3 artifact, 4 saturation,
        # 5 macro mean, 6 macro std, 7 meso std, 8 micro std, 9 lap std,
        # 10 grad std, 11 anisotropy, 12/13 dark water mean/top,
        # 14/15 specular mean/top, 16/17 texture erasure mean/top,
        # 18/19 value mean/std.
        selected = (
            (5, 7, 10, 16, 17, 18),
            (0, 5, 6, 7, 10, 11, 16, 17),
            (1, 2, 8, 9, 12, 13, 16, 17, 18),
            (2, 7, 12, 13, 16, 17, 18, 19),
            (0, 4, 6, 7, 11, 14, 15, 16, 17),
            (1, 8, 9, 10, 11, 18, 19),
        )
        for row, cols in enumerate(selected):
            masks[row, list(cols)] = 1.0
        self.register_buffer("expert_masks", masks, persistent=False)

        def projector(out_dim: int) -> nn.ModuleList:
            modules = nn.ModuleList()
            for _ in range(self.num_forms):
                modules.append(
                    nn.Sequential(
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, int(out_dim)),
                    )
                )
            return modules

        self.feature_projectors = projector(int(feature_dim))
        self.roughness_projectors = projector(int(token_dim))
        self.coupling_projectors = projector(int(token_dim))
        self.learned_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, max(hidden_dim // 2, 16)),
                    nn.GELU(),
                    nn.Linear(max(hidden_dim // 2, 16), 1),
                )
                for _ in range(self.num_forms)
            ]
        )
        for group in (self.feature_projectors, self.roughness_projectors, self.coupling_projectors):
            for module in group:
                last = module[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
        for module in self.learned_gates:
            last = module[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.constant_(last.bias, float(learned_gate_bias))

    @staticmethod
    def _hand_form_gates(values: torch.Tensor) -> torch.Tensor:
        macro_rough = values[:, 0:1].clamp(0.0, 1.0)
        micro_rough = values[:, 1:2].clamp(0.0, 1.0)
        film = values[:, 2:3].clamp(0.0, 1.0)
        artifact = values[:, 3:4].clamp(0.0, 1.0)
        saturation = values[:, 4:5].clamp(0.0, 1.0)
        macro_mean = values[:, 5:6].clamp(0.0, 1.0)
        macro_std = values[:, 6:7].clamp(0.0, 1.0)
        meso_std = values[:, 7:8].clamp(0.0, 1.0)
        micro_std = values[:, 8:9].clamp(0.0, 1.0)
        lap_std = values[:, 9:10].clamp(0.0, 1.0)
        grad_std = values[:, 10:11].clamp(0.0, 1.0)
        anisotropy = values[:, 11:12].clamp(0.0, 1.0)
        dark_water = values[:, 12:13].clamp(0.0, 1.0)
        dark_water_top = values[:, 13:14].clamp(0.0, 1.0)
        specular = values[:, 14:15].clamp(0.0, 1.0)
        specular_top = values[:, 15:16].clamp(0.0, 1.0)
        erasure = values[:, 16:17].clamp(0.0, 1.0)
        erasure_top = values[:, 17:18].clamp(0.0, 1.0)
        value_mean = values[:, 18:19].clamp(0.0, 1.0)
        value_std = values[:, 19:20].clamp(0.0, 1.0)

        dry_smooth_slight = (0.34 * meso_std + 0.24 * erasure_top + 0.18 * macro_mean + 0.14 * grad_std + 0.10 * value_std).clamp(0.0, 1.0)
        dry_slight_severe = (0.40 * macro_rough + 0.24 * macro_std + 0.16 * anisotropy + 0.12 * meso_std + 0.08 * grad_std).clamp(0.0, 1.0)
        water_asphalt_smooth_slight = (0.34 * dark_water_top + 0.24 * film + 0.18 * erasure_top + 0.14 * lap_std + 0.10 * micro_std).clamp(0.0, 1.0)
        water_concrete_smooth_slight = (0.30 * dark_water_top + 0.24 * dark_water + 0.18 * film + 0.16 * erasure + 0.12 * value_mean).clamp(0.0, 1.0)
        wet_concrete_slight_severe = (0.28 * macro_rough + 0.22 * anisotropy + 0.18 * erasure_top + 0.14 * saturation + 0.10 * specular_top + 0.08 * macro_std).clamp(0.0, 1.0)
        water_gravel_mud = (0.30 * micro_rough + 0.22 * micro_std + 0.18 * lap_std + 0.16 * grad_std + 0.14 * value_std).clamp(0.0, 1.0)

        film_occlusion = torch.maximum(film, torch.maximum(dark_water_top, specular_top)).clamp(0.0, 1.0)
        roughness_visibility = (1.0 - 0.45 * film_occlusion).clamp(0.12, 1.0)
        dry_guard = (1.0 - 0.55 * film_occlusion).clamp(0.20, 1.0)
        artifact_guard = (1.0 - 0.65 * artifact).clamp(0.15, 1.0)
        gates = torch.cat(
            [
                dry_smooth_slight * dry_guard,
                dry_slight_severe * dry_guard,
                water_asphalt_smooth_slight,
                water_concrete_smooth_slight,
                wet_concrete_slight_severe * roughness_visibility,
                water_gravel_mud,
            ],
            dim=1,
        )
        return (gates * artifact_guard).clamp(0.0, 1.0)

    def prepare(self, value_vector: torch.Tensor) -> dict[str, torch.Tensor]:
        values = value_vector[:, : self.value_dim].clamp(0.0, 1.0)
        if self.training and self.value_aug_std > 0.0:
            values = (values + torch.randn_like(values) * float(self.value_aug_std)).clamp(0.0, 1.0)
        hand_gates = self._hand_form_gates(values)
        contexts = []
        learned = []
        for idx in range(self.num_forms):
            expert_values = values * self.expert_masks[idx].to(device=values.device, dtype=values.dtype).unsqueeze(0)
            gate = hand_gates[:, idx : idx + 1]
            context = torch.cat([expert_values, gate, gate.square(), values[:, 3:4]], dim=1)
            if self.detach_context:
                context = context.detach()
            contexts.append(context)
            learned.append(torch.sigmoid(self.learned_gates[idx](context)))
        learned_gates = torch.cat(learned, dim=1)
        gates = hand_gates * learned_gates
        if self.gate_floor > 0.0:
            gates = self.gate_floor * hand_gates + (1.0 - self.gate_floor) * gates
        return {
            "values": values,
            "contexts": contexts,
            "hand_gates": hand_gates,
            "learned_gates": learned_gates,
            "gates": gates,
            "gate": gates.max(dim=1, keepdim=True).values,
        }

    def _mixture_delta(
        self,
        projectors: nn.ModuleList,
        prepared: dict[str, torch.Tensor],
        *,
        ref: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if scale <= 0.0:
            return torch.zeros_like(ref)
        gates = prepared["gates"].to(device=ref.device, dtype=ref.dtype)
        contexts = prepared["contexts"]
        delta = torch.zeros_like(ref)
        for idx, projector in enumerate(projectors):
            context = contexts[idx].to(device=ref.device, dtype=ref.dtype)
            expert_delta = torch.tanh(projector(context))
            delta = delta + gates[:, idx : idx + 1] * expert_delta
        return float(scale) * delta

    def condition_feature(
        self,
        feature: torch.Tensor,
        prepared: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self._mixture_delta(self.feature_projectors, prepared, ref=feature, scale=self.feature_scale)
        return feature + delta, delta

    def condition_tokens(
        self,
        tokens: dict[str, torch.Tensor],
        prepared: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        roughness = tokens["roughness"]
        coupling = tokens["coupling"]
        rough_delta = self._mixture_delta(
            self.roughness_projectors,
            prepared,
            ref=roughness,
            scale=self.token_scale,
        )
        coupling_delta = self._mixture_delta(
            self.coupling_projectors,
            prepared,
            ref=coupling,
            scale=self.token_scale,
        )
        updated = dict(tokens)
        updated["roughness"] = updated["roughness"] + rough_delta
        updated["coupling"] = updated["coupling"] + coupling_delta
        return updated, {
            "roughness_delta": rough_delta,
            "coupling_delta": coupling_delta,
        }


class PairValueStemConditioner(nn.Module):
    """Early hard-pair value conditioner before the ConvNeXt stem.

    Value-only classifiers failed on the hard RSCD classes, but their audited
    value families still identify where the image stream should pay attention:
    macro/meso roughness for dry concrete, dark-water/erasure for water films,
    and anisotropy/texture tails for wet concrete. This module moves that
    evidence before the backbone. It is form-routed and zero-initialized, so a
    resumed calibrated checkpoint starts from the same pixels and predictions.
    """

    value_dim = 20
    context_dim = 6 + 8 + 1

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        scale: float = 0.018,
        gate_floor: float = 0.0,
        value_aug_std: float = 0.0,
        learned_gate_bias: float = -1.6,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.value_aug_std = max(float(value_aug_std), 0.0)
        self.evidence_maps = C3PhysicsEvidenceMaps()
        hidden_dim = int(hidden_dim)
        self.learned_gate = nn.Sequential(
            nn.LayerNorm(self.value_dim + 6),
            nn.Linear(self.value_dim + 6, max(hidden_dim, 16)),
            nn.GELU(),
            nn.Linear(max(hidden_dim, 16), 6),
        )
        gate_last = self.learned_gate[-1]
        if isinstance(gate_last, nn.Linear):
            nn.init.zeros_(gate_last.weight)
            nn.init.constant_(gate_last.bias, float(learned_gate_bias))

        in_channels = 3 + 3 + 3 + C3PhysicsEvidenceMaps.out_channels + self.context_dim
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden_dim)), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=max(1, hidden_dim), bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )
        last = self.adapter[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _key_values(values: torch.Tensor) -> torch.Tensor:
        """Compact value channels used as broadcast physical context [B, 8]."""

        return torch.cat(
            [
                values[:, 0:1],   # macro roughness
                values[:, 2:3],   # wet/water film
                values[:, 6:7],   # macro std
                values[:, 7:8],   # meso std
                values[:, 11:12], # gradient anisotropy
                values[:, 13:14], # dark-water top tail
                values[:, 17:18], # texture-erasure top tail
                values[:, 19:20], # value std
            ],
            dim=1,
        )

    @staticmethod
    def _spatial_gate(local_maps: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        grad = local_maps[:, 2:3].clamp(0.0, 1.0)
        lap = local_maps[:, 3:4].clamp(0.0, 1.0)
        contrast = local_maps[:, 4:5].clamp(0.0, 1.0)
        specular = local_maps[:, 5:6].clamp(0.0, 1.0)
        dark_water = local_maps[:, 6:7].clamp(0.0, 1.0)
        wet_connected = local_maps[:, 7:8].clamp(0.0, 1.0)
        visible_rough = local_maps[:, 8:9].clamp(0.0, 1.0)
        artifact = local_maps[:, 9:10].clamp(0.0, 1.0)

        rough_map = (0.38 * grad + 0.28 * lap + 0.22 * contrast + 0.12 * visible_rough).clamp(0.0, 1.0)
        film_map = torch.maximum(torch.maximum(specular, dark_water), wet_connected).clamp(0.0, 1.0)
        dry_forms = gates[:, 0:2].amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
        wet_forms = gates[:, 2:5].amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
        texture_forms = gates[:, 5:6].view(-1, 1, 1, 1)
        spatial = dry_forms * rough_map + wet_forms * torch.maximum(film_map, 0.55 * rough_map) + texture_forms * contrast
        return (spatial * (1.0 - 0.65 * artifact).clamp(0.15, 1.0)).clamp(0.0, 1.0)

    def _prepare(self, value_vector: torch.Tensor) -> dict[str, torch.Tensor]:
        values = value_vector[:, : self.value_dim].clamp(0.0, 1.0)
        if self.training and self.value_aug_std > 0.0:
            values = (values + torch.randn_like(values) * float(self.value_aug_std)).clamp(0.0, 1.0)
        hand_gates = CoupledFormExpertConditioner._hand_form_gates(values)
        learned_gates = torch.sigmoid(self.learned_gate(torch.cat([values, hand_gates], dim=1)))
        gates = hand_gates * learned_gates
        if self.gate_floor > 0.0:
            gates = self.gate_floor * hand_gates + (1.0 - self.gate_floor) * gates
        gate = gates.max(dim=1, keepdim=True).values
        context = torch.cat([gates, self._key_values(values), gate], dim=1)
        return {
            "values": values,
            "hand_gates": hand_gates,
            "learned_gates": learned_gates,
            "gates": gates,
            "gate": gate,
            "context": context,
        }

    def forward(
        self,
        image: torch.Tensor,
        value_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.scale <= 0.0:
            zero_gate = image.new_zeros((image.size(0), 1))
            return image, {
                "hand_gates": image.new_zeros((image.size(0), 6)),
                "learned_gates": image.new_zeros((image.size(0), 6)),
                "gates": image.new_zeros((image.size(0), 6)),
                "gate": zero_gate,
                "spatial_gate": image.new_zeros((image.size(0), 1, image.size(2), image.size(3))),
                "delta": torch.zeros_like(image),
            }
        prepared = self._prepare(value_vector.to(device=image.device, dtype=image.dtype))
        local_maps = self.evidence_maps(image).to(device=image.device, dtype=image.dtype)
        low = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        high = image - low
        context = prepared["context"].to(dtype=image.dtype).view(image.size(0), self.context_dim, 1, 1)
        context_maps = context.expand(-1, -1, image.size(2), image.size(3))
        spatial_gate = self._spatial_gate(local_maps, prepared["gates"].to(dtype=image.dtype))
        gate = prepared["gate"].to(dtype=image.dtype).view(image.size(0), 1, 1, 1)
        total_gate = (gate * spatial_gate).clamp(0.0, 1.0)
        fields = torch.cat([image, low, high, local_maps, context_maps], dim=1)
        delta = self.scale * total_gate * torch.tanh(self.adapter(fields))
        return image + delta, {
            "hand_gates": prepared["hand_gates"],
            "learned_gates": prepared["learned_gates"],
            "gates": prepared["gates"],
            "gate": prepared["gate"],
            "spatial_gate": spatial_gate,
            "delta": delta,
        }


class WetWaterConcreteFilmDepthStemConditioner(nn.Module):
    """Early wet/water-concrete film-depth conditioner.

    This module targets the RSCD coupling boundary where wet-concrete-slight
    and water-concrete-slight share material and roughness but differ by water
    film depth. It writes a bounded input-level correction only where local
    reflectance, dark-water, texture-erasure, and concrete-like low-saturation
    evidence suggest a concrete contact patch with ambiguous film depth.
    """

    mechanism_channels = 8

    def __init__(
        self,
        *,
        hidden_dim: int = 36,
        scale: float = 0.030,
        gate_floor: float = 0.04,
        learned_gate_bias: float = -1.2,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.scale = max(float(scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.evidence_maps = C3PhysicsEvidenceMaps()
        global_dim = self.mechanism_channels * 2 + 5
        self.global_gate = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, max(hidden_dim, 16)),
            nn.GELU(),
            nn.Linear(max(hidden_dim, 16), 1),
        )
        gate_last = self.global_gate[-1]
        if isinstance(gate_last, nn.Linear):
            nn.init.zeros_(gate_last.weight)
            nn.init.constant_(gate_last.bias, float(learned_gate_bias))
        in_channels = 3 + 3 + 3 + C3PhysicsEvidenceMaps.out_channels + self.mechanism_channels + 1
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden_dim)), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=max(1, hidden_dim), bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )
        last = self.adapter[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _top_fraction_mean(x: torch.Tensor, fraction: float) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(1, int(flat.size(1) * float(fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def _mechanism_maps(self, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.evidence_maps(reference_image)
        gray = local[:, 0:1].clamp(0.0, 1.0)
        saturation = local[:, 1:2].clamp(0.0, 1.0)
        grad = local[:, 2:3].clamp(0.0, 1.0)
        lap = local[:, 3:4].clamp(0.0, 1.0)
        contrast = local[:, 4:5].clamp(0.0, 1.0)
        specular = local[:, 5:6].clamp(0.0, 1.0)
        dark_water = local[:, 6:7].clamp(0.0, 1.0)
        wet_connected = local[:, 7:8].clamp(0.0, 1.0)
        visible_rough = local[:, 8:9].clamp(0.0, 1.0)
        artifact = local[:, 9:10].clamp(0.0, 1.0)

        concrete_like = (
            torch.sigmoid((gray - 0.34) * 8.0)
            * torch.sigmoid((0.83 - gray) * 8.0)
            * torch.sigmoid((0.34 - saturation) * 10.0)
            * torch.sigmoid((0.86 - gray) * 10.0)
        ).clamp(0.0, 1.0)
        thin_wet_film = (
            concrete_like
            * torch.maximum(specular, wet_connected)
            * torch.sigmoid((0.16 - dark_water) * 18.0)
            * torch.sigmoid((0.11 - lap) * 18.0)
        ).clamp(0.0, 1.0)
        standing_water = (
            concrete_like
            * torch.maximum(dark_water, 0.55 * wet_connected)
            * torch.sigmoid((0.080 - grad) * 24.0)
            * torch.sigmoid((0.065 - contrast) * 26.0)
        ).clamp(0.0, 1.0)
        film_depth_gap = (standing_water * (1.0 - thin_wet_film)).clamp(0.0, 1.0)
        film_boundary = (
            concrete_like
            * torch.maximum(thin_wet_film, standing_water)
            * torch.maximum(grad, lap)
            * torch.sigmoid((0.32 - artifact) * 8.0)
        ).clamp(0.0, 1.0)
        rough_under_film = (
            concrete_like
            * torch.maximum(thin_wet_film, standing_water)
            * (0.45 * visible_rough + 0.35 * contrast + 0.20 * lap)
        ).clamp(0.0, 1.0)
        slight_contact = (
            rough_under_film
            * torch.sigmoid((rough_under_film - 0.055) * 24.0)
            * torch.sigmoid((0.28 - rough_under_film) * 16.0)
        ).clamp(0.0, 1.0)
        transition_band = (
            concrete_like
            * torch.minimum(thin_wet_film + 0.35 * film_boundary, standing_water + 0.35 * film_boundary)
        ).clamp(0.0, 1.0)
        mechanism = torch.cat(
            [
                concrete_like,
                thin_wet_film,
                standing_water,
                film_depth_gap,
                film_boundary,
                rough_under_film,
                slight_contact,
                transition_band,
            ],
            dim=1,
        )
        stats = torch.cat(
            [
                mechanism.mean(dim=(2, 3)),
                torch.cat([self._top_fraction_mean(mechanism[:, i : i + 1], 0.12) for i in range(self.mechanism_channels)], dim=1),
                grad.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                artifact.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        spatial_gate = (
            concrete_like
            * torch.maximum(torch.maximum(thin_wet_film, standing_water), transition_band)
            * (0.25 + 0.75 * torch.maximum(slight_contact, film_boundary)).clamp(0.0, 1.0)
            * (1.0 - 0.35 * artifact).clamp(0.20, 1.0)
        ).clamp(0.0, 1.0)
        return local, mechanism, torch.cat([stats, spatial_gate.mean(dim=(2, 3))], dim=1)

    def forward(
        self,
        image: torch.Tensor,
        *,
        reference_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.scale <= 0.0:
            zero = image.new_zeros((image.size(0), 1))
            return image, {
                "gate": zero,
                "spatial_gate": image.new_zeros((image.size(0), 1, image.size(2), image.size(3))),
                "mechanism": image.new_zeros((image.size(0), self.mechanism_channels, image.size(2), image.size(3))),
                "delta": torch.zeros_like(image),
            }
        ref = image if reference_image is None else reference_image
        local_maps, mechanism, stats = self._mechanism_maps(ref)
        local_maps = local_maps.to(device=image.device, dtype=image.dtype)
        mechanism = mechanism.to(device=image.device, dtype=image.dtype)
        stats = stats.to(device=image.device, dtype=image.dtype)
        low = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        high = image - low
        spatial_gate = (
            mechanism[:, 0:1]
            * torch.maximum(torch.maximum(mechanism[:, 1:2], mechanism[:, 2:3]), mechanism[:, 7:8])
            * (0.25 + 0.75 * torch.maximum(mechanism[:, 5:6], mechanism[:, 6:7])).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        learned_gate = torch.sigmoid(self.global_gate(stats)).to(dtype=image.dtype).view(image.size(0), 1, 1, 1)
        total_gate = (self.gate_floor * spatial_gate + (1.0 - self.gate_floor) * learned_gate * spatial_gate).clamp(0.0, 1.0)
        fields = torch.cat([image, low, high, local_maps, mechanism, total_gate], dim=1)
        delta = self.scale * total_gate * torch.tanh(self.adapter(fields))
        return image + delta, {
            "gate": learned_gate.flatten(1),
            "spatial_gate": spatial_gate,
            "mechanism": mechanism,
            "delta": delta,
        }


class WaterConcreteTopologyTextureStemConditioner(nn.Module):
    """Early topology-texture conditioner for water-concrete roughness coupling.

    Scalar roughness values overlap strongly for water-concrete slight and
    severe. This stem therefore builds soft morphology maps before the backbone:
    rough islands, rough rings, multi-scale persistence, and film-boundary maps
    inside concrete-like wet regions. It also exposes contrast-visibility cues
    that separate water-concrete slight/severe in RSCD audits: local gray
    variance, dark water-film quantiles, and saturation micro-variation. The
    final adapter is zero-initialized so resumed checkpoints start from the same
    predictions.
    """

    mechanism_channels = 12

    def __init__(
        self,
        *,
        hidden_dim: int = 36,
        scale: float = 0.026,
        gate_floor: float = 0.03,
        learned_gate_bias: float = -1.25,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.scale = max(float(scale), 0.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.evidence_maps = C3PhysicsEvidenceMaps()
        global_dim = self.mechanism_channels * 3 + 4
        self.global_gate = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, max(hidden_dim, 16)),
            nn.GELU(),
            nn.Linear(max(hidden_dim, 16), 1),
        )
        gate_last = self.global_gate[-1]
        if isinstance(gate_last, nn.Linear):
            nn.init.zeros_(gate_last.weight)
            nn.init.constant_(gate_last.bias, float(learned_gate_bias))

        in_channels = 3 + 3 + 3 + C3PhysicsEvidenceMaps.out_channels + self.mechanism_channels + 1
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden_dim)), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=max(1, hidden_dim), bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )
        last = self.adapter[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _min_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    @staticmethod
    def _top_fraction_mean(x: torch.Tensor, fraction: float) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(1, int(flat.size(1) * float(fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def _mechanism_maps(self, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.evidence_maps(reference_image)
        gray = local[:, 0:1].clamp(0.0, 1.0)
        saturation = local[:, 1:2].clamp(0.0, 1.0)
        grad = local[:, 2:3].clamp(0.0, 1.0)
        lap = local[:, 3:4].clamp(0.0, 1.0)
        contrast = local[:, 4:5].clamp(0.0, 1.0)
        specular = local[:, 5:6].clamp(0.0, 1.0)
        dark_water = local[:, 6:7].clamp(0.0, 1.0)
        wet_connected = local[:, 7:8].clamp(0.0, 1.0)
        visible_rough = local[:, 8:9].clamp(0.0, 1.0)
        artifact = local[:, 9:10].clamp(0.0, 1.0)
        gray_mean11 = F.avg_pool2d(gray, kernel_size=11, stride=1, padding=5)
        gray_std = torch.sqrt(
            F.avg_pool2d((gray - gray_mean11).square(), kernel_size=11, stride=1, padding=5) + 1e-6
        ).clamp(0.0, 1.0)
        sat_mean11 = F.avg_pool2d(saturation, kernel_size=11, stride=1, padding=5)
        sat_std = torch.sqrt(
            F.avg_pool2d((saturation - sat_mean11).square(), kernel_size=11, stride=1, padding=5) + 1e-6
        ).clamp(0.0, 1.0)

        concrete_like = (
            torch.sigmoid((gray - 0.34) * 8.0)
            * torch.sigmoid((0.84 - gray) * 8.0)
            * torch.sigmoid((0.35 - saturation) * 10.0)
            * torch.sigmoid((0.88 - gray) * 10.0)
        ).clamp(0.0, 1.0)
        film = (
            concrete_like
            * torch.maximum(torch.maximum(specular, dark_water), wet_connected)
            * (1.0 - 0.45 * artifact).clamp(0.15, 1.0)
        ).clamp(0.0, 1.0)
        rough = (
            concrete_like
            * (0.40 * visible_rough + 0.24 * grad + 0.22 * lap + 0.14 * contrast)
        ).clamp(0.0, 1.0)
        hidden_rough = (film * torch.maximum(rough, 0.58 * contrast + 0.42 * lap)).clamp(0.0, 1.0)

        rough_dilate3 = F.max_pool2d(hidden_rough, kernel_size=3, stride=1, padding=1)
        rough_erode3 = self._min_pool2d(hidden_rough, kernel_size=3)
        rough_ring = (rough_dilate3 - rough_erode3).clamp(0.0, 1.0)
        rough_dilate7 = F.max_pool2d(hidden_rough, kernel_size=7, stride=1, padding=3)
        rough_avg7 = F.avg_pool2d(hidden_rough, kernel_size=7, stride=1, padding=3)
        persistence = (rough_dilate7 - rough_avg7).clamp(0.0, 1.0)
        local_base = F.avg_pool2d(hidden_rough, kernel_size=9, stride=1, padding=4)
        rough_island = (hidden_rough * torch.sigmoid((hidden_rough - local_base) * 24.0)).clamp(0.0, 1.0)
        film_boundary = (
            film
            * (F.max_pool2d(film, kernel_size=7, stride=1, padding=3) - self._min_pool2d(film, kernel_size=7))
        ).clamp(0.0, 1.0)
        topology = (
            0.34 * rough_island
            + 0.26 * persistence
            + 0.22 * rough_ring
            + 0.18 * film_boundary
        ).clamp(0.0, 1.0)
        rough_band = (torch.sigmoid((rough - 0.006) * 80.0) * torch.sigmoid((0.090 - rough) * 36.0)).clamp(
            0.0,
            1.0,
        )
        film_occlusion = (
            film
            * rough_band
            * (0.40 + 0.60 * torch.maximum(film_boundary, topology)).clamp(0.0, 1.0)
            * torch.sigmoid((0.58 - concrete_like) * 5.5)
        ).clamp(0.0, 1.0)
        visible_slight_texture = (
            film
            * concrete_like
            * torch.sigmoid((rough - 0.010) * 70.0)
            * torch.sigmoid((0.120 - film) * 18.0)
            * (1.0 - 0.30 * film_boundary).clamp(0.20, 1.0)
        ).clamp(0.0, 1.0)
        contrast_visibility = (
            concrete_like
            * film
            * torch.sigmoid((gray_std - 0.020) * 70.0)
            * (1.0 - 0.30 * artifact).clamp(0.20, 1.0)
        ).clamp(0.0, 1.0)
        dark_film_quantile = (
            concrete_like
            * film
            * torch.sigmoid((0.54 - gray) * 10.0)
            * torch.sigmoid((0.34 - saturation) * 10.0)
            * (1.0 - 0.25 * artifact).clamp(0.20, 1.0)
        ).clamp(0.0, 1.0)
        chroma_micro_variation = (
            concrete_like
            * film
            * torch.sigmoid((sat_std - 0.010) * 80.0)
            * (1.0 - 0.25 * artifact).clamp(0.20, 1.0)
        ).clamp(0.0, 1.0)
        visible_low_variance_slight = (
            concrete_like
            * film
            * torch.sigmoid((gray - 0.52) * 8.0)
            * torch.sigmoid((0.075 - gray_std) * 60.0)
            * torch.sigmoid((0.055 - sat_std) * 70.0)
            * (1.0 - 0.35 * film_boundary).clamp(0.20, 1.0)
        ).clamp(0.0, 1.0)
        contrast_visibility_severe = (
            0.42 * contrast_visibility + 0.34 * dark_film_quantile + 0.24 * chroma_micro_variation
        ).clamp(0.0, 1.0)
        severe_form = (0.58 * film_occlusion + 0.42 * topology * torch.sigmoid((film - 0.045) * 16.0)).clamp(
            0.0,
            1.0,
        )
        severe_form = (0.74 * severe_form + 0.26 * contrast_visibility_severe).clamp(0.0, 1.0)
        slight_form = (
            0.52 * visible_slight_texture
            + 0.26 * visible_low_variance_slight
            + 0.22 * film * rough * (1.0 - 0.45 * topology)
        ).clamp(
            0.0,
            1.0,
        )
        mechanism = torch.cat(
            [
                concrete_like,
                film,
                rough,
                hidden_rough,
                rough_ring,
                persistence,
                rough_island,
                film_boundary,
                severe_form - 0.75 * slight_form,
                contrast_visibility,
                dark_film_quantile,
                chroma_micro_variation,
            ],
            dim=1,
        ).clamp(-1.0, 1.0)
        nonnegative = mechanism.clamp_min(0.0)
        stats = torch.cat(
            [
                nonnegative.mean(dim=(2, 3)),
                torch.cat([self._top_fraction_mean(nonnegative[:, i : i + 1], 0.10) for i in range(self.mechanism_channels)], dim=1),
                nonnegative.std(dim=(2, 3), unbiased=False),
                grad.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                contrast.mean(dim=(2, 3)),
                artifact.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        return local, mechanism, stats

    def forward(
        self,
        image: torch.Tensor,
        *,
        reference_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.scale <= 0.0:
            zero = image.new_zeros((image.size(0), 1))
            return image, {
                "gate": zero,
                "spatial_gate": image.new_zeros((image.size(0), 1, image.size(2), image.size(3))),
                "mechanism": image.new_zeros((image.size(0), self.mechanism_channels, image.size(2), image.size(3))),
                "delta": torch.zeros_like(image),
            }
        ref = image if reference_image is None else reference_image
        local_maps, mechanism, stats = self._mechanism_maps(ref)
        local_maps = local_maps.to(device=image.device, dtype=image.dtype)
        mechanism = mechanism.to(device=image.device, dtype=image.dtype)
        stats = stats.to(device=image.device, dtype=image.dtype)
        low = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        high = image - low
        topology = mechanism[:, 4:8].clamp_min(0.0).amax(dim=1, keepdim=True)
        concrete = mechanism[:, 0:1].clamp(0.0, 1.0)
        film = mechanism[:, 1:2].clamp(0.0, 1.0)
        rough = mechanism[:, 2:3].clamp(0.0, 1.0)
        film_boundary = mechanism[:, 7:8].clamp(0.0, 1.0)
        signed_severe = mechanism[:, 8:9].clamp(-1.0, 1.0)
        contrast_visibility = mechanism[:, 9:10].clamp(0.0, 1.0)
        dark_film_quantile = mechanism[:, 10:11].clamp(0.0, 1.0)
        chroma_micro_variation = mechanism[:, 11:12].clamp(0.0, 1.0)
        artifact = local_maps[:, 9:10].clamp(0.0, 1.0)
        signed_strength = signed_severe.abs().clamp(0.0, 1.0)
        visibility_cue = (0.45 * contrast_visibility + 0.35 * dark_film_quantile + 0.20 * chroma_micro_variation).clamp(
            0.0,
            1.0,
        )
        concrete_focus = torch.sigmoid((concrete - 0.360) * 12.0)
        film_band = (torch.sigmoid((film - 0.045) * 18.0) * torch.sigmoid((0.115 - film) * 20.0)).clamp(
            0.0,
            1.0,
        )
        rough_band = (torch.sigmoid((rough - 0.006) * 80.0) * torch.sigmoid((0.085 - rough) * 38.0)).clamp(
            0.0,
            1.0,
        )
        boundary_guard = torch.sigmoid((0.0075 - film_boundary) * 450.0)
        base_spatial_gate = (
            concrete
            * concrete_focus
            * film_band
            * rough_band
            * boundary_guard
            * (0.40 + 0.60 * signed_strength).clamp(0.0, 1.0)
            * (0.65 + 0.35 * topology).clamp(0.0, 1.0)
            * (0.62 + 0.38 * visibility_cue).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        visibility_lift = (
            0.10
            * concrete
            * concrete_focus
            * film_band
            * rough_band.clamp_min(0.35)
            * visibility_cue
            * (0.45 + 0.55 * signed_strength).clamp(0.0, 1.0)
            * (1.0 - 0.20 * artifact).clamp(0.20, 1.0)
        ).clamp(0.0, 1.0)
        spatial_gate = (base_spatial_gate + visibility_lift).clamp(0.0, 1.0)
        learned_gate = torch.sigmoid(self.global_gate(stats)).to(dtype=image.dtype).view(image.size(0), 1, 1, 1)
        total_gate = (spatial_gate * (self.gate_floor + (1.0 - self.gate_floor) * learned_gate)).clamp(0.0, 1.0)
        fields = torch.cat([image, low, high, local_maps, mechanism, total_gate], dim=1)
        delta = self.scale * total_gate * torch.tanh(self.adapter(fields))
        return image + delta, {
            "gate": learned_gate.flatten(1),
            "spatial_gate": spatial_gate,
            "mechanism": mechanism,
            "delta": delta,
        }


class CoupledTensorHead(nn.Module):
    """Factorized main, pairwise, and image-conditioned triple coupling head."""

    def __init__(
        self,
        *,
        spec: RSCDFactorSpec,
        token_dim: int,
        pair_rank: int = 8,
        triple_rank: int = 8,
        use_pairwise: bool = True,
        use_triple: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.use_pairwise = bool(use_pairwise)
        self.use_triple = bool(use_triple)
        self.num_f = len(FACTOR_LABELS["friction"])
        self.num_m = len(FACTOR_LABELS["material"])
        self.num_r = len(FACTOR_LABELS["roughness"])
        self.register_buffer("class_to_factor", spec.class_to_factor.clone())
        self.register_buffer("valid_tensor_mask", spec.valid_tensor_mask.clone())

        self.main_f = nn.Parameter(torch.randn(self.num_f, int(token_dim)) * 0.02)
        self.main_m = nn.Parameter(torch.randn(self.num_m, int(token_dim)) * 0.02)
        self.main_r = nn.Parameter(torch.randn(self.num_r, int(token_dim)) * 0.02)
        self.factor_bias = nn.ParameterDict(
            {
                "friction": nn.Parameter(torch.zeros(self.num_f)),
                "material": nn.Parameter(torch.zeros(self.num_m)),
                "roughness": nn.Parameter(torch.zeros(self.num_r)),
            }
        )
        q2 = max(int(pair_rank), 1)
        q3 = max(int(triple_rank), 1)
        self.fm_a = nn.Parameter(torch.randn(self.num_f, q2) * 0.02)
        self.fm_b = nn.Parameter(torch.randn(self.num_m, q2) * 0.02)
        self.fr_a = nn.Parameter(torch.randn(self.num_f, q2) * 0.02)
        self.fr_c = nn.Parameter(torch.randn(self.num_r, q2) * 0.02)
        self.mr_b = nn.Parameter(torch.randn(self.num_m, q2) * 0.02)
        self.mr_c = nn.Parameter(torch.randn(self.num_r, q2) * 0.02)
        self.lambda_fm = nn.Sequential(nn.LayerNorm(int(token_dim)), nn.Linear(int(token_dim), q2))
        self.lambda_fr = nn.Sequential(nn.LayerNorm(int(token_dim)), nn.Linear(int(token_dim), q2))
        self.lambda_mr = nn.Sequential(nn.LayerNorm(int(token_dim)), nn.Linear(int(token_dim), q2))
        self.tri_a = nn.Parameter(torch.randn(self.num_f, q3) * 0.02)
        self.tri_b = nn.Parameter(torch.randn(self.num_m, q3) * 0.02)
        self.tri_c = nn.Parameter(torch.randn(self.num_r, q3) * 0.02)
        self.lambda_tri = nn.Sequential(nn.LayerNorm(int(token_dim)), nn.Linear(int(token_dim), q3))

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_r: torch.Tensor,
        z_c: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sf = z_f @ self.main_f.t() + self.factor_bias["friction"]
        sm = z_m @ self.main_m.t() + self.factor_bias["material"]
        sr = z_r @ self.main_r.t() + self.factor_bias["roughness"]
        main_grid = sf[:, :, None, None] + sm[:, None, :, None] + sr[:, None, None, :]

        pair_grid = torch.zeros_like(main_grid)
        if self.use_pairwise:
            l_fm = self.lambda_fm(z_c)
            l_fr = self.lambda_fr(z_c)
            l_mr = self.lambda_mr(z_c)
            pair_grid = pair_grid + torch.einsum("bq,fq,mq->bfm", l_fm, self.fm_a, self.fm_b)[:, :, :, None]
            pair_grid = pair_grid + torch.einsum("bq,fq,rq->bfr", l_fr, self.fr_a, self.fr_c)[:, :, None, :]
            pair_grid = pair_grid + torch.einsum("bq,mq,rq->bmr", l_mr, self.mr_b, self.mr_c)[:, None, :, :]

        triple_grid = torch.zeros_like(main_grid)
        lambda_tri = self.lambda_tri(z_c)
        if self.use_triple:
            triple_grid = torch.einsum("bq,fq,mq,rq->bfmr", lambda_tri, self.tri_a, self.tri_b, self.tri_c)

        score_grid = main_grid + pair_grid + triple_grid
        score_grid = score_grid.masked_fill(~self.valid_tensor_mask.to(device=score_grid.device).unsqueeze(0), -1.0e4)
        class_factors = self.class_to_factor.to(device=score_grid.device)
        logits = score_grid[
            :,
            class_factors[:, 0].clamp_min(0),
            class_factors[:, 1].clamp_min(0),
            class_factors[:, 2].clamp_min(0),
        ]
        class_main = main_grid[
            :,
            class_factors[:, 0].clamp_min(0),
            class_factors[:, 1].clamp_min(0),
            class_factors[:, 2].clamp_min(0),
        ]
        class_pair = pair_grid[
            :,
            class_factors[:, 0].clamp_min(0),
            class_factors[:, 1].clamp_min(0),
            class_factors[:, 2].clamp_min(0),
        ]
        class_triple = triple_grid[
            :,
            class_factors[:, 0].clamp_min(0),
            class_factors[:, 1].clamp_min(0),
            class_factors[:, 2].clamp_min(0),
        ]
        return {
            "logits": logits,
            "factor_logits": {"friction": sf, "material": sm, "roughness": sr},
            "score_grid": score_grid,
            "main_logits": class_main,
            "pair_logits": class_pair,
            "triple_logits": class_triple,
            # Standard C3-FaRNet notation: image-conditioned low-rank
            # triple-coupling weights with shape [batch, triple_rank].
            "lambda_q": lambda_tri,
            "lambda_triple": lambda_tri,
        }


class FactorGraphProtectedLogitAdapter(nn.Module):
    """Bounded RSCD factor-graph logit adapter with explicit class protection.

    The adapter is designed for late-stage calibration after a strong anchor has
    already learned most classes. It writes a small residual only through
    friction/material/roughness factor embeddings and only on configured active
    RSCD classes; protected classes receive an exact zero residual.
    """

    def __init__(
        self,
        *,
        spec: RSCDFactorSpec,
        idx_to_class: dict[int, str],
        input_dim: int,
        rank: int = 6,
        hidden_dim: int = 96,
        scale: float = 0.08,
        gate_margin: float = 0.18,
        gate_temperature: float = 10.0,
        active_classes: list[str] | tuple[str, ...] | None = None,
        protected_classes: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.scale = max(float(scale), 0.0)
        self.gate_margin = float(gate_margin)
        self.gate_temperature = float(gate_temperature)
        q = max(int(rank), 1)
        num_classes = int(spec.class_to_factor.shape[0])
        self.register_buffer("class_to_factor", spec.class_to_factor.clone())
        active_names = {canonical_class_label(name) for name in (active_classes or [])}
        protected_names = {canonical_class_label(name) for name in (protected_classes or [])}
        active_mask = torch.zeros(num_classes, dtype=torch.float32)
        if active_names:
            for idx, name in idx_to_class.items():
                if canonical_class_label(name) in active_names:
                    active_mask[int(idx)] = 1.0
        else:
            active_mask.fill_(1.0)
        protected_mask = torch.zeros(num_classes, dtype=torch.float32)
        for idx, name in idx_to_class.items():
            if canonical_class_label(name) in protected_names:
                protected_mask[int(idx)] = 1.0
        active_mask = active_mask * (1.0 - protected_mask)
        self.register_buffer("active_mask", active_mask)
        self.register_buffer("protected_mask", protected_mask)

        self.friction_basis = nn.Parameter(torch.randn(len(FACTOR_LABELS["friction"]), q) * 0.02)
        self.material_basis = nn.Parameter(torch.randn(len(FACTOR_LABELS["material"]), q) * 0.02)
        self.roughness_basis = nn.Parameter(torch.randn(len(FACTOR_LABELS["roughness"]), q) * 0.02)
        self.fm_basis = nn.Parameter(torch.randn(len(FACTOR_LABELS["friction"]), len(FACTOR_LABELS["material"]), q) * 0.01)
        self.fr_basis = nn.Parameter(torch.randn(len(FACTOR_LABELS["friction"]), len(FACTOR_LABELS["roughness"]), q) * 0.01)
        self.mr_basis = nn.Parameter(torch.randn(len(FACTOR_LABELS["material"]), len(FACTOR_LABELS["roughness"]), q) * 0.01)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.condition = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), q),
        )
        last = self.condition[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _class_basis(self) -> torch.Tensor:
        factors = self.class_to_factor.clamp_min(0)
        f = factors[:, 0]
        m = factors[:, 1]
        r = factors[:, 2]
        return (
            self.friction_basis.index_select(0, f)
            + self.material_basis.index_select(0, m)
            + self.roughness_basis.index_select(0, r)
            + self.fm_basis[f, m]
            + self.fr_basis[f, r]
            + self.mr_basis[m, r]
        )

    @staticmethod
    def _prob_margin(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)
        top2 = probs.topk(k=min(2, probs.size(1)), dim=1).values
        if top2.size(1) == 1:
            return torch.ones_like(top2[:, 0:1])
        return (top2[:, 0:1] - top2[:, 1:2]).clamp(0.0, 1.0)

    def forward(self, linear_logits: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return adapted logits and the applied residual, both [B, num_classes]."""

        active = self.active_mask.to(device=linear_logits.device, dtype=linear_logits.dtype)
        if self.scale <= 0.0 or not bool(active.bool().any()):
            zero = torch.zeros_like(linear_logits)
            return linear_logits, zero
        gamma = torch.tanh(self.condition(condition.to(dtype=linear_logits.dtype)))
        raw = gamma @ self._class_basis().to(device=linear_logits.device, dtype=linear_logits.dtype).t()
        raw = raw + self.class_bias.to(device=linear_logits.device, dtype=linear_logits.dtype).unsqueeze(0)
        active_sum = active.sum().clamp_min(1.0)
        active_mean = (raw * active.unsqueeze(0)).sum(dim=1, keepdim=True) / active_sum
        centered = raw - active_mean
        ambiguity = torch.sigmoid(
            (self.gate_margin - self._prob_margin(linear_logits).to(dtype=linear_logits.dtype)) * self.gate_temperature
        )
        residual = self.scale * ambiguity * torch.tanh(centered) * active.unsqueeze(0)
        return linear_logits + residual, residual


class ClosedSetFactorRedistributor(nn.Module):
    """Physics-gated zero-sum logit redistribution inside RSCD closed sets.

    The module targets hard RSCD factor couplings such as wet/water + concrete
    + roughness. It does not act as a global residual head: probability evidence
    is only redistributed among a configured closed set, and the residual is
    centered so the set-level logit mass is preserved.
    """

    _ALIASES = {
        "wet_water_concrete_roughness": (
            "water_concrete_smooth",
            "wet_concrete_smooth",
            "water_concrete_slight",
            "wet_concrete_slight",
            "water_concrete_severe",
            "wet_concrete_severe",
        ),
        "wet_water_concrete_hard": (
            "water_concrete_slight",
            "wet_concrete_slight",
            "water_concrete_severe",
            "wet_concrete_severe",
        ),
        "dry_concrete_roughness": (
            "dry_concrete_smooth",
            "dry_concrete_slight",
            "dry_concrete_severe",
        ),
        "water_asphalt_roughness": (
            "water_asphalt_smooth",
            "water_asphalt_slight",
            "water_asphalt_severe",
        ),
        "wet_asphalt_roughness": (
            "wet_asphalt_smooth",
            "wet_asphalt_slight",
            "wet_asphalt_severe",
        ),
    }

    def __init__(
        self,
        class_to_idx: dict[str, int],
        set_specs: list[str] | tuple[str, ...] | None,
        *,
        input_dim: int,
        hidden_dim: int = 96,
        scale: float = 0.06,
        gate_floor: float = 0.0,
        mass_threshold: float = 0.08,
        margin_threshold: float = 0.25,
        temperature: float = 8.0,
        dropout: float = 0.0,
        gate_bias_init: float = -2.5,
        use_graph_locality_guard: bool = False,
        graph_max_distance: float = 2.0,
        graph_guard_floor: float = 0.0,
        graph_guard_temperature: float = 12.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_floor = float(gate_floor)
        self.mass_threshold = float(mass_threshold)
        self.margin_threshold = float(margin_threshold)
        self.temperature = float(temperature)
        self.use_graph_locality_guard = bool(use_graph_locality_guard)
        self.graph_max_distance = float(graph_max_distance)
        self.graph_guard_floor = float(graph_guard_floor)
        self.graph_guard_temperature = float(graph_guard_temperature)
        parsed_sets = self._parse_sets(class_to_idx, set_specs)
        self.set_names = tuple(item[0] for item in parsed_sets)
        self.set_kinds = tuple(item[2] for item in parsed_sets)
        max_len = max((len(item[1]) for item in parsed_sets), default=1)
        index_tensor = torch.full((len(parsed_sets), max_len), -1, dtype=torch.long)
        mask_tensor = torch.zeros((len(parsed_sets), max_len), dtype=torch.bool)
        distance_tensor = torch.zeros((len(parsed_sets), max_len, max_len), dtype=torch.float32)
        spec = build_rscd_factor_spec(class_to_idx)
        for row, (_, indices, _) in enumerate(parsed_sets):
            index_tensor[row, : len(indices)] = torch.as_tensor(indices, dtype=torch.long)
            mask_tensor[row, : len(indices)] = True
            for left_pos, left_idx in enumerate(indices):
                for right_pos, right_idx in enumerate(indices):
                    distance_tensor[row, left_pos, right_pos] = self._factor_graph_distance(
                        spec.class_to_factor[int(left_idx)].tolist(),
                        spec.class_to_factor[int(right_idx)].tolist(),
                    )
        self.register_buffer("set_indices", index_tensor, persistent=False)
        self.register_buffer("set_mask", mask_tensor, persistent=False)
        self.register_buffer("set_graph_distance", distance_tensor, persistent=False)
        net_in = int(input_dim) + 6
        self.residual_nets = nn.ModuleList()
        self.gate_nets = nn.ModuleList()
        for _ in parsed_sets:
            residual_net = nn.Sequential(
                nn.LayerNorm(net_in),
                nn.Linear(net_in, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), max_len),
            )
            residual_last = residual_net[-1]
            if isinstance(residual_last, nn.Linear):
                nn.init.zeros_(residual_last.weight)
                nn.init.zeros_(residual_last.bias)
            gate_net = nn.Sequential(
                nn.LayerNorm(net_in),
                nn.Linear(net_in, max(int(hidden_dim) // 2, 16)),
                nn.GELU(),
                nn.Linear(max(int(hidden_dim) // 2, 16), 1),
            )
            gate_last = gate_net[-1]
            if isinstance(gate_last, nn.Linear):
                nn.init.zeros_(gate_last.weight)
                nn.init.constant_(gate_last.bias, float(gate_bias_init))
            self.residual_nets.append(residual_net)
            self.gate_nets.append(gate_net)

    @classmethod
    def _parse_sets(
        cls,
        class_to_idx: dict[str, int],
        set_specs: list[str] | tuple[str, ...] | None,
    ) -> list[tuple[str, list[int], str]]:
        specs = list(set_specs or ["wet_water_concrete_roughness"])
        parsed: list[tuple[str, list[int], str]] = []
        for raw in specs:
            if isinstance(raw, str):
                key = raw.strip()
                if key in cls._ALIASES:
                    names = list(cls._ALIASES[key])
                    set_name = key
                else:
                    names = [
                        part.strip()
                        for part in key.replace("<->", "|").replace(",", "|").split("|")
                        if part.strip()
                    ]
                    set_name = "_".join(names[:3]) if names else "closed_set"
            else:
                names = [str(part).strip() for part in raw if str(part).strip()]
                set_name = "_".join(names[:3]) if names else "closed_set"
            canonical_names = [canonical_class_label(name) for name in names]
            indices = [int(class_to_idx[name]) for name in canonical_names if name in class_to_idx]
            if len(indices) < 2:
                continue
            parsed.append((set_name, indices, cls._infer_kind(canonical_names)))
        return parsed

    @staticmethod
    def _infer_kind(names: list[str]) -> str:
        joined = "|".join(names)
        if "concrete" in joined and ("water_" in joined or "wet_" in joined):
            return "wet_water_concrete"
        if "dry_concrete" in joined:
            return "dry_concrete"
        if "asphalt" in joined and ("water_" in joined or "wet_" in joined):
            return "wet_water_asphalt"
        return "generic"

    @staticmethod
    def _factor_graph_distance(left_factor: list[int], right_factor: list[int]) -> float:
        if min(left_factor + right_factor) < 0:
            return 0.0 if left_factor == right_factor else 3.0
        left_f, left_m, left_r = left_factor
        right_f, right_m, right_r = right_factor
        rough_rank = {"none": 0, "smooth": 0, "slight": 1, "severe": 2}
        rough_labels = FACTOR_LABELS["roughness"]
        distance = 0.0
        if int(left_f) != int(right_f):
            distance += 1.0
        if int(left_m) != int(right_m):
            distance += 1.0
        left_rank = rough_rank.get(rough_labels[int(left_r)], 0)
        right_rank = rough_rank.get(rough_labels[int(right_r)], 0)
        distance += float(abs(left_rank - right_rank))
        return distance

    def _physics_gate(self, kind: str, evidence_stats: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
        if evidence_stats is None:
            return like.new_ones((like.size(0), 1))
        stats = evidence_stats.to(device=like.device, dtype=like.dtype)
        specular = stats[:, 8:9].clamp(0.0, 1.0)
        dark_water = stats[:, 9:10].clamp(0.0, 1.0)
        wet = stats[:, 10:11].clamp(0.0, 1.0)
        rough = stats[:, 11:12].clamp(0.0, 1.0)
        erasure = stats[:, 12:13].clamp(0.0, 1.0)
        winter = torch.maximum(stats[:, 13:14], stats[:, 14:15]).clamp(0.0, 1.0)
        artifact_guard = (1.0 - 0.55 * winter).clamp(0.20, 1.0)
        film = (0.50 * wet + 0.30 * dark_water + 0.20 * specular).clamp(0.0, 1.0)
        if kind == "wet_water_concrete":
            visible_rough = (0.65 * rough + 0.35 * (1.0 - erasure)).clamp(0.0, 1.0)
            gate = (0.35 + 0.65 * (0.62 * film + 0.38 * visible_rough)).clamp(0.0, 1.0)
        elif kind == "dry_concrete":
            dry_guard = (1.0 - 0.65 * film).clamp(0.15, 1.0)
            gate = (0.38 + 0.62 * rough).clamp(0.0, 1.0) * dry_guard
        elif kind == "wet_water_asphalt":
            gate = (0.35 + 0.65 * film).clamp(0.0, 1.0)
        else:
            gate = like.new_ones((like.size(0), 1))
        return (gate * artifact_guard).clamp(0.0, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        condition: torch.Tensor,
        evidence_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return logits after closed-set redistribution plus diagnostic tensors."""

        if self.scale <= 0.0 or len(self.residual_nets) == 0:
            zero = torch.zeros_like(logits)
            return logits, {"residual": zero, "gates": {}, "raw": {}}
        probs = F.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
        residual = torch.zeros_like(logits)
        gates: dict[str, torch.Tensor] = {}
        raws: dict[str, torch.Tensor] = {}
        masses: dict[str, torch.Tensor] = {}
        margins: dict[str, torch.Tensor] = {}
        graph_guards: dict[str, torch.Tensor] = {}
        for set_id, (residual_net, gate_net) in enumerate(zip(self.residual_nets, self.gate_nets, strict=True)):
            mask = self.set_mask[set_id].to(device=logits.device)
            indices = self.set_indices[set_id].to(device=logits.device)[mask]
            if indices.numel() < 2:
                continue
            set_logits = logits.index_select(1, indices)
            set_probs = probs.index_select(1, indices).clamp(0.0, 1.0)
            set_mass = set_probs.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            local_probs = set_probs / set_mass.clamp_min(1e-6)
            top2 = local_probs.topk(k=min(2, local_probs.size(1)), dim=1).values
            if top2.size(1) == 1:
                local_margin = torch.ones_like(set_mass)
            else:
                local_margin = (top2[:, 0:1] - top2[:, 1:2]).clamp(0.0, 1.0)
            denom = torch.log(logits.new_tensor(float(max(indices.numel(), 2)))).clamp_min(1e-6)
            entropy = -(local_probs.clamp_min(1e-6) * local_probs.clamp_min(1e-6).log()).sum(dim=1, keepdim=True) / denom
            mean_logit = set_logits.mean(dim=1, keepdim=True)
            std_logit = set_logits.std(dim=1, keepdim=True, unbiased=False)
            top_prob = local_probs.max(dim=1, keepdim=True).values
            summary = torch.cat(
                [
                    condition.to(dtype=logits.dtype),
                    set_mass,
                    local_margin,
                    entropy.clamp(0.0, 1.0),
                    torch.tanh(0.25 * mean_logit),
                    torch.tanh(0.25 * std_logit),
                    top_prob,
                ],
                dim=1,
            )
            raw_all = residual_net(summary)
            raw = raw_all[:, : indices.numel()]
            centered = raw - raw.mean(dim=1, keepdim=True)
            learned_gate = torch.sigmoid(gate_net(summary))
            if self.gate_floor > 0.0:
                learned_gate = self.gate_floor + (1.0 - self.gate_floor) * learned_gate
            mass_gate = torch.sigmoid((set_mass - self.mass_threshold) * self.temperature)
            ambiguity_gate = torch.sigmoid((self.margin_threshold - local_margin) * self.temperature)
            physics_gate = self._physics_gate(self.set_kinds[set_id], evidence_stats, learned_gate)
            gate = (learned_gate * mass_gate * ambiguity_gate * physics_gate).clamp(0.0, 1.0)
            delta = (float(self.scale) * gate * torch.tanh(centered)).to(dtype=logits.dtype)
            if self.use_graph_locality_guard:
                anchor_pos = local_probs.argmax(dim=1)
                distance = self.set_graph_distance[set_id].to(device=logits.device, dtype=logits.dtype)
                distance = distance[: indices.numel(), : indices.numel()]
                guard = torch.sigmoid(
                    (float(self.graph_max_distance) + 0.5 - distance[anchor_pos])
                    * float(self.graph_guard_temperature)
                ).to(dtype=logits.dtype)
                if self.graph_guard_floor > 0.0:
                    guard = float(self.graph_guard_floor) + (1.0 - float(self.graph_guard_floor)) * guard
                delta = torch.where(delta > 0.0, delta * guard, delta)
                graph_guards[self.set_names[set_id]] = guard
            residual[:, indices] = residual[:, indices] + delta
            set_name = self.set_names[set_id]
            gates[set_name] = gate.squeeze(1)
            raws[set_name] = raw
            masses[set_name] = set_mass.squeeze(1)
            margins[set_name] = local_margin.squeeze(1)
        return logits + residual, {
            "residual": residual,
            "gates": gates,
            "raw": raws,
            "masses": masses,
            "margins": margins,
            "graph_guards": graph_guards,
        }


class FactorGraphEdgeFlowCorrector(nn.Module):
    """Adjacent factor-graph edge flow for RSCD hard boundaries.

    Unlike closed-set redistribution, this module can only move score along
    configured hard-pair edges where two RSCD factors are shared and one factor
    changes. It is intended for wet/water-concrete roughness and friction edges,
    where a free six-class residual proved too fragile.
    """

    _ALIASES = {
        "wet_water_concrete_adjacent": (
            "water_concrete_smooth|wet_concrete_smooth",
            "water_concrete_slight|wet_concrete_slight",
            "water_concrete_severe|wet_concrete_severe",
            "water_concrete_smooth|water_concrete_slight",
            "water_concrete_slight|water_concrete_severe",
            "wet_concrete_smooth|wet_concrete_slight",
            "wet_concrete_slight|wet_concrete_severe",
        ),
        "wet_water_concrete_hard_adjacent": (
            "water_concrete_slight|wet_concrete_slight",
            "water_concrete_severe|wet_concrete_severe",
            "water_concrete_slight|water_concrete_severe",
            "wet_concrete_slight|wet_concrete_severe",
        ),
        "water_asphalt_adjacent": (
            "water_asphalt_smooth|water_asphalt_slight",
            "water_asphalt_slight|water_asphalt_severe",
        ),
        "dry_concrete_adjacent": (
            "dry_concrete_smooth|dry_concrete_slight",
            "dry_concrete_slight|dry_concrete_severe",
        ),
    }

    def __init__(
        self,
        class_to_idx: dict[str, int],
        pair_specs: list[str] | tuple[str, ...] | None,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        scale: float = 0.10,
        gate_margin: float = 0.90,
        gate_temperature: float = 4.0,
        gate_floor: float = 0.0,
        confidence_protect: float = 0.74,
        confidence_temperature: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_margin = float(gate_margin)
        self.gate_temperature = float(gate_temperature)
        self.gate_floor = float(gate_floor)
        self.confidence_protect = float(confidence_protect)
        self.confidence_temperature = float(confidence_temperature)
        spec = build_rscd_factor_spec(class_to_idx)
        pairs = self._parse_pairs(spec, pair_specs)
        self.edge_keys = tuple(item[0] for item in pairs)
        self.edge_axes = tuple(item[3] for item in pairs)
        self.edge_kinds = tuple(item[4] for item in pairs)
        edge_tensor = torch.as_tensor([[left, right] for _, left, right, _, _ in pairs], dtype=torch.long)
        if edge_tensor.numel() == 0:
            edge_tensor = torch.empty((0, 2), dtype=torch.long)
        self.register_buffer("edge_pairs", edge_tensor, persistent=False)
        edge_in = int(input_dim) + 5
        self.edge_heads = nn.ModuleDict()
        for key, _, _, _, _ in pairs:
            head = nn.Sequential(
                nn.LayerNorm(edge_in),
                nn.Linear(edge_in, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            self.edge_heads[key] = head

    @classmethod
    def _expand_pair_specs(cls, pair_specs: list[str] | tuple[str, ...] | None) -> set[frozenset[str]]:
        specs = list(pair_specs or ["wet_water_concrete_hard_adjacent"])
        expanded: list[str] = []
        for item in specs:
            key = str(item).strip()
            if key in cls._ALIASES:
                expanded.extend(cls._ALIASES[key])
            else:
                expanded.append(key)
        requested: set[frozenset[str]] = set()
        for item in expanded:
            parts = [part.strip() for part in str(item).replace("<->", "|").replace(",", "|").split("|") if part.strip()]
            if len(parts) != 2:
                continue
            requested.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
        return requested

    @classmethod
    def _parse_pairs(
        cls,
        spec: RSCDFactorSpec,
        pair_specs: list[str] | tuple[str, ...] | None,
    ) -> list[tuple[str, int, int, str, str]]:
        requested = cls._expand_pair_specs(pair_specs)
        idx_to_class = {int(idx): canonical_class_label(name) for name, idx in spec.class_to_idx.items()}
        parsed: list[tuple[str, int, int, str, str]] = []
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            pair_names = frozenset((idx_to_class[left], idx_to_class[right]))
            if pair_names not in requested:
                continue
            key = f"p{left}_{right}"
            parsed.append((key, left, right, str(pair.axis), cls._infer_kind(idx_to_class[left], idx_to_class[right], str(pair.axis))))
        return parsed

    @staticmethod
    def _infer_kind(left_name: str, right_name: str, axis: str) -> str:
        joined = f"{left_name}|{right_name}"
        if "concrete" in joined and ("water_" in joined or "wet_" in joined):
            return f"wet_water_concrete_{axis}"
        if "asphalt" in joined and ("water_" in joined or "wet_" in joined):
            return f"wet_water_asphalt_{axis}"
        if "dry_concrete" in joined:
            return f"dry_concrete_{axis}"
        return str(axis)

    @staticmethod
    def _pair_features(logits: torch.Tensor, probs: torch.Tensor, left: int, right: int) -> torch.Tensor:
        lp = probs[:, left : left + 1]
        rp = probs[:, right : right + 1]
        pair_mass = (lp + rp).clamp(0.0, 1.0)
        prob_gap = (lp - rp).abs().clamp(0.0, 1.0)
        logit_gap = torch.tanh(0.25 * (logits[:, left : left + 1] - logits[:, right : right + 1]).abs())
        local = torch.cat([lp, rp], dim=1).clamp_min(1e-6)
        local_norm = local / local.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(local_norm * local_norm.log()).sum(dim=1, keepdim=True) / 0.6931471805599453
        return torch.cat([lp, rp, pair_mass, prob_gap + logit_gap, entropy.clamp(0.0, 1.0)], dim=1)

    def _physics_gate(self, kind: str, evidence_stats: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
        if evidence_stats is None:
            return like.new_ones(like.shape)
        stats = evidence_stats.to(device=like.device, dtype=like.dtype)
        specular = stats[:, 8].clamp(0.0, 1.0)
        dark_water = stats[:, 9].clamp(0.0, 1.0)
        wet = stats[:, 10].clamp(0.0, 1.0)
        rough = stats[:, 11].clamp(0.0, 1.0)
        erasure = stats[:, 12].clamp(0.0, 1.0)
        winter = torch.maximum(stats[:, 13], stats[:, 14]).clamp(0.0, 1.0)
        artifact_guard = (1.0 - 0.55 * winter).clamp(0.20, 1.0)
        film = (0.50 * wet + 0.30 * dark_water + 0.20 * specular).clamp(0.0, 1.0)
        if "wet_water_concrete_roughness" in kind:
            visible_rough = (0.70 * rough + 0.30 * (1.0 - erasure)).clamp(0.0, 1.0)
            gate = (0.25 + 0.75 * visible_rough).clamp(0.0, 1.0)
            gate = gate * (0.35 + 0.65 * film).clamp(0.0, 1.0)
        elif "wet_water_concrete_friction" in kind:
            gate = (0.30 + 0.70 * film).clamp(0.0, 1.0)
        elif "dry_concrete_roughness" in kind:
            gate = (0.35 + 0.65 * rough).clamp(0.0, 1.0) * (1.0 - 0.60 * film).clamp(0.20, 1.0)
        elif "asphalt" in kind and "roughness" in kind:
            gate = (0.30 + 0.70 * torch.maximum(rough, film)).clamp(0.0, 1.0)
        else:
            gate = like.new_ones(like.shape)
        return (gate * artifact_guard).clamp(0.0, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        condition: torch.Tensor,
        evidence_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor] | torch.Tensor]]:
        if self.scale <= 0.0 or self.edge_pairs.numel() == 0:
            zero = torch.zeros_like(logits)
            return logits, {"residual": zero, "raw": {}, "delta": {}, "gate": {}}
        probs = F.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
        top_conf = probs.max(dim=1).values
        confidence_gate = torch.sigmoid(
            (float(self.confidence_protect) - top_conf) * float(self.confidence_temperature)
        ).to(dtype=logits.dtype)
        residual = torch.zeros_like(logits)
        raw_dict: dict[str, torch.Tensor] = {}
        delta_dict: dict[str, torch.Tensor] = {}
        gate_dict: dict[str, torch.Tensor] = {}
        for edge_id, key in enumerate(self.edge_keys):
            left = int(self.edge_pairs[edge_id, 0])
            right = int(self.edge_pairs[edge_id, 1])
            pair_features = self._pair_features(logits, probs, left, right).to(device=condition.device, dtype=condition.dtype)
            pair_mass = pair_features[:, 2].clamp(0.0, 1.0)
            pair_gap = pair_features[:, 3].clamp(0.0, 1.0)
            pair_entropy = pair_features[:, 4].clamp(0.0, 1.0)
            boundary_gate = torch.sigmoid((self.gate_margin - pair_gap) * self.gate_temperature)
            physics_gate = self._physics_gate(self.edge_kinds[edge_id], evidence_stats, pair_mass)
            gate = pair_mass * boundary_gate * pair_entropy * confidence_gate * physics_gate
            if self.gate_floor > 0.0:
                gate = float(self.gate_floor) + (1.0 - float(self.gate_floor)) * gate
            raw = self.edge_heads[key](torch.cat([condition.to(dtype=pair_features.dtype), pair_features], dim=1)).squeeze(1)
            delta = (float(self.scale) * gate * torch.tanh(raw)).to(dtype=logits.dtype)
            residual[:, left] = residual[:, left] + delta
            residual[:, right] = residual[:, right] - delta
            raw_dict[key] = raw
            delta_dict[key] = delta
            gate_dict[key] = gate
        return logits + residual, {
            "residual": residual,
            "raw": raw_dict,
            "delta": delta_dict,
            "gate": gate_dict,
        }


class TriStateWetConcreteBoundaryExpert(nn.Module):
    """Direct hard-pair expert for wet/water concrete roughness coupling.

    The current calibrated head is mostly linear-head driven, so token-only
    roughness reliability can be too indirect. This module routes the same
    tri-state evidence directly to selected adjacent RSCD hard-pair boundaries.
    It is still mechanism constrained: it only acts on configured wet/water
    concrete friction or roughness edges and moves probability along that edge.
    """

    def __init__(
        self,
        class_to_idx: dict[str, int],
        pairs: list[str] | tuple[str, ...] | None = None,
        *,
        hidden_dim: int = 64,
        scale: float = 0.08,
        gate_margin: float = 0.85,
        gate_temperature: float = 5.0,
        gate_floor: float = 0.0,
        confidence_protect: float = 0.78,
        confidence_temperature: float = 16.0,
        dropout: float = 0.0,
        severe_protect: bool = False,
        severe_protect_prob: float = 0.30,
        severe_protect_raw_margin: float = 0.0,
        severe_protect_temperature: float = 12.0,
        severe_protect_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.gate_margin = float(gate_margin)
        self.gate_temperature = float(gate_temperature)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.confidence_protect = float(confidence_protect)
        self.confidence_temperature = float(confidence_temperature)
        self.severe_protect = bool(severe_protect)
        self.severe_protect_prob = float(severe_protect_prob)
        self.severe_protect_raw_margin = float(severe_protect_raw_margin)
        self.severe_protect_temperature = float(severe_protect_temperature)
        self.severe_protect_strength = min(max(float(severe_protect_strength), 0.0), 1.0)
        self.spec = build_rscd_factor_spec(class_to_idx)
        self.idx_to_class = {int(idx): name for name, idx in self.spec.class_to_idx.items()}
        configured: set[frozenset[str]] = set()
        for item in pairs or []:
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) != 2:
                raise ValueError(f"tristate wet-concrete boundary pair must contain two class names: {item}")
            configured.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

        self.pair_specs: list[dict[str, Any]] = []
        for pair in self.spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            left_name = canonical_class_label(self.idx_to_class[left])
            right_name = canonical_class_label(self.idx_to_class[right])
            pair_names = frozenset((left_name, right_name))
            if configured and pair_names not in configured:
                continue
            left_factor = self.spec.class_to_factor[left].tolist()
            right_factor = self.spec.class_to_factor[right].tolist()
            if min(left_factor + right_factor) < 0:
                continue
            f_labels = FACTOR_LABELS["friction"]
            m_labels = FACTOR_LABELS["material"]
            r_labels = FACTOR_LABELS["roughness"]
            lf, lm, lr = left_factor
            rf, rm, rr = right_factor
            friction_names = {f_labels[int(lf)], f_labels[int(rf)]}
            material_names = {m_labels[int(lm)], m_labels[int(rm)]}
            if material_names != {"concrete"}:
                continue
            if str(pair.axis) == "roughness":
                if int(lf) != int(rf) or f_labels[int(lf)] not in {"wet", "water"}:
                    continue
                rough_rank = {"none": -1, "smooth": 0, "slight": 1, "severe": 2}
                left_rank = rough_rank.get(r_labels[int(lr)], -1)
                right_rank = rough_rank.get(r_labels[int(rr)], -1)
                if min(left_rank, right_rank) < 0 or left_rank == right_rank:
                    continue
                mode = "roughness"
                sign = 1.0 if left_rank > right_rank else -1.0
                if left_rank == 2:
                    severe_index = left
                    severe_direction = 1.0
                elif right_rank == 2:
                    severe_index = right
                    severe_direction = -1.0
                else:
                    severe_index = None
                    severe_direction = 0.0
            elif str(pair.axis) == "friction":
                if friction_names != {"wet", "water"} or int(lm) != int(rm):
                    continue
                mode = "friction"
                sign = 1.0 if f_labels[int(lf)] == "water" else -1.0
                severe_index = None
                severe_direction = 0.0
            else:
                continue
            self.pair_specs.append(
                {
                    "key": f"p{left}_{right}",
                    "left": left,
                    "right": right,
                    "left_name": left_name,
                    "right_name": right_name,
                    "mode": mode,
                    "hand_sign": sign,
                    "severe_index": severe_index,
                    "severe_direction": severe_direction,
                }
            )
        input_dim = C3PhysicsEvidenceStats.out_dim + 20 + 7 + 5 + 2
        self.experts = nn.ModuleDict()
        self.hand_scales = nn.ParameterDict()
        for spec in self.pair_specs:
            key = str(spec["key"])
            expert = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 2),
            )
            last = expert[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            self.experts[key] = expert
            self.hand_scales[key] = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _pair_features(logits: torch.Tensor, probs: torch.Tensor, left: int, right: int) -> torch.Tensor:
        lp = probs[:, left : left + 1]
        rp = probs[:, right : right + 1]
        pair_mass = (lp + rp).clamp(0.0, 1.0)
        prob_gap = (lp - rp).abs().clamp(0.0, 1.0)
        logit_gap = torch.tanh(0.25 * (logits[:, left : left + 1] - logits[:, right : right + 1]).abs())
        local = torch.cat([lp, rp], dim=1).clamp_min(1e-6)
        local_norm = local / local.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(local_norm * local_norm.log()).sum(dim=1, keepdim=True) / 0.6931471805599453
        return torch.cat([lp, rp, pair_mass, prob_gap + logit_gap, entropy.clamp(0.0, 1.0)], dim=1)

    @staticmethod
    def _hand_raw(states: torch.Tensor, mode: str, sign: float) -> torch.Tensor:
        true_visible = states[:, 0]
        hidden_by_film = states[:, 1]
        pseudo_rough = states[:, 2]
        film_conflict = states[:, 3]
        wetness = states[:, 5]
        geometric_rough = states[:, 6]
        if mode == "roughness":
            raw = true_visible + 0.35 * geometric_rough - 0.60 * pseudo_rough - 0.35 * hidden_by_film
        else:
            raw = hidden_by_film + 0.55 * wetness + 0.35 * film_conflict - 0.35 * true_visible
        return float(sign) * raw

    def forward(
        self,
        logits: torch.Tensor,
        evidence_stats: torch.Tensor,
        pair_value_evidence: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.pair_specs or self.scale <= 0.0:
            return logits, {
                "residual": torch.zeros_like(logits),
                "raw": {},
                "delta": {},
                "gate": {},
                "hand_raw": {},
                "severe_protect": {},
            }
        probs = F.softmax(logits, dim=1)
        residual = torch.zeros_like(logits)
        value_vector = pair_value_evidence["vector"].to(device=logits.device, dtype=logits.dtype)
        evidence_stats = evidence_stats.to(device=logits.device, dtype=logits.dtype)
        states, hand_gate = PseudoRoughnessAwareReliability._hand_states(value_vector, evidence_stats)
        raw_dict: dict[str, torch.Tensor] = {}
        delta_dict: dict[str, torch.Tensor] = {}
        gate_dict: dict[str, torch.Tensor] = {}
        hand_raw_dict: dict[str, torch.Tensor] = {}
        severe_protect_dict: dict[str, torch.Tensor] = {}
        top_conf = probs.amax(dim=1, keepdim=True)
        confidence_gate = torch.sigmoid(
            (float(self.confidence_protect) - top_conf) * float(self.confidence_temperature)
        ).squeeze(1)
        for spec in self.pair_specs:
            key = str(spec["key"])
            left = int(spec["left"])
            right = int(spec["right"])
            pair_features = self._pair_features(logits, probs, left, right).to(dtype=logits.dtype)
            signed_gap = torch.tanh(0.25 * (logits[:, left : left + 1] - logits[:, right : right + 1]))
            mode_flag = logits.new_full((logits.size(0), 1), 1.0 if spec["mode"] == "roughness" else -1.0)
            x = torch.cat([evidence_stats, value_vector, states.to(dtype=logits.dtype), pair_features, signed_gap, mode_flag], dim=1)
            out = self.experts[key](x)
            hand_raw = self._hand_raw(states, str(spec["mode"]), float(spec["hand_sign"])).to(dtype=logits.dtype)
            raw = out[:, 0] + self.hand_scales[key].to(dtype=logits.dtype) * hand_raw
            learned_gate = torch.sigmoid(out[:, 1])
            if self.gate_floor > 0.0:
                learned_gate = self.gate_floor + (1.0 - self.gate_floor) * learned_gate
            pair_mass = pair_features[:, 2].clamp(0.0, 1.0)
            gap = pair_features[:, 3].clamp(0.0, 2.0)
            boundary_gate = torch.sigmoid((self.gate_margin - gap) * self.gate_temperature) * pair_mass
            tri_gate = hand_gate.to(dtype=logits.dtype).squeeze(1).clamp(0.0, 1.0)
            gate = (boundary_gate * learned_gate * tri_gate * confidence_gate).clamp(0.0, 1.0)
            protect_gate = torch.zeros_like(gate)
            severe_index = spec.get("severe_index")
            if self.severe_protect and str(spec["mode"]) == "roughness" and severe_index is not None:
                severe_prob = probs[:, int(severe_index)].clamp(0.0, 1.0)
                severe_prob_gate = torch.sigmoid(
                    (severe_prob - float(self.severe_protect_prob)) * float(self.severe_protect_temperature)
                )
                toward_severe = raw * float(spec.get("severe_direction", 0.0))
                away_gate = torch.sigmoid(
                    (float(self.severe_protect_raw_margin) - toward_severe)
                    * float(self.severe_protect_temperature)
                )
                protect_gate = (severe_prob_gate * away_gate).clamp(0.0, 1.0)
                gate = gate * (1.0 - float(self.severe_protect_strength) * protect_gate).clamp(0.0, 1.0)
            delta = self.scale * gate * torch.tanh(raw)
            residual[:, left] = residual[:, left] + delta
            residual[:, right] = residual[:, right] - delta
            raw_dict[key] = raw
            delta_dict[key] = delta
            gate_dict[key] = gate
            hand_raw_dict[key] = hand_raw
            severe_protect_dict[key] = protect_gate
        return logits + residual, {
            "residual": residual,
            "raw": raw_dict,
            "delta": delta_dict,
            "gate": gate_dict,
            "hand_raw": hand_raw_dict,
            "severe_protect": severe_protect_dict,
            "states": states,
            "hand_gate": hand_gate.to(dtype=logits.dtype),
        }


class ParetoEdgeExpertLogitCorrector(nn.Module):
    """Validation-selected RSCD edge expert with physics-gated no-harm locality.

    The module turns Pareto-safe source->target edge discoveries into a
    trainable single-model mechanism. It is deliberately narrower than a generic
    residual head: an edge can move only when the current logits are already in
    the same local ambiguity region as the selected rule, and the movement is
    gated by RSCD physical evidence from PhysicsTexture/pair-value statistics.
    """

    def __init__(
        self,
        class_to_idx: dict[str, int],
        rules: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        *,
        hidden_dim: int = 48,
        scale: float = 1.0,
        gate_temperature: float = 18.0,
        gate_floor: float = 0.0,
        learned_gate_bias: float = -1.6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.gate_temperature = float(gate_temperature)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.rule_specs: list[dict[str, Any]] = []
        canonical_to_idx = {canonical_class_label(name): int(idx) for name, idx in class_to_idx.items()}
        seen: set[tuple[int, int, int, float, float]] = set()
        for item in rules or []:
            if not isinstance(item, dict):
                continue
            rule = item.get("rule_raw", item)
            if not isinstance(rule, dict):
                continue
            source_name = canonical_class_label(str(rule.get("source", "")))
            target_name = canonical_class_label(str(rule.get("target", "")))
            if source_name not in canonical_to_idx or target_name not in canonical_to_idx:
                continue
            source = int(canonical_to_idx[source_name])
            target = int(canonical_to_idx[target_name])
            topk = int(rule.get("topk", 2))
            margin = float(rule.get("margin", 0.2))
            delta = float(rule.get("delta", 0.2))
            key = (source, target, topk, round(margin, 6), round(delta, 6))
            if key in seen:
                continue
            seen.add(key)
            self.rule_specs.append(
                {
                    "key": f"e{source}_{target}_{len(self.rule_specs)}",
                    "source": source,
                    "target": target,
                    "source_name": source_name,
                    "target_name": target_name,
                    "topk": max(1, topk),
                    "margin": margin,
                    "delta": max(delta, 0.0),
                    "kind": self._edge_kind(source_name, target_name),
                }
            )

        input_dim = C3PhysicsEvidenceStats.out_dim + 20 + 5 + 1 + 1
        self.experts = nn.ModuleDict()
        self.edge_bias = nn.ParameterDict()
        for spec in self.rule_specs:
            expert = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 2),
            )
            last = expert[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
                with torch.no_grad():
                    last.bias[1].fill_(float(learned_gate_bias))
            self.experts[str(spec["key"])] = expert
            self.edge_bias[str(spec["key"])] = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _parts(label: str) -> tuple[str, str, str]:
        parts = canonical_class_label(label).split("_")
        if len(parts) == 2:
            return parts[0], parts[1], "none"
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        return label, "", "none"

    @classmethod
    def _edge_kind(cls, source: str, target: str) -> str:
        sf, sm, sr = cls._parts(source)
        tf, tm, tr = cls._parts(target)
        if {source, target} == {"dry_concrete_severe", "dry_concrete_slight"}:
            return "dry_concrete_slight_severe"
        if {sm, tm} == {"mud", "gravel"}:
            return "gravel_mud_texture"
        if sf == tf == "water" and sm == tm == "asphalt":
            return "water_asphalt_film"
        if sm == tm == "concrete" and {sf, tf} <= {"wet", "water"}:
            return "wet_water_concrete_film"
        if sm == tm == "concrete" and {sf, tf} == {"dry", "wet"}:
            return "dry_wet_concrete_smooth"
        if sf == tf == "dry" and sr == tr == "smooth" and {sm, tm} == {"asphalt", "concrete"}:
            return "dry_paved_material_smooth"
        if sf == tf == "water" and {sm, tm} == {"gravel", "concrete"}:
            return "water_loose_concrete"
        return "generic_local_edge"

    @staticmethod
    def _pair_features(logits: torch.Tensor, probs: torch.Tensor, source: int, target: int) -> torch.Tensor:
        sp = probs[:, source : source + 1]
        tp = probs[:, target : target + 1]
        pair_mass = (sp + tp).clamp(0.0, 1.0)
        prob_gap = (sp - tp).abs().clamp(0.0, 1.0)
        logit_gap = torch.tanh(0.25 * (logits[:, source : source + 1] - logits[:, target : target + 1]).abs())
        local = torch.cat([sp, tp], dim=1).clamp_min(1e-6)
        local_norm = local / local.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(local_norm * local_norm.log()).sum(dim=1, keepdim=True) / 0.6931471805599453
        return torch.cat([sp, tp, pair_mass, prob_gap + logit_gap, entropy.clamp(0.0, 1.0)], dim=1)

    @staticmethod
    def _hand_gate(kind: str, evidence: torch.Tensor, value_vector: torch.Tensor) -> torch.Tensor:
        form_gates = CoupledFormExpertConditioner._hand_form_gates(value_vector).to(dtype=value_vector.dtype)
        macro_rough = value_vector[:, 0:1].clamp(0.0, 1.0)
        micro_rough = value_vector[:, 1:2].clamp(0.0, 1.0)
        film = value_vector[:, 2:3].clamp(0.0, 1.0)
        artifact = value_vector[:, 3:4].clamp(0.0, 1.0)
        dark_water_top = value_vector[:, 13:14].clamp(0.0, 1.0)
        erasure_top = value_vector[:, 17:18].clamp(0.0, 1.0)
        value_std = value_vector[:, 19:20].clamp(0.0, 1.0)
        wet = evidence[:, 10:11].clamp(0.0, 1.0)
        rough = evidence[:, 11:12].clamp(0.0, 1.0)
        erasure = evidence[:, 12:13].clamp(0.0, 1.0)
        snow = evidence[:, 13:14].clamp(0.0, 1.0)
        ice = evidence[:, 14:15].clamp(0.0, 1.0)
        artifact_guard = (1.0 - 0.60 * torch.maximum(torch.maximum(snow, ice), artifact)).clamp(0.12, 1.0)
        if kind == "dry_concrete_slight_severe":
            gate = form_gates[:, 1:2]
        elif kind == "gravel_mud_texture":
            gate = torch.maximum(form_gates[:, 5:6], 0.45 * micro_rough + 0.35 * value_std + 0.20 * rough)
        elif kind == "water_asphalt_film":
            gate = torch.maximum(form_gates[:, 2:3], 0.45 * dark_water_top + 0.35 * film + 0.20 * erasure_top)
        elif kind == "wet_water_concrete_film":
            gate = torch.maximum(form_gates[:, 3:4], 0.40 * wet + 0.30 * film + 0.30 * erasure)
        elif kind == "dry_wet_concrete_smooth":
            gate = torch.maximum(0.45 * film + 0.35 * wet + 0.20 * erasure, 1.0 - rough)
        elif kind == "dry_paved_material_smooth":
            gate = (0.42 * (1.0 - film) + 0.28 * (1.0 - erasure) + 0.18 * rough + 0.12 * macro_rough).clamp(0.0, 1.0)
        elif kind == "water_loose_concrete":
            gate = torch.maximum(form_gates[:, 5:6], 0.35 * film + 0.35 * micro_rough + 0.30 * rough)
        else:
            gate = form_gates.max(dim=1, keepdim=True).values
        return (gate * artifact_guard).clamp(0.0, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        evidence_stats: torch.Tensor,
        pair_value_evidence: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.rule_specs or self.scale <= 0.0:
            return logits, {
                "residual": torch.zeros_like(logits),
                "raw": {},
                "delta": {},
                "gate": {},
                "hand_gate": {},
            }
        probs = F.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
        order = torch.argsort(logits, dim=1, descending=True)
        pred = order[:, 0]
        value_vector = pair_value_evidence["vector"].to(device=logits.device, dtype=logits.dtype)
        evidence_stats = evidence_stats.to(device=logits.device, dtype=logits.dtype)
        residual = torch.zeros_like(logits)
        raw_dict: dict[str, torch.Tensor] = {}
        delta_dict: dict[str, torch.Tensor] = {}
        gate_dict: dict[str, torch.Tensor] = {}
        hand_gate_dict: dict[str, torch.Tensor] = {}
        for spec in self.rule_specs:
            key = str(spec["key"])
            source = int(spec["source"])
            target = int(spec["target"])
            topk = max(1, min(int(spec["topk"]), int(logits.shape[1])))
            source_logit = logits[:, source : source + 1]
            target_logit = logits[:, target : target + 1]
            margin = source_logit - target_logit
            source_pred = pred.eq(source).to(dtype=logits.dtype).unsqueeze(1)
            target_in_topk = order[:, :topk].eq(target).any(dim=1).to(dtype=logits.dtype).unsqueeze(1)
            close_gate = torch.sigmoid((float(spec["margin"]) - margin) * float(self.gate_temperature))
            pair_features = self._pair_features(logits, probs, source, target)
            hand_gate = self._hand_gate(str(spec["kind"]), evidence_stats, value_vector).to(dtype=logits.dtype)
            x = torch.cat([evidence_stats, value_vector, pair_features, torch.tanh(0.25 * margin), hand_gate], dim=1)
            raw_gate = self.experts[key](x)
            raw = raw_gate[:, 0:1] + self.edge_bias[key].to(dtype=logits.dtype).view(1, 1)
            learned_gate = torch.sigmoid(raw_gate[:, 1:2])
            if self.gate_floor > 0.0:
                learned_gate = self.gate_floor + (1.0 - self.gate_floor) * learned_gate
            gate = (source_pred * target_in_topk * close_gate * pair_features[:, 2:3] * hand_gate * learned_gate).clamp(0.0, 1.0)
            delta = float(self.scale) * float(spec["delta"]) * gate.squeeze(1) * torch.tanh(raw.squeeze(1))
            residual[:, target] = residual[:, target] + delta
            residual[:, source] = residual[:, source] - 0.25 * delta
            raw_dict[key] = raw.squeeze(1)
            delta_dict[key] = delta
            gate_dict[key] = gate.squeeze(1)
            hand_gate_dict[key] = hand_gate.squeeze(1)
        return logits + residual, {
            "residual": residual,
            "raw": raw_dict,
            "delta": delta_dict,
            "gate": gate_dict,
            "hand_gate": hand_gate_dict,
        }


class SourceReliableBoundaryFeatureRouter(nn.Module):
    """Source-reliability gated feature router for validated RSCD hard edges.

    Full-validation logit rules showed a small but robust principle: moving
    samples out of a source class is safest only when that source class is
    already reliable. This module converts that diagnostic into a pre-head
    feature mechanism. A route opens only inside a local source->target
    ambiguity region, is weighted by validation source reliability, and moves
    features along the classifier discriminant direction ``w_target - w_source``
    instead of adding a free-form residual.
    """

    def __init__(
        self,
        class_to_idx: dict[str, int],
        routes: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        *,
        hidden_dim: int = 32,
        scale: float = 0.012,
        gate_temperature: float = 6.0,
        physics_gate_floor: float = 0.0,
        base_strength: float = 0.0,
        source_temperature: float = 28.0,
        learned_gate_bias: float = -2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = max(float(scale), 0.0)
        self.gate_temperature = float(gate_temperature)
        self.physics_gate_floor = min(max(float(physics_gate_floor), 0.0), 1.0)
        self.base_strength = float(base_strength)
        self.source_temperature = float(source_temperature)
        canonical_to_idx = {canonical_class_label(name): int(idx) for name, idx in class_to_idx.items()}
        self.route_specs: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int, float]] = set()
        for item in routes or []:
            if not isinstance(item, dict):
                continue
            source_name = canonical_class_label(str(item.get("source", "")))
            target_name = canonical_class_label(str(item.get("target", "")))
            if source_name not in canonical_to_idx or target_name not in canonical_to_idx:
                continue
            source_f1 = float(item.get("source_f1", item.get("source_val_f1", 1.0)))
            min_source_f1 = float(item.get("min_source_f1", 0.0))
            if source_f1 + 1e-12 < min_source_f1:
                continue
            source = int(canonical_to_idx[source_name])
            target = int(canonical_to_idx[target_name])
            topk = max(int(item.get("topk", 2)), 1)
            margin = float(item.get("margin", 0.5))
            key = (source, target, topk, round(margin, 6))
            if key in seen:
                continue
            seen.add(key)
            reliability = float(torch.sigmoid(torch.tensor((source_f1 - min_source_f1) * self.source_temperature)))
            self.route_specs.append(
                {
                    "key": f"srbr_{source}_{target}_{len(self.route_specs)}",
                    "source": source,
                    "target": target,
                    "source_name": source_name,
                    "target_name": target_name,
                    "topk": topk,
                    "margin": margin,
                    "source_f1": source_f1,
                    "min_source_f1": min_source_f1,
                    "reliability": min(max(reliability, 0.0), 1.0),
                    "route_scale": max(float(item.get("route_scale", item.get("scale", 1.0))), 0.0),
                    "kind": str(item.get("kind", self._edge_kind(source_name, target_name))),
                }
            )

        context_dim = C3PhysicsEvidenceStats.out_dim + 5 + 3
        self.gates = nn.ModuleDict()
        for spec in self.route_specs:
            net = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
            last = net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            self.gates[str(spec["key"])] = net

    @staticmethod
    def _parts(label: str) -> tuple[str, str, str]:
        parts = canonical_class_label(label).split("_")
        if len(parts) == 2:
            return parts[0], parts[1], "none"
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        return label, "", "none"

    @classmethod
    def _edge_kind(cls, source: str, target: str) -> str:
        sf, sm, sr = cls._parts(source)
        tf, tm, tr = cls._parts(target)
        if sf == tf == "dry" and sm == tm == "concrete" and {sr, tr} <= {"smooth", "slight", "severe"}:
            return "dry_concrete_roughness"
        if sf == tf == "dry" and sm == tm == "asphalt" and {sr, tr} <= {"smooth", "slight", "severe"}:
            return "dry_asphalt_roughness"
        if sf == tf and sf in {"wet", "water"} and sm == tm == "concrete" and sr != tr:
            return "concrete_film_roughness"
        if sm == tm == "concrete" and sr == tr == "smooth" and {sf, tf} <= {"wet", "water"}:
            return "wet_water_smooth_film"
        if sm == tm == "concrete" and {sf, tf} <= {"wet", "water"}:
            return "wet_water_concrete_film"
        if sf == tf == "water" and sm == tm == "asphalt":
            return "water_asphalt_film"
        return "generic"

    @staticmethod
    def _pair_features(logits: torch.Tensor, probs: torch.Tensor, source: int, target: int) -> torch.Tensor:
        sp = probs[:, source : source + 1]
        tp = probs[:, target : target + 1]
        pair_mass = (sp + tp).clamp(0.0, 1.0)
        prob_gap = (sp - tp).abs().clamp(0.0, 1.0)
        logit_gap = torch.tanh(0.25 * (logits[:, source : source + 1] - logits[:, target : target + 1]).abs())
        local = torch.cat([sp, tp], dim=1).clamp_min(1e-6)
        local_norm = local / local.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(local_norm * local_norm.log()).sum(dim=1, keepdim=True) / 0.6931471805599453
        return torch.cat([sp, tp, pair_mass, prob_gap + logit_gap, entropy.clamp(0.0, 1.0)], dim=1)

    @staticmethod
    def _physics_gate(kind: str, evidence: torch.Tensor) -> torch.Tensor:
        grad_std = evidence[:, 5:6].clamp(0.0, 1.0)
        lap_mean = evidence[:, 6:7].clamp(0.0, 1.0)
        contrast = evidence[:, 7:8].clamp(0.0, 1.0)
        specular = evidence[:, 8:9].clamp(0.0, 1.0)
        dark_water = evidence[:, 9:10].clamp(0.0, 1.0)
        wet = evidence[:, 10:11].clamp(0.0, 1.0)
        rough = evidence[:, 11:12].clamp(0.0, 1.0)
        erasure = evidence[:, 12:13].clamp(0.0, 1.0)
        snow = evidence[:, 13:14].clamp(0.0, 1.0)
        ice = evidence[:, 14:15].clamp(0.0, 1.0)
        film = torch.clamp(0.45 * wet + 0.25 * dark_water + 0.15 * specular + 0.15 * erasure, 0.0, 1.0)
        artifact_guard = (1.0 - torch.maximum(torch.maximum(snow, ice), 0.65 * film)).clamp(0.10, 1.0)
        if kind == "dry_concrete_roughness":
            visible_rough = torch.clamp(0.36 * rough + 0.26 * grad_std + 0.22 * lap_mean + 0.16 * contrast, 0.0, 1.0)
            gate = torch.sigmoid((visible_rough - 0.105) * 18.0)
            return (gate * artifact_guard).clamp(0.0, 1.0)
        if kind == "dry_asphalt_roughness":
            visible_rough = torch.clamp(0.34 * rough + 0.28 * grad_std + 0.22 * lap_mean + 0.16 * contrast, 0.0, 1.0)
            moderate = torch.sigmoid((visible_rough - 0.085) * 20.0) * torch.sigmoid((0.175 - visible_rough) * 20.0)
            dry_guard = (1.0 - torch.clamp(0.60 * wet + 0.25 * dark_water + 0.15 * specular, 0.0, 1.0)).clamp(0.15, 1.0)
            return (moderate * dry_guard * artifact_guard).clamp(0.0, 1.0)
        if kind == "dry_concrete_moderate_roughness":
            visible_rough = torch.clamp(0.36 * rough + 0.26 * grad_std + 0.22 * lap_mean + 0.16 * contrast, 0.0, 1.0)
            moderate = torch.sigmoid((visible_rough - 0.082) * 22.0) * torch.sigmoid((0.178 - visible_rough) * 22.0)
            dry_guard = (1.0 - torch.clamp(0.58 * wet + 0.24 * dark_water + 0.18 * specular, 0.0, 1.0)).clamp(0.16, 1.0)
            erasure_guard = (1.0 - 0.40 * erasure).clamp(0.20, 1.0)
            return (moderate * dry_guard * erasure_guard * artifact_guard).clamp(0.0, 1.0)
        if kind == "wet_concrete_high_roughness":
            visible_rough = torch.clamp(0.35 * rough + 0.25 * grad_std + 0.24 * lap_mean + 0.16 * contrast, 0.0, 1.0)
            film = torch.clamp(0.48 * wet + 0.24 * specular + 0.18 * erasure + 0.10 * dark_water, 0.0, 1.0)
            hidden_rough = torch.clamp(0.62 * visible_rough + 0.24 * erasure + 0.14 * film, 0.0, 1.0)
            high = torch.sigmoid((hidden_rough - 0.145) * 18.0)
            water_guard = (1.0 - 0.45 * dark_water).clamp(0.18, 1.0)
            snow_ice_guard = (1.0 - 0.55 * torch.maximum(snow, ice)).clamp(0.10, 1.0)
            return (high * film * water_guard * snow_ice_guard).clamp(0.0, 1.0)
        if kind == "water_asphalt_shallow_film_roughness":
            visible_rough = torch.clamp(0.32 * rough + 0.27 * grad_std + 0.23 * lap_mean + 0.18 * contrast, 0.0, 1.0)
            film = torch.clamp(0.46 * dark_water + 0.30 * wet + 0.16 * specular + 0.08 * erasure, 0.0, 1.0)
            shallow_rough = torch.sigmoid((visible_rough - 0.070) * 20.0) * torch.sigmoid((0.165 - visible_rough) * 18.0)
            snow_ice_guard = (1.0 - 0.52 * torch.maximum(snow, ice)).clamp(0.10, 1.0)
            return (shallow_rough * film * snow_ice_guard).clamp(0.0, 1.0)
        if kind == "wet_asphalt_high_roughness":
            visible_rough = torch.clamp(0.34 * rough + 0.27 * grad_std + 0.22 * lap_mean + 0.17 * contrast, 0.0, 1.0)
            wet_film = torch.clamp(0.50 * wet + 0.30 * specular + 0.20 * erasure, 0.0, 1.0)
            high = torch.sigmoid((visible_rough - 0.135) * 18.0)
            not_deep_water = (1.0 - 0.50 * dark_water).clamp(0.16, 1.0)
            return (high * wet_film * not_deep_water * artifact_guard).clamp(0.0, 1.0)
        if kind == "dry_loose_gravel_texture":
            texture = torch.clamp(0.40 * rough + 0.24 * grad_std + 0.20 * lap_mean + 0.16 * contrast, 0.0, 1.0)
            dry_guard = (1.0 - torch.clamp(0.52 * wet + 0.28 * dark_water + 0.20 * specular, 0.0, 1.0)).clamp(0.16, 1.0)
            granular = torch.sigmoid((texture - 0.115) * 18.0) * (1.0 - 0.35 * erasure).clamp(0.18, 1.0)
            return (granular * dry_guard * artifact_guard).clamp(0.0, 1.0)
        if kind == "concrete_film_roughness":
            visible_rough = torch.clamp(0.34 * rough + 0.25 * grad_std + 0.24 * lap_mean + 0.17 * contrast, 0.0, 1.0)
            hidden_tail = torch.clamp(0.52 * visible_rough + 0.30 * erasure + 0.18 * film, 0.0, 1.0)
            gate = torch.sigmoid((hidden_tail - 0.125) * 16.0)
            return gate * (1.0 - 0.45 * torch.maximum(snow, ice)).clamp(0.12, 1.0)
        if kind == "wet_water_smooth_film":
            smooth = torch.sigmoid((0.115 - rough) * 14.0)
            gate = torch.clamp(0.50 * film + 0.30 * wet + 0.20 * erasure, 0.0, 1.0)
            return gate * smooth * (1.0 - 0.40 * torch.maximum(snow, ice)).clamp(0.12, 1.0)
        if kind == "water_concrete_moderate_roughness":
            visible_rough = torch.clamp(0.32 * rough + 0.25 * grad_std + 0.25 * lap_mean + 0.18 * contrast, 0.0, 1.0)
            hidden_rough = torch.clamp(0.48 * visible_rough + 0.32 * erasure + 0.20 * film, 0.0, 1.0)
            moderate = torch.sigmoid((hidden_rough - 0.105) * 18.0) * torch.sigmoid((0.190 - hidden_rough) * 18.0)
            water_guard = torch.clamp(0.45 * dark_water + 0.30 * wet + 0.25 * erasure, 0.0, 1.0)
            snow_ice_guard = (1.0 - 0.55 * torch.maximum(snow, ice)).clamp(0.08, 1.0)
            return (moderate * water_guard * snow_ice_guard).clamp(0.0, 1.0)
        if kind == "wet_water_concrete_film":
            gate = torch.clamp(0.42 * film + 0.30 * wet + 0.28 * erasure, 0.0, 1.0)
            return gate * (1.0 - 0.35 * torch.maximum(snow, ice)).clamp(0.15, 1.0)
        if kind == "water_asphalt_film":
            gate = torch.clamp(0.45 * dark_water + 0.35 * film + 0.20 * erasure, 0.0, 1.0)
            return gate * (1.0 - 0.35 * torch.maximum(snow, ice)).clamp(0.15, 1.0)
        return torch.ones_like(wet) * 0.5

    def forward(
        self,
        feature: torch.Tensor,
        evidence: torch.Tensor,
        base_logits: torch.Tensor,
        classifier_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.route_specs or self.scale <= 0.0:
            return feature, {
                "delta": torch.zeros_like(feature),
                "gate": {},
                "raw": {},
                "physics_gate": {},
            }

        probs = F.softmax(base_logits, dim=1)
        pred = base_logits.argmax(dim=1)
        max_topk = min(max(int(spec["topk"]) for spec in self.route_specs), base_logits.size(1))
        top_order = base_logits.topk(k=max_topk, dim=1).indices
        total_delta = torch.zeros_like(feature)
        gate_dict: dict[str, torch.Tensor] = {}
        raw_dict: dict[str, torch.Tensor] = {}
        physics_dict: dict[str, torch.Tensor] = {}
        for spec in self.route_specs:
            source = int(spec["source"])
            target = int(spec["target"])
            topk = min(int(spec["topk"]), top_order.size(1))
            in_region = pred.eq(source).float().view(-1, 1)
            in_region = in_region * top_order[:, :topk].eq(target).any(dim=1).float().view(-1, 1)
            source_target_margin = base_logits[:, source : source + 1] - base_logits[:, target : target + 1]
            close_gate = torch.sigmoid((float(spec["margin"]) - source_target_margin) * self.gate_temperature)
            physics_gate = self._physics_gate(str(spec["kind"]), evidence).to(device=feature.device, dtype=feature.dtype)
            if self.physics_gate_floor > 0.0:
                physics_gate = self.physics_gate_floor + (1.0 - self.physics_gate_floor) * physics_gate
            reliability = feature.new_full((feature.size(0), 1), float(spec["reliability"]))
            pair_features = self._pair_features(base_logits, probs, source, target).to(dtype=feature.dtype)
            context = torch.cat([evidence.to(dtype=feature.dtype), pair_features, physics_gate, reliability, close_gate], dim=1)
            raw = self.base_strength + torch.tanh(self.gates[str(spec["key"])](context))
            gate = in_region.to(dtype=feature.dtype) * close_gate.to(dtype=feature.dtype) * physics_gate * reliability
            direction = classifier_weight[target] - classifier_weight[source]
            direction = F.normalize(direction.detach().to(device=feature.device, dtype=feature.dtype), dim=0).view(1, -1)
            delta = float(self.scale) * float(spec.get("route_scale", 1.0)) * gate * raw.to(dtype=feature.dtype) * direction
            total_delta = total_delta + delta
            gate_dict[str(spec["key"])] = gate.squeeze(1)
            raw_dict[str(spec["key"])] = raw.squeeze(1)
            physics_dict[str(spec["key"])] = physics_gate.squeeze(1)
        return feature + total_delta, {
            "delta": total_delta,
            "gate": gate_dict,
            "raw": raw_dict,
            "physics_gate": physics_dict,
        }


class C3FaRNetSurfaceClassifier(nn.Module):
    """Causal-compositional-coupled factorization model for RSCD-27."""

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        backbone: str = "convnext_tiny",
        embedding_dim: int = 768,
        pretrained: bool = False,
        dropout: float = 0.2,
        token_dim: int = 256,
        pair_rank: int = 8,
        triple_rank: int = 8,
        head_type: str = "coupled_tensor",
        hybrid_coupled_scale: float = 0.10,
        hardpair_correction_scale: float = 0.08,
        hardpair_margin_scale: float = 0.18,
        hardpair_gate_margin: float = 1.00,
        hardpair_gate_temperature: float = 4.00,
        hardpair_error_gate_bias_init: float = -3.5,
        hardpair_error_gate_floor: float = 0.0,
        hardpair_physics_gate: str = "none",
        hardpair_physics_gate_floor: float = 0.0,
        hardpair_physics_gate_power: float = 1.0,
        use_hardpair_value_signed_adapter: bool = False,
        hardpair_value_adapter_pairs: list[str] | tuple[str, ...] | None = None,
        hardpair_value_adapter_pair_scales: dict[str, float] | None = None,
        hardpair_value_adapter_hidden_dim: int = 48,
        hardpair_value_adapter_scale: float = 0.10,
        hardpair_value_adapter_gate_floor: float = 0.0,
        hardpair_value_adapter_value_aug_std: float = 0.0,
        hardpair_value_adapter_dropout: float = 0.0,
        use_hardpair_value_rough_tail_guard: bool = False,
        hardpair_value_rough_tail_guard_pairs: list[str] | tuple[str, ...] | None = None,
        hardpair_value_rough_tail_guard_threshold: float = 0.52,
        hardpair_value_rough_tail_guard_temperature: float = 10.0,
        hardpair_value_rough_tail_guard_strength: float = 0.85,
        hardpair_focus_classes: list[str] | tuple[str, ...] | None = None,
        hardpair_focus_boundaries: list[str] | tuple[str, ...] | None = None,
        hardpair_disabled_class_pairs: list[str] | tuple[str, ...] | None = None,
        hardpair_pair_scales: dict[str, float] | None = None,
        hardpair_protected_classes: list[str] | tuple[str, ...] | None = None,
        hardpair_sample_protect_classes: list[str] | tuple[str, ...] | None = None,
        hardpair_sample_protect_threshold: float = 0.08,
        hardpair_sample_protect_temperature: float = 30.0,
        boundary_use_physics_feature: bool = False,
        use_physics_branch: bool = True,
        physics_dim: int = 96,
        physics_quality_cues: bool = True,
        physics_quality_region_cues: bool = False,
        use_semantic_physics_attention_branch: bool = True,
        semantic_physics_attention_dim: int = 64,
        use_local_physics_field_branch: bool = True,
        local_physics_field_dim: int = 64,
        local_physics_field_scale: float = 0.08,
        use_physics_texture_stem_adapter: bool = False,
        physics_texture_stem_hidden_dim: int = 32,
        physics_texture_stem_scale: float = 0.035,
        physics_texture_stem_gate_floor: float = 0.18,
        use_scale_space_roughness_stem_adapter: bool = False,
        scale_space_roughness_stem_hidden_dim: int = 32,
        scale_space_roughness_stem_scale: float = 0.020,
        scale_space_roughness_stem_gate_floor: float = 0.10,
        scale_space_roughness_stem_gate_mode: str = "concrete_tail",
        scale_space_roughness_stem_dry_tail_weight: float = 1.0,
        scale_space_roughness_stem_wet_hidden_tail_weight: float = 1.0,
        use_pair_value_stem_conditioner: bool = False,
        pair_value_stem_hidden_dim: int = 32,
        pair_value_stem_scale: float = 0.018,
        pair_value_stem_gate_floor: float = 0.0,
        pair_value_stem_value_aug_std: float = 0.0,
        pair_value_stem_learned_gate_bias: float = -1.6,
        use_wet_water_concrete_film_depth_stem_conditioner: bool = False,
        wet_water_concrete_film_depth_stem_hidden_dim: int = 36,
        wet_water_concrete_film_depth_stem_scale: float = 0.030,
        wet_water_concrete_film_depth_stem_gate_floor: float = 0.04,
        wet_water_concrete_film_depth_stem_learned_gate_bias: float = -1.2,
        use_water_concrete_topology_texture_stem_conditioner: bool = False,
        water_concrete_topology_texture_stem_hidden_dim: int = 36,
        water_concrete_topology_texture_stem_scale: float = 0.026,
        water_concrete_topology_texture_stem_gate_floor: float = 0.03,
        water_concrete_topology_texture_stem_learned_gate_bias: float = -1.25,
        use_scale_space_roughness_token_conditioner: bool = False,
        scale_space_roughness_token_hidden_dim: int = 64,
        scale_space_roughness_token_scale: float = 0.10,
        scale_space_roughness_token_gate_floor: float = 0.0,
        scale_space_roughness_token_dry_tail_weight: float = 1.0,
        scale_space_roughness_token_wet_hidden_tail_weight: float = 0.75,
        use_local_global_scale_token_conditioner: bool = False,
        local_global_scale_token_hidden_dim: int = 96,
        local_global_scale_token_scale: float = 0.050,
        local_global_scale_token_feature_scale: float = 0.010,
        local_global_scale_token_gate_floor: float = 0.0,
        local_global_scale_token_dropout: float = 0.0,
        local_global_scale_token_detach_context: bool = False,
        use_water_film_roughness_feature_film: bool = False,
        water_film_roughness_feature_film_hidden_dim: int = 128,
        water_film_roughness_feature_film_scale: float = 0.080,
        water_film_roughness_feature_film_gate_floor: float = 0.0,
        water_film_roughness_feature_film_max_gamma: float = 0.18,
        water_film_roughness_feature_film_dropout: float = 0.0,
        water_film_roughness_feature_film_detach_context: bool = False,
        use_pseudo_roughness_aware_reliability: bool = False,
        roughness_reliability_use_coupling_context: bool = False,
        pseudo_roughness_aware_reliability_hidden_dim: int = 128,
        pseudo_roughness_aware_reliability_scale: float = 0.060,
        pseudo_roughness_aware_reliability_rho_scale: float = 0.100,
        pseudo_roughness_aware_reliability_gate_floor: float = 0.0,
        pseudo_roughness_aware_reliability_dropout: float = 0.0,
        pseudo_roughness_aware_reliability_detach_context: bool = False,
        use_spatial_factor_queries: bool = False,
        spatial_factor_query_map_dim: int = 768,
        spatial_factor_query_heads: int = 4,
        spatial_factor_query_scale: float = 0.25,
        use_dry_concrete_roughness_vor_residual: bool = False,
        dry_concrete_roughness_hidden_dim: int = 48,
        dry_concrete_roughness_scale: float = 0.12,
        dry_concrete_roughness_gate_threshold: float = 0.12,
        dry_concrete_roughness_gate_temperature: float = 14.0,
        use_dry_concrete_ordinal_chart_residual: bool = False,
        dry_concrete_ordinal_chart_hidden_dim: int = 48,
        dry_concrete_ordinal_chart_scale: float = 0.06,
        dry_concrete_ordinal_chart_gate_threshold: float = 0.12,
        dry_concrete_ordinal_chart_gate_temperature: float = 14.0,
        dry_concrete_ordinal_chart_protect_confidence: float = 0.72,
        dry_concrete_ordinal_chart_protect_temperature: float = 18.0,
        use_dry_concrete_validation_transition: bool = False,
        dry_concrete_validation_transition_source: str = "dry_concrete_severe",
        dry_concrete_validation_transition_target: str = "dry_concrete_slight",
        dry_concrete_validation_transition_topk: int = 2,
        dry_concrete_validation_transition_margin: float = 0.20,
        dry_concrete_validation_transition_delta: float = 0.20,
        use_backbone_isolated_dry_concrete_adapter: bool = False,
        backbone_isolated_dry_concrete_branch_dim: int = 96,
        backbone_isolated_dry_concrete_hidden_dim: int = 64,
        backbone_isolated_dry_concrete_scale: float = 0.18,
        backbone_isolated_dry_concrete_gate_threshold: float = 0.10,
        backbone_isolated_dry_concrete_gate_temperature: float = 14.0,
        backbone_isolated_dry_concrete_dropout: float = 0.02,
        backbone_isolated_dry_concrete_output_mode: str = "free",
        use_dry_concrete_pair_signed_selector: bool = False,
        dry_concrete_pair_selector_pairs: list[str] | tuple[str, ...] | None = None,
        dry_concrete_pair_selector_hidden_dim: int = 48,
        dry_concrete_pair_selector_shift_scale: float = 0.65,
        dry_concrete_pair_selector_gain_scale: float = 0.50,
        dry_concrete_pair_selector_direct_delta_scale: float = 0.0,
        dry_concrete_pair_selector_safe_margin: float = 0.20,
        dry_concrete_pair_selector_safe_temperature: float = 28.0,
        protected_factor_adapter_rank: int = 6,
        protected_factor_adapter_hidden_dim: int = 96,
        protected_factor_adapter_scale: float = 0.08,
        protected_factor_adapter_gate_margin: float = 0.18,
        protected_factor_adapter_gate_temperature: float = 10.0,
        protected_factor_adapter_active_classes: list[str] | tuple[str, ...] | None = None,
        protected_factor_adapter_protected_classes: list[str] | tuple[str, ...] | None = None,
        use_feature_value_boundary_corrector: bool = False,
        feature_value_boundary_pairs: list[str] | tuple[str, ...] | None = None,
        feature_value_boundary_hidden_dim: int = 64,
        feature_value_boundary_scale: float = 0.22,
        feature_value_boundary_gate_margin: float = 1.05,
        feature_value_boundary_gate_temperature: float = 4.5,
        feature_value_boundary_gate_floor: float = 0.0,
        feature_value_boundary_value_aug_std: float = 0.0,
        feature_value_boundary_dropout: float = 0.0,
        feature_value_boundary_severe_tail_protect: bool = False,
        feature_value_boundary_severe_tail_protect_pairs: list[str] | tuple[str, ...] | None = None,
        feature_value_boundary_severe_tail_protect_strength: float = 0.85,
        feature_value_boundary_severe_tail_protect_prob: float = 0.34,
        feature_value_boundary_severe_tail_protect_tail_threshold: float = 0.115,
        feature_value_boundary_severe_tail_protect_temperature: float = 16.0,
        use_water_concrete_opponent_feature_conditioner: bool = False,
        water_concrete_opponent_pairs: list[str] | tuple[str, ...] | None = None,
        water_concrete_opponent_hidden_dim: int = 64,
        water_concrete_opponent_scale: float = 0.018,
        water_concrete_opponent_gate_margin: float = 1.08,
        water_concrete_opponent_gate_temperature: float = 4.5,
        water_concrete_opponent_gate_floor: float = 0.03,
        water_concrete_opponent_value_aug_std: float = 0.0,
        water_concrete_opponent_dropout: float = 0.0,
        use_factor_graph_edge_flow_corrector: bool = False,
        factor_graph_edge_flow_pairs: list[str] | tuple[str, ...] | None = None,
        factor_graph_edge_flow_hidden_dim: int = 64,
        factor_graph_edge_flow_scale: float = 0.10,
        factor_graph_edge_flow_gate_margin: float = 0.90,
        factor_graph_edge_flow_gate_temperature: float = 4.0,
        factor_graph_edge_flow_gate_floor: float = 0.0,
        factor_graph_edge_flow_confidence_protect: float = 0.74,
        factor_graph_edge_flow_confidence_temperature: float = 16.0,
        factor_graph_edge_flow_dropout: float = 0.0,
        use_tristate_wet_concrete_boundary_expert: bool = False,
        tristate_wet_concrete_boundary_pairs: list[str] | tuple[str, ...] | None = None,
        tristate_wet_concrete_boundary_hidden_dim: int = 64,
        tristate_wet_concrete_boundary_scale: float = 0.08,
        tristate_wet_concrete_boundary_gate_margin: float = 0.85,
        tristate_wet_concrete_boundary_gate_temperature: float = 5.0,
        tristate_wet_concrete_boundary_gate_floor: float = 0.0,
        tristate_wet_concrete_boundary_confidence_protect: float = 0.78,
        tristate_wet_concrete_boundary_confidence_temperature: float = 16.0,
        tristate_wet_concrete_boundary_dropout: float = 0.0,
        tristate_wet_concrete_boundary_severe_protect: bool = False,
        tristate_wet_concrete_boundary_severe_protect_prob: float = 0.30,
        tristate_wet_concrete_boundary_severe_protect_raw_margin: float = 0.0,
        tristate_wet_concrete_boundary_severe_protect_temperature: float = 12.0,
        tristate_wet_concrete_boundary_severe_protect_strength: float = 1.0,
        use_closed_set_factor_redistributor: bool = False,
        closed_set_factor_redistributor_sets: list[str] | tuple[str, ...] | None = None,
        closed_set_factor_redistributor_hidden_dim: int = 96,
        closed_set_factor_redistributor_scale: float = 0.06,
        closed_set_factor_redistributor_gate_floor: float = 0.0,
        closed_set_factor_redistributor_mass_threshold: float = 0.08,
        closed_set_factor_redistributor_margin_threshold: float = 0.25,
        closed_set_factor_redistributor_temperature: float = 8.0,
        closed_set_factor_redistributor_dropout: float = 0.0,
        closed_set_factor_redistributor_gate_bias_init: float = -2.5,
        closed_set_factor_redistributor_use_graph_locality_guard: bool = False,
        closed_set_factor_redistributor_graph_max_distance: float = 2.0,
        closed_set_factor_redistributor_graph_guard_floor: float = 0.0,
        closed_set_factor_redistributor_graph_guard_temperature: float = 12.0,
        use_backbone_family_ordinal_no_spill_adapter: bool = False,
        backbone_family_ordinal_no_spill_hidden_dim: int = 96,
        backbone_family_ordinal_no_spill_family_embed_dim: int = 12,
        backbone_family_ordinal_no_spill_scale: float = 0.18,
        backbone_family_ordinal_no_spill_gate_threshold: float = 0.055,
        backbone_family_ordinal_no_spill_gate_temperature: float = 10.0,
        backbone_family_ordinal_no_spill_dropout: float = 0.02,
        backbone_family_ordinal_no_spill_families: Any = None,
        use_pair_value_mechanism_conditioner: bool = False,
        pair_value_mechanism_hidden_dim: int = 64,
        pair_value_mechanism_feature_scale: float = 0.010,
        pair_value_mechanism_token_scale: float = 0.060,
        pair_value_mechanism_gate_floor: float = 0.0,
        pair_value_mechanism_value_aug_std: float = 0.0,
        pair_value_mechanism_protect_classes: list[str] | tuple[str, ...] | None = None,
        pair_value_mechanism_protect_threshold: float = 0.18,
        pair_value_mechanism_protect_temperature: float = 24.0,
        use_coupled_form_expert_conditioner: bool = False,
        coupled_form_expert_hidden_dim: int = 64,
        coupled_form_expert_feature_scale: float = 0.010,
        coupled_form_expert_token_scale: float = 0.060,
        coupled_form_expert_gate_floor: float = 0.0,
        coupled_form_expert_value_aug_std: float = 0.0,
        coupled_form_expert_learned_gate_bias: float = -1.5,
        coupled_form_expert_detach_context: bool = False,
        coupled_form_expert_protect_classes: list[str] | tuple[str, ...] | None = None,
        coupled_form_expert_protect_threshold: float = 0.18,
        coupled_form_expert_protect_temperature: float = 24.0,
        use_pareto_edge_expert: bool = False,
        pareto_edge_expert_rules: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        pareto_edge_expert_hidden_dim: int = 48,
        pareto_edge_expert_scale: float = 1.0,
        pareto_edge_expert_gate_temperature: float = 18.0,
        pareto_edge_expert_gate_floor: float = 0.0,
        pareto_edge_expert_learned_gate_bias: float = -1.6,
        pareto_edge_expert_dropout: float = 0.0,
        use_source_reliable_boundary_router: bool = False,
        source_reliable_boundary_routes: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        source_reliable_boundary_hidden_dim: int = 32,
        source_reliable_boundary_scale: float = 0.012,
        source_reliable_boundary_gate_temperature: float = 6.0,
        source_reliable_boundary_physics_gate_floor: float = 0.0,
        source_reliable_boundary_base_strength: float = 0.0,
        source_reliable_boundary_source_temperature: float = 28.0,
        source_reliable_boundary_learned_gate_bias: float = -2.0,
        source_reliable_boundary_dropout: float = 0.0,
        expose_hardpair_pair_value_evidence: bool = False,
    ) -> None:
        super().__init__()
        self.spec = build_rscd_factor_spec(class_to_idx)
        self.idx_to_class = {int(idx): name for name, idx in self.spec.class_to_idx.items()}
        self.expose_hardpair_pair_value_evidence = bool(expose_hardpair_pair_value_evidence)
        self.input_stem_adapter = (
            PhysicsTextureStemAdapter(
                hidden_dim=int(physics_texture_stem_hidden_dim),
                scale=float(physics_texture_stem_scale),
                gate_floor=float(physics_texture_stem_gate_floor),
            )
            if bool(use_physics_texture_stem_adapter)
            else None
        )
        self.scale_space_roughness_stem_adapter = (
            ScaleSpaceRoughnessStemAdapter(
                hidden_dim=int(scale_space_roughness_stem_hidden_dim),
                scale=float(scale_space_roughness_stem_scale),
                gate_floor=float(scale_space_roughness_stem_gate_floor),
                gate_mode=str(scale_space_roughness_stem_gate_mode),
                dry_tail_weight=float(scale_space_roughness_stem_dry_tail_weight),
                wet_hidden_tail_weight=float(scale_space_roughness_stem_wet_hidden_tail_weight),
            )
            if bool(use_scale_space_roughness_stem_adapter)
            else None
        )
        self.pair_value_stem_conditioner = (
            PairValueStemConditioner(
                hidden_dim=int(pair_value_stem_hidden_dim),
                scale=float(pair_value_stem_scale),
                gate_floor=float(pair_value_stem_gate_floor),
                value_aug_std=float(pair_value_stem_value_aug_std),
                learned_gate_bias=float(pair_value_stem_learned_gate_bias),
            )
            if bool(use_pair_value_stem_conditioner)
            else None
        )
        self.wet_water_concrete_film_depth_stem_conditioner = (
            WetWaterConcreteFilmDepthStemConditioner(
                hidden_dim=int(wet_water_concrete_film_depth_stem_hidden_dim),
                scale=float(wet_water_concrete_film_depth_stem_scale),
                gate_floor=float(wet_water_concrete_film_depth_stem_gate_floor),
                learned_gate_bias=float(wet_water_concrete_film_depth_stem_learned_gate_bias),
            )
            if bool(use_wet_water_concrete_film_depth_stem_conditioner)
            else None
        )
        self.water_concrete_topology_texture_stem_conditioner = (
            WaterConcreteTopologyTextureStemConditioner(
                hidden_dim=int(water_concrete_topology_texture_stem_hidden_dim),
                scale=float(water_concrete_topology_texture_stem_scale),
                gate_floor=float(water_concrete_topology_texture_stem_gate_floor),
                learned_gate_bias=float(water_concrete_topology_texture_stem_learned_gate_bias),
            )
            if bool(use_water_concrete_topology_texture_stem_conditioner)
            else None
        )
        self.backbone = build_backbone(backbone, int(embedding_dim), pretrained=pretrained)
        self.physics_branch = (
            PhysicsTextureBranch(
                int(physics_dim),
                quality_cues=bool(physics_quality_cues),
                quality_region_cues=bool(physics_quality_region_cues),
            )
            if use_physics_branch
            else None
        )
        self.semantic_physics_attention_branch = (
            SemanticPhysicsAttentionBranch(int(semantic_physics_attention_dim))
            if use_semantic_physics_attention_branch
            else None
        )
        self.local_physics_field_branch = (
            LocalPhysicsFieldBranch(int(local_physics_field_dim))
            if use_local_physics_field_branch
            else None
        )
        self.local_physics_field_scale = float(local_physics_field_scale)
        self.local_physics_field_adapter = (
            nn.Sequential(nn.LayerNorm(int(local_physics_field_dim)), nn.Linear(int(local_physics_field_dim), int(embedding_dim)))
            if self.local_physics_field_branch is not None
            else None
        )
        if self.local_physics_field_adapter is not None:
            last = self.local_physics_field_adapter[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        low_dim = (
            (int(physics_dim) if self.physics_branch is not None else 0)
            + (int(semantic_physics_attention_dim) if self.semantic_physics_attention_branch is not None else 0)
        )
        self.head_dim = int(embedding_dim) + low_dim
        self.norm = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.evidence_stats = C3PhysicsEvidenceStats()
        self.spatial_evidence_maps = C3PhysicsEvidenceMaps() if bool(use_spatial_factor_queries) else None
        self.decoder = FactorQueryDecoder(
            self.head_dim,
            int(token_dim),
            evidence_dim=self.evidence_stats.out_dim,
            spatial_map_dim=int(spatial_factor_query_map_dim) if bool(use_spatial_factor_queries) else None,
            spatial_evidence_dim=C3PhysicsEvidenceMaps.out_channels,
            spatial_heads=int(spatial_factor_query_heads),
            spatial_scale=float(spatial_factor_query_scale),
        )
        self.roughness_reliability = RoughnessReliability(
            int(token_dim),
            evidence_dim=self.evidence_stats.out_dim,
            use_coupling_context=bool(roughness_reliability_use_coupling_context),
        )
        self.pair_value_mechanism_conditioner = (
            PairValueMechanismConditioner(
                feature_dim=self.head_dim,
                token_dim=int(token_dim),
                hidden_dim=int(pair_value_mechanism_hidden_dim),
                feature_scale=float(pair_value_mechanism_feature_scale),
                token_scale=float(pair_value_mechanism_token_scale),
                gate_floor=float(pair_value_mechanism_gate_floor),
                value_aug_std=float(pair_value_mechanism_value_aug_std),
            )
            if bool(use_pair_value_mechanism_conditioner)
            else None
        )
        self.coupled_form_expert_conditioner = (
            CoupledFormExpertConditioner(
                feature_dim=self.head_dim,
                token_dim=int(token_dim),
                hidden_dim=int(coupled_form_expert_hidden_dim),
                feature_scale=float(coupled_form_expert_feature_scale),
                token_scale=float(coupled_form_expert_token_scale),
                gate_floor=float(coupled_form_expert_gate_floor),
                value_aug_std=float(coupled_form_expert_value_aug_std),
                learned_gate_bias=float(coupled_form_expert_learned_gate_bias),
                detach_context=bool(coupled_form_expert_detach_context),
            )
            if bool(use_coupled_form_expert_conditioner)
            else None
        )
        self.scale_space_roughness_token_conditioner = (
            ScaleSpaceRoughnessTokenConditioner(
                token_dim=int(token_dim),
                hidden_dim=int(scale_space_roughness_token_hidden_dim),
                scale=float(scale_space_roughness_token_scale),
                gate_floor=float(scale_space_roughness_token_gate_floor),
                dry_tail_weight=float(scale_space_roughness_token_dry_tail_weight),
                wet_hidden_tail_weight=float(scale_space_roughness_token_wet_hidden_tail_weight),
            )
            if bool(use_scale_space_roughness_token_conditioner)
            else None
        )
        self.local_global_scale_token_conditioner = (
            LocalGlobalScaleTokenConditioner(
                feature_dim=self.head_dim,
                token_dim=int(token_dim),
                evidence_dim=self.evidence_stats.out_dim,
                local_dim=int(local_physics_field_dim),
                hidden_dim=int(local_global_scale_token_hidden_dim),
                scale=float(local_global_scale_token_scale),
                feature_scale=float(local_global_scale_token_feature_scale),
                gate_floor=float(local_global_scale_token_gate_floor),
                dropout=float(local_global_scale_token_dropout),
                detach_context=bool(local_global_scale_token_detach_context),
            )
            if bool(use_local_global_scale_token_conditioner)
            else None
        )
        self.water_film_roughness_feature_film = (
            WaterFilmRoughnessFeatureFiLM(
                feature_dim=self.head_dim,
                hidden_dim=int(water_film_roughness_feature_film_hidden_dim),
                scale=float(water_film_roughness_feature_film_scale),
                gate_floor=float(water_film_roughness_feature_film_gate_floor),
                max_gamma=float(water_film_roughness_feature_film_max_gamma),
                dropout=float(water_film_roughness_feature_film_dropout),
                detach_context=bool(water_film_roughness_feature_film_detach_context),
            )
            if bool(use_water_film_roughness_feature_film)
            else None
        )
        self.pseudo_roughness_aware_reliability = (
            PseudoRoughnessAwareReliability(
                token_dim=int(token_dim),
                evidence_dim=self.evidence_stats.out_dim,
                hidden_dim=int(pseudo_roughness_aware_reliability_hidden_dim),
                scale=float(pseudo_roughness_aware_reliability_scale),
                rho_scale=float(pseudo_roughness_aware_reliability_rho_scale),
                gate_floor=float(pseudo_roughness_aware_reliability_gate_floor),
                dropout=float(pseudo_roughness_aware_reliability_dropout),
                detach_context=bool(pseudo_roughness_aware_reliability_detach_context),
            )
            if bool(use_pseudo_roughness_aware_reliability)
            else None
        )
        head_type = str(head_type)
        if head_type not in {
            "linear",
            "factor_only",
            "coupled_tensor",
            "coupled_no_triple",
            "coupled_no_pairwise",
            "hybrid_coupled",
            "hardpair_gated_coupled",
            "hardpair_pairwise_calibrated",
            "hardpair_error_gated_calibrated",
            "hardpair_margin_directed_calibrated",
            "hardpair_benefit_gated_margin_calibrated",
            "protected_factor_graph_adapter",
        }:
            raise ValueError(f"unknown C3 head_type: {head_type}")
        self.head_type = head_type
        self.boundary_use_physics_feature = bool(boundary_use_physics_feature) and self.physics_branch is not None
        self.hybrid_coupled_scale = float(hybrid_coupled_scale)
        self.hardpair_correction_scale = float(hardpair_correction_scale)
        self.hardpair_margin_scale = float(hardpair_margin_scale)
        self.hardpair_gate_margin = float(hardpair_gate_margin)
        self.hardpair_gate_temperature = float(hardpair_gate_temperature)
        self.hardpair_error_gate_floor = float(hardpair_error_gate_floor)
        self.hardpair_physics_gate = str(hardpair_physics_gate)
        self.hardpair_physics_gate_floor = float(hardpair_physics_gate_floor)
        self.hardpair_physics_gate_power = max(float(hardpair_physics_gate_power), 1e-3)
        self.use_hardpair_value_signed_adapter = bool(use_hardpair_value_signed_adapter)
        self.hardpair_value_adapter_scale = float(hardpair_value_adapter_scale)
        self.hardpair_value_adapter_gate_floor = float(hardpair_value_adapter_gate_floor)
        self.hardpair_value_adapter_value_aug_std = max(float(hardpair_value_adapter_value_aug_std), 0.0)
        self.use_hardpair_value_rough_tail_guard = bool(use_hardpair_value_rough_tail_guard)
        self.hardpair_value_rough_tail_guard_threshold = float(hardpair_value_rough_tail_guard_threshold)
        self.hardpair_value_rough_tail_guard_temperature = float(hardpair_value_rough_tail_guard_temperature)
        self.hardpair_value_rough_tail_guard_strength = float(
            min(max(hardpair_value_rough_tail_guard_strength, 0.0), 1.0)
        )
        value_adapter_pairs: set[frozenset[str]] = set()
        for item in hardpair_value_adapter_pairs or []:
            if isinstance(item, str):
                parts = item.replace("<->", "|").replace(",", "|").split("|")
            else:
                parts = list(item)
            if len(parts) != 2:
                raise ValueError(f"hardpair_value_adapter_pairs entry must contain two class names: {item}")
            value_adapter_pairs.add(frozenset((canonical_class_label(str(parts[0])), canonical_class_label(str(parts[1])))))
        self.hardpair_value_adapter_pairs = value_adapter_pairs
        value_adapter_pair_scales = {}
        for item, scale in (hardpair_value_adapter_pair_scales or {}).items():
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) != 2:
                raise ValueError(f"hardpair_value_adapter_pair_scales key must contain two class names: {item}")
            left_name, right_name = (canonical_class_label(parts[0]), canonical_class_label(parts[1]))
            value_adapter_pair_scales[frozenset((left_name, right_name))] = float(scale)
        self.hardpair_value_adapter_pair_scales = value_adapter_pair_scales
        rough_tail_guard_pairs: set[frozenset[str]] = set()
        for item in hardpair_value_rough_tail_guard_pairs or []:
            if isinstance(item, str):
                parts = item.replace("<->", "|").replace(",", "|").split("|")
            else:
                parts = list(item)
            if len(parts) != 2:
                raise ValueError(f"hardpair_value_rough_tail_guard_pairs entry must contain two class names: {item}")
            rough_tail_guard_pairs.add(
                frozenset((canonical_class_label(str(parts[0])), canonical_class_label(str(parts[1]))))
            )
        self.hardpair_value_rough_tail_guard_pairs = rough_tail_guard_pairs
        focus_classes = {canonical_class_label(name) for name in (hardpair_focus_classes or [])}
        focus_boundaries = {str(name) for name in (hardpair_focus_boundaries or [])}
        disabled_pairs = set()
        for item in hardpair_disabled_class_pairs or []:
            if isinstance(item, str):
                parts = item.replace("<->", "|").replace(",", "|").split("|")
            else:
                parts = list(item)
            if len(parts) != 2:
                raise ValueError(f"hardpair_disabled_class_pairs entry must contain two class names: {item}")
            left_name, right_name = (canonical_class_label(str(parts[0])), canonical_class_label(str(parts[1])))
            disabled_pairs.add(frozenset((left_name, right_name)))
        pair_scales = {}
        for item, scale in (hardpair_pair_scales or {}).items():
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) != 2:
                raise ValueError(f"hardpair_pair_scales key must contain two class names: {item}")
            left_name, right_name = (canonical_class_label(parts[0]), canonical_class_label(parts[1]))
            pair_scales[frozenset((left_name, right_name))] = float(scale)
        self.hardpair_pair_scales = pair_scales
        protected_hardpair_classes = {canonical_class_label(name) for name in (hardpair_protected_classes or [])}
        sample_protect_classes = {canonical_class_label(name) for name in (hardpair_sample_protect_classes or [])}
        pair_value_protect_classes = {
            canonical_class_label(name) for name in (pair_value_mechanism_protect_classes or [])
        }
        coupled_form_expert_protect_classes = {
            canonical_class_label(name) for name in (coupled_form_expert_protect_classes or [])
        }
        idx_to_class = self.idx_to_class
        sample_protect_idx = [
            int(idx)
            for idx, name in idx_to_class.items()
            if canonical_class_label(name) in sample_protect_classes
        ]
        self.register_buffer(
            "hardpair_sample_protect_idx",
            torch.as_tensor(sample_protect_idx, dtype=torch.long),
            persistent=False,
        )
        pair_value_protect_idx = [
            int(idx)
            for idx, name in idx_to_class.items()
            if canonical_class_label(name) in pair_value_protect_classes
        ]
        self.register_buffer(
            "pair_value_mechanism_protect_idx",
            torch.as_tensor(pair_value_protect_idx, dtype=torch.long),
            persistent=False,
        )
        coupled_form_expert_protect_idx = [
            int(idx)
            for idx, name in idx_to_class.items()
            if canonical_class_label(name) in coupled_form_expert_protect_classes
        ]
        self.register_buffer(
            "coupled_form_expert_protect_idx",
            torch.as_tensor(coupled_form_expert_protect_idx, dtype=torch.long),
            persistent=False,
        )
        self.hardpair_sample_protect_threshold = float(hardpair_sample_protect_threshold)
        self.hardpair_sample_protect_temperature = float(hardpair_sample_protect_temperature)
        self.pair_value_mechanism_protect_threshold = float(pair_value_mechanism_protect_threshold)
        self.pair_value_mechanism_protect_temperature = float(pair_value_mechanism_protect_temperature)
        self.coupled_form_expert_protect_threshold = float(coupled_form_expert_protect_threshold)
        self.coupled_form_expert_protect_temperature = float(coupled_form_expert_protect_temperature)
        active_hardpairs = []
        for pair in self.spec.hard_pairs:
            left_name = idx_to_class[int(pair.left)]
            right_name = idx_to_class[int(pair.right)]
            pair_names = frozenset((canonical_class_label(left_name), canonical_class_label(right_name)))
            if pair_names in disabled_pairs:
                active_hardpairs.append(False)
                continue
            if left_name in protected_hardpair_classes or right_name in protected_hardpair_classes:
                active_hardpairs.append(False)
                continue
            if focus_classes or focus_boundaries:
                active = (
                    str(pair.boundary) in focus_boundaries
                    or left_name in focus_classes
                    or right_name in focus_classes
                )
            else:
                active = True
            active_hardpairs.append(bool(active))
        self.hardpair_active = tuple(active_hardpairs)
        self.linear_head = nn.Linear(self.head_dim, len(class_to_idx))
        self.source_reliable_boundary_router = (
            SourceReliableBoundaryFeatureRouter(
                self.spec.class_to_idx,
                source_reliable_boundary_routes,
                hidden_dim=int(source_reliable_boundary_hidden_dim),
                scale=float(source_reliable_boundary_scale),
                gate_temperature=float(source_reliable_boundary_gate_temperature),
                physics_gate_floor=float(source_reliable_boundary_physics_gate_floor),
                base_strength=float(source_reliable_boundary_base_strength),
                source_temperature=float(source_reliable_boundary_source_temperature),
                learned_gate_bias=float(source_reliable_boundary_learned_gate_bias),
                dropout=float(source_reliable_boundary_dropout),
            )
            if bool(use_source_reliable_boundary_router)
            else None
        )
        self.coupled_head = CoupledTensorHead(
            spec=self.spec,
            token_dim=int(token_dim),
            pair_rank=int(pair_rank),
            triple_rank=int(triple_rank),
            use_pairwise=head_type
            in {"coupled_tensor", "coupled_no_triple", "hybrid_coupled", "hardpair_gated_coupled"},
            use_triple=head_type
            in {"coupled_tensor", "coupled_no_pairwise", "hybrid_coupled", "hardpair_gated_coupled"},
        )
        self.boundary_types = tuple(sorted({pair.boundary for pair in self.spec.hard_pairs}))
        judge_in = int(token_dim) + self.evidence_stats.out_dim + 1
        if self.boundary_use_physics_feature:
            judge_in += int(physics_dim)
        self.protected_factor_adapter = (
            FactorGraphProtectedLogitAdapter(
                spec=self.spec,
                idx_to_class=self.idx_to_class,
                input_dim=judge_in,
                rank=int(protected_factor_adapter_rank),
                hidden_dim=int(protected_factor_adapter_hidden_dim),
                scale=float(protected_factor_adapter_scale),
                gate_margin=float(protected_factor_adapter_gate_margin),
                gate_temperature=float(protected_factor_adapter_gate_temperature),
                active_classes=protected_factor_adapter_active_classes,
                protected_classes=protected_factor_adapter_protected_classes,
            )
            if head_type == "protected_factor_graph_adapter"
            else None
        )
        self.boundary_experts = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(judge_in),
                    nn.Linear(judge_in, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                )
                for name in self.boundary_types
            }
        )
        self.pairwise_hardpair_experts = nn.ModuleDict()
        self.pairwise_hardpair_error_gates = nn.ModuleDict()
        self.pairwise_hardpair_margin_heads = nn.ModuleDict()
        self.pairwise_hardpair_value_adapters = nn.ModuleDict()
        needs_pair_expert = head_type in {
            "hardpair_pairwise_calibrated",
            "hardpair_error_gated_calibrated",
        }
        needs_error_gate = head_type in {
            "hardpair_error_gated_calibrated",
            "hardpair_benefit_gated_margin_calibrated",
        }
        needs_margin_head = head_type in {
            "hardpair_margin_directed_calibrated",
            "hardpair_benefit_gated_margin_calibrated",
        }
        pair_gate_in = judge_in + 5
        pair_value_in = 20 + self.evidence_stats.out_dim + 5 + 1
        for pair, active in zip(self.spec.hard_pairs, self.hardpair_active, strict=True):
            if not active:
                continue
            key = self._hardpair_pair_key(pair)
            if needs_pair_expert:
                expert = nn.Sequential(
                    nn.LayerNorm(judge_in),
                    nn.Linear(judge_in, 96),
                    nn.GELU(),
                    nn.Linear(96, 1),
                )
                last = expert[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
                self.pairwise_hardpair_experts[key] = expert
            if needs_error_gate:
                gate = nn.Sequential(
                    nn.LayerNorm(pair_gate_in),
                    nn.Linear(pair_gate_in, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                )
                gate_last = gate[-1]
                if isinstance(gate_last, nn.Linear):
                    nn.init.zeros_(gate_last.weight)
                    nn.init.constant_(gate_last.bias, float(hardpair_error_gate_bias_init))
                self.pairwise_hardpair_error_gates[key] = gate
            if needs_margin_head:
                margin_head = nn.Sequential(
                    nn.LayerNorm(pair_gate_in),
                    nn.Linear(pair_gate_in, 96),
                    nn.GELU(),
                    nn.Linear(96, 1),
                )
                margin_last = margin_head[-1]
                if isinstance(margin_last, nn.Linear):
                    nn.init.zeros_(margin_last.weight)
                    nn.init.zeros_(margin_last.bias)
                self.pairwise_hardpair_margin_heads[key] = margin_head
            left_name = canonical_class_label(self.idx_to_class[int(pair.left)])
            right_name = canonical_class_label(self.idx_to_class[int(pair.right)])
            pair_names = frozenset((left_name, right_name))
            use_value_adapter = self.use_hardpair_value_signed_adapter and (
                not self.hardpair_value_adapter_pairs or pair_names in self.hardpair_value_adapter_pairs
            )
            if use_value_adapter:
                adapter = nn.Sequential(
                    nn.LayerNorm(pair_value_in),
                    nn.Linear(pair_value_in, int(hardpair_value_adapter_hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(hardpair_value_adapter_dropout)),
                    nn.Linear(int(hardpair_value_adapter_hidden_dim), 2),
                )
                adapter_last = adapter[-1]
                if isinstance(adapter_last, nn.Linear):
                    nn.init.zeros_(adapter_last.weight)
                    nn.init.zeros_(adapter_last.bias)
                self.pairwise_hardpair_value_adapters[key] = adapter
        self.dry_concrete_roughness_vor_residual = (
            DryConcreteRoughnessVORResidual(
                class_to_idx=self.spec.class_to_idx,
                hidden_dim=int(dry_concrete_roughness_hidden_dim),
                scale=float(dry_concrete_roughness_scale),
                gate_threshold=float(dry_concrete_roughness_gate_threshold),
                gate_temperature=float(dry_concrete_roughness_gate_temperature),
                zero_init=True,
            )
            if use_dry_concrete_roughness_vor_residual
            else None
        )
        self.dry_concrete_ordinal_chart_residual = (
            DryConcreteOrdinalChartResidual(
                class_to_idx=self.spec.class_to_idx,
                hidden_dim=int(dry_concrete_ordinal_chart_hidden_dim),
                scale=float(dry_concrete_ordinal_chart_scale),
                gate_threshold=float(dry_concrete_ordinal_chart_gate_threshold),
                gate_temperature=float(dry_concrete_ordinal_chart_gate_temperature),
                protect_confidence=float(dry_concrete_ordinal_chart_protect_confidence),
                protect_temperature=float(dry_concrete_ordinal_chart_protect_temperature),
            )
            if use_dry_concrete_ordinal_chart_residual
            else None
        )
        self.dry_concrete_validation_transition = (
            DryConcreteValidationSafeTransition(
                class_to_idx=self.spec.class_to_idx,
                source=str(dry_concrete_validation_transition_source),
                target=str(dry_concrete_validation_transition_target),
                topk=int(dry_concrete_validation_transition_topk),
                margin=float(dry_concrete_validation_transition_margin),
                delta=float(dry_concrete_validation_transition_delta),
            )
            if bool(use_dry_concrete_validation_transition)
            else None
        )
        self.backbone_isolated_dry_concrete_adapter = (
            BackboneIsolatedDryConcreteLogitAdapter(
                class_to_idx=self.spec.class_to_idx,
                branch_dim=int(backbone_isolated_dry_concrete_branch_dim),
                hidden_dim=int(backbone_isolated_dry_concrete_hidden_dim),
                scale=float(backbone_isolated_dry_concrete_scale),
                gate_threshold=float(backbone_isolated_dry_concrete_gate_threshold),
                gate_temperature=float(backbone_isolated_dry_concrete_gate_temperature),
                dropout=float(backbone_isolated_dry_concrete_dropout),
                output_mode=str(backbone_isolated_dry_concrete_output_mode),
            )
            if bool(use_backbone_isolated_dry_concrete_adapter)
            else None
        )
        selector_pairs = dry_concrete_pair_selector_pairs or ["dry_concrete_severe|dry_concrete_slight"]
        self.dry_concrete_pair_selector_pairs = set()
        for item in selector_pairs:
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) != 2:
                raise ValueError(f"dry_concrete_pair_selector_pairs entry must contain two class names: {item}")
            self.dry_concrete_pair_selector_pairs.add(
                frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))
            )
        self.dry_concrete_pair_signed_selector = (
            DryConcretePairSignedSelector(
                hidden_dim=int(dry_concrete_pair_selector_hidden_dim),
                shift_scale=float(dry_concrete_pair_selector_shift_scale),
                gain_scale=float(dry_concrete_pair_selector_gain_scale),
            )
            if bool(use_dry_concrete_pair_signed_selector)
            else None
        )
        self.dry_concrete_pair_selector_direct_delta_scale = float(dry_concrete_pair_selector_direct_delta_scale)
        self.dry_concrete_pair_selector_safe_margin = float(dry_concrete_pair_selector_safe_margin)
        self.dry_concrete_pair_selector_safe_temperature = float(dry_concrete_pair_selector_safe_temperature)
        self.feature_value_boundary_corrector = (
            FeatureValueBoundaryCorrector(
                self.spec.class_to_idx,
                feature_value_boundary_pairs,
                hidden_dim=int(feature_value_boundary_hidden_dim),
                scale=float(feature_value_boundary_scale),
                gate_margin=float(feature_value_boundary_gate_margin),
                gate_temperature=float(feature_value_boundary_gate_temperature),
                gate_floor=float(feature_value_boundary_gate_floor),
                value_aug_std=float(feature_value_boundary_value_aug_std),
                dropout=float(feature_value_boundary_dropout),
                severe_tail_protect=bool(feature_value_boundary_severe_tail_protect),
                severe_tail_protect_pairs=feature_value_boundary_severe_tail_protect_pairs,
                severe_tail_protect_strength=float(feature_value_boundary_severe_tail_protect_strength),
                severe_tail_protect_prob=float(feature_value_boundary_severe_tail_protect_prob),
                severe_tail_protect_tail_threshold=float(feature_value_boundary_severe_tail_protect_tail_threshold),
                severe_tail_protect_temperature=float(feature_value_boundary_severe_tail_protect_temperature),
            )
            if bool(use_feature_value_boundary_corrector)
            else None
        )
        self.water_concrete_opponent_feature_conditioner = (
            WaterConcreteOpponentFeatureConditioner(
                self.spec.class_to_idx,
                water_concrete_opponent_pairs,
                feature_dim=self.head_dim,
                hidden_dim=int(water_concrete_opponent_hidden_dim),
                scale=float(water_concrete_opponent_scale),
                gate_margin=float(water_concrete_opponent_gate_margin),
                gate_temperature=float(water_concrete_opponent_gate_temperature),
                gate_floor=float(water_concrete_opponent_gate_floor),
                value_aug_std=float(water_concrete_opponent_value_aug_std),
                dropout=float(water_concrete_opponent_dropout),
            )
            if bool(use_water_concrete_opponent_feature_conditioner)
            else None
        )
        self.factor_graph_edge_flow_corrector = (
            FactorGraphEdgeFlowCorrector(
                self.spec.class_to_idx,
                factor_graph_edge_flow_pairs,
                input_dim=judge_in,
                hidden_dim=int(factor_graph_edge_flow_hidden_dim),
                scale=float(factor_graph_edge_flow_scale),
                gate_margin=float(factor_graph_edge_flow_gate_margin),
                gate_temperature=float(factor_graph_edge_flow_gate_temperature),
                gate_floor=float(factor_graph_edge_flow_gate_floor),
                confidence_protect=float(factor_graph_edge_flow_confidence_protect),
                confidence_temperature=float(factor_graph_edge_flow_confidence_temperature),
                dropout=float(factor_graph_edge_flow_dropout),
            )
            if bool(use_factor_graph_edge_flow_corrector)
            else None
        )
        self.tristate_wet_concrete_boundary_expert = (
            TriStateWetConcreteBoundaryExpert(
                self.spec.class_to_idx,
                tristate_wet_concrete_boundary_pairs,
                hidden_dim=int(tristate_wet_concrete_boundary_hidden_dim),
                scale=float(tristate_wet_concrete_boundary_scale),
                gate_margin=float(tristate_wet_concrete_boundary_gate_margin),
                gate_temperature=float(tristate_wet_concrete_boundary_gate_temperature),
                gate_floor=float(tristate_wet_concrete_boundary_gate_floor),
                confidence_protect=float(tristate_wet_concrete_boundary_confidence_protect),
                confidence_temperature=float(tristate_wet_concrete_boundary_confidence_temperature),
                dropout=float(tristate_wet_concrete_boundary_dropout),
                severe_protect=bool(tristate_wet_concrete_boundary_severe_protect),
                severe_protect_prob=float(tristate_wet_concrete_boundary_severe_protect_prob),
                severe_protect_raw_margin=float(tristate_wet_concrete_boundary_severe_protect_raw_margin),
                severe_protect_temperature=float(tristate_wet_concrete_boundary_severe_protect_temperature),
                severe_protect_strength=float(tristate_wet_concrete_boundary_severe_protect_strength),
            )
            if bool(use_tristate_wet_concrete_boundary_expert)
            else None
        )
        self.closed_set_factor_redistributor = (
            ClosedSetFactorRedistributor(
                self.spec.class_to_idx,
                closed_set_factor_redistributor_sets,
                input_dim=judge_in,
                hidden_dim=int(closed_set_factor_redistributor_hidden_dim),
                scale=float(closed_set_factor_redistributor_scale),
                gate_floor=float(closed_set_factor_redistributor_gate_floor),
                mass_threshold=float(closed_set_factor_redistributor_mass_threshold),
                margin_threshold=float(closed_set_factor_redistributor_margin_threshold),
                temperature=float(closed_set_factor_redistributor_temperature),
                dropout=float(closed_set_factor_redistributor_dropout),
                gate_bias_init=float(closed_set_factor_redistributor_gate_bias_init),
                use_graph_locality_guard=bool(closed_set_factor_redistributor_use_graph_locality_guard),
                graph_max_distance=float(closed_set_factor_redistributor_graph_max_distance),
                graph_guard_floor=float(closed_set_factor_redistributor_graph_guard_floor),
                graph_guard_temperature=float(closed_set_factor_redistributor_graph_guard_temperature),
            )
            if bool(use_closed_set_factor_redistributor)
            else None
        )
        self.backbone_family_ordinal_no_spill_adapter = (
            BackboneFamilyOrdinalNoSpillAdapter(
                class_to_idx=self.spec.class_to_idx,
                hidden_dim=int(backbone_family_ordinal_no_spill_hidden_dim),
                family_embed_dim=int(backbone_family_ordinal_no_spill_family_embed_dim),
                scale=float(backbone_family_ordinal_no_spill_scale),
                gate_threshold=float(backbone_family_ordinal_no_spill_gate_threshold),
                gate_temperature=float(backbone_family_ordinal_no_spill_gate_temperature),
                dropout=float(backbone_family_ordinal_no_spill_dropout),
                families=backbone_family_ordinal_no_spill_families,
            )
            if bool(use_backbone_family_ordinal_no_spill_adapter)
            else None
        )
        self.pareto_edge_expert = (
            ParetoEdgeExpertLogitCorrector(
                self.spec.class_to_idx,
                pareto_edge_expert_rules,
                hidden_dim=int(pareto_edge_expert_hidden_dim),
                scale=float(pareto_edge_expert_scale),
                gate_temperature=float(pareto_edge_expert_gate_temperature),
                gate_floor=float(pareto_edge_expert_gate_floor),
                learned_gate_bias=float(pareto_edge_expert_learned_gate_bias),
                dropout=float(pareto_edge_expert_dropout),
            )
            if bool(use_pareto_edge_expert)
            else None
        )

    @staticmethod
    def _hardpair_pair_key(pair: Any) -> str:
        return f"p{int(pair.left)}_{int(pair.right)}"

    def _hardpair_pair_scale_value(self, pair: Any) -> float:
        left_name = canonical_class_label(self.idx_to_class[int(pair.left)])
        right_name = canonical_class_label(self.idx_to_class[int(pair.right)])
        return float(self.hardpair_pair_scales.get(frozenset((left_name, right_name)), 1.0))

    def _hardpair_value_pair_scale_value(self, pair: Any) -> float:
        left_name = canonical_class_label(self.idx_to_class[int(pair.left)])
        right_name = canonical_class_label(self.idx_to_class[int(pair.right)])
        return float(self.hardpair_value_adapter_pair_scales.get(frozenset((left_name, right_name)), 1.0))

    def _hardpair_value_rough_tail_guard(
        self,
        pair: Any,
        value_delta: torch.Tensor,
        pair_value_evidence: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        """Suppress value residuals that move rough dry-concrete samples toward smoother classes."""

        if not self.use_hardpair_value_rough_tail_guard or pair_value_evidence is None:
            return None
        left_name = canonical_class_label(self.idx_to_class[int(pair.left)])
        right_name = canonical_class_label(self.idx_to_class[int(pair.right)])
        pair_names = frozenset((left_name, right_name))
        if self.hardpair_value_rough_tail_guard_pairs and pair_names not in self.hardpair_value_rough_tail_guard_pairs:
            return None
        left_factor = self.spec.class_to_factor[int(pair.left)].tolist()
        right_factor = self.spec.class_to_factor[int(pair.right)].tolist()
        if min(left_factor + right_factor) < 0 or str(pair.axis) != "roughness":
            return None
        friction_labels = FACTOR_LABELS["friction"]
        material_labels = FACTOR_LABELS["material"]
        roughness_labels = FACTOR_LABELS["roughness"]
        left_f, left_m, left_r = left_factor
        right_f, right_m, right_r = right_factor
        if friction_labels[int(left_f)] != "dry" or friction_labels[int(right_f)] != "dry":
            return None
        if material_labels[int(left_m)] != "concrete" or material_labels[int(right_m)] != "concrete":
            return None
        rough_rank = {"none": -1, "smooth": 0, "slight": 1, "severe": 2}
        left_rank = rough_rank.get(roughness_labels[int(left_r)], -1)
        right_rank = rough_rank.get(roughness_labels[int(right_r)], -1)
        if left_rank == right_rank or min(left_rank, right_rank) < 0:
            return None

        macro_rough = pair_value_evidence["macro_rough"].to(device=value_delta.device, dtype=value_delta.dtype).squeeze(1)
        micro_rough = pair_value_evidence["micro_rough"].to(device=value_delta.device, dtype=value_delta.dtype).squeeze(1)
        value_vector = pair_value_evidence["vector"].to(device=value_delta.device, dtype=value_delta.dtype)
        # Macro/meso tail is the strongest dry-concrete severe/slight clue in the
        # feature audit; micro/Laplacian/gradient terms catch sharp local roughness.
        rough_tail = torch.maximum(
            macro_rough,
            torch.maximum(micro_rough, 0.50 * value_vector[:, 6] + 0.30 * value_vector[:, 7] + 0.20 * value_vector[:, 9]),
        ).clamp(0.0, 1.0)
        rough_support = torch.sigmoid(
            (rough_tail - float(self.hardpair_value_rough_tail_guard_threshold))
            * float(self.hardpair_value_rough_tail_guard_temperature)
        )
        if left_rank > right_rank:
            moves_toward_smoother = value_delta < 0.0
        else:
            moves_toward_smoother = value_delta > 0.0
        guard = 1.0 - float(self.hardpair_value_rough_tail_guard_strength) * rough_support
        guard = guard.clamp(1.0 - float(self.hardpair_value_rough_tail_guard_strength), 1.0)
        return torch.where(moves_toward_smoother, guard, torch.ones_like(guard))

    def _uses_dry_concrete_pair_selector(self, pair: Any) -> bool:
        if self.dry_concrete_pair_signed_selector is None:
            return False
        left_name = canonical_class_label(self.idx_to_class[int(pair.left)])
        right_name = canonical_class_label(self.idx_to_class[int(pair.right)])
        return frozenset((left_name, right_name)) in self.dry_concrete_pair_selector_pairs

    def _dry_concrete_pair_selector_safe_gate(
        self,
        pair: Any,
        linear_logits: torch.Tensor,
    ) -> torch.Tensor | None:
        """Open the direct selector path only on validation-safe ordinal states.

        The accepted offline rule was asymmetric: move `dry_concrete_severe`
        toward `dry_concrete_slight` only when the anchor already puts slight
        in the top-2 and the severe-vs-slight margin is small. This makes the
        learnable selector a structural version of that safe transition rather
        than a free late classifier.
        """

        if self.dry_concrete_pair_selector_direct_delta_scale <= 0.0:
            return None
        left = int(pair.left)
        right = int(pair.right)
        left_name = canonical_class_label(self.idx_to_class[left])
        right_name = canonical_class_label(self.idx_to_class[right])
        severe_name = "dry_concrete_severe"
        slight_name = "dry_concrete_slight"
        if {left_name, right_name} != {severe_name, slight_name}:
            return None
        severe_idx = left if left_name == severe_name else right
        slight_idx = left if left_name == slight_name else right
        pred = linear_logits.argmax(dim=1)
        order = torch.argsort(linear_logits, dim=1, descending=True)
        slight_in_top2 = order[:, :2].eq(int(slight_idx)).any(dim=1).to(dtype=linear_logits.dtype)
        severe_pred = pred.eq(int(severe_idx)).to(dtype=linear_logits.dtype)
        margin = linear_logits[:, severe_idx] - linear_logits[:, slight_idx]
        close = torch.sigmoid(
            (float(self.dry_concrete_pair_selector_safe_margin) - margin)
            * float(self.dry_concrete_pair_selector_safe_temperature)
        )
        return (severe_pred * slight_in_top2 * close).to(dtype=linear_logits.dtype)

    @staticmethod
    def _hardpair_pair_features(
        linear_logits: torch.Tensor,
        probs: torch.Tensor,
        left: int,
        right: int,
    ) -> torch.Tensor:
        """Return pair-local anchor state features with shape [B, 5]."""

        lp = probs[:, left : left + 1]
        rp = probs[:, right : right + 1]
        pair_mass = (lp + rp).clamp(0.0, 1.0)
        prob_gap = (lp - rp).abs().clamp(0.0, 1.0)
        logit_gap = torch.tanh(0.25 * (linear_logits[:, left : left + 1] - linear_logits[:, right : right + 1]).abs())
        local = torch.cat([lp, rp], dim=1).clamp_min(1e-6)
        local_norm = local / local.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(local_norm * local_norm.log()).sum(dim=1, keepdim=True) / 0.6931471805599453
        return torch.cat([lp, rp, pair_mass, prob_gap + logit_gap, entropy.clamp(0.0, 1.0)], dim=1)

    def _hardpair_sample_protection_gate(self, probs: torch.Tensor) -> torch.Tensor:
        """Return [B] residual gate that closes on protected-class-like samples.

        This is sample-conditional protection: even if a hard-pair edge is
        globally active, the residual is suppressed when the calibrated linear
        anchor assigns enough probability mass to fragile classes such as
        `water_concrete_slight`.
        """

        idx = self.hardpair_sample_protect_idx.to(device=probs.device)
        if idx.numel() == 0:
            return probs.new_ones((probs.size(0),))
        protect_mass = probs.index_select(1, idx).amax(dim=1).clamp(0.0, 1.0)
        return torch.sigmoid(
            (float(self.hardpair_sample_protect_threshold) - protect_mass)
            * float(self.hardpair_sample_protect_temperature)
        ).to(dtype=probs.dtype)

    def _hardpair_pair_value_evidence(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return pair-boundary value evidence used to gate hard-pair experts.

        The features are not a separate classifier. They summarize the value
        families found in high-error pair diagnosis: macro/meso roughness for
        slight/severe edges, wet-film and dark-water for wet/water edges, and
        micro/Laplacian roughness for mud/gravel-like texture edges.
        """

        rgb = (image * self.evidence_stats.std + self.evidence_stats.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 128:
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
        r, g_ch, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.299 * r + 0.587 * g_ch + 0.114 * b
        value = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        saturation = ((value - minc) / value.clamp_min(1e-4)).clamp(0.0, 1.0)
        gx = F.conv2d(gray, self.evidence_stats.sobel_x, padding=1)
        gy = F.conv2d(gray, self.evidence_stats.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6).clamp(0.0, 1.0)
        lap = F.conv2d(gray, self.evidence_stats.laplace, padding=1).abs().clamp(0.0, 1.0)
        blur3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        blur9 = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        blur21 = F.avg_pool2d(gray, kernel_size=21, stride=1, padding=10)
        contrast = (gray - blur9).abs().clamp(0.0, 1.0)
        micro = (gray - blur3).abs().clamp(0.0, 1.0)
        meso = (blur3 - blur9).abs().clamp(0.0, 1.0)
        macro = (blur9 - blur21).abs().clamp(0.0, 1.0)
        specular = torch.sigmoid((value - 0.80) * 14.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        dark_water = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.34 - saturation) * 10.0)
            * torch.sigmoid((0.055 - grad) * 30.0)
        )
        wet = torch.clamp(specular + 0.65 * dark_water, 0.0, 1.0)
        low_texture = torch.sigmoid((0.050 - grad) * 32.0)
        low_contrast = torch.sigmoid((0.035 - contrast) * 40.0)
        texture_erasure = low_texture * low_contrast
        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.30 - saturation) * 12.0)
        marking_like = torch.sigmoid((value - 0.76) * 15.0) * torch.sigmoid((grad - 0.08) * 18.0)

        def mean(x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(2, 3))

        def std(x: torch.Tensor) -> torch.Tensor:
            return x.std(dim=(2, 3), unbiased=False)

        def top_mean(x: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
            flat = x.flatten(1)
            k = max(1, int(flat.size(1) * float(fraction)))
            return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

        macro_rough = (18.0 * std(macro) + 12.0 * std(meso) + 10.0 * std(lap) + 6.0 * std(grad)).clamp(0.0, 1.0)
        micro_rough = (16.0 * std(micro) + 12.0 * std(lap) + 7.0 * std(grad)).clamp(0.0, 1.0)
        film = (1.2 * mean(wet) + 1.4 * mean(dark_water) + 0.9 * mean(texture_erasure) + mean(specular)).clamp(0.0, 1.0)
        artifact = torch.maximum(mean(snow_like), mean(marking_like)).clamp(0.0, 1.0)
        gx_energy = gx.abs().mean(dim=(2, 3))
        gy_energy = gy.abs().mean(dim=(2, 3))
        grad_anisotropy = ((gx_energy - gy_energy).abs() / (gx_energy + gy_energy + 1e-4)).clamp(0.0, 1.0)
        value_vector = torch.cat(
            [
                macro_rough,
                micro_rough,
                film,
                artifact,
                mean(saturation).clamp(0.0, 1.0),
                (10.0 * mean(macro)).clamp(0.0, 1.0),
                (10.0 * std(macro)).clamp(0.0, 1.0),
                (10.0 * std(meso)).clamp(0.0, 1.0),
                (10.0 * std(micro)).clamp(0.0, 1.0),
                (5.0 * std(lap)).clamp(0.0, 1.0),
                (5.0 * std(grad)).clamp(0.0, 1.0),
                grad_anisotropy,
                mean(dark_water).clamp(0.0, 1.0),
                top_mean(dark_water).clamp(0.0, 1.0),
                mean(specular).clamp(0.0, 1.0),
                top_mean(specular).clamp(0.0, 1.0),
                mean(texture_erasure).clamp(0.0, 1.0),
                top_mean(texture_erasure).clamp(0.0, 1.0),
                mean(value).clamp(0.0, 1.0),
                std(value).clamp(0.0, 1.0),
            ],
            dim=1,
        )
        return {
            "macro_rough": macro_rough,
            "micro_rough": micro_rough,
            "film": film,
            "artifact": artifact,
            "saturation": mean(saturation).clamp(0.0, 1.0),
            "vector": value_vector,
        }

    def _hardpair_physics_gate_value(
        self,
        pair: Any,
        evidence_stats: torch.Tensor | None,
        pair_value_evidence: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Gate wet/water-concrete roughness corrections by visible rough evidence."""

        if self.hardpair_physics_gate in {"", "none", "off", "false"} or evidence_stats is None:
            return None
        if self.hardpair_physics_gate == "pair_value_evidence":
            if pair_value_evidence is None:
                return None
            if str(pair.axis) not in {"roughness", "friction"}:
                return None
            left_factor = self.spec.class_to_factor[int(pair.left)].tolist()
            right_factor = self.spec.class_to_factor[int(pair.right)].tolist()
            if min(left_factor + right_factor) < 0:
                return None
            friction_labels = FACTOR_LABELS["friction"]
            material_labels = FACTOR_LABELS["material"]
            left_f, left_m, _ = left_factor
            right_f, right_m, _ = right_factor
            friction_names = {friction_labels[int(left_f)], friction_labels[int(right_f)]}
            material_names = {material_labels[int(left_m)], material_labels[int(right_m)]}
            macro_rough = pair_value_evidence["macro_rough"].float()
            micro_rough = pair_value_evidence["micro_rough"].float()
            film = pair_value_evidence["film"].float()
            artifact_guard = (1.0 - 0.55 * pair_value_evidence["artifact"].float()).clamp(0.15, 1.0)
            if str(pair.axis) == "roughness":
                rough_score = macro_rough if material_names <= {"concrete"} else torch.maximum(macro_rough, micro_rough)
                if friction_names & {"wet", "water"}:
                    score = 0.58 * rough_score + 0.42 * film
                    threshold = 0.50
                    temperature = 7.0
                elif friction_names & {"mud", "gravel"}:
                    score = 0.70 * micro_rough + 0.30 * rough_score
                    threshold = 0.48
                    temperature = 7.5
                else:
                    score = rough_score
                    threshold = 0.54
                    temperature = 7.5
            else:
                score = film
                threshold = 0.46
                temperature = 8.0
            gate = torch.sigmoid((score - threshold) * temperature) * artifact_guard
            gate = gate.pow(float(self.hardpair_physics_gate_power)).clamp(0.0, 1.0)
            floor = float(self.hardpair_physics_gate_floor)
            if floor > 0.0:
                gate = floor + (1.0 - floor) * gate
            return gate.to(device=evidence_stats.device, dtype=evidence_stats.dtype)
        if self.hardpair_physics_gate != "wet_concrete_rough":
            raise ValueError(f"unknown hardpair_physics_gate: {self.hardpair_physics_gate}")
        if str(pair.axis) != "roughness":
            return None
        left_factor = self.spec.class_to_factor[int(pair.left)].tolist()
        right_factor = self.spec.class_to_factor[int(pair.right)].tolist()
        friction_labels = FACTOR_LABELS["friction"]
        material_labels = FACTOR_LABELS["material"]
        roughness_labels = FACTOR_LABELS["roughness"]
        left_f, left_m, left_r = left_factor
        right_f, right_m, right_r = right_factor
        if min(left_factor + right_factor) < 0:
            return None
        same_state = int(left_f) == int(right_f) and int(left_m) == int(right_m)
        friction_name = friction_labels[int(left_f)]
        material_name = material_labels[int(left_m)]
        rough_pair = {roughness_labels[int(left_r)], roughness_labels[int(right_r)]}
        target = (
            same_state
            and friction_name in {"wet", "water"}
            and material_name == "concrete"
            and rough_pair <= {"smooth", "slight", "severe"}
        )
        if not target:
            return None

        stats = evidence_stats.float()
        rough = stats[:, 11:12].clamp(0.0, 1.0)
        erasure = stats[:, 12:13].clamp(0.0, 1.0)
        snow_ice = torch.maximum(stats[:, 13:14], stats[:, 14:15]).clamp(0.0, 1.0)
        # This gate protects smooth wet/water concrete: film evidence alone is
        # not enough; the pair correction opens only when rough texture is visible.
        rough_gate = torch.sigmoid((rough - 0.020) * 160.0)
        erasure_guard = torch.sigmoid((0.58 - erasure) * 10.0)
        winter_guard = torch.sigmoid((0.30 - snow_ice) * 20.0)
        gate = (rough_gate * erasure_guard * winter_guard).clamp(0.0, 1.0)
        gate = gate.pow(float(self.hardpair_physics_gate_power))
        floor = float(self.hardpair_physics_gate_floor)
        if floor > 0.0:
            gate = floor + (1.0 - floor) * gate
        return gate.to(dtype=evidence_stats.dtype)

    def _hardpair_gated_logits(
        self,
        linear_logits: torch.Tensor,
        coupled_logits: torch.Tensor,
        boundary_logits: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        probs = F.softmax(linear_logits, dim=1)
        sample_gate = self._hardpair_sample_protection_gate(probs)
        residual = torch.zeros_like(linear_logits)
        for pair, active in zip(self.spec.hard_pairs, self.hardpair_active, strict=True):
            if not active:
                continue
            left = int(pair.left)
            right = int(pair.right)
            base_gap = (linear_logits[:, left] - linear_logits[:, right]).abs()
            pair_mass = (probs[:, left] + probs[:, right]).clamp(0.0, 1.0)
            gate = torch.sigmoid((self.hardpair_gate_margin - base_gap) * self.hardpair_gate_temperature) * pair_mass * sample_gate
            raw = coupled_logits[:, left] - coupled_logits[:, right]
            expert = boundary_logits.get(pair.boundary)
            if expert is not None:
                raw = raw + expert
            pair_scale = self._hardpair_pair_scale_value(pair)
            delta = pair_scale * self.hardpair_correction_scale * gate * torch.tanh(raw)
            residual[:, left] = residual[:, left] + delta
            residual[:, right] = residual[:, right] - delta
        return linear_logits + residual

    def _hardpair_pairwise_calibrated_logits(
        self,
        linear_logits: torch.Tensor,
        judge_input: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(linear_logits, dim=1)
        sample_gate = self._hardpair_sample_protection_gate(probs)
        residual = torch.zeros_like(linear_logits)
        for pair, active in zip(self.spec.hard_pairs, self.hardpair_active, strict=True):
            if not active:
                continue
            key = self._hardpair_pair_key(pair)
            if key not in self.pairwise_hardpair_experts:
                continue
            expert = self.pairwise_hardpair_experts[key]
            left = int(pair.left)
            right = int(pair.right)
            base_gap = (linear_logits[:, left] - linear_logits[:, right]).abs()
            pair_mass = (probs[:, left] + probs[:, right]).clamp(0.0, 1.0)
            gate = torch.sigmoid((self.hardpair_gate_margin - base_gap) * self.hardpair_gate_temperature) * pair_mass * sample_gate
            raw = expert(judge_input).squeeze(1)
            pair_scale = self._hardpair_pair_scale_value(pair)
            delta = pair_scale * self.hardpair_correction_scale * gate * torch.tanh(raw)
            residual[:, left] = residual[:, left] + delta
            residual[:, right] = residual[:, right] - delta
        return linear_logits + residual

    def _hardpair_error_gated_calibrated_logits(
        self,
        linear_logits: torch.Tensor,
        judge_input: torch.Tensor,
        evidence_stats: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Apply pair-specific corrections only when a learned anchor-error gate opens."""

        probs = F.softmax(linear_logits, dim=1)
        sample_gate = self._hardpair_sample_protection_gate(probs)
        residual = torch.zeros_like(linear_logits)
        gate_logits: dict[str, torch.Tensor] = {}
        gate_values: dict[str, torch.Tensor] = {}
        value_adapter_logits: dict[str, torch.Tensor] = {}
        value_adapter_delta: dict[str, torch.Tensor] = {}
        value_adapter_gate: dict[str, torch.Tensor] = {}
        pair_value_evidence = (
            self._hardpair_pair_value_evidence(image)
            if image is not None
            and (self.hardpair_physics_gate == "pair_value_evidence" or bool(self.pairwise_hardpair_value_adapters))
            else None
        )
        for pair, active in zip(self.spec.hard_pairs, self.hardpair_active, strict=True):
            if not active:
                continue
            key = self._hardpair_pair_key(pair)
            if key not in self.pairwise_hardpair_experts or key not in self.pairwise_hardpair_error_gates:
                continue
            expert = self.pairwise_hardpair_experts[key]
            error_gate = self.pairwise_hardpair_error_gates[key]
            left = int(pair.left)
            right = int(pair.right)
            base_gap = (linear_logits[:, left] - linear_logits[:, right]).abs()
            pair_mass = (probs[:, left] + probs[:, right]).clamp(0.0, 1.0)
            boundary_gate = torch.sigmoid((self.hardpair_gate_margin - base_gap) * self.hardpair_gate_temperature) * pair_mass * sample_gate
            pair_features = self._hardpair_pair_features(linear_logits, probs, left, right).to(dtype=judge_input.dtype)
            error_logit = error_gate(torch.cat([judge_input, pair_features], dim=1)).squeeze(1)
            learned_gate = torch.sigmoid(error_logit)
            if self.hardpair_error_gate_floor > 0.0:
                learned_gate = torch.clamp(learned_gate, min=float(self.hardpair_error_gate_floor), max=1.0)
            physics_gate = self._hardpair_physics_gate_value(pair, evidence_stats, pair_value_evidence)
            if physics_gate is not None:
                physics_gate = physics_gate.to(device=learned_gate.device, dtype=learned_gate.dtype).squeeze(1)
                learned_gate = learned_gate * physics_gate
            raw = expert(judge_input).squeeze(1)
            selector_gain = raw.new_ones(raw.shape)
            if image is not None and self._uses_dry_concrete_pair_selector(pair):
                raw, selector_gain, selector_shift = self.dry_concrete_pair_signed_selector(
                    image,
                    pair_features,
                    raw,
                )
                gate_values[f"dry_selector_gain/{key}"] = selector_gain
                gate_values[f"dry_selector_shift/{key}"] = selector_shift
                selector_safe_gate = self._dry_concrete_pair_selector_safe_gate(pair, linear_logits)
                if selector_safe_gate is not None:
                    selector_safe_gate = selector_safe_gate.to(device=raw.device, dtype=raw.dtype)
                    gate_values[f"dry_selector_safe_gate/{key}"] = selector_safe_gate
                else:
                    selector_safe_gate = raw.new_zeros(raw.shape)
            else:
                selector_shift = raw.new_zeros(raw.shape)
                selector_safe_gate = raw.new_zeros(raw.shape)
            pair_scale = self._hardpair_pair_scale_value(pair)
            delta = (
                pair_scale
                * self.hardpair_correction_scale
                * selector_gain
                * boundary_gate
                * learned_gate
                * torch.tanh(raw)
            )
            if self.dry_concrete_pair_selector_direct_delta_scale > 0.0:
                selector_delta = (
                    pair_scale
                    * float(self.dry_concrete_pair_selector_direct_delta_scale)
                    * selector_safe_gate
                    * selector_shift.to(dtype=raw.dtype)
                )
                delta = delta + selector_delta
                if bool(torch.is_tensor(selector_delta)):
                    gate_values[f"dry_selector_delta/{key}"] = selector_delta
            residual[:, left] = residual[:, left] + delta
            residual[:, right] = residual[:, right] - delta
            if pair_value_evidence is not None and key in self.pairwise_hardpair_value_adapters:
                value_vector = pair_value_evidence["vector"].to(device=judge_input.device, dtype=judge_input.dtype)
                if self.training and self.hardpair_value_adapter_value_aug_std > 0.0:
                    value_vector = (
                        value_vector
                        + torch.randn_like(value_vector) * float(self.hardpair_value_adapter_value_aug_std)
                    ).clamp(0.0, 1.0)
                if evidence_stats is None:
                    evidence_input = value_vector.new_zeros((value_vector.size(0), self.evidence_stats.out_dim))
                else:
                    evidence_input = evidence_stats.to(device=judge_input.device, dtype=judge_input.dtype)
                signed_gap = torch.tanh(0.25 * (linear_logits[:, left : left + 1] - linear_logits[:, right : right + 1]))
                value_input = torch.cat([value_vector, evidence_input, pair_features, signed_gap], dim=1)
                value_raw = self.pairwise_hardpair_value_adapters[key](value_input)
                value_signed = torch.tanh(value_raw[:, 0])
                value_gate = torch.sigmoid(value_raw[:, 1])
                if self.hardpair_value_adapter_gate_floor > 0.0:
                    value_gate = self.hardpair_value_adapter_gate_floor + (1.0 - self.hardpair_value_adapter_gate_floor) * value_gate
                value_pair_scale = self._hardpair_value_pair_scale_value(pair)
                value_delta = (
                    value_pair_scale
                    * float(self.hardpair_value_adapter_scale)
                    * boundary_gate
                    * value_gate
                    * value_signed
                )
                rough_tail_guard = self._hardpair_value_rough_tail_guard(pair, value_delta, pair_value_evidence)
                if rough_tail_guard is not None:
                    value_delta = value_delta * rough_tail_guard
                    gate_values[f"value_rough_tail_guard/{key}"] = rough_tail_guard
                residual[:, left] = residual[:, left] + value_delta
                residual[:, right] = residual[:, right] - value_delta
                gate_values[f"value_adapter/{key}"] = value_gate
                value_adapter_logits[key] = value_raw[:, 0]
                value_adapter_delta[key] = value_delta
                value_adapter_gate[key] = value_gate if rough_tail_guard is None else value_gate * rough_tail_guard
            gate_logits[key] = error_logit
            gate_values[key] = learned_gate
        return (
            linear_logits + residual,
            gate_logits,
            gate_values,
            value_adapter_logits,
            value_adapter_delta,
            value_adapter_gate,
        )

    def _hardpair_margin_directed_logits(
        self,
        linear_logits: torch.Tensor,
        judge_input: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Apply pair-local signed margin corrections on calibrated hard boundaries."""

        probs = F.softmax(linear_logits, dim=1)
        sample_gate = self._hardpair_sample_protection_gate(probs)
        residual = torch.zeros_like(linear_logits)
        margin_raw: dict[str, torch.Tensor] = {}
        margin_delta: dict[str, torch.Tensor] = {}
        margin_gate: dict[str, torch.Tensor] = {}
        selector_shift: dict[str, torch.Tensor] = {}
        for pair, active in zip(self.spec.hard_pairs, self.hardpair_active, strict=True):
            if not active:
                continue
            key = self._hardpair_pair_key(pair)
            if key not in self.pairwise_hardpair_margin_heads:
                continue
            margin_head = self.pairwise_hardpair_margin_heads[key]
            left = int(pair.left)
            right = int(pair.right)
            base_gap = (linear_logits[:, left] - linear_logits[:, right]).abs()
            pair_mass = (probs[:, left] + probs[:, right]).clamp(0.0, 1.0)
            gate = torch.sigmoid((self.hardpair_gate_margin - base_gap) * self.hardpair_gate_temperature) * pair_mass * sample_gate
            pair_features = self._hardpair_pair_features(linear_logits, probs, left, right).to(dtype=judge_input.dtype)
            raw = margin_head(torch.cat([judge_input, pair_features], dim=1)).squeeze(1)
            selector_gain = raw.new_ones(raw.shape)
            if image is not None and self._uses_dry_concrete_pair_selector(pair):
                raw, selector_gain, shift = self.dry_concrete_pair_signed_selector(image, pair_features, raw)
                selector_shift[key] = shift
            pair_scale = self._hardpair_pair_scale_value(pair)
            delta = pair_scale * self.hardpair_margin_scale * selector_gain * gate * torch.tanh(raw)
            residual[:, left] = residual[:, left] + delta
            residual[:, right] = residual[:, right] - delta
            margin_raw[key] = raw
            margin_delta[key] = delta
            margin_gate[key] = gate * selector_gain
        return linear_logits + residual, margin_raw, margin_delta, margin_gate, selector_shift

    def _hardpair_benefit_gated_margin_logits(
        self,
        linear_logits: torch.Tensor,
        judge_input: torch.Tensor,
        evidence_stats: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Apply signed pair-margin updates only when an anchor-error gate opens."""

        probs = F.softmax(linear_logits, dim=1)
        sample_gate = self._hardpair_sample_protection_gate(probs)
        residual = torch.zeros_like(linear_logits)
        gate_logits: dict[str, torch.Tensor] = {}
        gate_values: dict[str, torch.Tensor] = {}
        margin_raw: dict[str, torch.Tensor] = {}
        margin_delta: dict[str, torch.Tensor] = {}
        margin_gate: dict[str, torch.Tensor] = {}
        pair_value_evidence = (
            self._hardpair_pair_value_evidence(image)
            if image is not None and self.hardpair_physics_gate == "pair_value_evidence"
            else None
        )
        for pair, active in zip(self.spec.hard_pairs, self.hardpair_active, strict=True):
            if not active:
                continue
            key = self._hardpair_pair_key(pair)
            if key not in self.pairwise_hardpair_margin_heads or key not in self.pairwise_hardpair_error_gates:
                continue
            margin_head = self.pairwise_hardpair_margin_heads[key]
            error_gate = self.pairwise_hardpair_error_gates[key]
            left = int(pair.left)
            right = int(pair.right)
            base_gap = (linear_logits[:, left] - linear_logits[:, right]).abs()
            pair_mass = (probs[:, left] + probs[:, right]).clamp(0.0, 1.0)
            boundary_gate = torch.sigmoid((self.hardpair_gate_margin - base_gap) * self.hardpair_gate_temperature) * pair_mass * sample_gate
            pair_features = self._hardpair_pair_features(linear_logits, probs, left, right).to(dtype=judge_input.dtype)
            pair_input = torch.cat([judge_input, pair_features], dim=1)
            raw = margin_head(pair_input).squeeze(1)
            error_logit = error_gate(pair_input).squeeze(1)
            learned_gate = torch.sigmoid(error_logit)
            if self.hardpair_error_gate_floor > 0.0:
                learned_gate = torch.clamp(learned_gate, min=float(self.hardpair_error_gate_floor), max=1.0)
            physics_gate = self._hardpair_physics_gate_value(pair, evidence_stats, pair_value_evidence)
            if physics_gate is not None:
                physics_gate = physics_gate.to(device=learned_gate.device, dtype=learned_gate.dtype).squeeze(1)
                learned_gate = learned_gate * physics_gate
            benefit_gate = boundary_gate * learned_gate
            pair_scale = self._hardpair_pair_scale_value(pair)
            delta = pair_scale * self.hardpair_margin_scale * benefit_gate * torch.tanh(raw)
            residual[:, left] = residual[:, left] + delta
            residual[:, right] = residual[:, right] - delta
            gate_logits[key] = error_logit
            gate_values[key] = learned_gate
            margin_raw[key] = raw
            margin_delta[key] = delta
            margin_gate[key] = benefit_gate
        return linear_logits + residual, gate_logits, gate_values, margin_raw, margin_delta, margin_gate

    def forward(self, image: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        backbone_image = self.input_stem_adapter(image) if self.input_stem_adapter is not None else image
        scale_space_roughness_stem_stats = None
        pair_value_stem_aux: dict[str, torch.Tensor] | None = None
        wet_water_concrete_film_depth_stem_aux: dict[str, torch.Tensor] | None = None
        water_concrete_topology_texture_stem_aux: dict[str, torch.Tensor] | None = None
        if self.scale_space_roughness_stem_adapter is not None:
            if return_aux:
                backbone_image, scale_space_roughness_stem_stats = self.scale_space_roughness_stem_adapter(
                    backbone_image,
                    return_stats=True,
                )
            else:
                backbone_image = self.scale_space_roughness_stem_adapter(backbone_image)
        if self.pair_value_stem_conditioner is not None:
            pair_value_stem_evidence = self._hardpair_pair_value_evidence(image)
            backbone_image, pair_value_stem_aux = self.pair_value_stem_conditioner(
                backbone_image,
                pair_value_stem_evidence["vector"],
            )
        if self.wet_water_concrete_film_depth_stem_conditioner is not None:
            backbone_image, wet_water_concrete_film_depth_stem_aux = (
                self.wet_water_concrete_film_depth_stem_conditioner(
                    backbone_image,
                    reference_image=image,
                )
            )
        if self.water_concrete_topology_texture_stem_conditioner is not None:
            backbone_image, water_concrete_topology_texture_stem_aux = (
                self.water_concrete_topology_texture_stem_conditioner(
                    backbone_image,
                    reference_image=image,
                )
            )
        backbone_feature = self.backbone(backbone_image)
        parts = [backbone_feature]
        local_physics_feature = None
        if self.local_physics_field_branch is not None and self.local_physics_field_adapter is not None:
            local_physics_feature = self.local_physics_field_branch(image)
            backbone_feature = backbone_feature + self.local_physics_field_scale * self.local_physics_field_adapter(
                local_physics_feature
            )
            parts[0] = backbone_feature
        physics_feature = None
        if self.physics_branch is not None:
            physics_feature = self.physics_branch(image)
            parts.append(physics_feature)
        semantic_feature = None
        if self.semantic_physics_attention_branch is not None:
            semantic_feature = self.semantic_physics_attention_branch(image)
            parts.append(semantic_feature)
        feature = self.dropout(self.norm(torch.cat(parts, dim=1)))
        pair_value_mechanism_prepared: dict[str, torch.Tensor] | None = None
        pair_value_mechanism_aux: dict[str, torch.Tensor] | None = None
        coupled_form_expert_prepared: dict[str, torch.Tensor] | None = None
        coupled_form_expert_aux: dict[str, torch.Tensor] | None = None
        water_concrete_opponent_aux: dict[str, dict[str, torch.Tensor]] | None = None
        if self.pair_value_mechanism_conditioner is not None:
            pair_value_evidence = self._hardpair_pair_value_evidence(image)
            pair_value_mechanism_prepared = self.pair_value_mechanism_conditioner.prepare(
                pair_value_evidence["vector"].to(device=feature.device, dtype=feature.dtype)
            )
            if self.pair_value_mechanism_protect_idx.numel() > 0:
                protect_idx = self.pair_value_mechanism_protect_idx.to(device=feature.device)
                base_probs = F.softmax(self.linear_head(feature), dim=1)
                protect_mass = base_probs.index_select(1, protect_idx).max(dim=1, keepdim=True).values
                protect_gate = torch.sigmoid(
                    (float(self.pair_value_mechanism_protect_threshold) - protect_mass)
                    * float(self.pair_value_mechanism_protect_temperature)
                ).to(dtype=feature.dtype)
                pair_value_mechanism_prepared["gate"] = pair_value_mechanism_prepared["gate"].to(
                    device=feature.device,
                    dtype=feature.dtype,
                ) * protect_gate
            else:
                protect_gate = feature.new_ones((feature.size(0), 1))
            feature, feature_delta = self.pair_value_mechanism_conditioner.condition_feature(
                feature,
                pair_value_mechanism_prepared,
            )
            pair_value_mechanism_aux = {
                "gates": pair_value_mechanism_prepared["gates"],
                "gate": pair_value_mechanism_prepared["gate"],
                "protect_gate": protect_gate,
                "feature_delta": feature_delta,
            }
        if self.coupled_form_expert_conditioner is not None:
            coupled_value_evidence = self._hardpair_pair_value_evidence(image)
            coupled_form_expert_prepared = self.coupled_form_expert_conditioner.prepare(
                coupled_value_evidence["vector"].to(device=feature.device, dtype=feature.dtype)
            )
            if self.coupled_form_expert_protect_idx.numel() > 0:
                protect_idx = self.coupled_form_expert_protect_idx.to(device=feature.device)
                base_probs = F.softmax(self.linear_head(feature), dim=1)
                protect_mass = base_probs.index_select(1, protect_idx).max(dim=1, keepdim=True).values
                coupled_protect_gate = torch.sigmoid(
                    (float(self.coupled_form_expert_protect_threshold) - protect_mass)
                    * float(self.coupled_form_expert_protect_temperature)
                ).to(dtype=feature.dtype)
                coupled_form_expert_prepared["gates"] = coupled_form_expert_prepared["gates"].to(
                    device=feature.device,
                    dtype=feature.dtype,
                ) * coupled_protect_gate
                coupled_form_expert_prepared["gate"] = coupled_form_expert_prepared["gate"].to(
                    device=feature.device,
                    dtype=feature.dtype,
                ) * coupled_protect_gate
            else:
                coupled_protect_gate = feature.new_ones((feature.size(0), 1))
            feature, coupled_feature_delta = self.coupled_form_expert_conditioner.condition_feature(
                feature,
                coupled_form_expert_prepared,
            )
            coupled_form_expert_aux = {
                "hand_gates": coupled_form_expert_prepared["hand_gates"],
                "learned_gates": coupled_form_expert_prepared["learned_gates"],
                "gates": coupled_form_expert_prepared["gates"],
                "gate": coupled_form_expert_prepared["gate"],
                "protect_gate": coupled_protect_gate,
                "feature_delta": coupled_feature_delta,
            }
        if self.water_concrete_opponent_feature_conditioner is not None:
            opponent_value_evidence = self._hardpair_pair_value_evidence(image)
            opponent_base_logits = self.linear_head(feature)
            (
                feature,
                opponent_raw,
                opponent_delta,
                opponent_gate,
            ) = self.water_concrete_opponent_feature_conditioner(
                feature,
                opponent_base_logits,
                opponent_value_evidence["vector"],
                self.spec,
            )
            water_concrete_opponent_aux = {
                "raw": opponent_raw,
                "delta": opponent_delta,
                "gate": opponent_gate,
            }
        evidence = self.evidence_stats(image).to(dtype=feature.dtype)
        water_film_roughness_feature_film_aux: dict[str, torch.Tensor] | None = None
        if self.water_film_roughness_feature_film is not None:
            feature, water_film_roughness_feature_film_aux = self.water_film_roughness_feature_film(
                image,
                feature,
                evidence,
            )
        local_global_scale_token_prepared: dict[str, torch.Tensor] | None = None
        local_global_scale_token_aux: dict[str, torch.Tensor] | None = None
        if self.local_global_scale_token_conditioner is not None:
            local_global_scale_token_prepared = self.local_global_scale_token_conditioner.prepare(
                image,
                evidence,
                local_physics_feature,
                device=feature.device,
                dtype=feature.dtype,
            )
            feature, local_global_scale_feature_delta = self.local_global_scale_token_conditioner.condition_feature(
                feature,
                local_global_scale_token_prepared,
            )
            local_global_scale_token_aux = {
                "scale_stats": local_global_scale_token_prepared["scale_stats"],
                "mechanism": local_global_scale_token_prepared["mechanism"],
                "gate": local_global_scale_token_prepared["gate"],
                "learned_gate": local_global_scale_token_prepared["learned_gate"],
                "hand_gate": local_global_scale_token_prepared["hand_gate"],
                "feature_delta": local_global_scale_feature_delta,
            }
        source_reliable_boundary_router_aux: dict[str, Any] | None = None
        if self.source_reliable_boundary_router is not None:
            source_base_logits = self.linear_head(feature)
            feature, source_reliable_boundary_router_aux = self.source_reliable_boundary_router(
                feature,
                evidence,
                source_base_logits,
                self.linear_head.weight,
            )
        spatial_map = getattr(self.backbone, "last_feature_map", None)
        spatial_evidence = self.spatial_evidence_maps(image) if self.spatial_evidence_maps is not None else None
        tokens = self.decoder(
            feature,
            evidence,
            spatial_map=spatial_map if isinstance(spatial_map, torch.Tensor) else None,
            spatial_evidence=spatial_evidence,
        )
        scale_space_roughness_token_aux = None
        if self.scale_space_roughness_token_conditioner is not None:
            tokens, scale_space_roughness_token_aux = self.scale_space_roughness_token_conditioner(image, tokens)
        if (
            self.local_global_scale_token_conditioner is not None
            and local_global_scale_token_prepared is not None
            and local_global_scale_token_aux is not None
        ):
            tokens, local_global_scale_token_delta_aux = self.local_global_scale_token_conditioner.condition_tokens(
                tokens,
                local_global_scale_token_prepared,
            )
            local_global_scale_token_aux.update(local_global_scale_token_delta_aux)
        if self.pair_value_mechanism_conditioner is not None and pair_value_mechanism_prepared is not None:
            tokens, pair_value_token_aux = self.pair_value_mechanism_conditioner.condition_tokens(
                tokens,
                pair_value_mechanism_prepared,
            )
            if pair_value_mechanism_aux is not None:
                pair_value_mechanism_aux.update(pair_value_token_aux)
        if self.coupled_form_expert_conditioner is not None and coupled_form_expert_prepared is not None:
            tokens, coupled_form_token_aux = self.coupled_form_expert_conditioner.condition_tokens(
                tokens,
                coupled_form_expert_prepared,
            )
            if coupled_form_expert_aux is not None:
                coupled_form_expert_aux.update(coupled_form_token_aux)
        z_r, rho = self.roughness_reliability(
            tokens["friction"],
            tokens["material"],
            tokens["roughness"],
            evidence,
            tokens["coupling"],
        )
        pseudo_roughness_reliability_aux: dict[str, torch.Tensor] | None = None
        if self.pseudo_roughness_aware_reliability is not None:
            reliability_value_evidence = self._hardpair_pair_value_evidence(image)
            z_r, rho, pseudo_roughness_reliability_aux = self.pseudo_roughness_aware_reliability(
                tokens["friction"],
                tokens["material"],
                tokens["roughness"],
                z_r,
                rho,
                evidence,
                reliability_value_evidence["vector"],
            )
        head_out = self.coupled_head(tokens["friction"], tokens["material"], z_r, tokens["coupling"])
        linear_logits = self.linear_head(feature)
        judge_parts = [tokens["coupling"], evidence, rho]
        if self.boundary_use_physics_feature and physics_feature is not None:
            judge_parts.append(physics_feature.to(dtype=feature.dtype))
        judge_input = torch.cat(judge_parts, dim=1)
        boundary_logits = {
            name: expert(judge_input).squeeze(1)
            for name, expert in self.boundary_experts.items()
        }
        hardpair_error_gate_logits: dict[str, torch.Tensor] = {}
        hardpair_error_gate_values: dict[str, torch.Tensor] = {}
        hardpair_margin_raw: dict[str, torch.Tensor] = {}
        hardpair_margin_delta: dict[str, torch.Tensor] = {}
        hardpair_margin_gate: dict[str, torch.Tensor] = {}
        hardpair_selector_shift: dict[str, torch.Tensor] = {}
        hardpair_value_adapter_logits: dict[str, torch.Tensor] = {}
        hardpair_value_adapter_delta: dict[str, torch.Tensor] = {}
        hardpair_value_adapter_gate: dict[str, torch.Tensor] = {}
        feature_value_boundary_logits: dict[str, torch.Tensor] = {}
        feature_value_boundary_delta: dict[str, torch.Tensor] = {}
        feature_value_boundary_gate: dict[str, torch.Tensor] = {}
        factor_graph_edge_flow_aux: dict[str, Any] | None = None
        tristate_wet_concrete_boundary_aux: dict[str, Any] | None = None
        closed_set_factor_redistributor_aux: dict[str, Any] | None = None
        backbone_family_ordinal_no_spill_aux: dict[str, torch.Tensor] | None = None
        backbone_isolated_dry_concrete_aux: dict[str, torch.Tensor] | None = None
        pareto_edge_expert_aux: dict[str, Any] | None = None
        protected_factor_adapter_delta: torch.Tensor | None = None
        if self.head_type == "linear":
            logits = linear_logits
        elif self.head_type == "hybrid_coupled":
            logits = linear_logits + self.hybrid_coupled_scale * head_out["logits"]
        elif self.head_type == "hardpair_gated_coupled":
            logits = self._hardpair_gated_logits(linear_logits, head_out["logits"], boundary_logits)
        elif self.head_type == "hardpair_pairwise_calibrated":
            logits = self._hardpair_pairwise_calibrated_logits(linear_logits, judge_input)
        elif self.head_type == "hardpair_error_gated_calibrated":
            (
                logits,
                hardpair_error_gate_logits,
                hardpair_error_gate_values,
                hardpair_value_adapter_logits,
                hardpair_value_adapter_delta,
                hardpair_value_adapter_gate,
            ) = self._hardpair_error_gated_calibrated_logits(
                linear_logits,
                judge_input,
                evidence,
                image=image,
            )
        elif self.head_type == "hardpair_margin_directed_calibrated":
            (
                logits,
                hardpair_margin_raw,
                hardpair_margin_delta,
                hardpair_margin_gate,
                hardpair_selector_shift,
            ) = self._hardpair_margin_directed_logits(linear_logits, judge_input, image=image)
        elif self.head_type == "hardpair_benefit_gated_margin_calibrated":
            (
                logits,
                hardpair_error_gate_logits,
                hardpair_error_gate_values,
                hardpair_margin_raw,
                hardpair_margin_delta,
                hardpair_margin_gate,
            ) = self._hardpair_benefit_gated_margin_logits(linear_logits, judge_input, evidence, image=image)
        elif self.head_type == "protected_factor_graph_adapter":
            if self.protected_factor_adapter is None:
                logits = linear_logits
                protected_factor_adapter_delta = torch.zeros_like(linear_logits)
            else:
                logits, protected_factor_adapter_delta = self.protected_factor_adapter(linear_logits, judge_input)
        else:
            logits = head_out["logits"]
        if self.tristate_wet_concrete_boundary_expert is not None:
            tristate_value_evidence = self._hardpair_pair_value_evidence(image)
            logits, tristate_wet_concrete_boundary_aux = self.tristate_wet_concrete_boundary_expert(
                logits,
                evidence,
                tristate_value_evidence,
            )
        if self.feature_value_boundary_corrector is not None:
            (
                logits,
                feature_value_boundary_logits,
                feature_value_boundary_delta,
                feature_value_boundary_gate,
            ) = self.feature_value_boundary_corrector(
                image,
                logits,
                evidence,
                self.spec,
            )
        if self.factor_graph_edge_flow_corrector is not None:
            logits, factor_graph_edge_flow_aux = self.factor_graph_edge_flow_corrector(
                logits,
                judge_input,
                evidence,
            )
            hardpair_margin_raw.update(factor_graph_edge_flow_aux["raw"])
            hardpair_margin_delta.update(factor_graph_edge_flow_aux["delta"])
            hardpair_margin_gate.update(factor_graph_edge_flow_aux["gate"])
        if self.closed_set_factor_redistributor is not None:
            logits, closed_set_factor_redistributor_aux = self.closed_set_factor_redistributor(
                logits,
                judge_input,
                evidence,
            )
        if self.backbone_family_ordinal_no_spill_adapter is not None:
            stage_maps = getattr(self.backbone, "stage_feature_maps", None)
            boundary_evidence = getattr(self.backbone, "last_boundary_evidence", None)
            boundary_stats = getattr(self.backbone, "last_boundary_stats", None)
            if (
                isinstance(stage_maps, dict)
                and isinstance(boundary_evidence, torch.Tensor)
                and isinstance(boundary_stats, torch.Tensor)
            ):
                logits, backbone_family_ordinal_no_spill_aux = self.backbone_family_ordinal_no_spill_adapter(
                    logits,
                    stage_maps,
                    boundary_evidence,
                    boundary_stats,
                )
        if self.backbone_isolated_dry_concrete_adapter is not None:
            branch_feature = getattr(self.backbone, "last_dry_concrete_isolated_feature", None)
            branch_gate = getattr(self.backbone, "last_dry_concrete_isolated_gate", None)
            if isinstance(branch_feature, torch.Tensor) and isinstance(branch_gate, torch.Tensor):
                logits, backbone_isolated_dry_concrete_aux = self.backbone_isolated_dry_concrete_adapter(
                    logits,
                    branch_feature,
                    branch_gate,
                    evidence,
                )
        logits_before_vor = logits
        dry_concrete_ordinal_chart_delta: torch.Tensor | None = None
        dry_concrete_validation_transition_aux: dict[str, torch.Tensor] | None = None
        if self.dry_concrete_roughness_vor_residual is not None:
            logits = logits + self.dry_concrete_roughness_vor_residual(image, logits)
        if self.dry_concrete_ordinal_chart_residual is not None:
            dry_concrete_ordinal_chart_delta = self.dry_concrete_ordinal_chart_residual(image, logits)
            logits = logits + dry_concrete_ordinal_chart_delta
        if self.dry_concrete_validation_transition is not None:
            logits, dry_concrete_validation_transition_aux = self.dry_concrete_validation_transition(logits)
        logits_before_pareto_edge_expert = logits
        if self.pareto_edge_expert is not None:
            pareto_value_evidence = self._hardpair_pair_value_evidence(image)
            logits, pareto_edge_expert_aux = self.pareto_edge_expert(
                logits,
                evidence,
                pareto_value_evidence,
            )
        if not return_aux:
            return logits
        aux = dict(head_out)
        aux["logits"] = logits
        aux["logits_before_vor"] = logits_before_vor
        aux["logits_before_pareto_edge_expert"] = logits_before_pareto_edge_expert
        aux["linear_logits"] = linear_logits
        aux["feature"] = feature
        aux["tokens"] = {
            "friction": tokens["friction"],
            "material": tokens["material"],
            "roughness_visible": tokens["roughness"],
            "roughness": z_r,
            "coupling": tokens["coupling"],
        }
        aux["rho_roughness"] = rho
        aux["rho_target"] = C3PhysicsEvidenceStats.roughness_reliability_target(evidence)
        aux["evidence_stats"] = evidence
        aux["boundary_logits"] = boundary_logits
        family_route_logits = getattr(self.backbone, "last_family_route_logits", None)
        family_route_probs = getattr(self.backbone, "last_family_route_probs", None)
        family_route_effective_probs = getattr(self.backbone, "last_family_route_effective_probs", None)
        if isinstance(family_route_logits, dict) and family_route_logits:
            aux["family_route_logits"] = family_route_logits
        if isinstance(family_route_probs, dict) and family_route_probs:
            aux["family_route_probs"] = family_route_probs
        if isinstance(family_route_effective_probs, dict) and family_route_effective_probs:
            aux["family_route_effective_probs"] = family_route_effective_probs
        if hardpair_error_gate_logits:
            aux["hardpair_error_gate_logits"] = hardpair_error_gate_logits
            aux["hardpair_error_gate_values"] = hardpair_error_gate_values
        if hardpair_margin_raw:
            aux["hardpair_margin_raw"] = hardpair_margin_raw
            aux["hardpair_margin_delta"] = hardpair_margin_delta
            aux["hardpair_margin_gate"] = hardpair_margin_gate
        if hardpair_value_adapter_logits:
            aux["hardpair_value_adapter_logits"] = hardpair_value_adapter_logits
            aux["hardpair_value_adapter_delta"] = hardpair_value_adapter_delta
            aux["hardpair_value_adapter_gate"] = hardpair_value_adapter_gate
        if hardpair_selector_shift:
            aux["hardpair_selector_shift"] = hardpair_selector_shift
        if feature_value_boundary_delta:
            aux["feature_value_boundary_logits"] = feature_value_boundary_logits
            aux["feature_value_boundary_delta"] = feature_value_boundary_delta
            aux["feature_value_boundary_gate"] = feature_value_boundary_gate
        if water_concrete_opponent_aux is not None:
            aux["water_concrete_opponent_feature_logits"] = water_concrete_opponent_aux["raw"]
            aux["water_concrete_opponent_feature_delta"] = water_concrete_opponent_aux["delta"]
            aux["water_concrete_opponent_feature_gate"] = water_concrete_opponent_aux["gate"]
        if factor_graph_edge_flow_aux is not None:
            aux["factor_graph_edge_flow_delta"] = factor_graph_edge_flow_aux["residual"]
            aux["factor_graph_edge_flow_raw"] = factor_graph_edge_flow_aux["raw"]
            aux["factor_graph_edge_flow_pair_delta"] = factor_graph_edge_flow_aux["delta"]
            aux["factor_graph_edge_flow_gate"] = factor_graph_edge_flow_aux["gate"]
        if tristate_wet_concrete_boundary_aux is not None:
            aux["tristate_wet_concrete_boundary_delta"] = tristate_wet_concrete_boundary_aux["residual"]
            aux["tristate_wet_concrete_boundary_raw"] = tristate_wet_concrete_boundary_aux["raw"]
            aux["tristate_wet_concrete_boundary_pair_delta"] = tristate_wet_concrete_boundary_aux["delta"]
            aux["tristate_wet_concrete_boundary_gate"] = tristate_wet_concrete_boundary_aux["gate"]
            aux["tristate_wet_concrete_boundary_hand_raw"] = tristate_wet_concrete_boundary_aux["hand_raw"]
            aux["tristate_wet_concrete_boundary_severe_protect"] = tristate_wet_concrete_boundary_aux[
                "severe_protect"
            ]
            aux["tristate_wet_concrete_boundary_states"] = tristate_wet_concrete_boundary_aux["states"]
            aux["tristate_wet_concrete_boundary_hand_gate"] = tristate_wet_concrete_boundary_aux["hand_gate"]
        if protected_factor_adapter_delta is not None:
            aux["protected_factor_adapter_delta"] = protected_factor_adapter_delta
        if closed_set_factor_redistributor_aux is not None:
            aux["closed_set_factor_redistributor_delta"] = closed_set_factor_redistributor_aux["residual"]
            aux["closed_set_factor_redistributor_gates"] = closed_set_factor_redistributor_aux["gates"]
            aux["closed_set_factor_redistributor_raw"] = closed_set_factor_redistributor_aux["raw"]
            aux["closed_set_factor_redistributor_masses"] = closed_set_factor_redistributor_aux["masses"]
            aux["closed_set_factor_redistributor_margins"] = closed_set_factor_redistributor_aux["margins"]
            aux["closed_set_factor_redistributor_graph_guards"] = closed_set_factor_redistributor_aux["graph_guards"]
        if backbone_family_ordinal_no_spill_aux is not None:
            aux["backbone_family_ordinal_no_spill_delta"] = backbone_family_ordinal_no_spill_aux["residual"]
            aux["backbone_family_ordinal_no_spill_gates"] = backbone_family_ordinal_no_spill_aux["gates"]
            aux["backbone_family_ordinal_no_spill_raw"] = backbone_family_ordinal_no_spill_aux["raw"]
        if backbone_isolated_dry_concrete_aux is not None:
            aux["backbone_isolated_dry_concrete_delta"] = backbone_isolated_dry_concrete_aux["residual"]
            aux["backbone_isolated_dry_concrete_raw"] = backbone_isolated_dry_concrete_aux["raw"]
            aux["backbone_isolated_dry_concrete_gate"] = backbone_isolated_dry_concrete_aux["gate"]
            aux["backbone_isolated_dry_concrete_dry_mass"] = backbone_isolated_dry_concrete_aux["dry_mass"]
            aux["backbone_isolated_dry_concrete_prob_gap"] = backbone_isolated_dry_concrete_aux["prob_gap"]
        if dry_concrete_ordinal_chart_delta is not None:
            aux["dry_concrete_ordinal_chart_delta"] = dry_concrete_ordinal_chart_delta
        if dry_concrete_validation_transition_aux is not None:
            aux["dry_concrete_validation_transition_delta"] = dry_concrete_validation_transition_aux["residual"]
            aux["dry_concrete_validation_transition_mask"] = dry_concrete_validation_transition_aux["mask"]
        if pareto_edge_expert_aux is not None:
            aux["pareto_edge_expert_delta"] = pareto_edge_expert_aux["residual"]
            aux["pareto_edge_expert_raw"] = pareto_edge_expert_aux["raw"]
            aux["pareto_edge_expert_pair_delta"] = pareto_edge_expert_aux["delta"]
            aux["pareto_edge_expert_gate"] = pareto_edge_expert_aux["gate"]
            aux["pareto_edge_expert_hand_gate"] = pareto_edge_expert_aux["hand_gate"]
        if source_reliable_boundary_router_aux is not None:
            aux["source_reliable_boundary_delta"] = source_reliable_boundary_router_aux["delta"]
            aux["source_reliable_boundary_gate"] = source_reliable_boundary_router_aux["gate"]
            aux["source_reliable_boundary_raw"] = source_reliable_boundary_router_aux["raw"]
            aux["source_reliable_boundary_physics_gate"] = source_reliable_boundary_router_aux["physics_gate"]
        if physics_feature is not None:
            aux["physics_feature"] = physics_feature
        if semantic_feature is not None:
            aux["semantic_physics_feature"] = semantic_feature
        if local_physics_feature is not None:
            aux["local_physics_feature"] = local_physics_feature
        if scale_space_roughness_stem_stats is not None:
            aux["scale_space_roughness_stem_stats"] = scale_space_roughness_stem_stats
        if pair_value_stem_aux is not None:
            aux["pair_value_stem_hand_gates"] = pair_value_stem_aux["hand_gates"]
            aux["pair_value_stem_learned_gates"] = pair_value_stem_aux["learned_gates"]
            aux["pair_value_stem_gates"] = pair_value_stem_aux["gates"]
            aux["pair_value_stem_gate"] = pair_value_stem_aux["gate"]
            aux["pair_value_stem_spatial_gate"] = pair_value_stem_aux["spatial_gate"]
            aux["pair_value_stem_delta"] = pair_value_stem_aux["delta"]
        if wet_water_concrete_film_depth_stem_aux is not None:
            aux["wet_water_concrete_film_depth_stem_gate"] = wet_water_concrete_film_depth_stem_aux["gate"]
            aux["wet_water_concrete_film_depth_stem_spatial_gate"] = wet_water_concrete_film_depth_stem_aux[
                "spatial_gate"
            ]
            aux["wet_water_concrete_film_depth_stem_mechanism"] = wet_water_concrete_film_depth_stem_aux[
                "mechanism"
            ]
            aux["wet_water_concrete_film_depth_stem_delta"] = wet_water_concrete_film_depth_stem_aux["delta"]
        if water_concrete_topology_texture_stem_aux is not None:
            aux["water_concrete_topology_texture_stem_gate"] = water_concrete_topology_texture_stem_aux["gate"]
            aux["water_concrete_topology_texture_stem_spatial_gate"] = water_concrete_topology_texture_stem_aux[
                "spatial_gate"
            ]
            aux["water_concrete_topology_texture_stem_mechanism"] = water_concrete_topology_texture_stem_aux[
                "mechanism"
            ]
            aux["water_concrete_topology_texture_stem_delta"] = water_concrete_topology_texture_stem_aux["delta"]
        if scale_space_roughness_token_aux is not None:
            aux["scale_space_roughness_token_stats"] = scale_space_roughness_token_aux["stats"]
            aux["scale_space_roughness_token_gate"] = scale_space_roughness_token_aux["gate"]
            aux["scale_space_roughness_token_roughness_delta"] = scale_space_roughness_token_aux["roughness_delta"]
            aux["scale_space_roughness_token_coupling_delta"] = scale_space_roughness_token_aux["coupling_delta"]
        if local_global_scale_token_aux is not None:
            aux["local_global_scale_token_scale_stats"] = local_global_scale_token_aux["scale_stats"]
            aux["local_global_scale_token_mechanism"] = local_global_scale_token_aux["mechanism"]
            aux["local_global_scale_token_gate"] = local_global_scale_token_aux["gate"]
            aux["local_global_scale_token_learned_gate"] = local_global_scale_token_aux["learned_gate"]
            aux["local_global_scale_token_hand_gate"] = local_global_scale_token_aux["hand_gate"]
            aux["local_global_scale_token_feature_delta"] = local_global_scale_token_aux["feature_delta"]
            aux["local_global_scale_token_roughness_delta"] = local_global_scale_token_aux["roughness_delta"]
            aux["local_global_scale_token_coupling_delta"] = local_global_scale_token_aux["coupling_delta"]
        if water_film_roughness_feature_film_aux is not None:
            aux["water_film_roughness_feature_film_scale_stats"] = water_film_roughness_feature_film_aux["scale_stats"]
            aux["water_film_roughness_feature_film_mechanism"] = water_film_roughness_feature_film_aux["mechanism"]
            aux["water_film_roughness_feature_film_gate"] = water_film_roughness_feature_film_aux["gate"]
            aux["water_film_roughness_feature_film_hand_gate"] = water_film_roughness_feature_film_aux["hand_gate"]
            aux["water_film_roughness_feature_film_learned_gate"] = water_film_roughness_feature_film_aux["learned_gate"]
            aux["water_film_roughness_feature_film_feature_delta"] = water_film_roughness_feature_film_aux["feature_delta"]
        if pseudo_roughness_reliability_aux is not None:
            aux["pseudo_roughness_reliability_states"] = pseudo_roughness_reliability_aux["states"]
            aux["pseudo_roughness_reliability_state_probs"] = pseudo_roughness_reliability_aux["state_probs"]
            aux["pseudo_roughness_reliability_gate"] = pseudo_roughness_reliability_aux["gate"]
            aux["pseudo_roughness_reliability_hand_gate"] = pseudo_roughness_reliability_aux["hand_gate"]
            aux["pseudo_roughness_reliability_learned_gate"] = pseudo_roughness_reliability_aux["learned_gate"]
            aux["pseudo_roughness_reliability_token_delta"] = pseudo_roughness_reliability_aux["token_delta"]
            aux["pseudo_roughness_reliability_rho_delta"] = pseudo_roughness_reliability_aux["rho_delta"]
        if pair_value_mechanism_aux is not None:
            aux["pair_value_mechanism_gates"] = pair_value_mechanism_aux["gates"]
            aux["pair_value_mechanism_gate"] = pair_value_mechanism_aux["gate"]
            aux["pair_value_mechanism_protect_gate"] = pair_value_mechanism_aux["protect_gate"]
            aux["pair_value_mechanism_feature_delta"] = pair_value_mechanism_aux["feature_delta"]
            aux["pair_value_mechanism_roughness_delta"] = pair_value_mechanism_aux["roughness_delta"]
            aux["pair_value_mechanism_coupling_delta"] = pair_value_mechanism_aux["coupling_delta"]
        if coupled_form_expert_aux is not None:
            aux["coupled_form_expert_hand_gates"] = coupled_form_expert_aux["hand_gates"]
            aux["coupled_form_expert_learned_gates"] = coupled_form_expert_aux["learned_gates"]
            aux["coupled_form_expert_gates"] = coupled_form_expert_aux["gates"]
            aux["coupled_form_expert_gate"] = coupled_form_expert_aux["gate"]
            aux["coupled_form_expert_protect_gate"] = coupled_form_expert_aux["protect_gate"]
            aux["coupled_form_expert_feature_delta"] = coupled_form_expert_aux["feature_delta"]
            aux["coupled_form_expert_roughness_delta"] = coupled_form_expert_aux["roughness_delta"]
            aux["coupled_form_expert_coupling_delta"] = coupled_form_expert_aux["coupling_delta"]
        if self.expose_hardpair_pair_value_evidence:
            value_evidence = self._hardpair_pair_value_evidence(image)
            aux["hardpair_pair_value_evidence_vector"] = value_evidence["vector"].to(
                device=logits.device,
                dtype=logits.dtype,
            )
        return aux

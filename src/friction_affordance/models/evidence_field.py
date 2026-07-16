from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from friction_affordance.ontology import RISK


class LocalFrictionEvidenceField(nn.Module):
    """Weakly supervised local friction evidence field.

    The public datasets provide image-level road-state labels rather than
    pixel-level friction labels. This branch turns low-level optical/texture
    evidence into a patch grid, learns an attention field over plausible road
    contact regions, and aggregates patch predictions with MIL-style pooling.
    """

    def __init__(
        self,
        out_dim: int = 64,
        hidden_dim: int = 48,
        patch_stride: int = 8,
        contact_prior_strength: float = 1.0,
        road_likelihood_prior_strength: float = 0.0,
        entropy_expansion: float = 0.08,
        use_region_mixture_cues: bool = False,
        region_mixture_expansion: float = 0.0,
        region_mixture_kernel_size: int = 9,
        num_queries: int = 1,
        query_disagreement_expansion: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.patch_stride = int(patch_stride)
        self.contact_prior_strength = float(contact_prior_strength)
        self.road_likelihood_prior_strength = float(road_likelihood_prior_strength)
        self.entropy_expansion = float(entropy_expansion)
        self.use_region_mixture_cues = bool(use_region_mixture_cues)
        self.region_mixture_expansion = float(region_mixture_expansion)
        self.region_mixture_kernel_size = max(int(region_mixture_kernel_size), 3)
        self.num_queries = max(int(num_queries), 1)
        self.query_disagreement_expansion = float(query_disagreement_expansion)
        if self.region_mixture_kernel_size % 2 == 0:
            self.region_mixture_kernel_size += 1
        self.region_mixture_channel_count = 5 if self.use_region_mixture_cues else 0
        self.x_coord_channel = 13 + self.region_mixture_channel_count
        self.y_coord_channel = 14 + self.region_mixture_channel_count
        self.contact_prior_channel = 15 + self.region_mixture_channel_count
        self.region_mixture_score_channel = 17 if self.use_region_mixture_cues else None
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

        self.encoder = nn.Sequential(
            nn.Conv2d(16 + self.region_mixture_channel_count, hidden_dim, kernel_size=3, padding=1, bias=False),
            _group_norm(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1, bias=False),
            _group_norm(hidden_dim),
            nn.GELU(),
        )
        self.risk_head = nn.Conv2d(hidden_dim, len(RISK), kernel_size=1)
        self.mu_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        self.attention_head = nn.Conv2d(hidden_dim, self.num_queries, kernel_size=1)
        self.query_gate = nn.Linear(hidden_dim, 1) if self.num_queries > 1 else None
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self._evidence_channels(x)
        evidence = F.avg_pool2d(
            evidence,
            kernel_size=self.patch_stride,
            stride=self.patch_stride,
            ceil_mode=False,
        )
        hidden = self.encoder(evidence)
        risk_logits_map = self.risk_head(hidden)
        mu_raw = self.mu_head(hidden)
        contact_prior = evidence[:, self.contact_prior_channel : self.contact_prior_channel + 1, :, :].clamp_min(1e-4)
        road_likelihood = self._road_likelihood(evidence).clamp_min(1e-4)
        attention_prior = contact_prior
        attention_logits = self.attention_head(hidden)
        attention_logits = attention_logits + self.contact_prior_strength * torch.log(contact_prior)
        if self.road_likelihood_prior_strength > 0:
            attention_logits = attention_logits + self.road_likelihood_prior_strength * torch.log(road_likelihood)
            attention_prior = (
                contact_prior.clamp_min(1e-4).pow(max(self.contact_prior_strength, 1e-4))
                * road_likelihood.clamp_min(1e-4).pow(self.road_likelihood_prior_strength)
            )
        attention_queries = torch.softmax(attention_logits.flatten(2), dim=2).view_as(attention_logits)

        risk_logits_queries = torch.einsum("bqhw,bchw->bqc", attention_queries, risk_logits_map)
        mu_mean_map = (1.2 * torch.sigmoid(mu_raw[:, 0:1])).clamp(1e-4, 1.2)
        # Patch-level evidence should start as a moderately uncertain weak
        # interval, not as the full physical range. The image-level set head can
        # still expand intervals when labels are ambiguous.
        mu_scale_map = (0.25 * F.softplus(mu_raw[:, 1:2])).clamp(1e-3, 0.35)
        z = 1.2815515655446004
        mu_low_map = (mu_mean_map - z * mu_scale_map).clamp(0.0, 1.2)
        mu_high_map = (mu_mean_map + z * mu_scale_map).clamp(0.0, 1.2)
        mu_low_queries = (attention_queries * mu_low_map).sum(dim=(2, 3))
        mu_high_queries = (attention_queries * mu_high_map).sum(dim=(2, 3))

        query_summary = torch.einsum("bqhw,bchw->bqc", attention_queries, hidden)
        if self.query_gate is None:
            query_weights = attention_queries.new_ones(attention_queries.size(0), 1)
        else:
            query_weights = torch.softmax(self.query_gate(query_summary).squeeze(-1), dim=1)
        attention = (attention_queries * query_weights.view(query_weights.size(0), -1, 1, 1)).sum(
            dim=1,
            keepdim=True,
        )
        risk_logits = (risk_logits_queries * query_weights.unsqueeze(-1)).sum(dim=1)
        mu_low = (mu_low_queries * query_weights).sum(dim=1)
        mu_high = (mu_high_queries * query_weights).sum(dim=1)
        mu_mid_queries = 0.5 * (mu_low_queries + mu_high_queries)
        query_disagreement = (
            ((mu_mid_queries - (mu_mid_queries * query_weights).sum(dim=1, keepdim=True)).square() * query_weights)
            .sum(dim=1)
            .sqrt()
        )

        entropy = -(attention.flatten(1) * attention.flatten(1).clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / math.log(float(attention.size(2) * attention.size(3)))
        region_mixture_map = self._region_mixture_score_from_evidence(evidence)
        region_mixture_signal = (
            (attention * region_mixture_map).sum(dim=(2, 3)).squeeze(1)
            if region_mixture_map is not None
            else torch.zeros_like(entropy)
        )
        expansion = self.entropy_expansion * entropy
        if self.region_mixture_expansion > 0:
            expansion = expansion + self.region_mixture_expansion * region_mixture_signal
        if self.query_disagreement_expansion > 0 and self.num_queries > 1:
            expansion = expansion + self.query_disagreement_expansion * query_disagreement
        interval = torch.stack(
            [
                (mu_low - expansion).clamp(0.0, 1.2),
                (mu_high + expansion).clamp(0.0, 1.2),
            ],
            dim=1,
        )
        mean = interval.mean(dim=1)
        scale = ((interval[:, 1] - interval[:, 0]) / (2.0 * z)).clamp(1e-3, 1.0)
        summary = (attention * hidden).sum(dim=(2, 3))

        out = {
            "summary": self.summary_proj(summary),
            "risk_logits": risk_logits,
            "risk_logits_map": risk_logits_map,
            "mu_interval": interval,
            "mu_mean": mean,
            "mu_scale": scale,
            "mu_interval_map": torch.cat([mu_low_map, mu_high_map], dim=1),
            "attention": attention,
            "attention_queries": attention_queries,
            "query_weights": query_weights,
            "query_disagreement": query_disagreement,
            "attention_entropy": entropy,
            "region_mixture_signal": region_mixture_signal,
            "contact_prior": contact_prior,
            "road_likelihood": road_likelihood,
            "attention_prior": attention_prior.clamp_min(1e-4),
        }
        if region_mixture_map is not None:
            out["region_mixture_map"] = region_mixture_map
        return out

    def _evidence_channels(self, x: torch.Tensor) -> torch.Tensor:
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
        wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)
        roughness = torch.sigmoid((grad + 0.5 * lap - 0.12) * 12.0)
        region_mixture_cues = self._region_mixture_cues(
            rgb=rgb,
            gray=gray,
            saturation=saturation,
            value=value,
            grad=grad,
            lap=lap,
            snow_like=snow_like,
            specular=specular,
            dark_water=dark_water,
            wet_proxy=wet_proxy,
            roughness=roughness,
        )
        x_coord, y_coord, contact_prior = self._coordinate_maps(x)
        return torch.cat(
            [
                rgb,
                gray,
                saturation,
                value,
                grad,
                lap,
                snow_like,
                specular,
                dark_water,
                wet_proxy,
                roughness,
                *region_mixture_cues,
                x_coord,
                y_coord,
                contact_prior,
            ],
            dim=1,
        )

    def _coordinate_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        yy = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xx = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        x_coord = xx.expand(b, 1, h, w)
        y_coord = yy.expand(b, 1, h, w)
        bottom = torch.sigmoid((y_coord - 0.45) * 12.0)
        center = torch.exp(-((x_coord - 0.5).square()) / (2.0 * 0.36**2))
        contact_prior = (bottom * (0.35 + 0.65 * center)).clamp_min(1e-4)
        return x_coord, y_coord, contact_prior

    def _road_likelihood(self, evidence: torch.Tensor) -> torch.Tensor:
        # Base channel layout: rgb(3), gray, saturation, value, grad, lap,
        # snow_like, specular, dark_water, wet_proxy, roughness. Optional
        # region-mixture cues are inserted before x/y/contact.
        saturation = evidence[:, 4:5]
        value = evidence[:, 5:6]
        grad = evidence[:, 6:7]
        lap = evidence[:, 7:8]
        snow_like = evidence[:, 8:9]
        wet_proxy = evidence[:, 11:12]
        roughness = evidence[:, 12:13]
        y_coord = evidence[:, self.y_coord_channel : self.y_coord_channel + 1]
        bottom = torch.sigmoid((y_coord - 0.35) * 10.0)
        textured_road = torch.sigmoid((grad + 0.4 * lap + 0.4 * roughness - 0.08) * 10.0)
        flat_bright_road = snow_like * torch.sigmoid((value - 0.45) * 8.0)
        wet_road = wet_proxy * torch.sigmoid((0.85 - saturation) * 6.0)
        roadness = torch.clamp(0.45 * textured_road + 0.30 * flat_bright_road + 0.25 * wet_road, 0.0, 1.0)
        return (0.15 + 0.85 * bottom * roadness).clamp(1e-4, 1.0)

    def _region_mixture_cues(
        self,
        *,
        rgb: torch.Tensor,
        gray: torch.Tensor,
        saturation: torch.Tensor,
        value: torch.Tensor,
        grad: torch.Tensor,
        lap: torch.Tensor,
        snow_like: torch.Tensor,
        specular: torch.Tensor,
        dark_water: torch.Tensor,
        wet_proxy: torch.Tensor,
        roughness: torch.Tensor,
    ) -> list[torch.Tensor]:
        if not self.use_region_mixture_cues:
            return []
        color_std = self._local_std(rgb).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        value_std = self._local_std(value).clamp(0.0, 1.0)
        saturation_std = self._local_std(saturation).clamp(0.0, 1.0)
        texture_span = self._local_mean(grad + 0.5 * lap).clamp(0.0, 1.0)
        state_stack = torch.cat([snow_like, specular, dark_water, wet_proxy, roughness], dim=1)
        state_mean = self._local_mean(state_stack).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        mixture_score = torch.sigmoid(
            9.0 * (1.4 * color_std + value_std + 0.8 * saturation_std + 0.7 * texture_span + 0.8 * state_mean - 0.32)
        )
        return [color_std, value_std, saturation_std, texture_span, mixture_score]

    def _region_mixture_score_from_evidence(self, evidence: torch.Tensor) -> torch.Tensor | None:
        if self.region_mixture_score_channel is None:
            return None
        return evidence[
            :,
            self.region_mixture_score_channel : self.region_mixture_score_channel + 1,
            :,
            :,
        ].clamp(0.0, 1.0)

    def _local_mean(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.region_mixture_kernel_size // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=self.region_mixture_kernel_size, stride=1)

    def _local_std(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._local_mean(x)
        mean_sq = self._local_mean(x.square())
        return (mean_sq - mean.square()).clamp_min(0.0).sqrt()


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from friction_affordance.ontology import TASKS
from .backbone import build_backbone
from .evidence_field import LocalFrictionEvidenceField
from .friction_set import FrictionSetHead
from .texture import PhysicsTextureBranch


class FrictionAffordanceModel(nn.Module):
    def __init__(
        self,
        backbone: str = "simple_cnn",
        embedding_dim: int = 256,
        dropout: float = 0.1,
        pretrained: bool = False,
        use_physics_branch: bool = False,
        physics_dim: int = 64,
        physics_quality_cues: bool = False,
        physics_quality_region_cues: bool = True,
        num_domains: int = 0,
        use_friction_set: bool = False,
        friction_set_entropy_expansion: float = 0.10,
        friction_set_interval_mix: float = 1.0,
        use_evidence_field: bool = False,
        evidence_dim: int = 64,
        evidence_hidden_dim: int = 48,
        evidence_patch_stride: int = 8,
        evidence_contact_prior_strength: float = 1.0,
        evidence_road_likelihood_prior_strength: float = 0.0,
        evidence_entropy_expansion: float = 0.08,
        evidence_interval_mix: float = 0.0,
        evidence_risk_logit_mix: float = 0.0,
        evidence_region_mixture_cues: bool = False,
        evidence_region_mixture_expansion: float = 0.0,
        evidence_region_mixture_kernel_size: int = 9,
        evidence_num_queries: int = 1,
        evidence_query_disagreement_expansion: float = 0.0,
        use_domain_adapters: bool = False,
        domain_adapter_scale: float = 0.15,
        use_feature_mixstyle: bool = False,
        feature_mixstyle_p: float = 0.5,
        feature_mixstyle_alpha: float = 0.1,
        feature_mixstyle_groups: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = build_backbone(backbone, embedding_dim, pretrained=pretrained)
        self.physics_branch = (
            PhysicsTextureBranch(
                physics_dim,
                quality_cues=physics_quality_cues,
                quality_region_cues=physics_quality_region_cues,
            )
            if use_physics_branch
            else None
        )
        self.evidence_field = (
            LocalFrictionEvidenceField(
                out_dim=evidence_dim,
                hidden_dim=evidence_hidden_dim,
                patch_stride=evidence_patch_stride,
                contact_prior_strength=evidence_contact_prior_strength,
                road_likelihood_prior_strength=evidence_road_likelihood_prior_strength,
                entropy_expansion=evidence_entropy_expansion,
                use_region_mixture_cues=evidence_region_mixture_cues,
                region_mixture_expansion=evidence_region_mixture_expansion,
                region_mixture_kernel_size=evidence_region_mixture_kernel_size,
                num_queries=evidence_num_queries,
                query_disagreement_expansion=evidence_query_disagreement_expansion,
            )
            if use_evidence_field
            else None
        )
        self.friction_set_interval_mix = float(friction_set_interval_mix)
        self.evidence_interval_mix = float(evidence_interval_mix)
        self.evidence_risk_logit_mix = float(evidence_risk_logit_mix)
        head_dim = (
            embedding_dim
            + (physics_dim if use_physics_branch else 0)
            + (evidence_dim if use_evidence_field else 0)
        )
        self.norm = nn.LayerNorm(head_dim)
        self.domain_adapter = (
            DomainAffineAdapter(num_domains, head_dim, scale=domain_adapter_scale)
            if use_domain_adapters and num_domains > 1
            else None
        )
        self.feature_mixstyle = (
            FeatureMixStyle(
                p=feature_mixstyle_p,
                alpha=feature_mixstyle_alpha,
                groups=feature_mixstyle_groups,
            )
            if use_feature_mixstyle
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(head_dim, len(labels)) for task, labels in TASKS.items()}
        )
        self.mu_head = nn.Sequential(
            nn.Linear(head_dim, max(head_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(head_dim // 2, 32), 2),
        )
        self.domain_head = (
            nn.Sequential(
                nn.Linear(head_dim, max(head_dim // 2, 32)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(head_dim // 2, 32), num_domains),
            )
            if num_domains > 1
            else None
        )
        self.friction_set_head = (
            FrictionSetHead(entropy_expansion=friction_set_entropy_expansion)
            if use_friction_set
            else None
        )

    def forward(
        self,
        image: torch.Tensor,
        grl_lambda: float = 0.0,
        domain_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        parts = [self.encoder(image)]
        if self.physics_branch is not None:
            parts.append(self.physics_branch(image))
        evidence_field = None
        if self.evidence_field is not None:
            evidence_field = self.evidence_field(image)
            parts.append(evidence_field["summary"])
        shared_feat = self.norm(torch.cat(parts, dim=1))
        feat = shared_feat
        if self.feature_mixstyle is not None:
            feat = self.feature_mixstyle(feat)
        if self.domain_adapter is not None:
            feat = self.domain_adapter(feat, domain_idx)
        feat = self.dropout(feat)
        logits = {task: head(feat) for task, head in self.heads.items()}
        if evidence_field is not None and self.evidence_risk_logit_mix > 0:
            mix = min(max(self.evidence_risk_logit_mix, 0.0), 1.0)
            logits["risk"] = (1.0 - mix) * logits["risk"] + mix * evidence_field["risk_logits"]
        raw = self.mu_head(feat)
        mu_mean = (1.2 * torch.sigmoid(raw[:, 0])).clamp(1e-4, 1.2)
        mu_scale = F.softplus(raw[:, 1]).clamp(1e-3, 1.0)
        z = 1.2815515655446004
        mu_low = (mu_mean - z * mu_scale).clamp(0.0, 1.2)
        mu_high = (mu_mean + z * mu_scale).clamp(0.0, 1.2)
        out = {
            "features": feat,
            "shared_features": shared_feat,
            "logits": logits,
            "mu_mean": mu_mean,
            "mu_scale": mu_scale,
            "mu_interval": torch.stack([mu_low, mu_high], dim=1),
        }
        if evidence_field is not None:
            out["evidence_field"] = evidence_field
        if self.friction_set_head is not None:
            friction_set = self.friction_set_head(logits)
            out["mu_mean_parametric"] = out["mu_mean"]
            out["mu_scale_parametric"] = out["mu_scale"]
            out["mu_interval_parametric"] = out["mu_interval"]
            mix = self.friction_set_interval_mix
            if mix >= 1.0:
                out["mu_mean"] = friction_set["mu_mean"]
                out["mu_scale"] = friction_set["mu_scale"]
                out["mu_interval"] = friction_set["mu_interval"]
            elif mix > 0.0:
                out["mu_interval"] = (
                    mix * friction_set["mu_interval"]
                    + (1.0 - mix) * out["mu_interval"]
                )
                out["mu_mean"] = out["mu_interval"].mean(dim=1)
                out["mu_scale"] = (
                    (out["mu_interval"][:, 1] - out["mu_interval"][:, 0])
                    / (2.0 * z)
                ).clamp(1e-3, 1.0)
            out["friction_set"] = friction_set
        if evidence_field is not None and self.evidence_interval_mix > 0:
            mix = min(max(self.evidence_interval_mix, 0.0), 1.0)
            out["mu_interval_before_evidence"] = out["mu_interval"]
            out["mu_interval"] = (
                mix * evidence_field["mu_interval"] + (1.0 - mix) * out["mu_interval"]
            )
            out["mu_mean"] = out["mu_interval"].mean(dim=1)
            out["mu_scale"] = (
                (out["mu_interval"][:, 1] - out["mu_interval"][:, 0]) / (2.0 * z)
            ).clamp(1e-3, 1.0)
        if self.domain_head is not None:
            out["domain_logits"] = self.domain_head(_grad_reverse(shared_feat, float(grl_lambda)))
        if self.domain_adapter is not None:
            out["domain_adapter_penalty"] = self.domain_adapter.regularization()
        return out


class DomainAffineAdapter(nn.Module):
    """Tiny per-domain affine residual on shared normalized features.

    The adapter starts as an identity map. It can absorb dataset-specific camera
    or annotation style offsets while leaving the shared representation available
    for shortcut diagnostics and domain-adversarial regularization.
    """

    def __init__(self, num_domains: int, feature_dim: int, scale: float = 0.15) -> None:
        super().__init__()
        self.num_domains = int(num_domains)
        self.scale = float(scale)
        self.gamma = nn.Embedding(self.num_domains, feature_dim)
        self.beta = nn.Embedding(self.num_domains, feature_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, features: torch.Tensor, domain_idx: torch.Tensor | None = None) -> torch.Tensor:
        if domain_idx is None:
            return features
        idx = domain_idx.to(device=features.device, dtype=torch.long).clamp(0, self.num_domains - 1)
        gamma = self.gamma(idx)
        beta = self.beta(idx)
        return features * (1.0 + self.scale * gamma) + self.scale * beta

    def regularization(self) -> torch.Tensor:
        return self.gamma.weight.square().mean() + self.beta.weight.square().mean()


class FeatureMixStyle(nn.Module):
    """Training-only grouped feature-statistic mixing for shortcut stress tests.

    The shared vector is layer-normalized before this module, so a single global
    mean/std would be nearly constant. Group-wise statistics still carry channel
    style, camera, and annotation-domain cues, making this a cheap fail-fast
    candidate on 4GB GPUs.
    """

    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.1,
        groups: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.p = min(max(float(p), 0.0), 1.0)
        self.alpha = max(float(alpha), 1e-3)
        self.groups = max(int(groups), 1)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0 or features.size(0) < 2:
            return features
        if torch.rand((), device=features.device) > self.p:
            return features
        batch, dim = features.shape
        groups = min(self.groups, dim)
        group_width = dim // groups
        if group_width < 2:
            return features
        trim = groups * group_width
        grouped = features[:, :trim].reshape(batch, groups, group_width)
        mean = grouped.mean(dim=2, keepdim=True).detach()
        std = grouped.var(dim=2, keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
        normalized = (grouped - mean) / std
        perm = torch.randperm(features.size(0), device=features.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((batch, groups, 1)).to(device=features.device, dtype=features.dtype)
        mixed_mean = lam * mean + (1.0 - lam) * mean[perm]
        mixed_std = lam * std + (1.0 - lam) * std[perm]
        mixed = (normalized * mixed_std + mixed_mean).reshape(batch, trim)
        if trim == dim:
            return mixed
        return torch.cat([mixed, features[:, trim:]], dim=1)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def _grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    if lambd <= 0:
        return x
    return _GradientReverse.apply(x, lambd)

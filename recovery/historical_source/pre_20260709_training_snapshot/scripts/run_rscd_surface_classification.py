from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from tqdm import tqdm

from friction_affordance.engine import dataloader_worker_settings
from friction_affordance.models.backbone import build_backbone
from friction_affordance.models.physics_evidence import (
    PhysicsEvidenceMapHeads,
    PhysicsEvidenceTarget,
    physics_evidence_loss,
)
from friction_affordance.models.texture import (
    AntiHumanTextureBranch,
    DirectionalTextureBranch,
    FactorCoupledPhysicsTokenBranch,
    FactorConditionedPhysicsTokenBranch,
    LocalPhysicsFieldBranch,
    PhysicsAttentionBranch,
    PhysicsTextureBranch,
    RetinexTextureBranch,
    RelationConditionedPhysicsExpertBranch,
    SemanticPhysicsAttentionBranch,
    TopologicalTextureBranch,
    VisibilityObservedRoughnessBranch,
    WaveletTextureBranch,
    _normalize_map,
    _soft_euler_curve_stats,
)
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed


ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_TRAIN = Path("data/manifests_full/rscd_prepared_train.csv")
DEFAULT_VAL = Path("data/manifests_full/rscd_prepared_val.csv")
DEFAULT_TEST = Path("data/manifests_full/rscd_prepared_test.csv")
DEFAULT_OUT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification\convnext_tiny")
FACTOR_LABELS = {
    "friction": ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"],
    "material": ["asphalt", "concrete", "mud", "gravel"],
    "unevenness": ["smooth", "slight", "severe"],
}
MECHANISM_TRAIN_SCOPES = (
    "all",
    "core_paved",
    "dry_visible",
    "dry_paved_roughness",
    "wet_water_paved",
    "wet_water_concrete",
    "granular",
    "winter",
    "hard_audited",
)
OBSERVER_HINF_SCOPES = MECHANISM_TRAIN_SCOPES + (
    "non_wet_water",
    "wet_water_guarded",
)
PHYSICS_EVIDENCE_FIELD_MODES = (
    "all",
    "roughness_coupling",
    "wet_concrete_hidden_roughness",
    "granular_guard",
)
DEFAULT_CHECKPOINT_HARD_SLICE_CLASSES = (
    "water_concrete_slight",
    "water_concrete_severe",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "wet_asphalt_severe",
    "water_asphalt_severe",
)


def _soft_connectedness(mask: torch.Tensor) -> torch.Tensor:
    pooled = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return (mask * pooled).mean(dim=(2, 3))


class ConditionalFactorConsistencyProjection(nn.Module):
    """Deterministic label-factor consistency projection for RSCD logits.

    RSCD classes are generated from partially observed factors rather than
    from 27 independent symbols. Given logits z_c for class c=(f,m,u), this
    module builds conditional log-probability fields P(f|m,u), P(m|f,u), and
    P(u|f,m) directly from the current logits, then adds a small centered
    consistency residual. It is parameter-free and can therefore be evaluated
    as a low-risk calibration layer on top of a trained classifier.
    """

    def __init__(
        self,
        class_to_idx: dict[str, int],
        *,
        scale: float = 0.04,
        gate_threshold: float = 0.35,
        gate_temperature: float = 10.0,
        friction_weight: float = 1.0,
        material_weight: float = 0.6,
        unevenness_weight: float = 1.2,
        focus: str = "core",
        protected_negative_limit: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        focus = str(focus)
        if focus not in {"all", "core", "hard"}:
            raise ValueError(f"unknown conditional factor projection focus: {focus}")

        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        num_classes = len(class_to_idx)
        factors = {idx: _factor_text(idx_to_class[idx]) for idx in range(num_classes)}
        hard_classes = {
            "water_concrete_slight",
            "water_asphalt_slight",
            "wet_concrete_slight",
            "water_concrete_severe",
            "dry_concrete_slight",
            "wet_concrete_severe",
            "water_asphalt_severe",
            "dry_concrete_severe",
        }
        axis_weights = {
            "friction": float(friction_weight),
            "material": float(material_weight),
            "unevenness": float(unevenness_weight),
        }
        numerators = []
        denominators = []
        weights = []
        valid_rows = []
        for axis in FACTOR_LABELS:
            other_axes = [name for name in FACTOR_LABELS if name != axis]
            numerator = torch.zeros((num_classes, num_classes), dtype=torch.bool)
            denominator = torch.zeros((num_classes, num_classes), dtype=torch.bool)
            valid = torch.zeros(num_classes, dtype=torch.float32)
            for i in range(num_classes):
                current = factors[i]
                if current[axis] is None:
                    continue
                is_core = (
                    current["friction"] in {"dry", "wet", "water"}
                    and current["material"] in {"asphalt", "concrete"}
                    and current["unevenness"] in {"smooth", "slight", "severe"}
                )
                if focus == "core" and not is_core:
                    continue
                if focus == "hard" and canonical_class_label(idx_to_class[i]) not in hard_classes:
                    continue
                active_other_axes = [name for name in other_axes if current[name] is not None]
                axis_values: set[str] = set()
                for j in range(num_classes):
                    candidate = factors[j]
                    if candidate[axis] is None:
                        continue
                    if any(candidate[name] != current[name] for name in active_other_axes):
                        continue
                    denominator[i, j] = True
                    axis_values.add(str(candidate[axis]))
                    if candidate[axis] == current[axis]:
                        numerator[i, j] = True
                if denominator[i].sum() > numerator[i].sum() and len(axis_values) >= 2:
                    valid[i] = 1.0
            numerators.append(numerator)
            denominators.append(denominator)
            weights.append(axis_weights[axis])
            valid_rows.append(valid)

        protected = torch.zeros(num_classes, dtype=torch.bool)
        for idx, row in factors.items():
            protected[idx] = row["friction"] in {"wet", "water"}
        self.register_buffer("numerator_masks", torch.stack(numerators, dim=0))
        self.register_buffer("denominator_masks", torch.stack(denominators, dim=0))
        self.register_buffer("axis_weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("valid_rows", torch.stack(valid_rows, dim=0))
        self.register_buffer("protected_classes", protected)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scale == 0.0:
            return logits.new_zeros(logits.shape)
        dtype = logits.dtype
        numerators = self.numerator_masks.to(device=logits.device)
        denominators = self.denominator_masks.to(device=logits.device)
        valid_rows = self.valid_rows.to(device=logits.device, dtype=dtype)
        axis_weights = self.axis_weights.to(device=logits.device, dtype=dtype)
        residual = logits.new_zeros(logits.shape)
        total_weight = logits.new_zeros(logits.shape)
        for axis_idx in range(numerators.shape[0]):
            numerator = numerators[axis_idx]
            denominator = denominators[axis_idx]
            valid = valid_rows[axis_idx].view(1, -1)
            if not bool((valid > 0).any()):
                continue
            numer_score = torch.logsumexp(logits.unsqueeze(1).masked_fill(~numerator.unsqueeze(0), -1.0e4), dim=2)
            denom_score = torch.logsumexp(logits.unsqueeze(1).masked_fill(~denominator.unsqueeze(0), -1.0e4), dim=2)
            conditional_score = (numer_score - denom_score) * valid
            residual = residual + axis_weights[axis_idx] * conditional_score
            total_weight = total_weight + axis_weights[axis_idx] * valid
        residual = residual / total_weight.clamp_min(1.0e-6)
        valid_any = (total_weight > 0).to(dtype=dtype)
        if bool((valid_any > 0).any()):
            mean = (residual * valid_any).sum(dim=1, keepdim=True) / valid_any.sum(dim=1, keepdim=True).clamp_min(1.0)
            residual = (residual - mean) * valid_any

        top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
        if top2.shape[1] < 2:
            margin = logits.new_zeros((logits.shape[0],))
        else:
            margin = top2[:, 0] - top2[:, 1]
        gate = torch.sigmoid((self.gate_threshold - margin) * self.gate_temperature).view(-1, 1).to(dtype=dtype)
        residual = float(self.scale) * gate * residual
        if self.protected_negative_limit > 0.0:
            protected = self.protected_classes.to(device=logits.device).view(1, -1)
            floor = -self.protected_negative_limit
            residual = torch.where(protected, residual.clamp_min(floor), residual)
        return residual


class HeterogeneousLabelRouter(nn.Module):
    """Zero-initialized residual head for RSCD's heterogeneous label topology.

    RSCD is not a single Cartesian product: paved road labels form
    friction-material-roughness cells, mud/gravel labels form friction-material
    cells, and snow/ice labels are standalone weather states. This head learns a
    coarse label family router plus a within-family residual classifier, so the
    base ConvNeXt classifier is refined by the correct subspace instead of by a
    homogeneous 27-way residual.
    """

    def __init__(
        self,
        in_dim: int,
        class_to_idx: dict[str, int],
        *,
        hidden_dim: int = 128,
        scale: float = 0.08,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        groups: dict[str, list[int]] = {"paved": [], "loose": [], "weather": []}
        group_to_idx = {"paved": 0, "loose": 1, "weather": 2}
        group_offsets = {name: 0 for name in groups}
        class_group = []
        branch_index = []
        for class_idx in range(len(idx_to_class)):
            label = idx_to_class[class_idx]
            factors = parse_rscd_factors(label)
            friction = FACTOR_LABELS["friction"][factors["friction"]] if factors["friction"] >= 0 else label
            material = FACTOR_LABELS["material"][factors["material"]] if factors["material"] >= 0 else None
            if label in {"fresh_snow", "melted_snow", "ice"} or friction in {"fresh_snow", "melted_snow", "ice"}:
                group_name = "weather"
            elif material in {"mud", "gravel"}:
                group_name = "loose"
            else:
                group_name = "paved"
            class_group.append(group_to_idx[group_name])
            branch_index.append(group_offsets[group_name])
            groups[group_name].append(class_idx)
            group_offsets[group_name] += 1

        self.register_buffer("class_group", torch.as_tensor(class_group, dtype=torch.long))
        self.register_buffer("branch_index", torch.as_tensor(branch_index, dtype=torch.long))
        self.route_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), len(groups)),
        )
        self.branch_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), max(len(groups[name]), 1)),
                )
                for name in ("paved", "loose", "weather")
            ]
        )
        if zero_init:
            self._zero_last(self.route_head)
            for head in self.branch_heads:
                self._zero_last(head)

    @staticmethod
    def _zero_last(module: nn.Module) -> None:
        for child in reversed(list(module.modules())):
            if isinstance(child, nn.Linear):
                nn.init.zeros_(child.weight)
                nn.init.zeros_(child.bias)
                return

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        class_group = self.class_group.to(device=feature.device)
        branch_index = self.branch_index.to(device=feature.device)
        route_logits = self.route_head(feature)
        residual = route_logits.index_select(1, class_group)
        for group_idx, branch_head in enumerate(self.branch_heads):
            mask = class_group.eq(int(group_idx))
            if not bool(mask.any()):
                continue
            branch_logits = branch_head(feature)
            residual[:, mask] = residual[:, mask] + branch_logits.index_select(1, branch_index[mask])
        return float(self.scale) * residual


class FactorizedLowRankHead(nn.Module):
    """Low-rank class residual built from RSCD friction/material/unevenness factors."""

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        class_to_idx: dict[str, int],
        rank: int = 64,
        scale: float = 0.25,
        normalize: bool = True,
        zero_init: bool = False,
        factors: tuple[str, ...] | None = None,
        use_class_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.use_class_embedding = bool(use_class_embedding)
        rank = int(rank)
        if rank <= 0:
            raise ValueError("factorized low-rank rank must be positive.")
        if factors is None:
            factors = tuple(FACTOR_LABELS)
        invalid = sorted(set(factors).difference(FACTOR_LABELS))
        if invalid:
            raise ValueError(f"unknown factorized factors: {invalid}")
        self.factor_names = tuple(factors)

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, rank, bias=False),
        )
        self.class_embedding = nn.Parameter(torch.empty(num_classes, rank))
        self.factor_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(len(labels), rank))
                for name, labels in FACTOR_LABELS.items()
                if name in self.factor_names
            }
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

        factor_buffers = build_class_factor_buffers(class_to_idx)
        for name, (indices, mask) in factor_buffers.items():
            self.register_buffer(f"{name}_class_factor_idx", indices)
            self.register_buffer(f"{name}_class_factor_mask", mask)

        if zero_init:
            nn.init.zeros_(self.class_embedding)
            for embedding in self.factor_embeddings.values():
                nn.init.zeros_(embedding)
        else:
            nn.init.trunc_normal_(self.class_embedding, std=0.02)
            for embedding in self.factor_embeddings.values():
                nn.init.trunc_normal_(embedding, std=0.02)

    def class_weight(self) -> torch.Tensor:
        if self.use_class_embedding:
            weight = self.class_embedding
            active = torch.ones((weight.shape[0], 1), device=weight.device, dtype=weight.dtype)
        else:
            weight = torch.zeros_like(self.class_embedding)
            active = torch.zeros((weight.shape[0], 1), device=weight.device, dtype=weight.dtype)
        for name, embedding in self.factor_embeddings.items():
            factor_idx = getattr(self, f"{name}_class_factor_idx")
            factor_mask = getattr(self, f"{name}_class_factor_mask").to(device=weight.device, dtype=weight.dtype)
            weight = weight + embedding[factor_idx] * factor_mask.unsqueeze(1)
            active = active + factor_mask.unsqueeze(1)
        return weight / active.clamp_min(1.0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        query = self.feature_proj(feature)
        weight = self.class_weight()
        if self.normalize:
            query = F.normalize(query, dim=1)
            weight = F.normalize(weight, dim=1)
        return self.scale * (query @ weight.t() + self.bias)


class SafeAdaptiveFactorizedLowRankHead(nn.Module):
    """Uncertainty-gated low-rank residual with wet/water safety protection.

    The ordinary factorized head helped fast average metrics but reduced some
    safety-relevant wet/water slices. This variant uses the same compositional
    class-factor idea, but only applies the residual strongly when the base
    classifier is uncertain, and it prevents the residual from strongly lowering
    protected wet/water class logits.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        class_to_idx: dict[str, int],
        rank: int = 64,
        scale: float = 0.25,
        normalize: bool = True,
        zero_init: bool = True,
        factors: tuple[str, ...] | None = None,
        use_class_embedding: bool = False,
        gate_threshold: float = 0.55,
        gate_temperature: float = 8.0,
        protected_negative_limit: float = 0.0,
        protected_friction: tuple[str, ...] = ("wet", "water"),
    ) -> None:
        super().__init__()
        self.base = FactorizedLowRankHead(
            in_dim=in_dim,
            num_classes=num_classes,
            class_to_idx=class_to_idx,
            rank=rank,
            scale=scale,
            normalize=normalize,
            zero_init=zero_init,
            factors=factors,
            use_class_embedding=use_class_embedding,
        )
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        protected = torch.zeros(num_classes, dtype=torch.bool)
        protected_set = {str(item) for item in protected_friction}
        for class_name, class_idx in class_to_idx.items():
            friction = _factor_text(class_name)["friction"]
            if friction in protected_set:
                protected[int(class_idx)] = True
        self.register_buffer("protected_class_mask", protected.view(1, -1))

    def forward(self, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        residual = self.base(feature)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return gate.to(dtype=residual.dtype) * residual


class FactorInteractionLowRankHead(nn.Module):
    """Low-rank residual over pairwise RSCD label-factor interactions.

    The first-order factorized head assumes class evidence is almost additive
    across friction, material, and unevenness. RSCD hard cases show that this is
    too weak: water-on-concrete and water-on-asphalt boundaries need interaction
    terms. This head learns only pairwise interaction embeddings and starts from
    a protected, uncertainty-gated residual so the validated base classifier is
    not overwritten early in training.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        class_to_idx: dict[str, int],
        rank: int = 64,
        scale: float = 0.20,
        normalize: bool = True,
        zero_init: bool = True,
        gate_threshold: float = 0.55,
        gate_temperature: float = 8.0,
        protected_negative_limit: float = 0.0,
        protected_friction: tuple[str, ...] = ("wet", "water"),
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        rank = int(rank)
        if rank <= 0:
            raise ValueError("factor interaction rank must be positive.")
        self.feature_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, rank, bias=False))
        pair_dims = {
            "friction_material": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["material"]),
            "friction_unevenness": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["unevenness"]),
            "material_unevenness": len(FACTOR_LABELS["material"]) * len(FACTOR_LABELS["unevenness"]),
        }
        self.pair_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(size, rank)) for name, size in pair_dims.items()}
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        pair_indices: dict[str, list[int]] = {name: [] for name in pair_dims}
        pair_masks: dict[str, list[float]] = {name: [] for name in pair_dims}
        protected = torch.zeros(num_classes, dtype=torch.bool)
        protected_set = {str(item) for item in protected_friction}
        for class_idx in range(num_classes):
            class_name = idx_to_class[class_idx]
            factors = parse_rscd_factors(class_name)
            f_idx = factors["friction"]
            m_idx = factors["material"]
            u_idx = factors["unevenness"]
            if _factor_text(class_name)["friction"] in protected_set:
                protected[class_idx] = True
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_material",
                f_idx,
                m_idx,
                len(FACTOR_LABELS["material"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_unevenness",
                f_idx,
                u_idx,
                len(FACTOR_LABELS["unevenness"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "material_unevenness",
                m_idx,
                u_idx,
                len(FACTOR_LABELS["unevenness"]),
            )
        for name in pair_dims:
            self.register_buffer(f"{name}_idx", torch.tensor(pair_indices[name], dtype=torch.long))
            self.register_buffer(f"{name}_mask", torch.tensor(pair_masks[name], dtype=torch.float32))
        self.register_buffer("protected_class_mask", protected.view(1, -1))

        for embedding in self.pair_embeddings.values():
            if zero_init:
                nn.init.zeros_(embedding)
            else:
                nn.init.trunc_normal_(embedding, std=0.02)

    def class_weight(self) -> torch.Tensor:
        first_embedding = next(iter(self.pair_embeddings.values()))
        weight = torch.zeros((self.bias.numel(), first_embedding.shape[1]), device=first_embedding.device)
        active = torch.zeros((self.bias.numel(), 1), device=first_embedding.device)
        for name, embedding in self.pair_embeddings.items():
            pair_idx = getattr(self, f"{name}_idx")
            pair_mask = getattr(self, f"{name}_mask").to(device=embedding.device, dtype=embedding.dtype)
            weight = weight + embedding[pair_idx] * pair_mask.unsqueeze(1)
            active = active + pair_mask.unsqueeze(1)
        return weight / active.clamp_min(1.0)

    def forward(self, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        query = self.feature_proj(feature)
        weight = self.class_weight()
        if self.normalize:
            query = F.normalize(query, dim=1)
            weight = F.normalize(weight, dim=1)
        residual = self.scale * (query @ weight.t() + self.bias)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return gate.to(dtype=residual.dtype) * residual


class ConditionalCouplingDecompositionField(nn.Module):
    """Dynamic pairwise coupling field for RSCD compositional labels.

    RSCD's 27 labels are not a complete Cartesian product: snow/ice classes
    have no material/roughness factors, and mud/gravel classes omit roughness.
    This module therefore keeps three relation-specific low-rank fields and
    predicts a per-image coupling gate instead of averaging all pair relations.
    """

    relation_names = ("friction_material", "friction_unevenness", "material_unevenness")

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        class_to_idx: dict[str, int],
        rank: int = 64,
        scale: float = 0.08,
        normalize: bool = True,
        zero_init: bool = True,
        gate_threshold: float = 0.35,
        gate_temperature: float = 8.0,
        protected_negative_limit: float = 0.0,
        protected_friction: tuple[str, ...] = ("wet", "water"),
        relation_gate_hidden_dim: int = 64,
        relation_gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        self.relation_gate_temperature = max(float(relation_gate_temperature), 1e-4)
        rank = int(rank)
        if rank <= 0:
            raise ValueError("conditional coupling rank must be positive.")
        relation_gate_hidden_dim = max(int(relation_gate_hidden_dim), 1)

        self.query_projs = nn.ModuleDict(
            {name: nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, rank, bias=False)) for name in self.relation_names}
        )
        pair_dims = {
            "friction_material": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["material"]),
            "friction_unevenness": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["unevenness"]),
            "material_unevenness": len(FACTOR_LABELS["material"]) * len(FACTOR_LABELS["unevenness"]),
        }
        self.pair_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(size, rank)) for name, size in pair_dims.items()}
        )
        self.relation_gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, relation_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_gate_hidden_dim, len(self.relation_names)),
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        pair_indices: dict[str, list[int]] = {name: [] for name in pair_dims}
        pair_masks: dict[str, list[float]] = {name: [] for name in pair_dims}
        protected = torch.zeros(num_classes, dtype=torch.bool)
        protected_set = {str(item) for item in protected_friction}
        for class_idx in range(num_classes):
            class_name = idx_to_class[class_idx]
            factors = parse_rscd_factors(class_name)
            if _factor_text(class_name)["friction"] in protected_set:
                protected[class_idx] = True
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_material",
                factors["friction"],
                factors["material"],
                len(FACTOR_LABELS["material"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_unevenness",
                factors["friction"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "material_unevenness",
                factors["material"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
        for name in pair_dims:
            self.register_buffer(f"{name}_idx", torch.tensor(pair_indices[name], dtype=torch.long))
            self.register_buffer(f"{name}_mask", torch.tensor(pair_masks[name], dtype=torch.float32))
        self.register_buffer("protected_class_mask", protected.view(1, -1))

        for embedding in self.pair_embeddings.values():
            if zero_init:
                nn.init.zeros_(embedding)
            else:
                nn.init.trunc_normal_(embedding, std=0.02)
        nn.init.zeros_(self.relation_gate[-1].weight)
        nn.init.zeros_(self.relation_gate[-1].bias)

    def relation_weight(self, name: str) -> torch.Tensor:
        embedding = self.pair_embeddings[name]
        pair_idx = getattr(self, f"{name}_idx")
        pair_mask = getattr(self, f"{name}_mask").to(device=embedding.device, dtype=embedding.dtype)
        return embedding[pair_idx] * pair_mask.unsqueeze(1)

    def forward(self, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        relation_scores = []
        for name in self.relation_names:
            query = self.query_projs[name](feature)
            weight = self.relation_weight(name)
            if self.normalize:
                query = F.normalize(query, dim=1)
                weight = F.normalize(weight, dim=1)
            relation_scores.append(query @ weight.t())
        stacked = torch.stack(relation_scores, dim=1)
        relation_gate = F.softmax(self.relation_gate(feature) / self.relation_gate_temperature, dim=1)
        residual = (stacked * relation_gate.unsqueeze(-1)).sum(dim=1)
        residual = self.scale * (residual + self.bias)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return uncertainty_gate.to(dtype=residual.dtype) * residual


class MobiusSheafFactorHead(nn.Module):
    """Möbius-style factor-interaction classifier for RSCD labels.

    RSCD labels are generated by partially observed factors rather than by 27
    unrelated symbols. This head decomposes class evidence into main effects,
    pairwise effects, and a paved-road triple interaction. It is designed as a
    primary classifier head for joint training, while the residual mode gives a
    cheap fail-fast screen from an existing checkpoint.
    """

    term_names = (
        "friction",
        "material",
        "unevenness",
        "friction_material",
        "friction_unevenness",
        "material_unevenness",
        "friction_material_unevenness",
    )

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        class_to_idx: dict[str, int],
        rank: int = 32,
        scale: float = 0.12,
        normalize: bool = True,
        zero_init: bool = True,
        use_triple: bool = True,
        gate_hidden_dim: int = 96,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError("MobiusSheafFactorHead rank must be positive.")
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.use_triple = bool(use_triple)
        self.gate_temperature = max(float(gate_temperature), 1e-4)
        gate_hidden_dim = max(int(gate_hidden_dim), 1)

        self.query_projs = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, rank, bias=False))
                for name in self.term_names
            }
        )
        self.effect_gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, len(self.term_names)),
        )
        self.single_embeddings = nn.ParameterDict(
            {
                "friction": nn.Parameter(torch.empty(len(FACTOR_LABELS["friction"]), rank)),
                "material": nn.Parameter(torch.empty(len(FACTOR_LABELS["material"]), rank)),
                "unevenness": nn.Parameter(torch.empty(len(FACTOR_LABELS["unevenness"]), rank)),
            }
        )
        pair_dims = {
            "friction_material": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["material"]),
            "friction_unevenness": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["unevenness"]),
            "material_unevenness": len(FACTOR_LABELS["material"]) * len(FACTOR_LABELS["unevenness"]),
        }
        self.pair_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(size, rank)) for name, size in pair_dims.items()}
        )
        self.triple_embedding = nn.Parameter(torch.empty(3 * 2 * 3, rank))
        self.bias = nn.Parameter(torch.zeros(num_classes))

        factor_buffers = build_class_factor_buffers(class_to_idx)
        for name, (indices, mask) in factor_buffers.items():
            self.register_buffer(f"{name}_class_factor_idx", indices)
            self.register_buffer(f"{name}_class_factor_mask", mask)

        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        pair_indices: dict[str, list[int]] = {name: [] for name in pair_dims}
        pair_masks: dict[str, list[float]] = {name: [] for name in pair_dims}
        triple_indices: list[int] = []
        triple_masks: list[float] = []
        friction_vocab = {"dry": 0, "wet": 1, "water": 2}
        material_vocab = {"asphalt": 0, "concrete": 1}
        unevenness_vocab = {"smooth": 0, "slight": 1, "severe": 2}
        for class_idx in range(num_classes):
            class_name = idx_to_class[class_idx]
            factors = parse_rscd_factors(class_name)
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_material",
                factors["friction"],
                factors["material"],
                len(FACTOR_LABELS["material"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_unevenness",
                factors["friction"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "material_unevenness",
                factors["material"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
            text = _factor_text(class_name)
            f_id = friction_vocab.get(str(text["friction"]), -1)
            m_id = material_vocab.get(str(text["material"]), -1)
            u_id = unevenness_vocab.get(str(text["unevenness"]), -1)
            if f_id >= 0 and m_id >= 0 and u_id >= 0:
                triple_indices.append(f_id * (2 * 3) + m_id * 3 + u_id)
                triple_masks.append(1.0)
            else:
                triple_indices.append(0)
                triple_masks.append(0.0)
        for name in pair_dims:
            self.register_buffer(f"{name}_idx", torch.tensor(pair_indices[name], dtype=torch.long))
            self.register_buffer(f"{name}_mask", torch.tensor(pair_masks[name], dtype=torch.float32))
        self.register_buffer("triple_idx", torch.tensor(triple_indices, dtype=torch.long))
        self.register_buffer("triple_mask", torch.tensor(triple_masks, dtype=torch.float32))

        embeddings = list(self.single_embeddings.values()) + list(self.pair_embeddings.values()) + [self.triple_embedding]
        for embedding in embeddings:
            if zero_init:
                nn.init.zeros_(embedding)
            else:
                nn.init.trunc_normal_(embedding, std=0.02)
        if zero_init:
            nn.init.zeros_(self.bias)
        else:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.effect_gate[-1].weight)
        nn.init.zeros_(self.effect_gate[-1].bias)

    @staticmethod
    def _center_rows(weight: torch.Tensor) -> torch.Tensor:
        return weight - weight.mean(dim=0, keepdim=True)

    def _single_weight(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self._center_rows(self.single_embeddings[name])
        factor_idx = getattr(self, f"{name}_class_factor_idx")
        factor_mask = getattr(self, f"{name}_class_factor_mask").to(device=embedding.device, dtype=embedding.dtype)
        return embedding[factor_idx], factor_mask

    def _pair_weight(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self._center_rows(self.pair_embeddings[name])
        pair_idx = getattr(self, f"{name}_idx")
        pair_mask = getattr(self, f"{name}_mask").to(device=embedding.device, dtype=embedding.dtype)
        return embedding[pair_idx], pair_mask

    def _triple_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self._center_rows(self.triple_embedding)
        mask = self.triple_mask.to(device=embedding.device, dtype=embedding.dtype)
        if not self.use_triple:
            mask = torch.zeros_like(mask)
        return embedding[self.triple_idx], mask

    def _term_weight(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name in self.single_embeddings:
            return self._single_weight(name)
        if name in self.pair_embeddings:
            return self._pair_weight(name)
        return self._triple_weight()

    def forward(self, feature: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        gates = torch.sigmoid(self.effect_gate(feature) / self.gate_temperature)
        logits = feature.new_zeros((feature.shape[0], self.bias.numel()))
        for term_idx, name in enumerate(self.term_names):
            query = self.query_projs[name](feature)
            weight, mask = self._term_weight(name)
            weight = weight.to(device=feature.device, dtype=feature.dtype)
            mask = mask.to(device=feature.device, dtype=feature.dtype).view(1, -1)
            if self.normalize:
                query = F.normalize(query, dim=1)
                weight = F.normalize(weight, dim=1)
            score = query @ weight.t()
            logits = logits + gates[:, term_idx : term_idx + 1].to(dtype=score.dtype) * score * mask
        logits = self.scale * (logits + self.bias.to(device=feature.device, dtype=feature.dtype))
        if not return_aux:
            return logits
        return {"logits": logits, "mobius_gates": gates}


class MechanismConditionalSheafHead(nn.Module):
    """Mechanism-conditioned factor sheaf residual for RSCD labels.

    RSCD hard classes mix multiple visual mechanisms: dry visible roughness,
    wet film reflection, water-film texture obstruction, granular loose
    material, and winter phase appearance. This head keeps the current
    ConvNeXt classifier as the base model and learns a zero-initialized
    residual whose factor decomposition and graph-edge correction are routed by
    physics evidence instead of shared across all labels.
    """

    term_names = (
        "friction",
        "material",
        "unevenness",
        "friction_material",
        "friction_unevenness",
        "material_unevenness",
        "friction_material_unevenness",
    )
    mechanism_names = ("dry_visible", "wet_film", "water_obstruction", "granular", "winter")

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        class_to_idx: dict[str, int],
        rank: int = 24,
        scale: float = 0.06,
        edge_scale: float = 0.04,
        router_hidden_dim: int = 96,
        edge_dim: int = 12,
        edge_hidden_dim: int = 64,
        class_scope: str = "all",
        normalize: bool = True,
        zero_init: bool = True,
        use_edge_flow: bool = True,
        protected_negative_limit: float = 0.0,
        sparse_router_topk: int = 0,
        router_temperature: float = 1.0,
        physics_prior_weight: float = 0.0,
    ) -> None:
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError("MechanismConditionalSheafHead rank must be positive.")
        self.scale = float(scale)
        self.edge_scale = float(edge_scale)
        self.normalize = bool(normalize)
        self.use_edge_flow = bool(use_edge_flow)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        self.sparse_router_topk = max(int(sparse_router_topk), 0)
        self.router_temperature = max(float(router_temperature), 1e-4)
        self.physics_prior_weight = max(float(physics_prior_weight), 0.0)
        self.class_scope = str(class_scope)
        if self.class_scope not in {"all", "asphalt_core", "wet_water_asphalt", "concrete_core"}:
            raise ValueError(f"unknown HMC-Sheaf++ class scope: {self.class_scope}")
        self.num_mechanisms = len(self.mechanism_names)
        self.num_stats = 40
        self.logit_context_dim = 8
        context_dim = int(in_dim) + self.num_stats + self.logit_context_dim
        router_hidden_dim = max(int(router_hidden_dim), 1)

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

        self.router = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, self.num_mechanisms),
        )
        self.effect_gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, self.num_mechanisms * len(self.term_names)),
        )
        self.query_projs = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, self.num_mechanisms * rank, bias=False))
                for name in self.term_names
            }
        )
        self.single_embeddings = nn.ParameterDict(
            {
                "friction": nn.Parameter(torch.empty(self.num_mechanisms, len(FACTOR_LABELS["friction"]), rank)),
                "material": nn.Parameter(torch.empty(self.num_mechanisms, len(FACTOR_LABELS["material"]), rank)),
                "unevenness": nn.Parameter(torch.empty(self.num_mechanisms, len(FACTOR_LABELS["unevenness"]), rank)),
            }
        )
        pair_dims = {
            "friction_material": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["material"]),
            "friction_unevenness": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["unevenness"]),
            "material_unevenness": len(FACTOR_LABELS["material"]) * len(FACTOR_LABELS["unevenness"]),
        }
        self.pair_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(self.num_mechanisms, size, rank)) for name, size in pair_dims.items()}
        )
        self.triple_embedding = nn.Parameter(torch.empty(self.num_mechanisms, 3 * 2 * 3, rank))
        self.bias = nn.Parameter(torch.zeros(self.num_mechanisms, num_classes))

        factor_buffers = build_class_factor_buffers(class_to_idx)
        for name, (indices, mask) in factor_buffers.items():
            self.register_buffer(f"{name}_class_factor_idx", indices)
            self.register_buffer(f"{name}_class_factor_mask", mask)

        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        pair_indices: dict[str, list[int]] = {name: [] for name in pair_dims}
        pair_masks: dict[str, list[float]] = {name: [] for name in pair_dims}
        triple_indices: list[int] = []
        triple_masks: list[float] = []
        friction_vocab = {"dry": 0, "wet": 1, "water": 2}
        material_vocab = {"asphalt": 0, "concrete": 1}
        unevenness_vocab = {"smooth": 0, "slight": 1, "severe": 2}
        dry_mask = torch.zeros(num_classes, dtype=torch.float32)
        wet_mask = torch.zeros(num_classes, dtype=torch.float32)
        water_mask = torch.zeros(num_classes, dtype=torch.float32)
        core_mask = torch.zeros(num_classes, dtype=torch.float32)
        rough_mask = torch.zeros(num_classes, dtype=torch.float32)
        scope_mask = torch.zeros(num_classes, dtype=torch.float32)
        protected = torch.zeros(num_classes, dtype=torch.bool)
        factor_texts: dict[int, dict[str, str | None]] = {}
        for class_idx in range(num_classes):
            class_name = idx_to_class[class_idx]
            factors = parse_rscd_factors(class_name)
            text = _factor_text(class_name)
            factor_texts[class_idx] = text
            if text["friction"] == "dry":
                dry_mask[class_idx] = 1.0
            if text["friction"] == "wet":
                wet_mask[class_idx] = 1.0
                protected[class_idx] = True
            if text["friction"] == "water":
                water_mask[class_idx] = 1.0
                protected[class_idx] = True
            if text["friction"] in {"dry", "wet", "water"} and text["material"] in {"asphalt", "concrete"}:
                core_mask[class_idx] = 1.0
            if text["unevenness"] in {"smooth", "slight", "severe"}:
                rough_mask[class_idx] = 1.0
            if self.class_scope == "all":
                scope_mask[class_idx] = 1.0
            elif self.class_scope == "asphalt_core":
                if text["friction"] in {"dry", "wet", "water"} and text["material"] == "asphalt":
                    scope_mask[class_idx] = 1.0
            elif self.class_scope == "wet_water_asphalt":
                if text["friction"] in {"wet", "water"} and text["material"] == "asphalt":
                    scope_mask[class_idx] = 1.0
            elif self.class_scope == "concrete_core":
                if text["friction"] in {"dry", "wet", "water"} and text["material"] == "concrete":
                    scope_mask[class_idx] = 1.0
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_material",
                factors["friction"],
                factors["material"],
                len(FACTOR_LABELS["material"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_unevenness",
                factors["friction"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "material_unevenness",
                factors["material"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
            f_id = friction_vocab.get(str(text["friction"]), -1)
            m_id = material_vocab.get(str(text["material"]), -1)
            u_id = unevenness_vocab.get(str(text["unevenness"]), -1)
            if f_id >= 0 and m_id >= 0 and u_id >= 0:
                triple_indices.append(f_id * (2 * 3) + m_id * 3 + u_id)
                triple_masks.append(1.0)
            else:
                triple_indices.append(0)
                triple_masks.append(0.0)
        for name in pair_dims:
            self.register_buffer(f"{name}_idx", torch.tensor(pair_indices[name], dtype=torch.long))
            self.register_buffer(f"{name}_mask", torch.tensor(pair_masks[name], dtype=torch.float32))
        self.register_buffer("triple_idx", torch.tensor(triple_indices, dtype=torch.long))
        self.register_buffer("triple_mask", torch.tensor(triple_masks, dtype=torch.float32))
        self.register_buffer("dry_class_mask", dry_mask.view(1, -1))
        self.register_buffer("wet_class_mask", wet_mask.view(1, -1))
        self.register_buffer("water_class_mask", water_mask.view(1, -1))
        self.register_buffer("core_class_mask", core_mask.view(1, -1))
        self.register_buffer("rough_class_mask", rough_mask.view(1, -1))
        self.register_buffer("scope_class_mask", scope_mask.view(1, -1))
        self.register_buffer("protected_class_mask", protected.view(1, -1))

        edge_left: list[int] = []
        edge_right: list[int] = []
        edge_relation: list[int] = []
        edge_weight: list[float] = []
        for i in range(num_classes):
            a = factor_texts[i]
            for j in range(i + 1, num_classes):
                if scope_mask[i] <= 0.0 or scope_mask[j] <= 0.0:
                    continue
                b = factor_texts[j]
                same_friction = a["friction"] is not None and a["friction"] == b["friction"]
                same_material = a["material"] is not None and a["material"] == b["material"]
                same_unevenness = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
                relation = -1
                weight = 1.0
                if same_material and same_unevenness and _friction_neighbors(a["friction"], b["friction"]):
                    relation = 0
                    weight = 1.15
                elif same_friction and same_material and _unevenness_neighbors(a["unevenness"], b["unevenness"]):
                    relation = 1
                    weight = 1.45
                elif same_friction and same_unevenness and _material_neighbors(a["material"], b["material"]):
                    relation = 2
                    weight = 0.90
                if relation >= 0:
                    edge_left.append(i)
                    edge_right.append(j)
                    edge_relation.append(relation)
                    edge_weight.append(weight)
        self.register_buffer("edge_left", torch.tensor(edge_left, dtype=torch.long))
        self.register_buffer("edge_right", torch.tensor(edge_right, dtype=torch.long))
        self.register_buffer("edge_relation", torch.tensor(edge_relation, dtype=torch.long))
        self.register_buffer("edge_weight", torch.tensor(edge_weight, dtype=torch.float32))
        self.edge_embedding = nn.Embedding(3, int(edge_dim))
        self.edge_head = nn.Sequential(
            nn.LayerNorm(context_dim + int(edge_dim) + 8),
            nn.Linear(context_dim + int(edge_dim) + 8, int(edge_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(edge_hidden_dim), 1),
        )

        embeddings = list(self.single_embeddings.values()) + list(self.pair_embeddings.values()) + [self.triple_embedding]
        for embedding in embeddings:
            if zero_init:
                nn.init.zeros_(embedding)
            else:
                nn.init.trunc_normal_(embedding, std=0.02)
        if zero_init:
            nn.init.zeros_(self.bias)
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
            nn.init.zeros_(self.effect_gate[-1].weight)
            nn.init.zeros_(self.effect_gate[-1].bias)
            nn.init.zeros_(self.edge_head[-1].weight)
            nn.init.zeros_(self.edge_head[-1].bias)
        else:
            nn.init.zeros_(self.bias)

    @staticmethod
    def _top_fraction_mean(x: torch.Tensor, fraction: float) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(1, int(flat.size(1) * float(fraction)))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    @staticmethod
    def _center_mechanism_rows(weight: torch.Tensor) -> torch.Tensor:
        return weight - weight.mean(dim=1, keepdim=True)

    def _field_summary(self, field: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                field.mean(dim=(2, 3)),
                self._top_fraction_mean(field, 0.15),
                _soft_connectedness(field),
            ],
            dim=1,
        )

    def _physics_stats(self, image: torch.Tensor) -> torch.Tensor:
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
        granular = rough_energy * torch.sigmoid((local_contrast - 0.035) * 35.0)
        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        obstruction = torch.clamp(0.45 * thin_film + 0.30 * dark_water + 0.20 * specular + 0.35 * texture_erasure, 0.0, 1.0)
        visible_rough = rough_energy * (1.0 - obstruction) * (1.0 - snow_like) * (1.0 - marking)
        hidden_rough = rough_energy * obstruction
        global_stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3)),
            ],
            dim=1,
        )
        field_stats = [
            self._field_summary(specular),
            self._field_summary(dark_water),
            self._field_summary(thin_film),
            self._field_summary(texture_erasure),
            self._field_summary(rough_energy),
            self._field_summary(granular),
            self._field_summary(snow_like),
            self._field_summary(marking),
            self._field_summary(visible_rough),
            self._field_summary(hidden_rough),
        ]
        return torch.cat([global_stats, *field_stats], dim=1)

    def _logit_context(self, base_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(base_logits, dim=1)
        top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        top_prob = top2[:, 0:1]
        if top2.shape[1] < 2:
            margin = probs.new_zeros((probs.shape[0], 1))
        else:
            margin = top2[:, 0:1] - top2[:, 1:2]
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / math.log(float(probs.shape[1]))
        dry_mass = (probs * self.dry_class_mask.to(device=probs.device, dtype=probs.dtype)).sum(dim=1, keepdim=True)
        wet_mass = (probs * self.wet_class_mask.to(device=probs.device, dtype=probs.dtype)).sum(dim=1, keepdim=True)
        water_mass = (probs * self.water_class_mask.to(device=probs.device, dtype=probs.dtype)).sum(dim=1, keepdim=True)
        core_mass = (probs * self.core_class_mask.to(device=probs.device, dtype=probs.dtype)).sum(dim=1, keepdim=True)
        rough_probs = probs * self.rough_class_mask.to(device=probs.device, dtype=probs.dtype)
        rough_mass = rough_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        rough_dist = rough_probs / rough_mass
        rough_entropy = -(rough_dist * rough_dist.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / math.log(float(probs.shape[1]))
        return torch.cat([entropy, margin, top_prob, dry_mass, wet_mass, water_mass, core_mass, rough_entropy], dim=1)

    def _physics_router_prior(self, stats: torch.Tensor) -> torch.Tensor:
        """Evidence-seeded prior for sparse mechanism routing.

        The learned router is zero-initialized for no-harm checkpoint screens.
        A hard sparse router would otherwise always select the first mechanism
        before gradients have separated modes. These logits provide a weak
        physics-chart prior: visible rough texture, wet film, water obstruction,
        granular texture, and winter-like appearance compete before the learned
        router fine-tunes the decision.
        """

        if self.physics_prior_weight <= 0.0:
            return stats.new_zeros((stats.shape[0], self.num_mechanisms))
        specular = 0.55 * stats[:, 10:11] + 0.45 * stats[:, 11:12]
        dark_water = 0.55 * stats[:, 13:14] + 0.45 * stats[:, 14:15]
        thin_film = 0.55 * stats[:, 16:17] + 0.45 * stats[:, 17:18]
        texture_erasure = 0.55 * stats[:, 19:20] + 0.45 * stats[:, 20:21]
        rough_energy = 0.55 * stats[:, 22:23] + 0.45 * stats[:, 23:24]
        granular = 0.55 * stats[:, 25:26] + 0.45 * stats[:, 26:27]
        snow_like = 0.55 * stats[:, 28:29] + 0.45 * stats[:, 29:30]
        visible_rough = 0.55 * stats[:, 34:35] + 0.45 * stats[:, 35:36]
        hidden_rough = 0.55 * stats[:, 37:38] + 0.45 * stats[:, 38:39]
        obstruction = torch.clamp(thin_film + 0.6 * dark_water + 0.5 * texture_erasure, 0.0, 2.0)
        prior = torch.cat(
            [
                1.25 * visible_rough + 0.35 * rough_energy - 0.50 * obstruction,
                0.75 * specular + 0.85 * thin_film + 0.45 * texture_erasure,
                0.75 * dark_water + 0.65 * thin_film + 0.85 * hidden_rough,
                1.20 * granular + 0.30 * visible_rough,
                1.25 * snow_like,
            ],
            dim=1,
        )
        return float(self.physics_prior_weight) * prior

    def _route_mechanisms(self, context: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        logits = self.router(context) + self._physics_router_prior(stats).to(device=context.device, dtype=context.dtype)
        probs = F.softmax(logits / self.router_temperature, dim=1)
        topk = self.sparse_router_topk
        if topk <= 0 or topk >= self.num_mechanisms:
            return probs
        _, indices = torch.topk(probs, k=topk, dim=1)
        mask = torch.zeros_like(probs).scatter_(1, indices, 1.0)
        sparse = probs * mask
        return sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _term_weight(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name in self.single_embeddings:
            embedding = self._center_mechanism_rows(self.single_embeddings[name])
            factor_idx = getattr(self, f"{name}_class_factor_idx")
            factor_mask = getattr(self, f"{name}_class_factor_mask").to(device=embedding.device, dtype=embedding.dtype)
            return embedding[:, factor_idx], factor_mask
        if name in self.pair_embeddings:
            embedding = self._center_mechanism_rows(self.pair_embeddings[name])
            pair_idx = getattr(self, f"{name}_idx")
            pair_mask = getattr(self, f"{name}_mask").to(device=embedding.device, dtype=embedding.dtype)
            return embedding[:, pair_idx], pair_mask
        embedding = self._center_mechanism_rows(self.triple_embedding)
        mask = self.triple_mask.to(device=embedding.device, dtype=embedding.dtype)
        return embedding[:, self.triple_idx], mask

    def _factor_residual(self, context: torch.Tensor, router: torch.Tensor, effect_gate: torch.Tensor) -> torch.Tensor:
        logits = context.new_zeros((context.shape[0], self.bias.shape[1]))
        for term_idx, name in enumerate(self.term_names):
            query = self.query_projs[name](context).view(context.shape[0], self.num_mechanisms, -1)
            weight, mask = self._term_weight(name)
            weight = weight.to(device=context.device, dtype=context.dtype)
            mask = mask.to(device=context.device, dtype=context.dtype).view(1, 1, -1)
            if self.normalize:
                query = F.normalize(query, dim=2)
                weight = F.normalize(weight, dim=2)
            score = torch.einsum("brd,rcd->brc", query, weight) * mask
            weighted = router.unsqueeze(-1) * effect_gate[:, :, term_idx : term_idx + 1] * score
            logits = logits + weighted.sum(dim=1)
        bias = (router.unsqueeze(-1) * self.bias.to(device=context.device, dtype=context.dtype).unsqueeze(0)).sum(dim=1)
        scope = self.scope_class_mask.to(device=context.device, dtype=context.dtype)
        return self.scale * (logits + bias) * scope

    def _edge_flow_residual(self, context: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        if (not self.use_edge_flow) or self.edge_left.numel() == 0 or self.edge_scale == 0.0:
            return torch.zeros_like(base_logits)
        left = self.edge_left.to(device=base_logits.device)
        right = self.edge_right.to(device=base_logits.device)
        relation = self.edge_relation.to(device=base_logits.device)
        edge_weight = self.edge_weight.to(device=base_logits.device, dtype=base_logits.dtype)
        probs = F.softmax(base_logits, dim=1)
        z_left = base_logits.index_select(1, left)
        z_right = base_logits.index_select(1, right)
        p_left = probs.index_select(1, left)
        p_right = probs.index_select(1, right)
        pair_mass = p_left + p_right
        pair_gap = (p_left - p_right).abs()
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / math.log(float(probs.shape[1]))
        edge_logit_features = torch.stack(
            [
                z_left,
                z_right,
                p_left,
                p_right,
                z_right - z_left,
                p_right - p_left,
                pair_mass,
                pair_gap,
            ],
            dim=2,
        )
        edge_emb = self.edge_embedding(relation).unsqueeze(0).expand(base_logits.shape[0], -1, -1)
        expanded_context = context.unsqueeze(1).expand(-1, left.numel(), -1)
        edge_input = torch.cat(
            [expanded_context, edge_emb.to(dtype=context.dtype), edge_logit_features.to(dtype=context.dtype)],
            dim=2,
        )
        raw_delta = torch.tanh(self.edge_head(edge_input).squeeze(2))
        uncertainty_gate = torch.sigmoid((entropy - 0.10) * 10.0)
        mass_gate = torch.sigmoid((pair_mass - 0.045) * 18.0)
        ambiguity_gate = torch.sigmoid((0.38 - pair_gap) * 10.0)
        delta = (
            raw_delta
            * mass_gate.to(dtype=raw_delta.dtype)
            * ambiguity_gate.to(dtype=raw_delta.dtype)
            * uncertainty_gate.to(dtype=raw_delta.dtype)
            * edge_weight.view(1, -1).to(dtype=raw_delta.dtype)
            * self.edge_scale
        )
        residual = torch.zeros_like(base_logits)
        residual.scatter_add_(1, left.view(1, -1).expand(base_logits.shape[0], -1), -delta.to(dtype=residual.dtype))
        residual.scatter_add_(1, right.view(1, -1).expand(base_logits.shape[0], -1), delta.to(dtype=residual.dtype))
        return residual * self.scope_class_mask.to(device=residual.device, dtype=residual.dtype)

    def forward(
        self,
        image: torch.Tensor,
        feature: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        stats = self._physics_stats(image).to(device=feature.device, dtype=feature.dtype)
        logit_context = self._logit_context(base_logits).to(device=feature.device, dtype=feature.dtype)
        context = torch.cat([feature, stats, logit_context], dim=1)
        router = self._route_mechanisms(context, stats)
        effect_gate = torch.sigmoid(self.effect_gate(context)).view(
            context.shape[0],
            self.num_mechanisms,
            len(self.term_names),
        )
        residual = self._factor_residual(context, router, effect_gate)
        residual = residual + self._edge_flow_residual(context, base_logits)
        if self.protected_negative_limit > 0.0 and bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        if not return_aux:
            return residual
        return {
            "logits": residual,
            "hmc_sheaf_router": router,
            "hmc_sheaf_effect_gate": effect_gate,
        }


class LocalGlobalFactorAttentionResidual(nn.Module):
    """Dynamic low-rank residual for local-global RSCD factor disentanglement.

    The residual decomposes every class into first-order factors
    (friction/material/unevenness) and pairwise coupled factors. A small
    attention gate, conditioned by the global image feature and the local
    physics field, decides which factor family should be trusted for each
    sample. This keeps the module compact while directly targeting RSCD's
    coupled-label failure mode.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        local_dim: int,
        num_classes: int,
        class_to_idx: dict[str, int],
        rank: int = 48,
        scale: float = 0.08,
        normalize: bool = True,
        zero_init: bool = True,
        gate_threshold: float = 0.35,
        gate_temperature: float = 10.0,
        neighbor_gate_floor: float = 0.15,
        protected_negative_limit: float = 0.0,
        protected_friction: tuple[str, ...] = ("wet", "water"),
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.local_dim = max(0, int(local_dim))
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.neighbor_gate_floor = float(neighbor_gate_floor)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        rank = int(rank)
        if rank <= 0:
            raise ValueError("local-global factor attention rank must be positive.")

        self.feature_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, rank, bias=False))
        self.local_proj = (
            nn.Sequential(nn.LayerNorm(self.local_dim), nn.Linear(self.local_dim, rank, bias=False))
            if self.local_dim > 0
            else None
        )
        context_dim = int(in_dim) + self.local_dim
        hidden_dim = max(32, min(128, rank * 2))
        self.factor_attention = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        self.single_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(len(labels), rank)) for name, labels in FACTOR_LABELS.items()}
        )
        pair_dims = {
            "friction_material": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["material"]),
            "friction_unevenness": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["unevenness"]),
            "material_unevenness": len(FACTOR_LABELS["material"]) * len(FACTOR_LABELS["unevenness"]),
        }
        self.pair_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(size, rank)) for name, size in pair_dims.items()}
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

        factor_buffers = build_class_factor_buffers(class_to_idx)
        for name, (indices, mask) in factor_buffers.items():
            self.register_buffer(f"{name}_class_factor_idx", indices)
            self.register_buffer(f"{name}_class_factor_mask", mask)

        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        pair_indices: dict[str, list[int]] = {name: [] for name in pair_dims}
        pair_masks: dict[str, list[float]] = {name: [] for name in pair_dims}
        protected = torch.zeros(num_classes, dtype=torch.bool)
        hard_neighbor = torch.zeros((num_classes, num_classes), dtype=torch.bool)
        protected_set = {str(item) for item in protected_friction}
        factor_texts: dict[int, dict[str, str | None]] = {}
        for class_idx in range(num_classes):
            class_name = idx_to_class[class_idx]
            factors = parse_rscd_factors(class_name)
            f_idx = factors["friction"]
            m_idx = factors["material"]
            u_idx = factors["unevenness"]
            factor_texts[class_idx] = _factor_text(class_name)
            if factor_texts[class_idx]["friction"] in protected_set:
                protected[class_idx] = True
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_material",
                f_idx,
                m_idx,
                len(FACTOR_LABELS["material"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_unevenness",
                f_idx,
                u_idx,
                len(FACTOR_LABELS["unevenness"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "material_unevenness",
                m_idx,
                u_idx,
                len(FACTOR_LABELS["unevenness"]),
            )
        for i in range(num_classes):
            a = factor_texts[i]
            for j in range(num_classes):
                if i == j:
                    continue
                b = factor_texts[j]
                same_friction = a["friction"] is not None and a["friction"] == b["friction"]
                same_material = a["material"] is not None and a["material"] == b["material"]
                same_unevenness = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
                factor_neighbor = (
                    (same_material and same_unevenness and _friction_neighbors(a["friction"], b["friction"]))
                    or (same_friction and same_material and _unevenness_neighbors(a["unevenness"], b["unevenness"]))
                    or (same_friction and same_unevenness and _material_neighbors(a["material"], b["material"]))
                )
                hard_neighbor[i, j] = bool(factor_neighbor)
        for name in pair_dims:
            self.register_buffer(f"{name}_idx", torch.tensor(pair_indices[name], dtype=torch.long))
            self.register_buffer(f"{name}_mask", torch.tensor(pair_masks[name], dtype=torch.float32))
        self.register_buffer("protected_class_mask", protected.view(1, -1))
        self.register_buffer("hard_neighbor_matrix", hard_neighbor)

        for embedding in list(self.single_embeddings.values()) + list(self.pair_embeddings.values()):
            if zero_init:
                nn.init.zeros_(embedding)
            else:
                nn.init.trunc_normal_(embedding, std=0.02)
        nn.init.zeros_(self.factor_attention[-1].weight)
        nn.init.zeros_(self.factor_attention[-1].bias)

    def _class_weight(self, attention: torch.Tensor) -> torch.Tensor:
        first_embedding = next(iter(self.single_embeddings.values()))
        bsz = attention.shape[0]
        weight = torch.zeros(
            (bsz, self.bias.numel(), first_embedding.shape[1]),
            device=attention.device,
            dtype=attention.dtype,
        )
        active = torch.zeros((bsz, self.bias.numel(), 1), device=attention.device, dtype=attention.dtype)
        offset = 0
        for name, embedding in self.single_embeddings.items():
            factor_idx = getattr(self, f"{name}_class_factor_idx")
            factor_mask = getattr(self, f"{name}_class_factor_mask").to(device=attention.device, dtype=attention.dtype)
            term = embedding[factor_idx].to(dtype=attention.dtype) * factor_mask.view(1, -1, 1)
            alpha = attention[:, offset].view(bsz, 1, 1)
            weight = weight + alpha * term
            active = active + alpha * factor_mask.view(1, -1, 1)
            offset += 1
        for name, embedding in self.pair_embeddings.items():
            pair_idx = getattr(self, f"{name}_idx")
            pair_mask = getattr(self, f"{name}_mask").to(device=attention.device, dtype=attention.dtype)
            term = embedding[pair_idx].to(dtype=attention.dtype) * pair_mask.view(1, -1, 1)
            alpha = attention[:, offset].view(bsz, 1, 1)
            weight = weight + alpha * term
            active = active + alpha * pair_mask.view(1, -1, 1)
            offset += 1
        return weight / active.clamp_min(1e-4)

    def forward(
        self,
        feature: torch.Tensor,
        base_logits: torch.Tensor,
        local_physics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.feature_proj(feature)
        if self.local_proj is not None:
            if local_physics is None:
                local_physics = feature.new_zeros((feature.shape[0], self.local_dim))
            local_query = self.local_proj(local_physics.to(dtype=feature.dtype))
            query = query + local_query
            context = torch.cat([feature, local_physics.to(dtype=feature.dtype)], dim=1)
        else:
            context = feature
        attention = torch.softmax(self.factor_attention(context), dim=1)
        weight = self._class_weight(attention)
        if self.normalize:
            query = F.normalize(query, dim=1)
            weight = F.normalize(weight, dim=2)
        residual = self.scale * (torch.einsum("br,bcr->bc", query, weight) + self.bias)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            top2 = base_logits.topk(k=2, dim=1).indices
            neighbor = self.hard_neighbor_matrix.to(device=base_logits.device)[top2[:, 0], top2[:, 1]].view(-1, 1)
            neighbor_gate = torch.where(
                neighbor,
                torch.ones_like(uncertainty_gate),
                uncertainty_gate.new_full(uncertainty_gate.shape, self.neighbor_gate_floor),
            )
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return uncertainty_gate.to(dtype=residual.dtype) * neighbor_gate.to(dtype=residual.dtype) * residual


class LabelGraphPrototypeResidual(nn.Module):
    """GCN-style residual over the RSCD composite label graph.

    RSCD classes are not independent symbols: they form a small factor graph
    over friction, material, and unevenness. This module builds class prototypes
    by propagating factor-node descriptors through that graph, then applies a
    weak uncertainty-gated residual to the main classifier logits.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        rank: int = 32,
        scale: float = 0.04,
        gate_threshold: float = 0.45,
        gate_temperature: float = 10.0,
        neighbor_gate_floor: float = 0.10,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.neighbor_gate_floor = float(neighbor_gate_floor)
        rank = int(rank)
        if rank <= 0:
            raise ValueError("label graph residual rank must be positive.")
        self.query = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, rank, bias=False))
        node_features, adjacency, neighbor_mask = build_label_graph_buffers(class_to_idx)
        self.register_buffer("node_features", node_features)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("neighbor_mask", neighbor_mask)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features.shape[1], rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, rank, bias=False),
        )
        self.output = nn.Linear(rank, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(len(class_to_idx)))
        if zero_init:
            nn.init.zeros_(self.output.weight)
        else:
            nn.init.trunc_normal_(self.output.weight, std=0.02)

    def class_prototypes(self) -> torch.Tensor:
        nodes = self.node_encoder(self.node_features.to(dtype=self.output.weight.dtype))
        propagated = self.adjacency.to(dtype=nodes.dtype) @ nodes
        return self.output(propagated)

    def forward(self, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.query(feature), dim=1)
        weight = F.normalize(self.class_prototypes(), dim=1)
        residual = self.scale * (query @ weight.t() + self.bias)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            top2 = torch.topk(prob, k=2, dim=1).indices
            c1 = top2[:, 0]
            c2 = top2[:, 1]
            is_neighbor = self.neighbor_mask.to(device=base_logits.device)[c1, c2].view(-1, 1)
            neighbor_gate = torch.where(
                is_neighbor,
                uncertainty_gate.new_ones(uncertainty_gate.shape),
                uncertainty_gate.new_full(uncertainty_gate.shape, self.neighbor_gate_floor),
            )
        return uncertainty_gate.to(dtype=residual.dtype) * neighbor_gate.to(dtype=residual.dtype) * residual


class ConditionalEvidenceMaskedCouplingField(nn.Module):
    """Physics-mask-pooled coupling residual over RSCD factor relations.

    Random input masking was harmful because it deletes friction evidence. This
    module instead uses deterministic soft masks as evidence selectors: snow,
    water film, dark water, texture erasure, rough aggregate, granular texture,
    and marking-like artifacts pool the ConvNeXt feature map into tokens. Three
    relation-specific queries then recouple friction-material, friction-
    roughness, and material-roughness evidence without graph smoothing.
    """

    relation_names = ("friction_material", "friction_unevenness", "material_unevenness")

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        feature_map_dim: int = 768,
        token_dim: int = 96,
        rank: int = 32,
        scale: float = 0.04,
        normalize: bool = True,
        gate_threshold: float = 0.35,
        gate_temperature: float = 10.0,
        neighbor_gate_floor: float = 0.05,
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.neighbor_gate_floor = float(neighbor_gate_floor)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        self.num_masks = 9

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

        token_dim = int(token_dim)
        rank = int(rank)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(int(feature_map_dim)),
            nn.Linear(int(feature_map_dim), token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.relation_query = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), token_dim),
            nn.GELU(),
            nn.Linear(token_dim, len(self.relation_names) * token_dim),
        )
        self.relation_proj = nn.ModuleDict(
            {name: nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, rank, bias=False)) for name in self.relation_names}
        )
        pair_dims = {
            "friction_material": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["material"]),
            "friction_unevenness": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["unevenness"]),
            "material_unevenness": len(FACTOR_LABELS["material"]) * len(FACTOR_LABELS["unevenness"]),
        }
        self.pair_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(size, rank)) for name, size in pair_dims.items()}
        )
        self.bias = nn.Parameter(torch.zeros(len(class_to_idx)))
        pair_indices: dict[str, list[int]] = {name: [] for name in pair_dims}
        pair_masks: dict[str, list[float]] = {name: [] for name in pair_dims}
        protected = torch.zeros(len(class_to_idx), dtype=torch.bool)
        for class_name, class_idx in sorted(class_to_idx.items(), key=lambda item: item[1]):
            factors = parse_rscd_factors(class_name)
            text = _factor_text(class_name)
            if text["friction"] in {"wet", "water"}:
                protected[int(class_idx)] = True
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_material",
                factors["friction"],
                factors["material"],
                len(FACTOR_LABELS["material"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_unevenness",
                factors["friction"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "material_unevenness",
                factors["material"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
        for name in self.relation_names:
            self.register_buffer(f"{name}_idx", torch.tensor(pair_indices[name], dtype=torch.long))
            self.register_buffer(f"{name}_mask", torch.tensor(pair_masks[name], dtype=torch.float32))
        self.register_buffer("protected_class_mask", protected.view(1, -1))
        self.register_buffer(
            "neighbor_mask",
            build_factor_neighbor_negative_mask(class_to_idx, torch.device("cpu"), core_only=False),
        )
        for embedding in self.pair_embeddings.values():
            if zero_init:
                nn.init.zeros_(embedding)
            else:
                nn.init.trunc_normal_(embedding, std=0.02)

    def relation_weight(self, name: str) -> torch.Tensor:
        embedding = self.pair_embeddings[name]
        pair_idx = getattr(self, f"{name}_idx")
        pair_mask = getattr(self, f"{name}_mask").to(device=embedding.device, dtype=embedding.dtype)
        return embedding[pair_idx] * pair_mask.unsqueeze(1)

    def forward(
        self,
        image: torch.Tensor,
        feature_map: torch.Tensor | None,
        feature: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        if feature_map is None or feature_map.ndim != 4:
            return base_logits.new_zeros(base_logits.shape)
        masks = self._evidence_masks(image, feature_map.shape[-2:]).to(device=feature_map.device, dtype=feature_map.dtype)
        mass = masks.flatten(2).sum(dim=2).clamp_min(1e-4)
        pooled = (feature_map.unsqueeze(1) * masks.unsqueeze(2)).flatten(3).sum(dim=3) / mass.unsqueeze(2)
        tokens = self.token_proj(pooled)
        relation_queries = self.relation_query(feature).view(feature.shape[0], len(self.relation_names), -1)
        attention = torch.softmax(
            torch.matmul(F.normalize(relation_queries, dim=2), F.normalize(tokens, dim=2).transpose(1, 2))
            / math.sqrt(float(tokens.shape[-1])),
            dim=2,
        )
        relation_tokens = torch.matmul(attention, tokens)
        relation_scores = []
        for idx, name in enumerate(self.relation_names):
            query = self.relation_proj[name](relation_tokens[:, idx])
            weight = self.relation_weight(name)
            if self.normalize:
                query = F.normalize(query, dim=1)
                weight = F.normalize(weight, dim=1)
            relation_scores.append(query @ weight.t())
        residual = torch.stack(relation_scores, dim=1).sum(dim=1)
        residual = self.scale * (residual + self.bias)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            top2 = torch.topk(prob, k=2, dim=1).indices
            neighbor = self.neighbor_mask.to(device=base_logits.device)[top2[:, 0], top2[:, 1]].view(-1, 1)
            neighbor_gate = torch.where(
                neighbor,
                uncertainty_gate.new_ones(uncertainty_gate.shape),
                uncertainty_gate.new_full(uncertainty_gate.shape, self.neighbor_gate_floor),
            )
        return uncertainty_gate.to(dtype=residual.dtype) * neighbor_gate.to(dtype=residual.dtype) * residual

    def _evidence_masks(self, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
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
        dry_microtexture = rough_aggregate * (1.0 - torch.clamp(mirror_water + dark_water + snow, 0.0, 1.0))
        masks = torch.cat(
            [
                snow,
                mirror_water,
                dark_water,
                thin_film,
                texture_erasure,
                rough_aggregate,
                granular,
                marking,
                dry_microtexture,
            ],
            dim=1,
        )
        return F.interpolate(masks, size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


class FullOrderCouplingTensorField(nn.Module):
    """ANOVA-style full third-order coupling tensor for core RSCD labels.

    This module implements a task-specific decomposition for the 18 paved-road
    labels dry/wet/water x asphalt/concrete x smooth/slight/severe:

        Z_fmr = A_f + B_m + C_r + D_fm + E_fr + G_mr + H_fmr.

    The first- and second-order terms are explicitly centered, while H is a
    full non-low-rank 3 x 2 x 3 tensor with zero marginal sums. H therefore
    cannot collapse into the easier main or pairwise effects. Each H_fmr term
    is scored from a class-specific PhysicsTexture-masked ConvNeXt token, so
    wet-concrete-slight, water-asphalt-smooth, and dry-concrete-severe are not
    forced through one shared visual extractor.
    """

    friction_values = ("dry", "wet", "water")
    material_values = ("asphalt", "concrete")
    roughness_values = ("smooth", "slight", "severe")

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        feature_map_dim: int = 768,
        token_dim: int = 96,
        hidden_dim: int = 96,
        scale: float = 0.05,
        gate_threshold: float = 0.35,
        gate_temperature: float = 10.0,
        core_gate_floor: float = 0.05,
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.core_gate_floor = float(core_gate_floor)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        self.num_masks = 12
        self.num_core = len(self.friction_values) * len(self.material_values) * len(self.roughness_values)

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

        self.factor_head = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3 + 2 + 3 + 3 * 2 + 3 * 3 + 2 * 3),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(int(feature_map_dim)),
            nn.Linear(int(feature_map_dim), int(token_dim)),
            nn.GELU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.triple_weight = nn.Parameter(torch.empty(self.num_core, int(token_dim)))
        self.triple_bias = nn.Parameter(torch.zeros(self.num_core))

        core_to_class = torch.full((self.num_core,), -1, dtype=torch.long)
        class_to_core = torch.full((len(class_to_idx),), -1, dtype=torch.long)
        protected = torch.zeros(len(class_to_idx), dtype=torch.bool)
        core_class_mask = torch.zeros(len(class_to_idx), dtype=torch.float32)
        for f_idx, friction in enumerate(self.friction_values):
            for m_idx, material in enumerate(self.material_values):
                for r_idx, roughness in enumerate(self.roughness_values):
                    core_idx = self._core_index(f_idx, m_idx, r_idx)
                    label = f"{friction}_{material}_{roughness}"
                    if label not in class_to_idx:
                        continue
                    class_idx = int(class_to_idx[label])
                    core_to_class[core_idx] = class_idx
                    class_to_core[class_idx] = core_idx
                    core_class_mask[class_idx] = 1.0
                    if friction in {"wet", "water"}:
                        protected[class_idx] = True
        self.register_buffer("core_to_class", core_to_class)
        self.register_buffer("class_to_core", class_to_core)
        self.register_buffer("core_class_mask", core_class_mask.view(1, -1))
        self.register_buffer("protected_class_mask", protected.view(1, -1))
        self.register_buffer("core_evidence_weights", self._build_core_evidence_weights())

        if zero_init:
            last = self.factor_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            nn.init.zeros_(self.triple_weight)
            nn.init.zeros_(self.triple_bias)
        else:
            nn.init.trunc_normal_(self.triple_weight, std=0.02)
            nn.init.zeros_(self.triple_bias)

    @classmethod
    def _core_index(cls, f_idx: int, m_idx: int, r_idx: int) -> int:
        return int(f_idx) * (len(cls.material_values) * len(cls.roughness_values)) + int(m_idx) * len(cls.roughness_values) + int(r_idx)

    def _build_core_evidence_weights(self) -> torch.Tensor:
        weights = torch.zeros(self.num_core, self.num_masks, dtype=torch.float32)
        mask_id = {
            "dry_micro": 0,
            "visible_rough": 1,
            "rough_aggregate": 2,
            "thin_film": 3,
            "specular": 4,
            "dark_water": 5,
            "texture_erasure": 6,
            "hidden_rough": 7,
            "granular": 8,
            "marking": 9,
            "asphalt_proxy": 10,
            "concrete_proxy": 11,
        }
        for f_idx, friction in enumerate(self.friction_values):
            for m_idx, material in enumerate(self.material_values):
                for r_idx, roughness in enumerate(self.roughness_values):
                    core_idx = self._core_index(f_idx, m_idx, r_idx)
                    row = weights[core_idx]
                    if friction == "dry":
                        row[mask_id["dry_micro"]] += 1.2
                        row[mask_id["visible_rough"]] += 1.0
                        row[mask_id["rough_aggregate"]] += 0.5
                    elif friction == "wet":
                        row[mask_id["thin_film"]] += 1.1
                        row[mask_id["specular"]] += 0.9
                        row[mask_id["texture_erasure"]] += 0.8
                    else:
                        row[mask_id["dark_water"]] += 1.0
                        row[mask_id["thin_film"]] += 0.8
                        row[mask_id["hidden_rough"]] += 1.0
                        row[mask_id["texture_erasure"]] += 0.7

                    if material == "asphalt":
                        row[mask_id["asphalt_proxy"]] += 0.8
                        if friction in {"wet", "water"}:
                            row[mask_id["specular"]] += 0.4
                    else:
                        row[mask_id["concrete_proxy"]] += 0.8
                        if friction in {"wet", "water"}:
                            row[mask_id["hidden_rough"]] += 0.5
                            row[mask_id["texture_erasure"]] += 0.3

                    if roughness == "smooth":
                        row[mask_id["texture_erasure"]] += 0.6
                        row[mask_id["thin_film"]] += 0.3
                    elif roughness == "slight":
                        row[mask_id["visible_rough"]] += 0.4
                        row[mask_id["hidden_rough"]] += 0.4
                        row[mask_id["rough_aggregate"]] += 0.5
                    else:
                        row[mask_id["rough_aggregate"]] += 0.9
                        row[mask_id["hidden_rough"]] += 0.5
                        row[mask_id["granular"]] += 0.2
                    row[mask_id["marking"]] -= 0.4
        weights = weights.clamp_min(0.0)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _center_1d(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=1, keepdim=True)

    @staticmethod
    def _center_2d(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=1, keepdim=True) - x.mean(dim=2, keepdim=True) + x.mean(dim=(1, 2), keepdim=True)

    @staticmethod
    def _center_3d(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=1, keepdim=True)
        x = x - x.mean(dim=2, keepdim=True)
        x = x - x.mean(dim=3, keepdim=True)
        return x

    def _factor_tensor(self, feature: torch.Tensor) -> torch.Tensor:
        raw = self.factor_head(feature)
        cursor = 0
        a = self._center_1d(raw[:, cursor : cursor + 3])
        cursor += 3
        b = self._center_1d(raw[:, cursor : cursor + 2])
        cursor += 2
        c = self._center_1d(raw[:, cursor : cursor + 3])
        cursor += 3
        d = self._center_2d(raw[:, cursor : cursor + 6].view(-1, 3, 2))
        cursor += 6
        e = self._center_2d(raw[:, cursor : cursor + 9].view(-1, 3, 3))
        cursor += 9
        g = self._center_2d(raw[:, cursor : cursor + 6].view(-1, 2, 3))
        z = (
            a[:, :, None, None]
            + b[:, None, :, None]
            + c[:, None, None, :]
            + d[:, :, :, None]
            + e[:, :, None, :]
            + g[:, None, :, :]
        )
        return z.reshape(feature.shape[0], self.num_core)

    def _triple_tensor(self, image: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        masks = self._evidence_masks(image, feature_map.shape[-2:]).to(device=feature_map.device, dtype=feature_map.dtype)
        mass = masks.flatten(2).sum(dim=2).clamp_min(1e-4)
        pooled = (feature_map.unsqueeze(1) * masks.unsqueeze(2)).flatten(3).sum(dim=3) / mass.unsqueeze(2)
        mask_tokens = self.token_proj(pooled)
        core_weights = self.core_evidence_weights.to(device=mask_tokens.device, dtype=mask_tokens.dtype)
        core_tokens = torch.einsum("cm,bmd->bcd", core_weights, mask_tokens)
        triple = (core_tokens * self.triple_weight.to(device=core_tokens.device, dtype=core_tokens.dtype).unsqueeze(0)).sum(dim=2)
        triple = triple + self.triple_bias.to(device=core_tokens.device, dtype=core_tokens.dtype).view(1, -1)
        return self._center_3d(triple.view(-1, 3, 2, 3)).reshape(feature_map.shape[0], self.num_core)

    def _scatter_core(self, core_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        residual = torch.zeros_like(base_logits)
        core_logits = core_logits.to(device=base_logits.device, dtype=base_logits.dtype)
        class_idx = self.core_to_class.to(device=base_logits.device)
        valid = class_idx.ge(0)
        if bool(valid.any()):
            residual[:, class_idx[valid]] = core_logits[:, valid]
        active = self.core_class_mask.to(device=residual.device, dtype=residual.dtype)
        if float(active.sum()) > 0.0:
            mean = (residual * active).sum(dim=1, keepdim=True) / active.sum().clamp_min(1.0)
            residual = (residual - mean) * active
        return residual

    def _gate(self, base_logits: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(base_logits, dim=1)
        uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
        uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
        top2 = torch.topk(prob, k=min(2, prob.shape[1]), dim=1).indices
        if top2.shape[1] < 2:
            return uncertainty_gate
        core_mask = self.core_class_mask.to(device=base_logits.device).bool().view(-1)
        both_core = core_mask[top2[:, 0]] & core_mask[top2[:, 1]]
        core_gate = torch.where(
            both_core.view(-1, 1),
            uncertainty_gate.new_ones(uncertainty_gate.shape),
            uncertainty_gate.new_full(uncertainty_gate.shape, self.core_gate_floor),
        )
        return uncertainty_gate * core_gate

    def forward(
        self,
        image: torch.Tensor,
        feature_map: torch.Tensor | None,
        feature: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        if feature_map is None or feature_map.ndim != 4:
            return base_logits.new_zeros(base_logits.shape)
        z = self._factor_tensor(feature) + self._triple_tensor(image, feature_map)
        residual = self.scale * self._scatter_core(z, base_logits)
        if self.protected_negative_limit > 0.0 and bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return self._gate(base_logits).to(dtype=residual.dtype) * residual

    def _evidence_masks(self, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
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
        dark_water = torch.sigmoid((0.42 - value) * 10.0) * torch.sigmoid((0.30 - saturation) * 12.0) * low_texture
        thin_film = torch.clamp(specular + 0.6 * dark_water, 0.0, 1.0) * low_contrast
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        rough_aggregate = torch.sigmoid((grad - 0.075) * 22.0) * torch.sigmoid((lap - 0.050) * 18.0)
        obstruction = torch.clamp(thin_film + 0.6 * dark_water + 0.5 * texture_erasure, 0.0, 1.0)
        visible_rough = rough_aggregate * (1.0 - obstruction)
        hidden_rough = rough_aggregate * obstruction
        granular = torch.sigmoid((local_contrast - 0.045) * 35.0) * torch.sigmoid((saturation - 0.05) * 8.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        dry_micro = rough_aggregate * (1.0 - torch.clamp(specular + dark_water + marking, 0.0, 1.0))
        asphalt_proxy = torch.sigmoid((0.58 - value) * 8.0) * torch.sigmoid((0.38 - saturation) * 6.0)
        concrete_proxy = torch.sigmoid((value - 0.48) * 8.0) * torch.sigmoid((0.42 - saturation) * 6.0)
        masks = torch.cat(
            [
                dry_micro,
                visible_rough,
                rough_aggregate,
                thin_film,
                specular,
                dark_water,
                texture_erasure,
                hidden_rough,
                granular,
                marking,
                asphalt_proxy,
                concrete_proxy,
            ],
            dim=1,
        )
        return F.interpolate(masks, size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


class MechanismChartedFullOrderCouplingTensorField(FullOrderCouplingTensorField):
    """Mechanism-charted full-order tensor field for RSCD core labels.

    The ordinary full-order tensor field learns one homogeneous three-way
    residual H_fmr. This variant keeps the ANOVA decomposition but routes H_fmr
    through separate physical charts: dry visible roughness, wet smooth film,
    wet hidden roughness, water smooth obstruction, water hidden roughness,
    concrete hidden roughness, and asphalt wet sheen. Each chart pools the
    ConvNeXt feature map with a different PhysicsTexture mask mixture and then
    writes only to the compatible core cells with a zero-marginal H tensor.
    """

    mechanism_names = (
        "dry_visible_roughness",
        "wet_smooth_film",
        "wet_hidden_roughness",
        "water_smooth_obstruction",
        "water_hidden_roughness",
        "concrete_hidden_roughness",
        "asphalt_wet_sheen",
    )

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        feature_map_dim: int = 768,
        token_dim: int = 96,
        hidden_dim: int = 96,
        scale: float = 0.04,
        gate_threshold: float = 0.35,
        gate_temperature: float = 10.0,
        core_gate_floor: float = 0.05,
        protected_negative_limit: float = 0.0,
        router_hidden_dim: int = 96,
        router_temperature: float = 1.0,
        sparse_router_topk: int = 2,
        physics_prior_weight: float = 1.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            class_to_idx=class_to_idx,
            feature_map_dim=feature_map_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            core_gate_floor=core_gate_floor,
            protected_negative_limit=protected_negative_limit,
            zero_init=zero_init,
        )
        self.num_mechanisms = len(self.mechanism_names)
        self.router_temperature = max(float(router_temperature), 1e-4)
        self.sparse_router_topk = max(int(sparse_router_topk), 0)
        self.physics_prior_weight = max(float(physics_prior_weight), 0.0)
        self.mechanism_router = nn.Sequential(
            nn.LayerNorm(int(in_dim) + self.num_masks),
            nn.Linear(int(in_dim) + self.num_masks, int(router_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(router_hidden_dim), self.num_mechanisms),
        )
        self.mechanism_triple_weight = nn.Parameter(
            torch.empty(self.num_mechanisms, self.num_core, int(token_dim))
        )
        self.mechanism_triple_bias = nn.Parameter(torch.zeros(self.num_mechanisms, self.num_core))
        self.register_buffer("mechanism_core_weights", self._build_mechanism_core_weights())
        self.register_buffer("mechanism_mask_weights", self._build_mechanism_mask_weights())
        self.register_buffer("mechanism_contrast_weights", self._build_mechanism_contrast_weights())
        self.mechanism_token_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(token_dim)),
                    nn.Linear(int(token_dim), int(token_dim)),
                    nn.GELU(),
                    nn.Linear(int(token_dim), int(token_dim)),
                )
                for _ in range(self.num_mechanisms)
            ]
        )
        if zero_init:
            nn.init.zeros_(self.mechanism_router[-1].weight)
            nn.init.zeros_(self.mechanism_router[-1].bias)
            nn.init.zeros_(self.mechanism_triple_weight)
            nn.init.zeros_(self.mechanism_triple_bias)
            for adapter in self.mechanism_token_adapters:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
        else:
            nn.init.trunc_normal_(self.mechanism_triple_weight, std=0.02)
            nn.init.zeros_(self.mechanism_triple_bias)

    def _build_mechanism_mask_weights(self) -> torch.Tensor:
        mask_id = {
            "dry_micro": 0,
            "visible_rough": 1,
            "rough_aggregate": 2,
            "thin_film": 3,
            "specular": 4,
            "dark_water": 5,
            "texture_erasure": 6,
            "hidden_rough": 7,
            "granular": 8,
            "marking": 9,
            "asphalt_proxy": 10,
            "concrete_proxy": 11,
        }
        rows = torch.zeros(self.num_mechanisms, self.num_masks, dtype=torch.float32)
        rows[0, [mask_id["dry_micro"], mask_id["visible_rough"], mask_id["rough_aggregate"]]] = torch.tensor([1.2, 1.0, 0.6])
        rows[1, [mask_id["thin_film"], mask_id["specular"], mask_id["texture_erasure"]]] = torch.tensor([1.1, 0.9, 0.8])
        rows[2, [mask_id["thin_film"], mask_id["hidden_rough"], mask_id["texture_erasure"], mask_id["rough_aggregate"]]] = torch.tensor([0.8, 1.1, 0.8, 0.5])
        rows[3, [mask_id["dark_water"], mask_id["thin_film"], mask_id["texture_erasure"]]] = torch.tensor([1.1, 0.8, 0.8])
        rows[4, [mask_id["dark_water"], mask_id["hidden_rough"], mask_id["texture_erasure"], mask_id["rough_aggregate"]]] = torch.tensor([0.9, 1.2, 0.7, 0.6])
        rows[5, [mask_id["concrete_proxy"], mask_id["hidden_rough"], mask_id["texture_erasure"], mask_id["rough_aggregate"]]] = torch.tensor([1.0, 1.1, 0.7, 0.5])
        rows[6, [mask_id["asphalt_proxy"], mask_id["specular"], mask_id["thin_film"], mask_id["dark_water"]]] = torch.tensor([1.0, 1.0, 0.7, 0.4])
        rows[:, mask_id["marking"]] = -0.35
        rows = rows.clamp_min(0.0)
        return rows / rows.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _build_mechanism_contrast_weights(self) -> torch.Tensor:
        mask_id = {
            "dry_micro": 0,
            "visible_rough": 1,
            "rough_aggregate": 2,
            "thin_film": 3,
            "specular": 4,
            "dark_water": 5,
            "texture_erasure": 6,
            "hidden_rough": 7,
            "granular": 8,
            "marking": 9,
            "asphalt_proxy": 10,
            "concrete_proxy": 11,
        }
        rows = torch.zeros(self.num_mechanisms, self.num_masks, dtype=torch.float32)
        rows[0, [mask_id["dry_micro"], mask_id["visible_rough"], mask_id["rough_aggregate"]]] = torch.tensor([1.0, 0.8, 0.6])
        rows[0, [mask_id["thin_film"], mask_id["specular"], mask_id["dark_water"], mask_id["texture_erasure"]]] = torch.tensor([-0.7, -0.5, -0.6, -0.5])
        rows[1, [mask_id["thin_film"], mask_id["specular"], mask_id["texture_erasure"]]] = torch.tensor([1.0, 0.9, 0.7])
        rows[1, [mask_id["visible_rough"], mask_id["rough_aggregate"], mask_id["hidden_rough"]]] = torch.tensor([-0.6, -0.7, -0.5])
        rows[2, [mask_id["thin_film"], mask_id["hidden_rough"], mask_id["rough_aggregate"]]] = torch.tensor([0.7, 1.0, 0.6])
        rows[2, [mask_id["specular"], mask_id["texture_erasure"]]] = torch.tensor([-0.4, -0.4])
        rows[3, [mask_id["dark_water"], mask_id["thin_film"], mask_id["texture_erasure"]]] = torch.tensor([1.0, 0.8, 0.7])
        rows[3, [mask_id["visible_rough"], mask_id["rough_aggregate"], mask_id["granular"]]] = torch.tensor([-0.6, -0.6, -0.4])
        rows[4, [mask_id["dark_water"], mask_id["hidden_rough"], mask_id["rough_aggregate"]]] = torch.tensor([0.9, 1.0, 0.7])
        rows[4, [mask_id["specular"], mask_id["thin_film"]]] = torch.tensor([-0.4, -0.3])
        rows[5, [mask_id["concrete_proxy"], mask_id["hidden_rough"], mask_id["texture_erasure"]]] = torch.tensor([1.0, 0.8, 0.5])
        rows[5, [mask_id["asphalt_proxy"], mask_id["specular"]]] = torch.tensor([-0.8, -0.4])
        rows[6, [mask_id["asphalt_proxy"], mask_id["specular"], mask_id["thin_film"]]] = torch.tensor([1.0, 0.9, 0.6])
        rows[6, [mask_id["concrete_proxy"], mask_id["hidden_rough"]]] = torch.tensor([-0.7, -0.5])
        rows[:, mask_id["marking"]] = rows[:, mask_id["marking"]] - 0.3
        return rows / rows.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _build_mechanism_core_weights(self) -> torch.Tensor:
        weights = torch.zeros(self.num_mechanisms, self.num_core, dtype=torch.float32)
        for f_idx, friction in enumerate(self.friction_values):
            for m_idx, material in enumerate(self.material_values):
                for r_idx, roughness in enumerate(self.roughness_values):
                    core_idx = self._core_index(f_idx, m_idx, r_idx)
                    rough_weight = 0.45 if roughness == "smooth" else (0.90 if roughness == "slight" else 1.15)
                    if friction == "dry":
                        weights[0, core_idx] = rough_weight
                    if friction == "wet" and roughness == "smooth":
                        weights[1, core_idx] = 1.0
                    if friction == "wet" and roughness in {"slight", "severe"}:
                        weights[2, core_idx] = rough_weight
                    if friction == "water" and roughness == "smooth":
                        weights[3, core_idx] = 1.0
                    if friction == "water" and roughness in {"slight", "severe"}:
                        weights[4, core_idx] = rough_weight
                    if material == "concrete" and friction in {"wet", "water"} and roughness in {"slight", "severe"}:
                        weights[5, core_idx] = rough_weight
                    if material == "asphalt" and friction in {"wet", "water"}:
                        weights[6, core_idx] = 1.0 if roughness == "smooth" else 0.75
        return weights

    def _physics_router_prior(self, mask_stats: torch.Tensor) -> torch.Tensor:
        if self.physics_prior_weight <= 0.0:
            return mask_stats.new_zeros((mask_stats.shape[0], self.num_mechanisms))
        dry_micro = mask_stats[:, 0:1]
        visible_rough = mask_stats[:, 1:2]
        rough_aggregate = mask_stats[:, 2:3]
        thin_film = mask_stats[:, 3:4]
        specular = mask_stats[:, 4:5]
        dark_water = mask_stats[:, 5:6]
        texture_erasure = mask_stats[:, 6:7]
        hidden_rough = mask_stats[:, 7:8]
        asphalt_proxy = mask_stats[:, 10:11]
        concrete_proxy = mask_stats[:, 11:12]
        obstruction = torch.clamp(thin_film + 0.6 * dark_water + 0.5 * texture_erasure, 0.0, 2.0)
        prior = torch.cat(
            [
                dry_micro + 0.75 * visible_rough + 0.35 * rough_aggregate - 0.40 * obstruction,
                thin_film + 0.75 * specular + 0.65 * texture_erasure,
                thin_film + hidden_rough + 0.55 * texture_erasure,
                dark_water + 0.75 * thin_film + 0.55 * texture_erasure,
                dark_water + 1.05 * hidden_rough + 0.45 * rough_aggregate,
                concrete_proxy + hidden_rough + 0.50 * texture_erasure,
                asphalt_proxy + specular + 0.45 * thin_film,
            ],
            dim=1,
        )
        return float(self.physics_prior_weight) * prior

    def _route_mechanisms(self, feature: torch.Tensor, mask_stats: torch.Tensor) -> torch.Tensor:
        context = torch.cat([feature, mask_stats.to(device=feature.device, dtype=feature.dtype)], dim=1)
        logits = self.mechanism_router(context)
        logits = logits + self._physics_router_prior(mask_stats).to(device=feature.device, dtype=feature.dtype)
        probs = F.softmax(logits / self.router_temperature, dim=1)
        topk = self.sparse_router_topk
        if topk <= 0 or topk >= self.num_mechanisms:
            return probs
        _, indices = torch.topk(probs, k=topk, dim=1)
        mask = torch.zeros_like(probs).scatter_(1, indices, 1.0)
        sparse = probs * mask
        return sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _charted_triple_tensor(
        self,
        image: torch.Tensor,
        feature_map: torch.Tensor,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        masks = self._evidence_masks(image, feature_map.shape[-2:]).to(device=feature_map.device, dtype=feature_map.dtype)
        mass = masks.flatten(2).sum(dim=2).clamp_min(1e-4)
        pooled = (feature_map.unsqueeze(1) * masks.unsqueeze(2)).flatten(3).sum(dim=3) / mass.unsqueeze(2)
        mask_tokens = self.token_proj(pooled)
        mask_stats = masks.mean(dim=(2, 3)).to(device=feature.device, dtype=feature.dtype)
        router = self._route_mechanisms(feature, mask_stats)
        mechanism_mask_weights = self.mechanism_mask_weights.to(device=mask_tokens.device, dtype=mask_tokens.dtype)
        mechanism_contrast_weights = self.mechanism_contrast_weights.to(
            device=mask_tokens.device,
            dtype=mask_tokens.dtype,
        )
        mechanism_tokens = torch.einsum("km,bmd->bkd", mechanism_mask_weights, mask_tokens)
        mechanism_tokens = mechanism_tokens + torch.einsum("km,bmd->bkd", mechanism_contrast_weights, mask_tokens)
        adapted_tokens = [
            mechanism_tokens[:, mechanism_idx, :]
            + self.mechanism_token_adapters[mechanism_idx](mechanism_tokens[:, mechanism_idx, :])
            for mechanism_idx in range(self.num_mechanisms)
        ]
        mechanism_tokens = torch.stack(adapted_tokens, dim=1)
        triple_weight = self.mechanism_triple_weight.to(device=feature_map.device, dtype=feature_map.dtype)
        triple_bias = self.mechanism_triple_bias.to(device=feature_map.device, dtype=feature_map.dtype)
        scores = torch.einsum("bkd,kcd->bkc", mechanism_tokens, triple_weight) + triple_bias.unsqueeze(0)
        centered = self._center_3d(scores.reshape(-1, 3, 2, 3)).reshape(
            feature.shape[0],
            self.num_mechanisms,
            self.num_core,
        )
        core_weights = self.mechanism_core_weights.to(device=centered.device, dtype=centered.dtype).unsqueeze(0)
        charted = centered * core_weights
        return (router.to(dtype=charted.dtype).unsqueeze(-1) * charted).sum(dim=1)

    def forward(
        self,
        image: torch.Tensor,
        feature_map: torch.Tensor | None,
        feature: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        if feature_map is None or feature_map.ndim != 4:
            return base_logits.new_zeros(base_logits.shape)
        z = self._factor_tensor(feature) + self._charted_triple_tensor(image, feature_map, feature)
        residual = self.scale * self._scatter_core(z, base_logits)
        if self.protected_negative_limit > 0.0 and bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return self._gate(base_logits).to(dtype=residual.dtype) * residual


class CoreFactorCoupledResidual(nn.Module):
    """Core-only low-rank residual for the RSCD hard compositional subgraph.

    The difficult RSCD cells are concentrated in dry/wet/water road patches on
    asphalt/concrete with smooth/slight/severe unevenness. This module keeps the
    validated classifier intact and learns a small residual only for those 18
    cells. The residual combines low-rank factor-pair embeddings with explicit
    optical/roughness statistics, and it is strongly activated only when the
    base model's top-2 classes are neighboring core cells.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        rank: int = 32,
        scale: float = 0.08,
        normalize: bool = True,
        neighbor_gate_floor: float = 0.05,
        uncertainty_threshold: float = 0.40,
        uncertainty_temperature: float = 10.0,
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError("core factor-coupled rank must be positive.")
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.neighbor_gate_floor = float(neighbor_gate_floor)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.uncertainty_temperature = float(uncertainty_temperature)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)

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

        stat_dim = 24
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(int(in_dim) + stat_dim),
            nn.Linear(int(in_dim) + stat_dim, rank, bias=False),
        )
        pair_dims = {
            "friction_material": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["material"]),
            "friction_unevenness": len(FACTOR_LABELS["friction"]) * len(FACTOR_LABELS["unevenness"]),
            "material_unevenness": len(FACTOR_LABELS["material"]) * len(FACTOR_LABELS["unevenness"]),
        }
        self.pair_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(size, rank)) for name, size in pair_dims.items()}
        )
        num_classes = len(class_to_idx)
        self.bias = nn.Parameter(torch.zeros(num_classes))

        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        pair_indices: dict[str, list[int]] = {name: [] for name in pair_dims}
        pair_masks: dict[str, list[float]] = {name: [] for name in pair_dims}
        core_mask = torch.zeros(num_classes, dtype=torch.float32)
        protected = torch.zeros(num_classes, dtype=torch.bool)
        friction_vocab = {"dry": 0, "wet": 1, "water": 2}
        material_vocab = {"asphalt": 0, "concrete": 1}
        roughness_vocab = {"smooth": 0, "slight": 1, "severe": 2}
        friction_ids = torch.full((num_classes,), -1, dtype=torch.long)
        material_ids = torch.full((num_classes,), -1, dtype=torch.long)
        roughness_ids = torch.full((num_classes,), -1, dtype=torch.long)
        for class_idx in range(num_classes):
            class_name = idx_to_class[class_idx]
            factors_text = _factor_text(class_name)
            is_core = (
                factors_text["friction"] in friction_vocab
                and factors_text["material"] in material_vocab
                and factors_text["unevenness"] in roughness_vocab
            )
            if is_core:
                core_mask[class_idx] = 1.0
                friction_ids[class_idx] = friction_vocab[str(factors_text["friction"])]
                material_ids[class_idx] = material_vocab[str(factors_text["material"])]
                roughness_ids[class_idx] = roughness_vocab[str(factors_text["unevenness"])]
            if factors_text["friction"] in {"wet", "water"}:
                protected[class_idx] = True
            factors = parse_rscd_factors(class_name)
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_material",
                factors["friction"],
                factors["material"],
                len(FACTOR_LABELS["material"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "friction_unevenness",
                factors["friction"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
            _append_pair_index(
                pair_indices,
                pair_masks,
                "material_unevenness",
                factors["material"],
                factors["unevenness"],
                len(FACTOR_LABELS["unevenness"]),
            )
        for name in pair_dims:
            self.register_buffer(f"{name}_idx", torch.tensor(pair_indices[name], dtype=torch.long))
            self.register_buffer(f"{name}_mask", torch.tensor(pair_masks[name], dtype=torch.float32))
        self.register_buffer("core_class_mask", core_mask.view(1, -1))
        self.register_buffer("protected_class_mask", protected.view(1, -1))
        self.register_buffer("core_friction_ids", friction_ids)
        self.register_buffer("core_material_ids", material_ids)
        self.register_buffer("core_roughness_ids", roughness_ids)

        for embedding in self.pair_embeddings.values():
            if zero_init:
                nn.init.zeros_(embedding)
            else:
                nn.init.trunc_normal_(embedding, std=0.02)

    def class_weight(self) -> torch.Tensor:
        first_embedding = next(iter(self.pair_embeddings.values()))
        weight = torch.zeros((self.bias.numel(), first_embedding.shape[1]), device=first_embedding.device)
        active = torch.zeros((self.bias.numel(), 1), device=first_embedding.device)
        for name, embedding in self.pair_embeddings.items():
            pair_idx = getattr(self, f"{name}_idx")
            pair_mask = getattr(self, f"{name}_mask").to(device=embedding.device, dtype=embedding.dtype)
            weight = weight + embedding[pair_idx] * pair_mask.unsqueeze(1)
            active = active + pair_mask.unsqueeze(1)
        weight = weight / active.clamp_min(1.0)
        return weight * self.core_class_mask.t().to(device=weight.device, dtype=weight.dtype)

    def forward(self, image: torch.Tensor, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        stats = self._stats(image).to(dtype=feature.dtype)
        query = self.feature_proj(torch.cat([feature, stats], dim=1))
        weight = self.class_weight()
        if self.normalize:
            query = F.normalize(query, dim=1)
            weight = F.normalize(weight, dim=1)
        residual = self.scale * (query @ weight.t() + self.bias)
        residual = residual * self.core_class_mask.to(device=residual.device, dtype=residual.dtype)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            uncertainty_gate = torch.sigmoid(
                (uncertainty - self.uncertainty_threshold) * self.uncertainty_temperature
            )
            top2 = base_logits.topk(k=2, dim=1).indices
            c1, c2 = top2[:, 0], top2[:, 1]
            f_ids = self.core_friction_ids.to(device=base_logits.device)
            m_ids = self.core_material_ids.to(device=base_logits.device)
            r_ids = self.core_roughness_ids.to(device=base_logits.device)
            valid = (f_ids[c1] >= 0) & (f_ids[c2] >= 0)
            f_diff = f_ids[c1] != f_ids[c2]
            m_diff = m_ids[c1] != m_ids[c2]
            r_diff = r_ids[c1] != r_ids[c2]
            diff_count = f_diff.long() + m_diff.long() + r_diff.long()
            friction_neighbor = (f_ids[c1] - f_ids[c2]).abs() == 1
            roughness_neighbor = (r_ids[c1] - r_ids[c2]).abs() == 1
            material_neighbor = m_diff
            neighbor = valid & (diff_count == 1) & (
                (f_diff & friction_neighbor) | (r_diff & roughness_neighbor) | (m_diff & material_neighbor)
            )
            neighbor_gate = torch.where(
                neighbor.view(-1, 1),
                torch.ones_like(uncertainty_gate),
                uncertainty_gate.new_full(uncertainty_gate.shape, self.neighbor_gate_floor),
            )
        return uncertainty_gate.to(dtype=residual.dtype) * neighbor_gate.to(dtype=residual.dtype) * residual

    def _stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
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
        gx_abs = gx.abs()
        gy_abs = gy.abs()
        anisotropy = (gx_abs.mean(dim=(2, 3)) - gy_abs.mean(dim=(2, 3))).abs() / (
            gx_abs.mean(dim=(2, 3)) + gy_abs.mean(dim=(2, 3)) + 1e-4
        )
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
        wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)
        rough_power = F.avg_pool2d(grad.square(), kernel_size=7, stride=1, padding=3)
        rough_var = F.avg_pool2d((grad.square() - rough_power).abs(), kernel_size=7, stride=1, padding=3)
        return torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3)),
                specular.mean(dim=(2, 3)),
                specular.amax(dim=(2, 3)),
                dark_water.mean(dim=(2, 3)),
                dark_water.amax(dim=(2, 3)),
                thin_film.mean(dim=(2, 3)),
                texture_erasure.mean(dim=(2, 3)),
                low_texture.mean(dim=(2, 3)),
                low_contrast.mean(dim=(2, 3)),
                anisotropy,
                rough_power.mean(dim=(2, 3)),
                rough_var.mean(dim=(2, 3)),
                _soft_connectedness(wet_proxy),
                gray.mean(dim=(2, 3)),
                gray.std(dim=(2, 3)),
            ],
            dim=1,
        )


class WaterEvidenceLogitGate(nn.Module):
    """Patch-compatible wet/water logit residual from explicit optical evidence.

    RSCD images are cropped road patches, so this gate avoids vertical contact
    priors. It only summarizes photometric water-film cues: low-saturation
    specular highlights, dark smooth water, local contrast erasure, and soft
    connected wet regions.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        scale: float = 0.20,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
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
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, 24),
            nn.GELU(),
            nn.Linear(24, 2),
        )
        if zero_init:
            last = self.evidence_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

        wet_mask = torch.zeros(len(class_to_idx), dtype=torch.float32)
        water_mask = torch.zeros(len(class_to_idx), dtype=torch.float32)
        for class_name, class_idx in class_to_idx.items():
            factors = _factor_text(class_name)
            if factors["friction"] == "wet":
                wet_mask[int(class_idx)] = 1.0
            elif factors["friction"] == "water":
                water_mask[int(class_idx)] = 1.0
        self.register_buffer("wet_class_mask", wet_mask.view(1, -1))
        self.register_buffer("water_class_mask", water_mask.view(1, -1))

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
        mirror_smooth = specular * low_contrast
        hidden_water = dark_water * low_contrast
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)

        stats = torch.cat(
            [
                specular.mean(dim=(2, 3)),
                specular.amax(dim=(2, 3)),
                dark_water.mean(dim=(2, 3)),
                dark_water.amax(dim=(2, 3)),
                thin_film.mean(dim=(2, 3)),
                mirror_smooth.mean(dim=(2, 3)),
                hidden_water.mean(dim=(2, 3)),
                texture_erasure.mean(dim=(2, 3)),
                low_texture.mean(dim=(2, 3)),
                low_contrast.mean(dim=(2, 3)),
                _soft_connectedness(wet_proxy),
                value.std(dim=(2, 3)),
            ],
            dim=1,
        )
        wet_water_delta = torch.tanh(self.evidence_head(stats)) * self.scale
        wet_delta = wet_water_delta[:, 0:1] * self.wet_class_mask
        water_delta = wet_water_delta[:, 1:2] * self.water_class_mask
        return wet_delta + water_delta


class CoupledOpticalRoughnessResidual(nn.Module):
    """Hard-cell optical/roughness residual for coupled RSCD factor boundaries.

    This module is intentionally narrow: it does not replace the 27-way
    classifier. It learns a zero-initialized residual only for the dry/wet/water
    asphalt/concrete roughness cells where RSCD errors concentrate.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        hidden_dim: int = 96,
        scale: float = 0.12,
        gate_threshold: float = 0.35,
        gate_temperature: float = 8.0,
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        num_classes = len(class_to_idx)
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
        hard = torch.zeros(num_classes, dtype=torch.float32)
        protected = torch.zeros(num_classes, dtype=torch.bool)
        for class_name, class_idx in class_to_idx.items():
            factors = _factor_text(class_name)
            is_core_state = factors["friction"] in {"dry", "wet", "water"}
            is_core_material = factors["material"] in {"asphalt", "concrete"}
            is_core_rough = factors["unevenness"] in {"smooth", "slight", "severe"}
            if is_core_state and is_core_material and is_core_rough:
                hard[int(class_idx)] = 1.0
            if factors["friction"] in {"wet", "water"}:
                protected[int(class_idx)] = True
        self.register_buffer("hard_class_mask", hard.view(1, -1))
        self.register_buffer("protected_class_mask", protected.view(1, -1))
        stat_dim = 18
        self.head = nn.Sequential(
            nn.LayerNorm(int(in_dim) + stat_dim),
            nn.Linear(int(in_dim) + stat_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), num_classes),
        )
        if zero_init:
            last = self.head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, image: torch.Tensor, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        stats = self._optical_roughness_stats(image)
        residual = torch.tanh(self.head(torch.cat([feature, stats.to(dtype=feature.dtype)], dim=1))) * self.scale
        residual = residual * self.hard_class_mask.to(device=residual.device, dtype=residual.dtype)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
        return gate.to(dtype=residual.dtype) * residual

    def _optical_roughness_stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
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
        texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
        wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)
        gx_abs = gx.abs()
        gy_abs = gy.abs()
        anisotropy = (gx_abs.mean(dim=(2, 3)) - gy_abs.mean(dim=(2, 3))).abs() / (
            gx_abs.mean(dim=(2, 3)) + gy_abs.mean(dim=(2, 3)) + 1e-4
        )
        return torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3)),
                specular.mean(dim=(2, 3)),
                specular.amax(dim=(2, 3)),
                dark_water.mean(dim=(2, 3)),
                dark_water.amax(dim=(2, 3)),
                low_texture.mean(dim=(2, 3)),
                texture_erasure.mean(dim=(2, 3)),
                anisotropy,
                _soft_connectedness(wet_proxy),
            ],
            dim=1,
        )


class DryConcreteRoughnessVORResidual(nn.Module):
    """Mechanism-specific dry-concrete roughness residual.

    VOR screens showed a repeatable gain on dry concrete roughness but damage
    when the same cue was shared with wet/water states. This residual is the
    narrow HMC-Sheaf chart for the dry concrete roughness mechanism: it only
    redistributes logits among dry_concrete_smooth/slight/severe and activates
    only when the base model already assigns probability mass to that trio.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 48,
        scale: float = 0.05,
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
                field.std(dim=(2, 3)),
                self._top_fraction_mean(field, 0.05),
                self._top_fraction_mean(field, 0.15),
                _soft_connectedness(field),
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
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3)),
            ],
            dim=1,
        )
        field_stats = [
            self._field_stats(rough_base),
            self._field_stats(visible_rough),
            self._field_stats(dry_rough),
            self._field_stats(anti_glare_rough),
            self._field_stats(concrete_like),
        ]
        return torch.cat([global_stats, *field_stats], dim=1)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        dry_idx = self.dry_concrete_idx.to(device=base_logits.device)
        dry_logits = base_logits.index_select(1, dry_idx)
        probs = F.softmax(base_logits, dim=1)
        dry_probs = probs.index_select(1, dry_idx)
        dry_mass = dry_probs.sum(dim=1, keepdim=True)
        dry_top = dry_probs.amax(dim=1, keepdim=True)
        dry_entropy = -(dry_probs.clamp_min(1e-8) * dry_probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        sorted_logits = dry_logits.sort(dim=1, descending=True).values
        sorted_probs = dry_probs.sort(dim=1, descending=True).values
        logit_features = torch.cat(
            [
                dry_logits,
                dry_probs,
                dry_mass,
                dry_top,
                sorted_logits[:, 0:1] - sorted_logits[:, 1:2],
                sorted_probs[:, 0:1] - sorted_probs[:, 1:2],
            ],
            dim=1,
        )
        stats = self._stats(image).to(device=base_logits.device, dtype=base_logits.dtype)
        raw_delta = torch.tanh(self.head(torch.cat([stats, logit_features.to(dtype=stats.dtype)], dim=1)))
        centered_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
        gate = torch.sigmoid((dry_mass - self.gate_threshold) * self.gate_temperature)
        # Ambiguous dry-concrete roughness cases deserve the strongest correction;
        # very confident cases should remain almost unchanged.
        ambiguity = torch.sigmoid((0.42 - (sorted_probs[:, 0:1] - sorted_probs[:, 1:2])) * 10.0)
        delta = centered_delta * gate.to(dtype=centered_delta.dtype) * ambiguity.to(dtype=centered_delta.dtype) * self.scale
        residual = torch.zeros_like(base_logits)
        residual.scatter_add_(1, dry_idx.view(1, -1).expand(base_logits.size(0), -1), delta.to(dtype=residual.dtype))
        return residual


class DryPavedRoughnessVORResidual(DryConcreteRoughnessVORResidual):
    """Dry-visible roughness chart for dry asphalt and dry concrete.

    This extends the only currently repeatable positive chart, dry-concrete
    VOR, but keeps the same narrow safety contract. It redistributes logits
    only inside `dry_asphalt_{smooth,slight,severe}` and
    `dry_concrete_{smooth,slight,severe}`. Each material trio is zero-sum, so
    the residual cannot directly change friction state, wet/water classes, or
    asphalt-vs-concrete total probability.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 48,
        material_dim: int = 6,
        scale: float = 0.05,
        gate_threshold: float = 0.12,
        gate_temperature: float = 14.0,
        head_mode: str = "shared",
        material_gate_threshold: float = 0.0,
        material_gate_temperature: float = 16.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            zero_init=zero_init,
        )
        self.head = None
        groups: list[list[int]] = []
        names: list[str] = []
        for material in ("asphalt", "concrete"):
            trio = [
                f"dry_{material}_smooth",
                f"dry_{material}_slight",
                f"dry_{material}_severe",
            ]
            if all(name in class_to_idx for name in trio):
                groups.append([int(class_to_idx[name]) for name in trio])
                names.append(material)
        if not groups:
            raise ValueError("DryPavedRoughnessVORResidual found no dry paved roughness trios.")
        self.material_names = tuple(names)
        self.register_buffer("dry_paved_group_idx", torch.as_tensor(groups, dtype=torch.long))
        self.head_mode = str(head_mode).lower().strip()
        if self.head_mode not in {"shared", "nonshared"}:
            raise ValueError(f"unknown dry paved roughness head_mode: {head_mode}")
        self.material_gate_threshold = float(material_gate_threshold)
        self.material_gate_temperature = float(material_gate_temperature)
        stat_dim = 40
        logit_dim = 10

        def make_head(input_dim: int) -> nn.Sequential:
            head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 3),
            )
            if zero_init:
                last = head[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            return head

        if self.head_mode == "shared":
            self.material_embedding = nn.Embedding(len(groups), int(material_dim))
            self.head = make_head(stat_dim + logit_dim + int(material_dim))
            self.heads = None
            nn.init.trunc_normal_(self.material_embedding.weight, std=0.02)
        else:
            self.material_embedding = None
            self.head = None
            self.heads = nn.ModuleList([make_head(stat_dim + logit_dim) for _ in groups])

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
        dry_visible_mask = (1.0 - obstruction) * (1.0 - snow_phase) * (1.0 - marking)
        concrete_like = (
            torch.sigmoid((value - 0.38) * 8.0)
            * torch.sigmoid((0.82 - value) * 8.0)
            * torch.sigmoid((0.30 - saturation) * 10.0)
            * dry_visible_mask
        )
        asphalt_like = (
            torch.sigmoid((value - 0.14) * 10.0)
            * torch.sigmoid((0.62 - value) * 8.0)
            * torch.sigmoid((0.42 - saturation) * 8.0)
            * dry_visible_mask
        )
        visible_rough = rough_base * dry_visible_mask
        concrete_rough = visible_rough * concrete_like
        asphalt_rough = visible_rough * asphalt_like
        anti_glare_rough = visible_rough * (1.0 - specular) * (1.0 - texture_erasure)
        global_stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3)),
            ],
            dim=1,
        )
        field_stats = [
            self._field_stats(rough_base),
            self._field_stats(visible_rough),
            self._field_stats(concrete_rough),
            self._field_stats(asphalt_rough),
            self._field_stats(anti_glare_rough),
        ]
        return torch.cat([global_stats, *field_stats], dim=1)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        group_idx = self.dry_paved_group_idx.to(device=base_logits.device)
        probs = F.softmax(base_logits, dim=1)
        stats = self._stats(image).to(device=base_logits.device, dtype=base_logits.dtype)
        material_embed = None
        if self.material_embedding is not None:
            material_ids = torch.arange(group_idx.shape[0], device=base_logits.device)
            material_embed = self.material_embedding(material_ids).to(dtype=base_logits.dtype)
        dry_paved_mass = probs.index_select(1, group_idx.reshape(-1)).sum(dim=1, keepdim=True)
        residual = torch.zeros_like(base_logits)
        for group_id in range(group_idx.shape[0]):
            idx = group_idx[group_id]
            group_logits = base_logits.index_select(1, idx)
            group_probs = probs.index_select(1, idx)
            group_mass = group_probs.sum(dim=1, keepdim=True)
            group_top = group_probs.amax(dim=1, keepdim=True)
            sorted_logits = group_logits.sort(dim=1, descending=True).values
            sorted_probs = group_probs.sort(dim=1, descending=True).values
            logit_features = torch.cat(
                [
                    group_logits,
                    group_probs,
                    group_mass,
                    group_top,
                    sorted_logits[:, 0:1] - sorted_logits[:, 1:2],
                    sorted_probs[:, 0:1] - sorted_probs[:, 1:2],
                ],
                dim=1,
            )
            if self.head_mode == "shared":
                assert self.head is not None and material_embed is not None
                material_feature = material_embed[group_id].view(1, -1).expand(base_logits.size(0), -1)
                head_input = torch.cat(
                    [stats, logit_features.to(dtype=stats.dtype), material_feature.to(dtype=stats.dtype)],
                    dim=1,
                )
                raw_delta = torch.tanh(self.head(head_input))
            else:
                assert self.heads is not None
                head_input = torch.cat([stats, logit_features.to(dtype=stats.dtype)], dim=1)
                raw_delta = torch.tanh(self.heads[group_id](head_input))
            centered_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
            gate = torch.sigmoid((group_mass - self.gate_threshold) * self.gate_temperature)
            if self.material_gate_threshold > 0.0:
                material_share = group_mass / dry_paved_mass.clamp_min(1e-6)
                material_gate = torch.sigmoid(
                    (material_share - self.material_gate_threshold) * self.material_gate_temperature
                )
                gate = gate * material_gate
            ambiguity = torch.sigmoid((0.42 - (sorted_probs[:, 0:1] - sorted_probs[:, 1:2])) * 10.0)
            delta = centered_delta * gate.to(dtype=centered_delta.dtype) * ambiguity.to(dtype=centered_delta.dtype) * self.scale
            residual.scatter_add_(1, idx.view(1, -1).expand(base_logits.size(0), -1), delta.to(dtype=residual.dtype))
        return residual


class WetWaterFilmVORResidual(nn.Module):
    """Mechanism-specific wet/water film residual for paved RSCD classes.

    Wet and water labels differ by optical film strength, dark smooth water,
    specular highlights, and texture erasure. This chart only applies
    antisymmetric corrections between matched wet/water pairs that share
    material and roughness, e.g. wet_concrete_slight versus
    water_concrete_slight. It is intentionally narrow to avoid the collateral
    wet/water damage observed with global VOR features.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 48,
        pair_dim: int = 8,
        scale: float = 0.05,
        material_scope: str = "all",
        gate_threshold: float = 0.12,
        gate_temperature: float = 14.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        material_scope = str(material_scope)
        if material_scope not in {"all", "asphalt", "concrete"}:
            raise ValueError(f"unknown wet/water film material scope: {material_scope}")
        pairs: list[tuple[int, int, str]] = []
        for material in ("asphalt", "concrete"):
            if material_scope != "all" and material != material_scope:
                continue
            for unevenness in ("smooth", "slight", "severe"):
                wet_name = f"wet_{material}_{unevenness}"
                water_name = f"water_{material}_{unevenness}"
                if wet_name in class_to_idx and water_name in class_to_idx:
                    pairs.append((int(class_to_idx[wet_name]), int(class_to_idx[water_name]), f"{material}_{unevenness}"))
        if not pairs:
            raise ValueError("WetWaterFilmVORResidual found no wet/water paved RSCD pairs.")
        self.register_buffer("wet_pair_idx", torch.as_tensor([left for left, _, _ in pairs], dtype=torch.long))
        self.register_buffer("water_pair_idx", torch.as_tensor([right for _, right, _ in pairs], dtype=torch.long))
        self.pair_names = tuple(name for _, _, name in pairs)
        self.pair_embedding = nn.Embedding(len(pairs), int(pair_dim))
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
            nn.LayerNorm(stat_dim + logit_dim + int(pair_dim)),
            nn.Linear(stat_dim + logit_dim + int(pair_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
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
                field.std(dim=(2, 3)),
                self._top_fraction_mean(field, 0.05),
                self._top_fraction_mean(field, 0.15),
                _soft_connectedness(field),
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
        mirror_smooth = specular * low_contrast
        hidden_water = dark_water * low_contrast
        wet_proxy = torch.clamp(specular + 0.5 * dark_water + 0.35 * thin_film, 0.0, 1.0)
        film_obstruction = torch.clamp(
            0.45 * thin_film + 0.30 * dark_water + 0.20 * specular + 0.35 * texture_erasure,
            0.0,
            1.0,
        )

        global_stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                low_texture.mean(dim=(2, 3)),
                low_contrast.mean(dim=(2, 3)),
                film_obstruction.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        field_stats = [
            self._field_stats(specular),
            self._field_stats(dark_water),
            self._field_stats(thin_film),
            self._field_stats(texture_erasure),
            self._field_stats(wet_proxy),
        ]
        return torch.cat([global_stats, *field_stats], dim=1)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        wet_idx = self.wet_pair_idx.to(device=base_logits.device)
        water_idx = self.water_pair_idx.to(device=base_logits.device)
        wet_logits = base_logits.index_select(1, wet_idx)
        water_logits = base_logits.index_select(1, water_idx)
        probs = F.softmax(base_logits, dim=1)
        wet_probs = probs.index_select(1, wet_idx)
        water_probs = probs.index_select(1, water_idx)
        pair_mass = wet_probs + water_probs
        logit_diff = water_logits - wet_logits
        prob_diff = water_probs - wet_probs
        pair_top = torch.maximum(wet_probs, water_probs)
        pair_margin = (water_probs - wet_probs).abs()
        all_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / math.log(float(base_logits.size(1)))
        stats = self._stats(image).to(device=base_logits.device, dtype=base_logits.dtype)
        stats = stats.unsqueeze(1).expand(-1, len(self.pair_names), -1)
        pair_ids = torch.arange(len(self.pair_names), device=base_logits.device)
        pair_feature = self.pair_embedding(pair_ids).unsqueeze(0).expand(base_logits.size(0), -1, -1)
        logit_features = torch.stack(
            [
                wet_logits,
                water_logits,
                wet_probs,
                water_probs,
                pair_mass,
                pair_top,
                logit_diff,
                prob_diff,
                pair_margin,
                all_entropy.expand(-1, len(self.pair_names)),
            ],
            dim=2,
        ).to(dtype=stats.dtype)
        raw_delta = torch.tanh(self.head(torch.cat([stats, pair_feature.to(dtype=stats.dtype), logit_features], dim=2))).squeeze(2)
        mass_gate = torch.sigmoid((pair_mass - self.gate_threshold) * self.gate_temperature)
        ambiguity = torch.sigmoid((0.38 - pair_margin) * 10.0)
        delta = raw_delta * mass_gate.to(dtype=raw_delta.dtype) * ambiguity.to(dtype=raw_delta.dtype) * self.scale
        residual = torch.zeros_like(base_logits)
        residual.scatter_add_(1, wet_idx.view(1, -1).expand(base_logits.size(0), -1), -delta.to(dtype=residual.dtype))
        residual.scatter_add_(1, water_idx.view(1, -1).expand(base_logits.size(0), -1), delta.to(dtype=residual.dtype))
        return residual


class SmoothFilmConcreteExpert(WetWaterFilmVORResidual):
    """Protected expert for the wet/water smooth-concrete film coupling.

    The retinex/anti-human stem showed a small positive signal on
    wet_concrete_smooth and water_concrete_smooth, but it damaged the
    slight/severe wet-water concrete classes. This expert makes that empirical
    observation explicit: it can only write residual logits to the two smooth
    concrete film classes, while rough wet-water concrete logits are left
    untouched by construction.

    Its two output modes separate the coupled label into (1) common smooth-film
    mass and (2) wet-versus-water transfer, which is closer to the RSCD label
    physics than a generic late residual head.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 48,
        scale: float = 0.05,
        gate_threshold: float = 0.05,
        gate_temperature: float = 14.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            pair_dim=1,
            scale=scale,
            material_scope="concrete",
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            zero_init=False,
        )
        target_names = ("wet_concrete_smooth", "water_concrete_smooth")
        missing = [name for name in target_names if name not in class_to_idx]
        if missing:
            raise ValueError(f"SmoothFilmConcreteExpert missing RSCD classes: {missing}")
        protected_names = (
            "wet_concrete_slight",
            "wet_concrete_severe",
            "water_concrete_slight",
            "water_concrete_severe",
        )
        protected_idx = [int(class_to_idx[name]) for name in protected_names if name in class_to_idx]
        self.target_names = target_names
        self.protected_names = tuple(name for name in protected_names if name in class_to_idx)
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.register_buffer(
            "smooth_film_target_idx",
            torch.as_tensor([int(class_to_idx[name]) for name in target_names], dtype=torch.long),
        )
        self.register_buffer("smooth_film_protected_idx", torch.as_tensor(protected_idx, dtype=torch.long))
        stat_dim = 52
        logit_dim = 13
        self.head = nn.Sequential(
            nn.LayerNorm(stat_dim + logit_dim),
            nn.Linear(stat_dim + logit_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        if zero_init:
            last = self.head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _retinex_stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 128:
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
        red = rgb[:, 0:1]
        green = rgb[:, 1:2]
        blue = rgb[:, 2:3]
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        chroma = maxc - minc
        saturation = chroma / maxc.clamp_min(1e-4)

        log_gray = torch.log(gray.clamp_min(1e-4))
        retinex = (log_gray - F.avg_pool2d(log_gray, kernel_size=15, stride=1, padding=7)).abs()
        red_green_opponent = (red - green).abs()
        blue_yellow_opponent = (blue - 0.5 * (red + green)).abs()
        neutral = torch.sigmoid((0.10 - chroma) * 28.0)
        shadow_film = torch.sigmoid((0.45 - value) * 10.0) * torch.sigmoid((0.30 - saturation) * 12.0)
        highlight_film = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)

        fields = [
            red_green_opponent,
            blue_yellow_opponent,
            retinex,
            neutral,
            shadow_film * neutral,
            highlight_film * neutral,
        ]
        stats = []
        for field in fields:
            stats.append(field.mean(dim=(2, 3)))
            stats.append(field.std(dim=(2, 3)))
        return torch.cat(stats, dim=1)

    def _stats(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat([super()._stats(image), self._retinex_stats(image)], dim=1)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        target_idx = self.smooth_film_target_idx.to(device=base_logits.device)
        protected_idx = self.smooth_film_protected_idx.to(device=base_logits.device)
        target_logits = base_logits.index_select(1, target_idx)
        probs = F.softmax(base_logits, dim=1)
        target_probs = probs.index_select(1, target_idx)
        pair_mass = target_probs.sum(dim=1, keepdim=True)
        pair_top = target_probs.amax(dim=1, keepdim=True)
        pair_margin = (target_probs[:, 1:2] - target_probs[:, 0:1]).abs()
        logit_gap = target_logits[:, 1:2] - target_logits[:, 0:1]
        if protected_idx.numel() > 0:
            protected_probs = probs.index_select(1, protected_idx)
            protected_mass = protected_probs.sum(dim=1, keepdim=True)
            protected_top = protected_probs.amax(dim=1, keepdim=True)
        else:
            protected_mass = torch.zeros_like(pair_mass)
            protected_top = torch.zeros_like(pair_mass)
        wetwater_concrete_mass = pair_mass + protected_mass
        smooth_share = pair_mass / wetwater_concrete_mass.clamp_min(1e-6)
        all_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / math.log(float(base_logits.size(1)))

        stats = self._stats(image).to(device=base_logits.device, dtype=base_logits.dtype)
        logit_features = torch.cat(
            [
                target_logits,
                target_probs,
                pair_mass,
                pair_top,
                pair_margin,
                logit_gap,
                all_entropy,
                protected_mass,
                protected_top,
                wetwater_concrete_mass,
                smooth_share,
            ],
            dim=1,
        ).to(dtype=stats.dtype)
        raw_modes = torch.tanh(self.head(torch.cat([stats, logit_features], dim=1)))
        common_mode = raw_modes[:, 0:1]
        wet_water_transfer = raw_modes[:, 1:2]
        modal_delta = torch.cat([common_mode - wet_water_transfer, common_mode + wet_water_transfer], dim=1)

        film_evidence = torch.clamp(stats[:, 9:10] + stats[:, 16:17] + stats[:, 22:23], 0.0, 1.0)
        mass_gate = torch.sigmoid((pair_mass - self.gate_threshold) * self.gate_temperature)
        smooth_gate = torch.sigmoid((smooth_share - 0.25) * 12.0)
        protected_gate = torch.sigmoid((0.70 - protected_mass / wetwater_concrete_mass.clamp_min(1e-6)) * 10.0)
        ambiguity = torch.sigmoid((0.40 - pair_margin) * 10.0)
        film_gate = 0.25 + 0.75 * torch.sigmoid((film_evidence - 0.12) * 12.0)
        gate = mass_gate * smooth_gate * protected_gate * ambiguity * film_gate

        residual = torch.zeros_like(base_logits)
        delta = modal_delta * gate.to(dtype=modal_delta.dtype) * self.scale
        residual.scatter_add_(1, target_idx.view(1, -1).expand(base_logits.size(0), -1), delta.to(dtype=residual.dtype))
        return residual


class ConcreteRoughnessVORResidual(DryConcreteRoughnessVORResidual):
    """Friction-charted concrete roughness residual.

    This generalizes the positive dry-concrete VOR chart while keeping the same
    safety contract: each chart is zero-sum inside one concrete
    smooth/slight/severe trio, so it cannot directly change material, friction
    state, or non-concrete logits. A chart embedding allows dry, wet, and water
    concrete to learn different visibility-to-roughness mappings.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 48,
        chart_dim: int = 6,
        scale: float = 0.05,
        gate_threshold: float = 0.12,
        gate_temperature: float = 14.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            zero_init=zero_init,
        )
        self.head = None
        groups: list[list[int]] = []
        names: list[str] = []
        for friction in ("dry", "wet", "water"):
            trio = [
                f"{friction}_concrete_smooth",
                f"{friction}_concrete_slight",
                f"{friction}_concrete_severe",
            ]
            if all(name in class_to_idx for name in trio):
                groups.append([int(class_to_idx[name]) for name in trio])
                names.append(friction)
        if not groups:
            raise ValueError("ConcreteRoughnessVORResidual found no concrete roughness trios.")
        self.chart_names = tuple(names)
        self.register_buffer("concrete_group_idx", torch.as_tensor(groups, dtype=torch.long))
        self.chart_embedding = nn.Embedding(len(groups), int(chart_dim))
        stat_dim = 40
        logit_dim = 10
        self.head = nn.Sequential(
            nn.LayerNorm(stat_dim + logit_dim + int(chart_dim)),
            nn.Linear(stat_dim + logit_dim + int(chart_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3),
        )
        nn.init.trunc_normal_(self.chart_embedding.weight, std=0.02)
        if zero_init:
            last = self.head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        group_idx = self.concrete_group_idx.to(device=base_logits.device)
        probs = F.softmax(base_logits, dim=1)
        stats = self._stats(image).to(device=base_logits.device, dtype=base_logits.dtype)
        chart_ids = torch.arange(group_idx.shape[0], device=base_logits.device)
        chart_embed = self.chart_embedding(chart_ids).to(dtype=base_logits.dtype)
        residual = torch.zeros_like(base_logits)
        for group_id in range(group_idx.shape[0]):
            idx = group_idx[group_id]
            group_logits = base_logits.index_select(1, idx)
            group_probs = probs.index_select(1, idx)
            group_mass = group_probs.sum(dim=1, keepdim=True)
            group_top = group_probs.amax(dim=1, keepdim=True)
            sorted_logits = group_logits.sort(dim=1, descending=True).values
            sorted_probs = group_probs.sort(dim=1, descending=True).values
            logit_features = torch.cat(
                [
                    group_logits,
                    group_probs,
                    group_mass,
                    group_top,
                    sorted_logits[:, 0:1] - sorted_logits[:, 1:2],
                    sorted_probs[:, 0:1] - sorted_probs[:, 1:2],
                ],
                dim=1,
            )
            chart_feature = chart_embed[group_id : group_id + 1].expand(base_logits.shape[0], -1)
            raw_delta = torch.tanh(
                self.head(torch.cat([stats, logit_features.to(dtype=stats.dtype), chart_feature], dim=1))
            )
            centered_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
            gate = torch.sigmoid((group_mass - self.gate_threshold) * self.gate_temperature)
            ambiguity = torch.sigmoid((0.42 - (sorted_probs[:, 0:1] - sorted_probs[:, 1:2])) * 10.0)
            delta = (
                centered_delta
                * gate.to(dtype=centered_delta.dtype)
                * ambiguity.to(dtype=centered_delta.dtype)
                * self.scale
            )
            residual.scatter_add_(
                1,
                idx.view(1, -1).expand(base_logits.size(0), -1),
                delta.to(dtype=residual.dtype),
            )
        return residual


class ObstructionAwareConcreteRoughnessVORResidual(DryConcreteRoughnessVORResidual):
    """Wet/water concrete roughness chart with film-obstruction evidence.

    Wet and water concrete are not just dry-concrete roughness plus a friction
    label. Water films erase high-frequency texture, create dark smooth pools
    or specular patches, and make slight/severe roughness visually nonuniform.
    This module therefore uses obstruction-conditioned physical fields and
    applies independent zero-sum corrections inside wet_concrete_* and
    water_concrete_* trios only.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 48,
        scale: float = 0.05,
        gate_threshold: float = 0.12,
        gate_temperature: float = 14.0,
        share_gate_threshold: float = 0.0,
        share_gate_temperature: float = 16.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            zero_init=zero_init,
        )
        self.head = None
        groups: list[list[int]] = []
        names: list[str] = []
        for friction in ("wet", "water"):
            trio = [
                f"{friction}_concrete_smooth",
                f"{friction}_concrete_slight",
                f"{friction}_concrete_severe",
            ]
            if all(name in class_to_idx for name in trio):
                groups.append([int(class_to_idx[name]) for name in trio])
                names.append(friction)
        if not groups:
            raise ValueError("ObstructionAwareConcreteRoughnessVORResidual found no wet/water concrete trios.")
        self.chart_names = tuple(names)
        self.register_buffer("obstruction_concrete_group_idx", torch.as_tensor(groups, dtype=torch.long))
        self.share_gate_threshold = float(share_gate_threshold)
        self.share_gate_temperature = float(share_gate_temperature)
        stat_dim = 48
        logit_dim = 12

        def make_head() -> nn.Sequential:
            head = nn.Sequential(
                nn.LayerNorm(stat_dim + logit_dim),
                nn.Linear(stat_dim + logit_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 3),
            )
            if zero_init:
                last = head[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            return head

        self.heads = nn.ModuleList([make_head() for _ in groups])

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
            0.45 * thin_film + 0.30 * dark_water + 0.20 * specular + 0.35 * texture_erasure,
            0.0,
            1.0,
        )
        snow_phase = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        marking = torch.sigmoid((value - 0.90) * 16.0) * torch.sigmoid((0.20 - saturation) * 14.0)
        road_mask = (1.0 - snow_phase) * (1.0 - marking)
        concrete_like = (
            torch.sigmoid((value - 0.30) * 8.0)
            * torch.sigmoid((0.86 - value) * 8.0)
            * torch.sigmoid((0.38 - saturation) * 9.0)
            * road_mask
        )
        visible_rough = rough_base * (1.0 - obstruction) * concrete_like
        hidden_rough = rough_base * obstruction * concrete_like
        water_smooth = torch.clamp(0.50 * dark_water + 0.35 * thin_film + 0.25 * texture_erasure, 0.0, 1.0)
        obstructed_concrete = obstruction * concrete_like
        water_obstructed_rough = hidden_rough * water_smooth

        global_stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3)),
                obstruction.mean(dim=(2, 3)),
                water_smooth.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        field_stats = [
            self._field_stats(rough_base),
            self._field_stats(visible_rough),
            self._field_stats(hidden_rough),
            self._field_stats(obstruction),
            self._field_stats(obstructed_concrete),
            self._field_stats(water_obstructed_rough),
        ]
        return torch.cat([global_stats, *field_stats], dim=1)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        group_idx = self.obstruction_concrete_group_idx.to(device=base_logits.device)
        probs = F.softmax(base_logits, dim=1)
        stats = self._stats(image).to(device=base_logits.device, dtype=base_logits.dtype)
        wetwater_concrete_mass = probs.index_select(1, group_idx.reshape(-1)).sum(dim=1, keepdim=True)
        residual = torch.zeros_like(base_logits)
        for group_id in range(group_idx.shape[0]):
            idx = group_idx[group_id]
            group_logits = base_logits.index_select(1, idx)
            group_probs = probs.index_select(1, idx)
            group_mass = group_probs.sum(dim=1, keepdim=True)
            group_top = group_probs.amax(dim=1, keepdim=True)
            group_share = group_mass / wetwater_concrete_mass.clamp_min(1e-6)
            sorted_logits = group_logits.sort(dim=1, descending=True).values
            sorted_probs = group_probs.sort(dim=1, descending=True).values
            logit_features = torch.cat(
                [
                    group_logits,
                    group_probs,
                    group_mass,
                    group_top,
                    sorted_logits[:, 0:1] - sorted_logits[:, 1:2],
                    sorted_probs[:, 0:1] - sorted_probs[:, 1:2],
                    wetwater_concrete_mass,
                    group_share,
                ],
                dim=1,
            )
            head_input = torch.cat([stats, logit_features.to(dtype=stats.dtype)], dim=1)
            raw_delta = torch.tanh(self.heads[group_id](head_input))
            centered_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
            gate = torch.sigmoid((group_mass - self.gate_threshold) * self.gate_temperature)
            if self.share_gate_threshold > 0.0:
                share_gate = torch.sigmoid((group_share - self.share_gate_threshold) * self.share_gate_temperature)
                gate = gate * share_gate
            ambiguity = torch.sigmoid((0.42 - (sorted_probs[:, 0:1] - sorted_probs[:, 1:2])) * 10.0)
            delta = centered_delta * gate.to(dtype=centered_delta.dtype) * ambiguity.to(dtype=centered_delta.dtype) * self.scale
            residual.scatter_add_(1, idx.view(1, -1).expand(base_logits.size(0), -1), delta.to(dtype=residual.dtype))
        return residual


class RoughnessNeighborResidual(CoupledOpticalRoughnessResidual):
    """Residual specialized for RSCD smooth/slight/severe neighbor confusions."""

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        hidden_dim: int = 96,
        scale: float = 0.10,
        gate_threshold: float = 0.42,
        gate_temperature: float = 10.0,
        protected_negative_limit: float = 0.0,
        neighbor_gate_floor: float = 0.15,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            protected_negative_limit=protected_negative_limit,
            zero_init=zero_init,
        )
        self.neighbor_gate_floor = float(neighbor_gate_floor)
        friction_vocab = {"dry": 0, "wet": 1, "water": 2}
        material_vocab = {"asphalt": 0, "concrete": 1}
        roughness_vocab = {"smooth": 0, "slight": 1, "severe": 2}
        num_classes = len(class_to_idx)
        friction_ids = torch.full((num_classes,), -1, dtype=torch.long)
        material_ids = torch.full((num_classes,), -1, dtype=torch.long)
        roughness_ids = torch.full((num_classes,), -1, dtype=torch.long)
        roughness_mask = torch.zeros(num_classes, dtype=torch.float32)
        for class_name, class_idx in class_to_idx.items():
            factors = _factor_text(class_name)
            idx = int(class_idx)
            friction_ids[idx] = friction_vocab.get(str(factors["friction"]), -1)
            material_ids[idx] = material_vocab.get(str(factors["material"]), -1)
            roughness_ids[idx] = roughness_vocab.get(str(factors["unevenness"]), -1)
            if friction_ids[idx] >= 0 and material_ids[idx] >= 0 and roughness_ids[idx] >= 0:
                roughness_mask[idx] = 1.0
        self.register_buffer("rough_friction_ids", friction_ids)
        self.register_buffer("rough_material_ids", material_ids)
        self.register_buffer("roughness_ids", roughness_ids)
        self.register_buffer("roughness_neighbor_mask", roughness_mask.view(1, -1))

    def forward(self, image: torch.Tensor, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        stats = self._optical_roughness_stats(image)
        residual = torch.tanh(self.head(torch.cat([feature, stats.to(dtype=feature.dtype)], dim=1))) * self.scale
        residual = residual * self.roughness_neighbor_mask.to(device=residual.device, dtype=residual.dtype)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            top2 = base_logits.topk(k=2, dim=1).indices
            c1, c2 = top2[:, 0], top2[:, 1]
            friction_ids = self.rough_friction_ids.to(device=base_logits.device)
            material_ids = self.rough_material_ids.to(device=base_logits.device)
            roughness_ids = self.roughness_ids.to(device=base_logits.device)
            same_friction = friction_ids[c1] == friction_ids[c2]
            same_material = material_ids[c1] == material_ids[c2]
            roughness_diff = roughness_ids[c1] != roughness_ids[c2]
            valid_roughness = (roughness_ids[c1] >= 0) & (roughness_ids[c2] >= 0)
            is_neighbor_case = (same_friction & same_material & roughness_diff & valid_roughness).view(-1, 1)
            neighbor_gate = torch.where(
                is_neighbor_case,
                torch.ones_like(uncertainty_gate),
                uncertainty_gate.new_full(uncertainty_gate.shape, self.neighbor_gate_floor),
            )
        return uncertainty_gate.to(dtype=residual.dtype) * neighbor_gate.to(dtype=residual.dtype) * residual


class RelationSignedGraphExpert(CoupledOpticalRoughnessResidual):
    """Relation-aware hard-cell expert for the heterophilic RSCD label graph.

    The RSCD label graph is not a pure smoothing graph: adjacent classes are
    often the exact classes that must be separated, e.g. wet vs water concrete
    or slight vs severe roughness. This expert therefore activates only when
    the current top-2 predictions form a hard graph edge touching one of the
    audited weak nodes, and it uses relation-specific zero-initialized offsets
    rather than undirected diffusion.
    """

    HARD_CLASS_NAMES = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "dry_concrete_slight",
        "wet_concrete_severe",
        "water_asphalt_severe",
    }

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        hidden_dim: int = 96,
        scale: float = 0.06,
        gate_threshold: float = 0.35,
        gate_temperature: float = 12.0,
        protected_negative_limit: float = 0.0,
        neighbor_gate_floor: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            protected_negative_limit=protected_negative_limit,
            zero_init=zero_init,
        )
        self.neighbor_gate_floor = float(neighbor_gate_floor)
        num_classes = len(class_to_idx)
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        hard = torch.zeros(num_classes, dtype=torch.float32)
        neighbor = torch.zeros((num_classes, num_classes), dtype=torch.bool)
        relation = torch.zeros((num_classes, num_classes), dtype=torch.long)

        for class_name, class_idx in class_to_idx.items():
            if class_name in self.HARD_CLASS_NAMES:
                hard[int(class_idx)] = 1.0

        for i in range(num_classes):
            a = _factor_text(idx_to_class[i])
            for j in range(num_classes):
                if i == j:
                    continue
                b = _factor_text(idx_to_class[j])
                same_material = a["material"] is not None and a["material"] == b["material"]
                same_uneven = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
                same_friction = a["friction"] is not None and a["friction"] == b["friction"]
                rel_id = 0
                if same_material and same_uneven and _friction_neighbors(a["friction"], b["friction"]):
                    rel_id = 1
                elif same_material and same_friction and _unevenness_neighbors(a["unevenness"], b["unevenness"]):
                    rel_id = 2
                elif same_friction and same_uneven and _material_neighbors(a["material"], b["material"]):
                    rel_id = 3
                if rel_id > 0 and (idx_to_class[i] in self.HARD_CLASS_NAMES or idx_to_class[j] in self.HARD_CLASS_NAMES):
                    neighbor[i, j] = True
                    relation[i, j] = rel_id

        self.register_buffer("hard_class_mask", hard.view(1, -1))
        self.register_buffer("hard_neighbor_mask", neighbor)
        self.register_buffer("hard_relation_idx", relation)
        self.relation_bias = nn.Parameter(torch.zeros(4, num_classes))

    def forward(self, image: torch.Tensor, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        stats = self._optical_roughness_stats(image)
        raw = self.head(torch.cat([feature, stats.to(dtype=feature.dtype)], dim=1))
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            top2 = base_logits.topk(k=2, dim=1).indices
            c1, c2 = top2[:, 0], top2[:, 1]
            neighbor = self.hard_neighbor_mask.to(device=base_logits.device)[c1, c2].view(-1, 1)
            relation_idx = self.hard_relation_idx.to(device=base_logits.device)[c1, c2]
            neighbor_gate = torch.where(
                neighbor,
                torch.ones_like(uncertainty_gate),
                uncertainty_gate.new_full(uncertainty_gate.shape, self.neighbor_gate_floor),
            )
        raw = raw + self.relation_bias.to(device=raw.device, dtype=raw.dtype).index_select(0, relation_idx)
        residual = torch.tanh(raw) * self.scale
        residual = residual * self.hard_class_mask.to(device=residual.device, dtype=residual.dtype)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return uncertainty_gate.to(dtype=residual.dtype) * neighbor_gate.to(dtype=residual.dtype) * residual


class HeterophilicLogitBoundaryExpert(nn.Module):
    """Pairwise top-2 logit reranker for heterophilic RSCD hard boundaries.

    This module distills the validation-fitted hard-pair reranker into a small
    trainable layer. It does not smooth the label graph. It only applies an
    antisymmetric correction to the two top-2 classes when they form an audited
    hard pair such as wet/water concrete or slight/severe concrete.
    """

    AUDITED_PAIRS = {
        ("dry_concrete_slight", "dry_concrete_severe"),
        ("water_concrete_smooth", "wet_concrete_smooth"),
        ("dry_mud", "dry_gravel"),
        ("wet_mud", "wet_gravel"),
        ("water_concrete_slight", "water_concrete_severe"),
        ("wet_concrete_slight", "wet_concrete_severe"),
        ("dry_asphalt_slight", "dry_asphalt_severe"),
        ("water_asphalt_slight", "water_asphalt_severe"),
        ("wet_asphalt_slight", "wet_asphalt_severe"),
    }

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        scale: float = 0.35,
        gate_threshold: float = 0.0,
        gate_temperature: float = 8.0,
        protected_negative_limit: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.protected_negative_limit = float(protected_negative_limit)
        num_classes = len(class_to_idx)
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        pairs = sorted(self._build_pairs(class_to_idx), key=lambda pair: (idx_to_class[pair[0]], idx_to_class[pair[1]]))
        if not pairs:
            raise ValueError("HeterophilicLogitBoundaryExpert found no valid RSCD hard pairs.")
        pair_lookup = torch.full((num_classes, num_classes), -1, dtype=torch.long)
        pair_a = []
        pair_b = []
        for pair_idx, (left, right) in enumerate(pairs):
            pair_lookup[left, right] = pair_idx
            pair_lookup[right, left] = pair_idx
            pair_a.append(left)
            pair_b.append(right)
        protected = torch.zeros(num_classes, dtype=torch.bool)
        for class_name, class_idx in class_to_idx.items():
            factors = _factor_text(class_name)
            if factors["friction"] in {"wet", "water"} or factors["unevenness"] == "severe":
                protected[int(class_idx)] = True
        self.register_buffer("pair_lookup", pair_lookup)
        self.register_buffer("pair_a", torch.tensor(pair_a, dtype=torch.long))
        self.register_buffer("pair_b", torch.tensor(pair_b, dtype=torch.long))
        self.register_buffer("protected_class_mask", protected.view(1, -1))
        self.weight = nn.Parameter(torch.zeros(len(pairs), 10))
        self.bias = nn.Parameter(torch.zeros(len(pairs)))

    @classmethod
    def _build_pairs(cls, class_to_idx: dict[str, int]) -> set[tuple[int, int]]:
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        pairs: set[tuple[int, int]] = set()
        for i, name_i in idx_to_class.items():
            factors_i = _factor_text(name_i)
            for j, name_j in idx_to_class.items():
                if i >= j:
                    continue
                factors_j = _factor_text(name_j)
                if cls._hard_relation(factors_i, factors_j):
                    pairs.add(tuple(sorted((i, j))))
        for left_name, right_name in cls.AUDITED_PAIRS:
            if left_name in class_to_idx and right_name in class_to_idx:
                pairs.add(tuple(sorted((class_to_idx[left_name], class_to_idx[right_name]))))
        return pairs

    @staticmethod
    def _hard_relation(a: dict[str, str | None], b: dict[str, str | None]) -> bool:
        same_material = a["material"] is not None and a["material"] == b["material"]
        same_uneven = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
        same_friction = a["friction"] is not None and a["friction"] == b["friction"]
        if same_material and same_uneven and _friction_neighbors(a["friction"], b["friction"]):
            return True
        if same_material and same_friction and _unevenness_neighbors(a["unevenness"], b["unevenness"]):
            return True
        if same_friction and same_uneven and _material_neighbors(a["material"], b["material"]):
            return True
        return False

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(base_logits, dim=1)
            top_prob, top_idx = probs.topk(k=2, dim=1)
            top_logits = base_logits.gather(1, top_idx)
            left = torch.minimum(top_idx[:, 0], top_idx[:, 1])
            right = torch.maximum(top_idx[:, 0], top_idx[:, 1])
            pair_ids = self.pair_lookup.to(device=base_logits.device)[left, right]
            active = pair_ids >= 0
            safe_pair_ids = pair_ids.clamp_min(0)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(base_logits.size(1)))
            margin = top_prob[:, 0] - top_prob[:, 1]
            uncertainty = 1.0 - top_prob[:, 0]
            gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            gate = gate * active.to(dtype=gate.dtype)
            prob_left = probs.gather(1, left.view(-1, 1)).squeeze(1)
            prob_right = probs.gather(1, right.view(-1, 1)).squeeze(1)
            logit_left = base_logits.gather(1, left.view(-1, 1)).squeeze(1)
            logit_right = base_logits.gather(1, right.view(-1, 1)).squeeze(1)
            features = torch.stack(
                [
                    logit_left - logit_right,
                    prob_left - prob_right,
                    logit_left,
                    logit_right,
                    prob_left,
                    prob_right,
                    top_prob[:, 0],
                    top_prob[:, 1],
                    margin,
                    entropy,
                ],
                dim=1,
            )
        selected_weight = self.weight.index_select(0, safe_pair_ids)
        selected_bias = self.bias.index_select(0, safe_pair_ids)
        score = (features.to(dtype=selected_weight.dtype) * selected_weight).sum(dim=1) + selected_bias
        delta = torch.tanh(score) * self.scale * gate.to(dtype=base_logits.dtype)
        residual = torch.zeros_like(base_logits)
        delta = delta.to(dtype=residual.dtype)
        residual.scatter_add_(1, left.view(-1, 1), -delta.view(-1, 1))
        residual.scatter_add_(1, right.view(-1, 1), delta.view(-1, 1))
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return residual


class HeterophilicFeatureBoundaryExpert(HeterophilicLogitBoundaryExpert):
    """Feature-conditioned hard-pair boundary expert for compositional RSCD labels.

    The logits-only boundary expert verified that hard-pair corrections are
    useful but too weak. This version conditions the pairwise antisymmetric
    correction on the classifier feature, so the decision can use backbone and
    PhysicsTexture evidence while still avoiding graph smoothing.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        hidden_dim: int = 96,
        pair_dim: int = 16,
        scale: float = 0.08,
        gate_threshold: float = 0.10,
        gate_temperature: float = 10.0,
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            protected_negative_limit=protected_negative_limit,
        )
        num_pairs = int(self.weight.shape[0])
        self.pair_embedding = nn.Embedding(num_pairs, int(pair_dim))
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(int(in_dim) + int(pair_dim) + 10),
            nn.Linear(int(in_dim) + int(pair_dim) + 10, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        if zero_init:
            last = self.boundary_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(base_logits, dim=1)
            top_prob, top_idx = probs.topk(k=2, dim=1)
            left = torch.minimum(top_idx[:, 0], top_idx[:, 1])
            right = torch.maximum(top_idx[:, 0], top_idx[:, 1])
            pair_ids = self.pair_lookup.to(device=base_logits.device)[left, right]
            active = pair_ids >= 0
            safe_pair_ids = pair_ids.clamp_min(0)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(base_logits.size(1)))
            margin = top_prob[:, 0] - top_prob[:, 1]
            uncertainty = 1.0 - top_prob[:, 0]
            gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            gate = gate * active.to(dtype=gate.dtype)
            prob_left = probs.gather(1, left.view(-1, 1)).squeeze(1)
            prob_right = probs.gather(1, right.view(-1, 1)).squeeze(1)
            logit_left = base_logits.gather(1, left.view(-1, 1)).squeeze(1)
            logit_right = base_logits.gather(1, right.view(-1, 1)).squeeze(1)
            logit_features = torch.stack(
                [
                    logit_left - logit_right,
                    prob_left - prob_right,
                    logit_left,
                    logit_right,
                    prob_left,
                    prob_right,
                    top_prob[:, 0],
                    top_prob[:, 1],
                    margin,
                    entropy,
                ],
                dim=1,
            )
        pair_feature = self.pair_embedding(safe_pair_ids)
        boundary_feature = torch.cat(
            [
                feature,
                pair_feature.to(dtype=feature.dtype),
                logit_features.to(device=feature.device, dtype=feature.dtype),
            ],
            dim=1,
        )
        score = self.boundary_head(boundary_feature).squeeze(1)
        delta = torch.tanh(score) * self.scale * gate.to(device=feature.device, dtype=score.dtype)
        residual = torch.zeros_like(base_logits)
        delta = delta.to(dtype=residual.dtype)
        residual.scatter_add_(1, left.view(-1, 1), -delta.view(-1, 1))
        residual.scatter_add_(1, right.view(-1, 1), delta.view(-1, 1))
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return residual


class HeterophilicPhysicsBoundaryExpert(HeterophilicLogitBoundaryExpert):
    """Low-dimensional physics-evidence expert for RSCD hard-pair boundaries.

    Human observers often rely on obvious color, context, and markings. This
    expert deliberately ignores high-dimensional semantic features and uses
    only compact optical/texture evidence: specular highlights, dark smooth
    water, low-texture films, rough gradient energy, snow-like whiteness, and
    local contrast. It is activated only for audited top-2 hard pairs.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 64,
        pair_dim: int = 12,
        scale: float = 0.08,
        gate_threshold: float = 0.10,
        gate_temperature: float = 10.0,
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            protected_negative_limit=protected_negative_limit,
        )
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
        num_pairs = int(self.weight.shape[0])
        self.pair_embedding = nn.Embedding(num_pairs, int(pair_dim))
        self.num_physics_stats = 28
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(self.num_physics_stats + int(pair_dim) + 10),
            nn.Linear(self.num_physics_stats + int(pair_dim) + 10, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        if zero_init:
            last = self.boundary_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _physics_stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
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
        granular = rough_energy * torch.sigmoid((local_contrast - 0.035) * 35.0)
        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        wet_proxy = torch.clamp(specular + 0.5 * dark_water + 0.35 * thin_film, 0.0, 1.0)
        gx_abs = gx.abs()
        gy_abs = gy.abs()
        anisotropy = (gx_abs.mean(dim=(2, 3)) - gy_abs.mean(dim=(2, 3))).abs() / (
            gx_abs.mean(dim=(2, 3)) + gy_abs.mean(dim=(2, 3)) + 1e-4
        )

        return torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                lap.mean(dim=(2, 3)),
                lap.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3)),
                specular.mean(dim=(2, 3)),
                specular.amax(dim=(2, 3)),
                dark_water.mean(dim=(2, 3)),
                dark_water.amax(dim=(2, 3)),
                thin_film.mean(dim=(2, 3)),
                thin_film.amax(dim=(2, 3)),
                texture_erasure.mean(dim=(2, 3)),
                texture_erasure.amax(dim=(2, 3)),
                low_texture.mean(dim=(2, 3)),
                low_contrast.mean(dim=(2, 3)),
                rough_energy.mean(dim=(2, 3)),
                rough_energy.std(dim=(2, 3)),
                granular.mean(dim=(2, 3)),
                snow_like.mean(dim=(2, 3)),
                snow_like.amax(dim=(2, 3)),
                anisotropy,
                _soft_connectedness(wet_proxy),
                _soft_connectedness(low_texture),
            ],
            dim=1,
        )

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(base_logits, dim=1)
            top_prob, top_idx = probs.topk(k=2, dim=1)
            left = torch.minimum(top_idx[:, 0], top_idx[:, 1])
            right = torch.maximum(top_idx[:, 0], top_idx[:, 1])
            pair_ids = self.pair_lookup.to(device=base_logits.device)[left, right]
            active = pair_ids >= 0
            safe_pair_ids = pair_ids.clamp_min(0)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(base_logits.size(1)))
            margin = top_prob[:, 0] - top_prob[:, 1]
            uncertainty = 1.0 - top_prob[:, 0]
            gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            gate = gate * active.to(dtype=gate.dtype)
            prob_left = probs.gather(1, left.view(-1, 1)).squeeze(1)
            prob_right = probs.gather(1, right.view(-1, 1)).squeeze(1)
            logit_left = base_logits.gather(1, left.view(-1, 1)).squeeze(1)
            logit_right = base_logits.gather(1, right.view(-1, 1)).squeeze(1)
            logit_features = torch.stack(
                [
                    logit_left - logit_right,
                    prob_left - prob_right,
                    logit_left,
                    logit_right,
                    prob_left,
                    prob_right,
                    top_prob[:, 0],
                    top_prob[:, 1],
                    margin,
                    entropy,
                ],
                dim=1,
            )
        physics_feature = self._physics_stats(image)
        pair_feature = self.pair_embedding(safe_pair_ids)
        boundary_feature = torch.cat(
            [
                physics_feature.to(dtype=pair_feature.dtype),
                pair_feature,
                logit_features.to(device=pair_feature.device, dtype=pair_feature.dtype),
            ],
            dim=1,
        )
        score = self.boundary_head(boundary_feature).squeeze(1)
        delta = torch.tanh(score) * self.scale * gate.to(device=score.device, dtype=score.dtype)
        residual = torch.zeros_like(base_logits)
        delta = delta.to(dtype=residual.dtype)
        residual.scatter_add_(1, left.view(-1, 1), -delta.view(-1, 1))
        residual.scatter_add_(1, right.view(-1, 1), delta.view(-1, 1))
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return residual


class ProtectedHeterophilicFactorBoundaryField(HeterophilicPhysicsBoundaryExpert):
    """Relation-conditioned physical boundary field for RSCD hard pairs.

    This is a narrower successor to the generic physical hard-pair expert. The
    complete RSCD error graph shows that hard pairs are mostly single-factor
    neighbors: roughness, wet/water friction, or material granularity. The
    module therefore injects a relation embedding into the antisymmetric
    top-2 correction and keeps the correction protected for fragile wet/water
    and severe classes.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 64,
        pair_dim: int = 12,
        relation_dim: int = 8,
        scale: float = 0.08,
        gate_threshold: float = 0.10,
        gate_temperature: float = 10.0,
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            pair_dim=pair_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            protected_negative_limit=protected_negative_limit,
            zero_init=zero_init,
        )
        num_classes = len(class_to_idx)
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        relation = torch.zeros((num_classes, num_classes), dtype=torch.long)
        fragile = self.protected_class_mask.view(-1).clone().to(dtype=torch.bool)
        fragile_names = {
            "water_asphalt_slight",
            "water_gravel",
            "water_mud",
            "water_concrete_slight",
            "wet_concrete_slight",
        }
        for class_name, class_idx in class_to_idx.items():
            if class_name in fragile_names:
                fragile[int(class_idx)] = True
        for i in range(num_classes):
            factors_i = _factor_text(idx_to_class[i])
            for j in range(num_classes):
                if i == j:
                    continue
                factors_j = _factor_text(idx_to_class[j])
                same_material = factors_i["material"] is not None and factors_i["material"] == factors_j["material"]
                same_uneven = factors_i["unevenness"] is not None and factors_i["unevenness"] == factors_j["unevenness"]
                same_friction = factors_i["friction"] is not None and factors_i["friction"] == factors_j["friction"]
                if same_material and same_uneven and _friction_neighbors(factors_i["friction"], factors_j["friction"]):
                    relation[i, j] = 1
                elif same_material and same_friction and _unevenness_neighbors(factors_i["unevenness"], factors_j["unevenness"]):
                    relation[i, j] = 2
                elif same_friction and same_uneven and _material_neighbors(factors_i["material"], factors_j["material"]):
                    relation[i, j] = 3
        self.register_buffer("boundary_relation_idx", relation)
        self.register_buffer("fragile_protected_class_mask", fragile.view(1, -1))
        self.relation_embedding = nn.Embedding(4, int(relation_dim))
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(self.num_physics_stats + int(pair_dim) + int(relation_dim) + 10),
            nn.Linear(self.num_physics_stats + int(pair_dim) + int(relation_dim) + 10, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        if zero_init:
            last = self.boundary_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(base_logits, dim=1)
            top_prob, top_idx = probs.topk(k=2, dim=1)
            left = torch.minimum(top_idx[:, 0], top_idx[:, 1])
            right = torch.maximum(top_idx[:, 0], top_idx[:, 1])
            pair_ids = self.pair_lookup.to(device=base_logits.device)[left, right]
            relation_idx = self.boundary_relation_idx.to(device=base_logits.device)[left, right]
            active = (pair_ids >= 0) & (relation_idx > 0)
            safe_pair_ids = pair_ids.clamp_min(0)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(base_logits.size(1)))
            margin = top_prob[:, 0] - top_prob[:, 1]
            uncertainty = 1.0 - top_prob[:, 0]
            gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            gate = gate * active.to(dtype=gate.dtype)
            prob_left = probs.gather(1, left.view(-1, 1)).squeeze(1)
            prob_right = probs.gather(1, right.view(-1, 1)).squeeze(1)
            logit_left = base_logits.gather(1, left.view(-1, 1)).squeeze(1)
            logit_right = base_logits.gather(1, right.view(-1, 1)).squeeze(1)
            logit_features = torch.stack(
                [
                    logit_left - logit_right,
                    prob_left - prob_right,
                    logit_left,
                    logit_right,
                    prob_left,
                    prob_right,
                    top_prob[:, 0],
                    top_prob[:, 1],
                    margin,
                    entropy,
                ],
                dim=1,
            )
        physics_feature = self._physics_stats(image)
        pair_feature = self.pair_embedding(safe_pair_ids)
        relation_feature = self.relation_embedding(relation_idx)
        boundary_feature = torch.cat(
            [
                physics_feature.to(dtype=pair_feature.dtype),
                pair_feature,
                relation_feature.to(dtype=pair_feature.dtype),
                logit_features.to(device=pair_feature.device, dtype=pair_feature.dtype),
            ],
            dim=1,
        )
        score = self.boundary_head(boundary_feature).squeeze(1)
        delta = torch.tanh(score) * self.scale * gate.to(device=score.device, dtype=score.dtype)
        residual = torch.zeros_like(base_logits)
        delta = delta.to(dtype=residual.dtype)
        residual.scatter_add_(1, left.view(-1, 1), -delta.view(-1, 1))
        residual.scatter_add_(1, right.view(-1, 1), delta.view(-1, 1))
        if bool(self.fragile_protected_class_mask.any()):
            protected_mask = self.fragile_protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return residual


class RelationSpecificHardEdgeRefiner(ProtectedHeterophilicFactorBoundaryField):
    """Relation-routed hard-edge refiner for RSCD compositional confusions.

    The complete RSCD graph shows three dominant hard-edge types: roughness
    boundaries, wet/water optical boundaries, and mud/gravel material
    granularity. A single boundary head can blur these mechanisms together.
    This refiner keeps the antisymmetric top-2 correction but routes each
    relation to its own small head fed by compact physics and topology cues.
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 64,
        pair_dim: int = 12,
        scale: float = 0.08,
        gate_threshold: float = 0.10,
        gate_temperature: float = 10.0,
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            pair_dim=pair_dim,
            relation_dim=1,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            protected_negative_limit=protected_negative_limit,
            zero_init=zero_init,
        )
        self.register_buffer("topology_thresholds", torch.linspace(0.15, 0.85, 8).view(1, 1, 8, 1, 1))
        # _field_summary returns 10 values; four mechanism fields give 40
        # relation-routed physics/topology statistics.
        self.relation_stat_dim = 40
        head_in = self.num_physics_stats + self.relation_stat_dim + int(pair_dim) + 10

        def make_head() -> nn.Sequential:
            head = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 1),
            )
            if zero_init:
                last = head[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            return head

        # relation id 1=friction/wet-water, 2=roughness, 3=material.
        self.friction_head = make_head()
        self.roughness_head = make_head()
        self.material_head = make_head()

    def _field_summary(self, field: torch.Tensor) -> torch.Tensor:
        topo = _soft_euler_curve_stats(field, self.topology_thresholds)
        return torch.cat(
            [
                field.mean(dim=(2, 3)),
                field.std(dim=(2, 3)),
                field.amax(dim=(2, 3)),
                _soft_connectedness(field),
                topo,
            ],
            dim=1,
        )

    def _relation_stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 96:
            rgb = F.interpolate(rgb, size=(96, 96), mode="bilinear", align_corners=False)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        grad_norm = _normalize_map(grad)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
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
        rough_energy = torch.sigmoid((grad - 0.075) * 22.0)
        granular = rough_energy * torch.sigmoid((local_contrast - 0.035) * 35.0)
        rough_fragment = torch.clamp(0.6 * grad_norm + 0.4 * contrast_norm, 0.0, 1.0)

        # Four summaries x 10 stats = 40 dims. These are relation-targeted:
        # wet film, texture erasure, rough fragments, and granular particles.
        return torch.cat(
            [
                self._field_summary(wet_proxy),
                self._field_summary(thin_film * low_contrast),
                self._field_summary(rough_fragment),
                self._field_summary(granular),
            ],
            dim=1,
        )

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(base_logits, dim=1)
            top_prob, top_idx = probs.topk(k=2, dim=1)
            left = torch.minimum(top_idx[:, 0], top_idx[:, 1])
            right = torch.maximum(top_idx[:, 0], top_idx[:, 1])
            pair_ids = self.pair_lookup.to(device=base_logits.device)[left, right]
            relation_idx = self.boundary_relation_idx.to(device=base_logits.device)[left, right]
            active = (pair_ids >= 0) & (relation_idx > 0)
            safe_pair_ids = pair_ids.clamp_min(0)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(base_logits.size(1)))
            margin = top_prob[:, 0] - top_prob[:, 1]
            uncertainty = 1.0 - top_prob[:, 0]
            gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            gate = gate * active.to(dtype=gate.dtype)
            prob_left = probs.gather(1, left.view(-1, 1)).squeeze(1)
            prob_right = probs.gather(1, right.view(-1, 1)).squeeze(1)
            logit_left = base_logits.gather(1, left.view(-1, 1)).squeeze(1)
            logit_right = base_logits.gather(1, right.view(-1, 1)).squeeze(1)
            logit_features = torch.stack(
                [
                    logit_left - logit_right,
                    prob_left - prob_right,
                    logit_left,
                    logit_right,
                    prob_left,
                    prob_right,
                    top_prob[:, 0],
                    top_prob[:, 1],
                    margin,
                    entropy,
                ],
                dim=1,
            )
        physics_feature = self._physics_stats(image)
        relation_stats = self._relation_stats(image)
        pair_feature = self.pair_embedding(safe_pair_ids)
        boundary_feature = torch.cat(
            [
                physics_feature.to(dtype=pair_feature.dtype),
                relation_stats.to(dtype=pair_feature.dtype),
                pair_feature,
                logit_features.to(device=pair_feature.device, dtype=pair_feature.dtype),
            ],
            dim=1,
        )
        zero = torch.zeros((base_logits.size(0),), device=base_logits.device, dtype=boundary_feature.dtype)
        scores = torch.stack(
            [
                zero,
                self.friction_head(boundary_feature).squeeze(1),
                self.roughness_head(boundary_feature).squeeze(1),
                self.material_head(boundary_feature).squeeze(1),
            ],
            dim=1,
        )
        score = scores.gather(1, relation_idx.clamp(0, 3).view(-1, 1)).squeeze(1)
        delta = torch.tanh(score) * self.scale * gate.to(device=score.device, dtype=score.dtype)
        residual = torch.zeros_like(base_logits)
        delta = delta.to(dtype=residual.dtype)
        residual.scatter_add_(1, left.view(-1, 1), -delta.view(-1, 1))
        residual.scatter_add_(1, right.view(-1, 1), delta.view(-1, 1))
        if bool(self.fragile_protected_class_mask.any()):
            protected_mask = self.fragile_protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return residual


class SelectiveMechanismTensorBoundaryField(ProtectedHeterophilicFactorBoundaryField):
    """Mechanism-charted high-order boundary field for RSCD factor couplings.

    A global low-rank tensor treats all triples as variations of the same
    algebraic pattern, but RSCD's hard cells are mechanism-specific: wet asphalt
    sheen, concrete water-film erasure, dry visible micro-texture, granular
    material, and winter transitions expose different image statistics. This
    module keeps the top-2 antisymmetric boundary correction, but routes each
    candidate edge to a mechanism-specific field and gates it by matching visual
    evidence. It therefore models the high-order term H_fmr only where a
    factor-neighbor boundary is actually active.
    """

    MECHANISM_NAMES = (
        "inactive",
        "dry_visible_microtexture",
        "asphalt_wet_sheen",
        "concrete_thin_film",
        "water_obstructed_roughness",
        "wet_water_smooth_transition",
        "granular_material",
        "winter_low_friction",
        "paved_material_under_state",
    )

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        hidden_dim: int = 64,
        pair_dim: int = 12,
        relation_dim: int = 6,
        mechanism_dim: int = 8,
        scale: float = 0.08,
        gate_threshold: float = 0.10,
        gate_temperature: float = 10.0,
        mechanism_gate_threshold: float = 0.08,
        mechanism_gate_temperature: float = 12.0,
        enabled_mechanisms: str = "all",
        protected_negative_limit: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(
            class_to_idx=class_to_idx,
            hidden_dim=hidden_dim,
            pair_dim=pair_dim,
            relation_dim=relation_dim,
            scale=scale,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            protected_negative_limit=protected_negative_limit,
            zero_init=zero_init,
        )
        self.mechanism_gate_threshold = float(mechanism_gate_threshold)
        self.mechanism_gate_temperature = float(mechanism_gate_temperature)
        self.num_mechanisms = len(self.MECHANISM_NAMES)
        self.mechanism_stat_dim = 10
        self.register_buffer("enabled_mechanism_mask", self._parse_enabled_mechanisms(enabled_mechanisms))
        self.register_buffer("topology_thresholds", torch.linspace(0.15, 0.85, 8).view(1, 1, 8, 1, 1))
        self.register_buffer("boundary_mechanism_idx", self._build_mechanism_index(class_to_idx))
        self.mechanism_embedding = nn.Embedding(self.num_mechanisms, int(mechanism_dim))
        head_in = (
            self.num_physics_stats
            + self.mechanism_stat_dim
            + int(pair_dim)
            + int(relation_dim)
            + int(mechanism_dim)
            + 10
        )

        def make_head() -> nn.Sequential:
            head = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 1),
            )
            if zero_init:
                last = head[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            return head

        self.mechanism_heads = nn.ModuleList(make_head() for _ in range(self.num_mechanisms - 1))

    @classmethod
    def _parse_enabled_mechanisms(cls, value: str) -> torch.Tensor:
        mask = torch.zeros(len(cls.MECHANISM_NAMES), dtype=torch.bool)
        text = str(value or "all").strip().lower()
        if text in {"all", "*"}:
            mask[1:] = True
            return mask
        if text in {"none", "off"}:
            return mask
        name_to_idx = {name.lower(): idx for idx, name in enumerate(cls.MECHANISM_NAMES)}
        aliases = {
            "dry": "dry_visible_microtexture",
            "asphalt": "asphalt_wet_sheen",
            "concrete": "concrete_thin_film",
            "obstruction": "water_obstructed_roughness",
            "water_obstruction": "water_obstructed_roughness",
            "smooth": "wet_water_smooth_transition",
            "granular": "granular_material",
            "winter": "winter_low_friction",
            "paved": "paved_material_under_state",
        }
        for item in text.split(","):
            key = item.strip().lower()
            if not key:
                continue
            key = aliases.get(key, key)
            if key.isdigit():
                idx = int(key)
            else:
                idx = name_to_idx.get(key, -1)
            if idx <= 0 or idx >= len(cls.MECHANISM_NAMES):
                raise ValueError(
                    f"unknown SM-TBF mechanism '{item}'. Valid names: {', '.join(cls.MECHANISM_NAMES[1:])}"
                )
            mask[idx] = True
        return mask

    @classmethod
    def _build_mechanism_index(cls, class_to_idx: dict[str, int]) -> torch.Tensor:
        num_classes = len(class_to_idx)
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        mechanism = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for i in range(num_classes):
            factors_i = _factor_text(idx_to_class[i])
            for j in range(num_classes):
                if i == j:
                    continue
                factors_j = _factor_text(idx_to_class[j])
                mechanism[i, j] = cls._mechanism_for_pair(factors_i, factors_j)
        return mechanism

    @staticmethod
    def _mechanism_for_pair(a: dict[str, str | None], b: dict[str, str | None]) -> int:
        frictions = {value for value in (a["friction"], b["friction"]) if value is not None}
        materials = {value for value in (a["material"], b["material"]) if value is not None}
        unevenness = {value for value in (a["unevenness"], b["unevenness"]) if value is not None}
        same_material = a["material"] is not None and a["material"] == b["material"]
        same_uneven = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
        same_friction = a["friction"] is not None and a["friction"] == b["friction"]
        friction_edge = same_material and same_uneven and _friction_neighbors(a["friction"], b["friction"])
        roughness_edge = same_material and same_friction and _unevenness_neighbors(a["unevenness"], b["unevenness"])
        material_edge = same_friction and same_uneven and _material_neighbors(a["material"], b["material"])
        if not (friction_edge or roughness_edge or material_edge):
            return 0
        if frictions.intersection({"fresh_snow", "melted_snow", "ice"}):
            return 7
        if materials == {"mud", "gravel"} or materials.intersection({"mud", "gravel"}):
            return 6
        if material_edge:
            if frictions.intersection({"wet", "water"}):
                return 8
            return 1
        if friction_edge:
            if frictions == {"wet", "water"}:
                if "concrete" in materials and unevenness.intersection({"slight", "severe"}):
                    return 4
                if "concrete" in materials:
                    return 3
                if "asphalt" in materials:
                    return 2 if unevenness.intersection({"slight", "severe"}) else 5
                return 8
            if frictions == {"dry", "wet"}:
                if "concrete" in materials:
                    return 3
                if "asphalt" in materials:
                    return 2
                return 8
        if roughness_edge:
            if frictions == {"dry"}:
                return 1
            if "concrete" in materials and frictions.intersection({"wet", "water"}):
                return 4
            if "asphalt" in materials and frictions.intersection({"wet", "water"}):
                return 2
            return 8
        return 8

    def _field_summary(self, field: torch.Tensor) -> torch.Tensor:
        topo = _soft_euler_curve_stats(field, self.topology_thresholds)
        return torch.cat(
            [
                field.mean(dim=(2, 3)),
                field.std(dim=(2, 3)),
                field.amax(dim=(2, 3)),
                _soft_connectedness(field),
                topo,
            ],
            dim=1,
        )

    def _mechanism_fields(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        if max(rgb.shape[-2:]) > 96:
            rgb = F.interpolate(rgb, size=(96, 96), mode="bilinear", align_corners=False)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        grad_norm = _normalize_map(grad)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
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
        rough_fragment = torch.clamp(0.6 * grad_norm + 0.4 * contrast_norm, 0.0, 1.0)
        granular = rough_energy * torch.sigmoid((local_contrast - 0.035) * 35.0)
        snow_like = torch.sigmoid((value - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
        concrete_like = (
            torch.sigmoid((value - 0.44) * 8.0)
            * torch.sigmoid((0.78 - value) * 8.0)
            * torch.sigmoid((0.35 - saturation) * 10.0)
        )
        asphalt_like = torch.sigmoid((0.48 - value) * 8.0) * torch.sigmoid((grad - 0.035) * 18.0)

        dry_visible = (1.0 - wet_proxy).clamp(0.0, 1.0) * torch.clamp(0.55 * rough_fragment + 0.45 * contrast_norm, 0.0, 1.0)
        asphalt_sheen = torch.clamp(0.65 * wet_proxy + 0.35 * asphalt_like * (1.0 - low_contrast), 0.0, 1.0)
        concrete_film = torch.clamp(0.55 * thin_film + 0.45 * concrete_like * texture_erasure, 0.0, 1.0)
        water_obstructed = torch.clamp(0.55 * dark_water + 0.45 * texture_erasure, 0.0, 1.0)
        wet_water_smooth = torch.clamp(0.55 * wet_proxy + 0.45 * low_texture * low_contrast, 0.0, 1.0)
        paved_material = torch.clamp((0.5 * concrete_like + 0.5 * asphalt_like) * (0.55 + 0.45 * wet_proxy), 0.0, 1.0)
        zero = torch.zeros_like(dry_visible)
        fields = [
            zero,
            dry_visible,
            asphalt_sheen,
            concrete_film,
            water_obstructed,
            wet_water_smooth,
            granular,
            snow_like,
            paved_material,
        ]
        summaries = torch.stack([self._field_summary(field) for field in fields], dim=1)
        scores = torch.cat([field.mean(dim=(2, 3)) for field in fields], dim=1)
        return summaries, scores

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(base_logits, dim=1)
            top_prob, top_idx = probs.topk(k=2, dim=1)
            left = torch.minimum(top_idx[:, 0], top_idx[:, 1])
            right = torch.maximum(top_idx[:, 0], top_idx[:, 1])
            pair_ids = self.pair_lookup.to(device=base_logits.device)[left, right]
            relation_idx = self.boundary_relation_idx.to(device=base_logits.device)[left, right]
            mechanism_idx = self.boundary_mechanism_idx.to(device=base_logits.device)[left, right]
            enabled = self.enabled_mechanism_mask.to(device=base_logits.device)[mechanism_idx.clamp(0, self.num_mechanisms - 1)]
            active = (pair_ids >= 0) & (relation_idx > 0) & (mechanism_idx > 0) & enabled
            safe_pair_ids = pair_ids.clamp_min(0)
            safe_relation_idx = relation_idx.clamp(0, 3)
            safe_mechanism_idx = mechanism_idx.clamp(0, self.num_mechanisms - 1)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(base_logits.size(1)))
            margin = top_prob[:, 0] - top_prob[:, 1]
            uncertainty = 1.0 - top_prob[:, 0]
            uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            prob_left = probs.gather(1, left.view(-1, 1)).squeeze(1)
            prob_right = probs.gather(1, right.view(-1, 1)).squeeze(1)
            logit_left = base_logits.gather(1, left.view(-1, 1)).squeeze(1)
            logit_right = base_logits.gather(1, right.view(-1, 1)).squeeze(1)
            logit_features = torch.stack(
                [
                    logit_left - logit_right,
                    prob_left - prob_right,
                    logit_left,
                    logit_right,
                    prob_left,
                    prob_right,
                    top_prob[:, 0],
                    top_prob[:, 1],
                    margin,
                    entropy,
                ],
                dim=1,
            )
        physics_feature = self._physics_stats(image)
        mechanism_summaries, mechanism_scores = self._mechanism_fields(image)
        selected_summary = mechanism_summaries.gather(
            1,
            safe_mechanism_idx.view(-1, 1, 1).expand(-1, 1, self.mechanism_stat_dim),
        ).squeeze(1)
        selected_score = mechanism_scores.gather(1, safe_mechanism_idx.view(-1, 1)).squeeze(1)
        mechanism_gate = torch.sigmoid(
            (selected_score - self.mechanism_gate_threshold) * self.mechanism_gate_temperature
        )
        pair_feature = self.pair_embedding(safe_pair_ids)
        relation_feature = self.relation_embedding(safe_relation_idx)
        mechanism_feature = self.mechanism_embedding(safe_mechanism_idx)
        boundary_feature = torch.cat(
            [
                physics_feature.to(dtype=pair_feature.dtype),
                selected_summary.to(dtype=pair_feature.dtype),
                pair_feature,
                relation_feature.to(dtype=pair_feature.dtype),
                mechanism_feature.to(dtype=pair_feature.dtype),
                logit_features.to(device=pair_feature.device, dtype=pair_feature.dtype),
            ],
            dim=1,
        )
        zero = torch.zeros((base_logits.size(0),), device=base_logits.device, dtype=boundary_feature.dtype)
        scores = torch.stack([zero] + [head(boundary_feature).squeeze(1) for head in self.mechanism_heads], dim=1)
        score = scores.gather(1, safe_mechanism_idx.view(-1, 1)).squeeze(1)
        gate = (
            uncertainty_gate.to(device=score.device, dtype=score.dtype)
            * mechanism_gate.to(device=score.device, dtype=score.dtype)
            * active.to(device=score.device, dtype=score.dtype)
        )
        delta = torch.tanh(score) * self.scale * gate
        residual = torch.zeros_like(base_logits)
        delta = delta.to(dtype=residual.dtype)
        residual.scatter_add_(1, left.view(-1, 1), -delta.view(-1, 1))
        residual.scatter_add_(1, right.view(-1, 1), delta.view(-1, 1))
        if bool(self.fragile_protected_class_mask.any()):
            protected_mask = self.fragile_protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        return residual


class SpectralRoughnessResidual(nn.Module):
    """Multi-scale optical-frequency residual for roughness-neighbor errors.

    Human observers often judge road roughness from salient marks or brightness.
    RSCD's largest residual error mode is different: smooth/slight/severe labels
    are confused within the same friction and material state. This module uses
    patch-scale band-pass texture energies as a macro/micro-texture cue and only
    activates on uncertain roughness-neighbor top-2 cases.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        hidden_dim: int = 96,
        scale: float = 0.08,
        gate_threshold: float = 0.35,
        gate_temperature: float = 12.0,
        protected_negative_limit: float = 0.0,
        neighbor_gate_floor: float = 0.02,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.protected_negative_limit = max(float(protected_negative_limit), 0.0)
        self.neighbor_gate_floor = float(neighbor_gate_floor)
        num_classes = len(class_to_idx)
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
        friction_vocab = {"dry": 0, "wet": 1, "water": 2}
        material_vocab = {"asphalt": 0, "concrete": 1}
        roughness_vocab = {"smooth": 0, "slight": 1, "severe": 2}
        friction_ids = torch.full((num_classes,), -1, dtype=torch.long)
        material_ids = torch.full((num_classes,), -1, dtype=torch.long)
        roughness_ids = torch.full((num_classes,), -1, dtype=torch.long)
        rough_class_mask = torch.zeros(num_classes, dtype=torch.float32)
        protected = torch.zeros(num_classes, dtype=torch.bool)
        for class_name, class_idx in class_to_idx.items():
            factors = _factor_text(class_name)
            idx = int(class_idx)
            friction_ids[idx] = friction_vocab.get(str(factors["friction"]), -1)
            material_ids[idx] = material_vocab.get(str(factors["material"]), -1)
            roughness_ids[idx] = roughness_vocab.get(str(factors["unevenness"]), -1)
            if friction_ids[idx] >= 0 and material_ids[idx] >= 0 and roughness_ids[idx] >= 0:
                rough_class_mask[idx] = 1.0
            if factors["friction"] in {"wet", "water"}:
                protected[idx] = True
        self.register_buffer("spectral_friction_ids", friction_ids)
        self.register_buffer("spectral_material_ids", material_ids)
        self.register_buffer("spectral_roughness_ids", roughness_ids)
        self.register_buffer("rough_class_mask", rough_class_mask.view(1, -1))
        self.register_buffer("protected_class_mask", protected.view(1, -1))
        stat_dim = 30
        self.head = nn.Sequential(
            nn.LayerNorm(int(in_dim) + stat_dim),
            nn.Linear(int(in_dim) + stat_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), num_classes),
        )
        if zero_init:
            last = self.head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, image: torch.Tensor, feature: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        stats = self._spectral_stats(image).to(dtype=feature.dtype)
        residual = torch.tanh(self.head(torch.cat([feature, stats], dim=1))) * self.scale
        residual = residual * self.rough_class_mask.to(device=residual.device, dtype=residual.dtype)
        if bool(self.protected_class_mask.any()):
            protected_mask = self.protected_class_mask.to(device=residual.device)
            floor = -self.protected_negative_limit
            residual = torch.where(protected_mask & (residual < floor), residual.new_full((), floor), residual)
        with torch.no_grad():
            prob = F.softmax(base_logits, dim=1)
            uncertainty = 1.0 - prob.amax(dim=1, keepdim=True)
            uncertainty_gate = torch.sigmoid((uncertainty - self.gate_threshold) * self.gate_temperature)
            top2 = base_logits.topk(k=2, dim=1).indices
            c1, c2 = top2[:, 0], top2[:, 1]
            friction_ids = self.spectral_friction_ids.to(device=base_logits.device)
            material_ids = self.spectral_material_ids.to(device=base_logits.device)
            roughness_ids = self.spectral_roughness_ids.to(device=base_logits.device)
            same_friction = friction_ids[c1] == friction_ids[c2]
            same_material = material_ids[c1] == material_ids[c2]
            roughness_step = (roughness_ids[c1] - roughness_ids[c2]).abs()
            valid = (roughness_ids[c1] >= 0) & (roughness_ids[c2] >= 0)
            neighbor = (same_friction & same_material & valid & (roughness_step == 1)).view(-1, 1)
            neighbor_gate = torch.where(
                neighbor,
                torch.ones_like(uncertainty_gate),
                uncertainty_gate.new_full(uncertainty_gate.shape, self.neighbor_gate_floor),
            )
        return uncertainty_gate.to(dtype=residual.dtype) * neighbor_gate.to(dtype=residual.dtype) * residual

    def _spectral_stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        blur3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        blur9 = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        blur21 = F.avg_pool2d(gray, kernel_size=21, stride=1, padding=10)
        blur45 = F.avg_pool2d(gray, kernel_size=45, stride=1, padding=22)
        micro = (gray - blur3).abs()
        meso = (blur3 - blur9).abs()
        macro = (blur9 - blur21).abs()
        illumination = (blur21 - blur45).abs()
        grad_micro = F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)
        grad_meso = F.avg_pool2d(grad, kernel_size=9, stride=1, padding=4)
        grad_macro = F.avg_pool2d(grad, kernel_size=21, stride=1, padding=10)
        high_texture = torch.sigmoid((micro + grad_micro - 0.075) * 24.0)
        low_texture = torch.sigmoid((0.045 - grad_meso) * 35.0)
        rough_patch = torch.sigmoid((grad_meso + macro - 0.080) * 20.0)
        fields = [micro, meso, macro, illumination, grad_micro, grad_meso, grad_macro, high_texture, rough_patch]
        stats = []
        for field in fields:
            stats.extend([field.mean(dim=(2, 3)), field.std(dim=(2, 3)), field.amax(dim=(2, 3))])
        denom = meso.mean(dim=(2, 3)).clamp_min(1e-4)
        stats.extend(
            [
                micro.mean(dim=(2, 3)) / denom,
                macro.mean(dim=(2, 3)) / denom,
                _soft_connectedness(low_texture),
            ]
        )
        return torch.cat(stats, dim=1)


class ArtifactAwareTextureGate(nn.Module):
    """Image-statistics gate for low-level texture reliability.

    RSCD contains both real friction evidence and shortcut-like artifacts:
    lane markings, grates, pale concrete, glare, puddle reflections, and shadow
    edges. This gate does not erase pixels. It summarizes ambiguity/reliability
    cues and learns a zero-initialized multiplicative correction for low-level
    physics-texture features.
    """

    def __init__(
        self,
        *,
        low_level_dim: int,
        scale: float = 0.20,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
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
        self.head = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, int(low_level_dim)),
        )
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, image: torch.Tensor, low_level_feature: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        near_white = torch.sigmoid((value - 0.86) * 16.0) * torch.sigmoid((0.24 - saturation) * 14.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_smooth = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * torch.sigmoid((0.050 - grad) * 35.0)
        )
        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        strong_edge = torch.sigmoid((grad - 0.12) * 20.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        wet_proxy = torch.clamp(specular + 0.5 * dark_smooth, 0.0, 1.0)
        white_edge = near_white * strong_edge

        gx_abs = gx.abs()
        gy_abs = gy.abs()
        anisotropy = (gx_abs.mean(dim=(2, 3)) - gy_abs.mean(dim=(2, 3))).abs() / (
            gx_abs.mean(dim=(2, 3)) + gy_abs.mean(dim=(2, 3)) + 1e-4
        )
        stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                near_white.mean(dim=(2, 3)),
                white_edge.mean(dim=(2, 3)),
                specular.mean(dim=(2, 3)),
                dark_smooth.mean(dim=(2, 3)),
                wet_proxy.mean(dim=(2, 3)),
                low_texture.mean(dim=(2, 3)),
                low_contrast.mean(dim=(2, 3)),
                anisotropy,
                _soft_connectedness(wet_proxy),
            ],
            dim=1,
        )
        delta = torch.tanh(self.head(stats))
        return low_level_feature * (1.0 + self.scale * delta)


class SmoothEvidenceTextureGate(nn.Module):
    """Smooth-state low-level evidence gate for RSCD compositional labels.

    The fair RSPNet-L audit shows that the remaining Top-1 gap is concentrated
    in high-support smooth cells, while wet/water slight and severe cells are
    already a strength of the current PhysicsTexture route. This gate therefore
    does not apply a global artifact correction. It uses image-level optical
    evidence for true smoothness and only modulates low-level physics-texture
    features when micro/macro texture statistics look smooth rather than rough,
    granular, or broken water-film like.
    """

    def __init__(
        self,
        *,
        low_level_dim: int,
        scale: float = 0.10,
        smooth_temperature: float = 16.0,
        rough_suppression: float = 0.65,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.smooth_temperature = float(smooth_temperature)
        self.rough_suppression = min(1.0, max(0.0, float(rough_suppression)))
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
        stat_dim = 28
        self.head = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, 72),
            nn.GELU(),
            nn.Linear(72, int(low_level_dim)),
        )
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, image: torch.Tensor, low_level_feature: torch.Tensor) -> torch.Tensor:
        stats, smooth_gate = self._stats_and_gate(image)
        stats = stats.to(device=low_level_feature.device, dtype=low_level_feature.dtype)
        smooth_gate = smooth_gate.to(device=low_level_feature.device, dtype=low_level_feature.dtype)
        delta = torch.tanh(self.head(stats))
        return low_level_feature * (1.0 + self.scale * smooth_gate * delta)

    def _stats_and_gate(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        blur3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        blur9 = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        blur21 = F.avg_pool2d(gray, kernel_size=21, stride=1, padding=10)
        micro = (gray - blur3).abs()
        meso = (blur3 - blur9).abs()
        macro = (blur9 - blur21).abs()
        local_mean = blur9
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)

        near_white = torch.sigmoid((value - 0.86) * 16.0) * torch.sigmoid((0.24 - saturation) * 14.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_smooth = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * torch.sigmoid((0.050 - grad) * 35.0)
        )
        wet_proxy = torch.clamp(specular + 0.5 * dark_smooth, 0.0, 1.0)
        strong_edge = torch.sigmoid((grad - 0.12) * 20.0)
        white_edge = near_white * strong_edge
        rough_patch = torch.sigmoid((grad + macro - 0.090) * 22.0)
        granular_proxy = torch.sigmoid((micro + meso + grad - 0.115) * 18.0)
        smooth_field = (
            torch.sigmoid((0.060 - grad) * self.smooth_temperature)
            * torch.sigmoid((0.030 - macro) * self.smooth_temperature)
            * torch.sigmoid((0.040 - local_contrast) * self.smooth_temperature)
        )
        broken_film = wet_proxy * rough_patch
        real_smooth = (smooth_field * (1.0 - self.rough_suppression * rough_patch) * (1.0 - 0.45 * granular_proxy)).clamp(
            0.0,
            1.0,
        )
        smooth_gate = real_smooth.mean(dim=(2, 3))

        gx_abs = gx.abs()
        gy_abs = gy.abs()
        anisotropy = (gx_abs.mean(dim=(2, 3)) - gy_abs.mean(dim=(2, 3))).abs() / (
            gx_abs.mean(dim=(2, 3)) + gy_abs.mean(dim=(2, 3)) + 1e-4
        )
        stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                micro.mean(dim=(2, 3)),
                micro.std(dim=(2, 3)),
                meso.mean(dim=(2, 3)),
                meso.std(dim=(2, 3)),
                macro.mean(dim=(2, 3)),
                macro.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                local_contrast.std(dim=(2, 3)),
                near_white.mean(dim=(2, 3)),
                white_edge.mean(dim=(2, 3)),
                specular.mean(dim=(2, 3)),
                dark_smooth.mean(dim=(2, 3)),
                wet_proxy.mean(dim=(2, 3)),
                rough_patch.mean(dim=(2, 3)),
                granular_proxy.mean(dim=(2, 3)),
                broken_film.mean(dim=(2, 3)),
                real_smooth.mean(dim=(2, 3)),
                real_smooth.std(dim=(2, 3)),
                real_smooth.amax(dim=(2, 3)),
                anisotropy,
                _soft_connectedness(real_smooth),
                _soft_connectedness(broken_film),
            ],
            dim=1,
        )
        return stats, smooth_gate


class TriChartEvidenceFiLM(nn.Module):
    """Mutually gated evidence charts for early feature conditioning.

    The RSCD audit separates three mechanisms that should not share one gate:
    smooth-state negative evidence, wet/water-film evidence, and granular
    mud/gravel evidence. This module lets each chart produce a zero-initialized
    FiLM update for the backbone feature, then gates the charts with deterministic
    optical statistics so smooth gains do not overwrite water-film or granular
    evidence.
    """

    def __init__(
        self,
        *,
        low_level_dim: int,
        embedding_dim: int,
        hidden_dim: int = 256,
        scale: float = 0.035,
        gate_temperature: float = 18.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gate_temperature = float(gate_temperature)
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
        stat_dim = 24
        input_dim = int(low_level_dim) + stat_dim
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), 2 * int(embedding_dim)),
                )
                for _ in range(3)
            ]
        )
        for expert in self.experts:
            last = expert[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        image: torch.Tensor,
        backbone_feature: torch.Tensor,
        low_level_feature: torch.Tensor,
    ) -> torch.Tensor:
        stats, gates = self._stats_and_gates(image)
        stats = stats.to(device=low_level_feature.device, dtype=low_level_feature.dtype)
        gates = gates.to(device=low_level_feature.device, dtype=low_level_feature.dtype)
        chart_input = torch.cat([low_level_feature, stats], dim=1)
        updates = torch.stack([expert(chart_input) for expert in self.experts], dim=1)
        gated_update = (updates * gates.unsqueeze(-1)).sum(dim=1)
        gamma, beta = gated_update.chunk(2, dim=1)
        scale = float(self.scale)
        return backbone_feature * (1.0 + scale * torch.tanh(gamma)) + scale * beta

    def _stats_and_gates(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        blur3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        blur9 = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        blur21 = F.avg_pool2d(gray, kernel_size=21, stride=1, padding=10)
        micro = (gray - blur3).abs()
        meso = (blur3 - blur9).abs()
        macro = (blur9 - blur21).abs()
        local_contrast = F.avg_pool2d((gray - blur9).abs(), kernel_size=9, stride=1, padding=4)

        near_white = torch.sigmoid((value - 0.86) * 16.0) * torch.sigmoid((0.24 - saturation) * 14.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_smooth = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * torch.sigmoid((0.050 - grad) * 35.0)
        )
        wet_proxy = torch.clamp(specular + 0.5 * dark_smooth, 0.0, 1.0)
        rough_patch = torch.sigmoid((grad + macro - 0.090) * 22.0)
        granular_proxy = torch.sigmoid((micro + meso + grad - 0.115) * 18.0)
        smooth_field = (
            torch.sigmoid((0.060 - grad) * self.gate_temperature)
            * torch.sigmoid((0.030 - macro) * self.gate_temperature)
            * torch.sigmoid((0.040 - local_contrast) * self.gate_temperature)
        )
        broken_film = wet_proxy * rough_patch
        smooth_score = (smooth_field * (1.0 - rough_patch) * (1.0 - 0.5 * granular_proxy)).mean(dim=(2, 3))
        film_score = (wet_proxy * (0.35 + 0.65 * rough_patch)).mean(dim=(2, 3))
        granular_score = (granular_proxy * (1.0 - 0.35 * wet_proxy)).mean(dim=(2, 3))

        smooth_gate = smooth_score * (1.0 - film_score.clamp(0.0, 1.0)) * (1.0 - 0.5 * granular_score.clamp(0.0, 1.0))
        film_gate = film_score * (1.0 - 0.45 * smooth_score.clamp(0.0, 1.0))
        granular_gate = granular_score * (1.0 - 0.35 * smooth_score.clamp(0.0, 1.0))
        gates = torch.cat([smooth_gate, film_gate, granular_gate], dim=1).clamp(0.0, 1.0)
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1.0)

        gx_abs = gx.abs()
        gy_abs = gy.abs()
        anisotropy = (gx_abs.mean(dim=(2, 3)) - gy_abs.mean(dim=(2, 3))).abs() / (
            gx_abs.mean(dim=(2, 3)) + gy_abs.mean(dim=(2, 3)) + 1e-4
        )
        stats = torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                micro.mean(dim=(2, 3)),
                meso.mean(dim=(2, 3)),
                macro.mean(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                near_white.mean(dim=(2, 3)),
                specular.mean(dim=(2, 3)),
                dark_smooth.mean(dim=(2, 3)),
                wet_proxy.mean(dim=(2, 3)),
                rough_patch.mean(dim=(2, 3)),
                granular_proxy.mean(dim=(2, 3)),
                broken_film.mean(dim=(2, 3)),
                smooth_field.mean(dim=(2, 3)),
                smooth_score,
                film_score,
                granular_score,
                anisotropy,
                _soft_connectedness(wet_proxy),
                _soft_connectedness(granular_proxy),
            ],
            dim=1,
        )
        return stats, gates


class MechanismConditionedArtifactGate(nn.Module):
    """Mechanism-split reliability gate for RSCD low-level physical texture.

    A single artifact gate helped asphalt/smooth samples but harmed concrete
    wet/water boundaries. RSCD's compositional labels suggest why: lane-marking
    shortcuts, asphalt smoothness, concrete water film, and granular texture are
    different visual-physical mechanisms. This gate therefore keeps separate
    reliability experts and lets the current global feature softly route among
    them before the low-level PhysicsTexture/LocalPhysicsField evidence is used
    by texture-FiLM.
    """

    mechanism_names = (
        "dry_asphalt_texture",
        "wet_water_asphalt_film",
        "dry_concrete_texture",
        "wet_water_concrete_film",
        "granular_mud_gravel",
        "winter_snow_ice",
    )

    def __init__(
        self,
        *,
        low_level_dim: int,
        embedding_dim: int,
        scale: float = 0.12,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.router_temperature = max(float(router_temperature), 1e-4)
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
        stat_dim = 16
        self.router = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), len(self.mechanism_names)),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(stat_dim),
                    nn.Linear(stat_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, int(low_level_dim)),
                )
                for _ in self.mechanism_names
            ]
        )
        for expert in self.experts:
            last = expert[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        image: torch.Tensor,
        backbone_feature: torch.Tensor,
        low_level_feature: torch.Tensor,
    ) -> torch.Tensor:
        stats = self._stats(image).to(device=low_level_feature.device, dtype=low_level_feature.dtype)
        router_logits = self.router(backbone_feature.to(dtype=low_level_feature.dtype))
        weights = F.softmax(router_logits / self.router_temperature, dim=1)
        expert_delta = torch.stack([expert(stats) for expert in self.experts], dim=1)
        delta = (expert_delta * weights.unsqueeze(-1)).sum(dim=1)
        return low_level_feature * (1.0 + self.scale * torch.tanh(delta))

    def _stats(self, image: torch.Tensor) -> torch.Tensor:
        rgb = (image * self.std + self.mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        maxc = rgb.max(dim=1, keepdim=True).values
        minc = rgb.min(dim=1, keepdim=True).values
        value = maxc
        saturation = (maxc - minc) / maxc.clamp_min(1e-4)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)
        near_white = torch.sigmoid((value - 0.86) * 16.0) * torch.sigmoid((0.24 - saturation) * 14.0)
        specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
        dark_smooth = (
            torch.sigmoid((0.42 - value) * 10.0)
            * torch.sigmoid((0.30 - saturation) * 12.0)
            * torch.sigmoid((0.050 - grad) * 35.0)
        )
        low_texture = torch.sigmoid((0.045 - grad) * 35.0)
        strong_edge = torch.sigmoid((grad - 0.12) * 20.0)
        low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
        wet_proxy = torch.clamp(specular + 0.5 * dark_smooth, 0.0, 1.0)
        white_edge = near_white * strong_edge
        gx_abs = gx.abs()
        gy_abs = gy.abs()
        anisotropy = (gx_abs.mean(dim=(2, 3)) - gy_abs.mean(dim=(2, 3))).abs() / (
            gx_abs.mean(dim=(2, 3)) + gy_abs.mean(dim=(2, 3)) + 1e-4
        )
        return torch.cat(
            [
                value.mean(dim=(2, 3)),
                value.std(dim=(2, 3)),
                saturation.mean(dim=(2, 3)),
                saturation.std(dim=(2, 3)),
                grad.mean(dim=(2, 3)),
                grad.std(dim=(2, 3)),
                local_contrast.mean(dim=(2, 3)),
                near_white.mean(dim=(2, 3)),
                white_edge.mean(dim=(2, 3)),
                specular.mean(dim=(2, 3)),
                dark_smooth.mean(dim=(2, 3)),
                wet_proxy.mean(dim=(2, 3)),
                low_texture.mean(dim=(2, 3)),
                low_contrast.mean(dim=(2, 3)),
                anisotropy,
                _soft_connectedness(wet_proxy),
            ],
            dim=1,
        )


class MechanismOrthogonalCouplingAuxiliary(nn.Module):
    """Training-only mechanism/factor subspace decoupling head.

    RSCD classes are compositional, but the visual coupling mechanism differs
    across dry paved texture, wet film, water obstruction, granular material,
    and winter surfaces. This auxiliary head asks the fused feature to expose
    separate factor subspaces and a separate coupling-mechanism subspace. The
    returned tensors are used only by training losses; inference logits are not
    modified.
    """

    def __init__(self, in_dim: int, *, proj_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        proj_dim = max(int(proj_dim), 8)
        hidden_dim = max(int(hidden_dim), proj_dim)

        def make_projector() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, proj_dim),
            )

        self.factor_proj = nn.ModuleDict(
            {
                "friction": make_projector(),
                "material": make_projector(),
                "unevenness": make_projector(),
                "coupling": make_projector(),
            }
        )
        self.factor_heads = nn.ModuleDict(
            {
                "friction": nn.Linear(proj_dim, len(FACTOR_LABELS["friction"])),
                "material": nn.Linear(proj_dim, len(FACTOR_LABELS["material"])),
                "unevenness": nn.Linear(proj_dim, len(FACTOR_LABELS["unevenness"])),
            }
        )
        self.mechanism_head = nn.Linear(proj_dim, 5)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        embeddings = {name: proj(feature) for name, proj in self.factor_proj.items()}
        factor_logits = {
            name: self.factor_heads[name](embeddings[name])
            for name in ("friction", "material", "unevenness")
        }
        return {
            "embeddings": embeddings,
            "factor_logits": factor_logits,
            "mechanism_logits": self.mechanism_head(embeddings["coupling"]),
        }


class RSCDSurfaceDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        *,
        class_to_idx: dict[str, int],
        transform,
        max_samples: int | None = None,
        max_samples_per_class: int | None = None,
        seed: int = 79,
        mechanism_scope: str = "all",
    ) -> None:
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        df = pd.read_csv(manifest, dtype=str, low_memory=False)
        if "class_label" not in df.columns or "image_path" not in df.columns:
            raise ValueError(f"{manifest} must contain image_path and class_label columns.")
        df["class_label_canonical"] = df["class_label"].map(canonical_class_label)
        df = df[df["class_label_canonical"].astype(str).isin(class_to_idx)].copy()
        mechanism_scope = str(mechanism_scope)
        if mechanism_scope not in MECHANISM_TRAIN_SCOPES:
            raise ValueError(f"unknown mechanism training scope: {mechanism_scope}")
        if mechanism_scope != "all":
            df = df[
                df["class_label_canonical"].map(
                    lambda label: include_class_for_mechanism_scope(str(label), mechanism_scope)
                )
            ].copy()
            if df.empty:
                raise ValueError(f"No RSCD rows remain after applying mechanism scope: {mechanism_scope}")
        if max_samples_per_class:
            parts = []
            for _, group in df.groupby("class_label_canonical", sort=True):
                if len(group) > int(max_samples_per_class):
                    group = group.sample(n=int(max_samples_per_class), random_state=seed)
                parts.append(group)
            df = pd.concat(parts, ignore_index=True)
        if max_samples:
            df = df.sample(n=min(int(max_samples), len(df)), random_state=seed).reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.mechanism_scope = mechanism_scope
        self._warned_bad_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start_idx = int(idx)
        last_error: Exception | None = None
        for offset in range(min(50, len(self.df))):
            row_idx = (start_idx + offset) % len(self.df)
            row = self.df.iloc[row_idx]
            path = Path(str(row["image_path"]))
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.load()
                    image = self.transform(image)
                label_text = str(row["class_label_canonical"])
                factors = parse_rscd_factors(label_text)
                return {
                    "image": image,
                    "label": torch.tensor(self.class_to_idx[label_text], dtype=torch.long),
                    "friction_factor": torch.tensor(factors["friction"], dtype=torch.long),
                    "material_factor": torch.tensor(factors["material"], dtype=torch.long),
                    "unevenness_factor": torch.tensor(factors["unevenness"], dtype=torch.long),
                    "class_label": label_text,
                    "image_path": str(path),
                }
            except (OSError, SyntaxError, ValueError) as exc:
                last_error = exc
                path_text = str(path)
                if path_text not in self._warned_bad_paths:
                    self._warned_bad_paths.add(path_text)
                    print(f"WARNING: skipped unreadable image: {path_text} ({type(exc).__name__}: {exc})", flush=True)
                continue
        raise RuntimeError(f"Could not load a valid image after retries near index {start_idx}: {last_error}")


class RSCDHardPairSampler(Sampler[int]):
    """Sample audited reciprocal factor-boundary pairs into the same batches."""

    def __init__(
        self,
        ds: RSCDSurfaceDataset,
        *,
        num_samples: int,
        seed: int,
        pair_fraction: float,
    ) -> None:
        self.num_samples = max(int(num_samples), 0)
        self.seed = int(seed)
        self.pair_fraction = min(1.0, max(0.0, float(pair_fraction)))
        self.class_to_indices: dict[str, np.ndarray] = {}
        for label, group in ds.df.groupby("class_label_canonical", sort=True):
            self.class_to_indices[str(label)] = group.index.to_numpy(dtype=np.int64)
        pairs = rscd_hard_boundary_pairs()
        self.hard_pairs = [
            (a, b)
            for a, b in pairs
            if a in self.class_to_indices and b in self.class_to_indices
            and len(self.class_to_indices[a]) > 0
            and len(self.class_to_indices[b]) > 0
        ]
        if not self.hard_pairs:
            raise ValueError("No valid RSCD hard pairs were found for hard-pair sampling.")
        self.labels = sorted(self.class_to_indices)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        emitted = 0
        while emitted < self.num_samples:
            if emitted + 1 < self.num_samples and rng.random() < self.pair_fraction:
                left, right = self.hard_pairs[int(rng.integers(0, len(self.hard_pairs)))]
                for label in (left, right):
                    pool = self.class_to_indices[label]
                    yield int(pool[int(rng.integers(0, len(pool)))])
                    emitted += 1
                    if emitted >= self.num_samples:
                        break
            else:
                label = self.labels[int(rng.integers(0, len(self.labels)))]
                pool = self.class_to_indices[label]
                yield int(pool[int(rng.integers(0, len(pool)))])
                emitted += 1


class RSCDControlledFactorTournamentSampler(Sampler[int]):
    """Sample 2x2 controlled-factor rectangles into the same mini-batches.

    Each rectangle holds two contexts and two values on one RSCD factor axis.
    For example, on the unevenness axis it samples:
    (dry, asphalt, smooth), (dry, asphalt, slight),
    (wet, concrete, smooth), (wet, concrete, slight).
    This supplies both same-factor positives and controlled single-factor
    opponents for tournament losses.
    """

    def __init__(
        self,
        ds: RSCDSurfaceDataset,
        *,
        num_samples: int,
        seed: int,
        rectangle_fraction: float,
    ) -> None:
        self.num_samples = max(int(num_samples), 0)
        self.seed = int(seed)
        self.rectangle_fraction = min(1.0, max(0.0, float(rectangle_fraction)))
        self.class_to_indices: dict[str, np.ndarray] = {}
        for label, group in ds.df.groupby("class_label_canonical", sort=True):
            self.class_to_indices[str(label)] = group.index.to_numpy(dtype=np.int64)
        self.labels = sorted(self.class_to_indices)
        self.rectangles = self._build_rectangles()
        if not self.rectangles:
            raise ValueError("No valid RSCD controlled-factor rectangles were found.")

    def _has(self, label: str) -> bool:
        return label in self.class_to_indices and len(self.class_to_indices[label]) > 0

    def _label(self, friction: str, material: str, unevenness: str) -> str:
        return f"{friction}_{material}_{unevenness}"

    def _append_if_valid(self, rectangles: list[tuple[str, str, str, str]], labels: tuple[str, str, str, str]) -> None:
        if all(self._has(label) for label in labels):
            rectangles.append(labels)

    def _build_rectangles(self) -> list[tuple[str, str, str, str]]:
        rectangles: list[tuple[str, str, str, str]] = []
        frictions = ("dry", "wet", "water")
        materials = ("asphalt", "concrete")
        unevenness = ("smooth", "slight", "severe")

        # Friction axis: two (material, unevenness) contexts x two friction values.
        friction_contexts = [(m, u) for m in materials for u in unevenness]
        for c1_idx, (m1, u1) in enumerate(friction_contexts):
            for m2, u2 in friction_contexts[c1_idx + 1 :]:
                for f_idx, f1 in enumerate(frictions):
                    for f2 in frictions[f_idx + 1 :]:
                        self._append_if_valid(
                            rectangles,
                            (
                                self._label(f1, m1, u1),
                                self._label(f2, m1, u1),
                                self._label(f1, m2, u2),
                                self._label(f2, m2, u2),
                            ),
                        )

        # Material axis: two (friction, unevenness) contexts x asphalt/concrete.
        material_contexts = [(f, u) for f in frictions for u in unevenness]
        for c1_idx, (f1, u1) in enumerate(material_contexts):
            for f2, u2 in material_contexts[c1_idx + 1 :]:
                self._append_if_valid(
                    rectangles,
                    (
                        self._label(f1, "asphalt", u1),
                        self._label(f1, "concrete", u1),
                        self._label(f2, "asphalt", u2),
                        self._label(f2, "concrete", u2),
                    ),
                )

        # Unevenness axis: two (friction, material) contexts x two roughness values.
        uneven_contexts = [(f, m) for f in frictions for m in materials]
        for c1_idx, (f1, m1) in enumerate(uneven_contexts):
            for f2, m2 in uneven_contexts[c1_idx + 1 :]:
                for u_idx, u1 in enumerate(unevenness):
                    for u2 in unevenness[u_idx + 1 :]:
                        self._append_if_valid(
                            rectangles,
                            (
                                self._label(f1, m1, u1),
                                self._label(f1, m1, u2),
                                self._label(f2, m2, u1),
                                self._label(f2, m2, u2),
                            ),
                        )
        return rectangles

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        emitted = 0
        while emitted < self.num_samples:
            if emitted + 3 < self.num_samples and rng.random() < self.rectangle_fraction:
                rectangle = self.rectangles[int(rng.integers(0, len(self.rectangles)))]
                order = rng.permutation(4)
                for pos in order:
                    label = rectangle[int(pos)]
                    pool = self.class_to_indices[label]
                    yield int(pool[int(rng.integers(0, len(pool)))])
                    emitted += 1
                    if emitted >= self.num_samples:
                        break
            else:
                label = self.labels[int(rng.integers(0, len(self.labels)))]
                pool = self.class_to_indices[label]
                yield int(pool[int(rng.integers(0, len(pool)))])
                emitted += 1


class SurfaceClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: str,
        embedding_dim: int,
        num_classes: int,
        pretrained: bool,
        dropout: float,
        use_physics_branch: bool = False,
        physics_dim: int = 96,
        physics_quality_cues: bool = False,
        physics_quality_region_cues: bool = True,
        use_directional_texture_branch: bool = False,
        directional_texture_dim: int = 64,
        use_wavelet_texture_branch: bool = False,
        wavelet_texture_dim: int = 64,
        use_retinex_texture_branch: bool = False,
        retinex_texture_dim: int = 48,
        retinex_region_cues: bool = True,
        use_physics_attention_branch: bool = False,
        physics_attention_dim: int = 64,
        use_semantic_physics_attention_branch: bool = False,
        semantic_physics_attention_dim: int = 64,
        use_visibility_observed_roughness_branch: bool = False,
        visibility_observed_roughness_dim: int = 64,
        use_visibility_observed_roughness_adapter: bool = False,
        visibility_observed_roughness_scale: float = 0.04,
        use_factor_conditioned_physics_token_branch: bool = False,
        factor_conditioned_physics_token_dim: int = 48,
        factor_conditioned_physics_token_inner_dim: int = 16,
        use_factor_coupled_physics_token_branch: bool = False,
        factor_coupled_physics_token_dim: int = 64,
        factor_coupled_physics_token_inner_dim: int = 16,
        use_local_physics_field_branch: bool = False,
        local_physics_field_dim: int = 64,
        local_physics_field_scale: float = 0.15,
        use_relation_conditioned_physics_expert_branch: bool = False,
        relation_conditioned_physics_expert_dim: int = 72,
        relation_conditioned_physics_expert_inner_dim: int = 24,
        use_relation_conditioned_physics_expert_adapter: bool = False,
        relation_conditioned_physics_expert_scale: float = 0.06,
        use_topological_texture_branch: bool = False,
        topological_texture_dim: int = 48,
        use_anti_human_texture_branch: bool = False,
        anti_human_texture_dim: int = 64,
        use_texture_gate: bool = False,
        use_texture_residual_adapter: bool = False,
        texture_residual_scale: float = 0.25,
        use_texture_film: bool = False,
        texture_film_scale: float = 0.20,
        use_material_conditioned_texture_gate: bool = False,
        material_conditioned_gate_scale: float = 0.25,
        use_artifact_aware_texture_gate: bool = False,
        artifact_aware_gate_scale: float = 0.20,
        use_smooth_evidence_texture_gate: bool = False,
        smooth_evidence_texture_gate_scale: float = 0.10,
        smooth_evidence_texture_gate_temperature: float = 16.0,
        smooth_evidence_texture_gate_rough_suppression: float = 0.65,
        use_tri_chart_evidence_film: bool = False,
        tri_chart_evidence_film_hidden_dim: int = 256,
        tri_chart_evidence_film_scale: float = 0.035,
        tri_chart_evidence_film_gate_temperature: float = 18.0,
        use_mechanism_conditioned_artifact_gate: bool = False,
        mechanism_conditioned_artifact_gate_scale: float = 0.12,
        mechanism_conditioned_artifact_gate_temperature: float = 1.0,
        use_factor_logit_adjustment: bool = False,
        factor_logit_adjustment_scale: float = 0.30,
        use_factorized_low_rank_head: bool = False,
        factorized_rank: int = 64,
        factorized_scale: float = 0.25,
        factorized_normalize: bool = True,
        factorized_zero_init: bool = False,
        factorized_factors: tuple[str, ...] | None = None,
        factorized_class_embedding: bool = True,
        use_safe_factorized_low_rank_head: bool = False,
        safe_factorized_rank: int = 64,
        safe_factorized_scale: float = 0.25,
        safe_factorized_gate_threshold: float = 0.55,
        safe_factorized_gate_temperature: float = 8.0,
        safe_factorized_protected_negative_limit: float = 0.0,
        use_factor_interaction_low_rank_head: bool = False,
        factor_interaction_rank: int = 64,
        factor_interaction_scale: float = 0.20,
        factor_interaction_gate_threshold: float = 0.55,
        factor_interaction_gate_temperature: float = 8.0,
        factor_interaction_protected_negative_limit: float = 0.0,
        use_conditional_coupling_decomposition_field: bool = False,
        conditional_coupling_rank: int = 64,
        conditional_coupling_scale: float = 0.08,
        conditional_coupling_gate_threshold: float = 0.35,
        conditional_coupling_gate_temperature: float = 8.0,
        conditional_coupling_protected_negative_limit: float = 0.0,
        conditional_coupling_relation_gate_hidden_dim: int = 64,
        conditional_coupling_relation_gate_temperature: float = 1.0,
        use_mobius_sheaf_factor_head: bool = False,
        mobius_sheaf_rank: int = 32,
        mobius_sheaf_scale: float = 0.12,
        mobius_sheaf_mode: str = "residual",
        mobius_sheaf_blend: float = 0.50,
        mobius_sheaf_gate_hidden_dim: int = 96,
        mobius_sheaf_gate_temperature: float = 1.0,
        mobius_sheaf_normalize: bool = True,
        mobius_sheaf_zero_init: bool = True,
        mobius_sheaf_use_triple: bool = True,
        use_mechanism_conditional_sheaf_head: bool = False,
        mechanism_sheaf_rank: int = 24,
        mechanism_sheaf_scale: float = 0.06,
        mechanism_sheaf_edge_scale: float = 0.04,
        mechanism_sheaf_router_hidden_dim: int = 96,
        mechanism_sheaf_edge_dim: int = 12,
        mechanism_sheaf_edge_hidden_dim: int = 64,
        mechanism_sheaf_class_scope: str = "all",
        mechanism_sheaf_use_edge_flow: bool = True,
        mechanism_sheaf_protected_negative_limit: float = 0.0,
        mechanism_sheaf_sparse_router_topk: int = 0,
        mechanism_sheaf_router_temperature: float = 1.0,
        mechanism_sheaf_physics_prior_weight: float = 0.0,
        use_local_global_factor_attention: bool = False,
        local_global_factor_rank: int = 48,
        local_global_factor_scale: float = 0.08,
        local_global_factor_gate_threshold: float = 0.35,
        local_global_factor_gate_temperature: float = 10.0,
        local_global_factor_neighbor_gate_floor: float = 0.15,
        local_global_factor_protected_negative_limit: float = 0.0,
        use_label_graph_residual: bool = False,
        label_graph_rank: int = 32,
        label_graph_scale: float = 0.04,
        label_graph_gate_threshold: float = 0.45,
        label_graph_gate_temperature: float = 10.0,
        label_graph_neighbor_gate_floor: float = 0.10,
        use_conditional_evidence_masked_coupling_field: bool = False,
        evidence_masked_coupling_feature_map_dim: int = 768,
        evidence_masked_coupling_token_dim: int = 96,
        evidence_masked_coupling_rank: int = 32,
        evidence_masked_coupling_scale: float = 0.04,
        evidence_masked_coupling_gate_threshold: float = 0.35,
        evidence_masked_coupling_gate_temperature: float = 10.0,
        evidence_masked_coupling_neighbor_gate_floor: float = 0.05,
        evidence_masked_coupling_protected_negative_limit: float = 0.0,
        use_full_order_coupling_tensor_field: bool = False,
        full_order_coupling_feature_map_dim: int = 768,
        full_order_coupling_token_dim: int = 96,
        full_order_coupling_hidden_dim: int = 96,
        full_order_coupling_scale: float = 0.05,
        full_order_coupling_gate_threshold: float = 0.35,
        full_order_coupling_gate_temperature: float = 10.0,
        full_order_coupling_core_gate_floor: float = 0.05,
        full_order_coupling_protected_negative_limit: float = 0.0,
        use_mechanism_charted_full_order_coupling_tensor_field: bool = False,
        mechanism_charted_full_order_feature_map_dim: int = 768,
        mechanism_charted_full_order_token_dim: int = 96,
        mechanism_charted_full_order_hidden_dim: int = 96,
        mechanism_charted_full_order_scale: float = 0.04,
        mechanism_charted_full_order_gate_threshold: float = 0.35,
        mechanism_charted_full_order_gate_temperature: float = 10.0,
        mechanism_charted_full_order_core_gate_floor: float = 0.05,
        mechanism_charted_full_order_protected_negative_limit: float = 0.0,
        mechanism_charted_full_order_router_hidden_dim: int = 96,
        mechanism_charted_full_order_router_temperature: float = 1.0,
        mechanism_charted_full_order_sparse_router_topk: int = 2,
        mechanism_charted_full_order_physics_prior_weight: float = 1.0,
        use_core_factor_coupled_residual: bool = False,
        core_factor_rank: int = 32,
        core_factor_scale: float = 0.08,
        core_factor_neighbor_gate_floor: float = 0.05,
        core_factor_uncertainty_threshold: float = 0.40,
        core_factor_uncertainty_temperature: float = 10.0,
        core_factor_protected_negative_limit: float = 0.0,
        use_water_evidence_logit_gate: bool = False,
        water_evidence_gate_scale: float = 0.20,
        water_evidence_gate_zero_init: bool = True,
        use_dry_concrete_roughness_vor_residual: bool = False,
        dry_concrete_roughness_hidden_dim: int = 48,
        dry_concrete_roughness_scale: float = 0.05,
        dry_concrete_roughness_gate_threshold: float = 0.12,
        dry_concrete_roughness_gate_temperature: float = 14.0,
        use_dry_paved_roughness_vor_residual: bool = False,
        dry_paved_roughness_hidden_dim: int = 48,
        dry_paved_roughness_material_dim: int = 6,
        dry_paved_roughness_scale: float = 0.05,
        dry_paved_roughness_gate_threshold: float = 0.12,
        dry_paved_roughness_gate_temperature: float = 14.0,
        dry_paved_roughness_head_mode: str = "shared",
        dry_paved_roughness_material_gate_threshold: float = 0.0,
        dry_paved_roughness_material_gate_temperature: float = 16.0,
        use_concrete_roughness_vor_residual: bool = False,
        concrete_roughness_hidden_dim: int = 48,
        concrete_roughness_chart_dim: int = 6,
        concrete_roughness_scale: float = 0.05,
        concrete_roughness_gate_threshold: float = 0.12,
        concrete_roughness_gate_temperature: float = 14.0,
        use_wet_water_film_vor_residual: bool = False,
        wet_water_film_hidden_dim: int = 48,
        wet_water_film_pair_dim: int = 8,
        wet_water_film_scale: float = 0.05,
        wet_water_film_material_scope: str = "all",
        wet_water_film_gate_threshold: float = 0.12,
        wet_water_film_gate_temperature: float = 14.0,
        use_smooth_film_concrete_expert: bool = False,
        smooth_film_concrete_hidden_dim: int = 48,
        smooth_film_concrete_scale: float = 0.05,
        smooth_film_concrete_gate_threshold: float = 0.05,
        smooth_film_concrete_gate_temperature: float = 14.0,
        use_obstruction_concrete_roughness_vor_residual: bool = False,
        obstruction_concrete_roughness_hidden_dim: int = 48,
        obstruction_concrete_roughness_scale: float = 0.05,
        obstruction_concrete_roughness_gate_threshold: float = 0.12,
        obstruction_concrete_roughness_gate_temperature: float = 14.0,
        obstruction_concrete_roughness_share_gate_threshold: float = 0.0,
        obstruction_concrete_roughness_share_gate_temperature: float = 16.0,
        use_coupled_optical_roughness_residual: bool = False,
        coupled_residual_hidden_dim: int = 96,
        coupled_residual_scale: float = 0.12,
        coupled_residual_gate_threshold: float = 0.35,
        coupled_residual_gate_temperature: float = 8.0,
        coupled_residual_protected_negative_limit: float = 0.0,
        use_roughness_neighbor_residual: bool = False,
        roughness_neighbor_hidden_dim: int = 96,
        roughness_neighbor_scale: float = 0.10,
        roughness_neighbor_gate_threshold: float = 0.42,
        roughness_neighbor_gate_temperature: float = 10.0,
        roughness_neighbor_protected_negative_limit: float = 0.0,
        roughness_neighbor_gate_floor: float = 0.15,
        use_spectral_roughness_residual: bool = False,
        spectral_roughness_hidden_dim: int = 96,
        spectral_roughness_scale: float = 0.08,
        spectral_roughness_gate_threshold: float = 0.35,
        spectral_roughness_gate_temperature: float = 12.0,
        spectral_roughness_protected_negative_limit: float = 0.0,
        spectral_roughness_neighbor_gate_floor: float = 0.02,
        use_relation_signed_graph_expert: bool = False,
        relation_signed_hidden_dim: int = 96,
        relation_signed_scale: float = 0.06,
        relation_signed_gate_threshold: float = 0.35,
        relation_signed_gate_temperature: float = 12.0,
        relation_signed_protected_negative_limit: float = 0.0,
        relation_signed_neighbor_gate_floor: float = 0.0,
        use_heterophilic_logit_boundary_expert: bool = False,
        heterophilic_boundary_scale: float = 0.35,
        heterophilic_boundary_gate_threshold: float = 0.0,
        heterophilic_boundary_gate_temperature: float = 8.0,
        heterophilic_boundary_protected_negative_limit: float = 0.0,
        use_heterophilic_feature_boundary_expert: bool = False,
        heterophilic_feature_boundary_hidden_dim: int = 96,
        heterophilic_feature_boundary_pair_dim: int = 16,
        heterophilic_feature_boundary_scale: float = 0.08,
        heterophilic_feature_boundary_gate_threshold: float = 0.10,
        heterophilic_feature_boundary_gate_temperature: float = 10.0,
        heterophilic_feature_boundary_protected_negative_limit: float = 0.0,
        use_heterophilic_physics_boundary_expert: bool = False,
        heterophilic_physics_boundary_hidden_dim: int = 64,
        heterophilic_physics_boundary_pair_dim: int = 12,
        heterophilic_physics_boundary_scale: float = 0.08,
        heterophilic_physics_boundary_gate_threshold: float = 0.10,
        heterophilic_physics_boundary_gate_temperature: float = 10.0,
        heterophilic_physics_boundary_protected_negative_limit: float = 0.0,
        use_protected_heterophilic_factor_boundary_field: bool = False,
        protected_factor_boundary_hidden_dim: int = 64,
        protected_factor_boundary_pair_dim: int = 12,
        protected_factor_boundary_relation_dim: int = 8,
        protected_factor_boundary_scale: float = 0.08,
        protected_factor_boundary_gate_threshold: float = 0.10,
        protected_factor_boundary_gate_temperature: float = 10.0,
        protected_factor_boundary_protected_negative_limit: float = 0.0,
        use_relation_specific_hard_edge_refiner: bool = False,
        relation_specific_refiner_hidden_dim: int = 64,
        relation_specific_refiner_pair_dim: int = 12,
        relation_specific_refiner_scale: float = 0.08,
        relation_specific_refiner_gate_threshold: float = 0.10,
        relation_specific_refiner_gate_temperature: float = 10.0,
        relation_specific_refiner_protected_negative_limit: float = 0.0,
        use_selective_mechanism_tensor_boundary_field: bool = False,
        selective_mechanism_tensor_boundary_hidden_dim: int = 64,
        selective_mechanism_tensor_boundary_pair_dim: int = 12,
        selective_mechanism_tensor_boundary_relation_dim: int = 6,
        selective_mechanism_tensor_boundary_mechanism_dim: int = 8,
        selective_mechanism_tensor_boundary_scale: float = 0.08,
        selective_mechanism_tensor_boundary_gate_threshold: float = 0.10,
        selective_mechanism_tensor_boundary_gate_temperature: float = 10.0,
        selective_mechanism_tensor_boundary_mechanism_gate_threshold: float = 0.08,
        selective_mechanism_tensor_boundary_mechanism_gate_temperature: float = 12.0,
        selective_mechanism_tensor_boundary_enabled_mechanisms: str = "all",
        selective_mechanism_tensor_boundary_protected_negative_limit: float = 0.0,
        use_conditional_factor_projection: bool = False,
        conditional_factor_projection_scale: float = 0.04,
        conditional_factor_projection_gate_threshold: float = 0.35,
        conditional_factor_projection_gate_temperature: float = 10.0,
        conditional_factor_projection_focus: str = "core",
        conditional_factor_projection_friction_weight: float = 1.0,
        conditional_factor_projection_material_weight: float = 0.6,
        conditional_factor_projection_unevenness_weight: float = 1.2,
        conditional_factor_projection_protected_negative_limit: float = 0.0,
        use_heterogeneous_label_router: bool = False,
        heterogeneous_router_hidden_dim: int = 128,
        heterogeneous_router_scale: float = 0.08,
        use_hard_pair_aux: bool = False,
        hard_pair_aux_num_pairs: int = 0,
        hard_pair_aux_hidden_dim: int = 128,
        use_mechanism_orthogonal_coupling_aux: bool = False,
        mechanism_orthogonal_dim: int = 64,
        mechanism_orthogonal_hidden_dim: int = 128,
        class_to_idx: dict[str, int] | None = None,
        use_factor_aux: bool = False,
        use_local_physics_factor_aux: bool = False,
        use_backbone_aux: bool = False,
        use_physics_aux: bool = False,
        use_physics_evidence_aux: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone, embedding_dim, pretrained=pretrained)
        self.use_factor_aux = bool(use_factor_aux)
        self.use_local_physics_factor_aux = bool(use_local_physics_factor_aux and use_local_physics_field_branch)
        self.use_backbone_aux = bool(use_backbone_aux)
        self.use_physics_aux = bool(use_physics_aux and use_physics_branch)
        self.use_physics_evidence_aux = bool(use_physics_evidence_aux)
        self.physics_branch = (
            PhysicsTextureBranch(
                physics_dim,
                quality_cues=physics_quality_cues,
                quality_region_cues=physics_quality_region_cues,
            )
            if use_physics_branch
            else None
        )
        self.directional_texture_branch = (
            DirectionalTextureBranch(directional_texture_dim)
            if use_directional_texture_branch
            else None
        )
        self.wavelet_texture_branch = (
            WaveletTextureBranch(wavelet_texture_dim)
            if use_wavelet_texture_branch
            else None
        )
        self.retinex_texture_branch = (
            RetinexTextureBranch(retinex_texture_dim, region_cues=retinex_region_cues)
            if use_retinex_texture_branch
            else None
        )
        self.physics_attention_branch = (
            PhysicsAttentionBranch(physics_attention_dim)
            if use_physics_attention_branch
            else None
        )
        self.semantic_physics_attention_branch = (
            SemanticPhysicsAttentionBranch(semantic_physics_attention_dim)
            if use_semantic_physics_attention_branch
            else None
        )
        self.visibility_observed_roughness_branch = (
            VisibilityObservedRoughnessBranch(visibility_observed_roughness_dim)
            if use_visibility_observed_roughness_branch or use_visibility_observed_roughness_adapter
            else None
        )
        self.use_visibility_observed_roughness_adapter = bool(
            use_visibility_observed_roughness_adapter
            and self.visibility_observed_roughness_branch is not None
        )
        self.visibility_observed_roughness_scale = float(visibility_observed_roughness_scale)
        self.factor_conditioned_physics_token_branch = (
            FactorConditionedPhysicsTokenBranch(
                factor_conditioned_physics_token_dim,
                token_dim=factor_conditioned_physics_token_inner_dim,
            )
            if use_factor_conditioned_physics_token_branch
            else None
        )
        self.factor_coupled_physics_token_branch = (
            FactorCoupledPhysicsTokenBranch(
                factor_coupled_physics_token_dim,
                token_dim=factor_coupled_physics_token_inner_dim,
            )
            if use_factor_coupled_physics_token_branch
            else None
        )
        self.local_physics_field_branch = (
            LocalPhysicsFieldBranch(local_physics_field_dim)
            if use_local_physics_field_branch
            else None
        )
        self.local_physics_field_scale = float(local_physics_field_scale)
        self.relation_conditioned_physics_expert_branch = (
            RelationConditionedPhysicsExpertBranch(
                relation_conditioned_physics_expert_dim,
                expert_dim=relation_conditioned_physics_expert_inner_dim,
            )
            if use_relation_conditioned_physics_expert_branch
            else None
        )
        self.use_relation_conditioned_physics_expert_adapter = bool(
            use_relation_conditioned_physics_expert_adapter
            and self.relation_conditioned_physics_expert_branch is not None
        )
        self.relation_conditioned_physics_expert_scale = float(relation_conditioned_physics_expert_scale)
        self.topological_texture_branch = (
            TopologicalTextureBranch(topological_texture_dim)
            if use_topological_texture_branch
            else None
        )
        self.anti_human_texture_branch = (
            AntiHumanTextureBranch(anti_human_texture_dim)
            if use_anti_human_texture_branch
            else None
        )
        low_level_dim = (
            (physics_dim if use_physics_branch else 0)
            + (directional_texture_dim if use_directional_texture_branch else 0)
            + (wavelet_texture_dim if use_wavelet_texture_branch else 0)
            + (retinex_texture_dim if use_retinex_texture_branch else 0)
            + (physics_attention_dim if use_physics_attention_branch else 0)
            + (semantic_physics_attention_dim if use_semantic_physics_attention_branch else 0)
            + (
                visibility_observed_roughness_dim
                if use_visibility_observed_roughness_branch
                and not self.use_visibility_observed_roughness_adapter
                else 0
            )
            + (
                factor_conditioned_physics_token_dim
                if use_factor_conditioned_physics_token_branch
                else 0
            )
            + (
                factor_coupled_physics_token_dim
                if use_factor_coupled_physics_token_branch
                else 0
            )
            + (
                relation_conditioned_physics_expert_dim
                if use_relation_conditioned_physics_expert_branch
                and not self.use_relation_conditioned_physics_expert_adapter
                else 0
            )
            + (topological_texture_dim if use_topological_texture_branch else 0)
            + (anti_human_texture_dim if use_anti_human_texture_branch else 0)
        )
        self.texture_gate = (
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, low_level_dim),
                nn.Sigmoid(),
            )
            if use_texture_gate and low_level_dim > 0
            else None
        )
        self.use_texture_residual_adapter = bool(use_texture_residual_adapter and low_level_dim > 0)
        self.texture_residual_scale = float(texture_residual_scale)
        self.use_texture_film = bool(use_texture_film and low_level_dim > 0)
        self.texture_film_scale = float(texture_film_scale)
        self.use_material_conditioned_texture_gate = bool(use_material_conditioned_texture_gate and low_level_dim > 0)
        self.material_conditioned_gate_scale = float(material_conditioned_gate_scale)
        self.use_artifact_aware_texture_gate = bool(use_artifact_aware_texture_gate and low_level_dim > 0)
        self.use_smooth_evidence_texture_gate = bool(use_smooth_evidence_texture_gate and low_level_dim > 0)
        self.use_tri_chart_evidence_film = bool(use_tri_chart_evidence_film and low_level_dim > 0)
        self.use_mechanism_conditioned_artifact_gate = bool(
            use_mechanism_conditioned_artifact_gate and low_level_dim > 0
        )
        self.use_factor_logit_adjustment = bool(use_factor_logit_adjustment)
        self.factor_logit_adjustment_scale = float(factor_logit_adjustment_scale)
        self.use_factorized_low_rank_head = bool(use_factorized_low_rank_head)
        self.use_safe_factorized_low_rank_head = bool(use_safe_factorized_low_rank_head)
        self.use_factor_interaction_low_rank_head = bool(use_factor_interaction_low_rank_head)
        self.use_conditional_coupling_decomposition_field = bool(use_conditional_coupling_decomposition_field)
        self.use_mobius_sheaf_factor_head = bool(use_mobius_sheaf_factor_head)
        self.use_mechanism_conditional_sheaf_head = bool(use_mechanism_conditional_sheaf_head)
        self.mobius_sheaf_mode = str(mobius_sheaf_mode)
        if self.mobius_sheaf_mode not in {"residual", "blend", "replace"}:
            raise ValueError(f"unknown mobius_sheaf_mode: {self.mobius_sheaf_mode}")
        self.mobius_sheaf_blend = min(1.0, max(0.0, float(mobius_sheaf_blend)))
        self.use_local_global_factor_attention = bool(use_local_global_factor_attention)
        self.use_label_graph_residual = bool(use_label_graph_residual)
        self.use_conditional_evidence_masked_coupling_field = bool(use_conditional_evidence_masked_coupling_field)
        self.use_full_order_coupling_tensor_field = bool(use_full_order_coupling_tensor_field)
        self.use_mechanism_charted_full_order_coupling_tensor_field = bool(
            use_mechanism_charted_full_order_coupling_tensor_field
        )
        self.use_core_factor_coupled_residual = bool(use_core_factor_coupled_residual)
        self.use_water_evidence_logit_gate = bool(use_water_evidence_logit_gate)
        self.use_dry_concrete_roughness_vor_residual = bool(use_dry_concrete_roughness_vor_residual)
        self.use_dry_paved_roughness_vor_residual = bool(use_dry_paved_roughness_vor_residual)
        self.use_concrete_roughness_vor_residual = bool(use_concrete_roughness_vor_residual)
        self.use_wet_water_film_vor_residual = bool(use_wet_water_film_vor_residual)
        self.use_smooth_film_concrete_expert = bool(use_smooth_film_concrete_expert)
        self.use_obstruction_concrete_roughness_vor_residual = bool(
            use_obstruction_concrete_roughness_vor_residual
        )
        self.use_coupled_optical_roughness_residual = bool(use_coupled_optical_roughness_residual)
        self.use_roughness_neighbor_residual = bool(use_roughness_neighbor_residual)
        self.use_spectral_roughness_residual = bool(use_spectral_roughness_residual)
        self.use_relation_signed_graph_expert = bool(use_relation_signed_graph_expert)
        self.use_heterophilic_logit_boundary_expert = bool(use_heterophilic_logit_boundary_expert)
        self.use_heterophilic_feature_boundary_expert = bool(use_heterophilic_feature_boundary_expert)
        self.use_heterophilic_physics_boundary_expert = bool(use_heterophilic_physics_boundary_expert)
        self.use_protected_heterophilic_factor_boundary_field = bool(use_protected_heterophilic_factor_boundary_field)
        self.use_relation_specific_hard_edge_refiner = bool(use_relation_specific_hard_edge_refiner)
        self.use_selective_mechanism_tensor_boundary_field = bool(use_selective_mechanism_tensor_boundary_field)
        self.use_conditional_factor_projection = bool(use_conditional_factor_projection)
        self.use_heterogeneous_label_router = bool(use_heterogeneous_label_router)
        self.use_hard_pair_aux = bool(use_hard_pair_aux and int(hard_pair_aux_num_pairs) > 0)
        self.use_mechanism_orthogonal_coupling_aux = bool(use_mechanism_orthogonal_coupling_aux)
        self.local_physics_field_adapter = (
            nn.Sequential(
                nn.LayerNorm(local_physics_field_dim),
                nn.Linear(local_physics_field_dim, embedding_dim),
            )
            if self.local_physics_field_branch is not None
            else None
        )
        self.relation_conditioned_physics_expert_adapter = (
            nn.Sequential(
                nn.LayerNorm(relation_conditioned_physics_expert_dim),
                nn.Linear(relation_conditioned_physics_expert_dim, embedding_dim),
            )
            if self.use_relation_conditioned_physics_expert_adapter
            else None
        )
        self.visibility_observed_roughness_adapter = (
            nn.Sequential(
                nn.LayerNorm(visibility_observed_roughness_dim),
                nn.Linear(visibility_observed_roughness_dim, embedding_dim),
            )
            if self.use_visibility_observed_roughness_adapter
            else None
        )
        if self.local_physics_field_adapter is not None:
            last = self.local_physics_field_adapter[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        if self.relation_conditioned_physics_expert_adapter is not None:
            last = self.relation_conditioned_physics_expert_adapter[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        if self.visibility_observed_roughness_adapter is not None:
            last = self.visibility_observed_roughness_adapter[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        self.texture_residual_adapter = (
            nn.Sequential(
                nn.LayerNorm(low_level_dim),
                nn.Linear(low_level_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            if self.use_texture_residual_adapter
            else None
        )
        self.texture_film = (
            nn.Sequential(
                nn.LayerNorm(low_level_dim),
                nn.Linear(low_level_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, 2 * embedding_dim),
            )
            if self.use_texture_film
            else None
        )
        self.material_conditioned_texture_gate = (
            nn.ModuleDict(
                {
                    "material_logits": nn.Linear(embedding_dim, len(FACTOR_LABELS["material"])),
                    "gate": nn.Linear(len(FACTOR_LABELS["material"]), low_level_dim),
                }
            )
            if self.use_material_conditioned_texture_gate
            else None
        )
        self.artifact_aware_texture_gate = (
            ArtifactAwareTextureGate(low_level_dim=low_level_dim, scale=float(artifact_aware_gate_scale))
            if self.use_artifact_aware_texture_gate
            else None
        )
        self.smooth_evidence_texture_gate = (
            SmoothEvidenceTextureGate(
                low_level_dim=low_level_dim,
                scale=float(smooth_evidence_texture_gate_scale),
                smooth_temperature=float(smooth_evidence_texture_gate_temperature),
                rough_suppression=float(smooth_evidence_texture_gate_rough_suppression),
            )
            if self.use_smooth_evidence_texture_gate
            else None
        )
        self.tri_chart_evidence_film = (
            TriChartEvidenceFiLM(
                low_level_dim=low_level_dim,
                embedding_dim=embedding_dim,
                hidden_dim=int(tri_chart_evidence_film_hidden_dim),
                scale=float(tri_chart_evidence_film_scale),
                gate_temperature=float(tri_chart_evidence_film_gate_temperature),
            )
            if self.use_tri_chart_evidence_film
            else None
        )
        self.mechanism_conditioned_artifact_gate = (
            MechanismConditionedArtifactGate(
                low_level_dim=low_level_dim,
                embedding_dim=embedding_dim,
                scale=float(mechanism_conditioned_artifact_gate_scale),
                router_temperature=float(mechanism_conditioned_artifact_gate_temperature),
            )
            if self.use_mechanism_conditioned_artifact_gate
            else None
        )
        if self.texture_residual_adapter is not None:
            last = self.texture_residual_adapter[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        if self.texture_film is not None:
            last = self.texture_film[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        if self.material_conditioned_texture_gate is not None:
            gate = self.material_conditioned_texture_gate["gate"]
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        head_dim = (
            embedding_dim
            + (0 if (self.use_texture_residual_adapter or self.use_texture_film) else low_level_dim)
        )
        self.norm = nn.LayerNorm(head_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_dim, num_classes)
        self.factorized_low_rank_head = None
        if self.use_factorized_low_rank_head:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_factorized_low_rank_head=True.")
            self.factorized_low_rank_head = FactorizedLowRankHead(
                in_dim=head_dim,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
                rank=int(factorized_rank),
                scale=float(factorized_scale),
                normalize=bool(factorized_normalize),
                zero_init=bool(factorized_zero_init),
                factors=factorized_factors,
                use_class_embedding=bool(factorized_class_embedding),
            )
        self.safe_factorized_low_rank_head = None
        if self.use_safe_factorized_low_rank_head:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_safe_factorized_low_rank_head=True.")
            self.safe_factorized_low_rank_head = SafeAdaptiveFactorizedLowRankHead(
                in_dim=head_dim,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
                rank=int(safe_factorized_rank),
                scale=float(safe_factorized_scale),
                normalize=bool(factorized_normalize),
                zero_init=True,
                factors=factorized_factors,
                use_class_embedding=False,
                gate_threshold=float(safe_factorized_gate_threshold),
                gate_temperature=float(safe_factorized_gate_temperature),
                protected_negative_limit=float(safe_factorized_protected_negative_limit),
            )
        self.factor_interaction_low_rank_head = None
        if self.use_factor_interaction_low_rank_head:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_factor_interaction_low_rank_head=True.")
            self.factor_interaction_low_rank_head = FactorInteractionLowRankHead(
                in_dim=head_dim,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
                rank=int(factor_interaction_rank),
                scale=float(factor_interaction_scale),
                normalize=bool(factorized_normalize),
                zero_init=True,
                gate_threshold=float(factor_interaction_gate_threshold),
                gate_temperature=float(factor_interaction_gate_temperature),
                protected_negative_limit=float(factor_interaction_protected_negative_limit),
            )
        self.conditional_coupling_decomposition_field = None
        if self.use_conditional_coupling_decomposition_field:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_conditional_coupling_decomposition_field=True.")
            self.conditional_coupling_decomposition_field = ConditionalCouplingDecompositionField(
                in_dim=head_dim,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
                rank=int(conditional_coupling_rank),
                scale=float(conditional_coupling_scale),
                normalize=bool(factorized_normalize),
                zero_init=True,
                gate_threshold=float(conditional_coupling_gate_threshold),
                gate_temperature=float(conditional_coupling_gate_temperature),
                protected_negative_limit=float(conditional_coupling_protected_negative_limit),
                relation_gate_hidden_dim=int(conditional_coupling_relation_gate_hidden_dim),
                relation_gate_temperature=float(conditional_coupling_relation_gate_temperature),
            )
        self.mobius_sheaf_factor_head = None
        if self.use_mobius_sheaf_factor_head:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_mobius_sheaf_factor_head=True.")
            self.mobius_sheaf_factor_head = MobiusSheafFactorHead(
                in_dim=head_dim,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
                rank=int(mobius_sheaf_rank),
                scale=float(mobius_sheaf_scale),
                normalize=bool(mobius_sheaf_normalize),
                zero_init=bool(mobius_sheaf_zero_init),
                use_triple=bool(mobius_sheaf_use_triple),
                gate_hidden_dim=int(mobius_sheaf_gate_hidden_dim),
                gate_temperature=float(mobius_sheaf_gate_temperature),
            )
        self.mechanism_conditional_sheaf_head = None
        if self.use_mechanism_conditional_sheaf_head:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_mechanism_conditional_sheaf_head=True.")
            self.mechanism_conditional_sheaf_head = MechanismConditionalSheafHead(
                in_dim=head_dim,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
                rank=int(mechanism_sheaf_rank),
                scale=float(mechanism_sheaf_scale),
                edge_scale=float(mechanism_sheaf_edge_scale),
                router_hidden_dim=int(mechanism_sheaf_router_hidden_dim),
                edge_dim=int(mechanism_sheaf_edge_dim),
                edge_hidden_dim=int(mechanism_sheaf_edge_hidden_dim),
                class_scope=str(mechanism_sheaf_class_scope),
                normalize=bool(mobius_sheaf_normalize),
                zero_init=True,
                use_edge_flow=bool(mechanism_sheaf_use_edge_flow),
                protected_negative_limit=float(mechanism_sheaf_protected_negative_limit),
                sparse_router_topk=int(mechanism_sheaf_sparse_router_topk),
                router_temperature=float(mechanism_sheaf_router_temperature),
                physics_prior_weight=float(mechanism_sheaf_physics_prior_weight),
            )
        self.local_global_factor_attention = None
        if self.use_local_global_factor_attention:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_local_global_factor_attention=True.")
            self.local_global_factor_attention = LocalGlobalFactorAttentionResidual(
                in_dim=head_dim,
                local_dim=local_physics_field_dim if self.local_physics_field_branch is not None else 0,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
                rank=int(local_global_factor_rank),
                scale=float(local_global_factor_scale),
                normalize=bool(factorized_normalize),
                zero_init=True,
                gate_threshold=float(local_global_factor_gate_threshold),
                gate_temperature=float(local_global_factor_gate_temperature),
                neighbor_gate_floor=float(local_global_factor_neighbor_gate_floor),
                protected_negative_limit=float(local_global_factor_protected_negative_limit),
            )
        self.label_graph_residual = None
        if self.use_label_graph_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_label_graph_residual=True.")
            self.label_graph_residual = LabelGraphPrototypeResidual(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                rank=int(label_graph_rank),
                scale=float(label_graph_scale),
                gate_threshold=float(label_graph_gate_threshold),
                gate_temperature=float(label_graph_gate_temperature),
                neighbor_gate_floor=float(label_graph_neighbor_gate_floor),
                zero_init=True,
            )
        self.conditional_evidence_masked_coupling_field = None
        if self.use_conditional_evidence_masked_coupling_field:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_conditional_evidence_masked_coupling_field=True.")
            self.conditional_evidence_masked_coupling_field = ConditionalEvidenceMaskedCouplingField(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                feature_map_dim=int(evidence_masked_coupling_feature_map_dim),
                token_dim=int(evidence_masked_coupling_token_dim),
                rank=int(evidence_masked_coupling_rank),
                scale=float(evidence_masked_coupling_scale),
                normalize=bool(factorized_normalize),
                gate_threshold=float(evidence_masked_coupling_gate_threshold),
                gate_temperature=float(evidence_masked_coupling_gate_temperature),
                neighbor_gate_floor=float(evidence_masked_coupling_neighbor_gate_floor),
                protected_negative_limit=float(evidence_masked_coupling_protected_negative_limit),
                zero_init=True,
            )
        self.full_order_coupling_tensor_field = None
        if self.use_full_order_coupling_tensor_field:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_full_order_coupling_tensor_field=True.")
            self.full_order_coupling_tensor_field = FullOrderCouplingTensorField(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                feature_map_dim=int(full_order_coupling_feature_map_dim),
                token_dim=int(full_order_coupling_token_dim),
                hidden_dim=int(full_order_coupling_hidden_dim),
                scale=float(full_order_coupling_scale),
                gate_threshold=float(full_order_coupling_gate_threshold),
                gate_temperature=float(full_order_coupling_gate_temperature),
                core_gate_floor=float(full_order_coupling_core_gate_floor),
                protected_negative_limit=float(full_order_coupling_protected_negative_limit),
                zero_init=True,
            )
        self.mechanism_charted_full_order_coupling_tensor_field = None
        if self.use_mechanism_charted_full_order_coupling_tensor_field:
            if class_to_idx is None:
                raise ValueError(
                    "class_to_idx is required when "
                    "use_mechanism_charted_full_order_coupling_tensor_field=True."
                )
            self.mechanism_charted_full_order_coupling_tensor_field = (
                MechanismChartedFullOrderCouplingTensorField(
                    in_dim=head_dim,
                    class_to_idx=class_to_idx,
                    feature_map_dim=int(mechanism_charted_full_order_feature_map_dim),
                    token_dim=int(mechanism_charted_full_order_token_dim),
                    hidden_dim=int(mechanism_charted_full_order_hidden_dim),
                    scale=float(mechanism_charted_full_order_scale),
                    gate_threshold=float(mechanism_charted_full_order_gate_threshold),
                    gate_temperature=float(mechanism_charted_full_order_gate_temperature),
                    core_gate_floor=float(mechanism_charted_full_order_core_gate_floor),
                    protected_negative_limit=float(mechanism_charted_full_order_protected_negative_limit),
                    router_hidden_dim=int(mechanism_charted_full_order_router_hidden_dim),
                    router_temperature=float(mechanism_charted_full_order_router_temperature),
                    sparse_router_topk=int(mechanism_charted_full_order_sparse_router_topk),
                    physics_prior_weight=float(mechanism_charted_full_order_physics_prior_weight),
                    zero_init=True,
                )
            )
        self.core_factor_coupled_residual = None
        if self.use_core_factor_coupled_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_core_factor_coupled_residual=True.")
            self.core_factor_coupled_residual = CoreFactorCoupledResidual(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                rank=int(core_factor_rank),
                scale=float(core_factor_scale),
                normalize=bool(factorized_normalize),
                neighbor_gate_floor=float(core_factor_neighbor_gate_floor),
                uncertainty_threshold=float(core_factor_uncertainty_threshold),
                uncertainty_temperature=float(core_factor_uncertainty_temperature),
                protected_negative_limit=float(core_factor_protected_negative_limit),
                zero_init=True,
            )
        self.water_evidence_logit_gate = None
        if self.use_water_evidence_logit_gate:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_water_evidence_logit_gate=True.")
            self.water_evidence_logit_gate = WaterEvidenceLogitGate(
                class_to_idx=class_to_idx,
                scale=float(water_evidence_gate_scale),
                zero_init=bool(water_evidence_gate_zero_init),
            )
        self.dry_concrete_roughness_vor_residual = None
        if self.use_dry_concrete_roughness_vor_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_dry_concrete_roughness_vor_residual=True.")
            self.dry_concrete_roughness_vor_residual = DryConcreteRoughnessVORResidual(
                class_to_idx=class_to_idx,
                hidden_dim=int(dry_concrete_roughness_hidden_dim),
                scale=float(dry_concrete_roughness_scale),
                gate_threshold=float(dry_concrete_roughness_gate_threshold),
                gate_temperature=float(dry_concrete_roughness_gate_temperature),
                zero_init=True,
            )
        self.dry_paved_roughness_vor_residual = None
        if self.use_dry_paved_roughness_vor_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_dry_paved_roughness_vor_residual=True.")
            self.dry_paved_roughness_vor_residual = DryPavedRoughnessVORResidual(
                class_to_idx=class_to_idx,
                hidden_dim=int(dry_paved_roughness_hidden_dim),
                material_dim=int(dry_paved_roughness_material_dim),
                scale=float(dry_paved_roughness_scale),
                gate_threshold=float(dry_paved_roughness_gate_threshold),
                gate_temperature=float(dry_paved_roughness_gate_temperature),
                head_mode=str(dry_paved_roughness_head_mode),
                material_gate_threshold=float(dry_paved_roughness_material_gate_threshold),
                material_gate_temperature=float(dry_paved_roughness_material_gate_temperature),
                zero_init=True,
            )
        self.concrete_roughness_vor_residual = None
        if self.use_concrete_roughness_vor_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_concrete_roughness_vor_residual=True.")
            self.concrete_roughness_vor_residual = ConcreteRoughnessVORResidual(
                class_to_idx=class_to_idx,
                hidden_dim=int(concrete_roughness_hidden_dim),
                chart_dim=int(concrete_roughness_chart_dim),
                scale=float(concrete_roughness_scale),
                gate_threshold=float(concrete_roughness_gate_threshold),
                gate_temperature=float(concrete_roughness_gate_temperature),
                zero_init=True,
            )
        self.wet_water_film_vor_residual = None
        if self.use_wet_water_film_vor_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_wet_water_film_vor_residual=True.")
            self.wet_water_film_vor_residual = WetWaterFilmVORResidual(
                class_to_idx=class_to_idx,
                hidden_dim=int(wet_water_film_hidden_dim),
                pair_dim=int(wet_water_film_pair_dim),
                scale=float(wet_water_film_scale),
                material_scope=str(wet_water_film_material_scope),
                gate_threshold=float(wet_water_film_gate_threshold),
                gate_temperature=float(wet_water_film_gate_temperature),
                zero_init=True,
            )
        self.smooth_film_concrete_expert = None
        if self.use_smooth_film_concrete_expert:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_smooth_film_concrete_expert=True.")
            self.smooth_film_concrete_expert = SmoothFilmConcreteExpert(
                class_to_idx=class_to_idx,
                hidden_dim=int(smooth_film_concrete_hidden_dim),
                scale=float(smooth_film_concrete_scale),
                gate_threshold=float(smooth_film_concrete_gate_threshold),
                gate_temperature=float(smooth_film_concrete_gate_temperature),
                zero_init=True,
            )
        self.obstruction_concrete_roughness_vor_residual = None
        if self.use_obstruction_concrete_roughness_vor_residual:
            if class_to_idx is None:
                raise ValueError(
                    "class_to_idx is required when use_obstruction_concrete_roughness_vor_residual=True."
                )
            self.obstruction_concrete_roughness_vor_residual = ObstructionAwareConcreteRoughnessVORResidual(
                class_to_idx=class_to_idx,
                hidden_dim=int(obstruction_concrete_roughness_hidden_dim),
                scale=float(obstruction_concrete_roughness_scale),
                gate_threshold=float(obstruction_concrete_roughness_gate_threshold),
                gate_temperature=float(obstruction_concrete_roughness_gate_temperature),
                share_gate_threshold=float(obstruction_concrete_roughness_share_gate_threshold),
                share_gate_temperature=float(obstruction_concrete_roughness_share_gate_temperature),
                zero_init=True,
            )
        self.coupled_optical_roughness_residual = None
        if self.use_coupled_optical_roughness_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_coupled_optical_roughness_residual=True.")
            self.coupled_optical_roughness_residual = CoupledOpticalRoughnessResidual(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                hidden_dim=int(coupled_residual_hidden_dim),
                scale=float(coupled_residual_scale),
                gate_threshold=float(coupled_residual_gate_threshold),
                gate_temperature=float(coupled_residual_gate_temperature),
                protected_negative_limit=float(coupled_residual_protected_negative_limit),
                zero_init=True,
            )
        self.roughness_neighbor_residual = None
        if self.use_roughness_neighbor_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_roughness_neighbor_residual=True.")
            self.roughness_neighbor_residual = RoughnessNeighborResidual(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                hidden_dim=int(roughness_neighbor_hidden_dim),
                scale=float(roughness_neighbor_scale),
                gate_threshold=float(roughness_neighbor_gate_threshold),
                gate_temperature=float(roughness_neighbor_gate_temperature),
                protected_negative_limit=float(roughness_neighbor_protected_negative_limit),
                neighbor_gate_floor=float(roughness_neighbor_gate_floor),
                zero_init=True,
            )
        self.spectral_roughness_residual = None
        if self.use_spectral_roughness_residual:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_spectral_roughness_residual=True.")
            self.spectral_roughness_residual = SpectralRoughnessResidual(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                hidden_dim=int(spectral_roughness_hidden_dim),
                scale=float(spectral_roughness_scale),
                gate_threshold=float(spectral_roughness_gate_threshold),
                gate_temperature=float(spectral_roughness_gate_temperature),
                protected_negative_limit=float(spectral_roughness_protected_negative_limit),
                neighbor_gate_floor=float(spectral_roughness_neighbor_gate_floor),
                zero_init=True,
            )
        self.relation_signed_graph_expert = None
        if self.use_relation_signed_graph_expert:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_relation_signed_graph_expert=True.")
            self.relation_signed_graph_expert = RelationSignedGraphExpert(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                hidden_dim=int(relation_signed_hidden_dim),
                scale=float(relation_signed_scale),
                gate_threshold=float(relation_signed_gate_threshold),
                gate_temperature=float(relation_signed_gate_temperature),
                protected_negative_limit=float(relation_signed_protected_negative_limit),
                neighbor_gate_floor=float(relation_signed_neighbor_gate_floor),
                zero_init=True,
            )
        self.heterophilic_logit_boundary_expert = None
        if self.use_heterophilic_logit_boundary_expert:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_heterophilic_logit_boundary_expert=True.")
            self.heterophilic_logit_boundary_expert = HeterophilicLogitBoundaryExpert(
                class_to_idx=class_to_idx,
                scale=float(heterophilic_boundary_scale),
                gate_threshold=float(heterophilic_boundary_gate_threshold),
                gate_temperature=float(heterophilic_boundary_gate_temperature),
                protected_negative_limit=float(heterophilic_boundary_protected_negative_limit),
            )
        self.heterophilic_feature_boundary_expert = None
        if self.use_heterophilic_feature_boundary_expert:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_heterophilic_feature_boundary_expert=True.")
            self.heterophilic_feature_boundary_expert = HeterophilicFeatureBoundaryExpert(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                hidden_dim=int(heterophilic_feature_boundary_hidden_dim),
                pair_dim=int(heterophilic_feature_boundary_pair_dim),
                scale=float(heterophilic_feature_boundary_scale),
                gate_threshold=float(heterophilic_feature_boundary_gate_threshold),
                gate_temperature=float(heterophilic_feature_boundary_gate_temperature),
                protected_negative_limit=float(heterophilic_feature_boundary_protected_negative_limit),
                zero_init=True,
            )
        self.heterophilic_physics_boundary_expert = None
        if self.use_heterophilic_physics_boundary_expert:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_heterophilic_physics_boundary_expert=True.")
            self.heterophilic_physics_boundary_expert = HeterophilicPhysicsBoundaryExpert(
                class_to_idx=class_to_idx,
                hidden_dim=int(heterophilic_physics_boundary_hidden_dim),
                pair_dim=int(heterophilic_physics_boundary_pair_dim),
                scale=float(heterophilic_physics_boundary_scale),
                gate_threshold=float(heterophilic_physics_boundary_gate_threshold),
                gate_temperature=float(heterophilic_physics_boundary_gate_temperature),
                protected_negative_limit=float(heterophilic_physics_boundary_protected_negative_limit),
                zero_init=True,
            )
        self.protected_heterophilic_factor_boundary_field = None
        if self.use_protected_heterophilic_factor_boundary_field:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_protected_heterophilic_factor_boundary_field=True.")
            self.protected_heterophilic_factor_boundary_field = ProtectedHeterophilicFactorBoundaryField(
                class_to_idx=class_to_idx,
                hidden_dim=int(protected_factor_boundary_hidden_dim),
                pair_dim=int(protected_factor_boundary_pair_dim),
                relation_dim=int(protected_factor_boundary_relation_dim),
                scale=float(protected_factor_boundary_scale),
                gate_threshold=float(protected_factor_boundary_gate_threshold),
                gate_temperature=float(protected_factor_boundary_gate_temperature),
                protected_negative_limit=float(protected_factor_boundary_protected_negative_limit),
                zero_init=True,
            )
        self.relation_specific_hard_edge_refiner = None
        if self.use_relation_specific_hard_edge_refiner:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_relation_specific_hard_edge_refiner=True.")
            self.relation_specific_hard_edge_refiner = RelationSpecificHardEdgeRefiner(
                class_to_idx=class_to_idx,
                hidden_dim=int(relation_specific_refiner_hidden_dim),
                pair_dim=int(relation_specific_refiner_pair_dim),
                scale=float(relation_specific_refiner_scale),
                gate_threshold=float(relation_specific_refiner_gate_threshold),
                gate_temperature=float(relation_specific_refiner_gate_temperature),
                protected_negative_limit=float(relation_specific_refiner_protected_negative_limit),
                zero_init=True,
            )
        self.selective_mechanism_tensor_boundary_field = None
        if self.use_selective_mechanism_tensor_boundary_field:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_selective_mechanism_tensor_boundary_field=True.")
            self.selective_mechanism_tensor_boundary_field = SelectiveMechanismTensorBoundaryField(
                class_to_idx=class_to_idx,
                hidden_dim=int(selective_mechanism_tensor_boundary_hidden_dim),
                pair_dim=int(selective_mechanism_tensor_boundary_pair_dim),
                relation_dim=int(selective_mechanism_tensor_boundary_relation_dim),
                mechanism_dim=int(selective_mechanism_tensor_boundary_mechanism_dim),
                scale=float(selective_mechanism_tensor_boundary_scale),
                gate_threshold=float(selective_mechanism_tensor_boundary_gate_threshold),
                gate_temperature=float(selective_mechanism_tensor_boundary_gate_temperature),
                mechanism_gate_threshold=float(selective_mechanism_tensor_boundary_mechanism_gate_threshold),
                mechanism_gate_temperature=float(selective_mechanism_tensor_boundary_mechanism_gate_temperature),
                enabled_mechanisms=str(selective_mechanism_tensor_boundary_enabled_mechanisms),
                protected_negative_limit=float(selective_mechanism_tensor_boundary_protected_negative_limit),
                zero_init=True,
            )
        self.conditional_factor_projection = None
        if self.use_conditional_factor_projection:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_conditional_factor_projection=True.")
            self.conditional_factor_projection = ConditionalFactorConsistencyProjection(
                class_to_idx=class_to_idx,
                scale=float(conditional_factor_projection_scale),
                gate_threshold=float(conditional_factor_projection_gate_threshold),
                gate_temperature=float(conditional_factor_projection_gate_temperature),
                friction_weight=float(conditional_factor_projection_friction_weight),
                material_weight=float(conditional_factor_projection_material_weight),
                unevenness_weight=float(conditional_factor_projection_unevenness_weight),
                focus=str(conditional_factor_projection_focus),
                protected_negative_limit=float(conditional_factor_projection_protected_negative_limit),
            )
        self.heterogeneous_label_router = None
        if self.use_heterogeneous_label_router:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_heterogeneous_label_router=True.")
            self.heterogeneous_label_router = HeterogeneousLabelRouter(
                in_dim=head_dim,
                class_to_idx=class_to_idx,
                hidden_dim=int(heterogeneous_router_hidden_dim),
                scale=float(heterogeneous_router_scale),
                zero_init=True,
            )
        self.factor_logit_heads = (
            nn.ModuleDict(
                {
                    name: nn.Linear(head_dim, len(labels))
                    for name, labels in FACTOR_LABELS.items()
                }
            )
            if self.use_factor_logit_adjustment
            else None
        )
        if self.factor_logit_heads is not None:
            if class_to_idx is None:
                raise ValueError("class_to_idx is required when use_factor_logit_adjustment=True.")
            factor_buffers = build_class_factor_buffers(class_to_idx)
            for name, (indices, mask) in factor_buffers.items():
                self.register_buffer(f"{name}_class_factor_idx", indices)
                self.register_buffer(f"{name}_class_factor_mask", mask)
            for head in self.factor_logit_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        self.factor_heads = (
            nn.ModuleDict(
                {
                    name: nn.Linear(head_dim, len(labels))
                    for name, labels in FACTOR_LABELS.items()
                }
            )
            if self.use_factor_aux
            else None
        )
        self.local_physics_factor_heads = (
            nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.LayerNorm(local_physics_field_dim),
                        nn.Linear(local_physics_field_dim, len(labels)),
                    )
                    for name, labels in FACTOR_LABELS.items()
                }
            )
            if self.use_local_physics_factor_aux
            else None
        )
        self.hard_pair_aux_head = (
            nn.Sequential(
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, int(hard_pair_aux_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hard_pair_aux_hidden_dim), int(hard_pair_aux_num_pairs)),
            )
            if self.use_hard_pair_aux
            else None
        )
        self.physics_aux_classifier = (
            nn.Linear(physics_dim, num_classes)
            if self.use_physics_aux
            else None
        )
        self.backbone_aux_classifier = (
            nn.Linear(embedding_dim, num_classes)
            if self.use_backbone_aux
            else None
        )
        self.physics_evidence_aux_heads = (
            PhysicsEvidenceMapHeads({"early": 96, "mid": 192, "late": 384, "final": 768})
            if self.use_physics_evidence_aux
            else None
        )
        self.mechanism_orthogonal_aux = (
            MechanismOrthogonalCouplingAuxiliary(
                head_dim,
                proj_dim=int(mechanism_orthogonal_dim),
                hidden_dim=int(mechanism_orthogonal_hidden_dim),
            )
            if self.use_mechanism_orthogonal_coupling_aux
            else None
        )

    def forward(self, image: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        backbone_feature = self.backbone(image)
        parts = [backbone_feature]
        local_physics_feature = None
        low_level_feature = None
        if self.local_physics_field_branch is not None and self.local_physics_field_adapter is not None:
            local_physics_feature = self.local_physics_field_branch(image)
            backbone_feature = backbone_feature + float(self.local_physics_field_scale) * self.local_physics_field_adapter(
                local_physics_feature
            )
            parts[0] = backbone_feature
        low_level_parts = []
        physics_feature = None
        if self.physics_branch is not None:
            physics_feature = self.physics_branch(image)
            low_level_parts.append(physics_feature)
        if self.directional_texture_branch is not None:
            low_level_parts.append(self.directional_texture_branch(image))
        if self.wavelet_texture_branch is not None:
            low_level_parts.append(self.wavelet_texture_branch(image))
        if self.retinex_texture_branch is not None:
            low_level_parts.append(self.retinex_texture_branch(image))
        if self.physics_attention_branch is not None:
            low_level_parts.append(self.physics_attention_branch(image))
        if self.semantic_physics_attention_branch is not None:
            low_level_parts.append(self.semantic_physics_attention_branch(image))
        if self.visibility_observed_roughness_branch is not None:
            visibility_observed_roughness_feature = self.visibility_observed_roughness_branch(image)
            if self.visibility_observed_roughness_adapter is not None:
                backbone_feature = backbone_feature + float(
                    self.visibility_observed_roughness_scale
                ) * self.visibility_observed_roughness_adapter(visibility_observed_roughness_feature)
                parts[0] = backbone_feature
            else:
                low_level_parts.append(visibility_observed_roughness_feature)
        if self.factor_conditioned_physics_token_branch is not None:
            low_level_parts.append(self.factor_conditioned_physics_token_branch(image))
        if self.factor_coupled_physics_token_branch is not None:
            low_level_parts.append(self.factor_coupled_physics_token_branch(image))
        if self.relation_conditioned_physics_expert_branch is not None:
            relation_expert_feature = self.relation_conditioned_physics_expert_branch(image)
            if self.relation_conditioned_physics_expert_adapter is not None:
                backbone_feature = backbone_feature + float(
                    self.relation_conditioned_physics_expert_scale
                ) * self.relation_conditioned_physics_expert_adapter(relation_expert_feature)
                parts[0] = backbone_feature
            else:
                low_level_parts.append(relation_expert_feature)
        if self.topological_texture_branch is not None:
            low_level_parts.append(self.topological_texture_branch(image))
        if self.anti_human_texture_branch is not None:
            low_level_parts.append(self.anti_human_texture_branch(image))
        if low_level_parts:
            low_level_feature = torch.cat(low_level_parts, dim=1)
            if self.texture_gate is not None:
                low_level_feature = low_level_feature * self.texture_gate(backbone_feature)
            if self.material_conditioned_texture_gate is not None:
                material_logits = self.material_conditioned_texture_gate["material_logits"](backbone_feature)
                material_prob = F.softmax(material_logits, dim=1)
                gate_delta = torch.tanh(self.material_conditioned_texture_gate["gate"](material_prob))
                low_level_feature = low_level_feature * (
                    1.0 + float(self.material_conditioned_gate_scale) * gate_delta
                )
            if self.artifact_aware_texture_gate is not None:
                low_level_feature = self.artifact_aware_texture_gate(image, low_level_feature)
            if self.smooth_evidence_texture_gate is not None:
                low_level_feature = self.smooth_evidence_texture_gate(image, low_level_feature)
            if self.mechanism_conditioned_artifact_gate is not None:
                low_level_feature = self.mechanism_conditioned_artifact_gate(
                    image,
                    backbone_feature,
                    low_level_feature,
                )
            if self.tri_chart_evidence_film is not None:
                backbone_feature = self.tri_chart_evidence_film(
                    image,
                    backbone_feature,
                    low_level_feature,
                )
                parts[0] = backbone_feature
            if self.texture_film is not None:
                gamma_beta = self.texture_film(low_level_feature)
                gamma, beta = gamma_beta.chunk(2, dim=1)
                scale = float(self.texture_film_scale)
                backbone_feature = backbone_feature * (1.0 + scale * torch.tanh(gamma)) + scale * beta
                parts[0] = backbone_feature
            if self.texture_residual_adapter is not None:
                backbone_feature = backbone_feature + self.texture_residual_scale * self.texture_residual_adapter(
                    low_level_feature
                )
                parts[0] = backbone_feature
            if self.texture_residual_adapter is None and self.texture_film is None:
                parts.append(low_level_feature)
        feature = torch.cat(parts, dim=1)
        feature = self.dropout(self.norm(feature))
        logits = self.classifier(feature)
        if self.factorized_low_rank_head is not None:
            logits = logits + self.factorized_low_rank_head(feature)
        if self.safe_factorized_low_rank_head is not None:
            logits = logits + self.safe_factorized_low_rank_head(feature, logits)
        if self.factor_interaction_low_rank_head is not None:
            logits = logits + self.factor_interaction_low_rank_head(feature, logits)
        if self.conditional_coupling_decomposition_field is not None:
            logits = logits + self.conditional_coupling_decomposition_field(feature, logits)
        mobius_aux = None
        hmc_sheaf_aux = None
        if self.mobius_sheaf_factor_head is not None:
            mobius_out = self.mobius_sheaf_factor_head(feature, return_aux=return_aux)
            if isinstance(mobius_out, dict):
                mobius_logits = mobius_out["logits"]
                mobius_aux = mobius_out
            else:
                mobius_logits = mobius_out
            if self.mobius_sheaf_mode == "replace":
                logits = mobius_logits
            elif self.mobius_sheaf_mode == "blend":
                blend = float(self.mobius_sheaf_blend)
                logits = (1.0 - blend) * logits + blend * mobius_logits
            else:
                logits = logits + mobius_logits
        if self.mechanism_conditional_sheaf_head is not None:
            hmc_out = self.mechanism_conditional_sheaf_head(image, feature, logits, return_aux=return_aux)
            if isinstance(hmc_out, dict):
                hmc_logits = hmc_out["logits"]
                hmc_sheaf_aux = hmc_out
            else:
                hmc_logits = hmc_out
            logits = logits + hmc_logits
        if self.local_global_factor_attention is not None:
            logits = logits + self.local_global_factor_attention(feature, logits, local_physics_feature)
        if self.label_graph_residual is not None:
            logits = logits + self.label_graph_residual(feature, logits)
        if self.conditional_evidence_masked_coupling_field is not None:
            feature_map = getattr(self.backbone, "last_feature_map", None)
            logits = logits + self.conditional_evidence_masked_coupling_field(image, feature_map, feature, logits)
        if self.full_order_coupling_tensor_field is not None:
            feature_map = getattr(self.backbone, "last_feature_map", None)
            logits = logits + self.full_order_coupling_tensor_field(image, feature_map, feature, logits)
        if self.mechanism_charted_full_order_coupling_tensor_field is not None:
            feature_map = getattr(self.backbone, "last_feature_map", None)
            logits = logits + self.mechanism_charted_full_order_coupling_tensor_field(
                image,
                feature_map,
                feature,
                logits,
            )
        if self.core_factor_coupled_residual is not None:
            logits = logits + self.core_factor_coupled_residual(image, feature, logits)
        if self.water_evidence_logit_gate is not None:
            logits = logits + self.water_evidence_logit_gate(image)
        if self.dry_concrete_roughness_vor_residual is not None:
            logits = logits + self.dry_concrete_roughness_vor_residual(image, logits)
        if self.dry_paved_roughness_vor_residual is not None:
            logits = logits + self.dry_paved_roughness_vor_residual(image, logits)
        if self.concrete_roughness_vor_residual is not None:
            logits = logits + self.concrete_roughness_vor_residual(image, logits)
        if self.wet_water_film_vor_residual is not None:
            logits = logits + self.wet_water_film_vor_residual(image, logits)
        if self.smooth_film_concrete_expert is not None:
            logits = logits + self.smooth_film_concrete_expert(image, logits)
        if self.obstruction_concrete_roughness_vor_residual is not None:
            logits = logits + self.obstruction_concrete_roughness_vor_residual(image, logits)
        if self.coupled_optical_roughness_residual is not None:
            logits = logits + self.coupled_optical_roughness_residual(image, feature, logits)
        if self.roughness_neighbor_residual is not None:
            logits = logits + self.roughness_neighbor_residual(image, feature, logits)
        if self.spectral_roughness_residual is not None:
            logits = logits + self.spectral_roughness_residual(image, feature, logits)
        if self.relation_signed_graph_expert is not None:
            logits = logits + self.relation_signed_graph_expert(image, feature, logits)
        if self.heterophilic_logit_boundary_expert is not None:
            logits = logits + self.heterophilic_logit_boundary_expert(logits)
        if self.heterophilic_feature_boundary_expert is not None:
            logits = logits + self.heterophilic_feature_boundary_expert(feature, logits)
        if self.heterophilic_physics_boundary_expert is not None:
            logits = logits + self.heterophilic_physics_boundary_expert(image, logits)
        if self.protected_heterophilic_factor_boundary_field is not None:
            logits = logits + self.protected_heterophilic_factor_boundary_field(image, logits)
        if self.relation_specific_hard_edge_refiner is not None:
            logits = logits + self.relation_specific_hard_edge_refiner(image, logits)
        if self.selective_mechanism_tensor_boundary_field is not None:
            logits = logits + self.selective_mechanism_tensor_boundary_field(image, logits)
        if self.conditional_factor_projection is not None:
            logits = logits + self.conditional_factor_projection(logits)
        if self.heterogeneous_label_router is not None:
            logits = logits + self.heterogeneous_label_router(feature)
        factor_adjustment_logits = None
        if self.factor_logit_heads is not None:
            factor_logits = {name: head(feature) for name, head in self.factor_logit_heads.items()}
            factor_adjustment_logits = factor_logits
            class_adjustment = 0.0
            active = 0.0
            for name, logits_by_factor in factor_logits.items():
                factor_idx = getattr(self, f"{name}_class_factor_idx")
                factor_mask = getattr(self, f"{name}_class_factor_mask").to(dtype=logits_by_factor.dtype)
                gathered = logits_by_factor[:, factor_idx] * factor_mask.unsqueeze(0)
                class_adjustment = class_adjustment + gathered
                active = active + factor_mask.unsqueeze(0)
            class_adjustment = class_adjustment / active.clamp_min(1.0)
            logits = logits + self.factor_logit_adjustment_scale * class_adjustment
        if not return_aux:
            return logits
        out: dict[str, Any] = {"logits": logits, "feature": feature}
        out["backbone_feature"] = backbone_feature
        if low_level_feature is not None:
            out["low_level_feature"] = low_level_feature
        if local_physics_feature is not None:
            out["local_physics_feature"] = local_physics_feature
        if physics_feature is not None:
            out["physics_feature"] = physics_feature
        if mobius_aux is not None and "mobius_gates" in mobius_aux:
            out["mobius_gates"] = mobius_aux["mobius_gates"]
        if hmc_sheaf_aux is not None:
            out["hmc_sheaf_router"] = hmc_sheaf_aux["hmc_sheaf_router"]
            out["hmc_sheaf_effect_gate"] = hmc_sheaf_aux["hmc_sheaf_effect_gate"]
        if self.factor_heads is not None:
            out["factor_logits"] = {
                name: head(feature)
                for name, head in self.factor_heads.items()
            }
        elif factor_adjustment_logits is not None:
            out["factor_logits"] = factor_adjustment_logits
        if self.local_physics_factor_heads is not None and local_physics_feature is not None:
            out["local_physics_factor_logits"] = {
                name: head(local_physics_feature)
                for name, head in self.local_physics_factor_heads.items()
            }
        if self.hard_pair_aux_head is not None:
            out["hard_pair_logits"] = self.hard_pair_aux_head(feature)
        if self.mechanism_orthogonal_aux is not None:
            out["mechanism_orthogonal_aux"] = self.mechanism_orthogonal_aux(feature)
        if self.physics_aux_classifier is not None and physics_feature is not None:
            out["physics_logits"] = self.physics_aux_classifier(physics_feature)
        if self.backbone_aux_classifier is not None:
            out["backbone_logits"] = self.backbone_aux_classifier(backbone_feature)
        if self.physics_evidence_aux_heads is not None:
            stage_maps = getattr(self.backbone, "stage_feature_maps", {})
            if isinstance(stage_maps, dict):
                evidence_logits = self.physics_evidence_aux_heads(stage_maps)
                if evidence_logits:
                    out["physics_evidence_logits"] = evidence_logits
        return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RSCD 27-class road-surface classification protocol. This is a separate "
            "benchmark from the weak friction/risk/interval protocol and is intended "
            "for fair discussion of RSCD-style papers such as RoadFormer."
        )
    )
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backbone", default="convnext_tiny")
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--eval-resize-mode",
        choices=("letterbox", "stretch", "bottom_square"),
        default="letterbox",
        help=(
            "Deterministic validation/test geometry. Existing FAF ConvNeXt runs use "
            "letterbox by default; public RSPNet checkpoints require stretch to "
            "match their ImageFolder-style Resize((224,224)) protocol."
        ),
    )
    parser.add_argument(
        "--train-resize-mode",
        choices=("letterbox", "stretch", "bottom_square"),
        default="letterbox",
        help=(
            "Training geometry. Keep letterbox for existing FAF ConvNeXt runs; use "
            "stretch when head-only training from public RSPNet checkpoints."
        ),
    )
    parser.add_argument(
        "--train-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use stochastic training augmentation. Disable for frozen-head "
            "calibration experiments where the pretrained backbone geometry and "
            "photometry should match evaluation exactly."
        ),
    )
    parser.add_argument(
        "--train-mechanism-scope",
        choices=MECHANISM_TRAIN_SCOPES,
        default="all",
        help=(
            "Filter only the training set for staged mechanism-curriculum runs. "
            "Validation and test remain the complete RSCD 27-class protocol. "
            "Use this to pre-shape the same backbone on dry-visible texture, "
            "wet/water film, granular surfaces, or winter phase before full-data "
            "fine-tuning."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--line-erasing-p",
        type=float,
        default=0.0,
        help=(
            "Training-only shortcut suppression probability. It erases a few thin "
            "horizontal/vertical strips so RSCD models rely less on lane markings, "
            "drain grates, arrows, and other elongated non-surface artifacts."
        ),
    )
    parser.add_argument("--line-erasing-min-lines", type=int, default=1)
    parser.add_argument("--line-erasing-max-lines", type=int, default=3)
    parser.add_argument("--line-erasing-min-length", type=float, default=0.35)
    parser.add_argument("--line-erasing-max-length", type=float, default=0.95)
    parser.add_argument("--line-erasing-min-width", type=float, default=0.015)
    parser.add_argument("--line-erasing-max-width", type=float, default=0.055)
    parser.add_argument(
        "--gray-world-alpha",
        type=float,
        default=0.0,
        help=(
            "Apply soft gray-world color constancy before normalization. This reduces "
            "global camera/illumination color cast while preserving road texture evidence."
        ),
    )
    parser.add_argument(
        "--fourier-low-freq-jitter-p",
        type=float,
        default=0.0,
        help=(
            "Training probability for Fourier low-frequency amplitude jitter. This "
            "perturbs image style and illumination without changing road texture phase."
        ),
    )
    parser.add_argument("--fourier-beta", type=float, default=0.08)
    parser.add_argument("--fourier-strength-min", type=float, default=0.75)
    parser.add_argument("--fourier-strength-max", type=float, default=1.25)
    parser.add_argument("--use-physics-branch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--physics-dim", type=int, default=96)
    parser.add_argument("--physics-quality-cues", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--physics-quality-region-cues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep vertical bottom-vs-top quality cues inside PhysicsTexture. Disable "
            "for RSCD-style close road patches where there is no reliable contact-zone geometry."
        ),
    )
    parser.add_argument("--use-directional-texture-branch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--directional-texture-dim", type=int, default=64)
    parser.add_argument("--use-wavelet-texture-branch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wavelet-texture-dim", type=int, default=64)
    parser.add_argument(
        "--use-retinex-texture-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Retinex/chromaticity illumination-invariant texture cues for "
            "wet-film, shadow-water, and specular-road ambiguity."
        ),
    )
    parser.add_argument("--retinex-texture-dim", type=int, default=48)
    parser.add_argument(
        "--retinex-region-cues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep vertical bottom-vs-top cues inside RetinexTexture. Disable "
            "for patch-style datasets such as RSCD where vertical position is "
            "not a stable tire-contact prior."
        ),
    )
    parser.add_argument("--use-physics-attention-branch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--physics-attention-dim", type=int, default=64)
    parser.add_argument(
        "--use-semantic-physics-attention-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use patch-invariant weak semantic attention over physics-derived "
            "regions: snow, mirror water, dark water, thin film, rough aggregate, "
            "granular texture, and marking-like artifacts. Unlike the older "
            "PhysicsAttention branch, it avoids vertical contact-zone priors."
        ),
    )
    parser.add_argument("--semantic-physics-attention-dim", type=int, default=64)
    parser.add_argument(
        "--use-visibility-observed-roughness-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add VOR: visibility-observed roughness descriptors that separate "
            "water-film/specular/dark-water obstruction from visible and hidden "
            "roughness evidence. This targets wet/water concrete roughness "
            "confusions without using new labels."
        ),
    )
    parser.add_argument("--visibility-observed-roughness-dim", type=int, default=64)
    parser.add_argument(
        "--use-visibility-observed-roughness-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject VOR through a zero-initialized feature adapter instead of "
            "expanding the classifier input. This preserves the calibrated "
            "PhysicsTexture/SemanticPhysicsAttention head while learning a small "
            "optical-roughness correction."
        ),
    )
    parser.add_argument("--visibility-observed-roughness-scale", type=float, default=0.04)
    parser.add_argument(
        "--use-factor-conditioned-physics-token-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use factor-query attention over weak physics evidence tokens. The "
            "branch builds snow/water/film/rough/granular/marking tokens and lets "
            "friction, material, and unevenness queries select different evidence."
        ),
    )
    parser.add_argument("--factor-conditioned-physics-token-dim", type=int, default=48)
    parser.add_argument("--factor-conditioned-physics-token-inner-dim", type=int, default=16)
    parser.add_argument(
        "--use-factor-coupled-physics-token-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add FCPE: factor-coupled physical evidence tokens. The branch learns "
            "single-factor, pairwise, and three-way coupling queries over weak "
            "physics evidence before the classifier, targeting RSCD compositional "
            "labels without using RSPNet-style backbones."
        ),
    )
    parser.add_argument("--factor-coupled-physics-token-dim", type=int, default=64)
    parser.add_argument("--factor-coupled-physics-token-inner-dim", type=int, default=16)
    parser.add_argument(
        "--use-local-physics-field-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized residual adapter from weak segmentation-style "
            "local physics evidence maps. This keeps the classifier shape unchanged "
            "while testing whether local water-film/texture fields improve hard RSCD cells."
        ),
    )
    parser.add_argument("--local-physics-field-dim", type=int, default=64)
    parser.add_argument(
        "--local-physics-field-scale",
        type=float,
        default=0.15,
        help="Scale for the local physics field residual adapter.",
    )
    parser.add_argument(
        "--use-relation-conditioned-physics-expert-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add RCPE: relation-conditioned physical evidence experts before "
            "classification. The branch keeps separate wet/water, roughness, "
            "and material-granularity experts over local physics/topology fields."
        ),
    )
    parser.add_argument("--relation-conditioned-physics-expert-dim", type=int, default=72)
    parser.add_argument("--relation-conditioned-physics-expert-inner-dim", type=int, default=24)
    parser.add_argument(
        "--use-relation-conditioned-physics-expert-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject RCPE through a zero-initialized residual adapter into the "
            "ConvNeXt feature instead of concatenating it into the classifier."
        ),
    )
    parser.add_argument(
        "--relation-conditioned-physics-expert-scale",
        type=float,
        default=0.06,
        help="Scale for the RCPE residual feature adapter.",
    )
    parser.add_argument(
        "--use-topological-texture-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use soft Euler-curve topology summaries over wet, snow, low-texture, "
            "gradient, and luminance maps. This is patch-invariant and needs no "
            "pixel-level labels."
        ),
    )
    parser.add_argument("--topological-texture-dim", type=int, default=48)
    parser.add_argument(
        "--use-anti-human-texture-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use patch-invariant tail statistics for visual cues humans often "
            "underweight: local glare, hidden dark water, contrast erasure, and "
            "low-saturation smooth films."
        ),
    )
    parser.add_argument("--anti-human-texture-dim", type=int, default=64)
    parser.add_argument("--use-texture-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-texture-residual-adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--texture-residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--use-texture-film",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use zero-initialized FiLM modulation from physics/directional texture cues "
            "instead of concatenating them into the classifier head."
        ),
    )
    parser.add_argument("--texture-film-scale", type=float, default=0.20)
    parser.add_argument(
        "--use-material-conditioned-texture-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a zero-initialized material-conditioned gate to low-level texture "
            "features. This targets wet/water concrete confusion by allowing asphalt, "
            "concrete, mud, and gravel cues to weight wetness/roughness evidence differently."
        ),
    )
    parser.add_argument("--material-conditioned-gate-scale", type=float, default=0.25)
    parser.add_argument(
        "--use-artifact-aware-texture-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a zero-initialized reliability gate over low-level texture features. "
            "The gate is driven by patch statistics for near-white markings, strong "
            "edges, specular highlights, dark smooth water, low texture, and "
            "anisotropic line-like evidence."
        ),
    )
    parser.add_argument(
        "--artifact-aware-gate-scale",
        type=float,
        default=0.20,
        help="Maximum tanh-scaled reliability correction for low-level texture features.",
    )
    parser.add_argument(
        "--use-smooth-evidence-texture-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a smooth-state evidence gate over low-level physics texture "
            "features. It targets the audited RSCD gap on high-support smooth "
            "classes while suppressing activation on rough/granular/broken-film "
            "patches that already benefit from PhysicsTexture and LocalPhysicsField."
        ),
    )
    parser.add_argument(
        "--smooth-evidence-texture-gate-scale",
        type=float,
        default=0.10,
        help="Maximum tanh-scaled smooth-evidence correction for low-level texture features.",
    )
    parser.add_argument(
        "--smooth-evidence-texture-gate-temperature",
        type=float,
        default=16.0,
        help="Temperature for optical smoothness evidence computed from gradient and macro texture.",
    )
    parser.add_argument(
        "--smooth-evidence-texture-gate-rough-suppression",
        type=float,
        default=0.65,
        help="How strongly rough/granular/broken-film evidence suppresses smooth-state gating.",
    )
    parser.add_argument(
        "--use-tri-chart-evidence-film",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use mutually gated smooth/film/granular evidence charts to FiLM the "
            "backbone feature before classification. This is a task-specific "
            "alternative to one global artifact gate for RSCD coupled labels."
        ),
    )
    parser.add_argument("--tri-chart-evidence-film-hidden-dim", type=int, default=256)
    parser.add_argument("--tri-chart-evidence-film-scale", type=float, default=0.035)
    parser.add_argument("--tri-chart-evidence-film-gate-temperature", type=float, default=18.0)
    parser.add_argument(
        "--use-mechanism-conditioned-artifact-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use mechanism-split artifact reliability experts over low-level "
            "physics texture evidence. This targets the RSCD failure mode where "
            "one global artifact gate helps asphalt smooth classes but damages "
            "concrete wet/water film boundaries."
        ),
    )
    parser.add_argument(
        "--mechanism-conditioned-artifact-gate-scale",
        type=float,
        default=0.12,
        help="Maximum tanh-scaled mechanism-conditioned reliability correction.",
    )
    parser.add_argument(
        "--mechanism-conditioned-artifact-gate-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for mechanism-conditioned artifact routing.",
    )
    parser.add_argument(
        "--use-factor-logit-adjustment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized factor-aware calibration head. It predicts "
            "friction/material/unevenness logits and projects them back to the "
            "27 RSCD class logits using the known label factor structure."
        ),
    )
    parser.add_argument(
        "--factor-logit-adjustment-scale",
        type=float,
        default=0.30,
        help="Scale for the factor-aware class-logit residual.",
    )
    parser.add_argument(
        "--use-factorized-low-rank-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a low-rank label-embedding residual head whose class weights are "
            "composed from RSCD friction/material/unevenness factors."
        ),
    )
    parser.add_argument("--factorized-rank", type=int, default=64)
    parser.add_argument(
        "--factorized-scale",
        type=float,
        default=0.25,
        help="Scale for the compositional low-rank class-logit residual.",
    )
    parser.add_argument(
        "--factorized-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cosine-style normalization inside the factorized low-rank head.",
    )
    parser.add_argument(
        "--factorized-zero-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize the low-rank residual to zero so the model starts as the baseline classifier.",
    )
    parser.add_argument(
        "--factorized-factors",
        type=str,
        default="friction,material,unevenness",
        help=(
            "Comma-separated RSCD factors used by the low-rank residual head. "
            "Use subsets such as friction or friction,material to test missing-factor interference."
        ),
    )
    parser.add_argument(
        "--factorized-class-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a per-class low-rank residual embedding in addition to shared factor embeddings.",
    )
    parser.add_argument(
        "--use-safe-factorized-low-rank-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add an uncertainty-gated compositional low-rank residual that protects "
            "wet/water class logits from large negative corrections."
        ),
    )
    parser.add_argument("--safe-factorized-rank", type=int, default=64)
    parser.add_argument(
        "--safe-factorized-scale",
        type=float,
        default=0.25,
        help="Scale for the safe adaptive compositional low-rank residual.",
    )
    parser.add_argument(
        "--safe-factorized-gate-threshold",
        type=float,
        default=0.55,
        help="Uncertainty threshold; residual is strongest above this base-model uncertainty.",
    )
    parser.add_argument(
        "--safe-factorized-gate-temperature",
        type=float,
        default=8.0,
        help="Sharpness of the uncertainty gate for the safe factorized residual.",
    )
    parser.add_argument(
        "--safe-factorized-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum allowed negative logit residual for protected wet/water classes.",
    )
    parser.add_argument(
        "--use-factor-interaction-low-rank-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized, uncertainty-gated low-rank residual over "
            "pairwise RSCD factor interactions such as friction x material and "
            "friction x unevenness."
        ),
    )
    parser.add_argument("--factor-interaction-rank", type=int, default=64)
    parser.add_argument(
        "--factor-interaction-scale",
        type=float,
        default=0.20,
        help="Scale for the pairwise factor-interaction residual.",
    )
    parser.add_argument(
        "--factor-interaction-gate-threshold",
        type=float,
        default=0.55,
        help="Base-model uncertainty threshold for activating pairwise interaction residuals.",
    )
    parser.add_argument(
        "--factor-interaction-gate-temperature",
        type=float,
        default=8.0,
        help="Sharpness of the uncertainty gate for pairwise interaction residuals.",
    )
    parser.add_argument(
        "--factor-interaction-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for protected wet/water classes.",
    )
    parser.add_argument(
        "--use-conditional-coupling-decomposition-field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add C-CDF: a dynamic low-rank field that routes each image over "
            "friction-material, friction-unevenness, and material-unevenness "
            "coupling experts."
        ),
    )
    parser.add_argument("--conditional-coupling-rank", type=int, default=64)
    parser.add_argument(
        "--conditional-coupling-scale",
        type=float,
        default=0.08,
        help="Scale for the conditional coupling decomposition residual.",
    )
    parser.add_argument(
        "--conditional-coupling-gate-threshold",
        type=float,
        default=0.35,
        help="Base-model uncertainty threshold for activating C-CDF.",
    )
    parser.add_argument(
        "--conditional-coupling-gate-temperature",
        type=float,
        default=8.0,
        help="Sharpness of the uncertainty gate for C-CDF.",
    )
    parser.add_argument(
        "--conditional-coupling-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for protected wet/water classes in C-CDF.",
    )
    parser.add_argument(
        "--conditional-coupling-relation-gate-hidden-dim",
        type=int,
        default=64,
        help="Hidden width of the per-image C-CDF relation gate.",
    )
    parser.add_argument(
        "--conditional-coupling-relation-gate-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature of the C-CDF relation gate.",
    )
    parser.add_argument(
        "--use-mobius-sheaf-factor-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add MSF-FAF: a Möbius/ANOVA factor-interaction classifier over RSCD "
            "friction, material, and unevenness factors. It models main effects, "
            "pairwise couplings, and paved-road three-way coupling instead of "
            "treating all 27 labels as unrelated symbols."
        ),
    )
    parser.add_argument("--mobius-sheaf-rank", type=int, default=32)
    parser.add_argument(
        "--mobius-sheaf-scale",
        type=float,
        default=0.12,
        help="Scale for the MSF-FAF logits before they are residual/blended/replaced.",
    )
    parser.add_argument(
        "--mobius-sheaf-mode",
        choices=("residual", "blend", "replace"),
        default="residual",
        help=(
            "How MSF-FAF combines with the ordinary classifier. residual is for "
            "checkpoint-based fail-fast screens; blend/replace are for joint "
            "training from ImageNet or an early checkpoint."
        ),
    )
    parser.add_argument(
        "--mobius-sheaf-blend",
        type=float,
        default=0.50,
        help="Blend coefficient when --mobius-sheaf-mode=blend.",
    )
    parser.add_argument("--mobius-sheaf-gate-hidden-dim", type=int, default=96)
    parser.add_argument("--mobius-sheaf-gate-temperature", type=float, default=1.0)
    parser.add_argument(
        "--mobius-sheaf-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cosine-normalized projections and factor prototypes inside MSF-FAF.",
    )
    parser.add_argument(
        "--mobius-sheaf-zero-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize MSF-FAF effect embeddings to zero so residual mode starts as the loaded baseline.",
    )
    parser.add_argument(
        "--mobius-sheaf-use-triple",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the dry/wet/water x asphalt/concrete x smooth/slight/severe triple interaction.",
    )
    parser.add_argument(
        "--use-mechanism-conditional-sheaf-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add HMC-Sheaf++: a mechanism-routed residual over RSCD factors and "
            "heterophilic label edges. It uses physics evidence for dry roughness, "
            "wet/water film obstruction, granular texture, and winter phase cues."
        ),
    )
    parser.add_argument("--mechanism-sheaf-rank", type=int, default=24)
    parser.add_argument(
        "--mechanism-sheaf-scale",
        type=float,
        default=0.06,
        help="Scale for the mechanism-conditioned Moebius factor residual.",
    )
    parser.add_argument(
        "--mechanism-sheaf-edge-scale",
        type=float,
        default=0.04,
        help="Scale for the signed heterophilic edge-flow residual.",
    )
    parser.add_argument("--mechanism-sheaf-router-hidden-dim", type=int, default=96)
    parser.add_argument("--mechanism-sheaf-edge-dim", type=int, default=12)
    parser.add_argument("--mechanism-sheaf-edge-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--mechanism-sheaf-class-scope",
        choices=("all", "asphalt_core", "wet_water_asphalt", "concrete_core"),
        default="all",
        help=(
            "Restrict HMC-Sheaf++ residuals to a validated mechanism subgraph. "
            "Use wet_water_asphalt when broad HMC helps asphalt wet/water but "
            "harms concrete roughness."
        ),
    )
    parser.add_argument(
        "--mechanism-sheaf-use-edge-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable signed edge-flow correction on single-factor RSCD hard-neighbor edges.",
    )
    parser.add_argument(
        "--mechanism-sheaf-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative HMC-Sheaf++ residual allowed for protected wet/water classes.",
    )
    parser.add_argument(
        "--mechanism-sheaf-sparse-router-topk",
        type=int,
        default=0,
        help=(
            "If positive, keep only the top-k mechanism routes per sample inside "
            "HMC-Sheaf++. This tests a sparse mixture-of-mechanisms version of "
            "the factor sheaf so different RSCD couplings do not all update the "
            "same image."
        ),
    )
    parser.add_argument(
        "--mechanism-sheaf-router-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for HMC-Sheaf++ mechanism routing.",
    )
    parser.add_argument(
        "--mechanism-sheaf-physics-prior-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the PhysicsTexture evidence prior used to seed sparse "
            "HMC-Sheaf++ routing before the learned router separates modes."
        ),
    )
    parser.add_argument(
        "--use-local-global-factor-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add LG-FAR: a local-physics-conditioned factor attention residual "
            "that dynamically weighs decoupled RSCD factors and pairwise coupled "
            "factor interactions."
        ),
    )
    parser.add_argument("--local-global-factor-rank", type=int, default=48)
    parser.add_argument(
        "--local-global-factor-scale",
        type=float,
        default=0.08,
        help="Maximum scale for the local-global factor attention residual.",
    )
    parser.add_argument(
        "--local-global-factor-gate-threshold",
        type=float,
        default=0.35,
        help="Base uncertainty threshold for activating LG-FAR.",
    )
    parser.add_argument(
        "--local-global-factor-gate-temperature",
        type=float,
        default=10.0,
        help="Sharpness of the LG-FAR uncertainty gate.",
    )
    parser.add_argument(
        "--local-global-factor-neighbor-gate-floor",
        type=float,
        default=0.15,
        help="Residual gate multiplier when top-2 predictions are not coupled factor neighbors.",
    )
    parser.add_argument(
        "--local-global-factor-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative LG-FAR residual allowed for protected wet/water classes.",
    )
    parser.add_argument(
        "--use-label-graph-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a lightweight GCN-style label-graph prototype residual. RSCD "
            "classes are nodes connected by friction/material/roughness adjacency; "
            "the residual is uncertainty gated and keeps the original 27-class protocol."
        ),
    )
    parser.add_argument("--label-graph-rank", type=int, default=32)
    parser.add_argument("--label-graph-scale", type=float, default=0.04)
    parser.add_argument("--label-graph-gate-threshold", type=float, default=0.45)
    parser.add_argument("--label-graph-gate-temperature", type=float, default=10.0)
    parser.add_argument("--label-graph-neighbor-gate-floor", type=float, default=0.10)
    parser.add_argument(
        "--use-conditional-evidence-masked-coupling-field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add CEM-CF: physics soft masks pool the ConvNeXt feature map into "
            "evidence tokens, then friction-material, friction-roughness, and "
            "material-roughness queries produce a bounded heterophilic residual. "
            "This is a mask-based alternative to random Cutout consistency."
        ),
    )
    parser.add_argument("--evidence-masked-coupling-feature-map-dim", type=int, default=768)
    parser.add_argument("--evidence-masked-coupling-token-dim", type=int, default=96)
    parser.add_argument("--evidence-masked-coupling-rank", type=int, default=32)
    parser.add_argument("--evidence-masked-coupling-scale", type=float, default=0.04)
    parser.add_argument("--evidence-masked-coupling-gate-threshold", type=float, default=0.35)
    parser.add_argument("--evidence-masked-coupling-gate-temperature", type=float, default=10.0)
    parser.add_argument("--evidence-masked-coupling-neighbor-gate-floor", type=float, default=0.05)
    parser.add_argument("--evidence-masked-coupling-protected-negative-limit", type=float, default=0.0)
    parser.add_argument(
        "--use-full-order-coupling-tensor-field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add FOCT: a full-order ANOVA tensor field for RSCD core labels. "
            "It models A_f+B_m+C_r+D_fm+E_fr+G_mr+H_fmr with an unconstrained "
            "zero-marginal third-order term H_fmr from class-specific physics "
            "masked ConvNeXt tokens."
        ),
    )
    parser.add_argument("--full-order-coupling-feature-map-dim", type=int, default=768)
    parser.add_argument("--full-order-coupling-token-dim", type=int, default=96)
    parser.add_argument("--full-order-coupling-hidden-dim", type=int, default=96)
    parser.add_argument("--full-order-coupling-scale", type=float, default=0.05)
    parser.add_argument("--full-order-coupling-gate-threshold", type=float, default=0.35)
    parser.add_argument("--full-order-coupling-gate-temperature", type=float, default=10.0)
    parser.add_argument("--full-order-coupling-core-gate-floor", type=float, default=0.05)
    parser.add_argument("--full-order-coupling-protected-negative-limit", type=float, default=0.0)
    parser.add_argument(
        "--use-mechanism-charted-full-order-coupling-tensor-field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add MC-FOCT: a mechanism-charted full-order tensor field. It keeps "
            "A_f+B_m+C_r+D_fm+E_fr+G_mr, but splits H_fmr into dry-visible, "
            "wet-film, water-obstruction, concrete-hidden-roughness, and "
            "asphalt-sheen charts with different PhysicsTexture mask pooling."
        ),
    )
    parser.add_argument("--mechanism-charted-full-order-feature-map-dim", type=int, default=768)
    parser.add_argument("--mechanism-charted-full-order-token-dim", type=int, default=96)
    parser.add_argument("--mechanism-charted-full-order-hidden-dim", type=int, default=96)
    parser.add_argument("--mechanism-charted-full-order-scale", type=float, default=0.04)
    parser.add_argument("--mechanism-charted-full-order-gate-threshold", type=float, default=0.35)
    parser.add_argument("--mechanism-charted-full-order-gate-temperature", type=float, default=10.0)
    parser.add_argument("--mechanism-charted-full-order-core-gate-floor", type=float, default=0.05)
    parser.add_argument("--mechanism-charted-full-order-protected-negative-limit", type=float, default=0.0)
    parser.add_argument("--mechanism-charted-full-order-router-hidden-dim", type=int, default=96)
    parser.add_argument("--mechanism-charted-full-order-router-temperature", type=float, default=1.0)
    parser.add_argument("--mechanism-charted-full-order-sparse-router-topk", type=int, default=2)
    parser.add_argument("--mechanism-charted-full-order-physics-prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--use-core-factor-coupled-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a core-only factor-coupled residual for the 18 hard RSCD "
            "dry/wet/water asphalt/concrete smooth/slight/severe cells."
        ),
    )
    parser.add_argument("--core-factor-rank", type=int, default=32)
    parser.add_argument(
        "--core-factor-scale",
        type=float,
        default=0.08,
        help="Maximum scale for the core-only factor-coupled residual.",
    )
    parser.add_argument(
        "--core-factor-neighbor-gate-floor",
        type=float,
        default=0.05,
        help="Residual gate multiplier when top-2 predictions are not neighboring core cells.",
    )
    parser.add_argument(
        "--core-factor-uncertainty-threshold",
        type=float,
        default=0.40,
        help="Base uncertainty threshold for activating the core-only factor residual.",
    )
    parser.add_argument(
        "--core-factor-uncertainty-temperature",
        type=float,
        default=10.0,
        help="Sharpness of the uncertainty gate for the core-only factor residual.",
    )
    parser.add_argument(
        "--core-factor-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative core residual allowed for wet/water classes.",
    )
    parser.add_argument(
        "--use-water-evidence-logit-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a patch-compatible optical water-film evidence gate. It computes "
            "specular, dark-smooth water, low-texture, and contrast-erasure cues "
            "and maps them to wet/water class-logit residuals."
        ),
    )
    parser.add_argument(
        "--water-evidence-gate-scale",
        type=float,
        default=0.20,
        help="Maximum tanh-scaled wet/water logit residual from the water evidence gate.",
    )
    parser.add_argument(
        "--water-evidence-gate-zero-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the final evidence gate projection to zero so the model starts as the ungated baseline.",
    )
    parser.add_argument(
        "--use-dry-concrete-roughness-vor-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add the first HMC-Sheaf mechanism chart: a zero-initialized VOR "
            "residual that only redistributes logits among dry_concrete_smooth, "
            "dry_concrete_slight, and dry_concrete_severe. It is activated by "
            "dry-concrete probability mass to avoid wet/water collateral damage."
        ),
    )
    parser.add_argument("--dry-concrete-roughness-hidden-dim", type=int, default=48)
    parser.add_argument(
        "--dry-concrete-roughness-scale",
        type=float,
        default=0.05,
        help="Maximum zero-sum logit correction for the dry concrete roughness VOR chart.",
    )
    parser.add_argument(
        "--dry-concrete-roughness-gate-threshold",
        type=float,
        default=0.12,
        help="Dry-concrete probability mass threshold for activating the VOR chart.",
    )
    parser.add_argument(
        "--dry-concrete-roughness-gate-temperature",
        type=float,
        default=14.0,
        help="Sharpness of the dry-concrete probability-mass gate.",
    )
    parser.add_argument(
        "--use-dry-paved-roughness-vor-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a task-adapted dry paved roughness VOR chart. It separately "
            "redistributes logits inside dry_asphalt_{smooth,slight,severe} "
            "and dry_concrete_{smooth,slight,severe}, with zero-sum corrections "
            "per material so wet/water and material totals are protected."
        ),
    )
    parser.add_argument("--dry-paved-roughness-hidden-dim", type=int, default=48)
    parser.add_argument(
        "--dry-paved-roughness-material-dim",
        type=int,
        default=6,
        help="Learned material-token dimension for asphalt/concrete roughness charts.",
    )
    parser.add_argument(
        "--dry-paved-roughness-head-mode",
        choices=["shared", "nonshared"],
        default="shared",
        help=(
            "Use one shared material-conditioned chart head or independent "
            "heads per dry paved material. Nonshared mode tests the hypothesis "
            "that asphalt and concrete roughness have different visual coupling forms."
        ),
    )
    parser.add_argument(
        "--dry-paved-roughness-scale",
        type=float,
        default=0.05,
        help="Maximum zero-sum logit correction for each dry paved roughness chart.",
    )
    parser.add_argument(
        "--dry-paved-roughness-gate-threshold",
        type=float,
        default=0.12,
        help="Per-material dry probability mass threshold for activating the dry paved VOR chart.",
    )
    parser.add_argument(
        "--dry-paved-roughness-gate-temperature",
        type=float,
        default=14.0,
        help="Sharpness of the dry paved probability-mass gate.",
    )
    parser.add_argument(
        "--dry-paved-roughness-material-gate-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional safety gate on each material trio's share of dry paved "
            "probability mass. Values above 0 enable material-consistency "
            "protection before roughness redistribution."
        ),
    )
    parser.add_argument(
        "--dry-paved-roughness-material-gate-temperature",
        type=float,
        default=16.0,
        help="Sharpness of the dry paved material-consistency gate.",
    )
    parser.add_argument(
        "--use-concrete-roughness-vor-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a friction-charted concrete roughness VOR residual. It applies "
            "separate zero-sum smooth/slight/severe corrections inside dry, wet, "
            "and water concrete trios without changing total concrete-friction mass."
        ),
    )
    parser.add_argument("--concrete-roughness-hidden-dim", type=int, default=48)
    parser.add_argument("--concrete-roughness-chart-dim", type=int, default=6)
    parser.add_argument(
        "--concrete-roughness-scale",
        type=float,
        default=0.05,
        help="Maximum zero-sum logit correction for each concrete roughness chart.",
    )
    parser.add_argument(
        "--concrete-roughness-gate-threshold",
        type=float,
        default=0.12,
        help="Concrete-trio probability mass threshold for activating each chart.",
    )
    parser.add_argument(
        "--concrete-roughness-gate-temperature",
        type=float,
        default=14.0,
        help="Sharpness of the concrete-trio probability-mass gate.",
    )
    parser.add_argument(
        "--use-wet-water-film-vor-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add the HMC-Sheaf wet/water film chart. It only applies pairwise "
            "antisymmetric residuals between matched wet_* and water_* paved "
            "classes with the same material and roughness."
        ),
    )
    parser.add_argument("--wet-water-film-hidden-dim", type=int, default=48)
    parser.add_argument("--wet-water-film-pair-dim", type=int, default=8)
    parser.add_argument(
        "--wet-water-film-scale",
        type=float,
        default=0.05,
        help="Maximum wet/water pairwise logit transfer from the film VOR chart.",
    )
    parser.add_argument(
        "--wet-water-film-material-scope",
        choices=("all", "asphalt", "concrete"),
        default="all",
        help="Restrict the wet/water film chart to asphalt pairs, concrete pairs, or all paved pairs.",
    )
    parser.add_argument(
        "--wet-water-film-gate-threshold",
        type=float,
        default=0.12,
        help="Wet/water pair probability mass threshold for activating the film chart.",
    )
    parser.add_argument(
        "--wet-water-film-gate-temperature",
        type=float,
        default=14.0,
        help="Sharpness of the wet/water pair probability-mass gate.",
    )
    parser.add_argument(
        "--use-smooth-film-concrete-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a protected smooth-film concrete expert. It only writes to "
            "wet_concrete_smooth and water_concrete_smooth, decomposing the "
            "residual into common smooth-film mass and wet/water transfer while "
            "leaving wet/water concrete slight/severe logits unchanged."
        ),
    )
    parser.add_argument("--smooth-film-concrete-hidden-dim", type=int, default=48)
    parser.add_argument(
        "--smooth-film-concrete-scale",
        type=float,
        default=0.05,
        help="Maximum protected logit correction for the smooth wet/water concrete film expert.",
    )
    parser.add_argument(
        "--smooth-film-concrete-gate-threshold",
        type=float,
        default=0.05,
        help="Smooth wet/water concrete pair probability mass threshold for activating the expert.",
    )
    parser.add_argument(
        "--smooth-film-concrete-gate-temperature",
        type=float,
        default=14.0,
        help="Sharpness of the smooth-film concrete probability-mass gate.",
    )
    parser.add_argument(
        "--use-obstruction-concrete-roughness-vor-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add an obstruction-aware concrete roughness chart for wet_concrete_* "
            "and water_concrete_* trios. It uses visible roughness, hidden "
            "roughness, film obstruction, dark-water, and concrete-likeness "
            "fields, then applies zero-sum corrections only inside each trio."
        ),
    )
    parser.add_argument("--obstruction-concrete-roughness-hidden-dim", type=int, default=48)
    parser.add_argument(
        "--obstruction-concrete-roughness-scale",
        type=float,
        default=0.05,
        help="Maximum zero-sum logit correction for each wet/water concrete roughness chart.",
    )
    parser.add_argument(
        "--obstruction-concrete-roughness-gate-threshold",
        type=float,
        default=0.12,
        help="Wet/water concrete trio probability mass threshold for activating each chart.",
    )
    parser.add_argument(
        "--obstruction-concrete-roughness-gate-temperature",
        type=float,
        default=14.0,
        help="Sharpness of the obstruction-aware concrete roughness probability-mass gate.",
    )
    parser.add_argument(
        "--obstruction-concrete-roughness-share-gate-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional gate on a wet or water concrete trio's share of total "
            "wet+water concrete probability mass. Values above 0 enable the gate."
        ),
    )
    parser.add_argument(
        "--obstruction-concrete-roughness-share-gate-temperature",
        type=float,
        default=16.0,
        help="Sharpness of the wet/water concrete trio share gate.",
    )
    parser.add_argument(
        "--use-coupled-optical-roughness-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add CD-OWR: a zero-initialized uncertainty-gated residual for "
            "hard coupled dry/wet/water asphalt/concrete roughness cells. "
            "It uses optical water-film and roughness statistics plus the "
            "current classifier feature."
        ),
    )
    parser.add_argument("--coupled-residual-hidden-dim", type=int, default=96)
    parser.add_argument(
        "--coupled-residual-scale",
        type=float,
        default=0.12,
        help="Maximum tanh-scaled class-logit residual for CD-OWR.",
    )
    parser.add_argument(
        "--coupled-residual-gate-threshold",
        type=float,
        default=0.35,
        help="Base-model uncertainty threshold for activating CD-OWR.",
    )
    parser.add_argument(
        "--coupled-residual-gate-temperature",
        type=float,
        default=8.0,
        help="Sharpness of the CD-OWR uncertainty gate.",
    )
    parser.add_argument(
        "--coupled-residual-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative CD-OWR residual allowed for wet/water classes.",
    )
    parser.add_argument(
        "--use-roughness-neighbor-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized residual specialized for smooth/slight/severe "
            "neighbor confusions in dry/wet/water asphalt/concrete cells."
        ),
    )
    parser.add_argument("--roughness-neighbor-hidden-dim", type=int, default=96)
    parser.add_argument(
        "--roughness-neighbor-scale",
        type=float,
        default=0.10,
        help="Maximum tanh-scaled logit residual for the roughness-neighbor correction.",
    )
    parser.add_argument(
        "--roughness-neighbor-gate-threshold",
        type=float,
        default=0.42,
        help="Base-model uncertainty threshold for activating the roughness-neighbor residual.",
    )
    parser.add_argument(
        "--roughness-neighbor-gate-temperature",
        type=float,
        default=10.0,
        help="Sharpness of the roughness-neighbor uncertainty gate.",
    )
    parser.add_argument(
        "--roughness-neighbor-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative roughness-neighbor residual allowed for wet/water classes.",
    )
    parser.add_argument(
        "--roughness-neighbor-gate-floor",
        type=float,
        default=0.15,
        help="Residual gate multiplier for non-top2 roughness-neighbor cases.",
    )
    parser.add_argument(
        "--use-spectral-roughness-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized residual that uses multi-scale band-pass "
            "texture energy to separate smooth/slight/severe roughness neighbors."
        ),
    )
    parser.add_argument("--spectral-roughness-hidden-dim", type=int, default=96)
    parser.add_argument(
        "--spectral-roughness-scale",
        type=float,
        default=0.08,
        help="Maximum tanh-scaled logit residual for spectral roughness correction.",
    )
    parser.add_argument(
        "--spectral-roughness-gate-threshold",
        type=float,
        default=0.35,
        help="Base uncertainty threshold for activating the spectral roughness residual.",
    )
    parser.add_argument(
        "--spectral-roughness-gate-temperature",
        type=float,
        default=12.0,
        help="Sharpness of the spectral roughness uncertainty gate.",
    )
    parser.add_argument(
        "--spectral-roughness-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative spectral residual allowed for wet/water classes.",
    )
    parser.add_argument(
        "--spectral-roughness-neighbor-gate-floor",
        type=float,
        default=0.02,
        help="Residual gate multiplier for non-top2 roughness-neighbor cases.",
    )
    parser.add_argument(
        "--use-relation-signed-graph-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized heterophilic RSCD label-graph expert. It only "
            "activates when top-2 predictions form a hard graph edge touching an "
            "audited weak class, and uses relation-specific offsets instead of "
            "undirected graph smoothing."
        ),
    )
    parser.add_argument("--relation-signed-hidden-dim", type=int, default=96)
    parser.add_argument(
        "--relation-signed-scale",
        type=float,
        default=0.06,
        help="Maximum tanh-scaled logit residual for the relation-signed graph expert.",
    )
    parser.add_argument(
        "--relation-signed-gate-threshold",
        type=float,
        default=0.35,
        help="Base-model uncertainty threshold for activating the relation-signed expert.",
    )
    parser.add_argument(
        "--relation-signed-gate-temperature",
        type=float,
        default=12.0,
        help="Sharpness of the relation-signed uncertainty gate.",
    )
    parser.add_argument(
        "--relation-signed-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for protected wet/water classes.",
    )
    parser.add_argument(
        "--relation-signed-neighbor-gate-floor",
        type=float,
        default=0.0,
        help="Residual gate multiplier when top-2 is not an audited hard graph edge.",
    )
    parser.add_argument(
        "--use-heterophilic-logit-boundary-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a lightweight antisymmetric top-2 hard-pair boundary expert. "
            "It is trained on validation-audited RSCD heterophilic confusion pairs "
            "and applies a small logit residual only when the current top-2 labels "
            "match one of those pairs."
        ),
    )
    parser.add_argument(
        "--heterophilic-boundary-scale",
        type=float,
        default=0.35,
        help="Maximum tanh-scaled logit residual for the heterophilic boundary expert.",
    )
    parser.add_argument(
        "--heterophilic-boundary-gate-threshold",
        type=float,
        default=0.0,
        help="Uncertainty threshold for activating the heterophilic boundary expert.",
    )
    parser.add_argument(
        "--heterophilic-boundary-gate-temperature",
        type=float,
        default=8.0,
        help="Sharpness of the heterophilic boundary uncertainty gate.",
    )
    parser.add_argument(
        "--heterophilic-boundary-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for protected wet/water/severe classes.",
    )
    parser.add_argument(
        "--use-heterophilic-feature-boundary-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a feature-conditioned hard-pair boundary expert. Unlike the "
            "logits-only expert, this module uses the fused classifier feature "
            "that already contains PhysicsTexture evidence."
        ),
    )
    parser.add_argument("--heterophilic-feature-boundary-hidden-dim", type=int, default=96)
    parser.add_argument("--heterophilic-feature-boundary-pair-dim", type=int, default=16)
    parser.add_argument(
        "--heterophilic-feature-boundary-scale",
        type=float,
        default=0.08,
        help="Maximum tanh-scaled logit residual for the feature-conditioned boundary expert.",
    )
    parser.add_argument(
        "--heterophilic-feature-boundary-gate-threshold",
        type=float,
        default=0.10,
        help="Uncertainty threshold for activating the feature-conditioned boundary expert.",
    )
    parser.add_argument(
        "--heterophilic-feature-boundary-gate-temperature",
        type=float,
        default=10.0,
        help="Sharpness of the feature-conditioned boundary uncertainty gate.",
    )
    parser.add_argument(
        "--heterophilic-feature-boundary-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for protected wet/water/severe classes.",
    )
    parser.add_argument(
        "--use-heterophilic-physics-boundary-expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a low-dimensional physical-evidence hard-pair boundary expert. "
            "It uses optical and texture statistics instead of high-dimensional semantic features."
        ),
    )
    parser.add_argument("--heterophilic-physics-boundary-hidden-dim", type=int, default=64)
    parser.add_argument("--heterophilic-physics-boundary-pair-dim", type=int, default=12)
    parser.add_argument(
        "--heterophilic-physics-boundary-scale",
        type=float,
        default=0.08,
        help="Maximum tanh-scaled logit residual for the physical-evidence boundary expert.",
    )
    parser.add_argument(
        "--heterophilic-physics-boundary-gate-threshold",
        type=float,
        default=0.10,
        help="Uncertainty threshold for activating the physical-evidence boundary expert.",
    )
    parser.add_argument(
        "--heterophilic-physics-boundary-gate-temperature",
        type=float,
        default=10.0,
        help="Sharpness of the physical-evidence boundary uncertainty gate.",
    )
    parser.add_argument(
        "--heterophilic-physics-boundary-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for protected wet/water/severe classes.",
    )
    parser.add_argument(
        "--use-protected-heterophilic-factor-boundary-field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add PH-FBF: a relation-conditioned physical hard-pair boundary field. "
            "It activates only on top-k heterophilic RSCD factor-neighbor pairs "
            "and uses roughness, wet/water, or material relation embeddings."
        ),
    )
    parser.add_argument("--protected-factor-boundary-hidden-dim", type=int, default=64)
    parser.add_argument("--protected-factor-boundary-pair-dim", type=int, default=12)
    parser.add_argument("--protected-factor-boundary-relation-dim", type=int, default=8)
    parser.add_argument(
        "--protected-factor-boundary-scale",
        type=float,
        default=0.08,
        help="Maximum tanh-scaled logit residual for PH-FBF.",
    )
    parser.add_argument(
        "--protected-factor-boundary-gate-threshold",
        type=float,
        default=0.10,
        help="Uncertainty threshold for activating PH-FBF.",
    )
    parser.add_argument(
        "--protected-factor-boundary-gate-temperature",
        type=float,
        default=10.0,
        help="Sharpness of the PH-FBF uncertainty gate.",
    )
    parser.add_argument(
        "--protected-factor-boundary-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for fragile wet/water/severe classes in PH-FBF.",
    )
    parser.add_argument(
        "--use-relation-specific-hard-edge-refiner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a relation-routed hard-edge refiner. It uses separate heads for "
            "wet/water friction boundaries, roughness boundaries, and mud/gravel "
            "material boundaries, with compact physics/topology evidence."
        ),
    )
    parser.add_argument("--relation-specific-refiner-hidden-dim", type=int, default=64)
    parser.add_argument("--relation-specific-refiner-pair-dim", type=int, default=12)
    parser.add_argument(
        "--relation-specific-refiner-scale",
        type=float,
        default=0.08,
        help="Maximum tanh-scaled logit residual for the relation-specific hard-edge refiner.",
    )
    parser.add_argument(
        "--relation-specific-refiner-gate-threshold",
        type=float,
        default=0.10,
        help="Uncertainty threshold for activating the relation-specific hard-edge refiner.",
    )
    parser.add_argument(
        "--relation-specific-refiner-gate-temperature",
        type=float,
        default=10.0,
        help="Sharpness of the relation-specific hard-edge uncertainty gate.",
    )
    parser.add_argument(
        "--relation-specific-refiner-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for fragile wet/water/severe classes in the relation-specific refiner.",
    )
    parser.add_argument(
        "--use-selective-mechanism-tensor-boundary-field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add SM-TBF: a mechanism-charted high-order coupling boundary field. "
            "It routes top-2 RSCD factor-neighbor confusions to wet-film, water-"
            "obstruction, dry-microtexture, granular, winter, or paved-material "
            "experts instead of using one global low-rank H_fmr term."
        ),
    )
    parser.add_argument("--selective-mechanism-tensor-boundary-hidden-dim", type=int, default=64)
    parser.add_argument("--selective-mechanism-tensor-boundary-pair-dim", type=int, default=12)
    parser.add_argument("--selective-mechanism-tensor-boundary-relation-dim", type=int, default=6)
    parser.add_argument("--selective-mechanism-tensor-boundary-mechanism-dim", type=int, default=8)
    parser.add_argument(
        "--selective-mechanism-tensor-boundary-scale",
        type=float,
        default=0.08,
        help="Maximum tanh-scaled logit residual for SM-TBF.",
    )
    parser.add_argument(
        "--selective-mechanism-tensor-boundary-gate-threshold",
        type=float,
        default=0.10,
        help="Top-1 uncertainty threshold for activating SM-TBF.",
    )
    parser.add_argument(
        "--selective-mechanism-tensor-boundary-gate-temperature",
        type=float,
        default=10.0,
        help="Sharpness of the SM-TBF uncertainty gate.",
    )
    parser.add_argument(
        "--selective-mechanism-tensor-boundary-mechanism-gate-threshold",
        type=float,
        default=0.08,
        help="Minimum selected physical mechanism evidence required by SM-TBF.",
    )
    parser.add_argument(
        "--selective-mechanism-tensor-boundary-mechanism-gate-temperature",
        type=float,
        default=12.0,
        help="Sharpness of the SM-TBF mechanism-evidence gate.",
    )
    parser.add_argument(
        "--selective-mechanism-tensor-boundary-enabled-mechanisms",
        type=str,
        default="all",
        help=(
            "Comma-separated SM-TBF mechanisms to activate. Use all by default, "
            "or names such as asphalt_wet_sheen,concrete_thin_film to test a "
            "mechanism-selected high-order field."
        ),
    )
    parser.add_argument(
        "--selective-mechanism-tensor-boundary-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative residual allowed for fragile wet/water/severe classes in SM-TBF.",
    )
    parser.add_argument(
        "--use-conditional-factor-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add CFCP: a parameter-free conditional factor consistency projection. "
            "It recalibrates 27-class logits through P(friction|material,unevenness), "
            "P(material|friction,unevenness), and P(unevenness|friction,material) "
            "without changing the RSCD label space."
        ),
    )
    parser.add_argument(
        "--conditional-factor-projection-scale",
        type=float,
        default=0.04,
        help="Scale for the parameter-free conditional factor consistency residual.",
    )
    parser.add_argument("--conditional-factor-projection-gate-threshold", type=float, default=0.35)
    parser.add_argument("--conditional-factor-projection-gate-temperature", type=float, default=10.0)
    parser.add_argument(
        "--conditional-factor-projection-focus",
        choices=("all", "core", "hard"),
        default="core",
        help=(
            "Apply CFCP to all valid RSCD classes, dry/wet/water asphalt-concrete "
            "roughness cells, or only audited hard classes."
        ),
    )
    parser.add_argument("--conditional-factor-projection-friction-weight", type=float, default=1.0)
    parser.add_argument("--conditional-factor-projection-material-weight", type=float, default=0.6)
    parser.add_argument("--conditional-factor-projection-unevenness-weight", type=float, default=1.2)
    parser.add_argument(
        "--conditional-factor-projection-protected-negative-limit",
        type=float,
        default=0.0,
        help="Maximum negative CFCP residual allowed for protected wet/water classes.",
    )
    parser.add_argument(
        "--use-heterogeneous-label-router",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add HLR: a zero-initialized residual classifier that respects RSCD's "
            "heterogeneous label topology: paved 18-cell labels, loose mud/gravel "
            "6-cell labels, and snow/ice weather labels."
        ),
    )
    parser.add_argument("--heterogeneous-router-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--heterogeneous-router-scale",
        type=float,
        default=0.08,
        help="Scale for the heterogeneous label router residual logits.",
    )
    parser.add_argument(
        "--distill-teacher-probs",
        type=Path,
        default=None,
        help="Optional NPZ with image_path and probs arrays for soft-label calibration distillation.",
    )
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--distill-factor-weight",
        type=float,
        default=0.0,
        help=(
            "Teacher-probability distillation after factor-wise marginalization. "
            "The teacher 27-class distribution is projected to friction/material/"
            "unevenness factors before KL matching, which tests decoupled RSCD "
            "teacher transfer without changing inference."
        ),
    )
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--positive-congruent-weight",
        type=float,
        default=0.0,
        help=(
            "Extra teacher-preservation loss on samples that the teacher already "
            "classifies correctly. This targets negative flips: old model correct, "
            "new model wrong."
        ),
    )
    parser.add_argument(
        "--positive-congruent-beta",
        type=float,
        default=4.0,
        help="Confidence-dependent focal multiplier for positive-congruent distillation.",
    )
    parser.add_argument(
        "--positive-congruent-min-confidence",
        type=float,
        default=0.0,
        help="Only preserve teacher-correct samples whose teacher confidence exceeds this threshold.",
    )
    parser.add_argument(
        "--distill-missing-policy",
        choices=["error", "skip"],
        default="error",
        help=(
            "What to do when a batch image has no teacher probability row. "
            "`error` is safest for subset-matched fast screens; `skip` allows "
            "full-data formal training with partial teacher coverage."
        ),
    )
    parser.add_argument(
        "--online-teacher-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint for an online no-harm teacher. During training "
            "the frozen teacher protects samples it classifies correctly, avoiding "
            "negative flips while curriculum or hard-class losses move decision boundaries."
        ),
    )
    parser.add_argument(
        "--online-teacher-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only no-harm distillation weight from a frozen online teacher. "
            "Use with --online-teacher-checkpoint for protected mechanism-curriculum screens."
        ),
    )
    parser.add_argument("--online-teacher-temperature", type=float, default=2.0)
    parser.add_argument("--online-teacher-beta", type=float, default=4.0)
    parser.add_argument(
        "--online-teacher-min-confidence",
        type=float,
        default=0.0,
        help="Only preserve teacher-correct samples whose max probability is at least this value.",
    )
    parser.add_argument(
        "--teacher-error-replay-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only replay loss on samples that the frozen online teacher "
            "misclassifies inside audited RSCD hard cells. This targets the "
            "Top-1/Mean-F1 gap without class-wide natural-frequency weighting."
        ),
    )
    parser.add_argument(
        "--teacher-error-replay-focus",
        choices=("top1_gap_v1", "majority_smooth_v1", "concrete_wetwater_rough_v1", "all"),
        default="top1_gap_v1",
        help="Which RSCD class subset receives teacher-error replay.",
    )
    parser.add_argument(
        "--teacher-error-replay-beta",
        type=float,
        default=1.0,
        help="Teacher-confidence exponent for replayed teacher-error samples.",
    )
    parser.add_argument(
        "--teacher-error-replay-min-confidence",
        type=float,
        default=0.0,
        help="Only replay teacher-wrong samples whose frozen-teacher confidence is at least this value.",
    )
    parser.add_argument(
        "--class-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the standard 27-class RSCD cross-entropy. Keep the default "
            "1.0 for normal training. Use values below 1.0 only in staged "
            "factor-first pretraining, where friction/material/roughness losses "
            "temporarily shape the representation before full 27-class fine-tuning."
        ),
    )
    parser.add_argument("--factor-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--hard-pair-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only HCF hard-boundary auxiliary loss. Each audited RSCD "
            "hard pair receives a binary classifier on the shared feature, so the "
            "representation must learn relation-specific separability while the "
            "test-time 27-class head remains unchanged."
        ),
    )
    parser.add_argument(
        "--hard-pair-aux-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension of the hard-pair auxiliary boundary head.",
    )
    parser.add_argument(
        "--hard-pair-aux-focus",
        choices=("all", "concrete", "wet_water", "roughness", "material"),
        default="all",
        help=(
            "Which audited hard-pair boundaries are supervised by HCF. roughness "
            "keeps same-friction/material smooth-slight-severe edges; wet_water "
            "keeps water-vs-wet edges; concrete keeps pairs involving concrete."
        ),
    )
    parser.add_argument(
        "--mechanism-orthogonal-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only MOCA loss weight. MOCA factorizes the fused feature "
            "into friction/material/roughness/coupling subspaces and decorrelates "
            "them so water-film, material, and roughness evidence do not collapse "
            "into one coupled shortcut. Evaluation logits are unchanged."
        ),
    )
    parser.add_argument("--mechanism-orthogonal-dim", type=int, default=64)
    parser.add_argument("--mechanism-orthogonal-hidden-dim", type=int, default=128)
    parser.add_argument("--mechanism-orthogonal-factor-weight", type=float, default=1.0)
    parser.add_argument("--mechanism-orthogonal-mechanism-weight", type=float, default=0.35)
    parser.add_argument("--mechanism-orthogonal-cov-weight", type=float, default=0.08)
    parser.add_argument(
        "--hflip-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Symmetric KL consistency weight between the original image and a "
            "horizontal flip. This tries to internalize the large hflip-TTA gain "
            "into a single model without changing the RSCD test protocol."
        ),
    )
    parser.add_argument(
        "--hflip-consistency-temperature",
        type=float,
        default=2.0,
        help="Temperature for horizontal-flip consistency KL.",
    )
    parser.add_argument(
        "--hflip-physics-feature-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only consistency weight for PhysicsTexture evidence features "
            "under horizontal flip. This regularizes physical evidence rather "
            "than only final logits and adds no inference-time module."
        ),
    )
    parser.add_argument(
        "--hflip-local-physics-feature-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only consistency weight for LocalPhysicsField features under "
            "horizontal flip, targeting patch-local roughness/water evidence."
        ),
    )
    parser.add_argument(
        "--hflip-low-level-feature-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only consistency weight for the concatenated low-level "
            "physics/semantic evidence vector under horizontal flip."
        ),
    )
    parser.add_argument(
        "--masked-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Class-distribution consistency weight between a clean RSCD patch and "
            "a randomly masked view. This is a weak/self-supervised regularizer "
            "that discourages reliance on isolated markings, shadows, or accidental "
            "local cues while preserving the 27-class test protocol."
        ),
    )
    parser.add_argument(
        "--masked-factor-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Factor-marginal consistency weight for the masked view. The clean "
            "teacher distribution and masked student distribution are both "
            "projected to friction/material/unevenness marginals before KL matching."
        ),
    )
    parser.add_argument("--masked-consistency-temperature", type=float, default=2.0)
    parser.add_argument(
        "--masked-consistency-mode",
        choices=("random", "physics_protected"),
        default="random",
        help=(
            "Masked-view construction. The physics_protected mode preserves "
            "PhysicsTexture-style water-film, roughness, granular, and snow/ice "
            "evidence while still masking line-like nuisance inside random blocks."
        ),
    )
    parser.add_argument("--masked-consistency-ratio", type=float, default=0.18)
    parser.add_argument("--masked-consistency-block-frac", type=float, default=0.22)
    parser.add_argument("--masked-consistency-max-blocks", type=int, default=2)
    parser.add_argument(
        "--masked-consistency-value",
        choices=("mean", "zero", "random"),
        default="mean",
        help="Fill value for normalized random evidence masks.",
    )
    parser.add_argument(
        "--masked-consistency-confidence-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional clean-view confidence threshold before applying masked "
            "consistency. Keep at 0 for dense regularization; raise it if masked "
            "views over-regularize ambiguous wet/water boundaries."
        ),
    )
    parser.add_argument(
        "--observer-hinf-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only Observer-Hinf robustness weight. It creates nuisance "
            "disturbances that mimic markings, illumination drift, and local "
            "occlusion, then bounds the gain from image disturbance to RSCD "
            "physical-state features while preserving factor marginals."
        ),
    )
    parser.add_argument(
        "--observer-hinf-mode",
        choices=("mixed", "line", "illumination", "mask"),
        default="mixed",
        help="Nuisance disturbance family for Observer-Hinf training.",
    )
    parser.add_argument(
        "--observer-hinf-scope",
        choices=OBSERVER_HINF_SCOPES,
        default="all",
        help=(
            "Mechanism scope for Observer-Hinf. Use dry_paved_roughness or "
            "non_wet_water to apply disturbance attenuation where nuisance "
            "markings/exposure should be suppressed, while protecting wet/water "
            "film evidence from over-invariance."
        ),
    )
    parser.add_argument("--observer-hinf-strength", type=float, default=1.0)
    parser.add_argument("--observer-hinf-rho", type=float, default=0.20)
    parser.add_argument("--observer-hinf-max-lines", type=int, default=3)
    parser.add_argument("--observer-hinf-block-ratio", type=float, default=0.18)
    parser.add_argument("--observer-hinf-temperature", type=float, default=2.0)
    parser.add_argument("--observer-hinf-confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--observer-hinf-feature-weight",
        type=float,
        default=0.15,
        help="Gain-bound weight for the fused classifier feature.",
    )
    parser.add_argument(
        "--observer-hinf-physics-feature-weight",
        type=float,
        default=0.45,
        help="Gain-bound weight for PhysicsTexture wetness/material evidence.",
    )
    parser.add_argument(
        "--observer-hinf-local-physics-feature-weight",
        type=float,
        default=0.45,
        help="Gain-bound weight for LocalPhysicsField roughness evidence.",
    )
    parser.add_argument(
        "--observer-hinf-low-level-feature-weight",
        type=float,
        default=0.25,
        help="Gain-bound weight for concatenated low-level physical evidence.",
    )
    parser.add_argument(
        "--observer-hinf-class-consistency-weight",
        type=float,
        default=0.05,
        help="Weak 27-class consistency under nuisance disturbance.",
    )
    parser.add_argument(
        "--observer-hinf-factor-consistency-weight",
        type=float,
        default=0.35,
        help="Stronger friction/material/unevenness marginal consistency under nuisance disturbance.",
    )
    parser.add_argument(
        "--observer-hinf-disturbed-ce-weight",
        type=float,
        default=0.10,
        help="Small supervised CE on the disturbed view to keep true class boundaries observable.",
    )
    parser.add_argument(
        "--observer-hinf-barrier-weight",
        type=float,
        default=0.0,
        help=(
            "Control-barrier-style margin preservation weight. This protects "
            "selected mechanisms, typically wet/water film classes, from losing "
            "true-class evidence under the Observer-Hinf disturbance."
        ),
    )
    parser.add_argument(
        "--observer-hinf-barrier-scope",
        choices=OBSERVER_HINF_SCOPES,
        default="wet_water_paved",
        help="Mechanism scope receiving the Observer-Hinf margin barrier.",
    )
    parser.add_argument(
        "--observer-hinf-barrier-margin-drop",
        type=float,
        default=0.05,
        help="Allowed true-class logit-margin drop before the barrier activates.",
    )
    parser.add_argument(
        "--factor-marginal-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary consistency weight on the final 27-class logits. The class "
            "probabilities are marginalized into RSCD friction/material/unevenness "
            "factors and supervised by the corresponding factor labels, so the "
            "main classifier learns physically meaningful group boundaries without "
            "adding a separate test-time head."
        ),
    )
    parser.add_argument(
        "--relation-conditional-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only conditional factor decoupling weight. It computes "
            "P(friction|material,unevenness), P(material|friction,unevenness), "
            "and P(unevenness|friction,material) directly from the 27-class "
            "logits with logsumexp masks, so no extra test-time head is added."
        ),
    )
    parser.add_argument(
        "--relation-conditional-focus",
        choices=("all", "core", "hard"),
        default="core",
        help=(
            "Which samples receive the conditional factor loss: all valid RSCD "
            "classes, dry/wet/water asphalt-concrete core classes, or the audited "
            "hard classes only."
        ),
    )
    parser.add_argument("--relation-conditional-friction-weight", type=float, default=1.0)
    parser.add_argument("--relation-conditional-material-weight", type=float, default=0.6)
    parser.add_argument("--relation-conditional-unevenness-weight", type=float, default=1.2)
    parser.add_argument(
        "--relation-conditional-adaptive-weighting",
        choices=("none", "weak_f1_v1"),
        default="none",
        help=(
            "Optional per-class, per-factor curriculum for relation-conditional "
            "decoupling. weak_f1_v1 reads a previous evaluation JSON and gives "
            "more conditional supervision to audited weak classes without adding "
            "any test-time parameters."
        ),
    )
    parser.add_argument(
        "--relation-conditional-reference-json",
        default=None,
        help=(
            "Evaluation JSON used by --relation-conditional-adaptive-weighting "
            "to estimate per-class difficulty from prior full-test F1 scores."
        ),
    )
    parser.add_argument("--relation-conditional-adaptive-strength", type=float, default=0.5)
    parser.add_argument("--relation-conditional-adaptive-target-f1", type=float, default=0.90)
    parser.add_argument(
        "--relation-prototype-contrastive-weight",
        type=float,
        default=0.0,
        help=(
            "Feature-space prototype-NCA weight for relation-conditioned RSCD "
            "factor boundaries. It contrasts each image feature against classifier "
            "prototypes inside P(friction|material,unevenness), "
            "P(material|friction,unevenness), and "
            "P(unevenness|friction,material), so it changes training geometry "
            "without adding a test-time head."
        ),
    )
    parser.add_argument(
        "--relation-prototype-contrastive-temperature",
        type=float,
        default=0.18,
        help="Temperature for relation-conditioned prototype-NCA.",
    )
    parser.add_argument(
        "--relation-conditional-uncertainty-margin",
        type=float,
        default=0.35,
        help="Top-1 minus Top-2 logit margin below which conditional decoupling is strongest.",
    )
    parser.add_argument(
        "--relation-conditional-gate-temperature",
        type=float,
        default=12.0,
        help="Sharpness of the uncertainty gate for conditional factor decoupling.",
    )
    parser.add_argument(
        "--roughness-ordinal-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only ordinal supervision for RSCD smooth/slight/severe. "
            "It maps the 27-class logits to unevenness energies and learns the "
            "cumulative thresholds y>=slight and y>=severe, directly targeting "
            "the audited roughness-only failure mode."
        ),
    )
    parser.add_argument(
        "--roughness-boundary-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only hard-neighbor margin for same-friction/same-material "
            "roughness boundaries. Unlike graph smoothing, this explicitly pushes "
            "smooth/slight/severe RSCD neighbor logits apart."
        ),
    )
    parser.add_argument("--roughness-boundary-margin", type=float, default=0.08)
    parser.add_argument("--roughness-boundary-uncertainty-margin", type=float, default=0.35)
    parser.add_argument("--roughness-boundary-gate-temperature", type=float, default=12.0)
    parser.add_argument(
        "--roughness-boundary-focus",
        choices=("core", "hard"),
        default="hard",
        help="Apply the roughness boundary loss to all core asphalt/concrete classes or only audited hard classes.",
    )
    parser.add_argument(
        "--concrete-masked-roughness-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only CMR loss for wet/water concrete. It projects the 27-class "
            "logits to same-state concrete smooth/slight/severe energies and "
            "upweights samples with water-film, specular, dark-water, or "
            "texture-erasure evidence. No inference-time head is added."
        ),
    )
    parser.add_argument(
        "--concrete-masked-roughness-obstruction-weight",
        type=float,
        default=1.5,
        help="Multiplier for optical obstruction evidence inside CMR weighting.",
    )
    parser.add_argument(
        "--concrete-masked-roughness-uncertainty-margin",
        type=float,
        default=0.35,
        help="Top-1 minus Top-2 logit margin below which CMR is strongest.",
    )
    parser.add_argument(
        "--concrete-masked-roughness-gate-temperature",
        type=float,
        default=12.0,
        help="Sharpness of the CMR uncertainty gate.",
    )
    parser.add_argument(
        "--tensor-anova-boundary-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only Tensor-ANOVA boundary energy. It decomposes the core "
            "dry/wet/water x asphalt/concrete x smooth/slight/severe RSCD logits "
            "into main, pairwise, and full three-way coupling terms, then applies "
            "axis-specific margins that keep H_fmr active for hard combinations."
        ),
    )
    parser.add_argument("--tensor-anova-boundary-margin", type=float, default=0.06)
    parser.add_argument("--tensor-anova-boundary-uncertainty-margin", type=float, default=0.35)
    parser.add_argument("--tensor-anova-boundary-gate-temperature", type=float, default=12.0)
    parser.add_argument(
        "--tensor-anova-boundary-focus",
        choices=("core", "hard"),
        default="hard",
        help=(
            "Apply Tensor-ANOVA coupling margins to all factorized core labels or "
            "only the audited hard cells such as wet/water concrete slight/severe."
        ),
    )
    parser.add_argument("--tensor-anova-boundary-friction-weight", type=float, default=0.9)
    parser.add_argument("--tensor-anova-boundary-material-weight", type=float, default=0.6)
    parser.add_argument("--tensor-anova-boundary-roughness-weight", type=float, default=1.4)
    parser.add_argument(
        "--tensor-anova-boundary-obstruction-weight",
        type=float,
        default=1.5,
        help=(
            "Extra weight for wet/water concrete roughness coupling when optical "
            "obstruction statistics indicate water film or texture erasure."
        ),
    )
    parser.add_argument(
        "--factor-neighbor-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Margin loss weight against coupled RSCD factor-neighbor negatives. "
            "This targets observed errors such as water/wet concrete and "
            "slight/severe roughness confusion without changing the test protocol."
        ),
    )
    parser.add_argument(
        "--factor-neighbor-margin",
        type=float,
        default=0.25,
        help="Required logit margin over coupled factor-neighbor hard negatives.",
    )
    parser.add_argument(
        "--factor-neighbor-contrastive-weight",
        type=float,
        default=0.0,
        help=(
            "Feature-space penalty that pushes RSCD factor-neighbor samples apart "
            "without directly modifying class logits."
        ),
    )
    parser.add_argument(
        "--factor-neighbor-contrastive-margin",
        type=float,
        default=0.25,
        help="Maximum allowed cosine similarity for feature-space factor-neighbor pairs.",
    )
    parser.add_argument(
        "--factor-prototype-contrastive-weight",
        type=float,
        default=0.0,
        help=(
            "Feature-to-class-prototype angular margin against RSCD factor-neighbor "
            "classes. Unlike in-batch contrastive loss, this fires for every sample "
            "by using classifier rows as class prototypes."
        ),
    )
    parser.add_argument(
        "--factor-prototype-contrastive-margin",
        type=float,
        default=0.10,
        help="Required angular margin between the true class prototype and factor-neighbor prototypes.",
    )
    parser.add_argument(
        "--controlled-factor-tournament-weight",
        type=float,
        default=0.0,
        help=(
            "CFT loss weight. CFT performs image-feature tournaments between "
            "same-factor positives and controlled opponents that share the other "
            "two RSCD factors but differ on one factor. It is training-only and "
            "adds no inference-time parameters."
        ),
    )
    parser.add_argument(
        "--controlled-factor-tournament-temperature",
        type=float,
        default=0.15,
        help="Softmax temperature for controlled factor tournaments.",
    )
    parser.add_argument(
        "--controlled-factor-tournament-margin",
        type=float,
        default=0.10,
        help="Required tournament margin: same-factor commonality must beat controlled single-factor differences.",
    )
    parser.add_argument(
        "--controlled-factor-tournament-neg-weight",
        type=float,
        default=1.0,
        help="Weight of controlled single-factor opponents in the CFT denominator.",
    )
    parser.add_argument(
        "--controlled-factor-tournament-focus",
        choices=("core", "all"),
        default="core",
        help="Apply CFT to fully factorized RSCD core labels or all labels with valid factors.",
    )
    parser.add_argument("--controlled-factor-tournament-friction-weight", type=float, default=0.8)
    parser.add_argument("--controlled-factor-tournament-material-weight", type=float, default=0.6)
    parser.add_argument("--controlled-factor-tournament-unevenness-weight", type=float, default=1.2)
    parser.add_argument(
        "--mechanism-controlled-factor-tournament-weight",
        type=float,
        default=0.0,
        help=(
            "Mechanism-adapted CFT loss weight. Unlike generic CFT on the fused "
            "embedding, it compares friction on PhysicsTexture evidence, material "
            "on low-level texture/semantic-physics evidence, and unevenness on "
            "LocalPhysicsField evidence."
        ),
    )
    parser.add_argument("--mechanism-controlled-factor-tournament-temperature", type=float, default=0.15)
    parser.add_argument("--mechanism-controlled-factor-tournament-margin", type=float, default=0.08)
    parser.add_argument("--mechanism-controlled-factor-tournament-neg-weight", type=float, default=1.0)
    parser.add_argument(
        "--mechanism-controlled-factor-tournament-focus",
        choices=("core", "all"),
        default="core",
        help="Apply mechanism-CFT to fully factorized RSCD core labels or all valid-factor labels.",
    )
    parser.add_argument("--mechanism-controlled-factor-tournament-friction-weight", type=float, default=0.8)
    parser.add_argument("--mechanism-controlled-factor-tournament-material-weight", type=float, default=0.5)
    parser.add_argument("--mechanism-controlled-factor-tournament-unevenness-weight", type=float, default=1.4)
    parser.add_argument(
        "--factor-neighbor-core-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply factor-neighbor margin only to the RSCD bottleneck core: "
            "dry/wet/water asphalt-concrete classes. This tests whether targeted "
            "coupled boundaries help without disturbing gravel, mud, and winter classes."
        ),
    )
    parser.add_argument(
        "--factor-neighbor-loss-mode",
        choices=("hardest", "weighted"),
        default="hardest",
        help=(
            "Use the original hardest-neighbor margin or a weighted all-neighbor "
            "margin. The weighted mode is intended for first-principles RSCD "
            "factor-graph training: roughness, wet/water, and concrete hard-cell "
            "confusions can be emphasized without adding test-time modules."
        ),
    )
    parser.add_argument("--factor-neighbor-roughness-weight", type=float, default=1.0)
    parser.add_argument("--factor-neighbor-friction-weight", type=float, default=1.0)
    parser.add_argument("--factor-neighbor-material-weight", type=float, default=1.0)
    parser.add_argument(
        "--factor-neighbor-wet-water-weight",
        type=float,
        default=1.0,
        help="Extra multiplier for wet-water friction neighbors with matched material/roughness.",
    )
    parser.add_argument(
        "--factor-neighbor-concrete-weight",
        type=float,
        default=1.0,
        help="Extra multiplier when the true class is a concrete hard cell.",
    )
    parser.add_argument(
        "--factor-neighbor-hard-class-weight",
        type=float,
        default=1.0,
        help="Extra multiplier for the known worst RSCD hard classes.",
    )
    parser.add_argument(
        "--local-factor-graph-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only uncertainty-gated margin for active top-k RSCD "
            "factor-graph confusions. Unlike broad graph smoothing, it only "
            "fires when a current top-k alternative is a valid neighboring "
            "composition of the true friction/material/unevenness factors."
        ),
    )
    parser.add_argument(
        "--local-factor-graph-margin",
        type=float,
        default=0.10,
        help="Required logit margin over the active local factor-graph competitor.",
    )
    parser.add_argument(
        "--local-factor-graph-uncertainty-margin",
        type=float,
        default=0.35,
        help="Top-1 minus Top-2 margin below which the local graph loss is strongest.",
    )
    parser.add_argument(
        "--local-factor-graph-gate-temperature",
        type=float,
        default=12.0,
        help="Sharpness of the local factor-graph uncertainty gate.",
    )
    parser.add_argument(
        "--local-factor-graph-topk",
        type=int,
        default=3,
        help="How many current alternatives are checked for local graph-neighbor confusions.",
    )
    parser.add_argument(
        "--relation-specific-clean-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only RSCM loss. It applies clean-logit margins only when "
            "the current top-k alternatives contain a single-factor RSCD neighbor, "
            "with relation-specific friction, roughness, and material margins."
        ),
    )
    parser.add_argument(
        "--relation-specific-clean-margin-scope",
        choices=("all", "core", "hard"),
        default="core",
        help="Class graph scope for RSCM.",
    )
    parser.add_argument("--relation-specific-clean-margin-friction-margin", type=float, default=0.10)
    parser.add_argument("--relation-specific-clean-margin-roughness-margin", type=float, default=0.12)
    parser.add_argument("--relation-specific-clean-margin-material-margin", type=float, default=0.08)
    parser.add_argument("--relation-specific-clean-margin-friction-weight", type=float, default=1.1)
    parser.add_argument("--relation-specific-clean-margin-roughness-weight", type=float, default=1.4)
    parser.add_argument("--relation-specific-clean-margin-material-weight", type=float, default=0.8)
    parser.add_argument(
        "--relation-specific-clean-margin-protected-negative-scale",
        type=float,
        default=0.20,
        help=(
            "Scale applied when a wet/water protected class appears only as a "
            "negative competitor. Set to 0 for strict no-harm protection."
        ),
    )
    parser.add_argument("--relation-specific-clean-margin-uncertainty-margin", type=float, default=0.35)
    parser.add_argument("--relation-specific-clean-margin-gate-temperature", type=float, default=12.0)
    parser.add_argument("--relation-specific-clean-margin-topk", type=int, default=3)
    parser.add_argument(
        "--directed-confusion-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only asymmetric margin weight for high-frequency RSCD "
            "confusion directions observed in the current best model."
        ),
    )
    parser.add_argument(
        "--directed-confusion-margin",
        type=float,
        default=0.08,
        help="Required logit margin over selected directed confusion negatives.",
    )
    parser.add_argument(
        "--directed-confusion-preset",
        choices=("none", "rscd_hard_v1", "rscd_protected_v2", "rscd_smooth_gap_v3"),
        default="none",
        help=(
            "Preset of directed hard pairs. rscd_hard_v1 targets water/wet "
            "concrete, slight/severe roughness, and water-asphalt hard cells. "
            "rscd_protected_v2 keeps only concrete hard cells after the graph "
            "audit showed water-asphalt and granular protection risks. "
            "rscd_smooth_gap_v3 targets the latest full-test gap to RSPNet-L: "
            "high-support smooth and granular confusions."
        ),
    )
    parser.add_argument(
        "--graph-angular-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for Graph-Factor Angular Regularization (GFAR). GFAR treats "
            "classifier rows as class prototypes on a hypersphere and separates "
            "audited hard-confusion class pairs by an angular margin."
        ),
    )
    parser.add_argument(
        "--graph-angular-max-cosine",
        type=float,
        default=0.15,
        help="Maximum allowed cosine similarity for hard-confusion classifier prototypes.",
    )
    parser.add_argument(
        "--graph-angular-preset",
        choices=("none", "rscd_hard_v1", "rscd_protected_v2", "rscd_dry_roughness_v1"),
        default="none",
        help="Class-pair graph used by GFAR.",
    )
    parser.add_argument(
        "--backbone-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only auxiliary CE on the global backbone feature. Used with "
            "--physics-aux-weight as RoadMamba/RoadFormer-style global-local "
            "branch supervision; evaluation still uses only the fused classifier."
        ),
    )
    parser.add_argument("--physics-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--physics-evidence-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only dense physical-evidence supervision on ConvNeXt stage "
            "feature maps. It predicts obstruction, visible/hidden roughness, "
            "film, texture-erasure, and granular evidence maps from early/mid/"
            "late/final features, targeting RSCD mechanism observability rather "
            "than adding a late classification head."
        ),
    )
    parser.add_argument(
        "--physics-evidence-aux-scope",
        choices=MECHANISM_TRAIN_SCOPES,
        default="all",
        help=(
            "Mechanism gate for dense physical-evidence supervision. This keeps "
            "the auxiliary signal tied to the RSCD coupling mechanism, e.g. "
            "core_paved or hard_audited, instead of forcing granular, winter, "
            "and paved samples to share one observation model."
        ),
    )
    parser.add_argument(
        "--physics-evidence-aux-field-mode",
        choices=PHYSICS_EVIDENCE_FIELD_MODES,
        default="all",
        help=(
            "Which analytic evidence fields receive dense stage supervision. "
            "Use roughness_coupling or wet_concrete_hidden_roughness to target "
            "RSCD's dominant smooth/slight/severe and wet/water-concrete "
            "observability bottleneck without equally forcing unrelated fields."
        ),
    )
    parser.add_argument(
        "--local-physics-factor-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only factor supervision on the LocalPhysicsField feature. "
            "It encourages the local optical/texture evidence branch to decouple "
            "RSCD friction, material, and unevenness factors while the final "
            "evaluation still uses the original 27-class protocol."
        ),
    )
    parser.add_argument(
        "--factor-aux-friction-focus",
        choices=(
            "all",
            "dry_only",
            "non_wet_water",
            "dry_paved",
            "wet_water_concrete",
            "water_concrete",
            "wet_concrete",
            "water_asphalt",
            "wet_asphalt",
            "granular",
            "winter",
        ),
        default="all",
        help=(
            "Sample gate for the training-only factor auxiliary loss. The "
            "mechanism-specific options implement RSCD factor-first curricula "
            "without forcing all wet/water, granular, and winter mechanisms to "
            "share one factor-supervision rule."
        ),
    )
    parser.add_argument(
        "--factor-aux-axis-focus",
        choices=(
            "all",
            "friction",
            "material",
            "unevenness",
            "roughness",
            "friction_material",
            "friction_unevenness",
            "friction_roughness",
            "material_unevenness",
            "material_roughness",
        ),
        default="all",
        help=(
            "Which RSCD factor axes receive factor auxiliary supervision. This "
            "keeps mechanism curricula from over-constraining irrelevant axes, "
            "e.g. dry_paved + roughness or wet_water_concrete + friction_roughness."
        ),
    )
    parser.add_argument(
        "--hard-condition-boost",
        type=float,
        default=0.0,
        help=(
            "Extra training-loss weight for RSCD hard wet/water concrete classes. "
            "This targets the observed failure slice without changing labels, "
            "test data, or metrics. A value around 0.25-0.50 is intended for "
            "fast screening."
        ),
    )
    parser.add_argument(
        "--top1-gap-boost",
        type=float,
        default=0.0,
        help=(
            "Extra training-loss weight for audited high-support RSCD classes "
            "that dominate sample-weighted Top-1 errors. This keeps the "
            "class-balanced protocol but nudges concrete/roughness hard cells "
            "without using a global natural-frequency prior."
        ),
    )
    parser.add_argument(
        "--closed-loop-class-controller-reference-json",
        type=Path,
        default=None,
        help=(
            "Prior validation evaluation JSON used by the one-step class-loss "
            "controller. The controller reads per-class F1 as feedback state and "
            "builds bounded gain-scheduled class weights for the next fine-tune."
        ),
    )
    parser.add_argument(
        "--closed-loop-class-controller-strength",
        type=float,
        default=0.0,
        help=(
            "P-gain for the validation-feedback class controller. Values around "
            "0.10-0.35 are intended for fast screens; 0 disables it."
        ),
    )
    parser.add_argument("--closed-loop-class-controller-target-f1", type=float, default=0.90)
    parser.add_argument(
        "--closed-loop-class-controller-factor-strength",
        type=float,
        default=0.35,
        help=(
            "How much same-factor validation deficit contributes to each class "
            "gain, implementing RSCD friction/material/roughness gain scheduling."
        ),
    )
    parser.add_argument("--closed-loop-class-controller-min-gain", type=float, default=0.90)
    parser.add_argument("--closed-loop-class-controller-max-gain", type=float, default=1.35)
    parser.add_argument(
        "--hierarchical-smoothing",
        type=float,
        default=0.0,
        help=(
            "Soft-label mass for RSCD factor-neighbor classes. This uses the "
            "friction/material/unevenness structure of the 27 labels to reduce "
            "overconfidence on subjective dry-wet/wet-water boundaries."
        ),
    )
    parser.add_argument(
        "--roughness-neighbor-smoothing",
        type=float,
        default=0.0,
        help=(
            "Very narrow soft-label mass for same-friction, same-material, adjacent "
            "smooth/slight/severe RSCD roughness classes. This is a gentler "
            "alternative to roughness hard margins and logit residuals."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-smoothing",
        type=float,
        default=0.0,
        help=(
            "Small soft-label mass diffused over the RSCD physical class graph. "
            "This is a training-only graph heat-kernel regularizer and does not "
            "change the 27-class test protocol."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-temperature",
        type=float,
        default=0.35,
        help="Heat-kernel temperature for --graph-diffusion-smoothing.",
    )
    parser.add_argument(
        "--graph-diffusion-core-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply graph diffusion only to dry/wet/water asphalt-concrete "
            "smooth/slight/severe classes, where RSCD errors concentrate."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary uncertainty-gated graph diffusion loss weight. Unlike "
            "--graph-diffusion-smoothing, this keeps the primary CE target one-hot "
            "and only adds graph smoothing on uncertain hard classes."
        ),
    )
    parser.add_argument("--graph-diffusion-aux-smoothing", type=float, default=0.04)
    parser.add_argument("--graph-diffusion-uncertainty-margin", type=float, default=0.30)
    parser.add_argument("--graph-diffusion-gate-temperature", type=float, default=12.0)
    parser.add_argument(
        "--graph-diffusion-hard-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gate graph diffusion auxiliary loss to the audited six worst RSCD classes.",
    )
    parser.add_argument(
        "--factor-neighbor-prior-utility-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only local utility CE weight. It adds a natural-prior bias "
            "only inside uncertain top-k RSCD factor-neighbor candidates, targeting "
            "the Top-1 versus Mean-F1 mismatch without global class-prior collapse."
        ),
    )
    parser.add_argument("--factor-neighbor-prior-utility-lambda", type=float, default=0.08)
    parser.add_argument("--factor-neighbor-prior-utility-topk", type=int, default=3)
    parser.add_argument("--factor-neighbor-prior-utility-margin", type=float, default=0.50)
    parser.add_argument(
        "--factor-neighbor-prior-utility-mode",
        choices=["neighbor", "neighbor_hard"],
        default="neighbor_hard",
    )
    parser.add_argument("--factor-neighbor-prior-alpha", type=float, default=2.0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hard-pair-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a graph-audited sampler that places reciprocal RSCD hard-boundary "
            "class pairs into the training stream, e.g. slight/severe concrete and "
            "wet/water concrete. This is a training-only diagnostic switch."
        ),
    )
    parser.add_argument(
        "--hard-pair-sampling-fraction",
        type=float,
        default=0.65,
        help="Approximate fraction of sampled items produced by audited hard-boundary pairs.",
    )
    parser.add_argument(
        "--controlled-factor-tournament-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use 2x2 RSCD controlled-factor rectangles in the training stream. "
            "Each rectangle holds two contexts and two values on one factor axis, "
            "so CFT sees both same-factor positives and single-factor opponents."
        ),
    )
    parser.add_argument(
        "--controlled-factor-tournament-sampling-fraction",
        type=float,
        default=0.65,
        help="Approximate fraction of sampled items produced by controlled-factor rectangles.",
    )
    parser.add_argument("--samples-per-epoch", type=int, default=36000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-train-samples-per-class", type=int, default=None)
    parser.add_argument("--max-val-samples-per-class", type=int, default=None)
    parser.add_argument("--max-test-samples-per-class", type=int, default=None)
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=(
            "macro_f1",
            "top1",
            "weighted_f1",
            "balanced_accuracy",
            "top1_macro_guard",
            "top1_hardslice_guard",
            "top1_macro_hardslice_guard",
        ),
        default="macro_f1",
        help=(
            "Metric used to save best.pt during training. The default preserves "
            "previous macro-F1-oriented runs. top1_macro_guard selects higher "
            "Top-1 checkpoints only when validation macro-F1 stays within the "
            "configured guard band. top1_hardslice_guard additionally protects "
            "audited wet/water hard slices that repeatedly regress in RSCD screens."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-macro-tolerance",
        type=float,
        default=0.002,
        help=(
            "Allowed validation macro-F1 drop below the best macro-F1 seen so far "
            "when --checkpoint-selection-metric=top1_macro_guard. Default is 0.2pp."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-macro-floor",
        type=float,
        default=0.0,
        help=(
            "Absolute validation macro-F1 floor for top1_macro_guard selection. "
            "Use this with uncapped/natural validation if a formal run optimizes "
            "Top-1 while protecting Mean-F1."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-hard-slice-classes",
        default=",".join(DEFAULT_CHECKPOINT_HARD_SLICE_CLASSES),
        help=(
            "Comma-separated RSCD class labels used by hard-slice checkpoint "
            "guards. Defaults to the audited wet/water concrete and severe "
            "asphalt cells that most often collapse when Top-1 is optimized."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-hard-slice-tolerance",
        type=float,
        default=0.003,
        help=(
            "Allowed validation minimum hard-slice F1 drop below the best hard "
            "slice seen so far when using a hard-slice checkpoint guard. "
            "Default is 0.3pp."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-hard-slice-floor",
        type=float,
        default=0.0,
        help=(
            "Absolute floor for the minimum hard-slice validation F1 when using "
            "top1_hardslice_guard or top1_macro_hardslice_guard."
        ),
    )
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write per-sample test predictions for diagnostic confusion/graph analysis.",
    )
    parser.add_argument(
        "--save-probabilities-npz",
        type=Path,
        default=None,
        help=(
            "Optional NPZ path for per-sample class probabilities from the evaluated split. "
            "The file contains image_path, label, and probs arrays and can be reused as "
            "--distill-teacher-probs for protected expert training."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint whose model weights are loaded before training. "
            "This is for low-LR fine-tuning or continuing a completed formal run; "
            "optimizer state is intentionally not restored unless a future protocol adds it."
        ),
    )
    parser.add_argument(
        "--backbone-init-from",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint used only to initialize parameters whose names start with "
            "backbone. This supports self-supervised evidence pretraining without loading "
            "classification heads or other task-specific modules."
        ),
    )
    parser.add_argument(
        "--train-only-module-prefix",
        action="append",
        default=[],
        help=(
            "Freeze all parameters except modules whose parameter names start with this prefix. "
            "Can be passed multiple times, e.g. relation_signed_graph_expert. This is used "
            "to test whether a zero-initialized expert helps without disturbing the validated backbone."
        ),
    )
    args = parser.parse_args()

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    class_to_idx = build_class_map([args.train_manifest, args.val_manifest, args.test_manifest])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    checkpoint_hard_slice_classes = parse_checkpoint_guard_classes(args.checkpoint_selection_hard_slice_classes)
    write_protocol_manifest(args, class_to_idx)

    data_cfg = {"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor}
    if bool(args.train_augmentation):
        train_tf = build_transforms(
            int(args.image_size),
            train=True,
            aug_cfg={
                "random_resized_crop": False,
                "resize_mode": str(args.train_resize_mode),
                "color_jitter": {"brightness": 0.25, "contrast": 0.25, "saturation": 0.12, "hue": 0.025},
                "random_grayscale_p": 0.05,
                "gaussian_blur_p": 0.08,
                "blur_sigma": [0.1, 1.0],
                "random_erasing_p": 0.03,
                "erase_scale": [0.015, 0.06],
                "line_erasing_p": max(0.0, float(args.line_erasing_p)),
                "line_erasing_num_lines": [
                    max(1, int(args.line_erasing_min_lines)),
                    max(max(1, int(args.line_erasing_min_lines)), int(args.line_erasing_max_lines)),
                ],
                "line_erasing_length": [
                    max(0.01, float(args.line_erasing_min_length)),
                    min(1.0, max(float(args.line_erasing_min_length), float(args.line_erasing_max_length))),
                ],
                "line_erasing_width": [
                    max(0.001, float(args.line_erasing_min_width)),
                    min(0.50, max(float(args.line_erasing_min_width), float(args.line_erasing_max_width))),
                ],
                "gray_world_alpha": max(0.0, min(1.0, float(args.gray_world_alpha))),
                "fourier_low_freq_jitter_p": max(0.0, min(1.0, float(args.fourier_low_freq_jitter_p))),
                "fourier_beta": max(0.001, min(0.50, float(args.fourier_beta))),
                "fourier_strength": [
                    max(0.01, float(args.fourier_strength_min)),
                    max(max(0.01, float(args.fourier_strength_min)), float(args.fourier_strength_max)),
                ],
            },
        )
    else:
        train_tf = build_transforms(
            int(args.image_size),
            train=False,
            aug_cfg={"resize_mode": str(args.train_resize_mode)},
        )
    eval_tf = build_transforms(int(args.image_size), train=False, aug_cfg={"resize_mode": str(args.eval_resize_mode)})

    train_ds = RSCDSurfaceDataset(
        args.train_manifest,
        class_to_idx=class_to_idx,
        transform=train_tf,
        max_samples=args.max_train_samples,
        max_samples_per_class=args.max_train_samples_per_class,
        seed=int(args.seed),
        mechanism_scope=str(args.train_mechanism_scope),
    )
    val_ds = RSCDSurfaceDataset(
        args.val_manifest,
        class_to_idx=class_to_idx,
        transform=eval_tf,
        max_samples=args.max_val_samples,
        max_samples_per_class=args.max_val_samples_per_class,
        seed=int(args.seed) + 1,
    )
    test_ds = RSCDSurfaceDataset(
        args.test_manifest,
        class_to_idx=class_to_idx,
        transform=eval_tf,
        max_samples=args.max_test_samples,
        max_samples_per_class=args.max_test_samples_per_class,
        seed=int(args.seed) + 2,
    )
    print(
        "Dataset sizes: "
        f"train={len(train_ds)} (scope={args.train_mechanism_scope}) "
        f"val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )
    train_loader = build_loader(train_ds, args, data_cfg, train=True)
    val_loader = build_loader(val_ds, args, data_cfg, train=False)
    test_loader = build_loader(test_ds, args, data_cfg, train=False)

    hard_pair_aux_pairs = (
        build_hard_pair_aux_pairs(class_to_idx, focus=str(args.hard_pair_aux_focus))
        if float(args.hard_pair_aux_weight) > 0.0
        else []
    )
    if float(args.hard_pair_aux_weight) > 0.0:
        print(
            f"HCF hard-pair auxiliary boundaries: {len(hard_pair_aux_pairs)} "
            f"(focus={args.hard_pair_aux_focus})",
            flush=True,
        )
        if not hard_pair_aux_pairs:
            print("WARNING: hard-pair auxiliary loss requested but no valid hard pairs were found.", flush=True)

    model = SurfaceClassifier(
        backbone=str(args.backbone),
        embedding_dim=int(args.embedding_dim),
        num_classes=len(class_to_idx),
        pretrained=bool(args.pretrained),
        dropout=float(args.dropout),
        use_physics_branch=bool(args.use_physics_branch),
        physics_dim=int(args.physics_dim),
        physics_quality_cues=bool(args.physics_quality_cues),
        physics_quality_region_cues=bool(args.physics_quality_region_cues),
        use_directional_texture_branch=bool(args.use_directional_texture_branch),
        directional_texture_dim=int(args.directional_texture_dim),
        use_wavelet_texture_branch=bool(args.use_wavelet_texture_branch),
        wavelet_texture_dim=int(args.wavelet_texture_dim),
        use_retinex_texture_branch=bool(args.use_retinex_texture_branch),
        retinex_texture_dim=int(args.retinex_texture_dim),
        retinex_region_cues=bool(args.retinex_region_cues),
        use_physics_attention_branch=bool(args.use_physics_attention_branch),
        physics_attention_dim=int(args.physics_attention_dim),
        use_semantic_physics_attention_branch=bool(args.use_semantic_physics_attention_branch),
        semantic_physics_attention_dim=int(args.semantic_physics_attention_dim),
        use_visibility_observed_roughness_branch=bool(args.use_visibility_observed_roughness_branch),
        visibility_observed_roughness_dim=int(args.visibility_observed_roughness_dim),
        use_visibility_observed_roughness_adapter=bool(args.use_visibility_observed_roughness_adapter),
        visibility_observed_roughness_scale=float(args.visibility_observed_roughness_scale),
        use_factor_conditioned_physics_token_branch=bool(args.use_factor_conditioned_physics_token_branch),
        factor_conditioned_physics_token_dim=int(args.factor_conditioned_physics_token_dim),
        factor_conditioned_physics_token_inner_dim=int(args.factor_conditioned_physics_token_inner_dim),
        use_factor_coupled_physics_token_branch=bool(args.use_factor_coupled_physics_token_branch),
        factor_coupled_physics_token_dim=int(args.factor_coupled_physics_token_dim),
        factor_coupled_physics_token_inner_dim=int(args.factor_coupled_physics_token_inner_dim),
        use_local_physics_field_branch=bool(args.use_local_physics_field_branch),
        local_physics_field_dim=int(args.local_physics_field_dim),
        local_physics_field_scale=float(args.local_physics_field_scale),
        use_relation_conditioned_physics_expert_branch=bool(args.use_relation_conditioned_physics_expert_branch),
        relation_conditioned_physics_expert_dim=int(args.relation_conditioned_physics_expert_dim),
        relation_conditioned_physics_expert_inner_dim=int(args.relation_conditioned_physics_expert_inner_dim),
        use_relation_conditioned_physics_expert_adapter=bool(args.use_relation_conditioned_physics_expert_adapter),
        relation_conditioned_physics_expert_scale=float(args.relation_conditioned_physics_expert_scale),
        use_topological_texture_branch=bool(args.use_topological_texture_branch),
        topological_texture_dim=int(args.topological_texture_dim),
        use_anti_human_texture_branch=bool(args.use_anti_human_texture_branch),
        anti_human_texture_dim=int(args.anti_human_texture_dim),
        use_texture_gate=bool(args.use_texture_gate),
        use_texture_residual_adapter=bool(args.use_texture_residual_adapter),
        texture_residual_scale=float(args.texture_residual_scale),
        use_texture_film=bool(args.use_texture_film),
        texture_film_scale=float(args.texture_film_scale),
        use_material_conditioned_texture_gate=bool(args.use_material_conditioned_texture_gate),
        material_conditioned_gate_scale=float(args.material_conditioned_gate_scale),
        use_artifact_aware_texture_gate=bool(args.use_artifact_aware_texture_gate),
        artifact_aware_gate_scale=float(args.artifact_aware_gate_scale),
        use_smooth_evidence_texture_gate=bool(args.use_smooth_evidence_texture_gate),
        smooth_evidence_texture_gate_scale=float(args.smooth_evidence_texture_gate_scale),
        smooth_evidence_texture_gate_temperature=float(args.smooth_evidence_texture_gate_temperature),
        smooth_evidence_texture_gate_rough_suppression=float(
            args.smooth_evidence_texture_gate_rough_suppression
        ),
        use_tri_chart_evidence_film=bool(args.use_tri_chart_evidence_film),
        tri_chart_evidence_film_hidden_dim=int(args.tri_chart_evidence_film_hidden_dim),
        tri_chart_evidence_film_scale=float(args.tri_chart_evidence_film_scale),
        tri_chart_evidence_film_gate_temperature=float(args.tri_chart_evidence_film_gate_temperature),
        use_mechanism_conditioned_artifact_gate=bool(args.use_mechanism_conditioned_artifact_gate),
        mechanism_conditioned_artifact_gate_scale=float(args.mechanism_conditioned_artifact_gate_scale),
        mechanism_conditioned_artifact_gate_temperature=float(args.mechanism_conditioned_artifact_gate_temperature),
        use_factor_logit_adjustment=bool(args.use_factor_logit_adjustment),
        factor_logit_adjustment_scale=float(args.factor_logit_adjustment_scale),
        use_factorized_low_rank_head=bool(args.use_factorized_low_rank_head),
        factorized_rank=int(args.factorized_rank),
        factorized_scale=float(args.factorized_scale),
        factorized_normalize=bool(args.factorized_normalize),
        factorized_zero_init=bool(args.factorized_zero_init),
        factorized_factors=parse_factorized_factors(args.factorized_factors),
        factorized_class_embedding=bool(args.factorized_class_embedding),
        use_safe_factorized_low_rank_head=bool(args.use_safe_factorized_low_rank_head),
        safe_factorized_rank=int(args.safe_factorized_rank),
        safe_factorized_scale=float(args.safe_factorized_scale),
        safe_factorized_gate_threshold=float(args.safe_factorized_gate_threshold),
        safe_factorized_gate_temperature=float(args.safe_factorized_gate_temperature),
        safe_factorized_protected_negative_limit=float(args.safe_factorized_protected_negative_limit),
        use_factor_interaction_low_rank_head=bool(args.use_factor_interaction_low_rank_head),
        factor_interaction_rank=int(args.factor_interaction_rank),
        factor_interaction_scale=float(args.factor_interaction_scale),
        factor_interaction_gate_threshold=float(args.factor_interaction_gate_threshold),
        factor_interaction_gate_temperature=float(args.factor_interaction_gate_temperature),
        factor_interaction_protected_negative_limit=float(args.factor_interaction_protected_negative_limit),
        use_conditional_coupling_decomposition_field=bool(args.use_conditional_coupling_decomposition_field),
        conditional_coupling_rank=int(args.conditional_coupling_rank),
        conditional_coupling_scale=float(args.conditional_coupling_scale),
        conditional_coupling_gate_threshold=float(args.conditional_coupling_gate_threshold),
        conditional_coupling_gate_temperature=float(args.conditional_coupling_gate_temperature),
        conditional_coupling_protected_negative_limit=float(args.conditional_coupling_protected_negative_limit),
        conditional_coupling_relation_gate_hidden_dim=int(args.conditional_coupling_relation_gate_hidden_dim),
        conditional_coupling_relation_gate_temperature=float(args.conditional_coupling_relation_gate_temperature),
        use_mobius_sheaf_factor_head=bool(args.use_mobius_sheaf_factor_head),
        mobius_sheaf_rank=int(args.mobius_sheaf_rank),
        mobius_sheaf_scale=float(args.mobius_sheaf_scale),
        mobius_sheaf_mode=str(args.mobius_sheaf_mode),
        mobius_sheaf_blend=float(args.mobius_sheaf_blend),
        mobius_sheaf_gate_hidden_dim=int(args.mobius_sheaf_gate_hidden_dim),
        mobius_sheaf_gate_temperature=float(args.mobius_sheaf_gate_temperature),
        mobius_sheaf_normalize=bool(args.mobius_sheaf_normalize),
        mobius_sheaf_zero_init=bool(args.mobius_sheaf_zero_init),
        mobius_sheaf_use_triple=bool(args.mobius_sheaf_use_triple),
        use_mechanism_conditional_sheaf_head=bool(args.use_mechanism_conditional_sheaf_head),
        mechanism_sheaf_rank=int(args.mechanism_sheaf_rank),
        mechanism_sheaf_scale=float(args.mechanism_sheaf_scale),
        mechanism_sheaf_edge_scale=float(args.mechanism_sheaf_edge_scale),
        mechanism_sheaf_router_hidden_dim=int(args.mechanism_sheaf_router_hidden_dim),
        mechanism_sheaf_edge_dim=int(args.mechanism_sheaf_edge_dim),
        mechanism_sheaf_edge_hidden_dim=int(args.mechanism_sheaf_edge_hidden_dim),
        mechanism_sheaf_class_scope=str(args.mechanism_sheaf_class_scope),
        mechanism_sheaf_use_edge_flow=bool(args.mechanism_sheaf_use_edge_flow),
        mechanism_sheaf_protected_negative_limit=float(args.mechanism_sheaf_protected_negative_limit),
        mechanism_sheaf_sparse_router_topk=int(args.mechanism_sheaf_sparse_router_topk),
        mechanism_sheaf_router_temperature=float(args.mechanism_sheaf_router_temperature),
        mechanism_sheaf_physics_prior_weight=float(args.mechanism_sheaf_physics_prior_weight),
        use_local_global_factor_attention=bool(args.use_local_global_factor_attention),
        local_global_factor_rank=int(args.local_global_factor_rank),
        local_global_factor_scale=float(args.local_global_factor_scale),
        local_global_factor_gate_threshold=float(args.local_global_factor_gate_threshold),
        local_global_factor_gate_temperature=float(args.local_global_factor_gate_temperature),
        local_global_factor_neighbor_gate_floor=float(args.local_global_factor_neighbor_gate_floor),
        local_global_factor_protected_negative_limit=float(args.local_global_factor_protected_negative_limit),
        use_label_graph_residual=bool(args.use_label_graph_residual),
        label_graph_rank=int(args.label_graph_rank),
        label_graph_scale=float(args.label_graph_scale),
        label_graph_gate_threshold=float(args.label_graph_gate_threshold),
        label_graph_gate_temperature=float(args.label_graph_gate_temperature),
        label_graph_neighbor_gate_floor=float(args.label_graph_neighbor_gate_floor),
        use_conditional_evidence_masked_coupling_field=bool(args.use_conditional_evidence_masked_coupling_field),
        evidence_masked_coupling_feature_map_dim=int(args.evidence_masked_coupling_feature_map_dim),
        evidence_masked_coupling_token_dim=int(args.evidence_masked_coupling_token_dim),
        evidence_masked_coupling_rank=int(args.evidence_masked_coupling_rank),
        evidence_masked_coupling_scale=float(args.evidence_masked_coupling_scale),
        evidence_masked_coupling_gate_threshold=float(args.evidence_masked_coupling_gate_threshold),
        evidence_masked_coupling_gate_temperature=float(args.evidence_masked_coupling_gate_temperature),
        evidence_masked_coupling_neighbor_gate_floor=float(args.evidence_masked_coupling_neighbor_gate_floor),
        evidence_masked_coupling_protected_negative_limit=float(args.evidence_masked_coupling_protected_negative_limit),
        use_full_order_coupling_tensor_field=bool(args.use_full_order_coupling_tensor_field),
        full_order_coupling_feature_map_dim=int(args.full_order_coupling_feature_map_dim),
        full_order_coupling_token_dim=int(args.full_order_coupling_token_dim),
        full_order_coupling_hidden_dim=int(args.full_order_coupling_hidden_dim),
        full_order_coupling_scale=float(args.full_order_coupling_scale),
        full_order_coupling_gate_threshold=float(args.full_order_coupling_gate_threshold),
        full_order_coupling_gate_temperature=float(args.full_order_coupling_gate_temperature),
        full_order_coupling_core_gate_floor=float(args.full_order_coupling_core_gate_floor),
        full_order_coupling_protected_negative_limit=float(args.full_order_coupling_protected_negative_limit),
        use_mechanism_charted_full_order_coupling_tensor_field=bool(
            args.use_mechanism_charted_full_order_coupling_tensor_field
        ),
        mechanism_charted_full_order_feature_map_dim=int(args.mechanism_charted_full_order_feature_map_dim),
        mechanism_charted_full_order_token_dim=int(args.mechanism_charted_full_order_token_dim),
        mechanism_charted_full_order_hidden_dim=int(args.mechanism_charted_full_order_hidden_dim),
        mechanism_charted_full_order_scale=float(args.mechanism_charted_full_order_scale),
        mechanism_charted_full_order_gate_threshold=float(args.mechanism_charted_full_order_gate_threshold),
        mechanism_charted_full_order_gate_temperature=float(args.mechanism_charted_full_order_gate_temperature),
        mechanism_charted_full_order_core_gate_floor=float(args.mechanism_charted_full_order_core_gate_floor),
        mechanism_charted_full_order_protected_negative_limit=float(
            args.mechanism_charted_full_order_protected_negative_limit
        ),
        mechanism_charted_full_order_router_hidden_dim=int(args.mechanism_charted_full_order_router_hidden_dim),
        mechanism_charted_full_order_router_temperature=float(args.mechanism_charted_full_order_router_temperature),
        mechanism_charted_full_order_sparse_router_topk=int(args.mechanism_charted_full_order_sparse_router_topk),
        mechanism_charted_full_order_physics_prior_weight=float(args.mechanism_charted_full_order_physics_prior_weight),
        use_core_factor_coupled_residual=bool(args.use_core_factor_coupled_residual),
        core_factor_rank=int(args.core_factor_rank),
        core_factor_scale=float(args.core_factor_scale),
        core_factor_neighbor_gate_floor=float(args.core_factor_neighbor_gate_floor),
        core_factor_uncertainty_threshold=float(args.core_factor_uncertainty_threshold),
        core_factor_uncertainty_temperature=float(args.core_factor_uncertainty_temperature),
        core_factor_protected_negative_limit=float(args.core_factor_protected_negative_limit),
        use_water_evidence_logit_gate=bool(args.use_water_evidence_logit_gate),
        water_evidence_gate_scale=float(args.water_evidence_gate_scale),
        water_evidence_gate_zero_init=bool(args.water_evidence_gate_zero_init),
        use_dry_concrete_roughness_vor_residual=bool(args.use_dry_concrete_roughness_vor_residual),
        dry_concrete_roughness_hidden_dim=int(args.dry_concrete_roughness_hidden_dim),
        dry_concrete_roughness_scale=float(args.dry_concrete_roughness_scale),
        dry_concrete_roughness_gate_threshold=float(args.dry_concrete_roughness_gate_threshold),
        dry_concrete_roughness_gate_temperature=float(args.dry_concrete_roughness_gate_temperature),
        use_dry_paved_roughness_vor_residual=bool(args.use_dry_paved_roughness_vor_residual),
        dry_paved_roughness_hidden_dim=int(args.dry_paved_roughness_hidden_dim),
        dry_paved_roughness_material_dim=int(args.dry_paved_roughness_material_dim),
        dry_paved_roughness_scale=float(args.dry_paved_roughness_scale),
        dry_paved_roughness_gate_threshold=float(args.dry_paved_roughness_gate_threshold),
        dry_paved_roughness_gate_temperature=float(args.dry_paved_roughness_gate_temperature),
        dry_paved_roughness_head_mode=str(args.dry_paved_roughness_head_mode),
        dry_paved_roughness_material_gate_threshold=float(args.dry_paved_roughness_material_gate_threshold),
        dry_paved_roughness_material_gate_temperature=float(args.dry_paved_roughness_material_gate_temperature),
        use_concrete_roughness_vor_residual=bool(args.use_concrete_roughness_vor_residual),
        concrete_roughness_hidden_dim=int(args.concrete_roughness_hidden_dim),
        concrete_roughness_chart_dim=int(args.concrete_roughness_chart_dim),
        concrete_roughness_scale=float(args.concrete_roughness_scale),
        concrete_roughness_gate_threshold=float(args.concrete_roughness_gate_threshold),
        concrete_roughness_gate_temperature=float(args.concrete_roughness_gate_temperature),
        use_wet_water_film_vor_residual=bool(args.use_wet_water_film_vor_residual),
        wet_water_film_hidden_dim=int(args.wet_water_film_hidden_dim),
        wet_water_film_pair_dim=int(args.wet_water_film_pair_dim),
        wet_water_film_scale=float(args.wet_water_film_scale),
        wet_water_film_material_scope=str(args.wet_water_film_material_scope),
        wet_water_film_gate_threshold=float(args.wet_water_film_gate_threshold),
        wet_water_film_gate_temperature=float(args.wet_water_film_gate_temperature),
        use_smooth_film_concrete_expert=bool(args.use_smooth_film_concrete_expert),
        smooth_film_concrete_hidden_dim=int(args.smooth_film_concrete_hidden_dim),
        smooth_film_concrete_scale=float(args.smooth_film_concrete_scale),
        smooth_film_concrete_gate_threshold=float(args.smooth_film_concrete_gate_threshold),
        smooth_film_concrete_gate_temperature=float(args.smooth_film_concrete_gate_temperature),
        use_obstruction_concrete_roughness_vor_residual=bool(
            args.use_obstruction_concrete_roughness_vor_residual
        ),
        obstruction_concrete_roughness_hidden_dim=int(args.obstruction_concrete_roughness_hidden_dim),
        obstruction_concrete_roughness_scale=float(args.obstruction_concrete_roughness_scale),
        obstruction_concrete_roughness_gate_threshold=float(args.obstruction_concrete_roughness_gate_threshold),
        obstruction_concrete_roughness_gate_temperature=float(args.obstruction_concrete_roughness_gate_temperature),
        obstruction_concrete_roughness_share_gate_threshold=float(
            args.obstruction_concrete_roughness_share_gate_threshold
        ),
        obstruction_concrete_roughness_share_gate_temperature=float(
            args.obstruction_concrete_roughness_share_gate_temperature
        ),
        use_coupled_optical_roughness_residual=bool(args.use_coupled_optical_roughness_residual),
        coupled_residual_hidden_dim=int(args.coupled_residual_hidden_dim),
        coupled_residual_scale=float(args.coupled_residual_scale),
        coupled_residual_gate_threshold=float(args.coupled_residual_gate_threshold),
        coupled_residual_gate_temperature=float(args.coupled_residual_gate_temperature),
        coupled_residual_protected_negative_limit=float(args.coupled_residual_protected_negative_limit),
        use_roughness_neighbor_residual=bool(args.use_roughness_neighbor_residual),
        roughness_neighbor_hidden_dim=int(args.roughness_neighbor_hidden_dim),
        roughness_neighbor_scale=float(args.roughness_neighbor_scale),
        roughness_neighbor_gate_threshold=float(args.roughness_neighbor_gate_threshold),
        roughness_neighbor_gate_temperature=float(args.roughness_neighbor_gate_temperature),
        roughness_neighbor_protected_negative_limit=float(args.roughness_neighbor_protected_negative_limit),
        roughness_neighbor_gate_floor=float(args.roughness_neighbor_gate_floor),
        use_spectral_roughness_residual=bool(args.use_spectral_roughness_residual),
        spectral_roughness_hidden_dim=int(args.spectral_roughness_hidden_dim),
        spectral_roughness_scale=float(args.spectral_roughness_scale),
        spectral_roughness_gate_threshold=float(args.spectral_roughness_gate_threshold),
        spectral_roughness_gate_temperature=float(args.spectral_roughness_gate_temperature),
        spectral_roughness_protected_negative_limit=float(args.spectral_roughness_protected_negative_limit),
        spectral_roughness_neighbor_gate_floor=float(args.spectral_roughness_neighbor_gate_floor),
        use_relation_signed_graph_expert=bool(args.use_relation_signed_graph_expert),
        relation_signed_hidden_dim=int(args.relation_signed_hidden_dim),
        relation_signed_scale=float(args.relation_signed_scale),
        relation_signed_gate_threshold=float(args.relation_signed_gate_threshold),
        relation_signed_gate_temperature=float(args.relation_signed_gate_temperature),
        relation_signed_protected_negative_limit=float(args.relation_signed_protected_negative_limit),
        relation_signed_neighbor_gate_floor=float(args.relation_signed_neighbor_gate_floor),
        use_heterophilic_logit_boundary_expert=bool(args.use_heterophilic_logit_boundary_expert),
        heterophilic_boundary_scale=float(args.heterophilic_boundary_scale),
        heterophilic_boundary_gate_threshold=float(args.heterophilic_boundary_gate_threshold),
        heterophilic_boundary_gate_temperature=float(args.heterophilic_boundary_gate_temperature),
        heterophilic_boundary_protected_negative_limit=float(args.heterophilic_boundary_protected_negative_limit),
        use_heterophilic_feature_boundary_expert=bool(args.use_heterophilic_feature_boundary_expert),
        heterophilic_feature_boundary_hidden_dim=int(args.heterophilic_feature_boundary_hidden_dim),
        heterophilic_feature_boundary_pair_dim=int(args.heterophilic_feature_boundary_pair_dim),
        heterophilic_feature_boundary_scale=float(args.heterophilic_feature_boundary_scale),
        heterophilic_feature_boundary_gate_threshold=float(args.heterophilic_feature_boundary_gate_threshold),
        heterophilic_feature_boundary_gate_temperature=float(args.heterophilic_feature_boundary_gate_temperature),
        heterophilic_feature_boundary_protected_negative_limit=float(
            args.heterophilic_feature_boundary_protected_negative_limit
        ),
        use_heterophilic_physics_boundary_expert=bool(args.use_heterophilic_physics_boundary_expert),
        heterophilic_physics_boundary_hidden_dim=int(args.heterophilic_physics_boundary_hidden_dim),
        heterophilic_physics_boundary_pair_dim=int(args.heterophilic_physics_boundary_pair_dim),
        heterophilic_physics_boundary_scale=float(args.heterophilic_physics_boundary_scale),
        heterophilic_physics_boundary_gate_threshold=float(args.heterophilic_physics_boundary_gate_threshold),
        heterophilic_physics_boundary_gate_temperature=float(args.heterophilic_physics_boundary_gate_temperature),
        heterophilic_physics_boundary_protected_negative_limit=float(
            args.heterophilic_physics_boundary_protected_negative_limit
        ),
        use_protected_heterophilic_factor_boundary_field=bool(args.use_protected_heterophilic_factor_boundary_field),
        protected_factor_boundary_hidden_dim=int(args.protected_factor_boundary_hidden_dim),
        protected_factor_boundary_pair_dim=int(args.protected_factor_boundary_pair_dim),
        protected_factor_boundary_relation_dim=int(args.protected_factor_boundary_relation_dim),
        protected_factor_boundary_scale=float(args.protected_factor_boundary_scale),
        protected_factor_boundary_gate_threshold=float(args.protected_factor_boundary_gate_threshold),
        protected_factor_boundary_gate_temperature=float(args.protected_factor_boundary_gate_temperature),
        protected_factor_boundary_protected_negative_limit=float(
            args.protected_factor_boundary_protected_negative_limit
        ),
        use_relation_specific_hard_edge_refiner=bool(args.use_relation_specific_hard_edge_refiner),
        relation_specific_refiner_hidden_dim=int(args.relation_specific_refiner_hidden_dim),
        relation_specific_refiner_pair_dim=int(args.relation_specific_refiner_pair_dim),
        relation_specific_refiner_scale=float(args.relation_specific_refiner_scale),
        relation_specific_refiner_gate_threshold=float(args.relation_specific_refiner_gate_threshold),
        relation_specific_refiner_gate_temperature=float(args.relation_specific_refiner_gate_temperature),
        relation_specific_refiner_protected_negative_limit=float(
            args.relation_specific_refiner_protected_negative_limit
        ),
        use_selective_mechanism_tensor_boundary_field=bool(
            args.use_selective_mechanism_tensor_boundary_field
        ),
        selective_mechanism_tensor_boundary_hidden_dim=int(
            args.selective_mechanism_tensor_boundary_hidden_dim
        ),
        selective_mechanism_tensor_boundary_pair_dim=int(args.selective_mechanism_tensor_boundary_pair_dim),
        selective_mechanism_tensor_boundary_relation_dim=int(
            args.selective_mechanism_tensor_boundary_relation_dim
        ),
        selective_mechanism_tensor_boundary_mechanism_dim=int(
            args.selective_mechanism_tensor_boundary_mechanism_dim
        ),
        selective_mechanism_tensor_boundary_scale=float(args.selective_mechanism_tensor_boundary_scale),
        selective_mechanism_tensor_boundary_gate_threshold=float(
            args.selective_mechanism_tensor_boundary_gate_threshold
        ),
        selective_mechanism_tensor_boundary_gate_temperature=float(
            args.selective_mechanism_tensor_boundary_gate_temperature
        ),
        selective_mechanism_tensor_boundary_mechanism_gate_threshold=float(
            args.selective_mechanism_tensor_boundary_mechanism_gate_threshold
        ),
        selective_mechanism_tensor_boundary_mechanism_gate_temperature=float(
            args.selective_mechanism_tensor_boundary_mechanism_gate_temperature
        ),
        selective_mechanism_tensor_boundary_enabled_mechanisms=str(
            args.selective_mechanism_tensor_boundary_enabled_mechanisms
        ),
        selective_mechanism_tensor_boundary_protected_negative_limit=float(
            args.selective_mechanism_tensor_boundary_protected_negative_limit
        ),
        use_conditional_factor_projection=bool(args.use_conditional_factor_projection),
        conditional_factor_projection_scale=float(args.conditional_factor_projection_scale),
        conditional_factor_projection_gate_threshold=float(args.conditional_factor_projection_gate_threshold),
        conditional_factor_projection_gate_temperature=float(args.conditional_factor_projection_gate_temperature),
        conditional_factor_projection_focus=str(args.conditional_factor_projection_focus),
        conditional_factor_projection_friction_weight=float(args.conditional_factor_projection_friction_weight),
        conditional_factor_projection_material_weight=float(args.conditional_factor_projection_material_weight),
        conditional_factor_projection_unevenness_weight=float(args.conditional_factor_projection_unevenness_weight),
        conditional_factor_projection_protected_negative_limit=float(
            args.conditional_factor_projection_protected_negative_limit
        ),
        use_heterogeneous_label_router=bool(args.use_heterogeneous_label_router),
        heterogeneous_router_hidden_dim=int(args.heterogeneous_router_hidden_dim),
        heterogeneous_router_scale=float(args.heterogeneous_router_scale),
        use_hard_pair_aux=float(args.hard_pair_aux_weight) > 0.0,
        hard_pair_aux_num_pairs=len(hard_pair_aux_pairs),
        hard_pair_aux_hidden_dim=int(args.hard_pair_aux_hidden_dim),
        use_mechanism_orthogonal_coupling_aux=float(args.mechanism_orthogonal_aux_weight) > 0.0,
        mechanism_orthogonal_dim=int(args.mechanism_orthogonal_dim),
        mechanism_orthogonal_hidden_dim=int(args.mechanism_orthogonal_hidden_dim),
        class_to_idx=class_to_idx,
        use_factor_aux=float(args.factor_aux_weight) > 0.0,
        use_local_physics_factor_aux=float(args.local_physics_factor_aux_weight) > 0.0,
        use_backbone_aux=float(args.backbone_aux_weight) > 0.0,
        use_physics_aux=float(args.physics_aux_weight) > 0.0,
        use_physics_evidence_aux=float(args.physics_evidence_aux_weight) > 0.0,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    factor_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    hard_pair_aux_left = torch.as_tensor(
        [left for left, _, _, _ in hard_pair_aux_pairs],
        dtype=torch.long,
        device=device,
    )
    hard_pair_aux_right = torch.as_tensor(
        [right for _, right, _, _ in hard_pair_aux_pairs],
        dtype=torch.long,
        device=device,
    )
    hierarchical_targets = build_hierarchical_targets(class_to_idx, float(args.hierarchical_smoothing)).to(device)
    if float(args.hierarchical_smoothing) <= 0.0:
        hierarchical_targets = None
    roughness_neighbor_targets = build_roughness_neighbor_targets(
        class_to_idx,
        float(args.roughness_neighbor_smoothing),
    ).to(device)
    if float(args.roughness_neighbor_smoothing) <= 0.0:
        roughness_neighbor_targets = None
    if roughness_neighbor_targets is not None:
        if hierarchical_targets is not None:
            raise ValueError("--roughness-neighbor-smoothing and --hierarchical-smoothing are mutually exclusive.")
        hierarchical_targets = roughness_neighbor_targets
    graph_diffusion_targets = build_graph_diffusion_targets(
        class_to_idx,
        float(args.graph_diffusion_smoothing),
        temperature=float(args.graph_diffusion_temperature),
        core_only=bool(args.graph_diffusion_core_only),
    ).to(device)
    if float(args.graph_diffusion_smoothing) <= 0.0:
        graph_diffusion_targets = None
    if graph_diffusion_targets is not None:
        if hierarchical_targets is not None:
            raise ValueError(
                "--graph-diffusion-smoothing is mutually exclusive with "
                "--hierarchical-smoothing and --roughness-neighbor-smoothing."
            )
        hierarchical_targets = graph_diffusion_targets
    hard_condition_weights = build_hard_condition_weights(class_to_idx, float(args.hard_condition_boost)).to(device)
    if float(args.hard_condition_boost) <= 0.0:
        hard_condition_weights = None
    top1_gap_weights = build_top1_gap_weights(class_to_idx, float(args.top1_gap_boost)).to(device)
    if float(args.top1_gap_boost) > 0.0:
        if hard_condition_weights is None:
            hard_condition_weights = top1_gap_weights
        else:
            hard_condition_weights = hard_condition_weights * top1_gap_weights
            hard_condition_weights = hard_condition_weights / hard_condition_weights.mean().clamp_min(1e-6)
    closed_loop_controller_weights = build_closed_loop_class_controller_weights(
        class_to_idx,
        reference_json=args.closed_loop_class_controller_reference_json,
        strength=float(args.closed_loop_class_controller_strength),
        target_f1=float(args.closed_loop_class_controller_target_f1),
        factor_strength=float(args.closed_loop_class_controller_factor_strength),
        min_gain=float(args.closed_loop_class_controller_min_gain),
        max_gain=float(args.closed_loop_class_controller_max_gain),
        device=device,
    )
    if closed_loop_controller_weights is not None:
        if hard_condition_weights is None:
            hard_condition_weights = closed_loop_controller_weights
        else:
            hard_condition_weights = hard_condition_weights * closed_loop_controller_weights
            hard_condition_weights = hard_condition_weights / hard_condition_weights.mean().clamp_min(1e-6)
    factor_marginal_masks = build_factor_marginal_masks(class_to_idx, device)
    relation_conditional_masks = build_relation_conditional_masks(class_to_idx, device)
    relation_conditional_focus_mask = build_relation_conditional_focus_mask(
        class_to_idx,
        focus=str(args.relation_conditional_focus),
        device=device,
    )
    roughness_boundary_mask = build_roughness_boundary_mask(
        class_to_idx,
        focus=str(args.roughness_boundary_focus),
        device=device,
    )
    concrete_masked_roughness_masks = build_concrete_masked_roughness_masks(class_to_idx, device)
    tensor_anova_boundary_spec = build_tensor_anova_boundary_spec(class_to_idx, device)
    relation_conditional_class_axis_weights = build_relation_conditional_adaptive_weights(
        class_to_idx,
        mode=str(args.relation_conditional_adaptive_weighting),
        reference_json=args.relation_conditional_reference_json,
        strength=float(args.relation_conditional_adaptive_strength),
        target_f1=float(args.relation_conditional_adaptive_target_f1),
        device=device,
    )
    factor_neighbor_negative_mask = build_factor_neighbor_negative_mask(
        class_to_idx,
        device,
        core_only=bool(args.factor_neighbor_core_only),
    )
    factor_neighbor_weight_matrix = build_factor_neighbor_weight_matrix(
        class_to_idx,
        device,
        core_only=bool(args.factor_neighbor_core_only),
        roughness_weight=float(args.factor_neighbor_roughness_weight),
        friction_weight=float(args.factor_neighbor_friction_weight),
        material_weight=float(args.factor_neighbor_material_weight),
        wet_water_weight=float(args.factor_neighbor_wet_water_weight),
        concrete_weight=float(args.factor_neighbor_concrete_weight),
        hard_class_weight=float(args.factor_neighbor_hard_class_weight),
    )
    (
        relation_specific_clean_margin_relation_ids,
        relation_specific_clean_margin_class_weights,
        relation_specific_clean_margin_protected_mask,
    ) = build_relation_specific_margin_matrices(
        class_to_idx,
        device,
        scope=str(args.relation_specific_clean_margin_scope),
    )
    factor_neighbor_prior_bias = None
    factor_neighbor_prior_hard_mask = None
    if float(args.factor_neighbor_prior_utility_weight) > 0.0:
        factor_neighbor_prior_bias = build_natural_prior_log_bias(
            args.train_manifest,
            class_to_idx,
            alpha=float(args.factor_neighbor_prior_alpha),
            device=device,
        )
        factor_neighbor_prior_hard_mask = build_factor_neighbor_prior_hard_mask(class_to_idx, device)
    directed_confusion_weight_matrix = build_directed_confusion_weight_matrix(
        class_to_idx,
        device,
        preset=str(args.directed_confusion_preset),
    )
    graph_angular_weight_matrix = build_graph_angular_weight_matrix(
        class_to_idx,
        device,
        preset=str(args.graph_angular_preset),
    )
    graph_diffusion_aux_targets = None
    graph_diffusion_gate_mask = None
    if float(args.graph_diffusion_aux_weight) > 0.0:
        graph_diffusion_aux_targets = build_graph_diffusion_targets(
            class_to_idx,
            float(args.graph_diffusion_aux_smoothing),
            temperature=float(args.graph_diffusion_temperature),
            core_only=bool(args.graph_diffusion_core_only),
        ).to(device)
        graph_diffusion_gate_mask = build_graph_diffusion_gate_mask(
            class_to_idx,
            hard_only=bool(args.graph_diffusion_hard_only),
        ).to(device)
    teacher_error_replay_class_mask = build_teacher_error_replay_class_mask(
        class_to_idx,
        focus=str(args.teacher_error_replay_focus),
        device=device,
    )
    if args.backbone_init_from is not None:
        if not args.backbone_init_from.exists():
            raise FileNotFoundError(f"Checkpoint not found for --backbone-init-from: {args.backbone_init_from}")
        state = torch.load(args.backbone_init_from, map_location=device, weights_only=False)
        raw_state = extract_checkpoint_model_state(state)
        backbone_state = {str(k): v for k, v in raw_state.items() if str(k).startswith("backbone.")}
        if not backbone_state:
            raise ValueError(f"No backbone.* parameters found in {args.backbone_init_from}")
        own_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in backbone_state.items()
            if key in own_state and tuple(own_state[key].shape) == tuple(value.shape)
        }
        skipped = sorted(set(backbone_state) - set(compatible_state))
        if not compatible_state:
            raise ValueError(f"No compatible backbone parameters found in {args.backbone_init_from}")
        own_state.update(compatible_state)
        model.load_state_dict(own_state)
        print(
            f"Initialized backbone from {args.backbone_init_from}: "
            f"loaded={len(compatible_state)} skipped={len(skipped)}",
            flush=True,
        )
    if args.train_only_module_prefix:
        prefixes = tuple(str(item) for item in args.train_only_module_prefix)
        trainable_names = []
        for name, parameter in model.named_parameters():
            keep_trainable = name.startswith(prefixes)
            parameter.requires_grad_(keep_trainable)
            if keep_trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(f"--train-only-module-prefix matched no parameters: {list(prefixes)}")
        preview = ", ".join(trainable_names[:8])
        if len(trainable_names) > 8:
            preview += f", ... (+{len(trainable_names) - 8} more)"
        print(f"Training only parameter prefixes {list(prefixes)}: {preview}", flush=True)

    teacher_probs = load_teacher_probs(args.distill_teacher_probs, len(class_to_idx))
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint = args.checkpoint or (args.output_dir / "best.pt")
    if args.eval_only:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for --eval-only: {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        missing, unexpected, partial = load_state_dict_allow_expanded_head(model, extract_checkpoint_model_state(state))
        if missing or unexpected or partial:
            print(
                "WARNING: flexible eval checkpoint load "
                f"missing={len(missing)} unexpected={len(unexpected)} partial/skipped={len(partial)}",
                flush=True,
            )
        predictions_path = args.output_dir / "predictions_test.csv" if args.save_predictions else None
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            criterion,
            idx_to_class=idx_to_class,
            save_predictions_path=predictions_path,
            save_probabilities_path=args.save_probabilities_npz,
        )
        write_eval_outputs(args.output_dir, test_metrics, split="test")
        print(json.dumps(test_metrics["summary"], indent=2, ensure_ascii=False), flush=True)
        return

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"Checkpoint not found for --resume-from: {args.resume_from}")
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        missing, unexpected, partial = load_state_dict_allow_expanded_head(model, extract_checkpoint_model_state(state))
        if missing or unexpected or partial:
            print(
                "WARNING: flexible resume load "
                f"missing={list(missing)} unexpected={list(unexpected)} partial={list(partial)}",
                flush=True,
            )
        print(f"Resumed model weights from: {args.resume_from}", flush=True)

    online_teacher_model = None
    if float(args.online_teacher_weight) > 0.0 or float(args.teacher_error_replay_weight) > 0.0:
        online_teacher_model = copy.deepcopy(model).to(device)
        if args.online_teacher_checkpoint is not None:
            if not args.online_teacher_checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found for --online-teacher-checkpoint: {args.online_teacher_checkpoint}"
                )
            teacher_state = torch.load(args.online_teacher_checkpoint, map_location=device, weights_only=False)
            missing, unexpected, partial = load_state_dict_allow_expanded_head(
                online_teacher_model,
                extract_checkpoint_model_state(teacher_state),
            )
            if missing or unexpected or partial:
                print(
                    "WARNING: flexible online-teacher load "
                    f"missing={list(missing)} unexpected={list(unexpected)} partial={list(partial)}",
                    flush=True,
                )
            print(f"Loaded online no-harm teacher from: {args.online_teacher_checkpoint}", flush=True)
        else:
            print("Using current initialized/resumed model as online no-harm teacher.", flush=True)
        online_teacher_model.eval()
        for parameter in online_teacher_model.parameters():
            parameter.requires_grad_(False)

    physics_evidence_target_builder = (
        PhysicsEvidenceTarget().to(device)
        if float(args.physics_evidence_aux_weight) > 0.0
        else None
    )

    best_selection_key: tuple[float, ...] = (-math.inf,)
    best_macro_seen = -math.inf
    best_hard_slice_seen = -math.inf
    stale_epochs = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            factor_criterion,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=int(args.grad_accum_steps),
            log_every_steps=int(args.log_every_steps),
            class_loss_weight=float(args.class_loss_weight),
            factor_aux_weight=float(args.factor_aux_weight),
            factor_aux_friction_focus=str(args.factor_aux_friction_focus),
            factor_aux_axis_focus=str(args.factor_aux_axis_focus),
            hard_pair_aux_weight=float(args.hard_pair_aux_weight),
            hard_pair_aux_left=hard_pair_aux_left,
            hard_pair_aux_right=hard_pair_aux_right,
            mechanism_orthogonal_aux_weight=float(args.mechanism_orthogonal_aux_weight),
            mechanism_orthogonal_factor_weight=float(args.mechanism_orthogonal_factor_weight),
            mechanism_orthogonal_mechanism_weight=float(args.mechanism_orthogonal_mechanism_weight),
            mechanism_orthogonal_cov_weight=float(args.mechanism_orthogonal_cov_weight),
            local_physics_factor_aux_weight=float(args.local_physics_factor_aux_weight),
            factor_marginal_weight=float(args.factor_marginal_weight),
            relation_conditional_weight=float(args.relation_conditional_weight),
            relation_conditional_masks=relation_conditional_masks,
            relation_conditional_focus_mask=relation_conditional_focus_mask,
            relation_conditional_axis_weights={
                "friction": float(args.relation_conditional_friction_weight),
                "material": float(args.relation_conditional_material_weight),
                "unevenness": float(args.relation_conditional_unevenness_weight),
            },
            relation_conditional_class_axis_weights=relation_conditional_class_axis_weights,
            relation_prototype_contrastive_weight=float(args.relation_prototype_contrastive_weight),
            relation_prototype_contrastive_temperature=float(args.relation_prototype_contrastive_temperature),
            relation_conditional_uncertainty_margin=float(args.relation_conditional_uncertainty_margin),
            relation_conditional_gate_temperature=float(args.relation_conditional_gate_temperature),
            roughness_ordinal_weight=float(args.roughness_ordinal_weight),
            roughness_boundary_margin_weight=float(args.roughness_boundary_margin_weight),
            roughness_boundary_mask=roughness_boundary_mask,
            roughness_boundary_margin=float(args.roughness_boundary_margin),
            roughness_boundary_uncertainty_margin=float(args.roughness_boundary_uncertainty_margin),
            roughness_boundary_gate_temperature=float(args.roughness_boundary_gate_temperature),
            concrete_masked_roughness_weight=float(args.concrete_masked_roughness_weight),
            concrete_masked_roughness_masks=concrete_masked_roughness_masks,
            concrete_masked_roughness_obstruction_weight=float(args.concrete_masked_roughness_obstruction_weight),
            concrete_masked_roughness_uncertainty_margin=float(args.concrete_masked_roughness_uncertainty_margin),
            concrete_masked_roughness_gate_temperature=float(args.concrete_masked_roughness_gate_temperature),
            tensor_anova_boundary_weight=float(args.tensor_anova_boundary_weight),
            tensor_anova_boundary_spec=tensor_anova_boundary_spec,
            tensor_anova_boundary_focus=str(args.tensor_anova_boundary_focus),
            tensor_anova_boundary_margin=float(args.tensor_anova_boundary_margin),
            tensor_anova_boundary_uncertainty_margin=float(args.tensor_anova_boundary_uncertainty_margin),
            tensor_anova_boundary_gate_temperature=float(args.tensor_anova_boundary_gate_temperature),
            tensor_anova_boundary_friction_weight=float(args.tensor_anova_boundary_friction_weight),
            tensor_anova_boundary_material_weight=float(args.tensor_anova_boundary_material_weight),
            tensor_anova_boundary_roughness_weight=float(args.tensor_anova_boundary_roughness_weight),
            tensor_anova_boundary_obstruction_weight=float(args.tensor_anova_boundary_obstruction_weight),
            factor_marginal_masks=factor_marginal_masks,
            factor_neighbor_margin_weight=float(args.factor_neighbor_margin_weight),
            factor_neighbor_loss_mode=str(args.factor_neighbor_loss_mode),
            factor_neighbor_negative_mask=factor_neighbor_negative_mask,
            factor_neighbor_weight_matrix=factor_neighbor_weight_matrix,
            factor_neighbor_margin=float(args.factor_neighbor_margin),
            factor_neighbor_contrastive_weight=float(args.factor_neighbor_contrastive_weight),
            factor_neighbor_contrastive_margin=float(args.factor_neighbor_contrastive_margin),
            factor_prototype_contrastive_weight=float(args.factor_prototype_contrastive_weight),
            factor_prototype_contrastive_margin=float(args.factor_prototype_contrastive_margin),
            factor_neighbor_roughness_weight=float(args.factor_neighbor_roughness_weight),
            factor_neighbor_friction_weight=float(args.factor_neighbor_friction_weight),
            factor_neighbor_material_weight=float(args.factor_neighbor_material_weight),
            factor_neighbor_wet_water_weight=float(args.factor_neighbor_wet_water_weight),
            factor_neighbor_concrete_weight=float(args.factor_neighbor_concrete_weight),
            factor_neighbor_hard_class_weight=float(args.factor_neighbor_hard_class_weight),
            controlled_factor_tournament_weight=float(args.controlled_factor_tournament_weight),
            controlled_factor_tournament_temperature=float(args.controlled_factor_tournament_temperature),
            controlled_factor_tournament_margin=float(args.controlled_factor_tournament_margin),
            controlled_factor_tournament_neg_weight=float(args.controlled_factor_tournament_neg_weight),
            controlled_factor_tournament_focus=str(args.controlled_factor_tournament_focus),
            controlled_factor_tournament_friction_weight=float(args.controlled_factor_tournament_friction_weight),
            controlled_factor_tournament_material_weight=float(args.controlled_factor_tournament_material_weight),
            controlled_factor_tournament_unevenness_weight=float(args.controlled_factor_tournament_unevenness_weight),
            mechanism_controlled_factor_tournament_weight=float(args.mechanism_controlled_factor_tournament_weight),
            mechanism_controlled_factor_tournament_temperature=float(args.mechanism_controlled_factor_tournament_temperature),
            mechanism_controlled_factor_tournament_margin=float(args.mechanism_controlled_factor_tournament_margin),
            mechanism_controlled_factor_tournament_neg_weight=float(args.mechanism_controlled_factor_tournament_neg_weight),
            mechanism_controlled_factor_tournament_focus=str(args.mechanism_controlled_factor_tournament_focus),
            mechanism_controlled_factor_tournament_friction_weight=float(args.mechanism_controlled_factor_tournament_friction_weight),
            mechanism_controlled_factor_tournament_material_weight=float(args.mechanism_controlled_factor_tournament_material_weight),
            mechanism_controlled_factor_tournament_unevenness_weight=float(args.mechanism_controlled_factor_tournament_unevenness_weight),
            local_factor_graph_weight=float(args.local_factor_graph_weight),
            local_factor_graph_margin=float(args.local_factor_graph_margin),
            local_factor_graph_uncertainty_margin=float(args.local_factor_graph_uncertainty_margin),
            local_factor_graph_gate_temperature=float(args.local_factor_graph_gate_temperature),
            local_factor_graph_topk=int(args.local_factor_graph_topk),
            relation_specific_clean_margin_weight=float(args.relation_specific_clean_margin_weight),
            relation_specific_clean_margin_relation_ids=relation_specific_clean_margin_relation_ids,
            relation_specific_clean_margin_class_weights=relation_specific_clean_margin_class_weights,
            relation_specific_clean_margin_protected_mask=relation_specific_clean_margin_protected_mask,
            relation_specific_clean_margin_friction_margin=float(
                args.relation_specific_clean_margin_friction_margin
            ),
            relation_specific_clean_margin_roughness_margin=float(
                args.relation_specific_clean_margin_roughness_margin
            ),
            relation_specific_clean_margin_material_margin=float(
                args.relation_specific_clean_margin_material_margin
            ),
            relation_specific_clean_margin_friction_weight=float(
                args.relation_specific_clean_margin_friction_weight
            ),
            relation_specific_clean_margin_roughness_weight=float(
                args.relation_specific_clean_margin_roughness_weight
            ),
            relation_specific_clean_margin_material_weight=float(
                args.relation_specific_clean_margin_material_weight
            ),
            relation_specific_clean_margin_protected_negative_scale=float(
                args.relation_specific_clean_margin_protected_negative_scale
            ),
            relation_specific_clean_margin_uncertainty_margin=float(
                args.relation_specific_clean_margin_uncertainty_margin
            ),
            relation_specific_clean_margin_gate_temperature=float(
                args.relation_specific_clean_margin_gate_temperature
            ),
            relation_specific_clean_margin_topk=int(args.relation_specific_clean_margin_topk),
            directed_confusion_weight=float(args.directed_confusion_weight),
            directed_confusion_weight_matrix=directed_confusion_weight_matrix,
            directed_confusion_margin=float(args.directed_confusion_margin),
            graph_angular_weight=float(args.graph_angular_weight),
            graph_angular_weight_matrix=graph_angular_weight_matrix,
            graph_angular_max_cosine=float(args.graph_angular_max_cosine),
            graph_diffusion_aux_weight=float(args.graph_diffusion_aux_weight),
            graph_diffusion_aux_targets=graph_diffusion_aux_targets,
            graph_diffusion_gate_mask=graph_diffusion_gate_mask,
            graph_diffusion_uncertainty_margin=float(args.graph_diffusion_uncertainty_margin),
            graph_diffusion_gate_temperature=float(args.graph_diffusion_gate_temperature),
            factor_neighbor_prior_utility_weight=float(args.factor_neighbor_prior_utility_weight),
            factor_neighbor_prior_bias=factor_neighbor_prior_bias,
            factor_neighbor_prior_neighbor_mask=factor_neighbor_negative_mask,
            factor_neighbor_prior_hard_mask=factor_neighbor_prior_hard_mask,
            factor_neighbor_prior_utility_lambda=float(args.factor_neighbor_prior_utility_lambda),
            factor_neighbor_prior_utility_topk=int(args.factor_neighbor_prior_utility_topk),
            factor_neighbor_prior_utility_margin=float(args.factor_neighbor_prior_utility_margin),
            factor_neighbor_prior_utility_mode=str(args.factor_neighbor_prior_utility_mode),
            physics_aux_weight=float(args.physics_aux_weight),
            physics_evidence_aux_weight=float(args.physics_evidence_aux_weight),
            physics_evidence_target_builder=physics_evidence_target_builder,
            physics_evidence_aux_scope=str(args.physics_evidence_aux_scope),
            physics_evidence_aux_field_mode=str(args.physics_evidence_aux_field_mode),
            hierarchical_targets=hierarchical_targets,
            class_weights=hard_condition_weights,
            teacher_probs=teacher_probs,
            distill_weight=float(args.distill_weight),
            distill_factor_weight=float(args.distill_factor_weight),
            distill_temperature=float(args.distill_temperature),
            positive_congruent_weight=float(args.positive_congruent_weight),
            positive_congruent_beta=float(args.positive_congruent_beta),
            positive_congruent_min_confidence=float(args.positive_congruent_min_confidence),
            distill_missing_policy=str(args.distill_missing_policy),
            online_teacher_model=online_teacher_model,
            online_teacher_weight=float(args.online_teacher_weight),
            online_teacher_temperature=float(args.online_teacher_temperature),
            online_teacher_beta=float(args.online_teacher_beta),
            online_teacher_min_confidence=float(args.online_teacher_min_confidence),
            teacher_error_replay_weight=float(args.teacher_error_replay_weight),
            teacher_error_replay_class_mask=teacher_error_replay_class_mask,
            teacher_error_replay_beta=float(args.teacher_error_replay_beta),
            teacher_error_replay_min_confidence=float(args.teacher_error_replay_min_confidence),
            hflip_consistency_weight=float(args.hflip_consistency_weight),
            hflip_consistency_temperature=float(args.hflip_consistency_temperature),
            hflip_physics_feature_consistency_weight=float(args.hflip_physics_feature_consistency_weight),
            hflip_local_physics_feature_consistency_weight=float(args.hflip_local_physics_feature_consistency_weight),
            hflip_low_level_feature_consistency_weight=float(args.hflip_low_level_feature_consistency_weight),
            masked_consistency_weight=float(args.masked_consistency_weight),
            masked_factor_consistency_weight=float(args.masked_factor_consistency_weight),
            masked_consistency_temperature=float(args.masked_consistency_temperature),
            masked_consistency_mode=str(args.masked_consistency_mode),
            masked_consistency_ratio=float(args.masked_consistency_ratio),
            masked_consistency_block_frac=float(args.masked_consistency_block_frac),
            masked_consistency_max_blocks=int(args.masked_consistency_max_blocks),
            masked_consistency_value=str(args.masked_consistency_value),
            masked_consistency_confidence_threshold=float(args.masked_consistency_confidence_threshold),
            observer_hinf_weight=float(args.observer_hinf_weight),
            observer_hinf_mode=str(args.observer_hinf_mode),
            observer_hinf_scope=str(args.observer_hinf_scope),
            observer_hinf_strength=float(args.observer_hinf_strength),
            observer_hinf_rho=float(args.observer_hinf_rho),
            observer_hinf_max_lines=int(args.observer_hinf_max_lines),
            observer_hinf_block_ratio=float(args.observer_hinf_block_ratio),
            observer_hinf_temperature=float(args.observer_hinf_temperature),
            observer_hinf_confidence_threshold=float(args.observer_hinf_confidence_threshold),
            observer_hinf_feature_weight=float(args.observer_hinf_feature_weight),
            observer_hinf_physics_feature_weight=float(args.observer_hinf_physics_feature_weight),
            observer_hinf_local_physics_feature_weight=float(args.observer_hinf_local_physics_feature_weight),
            observer_hinf_low_level_feature_weight=float(args.observer_hinf_low_level_feature_weight),
            observer_hinf_class_consistency_weight=float(args.observer_hinf_class_consistency_weight),
            observer_hinf_factor_consistency_weight=float(args.observer_hinf_factor_consistency_weight),
            observer_hinf_disturbed_ce_weight=float(args.observer_hinf_disturbed_ce_weight),
            observer_hinf_barrier_weight=float(args.observer_hinf_barrier_weight),
            observer_hinf_barrier_scope=str(args.observer_hinf_barrier_scope),
            observer_hinf_barrier_margin_drop=float(args.observer_hinf_barrier_margin_drop),
            backbone_aux_weight=float(args.backbone_aux_weight),
        )
        val_metrics = evaluate(model, val_loader, device, criterion, idx_to_class=idx_to_class)
        val_summary = val_metrics["summary"]
        print(f"  train: loss={train_metrics['loss']:.4f} acc={train_metrics['top1']:.4f}", flush=True)
        print(
            "  val  : loss={loss:.4f} top1={top1:.4f} macro_f1={macro_f1:.4f} bal_acc={bal:.4f}".format(
                loss=val_summary["loss"],
                top1=val_summary["top1"],
                macro_f1=val_summary["macro_f1"],
                bal=val_summary["balanced_accuracy"],
            ),
            flush=True,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_summary,
        }
        hard_slice_stats = hard_slice_f1_stats(val_metrics, checkpoint_hard_slice_classes)
        selection_key = checkpoint_selection_key(
            val_metrics,
            metric=str(args.checkpoint_selection_metric),
            best_macro_seen=float(best_macro_seen),
            best_hard_slice_seen=float(best_hard_slice_seen),
            macro_tolerance=float(args.checkpoint_selection_macro_tolerance),
            macro_floor=float(args.checkpoint_selection_macro_floor),
            hard_slice_classes=checkpoint_hard_slice_classes,
            hard_slice_tolerance=float(args.checkpoint_selection_hard_slice_tolerance),
            hard_slice_floor=float(args.checkpoint_selection_hard_slice_floor),
        )
        best_macro_seen = max(float(best_macro_seen), float(val_summary["macro_f1"]))
        best_hard_slice_seen = max(float(best_hard_slice_seen), float(hard_slice_stats["min_f1"]))
        row["checkpoint_selection"] = {
            "metric": str(args.checkpoint_selection_metric),
            "key": [float(item) for item in selection_key],
            "best_key_before_epoch": [float(item) for item in best_selection_key],
            "best_macro_seen": float(best_macro_seen),
            "best_hard_slice_seen": float(best_hard_slice_seen),
            "hard_slice": hard_slice_stats,
        }
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "class_to_idx": class_to_idx,
                "args": vars(args),
                "val_summary": val_summary,
            },
            args.output_dir / "last.pt",
        )
        if selection_key > best_selection_key:
            best_selection_key = selection_key
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "class_to_idx": class_to_idx,
                    "args": vars(args),
                    "val_summary": val_summary,
                    "checkpoint_selection": {
                        "metric": str(args.checkpoint_selection_metric),
                        "key": [float(item) for item in selection_key],
                        "best_macro_seen": float(best_macro_seen),
                        "best_hard_slice_seen": float(best_hard_slice_seen),
                        "hard_slice": hard_slice_stats,
                    },
                },
                args.output_dir / "best.pt",
            )
            print(
                f"  saved best checkpoint ({args.checkpoint_selection_metric} key={selection_key}): "
                f"{args.output_dir / 'best.pt'}",
                flush=True,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.early_stop_patience):
                print(
                    f"  early stopping: no {args.checkpoint_selection_metric} improvement "
                    f"for {stale_epochs} epochs",
                    flush=True,
                )
                break

    state = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    predictions_path = args.output_dir / "predictions_test.csv" if args.save_predictions else None
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        criterion,
        idx_to_class=idx_to_class,
        save_predictions_path=predictions_path,
        save_probabilities_path=args.save_probabilities_npz,
    )
    write_eval_outputs(args.output_dir, test_metrics, split="test")
    print(json.dumps(test_metrics["summary"], indent=2, ensure_ascii=False), flush=True)


def build_class_map(manifests: list[Path]) -> dict[str, int]:
    labels: set[str] = set()
    for manifest in manifests:
        df = pd.read_csv(manifest, usecols=["class_label"], dtype=str, low_memory=False)
        labels.update(canonical_class_label(v) for v in df["class_label"].dropna().astype(str).unique().tolist())
    return {name: idx for idx, name in enumerate(sorted(labels))}


def build_loader(ds: RSCDSurfaceDataset, args: argparse.Namespace, data_cfg: dict[str, Any], *, train: bool) -> DataLoader:
    sampler = None
    shuffle = train
    if train and bool(args.controlled_factor_tournament_sampling):
        num_samples = int(args.samples_per_epoch) if int(args.samples_per_epoch) > 0 else len(ds)
        sampler = RSCDControlledFactorTournamentSampler(
            ds,
            num_samples=num_samples,
            seed=int(args.seed),
            rectangle_fraction=float(args.controlled_factor_tournament_sampling_fraction),
        )
        shuffle = False
    elif train and bool(args.hard_pair_sampling):
        num_samples = int(args.samples_per_epoch) if int(args.samples_per_epoch) > 0 else len(ds)
        sampler = RSCDHardPairSampler(
            ds,
            num_samples=num_samples,
            seed=int(args.seed),
            pair_fraction=float(args.hard_pair_sampling_fraction),
        )
        shuffle = False
    elif train and bool(args.balanced_sampling):
        weights = balanced_weights(ds.df)
        num_samples = int(args.samples_per_epoch) if int(args.samples_per_epoch) > 0 else len(ds)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)
        shuffle = False
    num_workers, loader_kwargs = dataloader_worker_settings(data_cfg)
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        **loader_kwargs,
    )


def balanced_weights(df: pd.DataFrame) -> list[float]:
    group_col = "class_label_canonical" if "class_label_canonical" in df.columns else "class_label"
    sizes = df.groupby(group_col)[group_col].transform("size").astype(float)
    return (1.0 / sizes.clip(lower=1.0)).tolist()


def rscd_hard_boundary_pairs() -> list[tuple[str, str]]:
    """Audited reciprocal RSCD boundaries from the current complete graph."""

    return [
        ("dry_concrete_slight", "dry_concrete_severe"),
        ("water_concrete_smooth", "wet_concrete_smooth"),
        ("dry_mud", "dry_gravel"),
        ("wet_mud", "wet_gravel"),
        ("water_concrete_slight", "water_concrete_severe"),
        ("dry_concrete_slight", "dry_concrete_smooth"),
        ("dry_asphalt_slight", "dry_asphalt_severe"),
        ("wet_concrete_severe", "wet_concrete_slight"),
        ("wet_asphalt_slight", "wet_asphalt_smooth"),
        ("dry_asphalt_smooth", "wet_asphalt_smooth"),
        ("dry_asphalt_slight", "dry_asphalt_smooth"),
        ("wet_mud", "water_mud"),
        ("dry_concrete_smooth", "wet_concrete_smooth"),
        ("dry_asphalt_slight", "wet_asphalt_slight"),
        ("wet_asphalt_severe", "wet_asphalt_slight"),
        ("wet_asphalt_slight", "water_asphalt_slight"),
        ("water_asphalt_severe", "water_asphalt_slight"),
        ("water_asphalt_smooth", "water_asphalt_slight"),
        ("water_mud", "water_gravel"),
    ]


def hard_pair_relation(left: str, right: str) -> tuple[str, ...]:
    left_factors = parse_rscd_factors(left)
    right_factors = parse_rscd_factors(right)
    return tuple(
        name
        for name in FACTOR_LABELS
        if int(left_factors[name]) != int(right_factors[name])
    )


def include_hard_pair_for_focus(left: str, right: str, focus: str) -> bool:
    focus = str(focus)
    if focus == "all":
        return True
    relation = hard_pair_relation(left, right)
    if focus == "roughness":
        return relation == ("unevenness",)
    if focus == "material":
        return relation == ("material",)
    if focus == "concrete":
        return "_concrete_" in f"_{left}_" or "_concrete_" in f"_{right}_"
    if focus == "wet_water":
        if relation != ("friction",):
            return False
        left_friction = FACTOR_LABELS["friction"][parse_rscd_factors(left)["friction"]]
        right_friction = FACTOR_LABELS["friction"][parse_rscd_factors(right)["friction"]]
        return {left_friction, right_friction} == {"wet", "water"}
    raise ValueError(f"unknown hard pair auxiliary focus: {focus}")


def build_hard_pair_aux_pairs(
    class_to_idx: dict[str, int],
    *,
    focus: str,
) -> list[tuple[int, int, str, str]]:
    pairs: list[tuple[int, int, str, str]] = []
    for left, right in rscd_hard_boundary_pairs():
        if left not in class_to_idx or right not in class_to_idx:
            continue
        if not include_hard_pair_for_focus(left, right, focus):
            continue
        pairs.append((int(class_to_idx[left]), int(class_to_idx[right]), left, right))
    return pairs


def hard_pair_auxiliary_loss(
    pair_logits: torch.Tensor,
    label: torch.Tensor,
    pair_left: torch.Tensor,
    pair_right: torch.Tensor,
) -> torch.Tensor | None:
    """Binary boundary supervision on only the samples belonging to each hard pair."""

    if pair_logits.numel() == 0 or pair_left.numel() == 0 or pair_right.numel() == 0:
        return None
    pair_left = pair_left.to(device=label.device)
    pair_right = pair_right.to(device=label.device)
    is_left = label.unsqueeze(1).eq(pair_left.unsqueeze(0))
    is_right = label.unsqueeze(1).eq(pair_right.unsqueeze(0))
    valid = is_left | is_right
    if not bool(valid.any()):
        return None
    targets = is_right.to(dtype=pair_logits.dtype)
    return F.binary_cross_entropy_with_logits(pair_logits[valid], targets[valid])


def load_state_dict_allow_expanded_head(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    """Load matching checkpoint tensors and copy overlapping head dimensions.

    Low-level evidence branches change the classifier input dimension. PyTorch's
    non-strict loading still rejects shape mismatches, so this helper preserves
    the old head weights in their overlapping columns and leaves new evidence
    dimensions initialized by the current model.
    """

    current = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    partial: list[str] = []
    skipped: list[str] = []
    state_dict = dict(state_dict)
    if any(str(name).startswith("stem.") for name in state_dict):
        remapped = dict(state_dict)
        for name, value in state_dict.items():
            name_text = str(name)
            if name_text.startswith(("stem.", "stages.", "norm.")):
                remapped[f"backbone.model.{name_text}"] = value
            elif name_text == "head.weight":
                remapped["classifier.weight"] = value
            elif name_text == "head.bias":
                remapped["classifier.bias"] = value
        state_dict = remapped
    compatible_aliases = {
        "backbone.global_proj.weight": "backbone.proj.weight",
        "backbone.global_proj.bias": "backbone.proj.bias",
    }
    for new_name, old_name in compatible_aliases.items():
        if new_name in current and new_name not in state_dict and old_name in state_dict:
            old_value = state_dict[old_name]
            if tuple(old_value.shape) == tuple(current[new_name].shape):
                state_dict[new_name] = old_value
                partial.append(f"alias:{old_name}->{new_name}")
    for name, value in state_dict.items():
        if name not in current:
            skipped.append(name)
            continue
        target = current[name]
        if tuple(value.shape) == tuple(target.shape):
            loadable[name] = value
            continue
        if (
            name == "backbone.features.0.0.weight"
            and value.ndim == target.ndim == 4
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
            and value.shape[1] <= target.shape[1]
        ):
            merged = target.clone()
            merged[:, : value.shape[1]].copy_(value.to(dtype=target.dtype))
            if target.shape[1] > value.shape[1]:
                merged[:, value.shape[1] :] = 0.0
            loadable[name] = merged
            partial.append(name)
            continue
        if name in {"norm.weight", "norm.bias"} and value.ndim == target.ndim == 1:
            merged = target.clone()
            n = min(int(value.shape[0]), int(target.shape[0]))
            merged[:n] = value[:n].to(dtype=target.dtype)
            loadable[name] = merged
            partial.append(name)
            continue
        if name == "classifier.weight" and value.ndim == target.ndim == 2 and value.shape[0] == target.shape[0]:
            merged = target.clone()
            n = min(int(value.shape[1]), int(target.shape[1]))
            merged[:, :n] = value[:, :n].to(dtype=target.dtype)
            if target.shape[1] > value.shape[1]:
                merged[:, n:] = 0.0
            loadable[name] = merged
            partial.append(name)
            continue
        skipped.append(name)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return list(missing), list(unexpected), partial + [f"skipped:{name}" for name in skipped]


def extract_checkpoint_model_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state_dict", "state_dict"):
        value = state.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(value, torch.Tensor) for value in state.values()):
        return state
    raise KeyError("checkpoint must contain one of: model, model_state_dict, state_dict.")


def canonical_class_label(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def parse_factorized_factors(value: str) -> tuple[str, ...]:
    factors = tuple(part.strip().lower() for part in str(value).split(",") if part.strip())
    if not factors:
        raise ValueError("--factorized-factors must contain at least one factor name.")
    invalid = sorted(set(factors).difference(FACTOR_LABELS))
    if invalid:
        raise ValueError(f"unknown --factorized-factors entries: {invalid}")
    return factors


def parse_checkpoint_guard_classes(value: str | None) -> tuple[str, ...]:
    labels = tuple(
        canonical_class_label(part)
        for part in str(value or "").split(",")
        if str(part).strip()
    )
    return labels or DEFAULT_CHECKPOINT_HARD_SLICE_CLASSES


def parse_rscd_factors(class_label: str) -> dict[str, int]:
    label = canonical_class_label(class_label)
    parts = label.split("_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        friction = label
        material = None
        unevenness = None
    else:
        friction = parts[0] if len(parts) >= 1 else None
        material = parts[1] if len(parts) >= 2 else None
        unevenness = parts[2] if len(parts) >= 3 else None
    values = {
        "friction": friction,
        "material": material,
        "unevenness": unevenness,
    }
    out = {}
    for name, labels in FACTOR_LABELS.items():
        value = values[name]
        out[name] = labels.index(value) if value in labels else -1
    return out


def hard_slice_f1_stats(metrics: dict[str, Any], hard_classes: tuple[str, ...]) -> dict[str, Any]:
    report = metrics.get("classification_report", {}) if isinstance(metrics, dict) else {}
    scores: list[float] = []
    present: list[str] = []
    missing: list[str] = []
    for label in hard_classes:
        item = report.get(label)
        if isinstance(item, dict) and "f1-score" in item:
            scores.append(float(item.get("f1-score", 0.0)))
            present.append(str(label))
        else:
            missing.append(str(label))
    if scores:
        return {
            "mean_f1": float(np.mean(scores)),
            "min_f1": float(np.min(scores)),
            "present": present,
            "missing": missing,
        }
    return {"mean_f1": 0.0, "min_f1": 0.0, "present": present, "missing": missing}


def checkpoint_selection_key(
    metrics: dict[str, Any],
    *,
    metric: str,
    best_macro_seen: float,
    best_hard_slice_seen: float,
    macro_tolerance: float,
    macro_floor: float,
    hard_slice_classes: tuple[str, ...],
    hard_slice_tolerance: float,
    hard_slice_floor: float,
) -> tuple[float, ...]:
    """Build a comparable key for validation checkpoint selection."""

    metric = str(metric)
    summary = metrics.get("summary", metrics)
    macro_f1 = float(summary.get("macro_f1", 0.0))
    top1 = float(summary.get("top1", 0.0))
    weighted_f1 = float(summary.get("weighted_f1", 0.0))
    balanced_accuracy = float(summary.get("balanced_accuracy", 0.0))
    hard_stats = hard_slice_f1_stats(metrics, hard_slice_classes)
    hard_mean = float(hard_stats["mean_f1"])
    hard_min = float(hard_stats["min_f1"])
    if metric == "macro_f1":
        return (macro_f1, top1)
    if metric == "top1":
        return (top1, macro_f1)
    if metric == "weighted_f1":
        return (weighted_f1, macro_f1, top1)
    if metric == "balanced_accuracy":
        return (balanced_accuracy, macro_f1, top1)
    if metric == "top1_macro_guard":
        if math.isfinite(float(best_macro_seen)):
            dynamic_floor = float(best_macro_seen) - max(float(macro_tolerance), 0.0)
        else:
            dynamic_floor = -math.inf
        guard_floor = max(float(macro_floor), dynamic_floor)
        guard_pass = 1.0 if macro_f1 >= guard_floor else 0.0
        return (guard_pass, top1, macro_f1)
    if metric in {"top1_hardslice_guard", "top1_macro_hardslice_guard"}:
        if math.isfinite(float(best_hard_slice_seen)):
            hard_dynamic_floor = float(best_hard_slice_seen) - max(float(hard_slice_tolerance), 0.0)
        else:
            hard_dynamic_floor = -math.inf
        hard_guard_floor = max(float(hard_slice_floor), hard_dynamic_floor)
        hard_pass = 1.0 if hard_min >= hard_guard_floor else 0.0
        if metric == "top1_hardslice_guard":
            return (hard_pass, top1, hard_mean, hard_min, macro_f1)
        if math.isfinite(float(best_macro_seen)):
            macro_dynamic_floor = float(best_macro_seen) - max(float(macro_tolerance), 0.0)
        else:
            macro_dynamic_floor = -math.inf
        macro_guard_floor = max(float(macro_floor), macro_dynamic_floor)
        macro_pass = 1.0 if macro_f1 >= macro_guard_floor else 0.0
        return (macro_pass, hard_pass, top1, macro_f1, hard_mean, hard_min)
    raise ValueError(f"unknown checkpoint selection metric: {metric}")


def include_class_for_mechanism_scope(class_label: str, scope: str) -> bool:
    """Select RSCD labels for mechanism-curriculum training phases."""

    label = canonical_class_label(class_label)
    scope = str(scope)
    if scope == "all":
        return True
    factors = _factor_text(label)
    friction = factors["friction"]
    material = factors["material"]
    unevenness = factors["unevenness"]
    is_paved_core = (
        friction in {"dry", "wet", "water"}
        and material in {"asphalt", "concrete"}
        and unevenness in {"smooth", "slight", "severe"}
    )
    if scope == "core_paved":
        return bool(is_paved_core)
    if scope == "dry_visible":
        return friction == "dry"
    if scope == "dry_paved_roughness":
        return friction == "dry" and material in {"asphalt", "concrete"} and unevenness in {"smooth", "slight", "severe"}
    if scope == "wet_water_paved":
        return friction in {"wet", "water"} and material in {"asphalt", "concrete"} and unevenness in {"smooth", "slight", "severe"}
    if scope == "wet_water_concrete":
        return friction in {"wet", "water"} and material == "concrete" and unevenness in {"smooth", "slight", "severe"}
    if scope == "granular":
        return material in {"mud", "gravel"}
    if scope == "winter":
        return friction in {"fresh_snow", "melted_snow", "ice"}
    if scope == "hard_audited":
        hard_classes = {
            "water_concrete_slight",
            "water_asphalt_slight",
            "wet_concrete_slight",
            "water_concrete_severe",
            "dry_concrete_slight",
            "wet_concrete_severe",
            "water_asphalt_severe",
            "water_gravel",
        }
        return label in hard_classes
    raise ValueError(f"unknown mechanism training scope: {scope}")


def _append_pair_index(
    pair_indices: dict[str, list[int]],
    pair_masks: dict[str, list[float]],
    name: str,
    left_idx: int,
    right_idx: int,
    right_size: int,
) -> None:
    if left_idx < 0 or right_idx < 0:
        pair_indices[name].append(0)
        pair_masks[name].append(0.0)
    else:
        pair_indices[name].append(int(left_idx) * int(right_size) + int(right_idx))
        pair_masks[name].append(1.0)


def build_class_factor_buffers(class_to_idx: dict[str, int]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in FACTOR_LABELS:
        indices = []
        mask = []
        for class_idx in range(len(idx_to_class)):
            factor_idx = parse_rscd_factors(idx_to_class[class_idx])[name]
            if factor_idx < 0:
                indices.append(0)
                mask.append(0.0)
            else:
                indices.append(int(factor_idx))
                mask.append(1.0)
        out[name] = (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )
    return out


def build_label_graph_buffers(class_to_idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build node descriptors and normalized adjacency for the RSCD class graph."""

    num_classes = len(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    feature_dim = sum(len(labels) for labels in FACTOR_LABELS.values()) + 3
    node_features = torch.zeros((num_classes, feature_dim), dtype=torch.float32)
    adjacency = torch.eye(num_classes, dtype=torch.float32)
    neighbor_mask = torch.eye(num_classes, dtype=torch.bool)
    hard_classes = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "dry_concrete_slight",
        "wet_concrete_severe",
    }
    offsets: dict[str, int] = {}
    cursor = 0
    for name, labels in FACTOR_LABELS.items():
        offsets[name] = cursor
        cursor += len(labels)

    factor_text = {idx: _factor_text(idx_to_class[idx]) for idx in range(num_classes)}
    for idx in range(num_classes):
        class_name = idx_to_class[idx]
        parsed = parse_rscd_factors(class_name)
        for name, labels in FACTOR_LABELS.items():
            factor_idx = int(parsed[name])
            if factor_idx >= 0:
                node_features[idx, offsets[name] + factor_idx] = 1.0
        text = factor_text[idx]
        is_core = (
            text["friction"] in {"dry", "wet", "water"}
            and text["material"] in {"asphalt", "concrete"}
            and text["unevenness"] in {"smooth", "slight", "severe"}
        )
        node_features[idx, cursor] = 1.0 if is_core else 0.0
        node_features[idx, cursor + 1] = 1.0 if class_name in hard_classes else 0.0
        node_features[idx, cursor + 2] = 1.0 if text["friction"] in {"wet", "water"} else 0.0

    for i in range(num_classes):
        a = factor_text[i]
        for j in range(i + 1, num_classes):
            b = factor_text[j]
            edge_weight = 0.0
            same_friction = a["friction"] is not None and a["friction"] == b["friction"]
            same_material = a["material"] is not None and a["material"] == b["material"]
            same_uneven = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
            if same_friction:
                edge_weight += 0.35
            elif _friction_neighbors(a["friction"], b["friction"]):
                edge_weight += 0.55
            if same_material:
                edge_weight += 0.20
            elif _material_neighbors(a["material"], b["material"]):
                edge_weight += 0.15
            if same_uneven:
                edge_weight += 0.20
            elif _unevenness_neighbors(a["unevenness"], b["unevenness"]):
                edge_weight += 0.35
            if edge_weight >= 0.35:
                adjacency[i, j] = edge_weight
                adjacency[j, i] = edge_weight
                neighbor_mask[i, j] = True
                neighbor_mask[j, i] = True

    degree = adjacency.sum(dim=1).clamp_min(1e-6)
    norm = degree.rsqrt().view(-1, 1) * adjacency * degree.rsqrt().view(1, -1)
    return node_features, norm, neighbor_mask


def build_hierarchical_targets(class_to_idx: dict[str, int], smoothing: float) -> torch.Tensor:
    """Build factor-aware soft labels for RSCD composite classes.

    The target remains mostly one-hot. The small smoothing mass is assigned to
    classes with related physical factors, especially same material/unevenness
    and neighboring friction states such as wet-water. This targets the label
    subjectivity noted by RSCD follow-up papers without changing the benchmark
    label space.
    """

    num_classes = len(class_to_idx)
    alpha = max(0.0, min(float(smoothing), 0.4))
    target = torch.eye(num_classes, dtype=torch.float32)
    if alpha <= 0.0:
        return target

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    factors = {idx: _factor_text(idx_to_class[idx]) for idx in range(num_classes)}
    for i in range(num_classes):
        scores = torch.zeros(num_classes, dtype=torch.float32)
        for j in range(num_classes):
            if i == j:
                continue
            scores[j] = _class_relation_score(factors[i], factors[j])
        if float(scores.sum()) <= 0.0:
            continue
        distribution = scores / scores.sum().clamp_min(1e-6)
        target[i] = (1.0 - alpha) * target[i] + alpha * distribution
    return target


def build_roughness_neighbor_targets(class_to_idx: dict[str, int], smoothing: float) -> torch.Tensor:
    """Build narrow soft labels for adjacent RSCD roughness states.

    Only classes with identical friction and material and neighboring
    smooth/slight/severe labels receive smoothing mass. Winter, mud/gravel, and
    cross-wetness relations stay untouched.
    """

    num_classes = len(class_to_idx)
    alpha = max(0.0, min(float(smoothing), 0.20))
    target = torch.eye(num_classes, dtype=torch.float32)
    if alpha <= 0.0:
        return target
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    factors = {idx: _factor_text(idx_to_class[idx]) for idx in range(num_classes)}
    for i in range(num_classes):
        a = factors[i]
        if a["friction"] not in {"dry", "wet", "water"}:
            continue
        if a["material"] not in {"asphalt", "concrete"}:
            continue
        if a["unevenness"] not in {"smooth", "slight", "severe"}:
            continue
        neighbors = []
        for j in range(num_classes):
            if i == j:
                continue
            b = factors[j]
            if a["friction"] != b["friction"] or a["material"] != b["material"]:
                continue
            if _unevenness_neighbors(a["unevenness"], b["unevenness"]):
                neighbors.append(j)
        if not neighbors:
            continue
        target[i, i] = 1.0 - alpha
        share = alpha / float(len(neighbors))
        for j in neighbors:
            target[i, j] = share
    return target


def build_graph_diffusion_targets(
    class_to_idx: dict[str, int],
    smoothing: float,
    *,
    temperature: float = 0.35,
    core_only: bool = True,
) -> torch.Tensor:
    """Build conservative heat-kernel soft labels on the RSCD class graph.

    This is a training-only graph regularizer: the true class keeps most target
    mass, while a small fraction diffuses only to physically adjacent RSCD
    labels. The default core-only mode avoids softening gravel, mud, and winter
    classes where the current model is already strong.
    """

    num_classes = len(class_to_idx)
    alpha = max(0.0, min(float(smoothing), 0.20))
    target = torch.eye(num_classes, dtype=torch.float32)
    if alpha <= 0.0:
        return target

    _, adjacency, neighbor_mask = build_label_graph_buffers(class_to_idx)
    adjacency = adjacency.clamp_min(0.0)
    adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-6)
    identity = torch.eye(num_classes, dtype=torch.float32)
    laplacian_generator = adjacency - identity
    heat = torch.linalg.matrix_exp(max(float(temperature), 1e-4) * laplacian_generator).clamp_min(0.0)
    heat = heat * neighbor_mask.to(dtype=heat.dtype)
    heat.fill_diagonal_(0.0)

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    for idx in range(num_classes):
        factors = _factor_text(idx_to_class[idx])
        is_core = (
            factors["friction"] in {"dry", "wet", "water"}
            and factors["material"] in {"asphalt", "concrete"}
            and factors["unevenness"] in {"smooth", "slight", "severe"}
        )
        if bool(core_only) and not is_core:
            continue
        row = heat[idx]
        row_sum = row.sum().clamp_min(1e-6)
        if float(row_sum) <= 1e-6:
            continue
        target[idx] = (1.0 - alpha) * identity[idx] + alpha * (row / row_sum)
    return target


def build_graph_diffusion_gate_mask(class_to_idx: dict[str, int], *, hard_only: bool = True) -> torch.Tensor:
    """Mask rows eligible for uncertainty-gated graph diffusion."""

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    hard_classes = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "dry_concrete_slight",
        "wet_concrete_severe",
    }
    mask = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for idx, class_name in idx_to_class.items():
        factors = _factor_text(class_name)
        is_core = (
            factors["friction"] in {"dry", "wet", "water"}
            and factors["material"] in {"asphalt", "concrete"}
            and factors["unevenness"] in {"smooth", "slight", "severe"}
        )
        if (bool(hard_only) and class_name in hard_classes) or (not bool(hard_only) and is_core):
            mask[idx] = 1.0
    return mask


def _factor_text(class_label: str) -> dict[str, str | None]:
    label = canonical_class_label(class_label)
    parts = label.split("_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return {"friction": label, "material": None, "unevenness": None}
    return {
        "friction": parts[0] if len(parts) >= 1 else None,
        "material": parts[1] if len(parts) >= 2 else None,
        "unevenness": parts[2] if len(parts) >= 3 else None,
    }


def _class_relation_score(a: dict[str, str | None], b: dict[str, str | None]) -> float:
    score = 0.0
    if a["friction"] and a["friction"] == b["friction"]:
        score += 0.45
    elif _friction_neighbors(a["friction"], b["friction"]):
        score += 0.25
    if a["material"] and a["material"] == b["material"]:
        score += 0.20
    if a["unevenness"] and a["unevenness"] == b["unevenness"]:
        score += 0.15
    if score < 0.25:
        return 0.0
    return score


def _friction_neighbors(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    neighbor_pairs = {
        frozenset(("dry", "wet")),
        frozenset(("wet", "water")),
        frozenset(("fresh_snow", "melted_snow")),
        frozenset(("melted_snow", "ice")),
    }
    return frozenset((a, b)) in neighbor_pairs


def _unevenness_neighbors(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    neighbor_pairs = {
        frozenset(("smooth", "slight")),
        frozenset(("slight", "severe")),
    }
    return frozenset((a, b)) in neighbor_pairs


def _material_neighbors(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    neighbor_pairs = {
        frozenset(("asphalt", "concrete")),
        frozenset(("mud", "gravel")),
    }
    return frozenset((a, b)) in neighbor_pairs


def build_hard_condition_weights(class_to_idx: dict[str, int], boost: float) -> torch.Tensor:
    """Class weights for the current RSCD wet-concrete failure mode.

    Fast class-slice audits show the weakest classes are concentrated in
    wet/water concrete with slight/severe unevenness. The weights are only a
    training loss prior; evaluation remains the standard unweighted RSCD metric.
    """

    weights = torch.ones(len(class_to_idx), dtype=torch.float32)
    beta = max(0.0, float(boost))
    if beta <= 0.0:
        return weights
    for name, idx in class_to_idx.items():
        factors = _factor_text(name)
        friction = factors["friction"]
        material = factors["material"]
        unevenness = factors["unevenness"]
        extra = 0.0
        if friction == "water":
            extra += 0.50 * beta
        if friction in {"wet", "water"} and material == "concrete":
            extra += beta
        if friction in {"wet", "water"} and material == "concrete" and unevenness in {"slight", "severe"}:
            extra += 0.25 * beta
        weights[int(idx)] += extra
    return weights / weights.mean().clamp_min(1e-6)


def build_top1_gap_weights(class_to_idx: dict[str, int], boost: float) -> torch.Tensor:
    """Class weights for the sample-weighted Top-1 versus Mean-F1 gap.

    The current model is macro-friendly, but full-test audits show that its
    sample-weighted errors concentrate in high-support concrete and paved
    roughness cells. This is a small RSCD-specific training prior over those
    coupled factor-neighbor cells, not a global natural-frequency prior.
    """

    weights = torch.ones(len(class_to_idx), dtype=torch.float32)
    beta = max(0.0, float(boost))
    if beta <= 0.0:
        return weights
    top1_error_cells = {
        "dry_concrete_slight": 1.00,
        "dry_concrete_severe": 0.80,
        "wet_concrete_smooth": 0.75,
        "water_concrete_smooth": 0.75,
        "wet_asphalt_slight": 0.55,
        "dry_asphalt_slight": 0.55,
        "water_concrete_slight": 0.50,
        "wet_concrete_slight": 0.45,
        "water_concrete_severe": 0.45,
        "wet_concrete_severe": 0.35,
    }
    for name, idx in class_to_idx.items():
        factors = _factor_text(name)
        friction = factors["friction"]
        material = factors["material"]
        unevenness = factors["unevenness"]
        extra = float(top1_error_cells.get(name, 0.0))
        if material == "concrete" and friction in {"dry", "wet", "water"}:
            extra += 0.20
        if material in {"asphalt", "concrete"} and unevenness in {"slight", "severe"}:
            extra += 0.15
        if material == "concrete" and unevenness in {"slight", "severe"}:
            extra += 0.15
        weights[int(idx)] += beta * extra
    return weights / weights.mean().clamp_min(1e-6)


def build_closed_loop_class_controller_weights(
    class_to_idx: dict[str, int],
    *,
    reference_json: str | Path | None,
    strength: float,
    target_f1: float,
    factor_strength: float,
    min_gain: float,
    max_gain: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Validation-feedback class weights for a one-step gain-scheduled controller.

    This is a training-only control law. A prior validation report is treated as
    the measured state, per-class F1 deficits are the tracking errors, and RSCD
    factor-level deficits provide shared friction/material/roughness feedback.
    The gain is clipped to avoid over-correcting fragile water/asphalt/granular
    cells when one local factor improves at another factor's expense.
    """

    strength = max(float(strength), 0.0)
    if strength <= 0.0:
        return None
    if reference_json is None:
        raise ValueError("--closed-loop-class-controller-reference-json is required when controller strength > 0.")
    ref_path = Path(reference_json)
    if not ref_path.exists():
        raise FileNotFoundError(f"--closed-loop-class-controller-reference-json does not exist: {ref_path}")
    report = json.loads(ref_path.read_text(encoding="utf-8"))
    class_report = report.get("classification_report", {})
    f1_by_class: dict[str, float] = {}
    for class_name, metrics in class_report.items():
        canonical = canonical_class_label(str(class_name))
        if canonical not in class_to_idx:
            continue
        if isinstance(metrics, dict) and "f1-score" in metrics:
            f1_by_class[canonical] = float(metrics["f1-score"])

    if not f1_by_class:
        raise ValueError(f"No per-class F1 rows found in controller reference: {ref_path}")

    target = max(1e-4, min(float(target_f1), 0.999))
    denom = max(1.0 - target, 1e-4)
    factor_strength = max(float(factor_strength), 0.0)
    min_gain = max(float(min_gain), 0.01)
    max_gain = max(float(max_gain), min_gain)

    factor_deficits: dict[tuple[str, str], list[float]] = {}
    class_deficit: dict[str, float] = {}
    for class_name in class_to_idx:
        canonical = canonical_class_label(class_name)
        observed = f1_by_class.get(canonical, target)
        deficit = max(0.0, target - float(observed))
        class_deficit[canonical] = deficit
        factors = _factor_text(canonical)
        for axis, value in factors.items():
            if value is None:
                continue
            factor_deficits.setdefault((axis, value), []).append(deficit)
    factor_mean = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in factor_deficits.items()
    }

    weights = torch.ones(len(class_to_idx), dtype=torch.float32)
    for class_name, idx in class_to_idx.items():
        canonical = canonical_class_label(class_name)
        factors = _factor_text(canonical)
        shared = []
        for axis, value in factors.items():
            if value is not None:
                shared.append(factor_mean.get((axis, value), 0.0))
        factor_error = float(np.mean(shared)) if shared else 0.0
        direct_error = class_deficit.get(canonical, 0.0)
        normalized_error = (direct_error + factor_strength * factor_error) / (1.0 + factor_strength)
        gain = 1.0 + strength * min(1.0, normalized_error / denom)
        weights[int(idx)] = float(min(max(gain, min_gain), max_gain))
    return (weights / weights.mean().clamp_min(1e-6)).to(device=device)


def build_teacher_error_replay_class_mask(
    class_to_idx: dict[str, int],
    *,
    focus: str,
    device: torch.device,
) -> torch.Tensor:
    """Class mask for teacher-error replay.

    This is deliberately narrower than a class-weight prior. It only defines
    which labels are allowed to receive extra loss when the frozen anchor is
    already wrong on that image, so anchor-correct samples are left to the
    no-harm teacher term.
    """

    focus = str(focus)
    masks: dict[str, set[str]] = {
        "top1_gap_v1": {
            "dry_concrete_slight",
            "dry_concrete_severe",
            "wet_concrete_smooth",
            "water_concrete_smooth",
            "wet_asphalt_slight",
            "dry_asphalt_slight",
            "water_concrete_slight",
            "wet_concrete_slight",
            "water_concrete_severe",
            "wet_concrete_severe",
            "water_asphalt_slight",
            "water_asphalt_severe",
            "wet_asphalt_severe",
            "dry_asphalt_severe",
        },
        "majority_smooth_v1": {
            "dry_asphalt_smooth",
            "dry_concrete_smooth",
            "wet_asphalt_smooth",
            "wet_concrete_smooth",
            "water_asphalt_smooth",
            "water_concrete_smooth",
            "dry_gravel",
            "wet_mud",
            "dry_asphalt_slight",
        },
        "concrete_wetwater_rough_v1": {
            "dry_concrete_slight",
            "dry_concrete_severe",
            "wet_concrete_smooth",
            "wet_concrete_slight",
            "wet_concrete_severe",
            "water_concrete_smooth",
            "water_concrete_slight",
            "water_concrete_severe",
        },
    }
    if focus not in {"top1_gap_v1", "majority_smooth_v1", "concrete_wetwater_rough_v1", "all"}:
        raise ValueError(f"unknown teacher-error replay focus: {focus}")
    mask = torch.zeros(len(class_to_idx), dtype=torch.bool)
    if focus == "all":
        mask.fill_(True)
    else:
        allowed = masks[focus]
        for name, idx in class_to_idx.items():
            if name in allowed:
                mask[int(idx)] = True
    return mask.to(device=device)


def _is_core_friction_material_class(factors: dict[str, str | None]) -> bool:
    return (
        factors["friction"] in {"dry", "wet", "water"}
        and factors["material"] in {"asphalt", "concrete"}
    )


def build_factor_neighbor_negative_mask(
    class_to_idx: dict[str, int],
    device: torch.device,
    *,
    core_only: bool = False,
) -> torch.Tensor:
    """Build hard-negative masks along RSCD factor axes.

    The low-F1 RSCD classes are mainly confused with coupled neighbors: same
    material/roughness but adjacent friction, same friction/material but adjacent
    roughness, or visually close material pairs such as asphalt/concrete.
    """

    num_classes = len(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    factors = {idx: _factor_text(idx_to_class[idx]) for idx in range(num_classes)}
    mask = torch.zeros((num_classes, num_classes), dtype=torch.bool)
    for i in range(num_classes):
        a = factors[i]
        for j in range(num_classes):
            if i == j:
                continue
            b = factors[j]
            if core_only and not (_is_core_friction_material_class(a) and _is_core_friction_material_class(b)):
                continue
            same_material = a["material"] is not None and a["material"] == b["material"]
            same_uneven = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
            same_friction = a["friction"] is not None and a["friction"] == b["friction"]
            if same_material and same_uneven and _friction_neighbors(a["friction"], b["friction"]):
                mask[i, j] = True
            if same_material and same_friction and _unevenness_neighbors(a["unevenness"], b["unevenness"]):
                mask[i, j] = True
            if same_friction and same_uneven and _material_neighbors(a["material"], b["material"]):
                mask[i, j] = True
    return mask.to(device=device)


def build_factor_neighbor_prior_hard_mask(class_to_idx: dict[str, int], device: torch.device) -> torch.Tensor:
    """Classes where sample-weighted Top-1 errors concentrate in current RSCD audits."""

    hard = torch.zeros(len(class_to_idx), dtype=torch.bool)
    for name, idx in class_to_idx.items():
        factors = _factor_text(name)
        friction = factors["friction"]
        material = factors["material"]
        unevenness = factors["unevenness"]
        hard[int(idx)] = bool(
            material == "concrete"
            or unevenness in {"slight", "severe"}
            or (friction in {"wet", "water"} and material in {"asphalt", "concrete"})
            or name in {"dry_asphalt_severe", "water_gravel"}
        )
    return hard.to(device=device)


def build_natural_prior_log_bias(
    manifest: Path,
    class_to_idx: dict[str, int],
    *,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """Natural RSCD class-prior log bias from the uncapped training manifest.

    Balanced sampling is useful for Mean-F1, but the official Top-1 metric is
    sample-weighted. This vector estimates the natural training prior without
    hand-written class boosts, so an auxiliary local-neighbor utility loss can
    teach margins that are less hostile to frequent, physically adjacent cells.
    """

    df = pd.read_csv(manifest, usecols=["class_label"], dtype=str, low_memory=False)
    counts = torch.full((len(class_to_idx),), float(alpha), dtype=torch.float32)
    for value, count in df["class_label"].dropna().astype(str).map(canonical_class_label).value_counts().items():
        idx = class_to_idx.get(str(value))
        if idx is not None:
            counts[int(idx)] += float(count)
    prior = counts / counts.sum().clamp_min(1.0)
    uniform = torch.full_like(prior, 1.0 / max(len(class_to_idx), 1))
    bias = torch.log(prior.clamp_min(1e-8) / uniform.clamp_min(1e-8))
    bias = bias - bias.mean()
    return bias.to(device=device)


def build_factor_neighbor_weight_matrix(
    class_to_idx: dict[str, int],
    device: torch.device,
    *,
    core_only: bool = False,
    roughness_weight: float = 1.0,
    friction_weight: float = 1.0,
    material_weight: float = 1.0,
    wet_water_weight: float = 1.0,
    concrete_weight: float = 1.0,
    hard_class_weight: float = 1.0,
) -> torch.Tensor:
    """Build weighted RSCD factor-neighbor penalties.

    The rows are true classes and columns are factor-neighbor negatives. This
    gives the loss a first-principles bias toward the observed RSCD bottleneck:
    roughness confusions dominate, wet/water is the hardest friction boundary,
    and concrete slight/severe water states are the weakest cells.
    """

    hard_classes = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "dry_concrete_slight",
        "wet_concrete_severe",
    }
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    factors = {idx: _factor_text(idx_to_class[idx]) for idx in range(len(class_to_idx))}
    weights = torch.zeros((len(class_to_idx), len(class_to_idx)), dtype=torch.float32)
    for i in range(len(class_to_idx)):
        a = factors[i]
        for j in range(len(class_to_idx)):
            if i == j:
                continue
            b = factors[j]
            if core_only and not (_is_core_friction_material_class(a) and _is_core_friction_material_class(b)):
                continue
            same_material = a["material"] is not None and a["material"] == b["material"]
            same_uneven = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
            same_friction = a["friction"] is not None and a["friction"] == b["friction"]
            edge_weight = 0.0
            if same_material and same_uneven and _friction_neighbors(a["friction"], b["friction"]):
                edge_weight = max(edge_weight, float(friction_weight))
                if {a["friction"], b["friction"]} == {"wet", "water"}:
                    edge_weight *= float(wet_water_weight)
            if same_material and same_friction and _unevenness_neighbors(a["unevenness"], b["unevenness"]):
                edge_weight = max(edge_weight, float(roughness_weight))
            if same_friction and same_uneven and _material_neighbors(a["material"], b["material"]):
                edge_weight = max(edge_weight, float(material_weight))
            if edge_weight <= 0.0:
                continue
            if a["material"] == "concrete":
                edge_weight *= float(concrete_weight)
            if idx_to_class[i] in hard_classes:
                edge_weight *= float(hard_class_weight)
            weights[i, j] = edge_weight
    return weights.to(device=device)


def build_factor_marginal_masks(class_to_idx: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]:
    """Masks that map 27 RSCD classes to friction/material/unevenness factors."""

    masks: dict[str, torch.Tensor] = {}
    for name, labels in FACTOR_LABELS.items():
        mask = torch.zeros((len(labels), len(class_to_idx)), dtype=torch.bool)
        for class_name, class_idx in class_to_idx.items():
            factor_idx = parse_rscd_factors(class_name)[name]
            if factor_idx >= 0:
                mask[int(factor_idx), int(class_idx)] = True
        masks[name] = mask.to(device=device)
    return masks


def build_relation_conditional_masks(
    class_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Masks for conditional factor probabilities from 27-class logits.

    For an RSCD class y=(f,m,u), the three conditional objectives are:
    P(f|m,u), P(m|f,u), and P(u|f,m). Each probability is computed by
    summing class probability mass over compatible labels, so evaluation still
    uses the original 27-class protocol.
    """

    num_classes = len(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    factors = {idx: _factor_text(idx_to_class[idx]) for idx in range(num_classes)}
    out: dict[str, dict[str, torch.Tensor]] = {}
    for axis in FACTOR_LABELS:
        denominator = torch.zeros((num_classes, num_classes), dtype=torch.bool)
        numerator = torch.zeros((num_classes, num_classes), dtype=torch.bool)
        valid = torch.zeros(num_classes, dtype=torch.float32)
        other_axes = [name for name in FACTOR_LABELS if name != axis]
        for i in range(num_classes):
            a = factors[i]
            if a[axis] is None:
                continue
            active_other_axes = [name for name in other_axes if a[name] is not None]
            axis_values: set[str] = set()
            for j in range(num_classes):
                b = factors[j]
                if b[axis] is None:
                    continue
                if any(b[name] != a[name] for name in active_other_axes):
                    continue
                denominator[i, j] = True
                axis_values.add(str(b[axis]))
                if b[axis] == a[axis]:
                    numerator[i, j] = True
            if denominator[i].sum() > numerator[i].sum() and len(axis_values) >= 2:
                valid[i] = 1.0
        out[axis] = {
            "denominator": denominator.to(device=device),
            "numerator": numerator.to(device=device),
            "valid": valid.to(device=device),
        }
    return out


def build_relation_conditional_focus_mask(
    class_to_idx: dict[str, int],
    *,
    focus: str,
    device: torch.device,
) -> torch.Tensor:
    focus = str(focus)
    hard_classes = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "dry_concrete_slight",
        "wet_concrete_severe",
    }
    mask = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for class_name, class_idx in class_to_idx.items():
        factors = _factor_text(class_name)
        is_core = (
            factors["friction"] in {"dry", "wet", "water"}
            and factors["material"] in {"asphalt", "concrete"}
            and factors["unevenness"] in {"smooth", "slight", "severe"}
        )
        if focus == "all":
            mask[int(class_idx)] = 1.0
        elif focus == "core" and is_core:
            mask[int(class_idx)] = 1.0
        elif focus == "hard" and class_name in hard_classes:
            mask[int(class_idx)] = 1.0
        elif focus not in {"all", "core", "hard"}:
            raise ValueError(f"unknown relation conditional focus: {focus}")
    return mask.to(device=device)


def build_tensor_anova_boundary_spec(class_to_idx: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]:
    """Index tensors for core RSCD Tensor-ANOVA coupling supervision.

    The core paved labels form a 3 x 2 x 3 tensor:
    friction in {dry, wet, water}, material in {asphalt, concrete}, and
    unevenness in {smooth, slight, severe}. Non-core classes stay in the normal
    CE objective; this auxiliary only changes training geometry for the
    factorized labels where full three-way coupling is observable.
    """

    friction_values = ("dry", "wet", "water")
    material_values = ("asphalt", "concrete")
    roughness_values = ("smooth", "slight", "severe")
    core_indices: list[int] = []
    class_cell = torch.full((len(class_to_idx), 3), -1, dtype=torch.long)
    missing: list[str] = []
    for f_idx, friction in enumerate(friction_values):
        for m_idx, material in enumerate(material_values):
            for r_idx, roughness in enumerate(roughness_values):
                class_name = f"{friction}_{material}_{roughness}"
                class_index = class_to_idx.get(class_name)
                if class_index is None:
                    missing.append(class_name)
                    core_indices.append(0)
                    continue
                core_indices.append(int(class_index))
                class_cell[int(class_index)] = torch.tensor([f_idx, m_idx, r_idx], dtype=torch.long)
    if missing:
        raise ValueError(f"RSCD core Tensor-ANOVA labels are missing: {missing}")

    hard_classes = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "dry_concrete_slight",
        "wet_concrete_severe",
        "water_asphalt_severe",
        "dry_concrete_severe",
    }
    core_focus = torch.zeros(len(class_to_idx), dtype=torch.float32)
    hard_focus = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for class_name, class_idx in class_to_idx.items():
        factors = _factor_text(class_name)
        is_core = (
            factors["friction"] in set(friction_values)
            and factors["material"] in set(material_values)
            and factors["unevenness"] in set(roughness_values)
        )
        if is_core:
            core_focus[int(class_idx)] = 1.0
        if canonical_class_label(class_name) in hard_classes:
            hard_focus[int(class_idx)] = 1.0
    return {
        "core_indices": torch.tensor(core_indices, dtype=torch.long, device=device),
        "class_cell": class_cell.to(device=device),
        "core_focus_mask": core_focus.to(device=device),
        "hard_focus_mask": hard_focus.to(device=device),
    }


def build_relation_conditional_adaptive_weights(
    class_to_idx: dict[str, int],
    *,
    mode: str,
    reference_json: str | None,
    strength: float,
    target_f1: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Per-class, per-factor curriculum weights for conditional decoupling.

    The weights are intentionally training-only. They use a prior evaluation
    report to identify weak RSCD classes, then route extra supervision to the
    label factor that is physically plausible for the class: wet/water classes
    get more friction-state pressure, asphalt/concrete core classes get some
    material pressure, and smooth/slight/severe classes get roughness pressure.
    """

    if str(mode) == "none":
        return None
    if str(mode) != "weak_f1_v1":
        raise ValueError(f"unknown relation conditional adaptive weighting: {mode}")
    num_classes = len(class_to_idx)
    axis_names = list(FACTOR_LABELS.keys())
    weights = torch.ones((num_classes, len(axis_names)), dtype=torch.float32)
    f1_by_class: dict[str, float] = {}
    if reference_json:
        ref_path = Path(reference_json)
        if ref_path.exists():
            report = json.loads(ref_path.read_text(encoding="utf-8"))
            class_report = report.get("classification_report", {})
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and "f1-score" in metrics:
                    f1_by_class[canonical_class_label(class_name)] = float(metrics["f1-score"])
        else:
            raise FileNotFoundError(f"--relation-conditional-reference-json does not exist: {ref_path}")
    hard_fallback = {
        "water_concrete_slight": 0.766,
        "water_asphalt_slight": 0.797,
        "water_concrete_severe": 0.805,
        "wet_concrete_slight": 0.812,
        "dry_concrete_slight": 0.827,
        "wet_concrete_severe": 0.838,
        "water_asphalt_severe": 0.841,
        "water_gravel": 0.856,
    }
    for class_name, idx in class_to_idx.items():
        canonical = canonical_class_label(class_name)
        observed_f1 = f1_by_class.get(canonical, hard_fallback.get(canonical, float(target_f1)))
        deficit = max(0.0, float(target_f1) - float(observed_f1))
        boost = min(1.0, deficit / max(1e-6, 1.0 - float(target_f1))) * float(strength)
        if boost <= 0.0:
            continue
        factors = _factor_text(canonical)
        priorities = {
            "friction": 0.25,
            "material": 0.15,
            "unevenness": 0.20,
        }
        if factors["friction"] in {"wet", "water"}:
            priorities["friction"] += 0.75
        elif factors["friction"] == "dry":
            priorities["friction"] += 0.25
        if factors["material"] in {"asphalt", "concrete"}:
            priorities["material"] += 0.35
        if factors["material"] == "concrete":
            priorities["material"] += 0.15
        if factors["unevenness"] in {"slight", "severe"}:
            priorities["unevenness"] += 0.65
        elif factors["unevenness"] == "smooth":
            priorities["unevenness"] += 0.25
        priority_sum = sum(priorities.values())
        for axis_idx, axis in enumerate(axis_names):
            weights[int(idx), axis_idx] += boost * priorities[axis] / max(priority_sum, 1e-6)
    return weights.to(device=device)


def relation_conditional_factor_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: dict[str, dict[str, torch.Tensor]],
    focus_mask: torch.Tensor,
    *,
    axis_weights: dict[str, float],
    class_axis_weights: torch.Tensor | None = None,
    uncertainty_margin: float,
    gate_temperature: float,
) -> torch.Tensor:
    """Conditional factor decoupling loss from the final class distribution."""

    if logits.numel() == 0:
        return logits.new_zeros(())
    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
    if top2.shape[1] < 2:
        pred_margin = logits.new_zeros((logits.shape[0],))
    else:
        pred_margin = top2[:, 0] - top2[:, 1]
    uncertainty_gate = torch.sigmoid((float(uncertainty_margin) - pred_margin) * float(gate_temperature))
    class_gate = focus_mask.to(device=logits.device, dtype=logits.dtype).index_select(0, labels)
    losses = []
    weights = []
    class_axis_weights_for_batch = None
    if class_axis_weights is not None:
        class_axis_weights_for_batch = class_axis_weights.to(device=logits.device, dtype=logits.dtype).index_select(0, labels)
    for axis_idx, (axis, mask_pack) in enumerate(masks.items()):
        axis_weight = float(axis_weights.get(axis, 1.0))
        if axis_weight <= 0.0:
            continue
        denominator = mask_pack["denominator"].to(device=logits.device).index_select(0, labels)
        numerator = mask_pack["numerator"].to(device=logits.device).index_select(0, labels)
        valid = mask_pack["valid"].to(device=logits.device, dtype=logits.dtype).index_select(0, labels)
        valid = valid * class_gate
        if not bool((valid > 0).any()):
            continue
        denom_logit = logits.masked_fill(~denominator, -1.0e4)
        numer_logit = logits.masked_fill(~numerator, -1.0e4)
        per_sample = torch.logsumexp(denom_logit, dim=1) - torch.logsumexp(numer_logit, dim=1)
        sample_weight = valid * uncertainty_gate.to(dtype=logits.dtype)
        if class_axis_weights_for_batch is not None:
            sample_weight = sample_weight * class_axis_weights_for_batch[:, int(axis_idx)]
        if float(sample_weight.sum().detach().cpu()) <= 1e-8:
            continue
        losses.append((per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6))
        weights.append(logits.new_tensor(axis_weight))
    if not losses:
        return logits.new_zeros(())
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights).to(device=loss_tensor.device, dtype=loss_tensor.dtype)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)


def factor_neighbor_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Require true logits to beat coupled factor-neighbor negatives."""

    mask = negative_mask.index_select(0, labels).to(device=logits.device)
    valid = mask.any(dim=1)
    if not bool(valid.any()):
        return logits.new_zeros(())
    true_logits = logits.gather(1, labels.view(-1, 1))
    neg_logits = logits.masked_fill(~mask, -1.0e4)
    hardest_neighbor = torch.logsumexp(neg_logits, dim=1, keepdim=True)
    loss = F.relu(float(margin) - (true_logits - hardest_neighbor)).squeeze(1)
    return loss[valid].mean()


def build_roughness_boundary_mask(
    class_to_idx: dict[str, int],
    *,
    focus: str,
    device: torch.device,
) -> torch.Tensor:
    """Hard-neighbor graph for same friction/material roughness boundaries."""

    focus = str(focus)
    num_classes = len(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    factors = {idx: _factor_text(idx_to_class[idx]) for idx in range(num_classes)}
    hard_classes = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "water_concrete_severe",
        "wet_concrete_slight",
        "dry_concrete_slight",
        "wet_concrete_severe",
        "water_asphalt_severe",
        "dry_concrete_severe",
    }
    mask = torch.zeros((num_classes, num_classes), dtype=torch.bool)
    for i in range(num_classes):
        a = factors[i]
        if a["friction"] not in {"dry", "wet", "water"}:
            continue
        if a["material"] not in {"asphalt", "concrete"}:
            continue
        if a["unevenness"] not in {"smooth", "slight", "severe"}:
            continue
        if focus == "hard" and canonical_class_label(idx_to_class[i]) not in hard_classes:
            continue
        if focus not in {"core", "hard"}:
            raise ValueError(f"unknown roughness boundary focus: {focus}")
        for j in range(num_classes):
            if i == j:
                continue
            b = factors[j]
            if b["friction"] != a["friction"] or b["material"] != a["material"]:
                continue
            if b["unevenness"] in {"smooth", "slight", "severe"} and b["unevenness"] != a["unevenness"]:
                mask[i, j] = True
    return mask.to(device=device)


def build_concrete_masked_roughness_masks(class_to_idx: dict[str, int], device: torch.device) -> torch.Tensor:
    """Conditioned roughness masks for wet/water concrete classes.

    Row y contains three masks over classes with the same friction and material
    as y, one for each roughness value. Only wet/water concrete rows are active.
    """

    roughness_order = {"smooth": 0, "slight": 1, "severe": 2}
    num_classes = len(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    factors = {idx: _factor_text(idx_to_class[idx]) for idx in range(num_classes)}
    masks = torch.zeros((num_classes, 3, num_classes), dtype=torch.bool)
    for row_idx in range(num_classes):
        row = factors[row_idx]
        if row["friction"] not in {"wet", "water"}:
            continue
        if row["material"] != "concrete":
            continue
        if row["unevenness"] not in roughness_order:
            continue
        for col_idx in range(num_classes):
            col = factors[col_idx]
            if col["friction"] != row["friction"] or col["material"] != row["material"]:
                continue
            rough_idx = roughness_order.get(str(col["unevenness"]))
            if rough_idx is not None:
                masks[row_idx, rough_idx, col_idx] = True
    return masks.to(device=device)


def roughness_boundary_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    boundary_mask: torch.Tensor,
    *,
    margin: float,
    uncertainty_margin: float,
    gate_temperature: float,
) -> torch.Tensor:
    """Separate roughness neighbors without smoothing the label graph.

    For a class y=(f,m,u), the negatives are only classes (f,m,u') with the same
    friction and material but a different roughness level. This directly targets
    the audited slight/severe failure mode while leaving other RSCD factors
    untouched.
    """

    mask = boundary_mask.index_select(0, labels).to(device=logits.device)
    valid = mask.any(dim=1)
    if not bool(valid.any()):
        return logits.new_zeros(())
    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
    if top2.shape[1] < 2:
        pred_margin = logits.new_zeros((logits.shape[0],))
    else:
        pred_margin = top2[:, 0] - top2[:, 1]
    gate = torch.sigmoid((float(uncertainty_margin) - pred_margin) * float(gate_temperature))
    true_logits = logits.gather(1, labels.view(-1, 1))
    neg_logits = logits.masked_fill(~mask, -1.0e4)
    hardest_neighbor = torch.logsumexp(neg_logits, dim=1, keepdim=True)
    per_sample = F.relu(float(margin) - (true_logits - hardest_neighbor)).squeeze(1)
    weights = gate.to(dtype=logits.dtype) * valid.to(dtype=logits.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def concrete_masked_roughness_ordinal_loss(
    logits: torch.Tensor,
    image: torch.Tensor,
    batch: dict[str, Any],
    conditioned_masks: torch.Tensor,
    *,
    obstruction_weight: float,
    uncertainty_margin: float,
    gate_temperature: float,
) -> torch.Tensor:
    """Training-only roughness supervision for wet/water concrete.

    Wet/water concrete roughness is hard because optical film can hide texture.
    This loss projects logits to smooth/slight/severe under the same friction
    and material condition, then upweights samples whose image statistics imply
    thin film, specular reflection, dark water, or texture erasure.
    """

    if conditioned_masks is None:
        return logits.new_zeros(())
    labels = batch["label"].to(device=logits.device)
    rough_target = batch["unevenness_factor"].to(device=logits.device)
    friction = batch["friction_factor"].to(device=logits.device)
    material = batch["material_factor"].to(device=logits.device)
    wet_idx = FACTOR_LABELS["friction"].index("wet")
    water_idx = FACTOR_LABELS["friction"].index("water")
    concrete_idx = FACTOR_LABELS["material"].index("concrete")
    valid = (
        ((friction == wet_idx) | (friction == water_idx))
        & (material == concrete_idx)
        & (rough_target >= 0)
        & (rough_target < 3)
    )
    masks = conditioned_masks.to(device=logits.device).index_select(0, labels)
    valid = valid & masks.any(dim=2).all(dim=1)
    if not bool(valid.any()):
        return logits.new_zeros(())
    rough_logits = torch.logsumexp(
        logits.unsqueeze(1).masked_fill(~masks, -1.0e4),
        dim=2,
    )
    selected = rough_logits.index_select(0, valid.nonzero(as_tuple=False).flatten())
    y = rough_target[valid]
    at_least_slight = torch.logsumexp(selected[:, 1:3], dim=1) - selected[:, 0]
    at_least_severe = selected[:, 2] - torch.logsumexp(selected[:, 0:2], dim=1)
    ordinal_logits = torch.stack([at_least_slight, at_least_severe], dim=1)
    ordinal_targets = torch.stack([(y >= 1), (y >= 2)], dim=1).to(dtype=ordinal_logits.dtype)
    per_sample = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets, reduction="none").mean(dim=1)

    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
    if top2.shape[1] < 2:
        margin = logits.new_zeros((logits.shape[0],))
    else:
        margin = top2[:, 0] - top2[:, 1]
    uncertainty_gate = torch.sigmoid((float(uncertainty_margin) - margin[valid]) * float(gate_temperature))
    obstruction = _concrete_optical_obstruction_score(image).to(device=logits.device, dtype=logits.dtype)[valid]
    weights = uncertainty_gate.to(dtype=logits.dtype) * (1.0 + float(obstruction_weight) * obstruction)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def _gather_core_tensor_cells(grid: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    batch_index = torch.arange(grid.shape[0], device=grid.device)
    return grid[batch_index, cells[:, 0], cells[:, 1], cells[:, 2]]


def tensor_anova_boundary_energy_loss(
    logits: torch.Tensor,
    image: torch.Tensor,
    labels: torch.Tensor,
    spec: dict[str, torch.Tensor],
    *,
    focus: str,
    margin: float,
    uncertainty_margin: float,
    gate_temperature: float,
    friction_weight: float,
    material_weight: float,
    roughness_weight: float,
    obstruction_weight: float,
) -> torch.Tensor:
    """Mechanism-specific boundary loss on the full three-factor RSCD tensor.

    For the 18 paved core labels, the class logits are reshaped as Z[f,m,r] and
    decomposed into main effects, pair effects, and the irreducible three-way
    term H_fmr. Axis comparisons intentionally keep only terms that vary along
    the tested axis:

    friction:  A_f + D_fm + E_fr + H_fmr
    material:  B_m + D_fm + G_mr + H_fmr
    roughness: C_r + E_fr + G_mr + H_fmr

    This prevents the hard wet-concrete-slight style cells from being explained
    away by additive or low-rank factors while still avoiding a new test-time
    module.
    """

    if logits.numel() == 0:
        return logits.new_zeros(())
    core_indices = spec["core_indices"].to(device=logits.device)
    if int(core_indices.numel()) != 18:
        return logits.new_zeros(())
    class_cell_table = spec["class_cell"].to(device=logits.device)
    cells = class_cell_table.index_select(0, labels)
    valid = cells.ge(0).all(dim=1)
    if not bool(valid.any()):
        return logits.new_zeros(())

    focus = str(focus)
    if focus == "hard":
        focus_mask = spec["hard_focus_mask"].to(device=logits.device, dtype=logits.dtype)
    elif focus == "core":
        focus_mask = spec["core_focus_mask"].to(device=logits.device, dtype=logits.dtype)
    else:
        raise ValueError(f"unknown Tensor-ANOVA boundary focus: {focus}")
    class_gate = focus_mask.index_select(0, labels) * valid.to(dtype=logits.dtype)
    if float(class_gate.sum().detach().cpu()) <= 1e-8:
        return logits.new_zeros(())

    z = logits.index_select(1, core_indices).float().view(-1, 3, 2, 3)
    grand = z.mean(dim=(1, 2, 3), keepdim=True)
    a_f = z.mean(dim=(2, 3), keepdim=True) - grand
    b_m = z.mean(dim=(1, 3), keepdim=True) - grand
    c_r = z.mean(dim=(1, 2), keepdim=True) - grand
    d_fm = z.mean(dim=3, keepdim=True) - grand - a_f - b_m
    e_fr = z.mean(dim=2, keepdim=True) - grand - a_f - c_r
    g_mr = z.mean(dim=1, keepdim=True) - grand - b_m - c_r
    h_fmr = z - grand - a_f - b_m - c_r - d_fm - e_fr - g_mr

    axis_scores = {
        "friction": a_f + d_fm + e_fr + h_fmr,
        "material": b_m + d_fm + g_mr + h_fmr,
        "roughness": c_r + e_fr + g_mr + h_fmr,
    }

    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
    if top2.shape[1] < 2:
        pred_margin = logits.new_zeros((logits.shape[0],), dtype=torch.float32)
    else:
        pred_margin = (top2[:, 0] - top2[:, 1]).float()
    uncertainty_gate = torch.sigmoid((float(uncertainty_margin) - pred_margin) * float(gate_temperature))

    friction = cells[:, 0]
    material = cells[:, 1]
    roughness = cells[:, 2]
    wet_or_water = friction.ge(1)
    concrete = material.eq(1)
    rough_visible = roughness.ge(1)
    obstruction = _concrete_optical_obstruction_score(image).to(device=logits.device, dtype=torch.float32)

    friction_mechanism = (
        0.30
        + 0.70 * wet_or_water.to(dtype=torch.float32)
        + 0.25 * concrete.to(dtype=torch.float32)
        + 0.20 * rough_visible.to(dtype=torch.float32)
    )
    material_mechanism = (
        0.25
        + 0.65 * wet_or_water.to(dtype=torch.float32)
        + 0.20 * rough_visible.to(dtype=torch.float32)
        + 0.30 * obstruction * wet_or_water.to(dtype=torch.float32)
    )
    roughness_mechanism = (
        0.35
        + 0.55 * rough_visible.to(dtype=torch.float32)
        + 0.35 * concrete.to(dtype=torch.float32)
        + 0.35 * wet_or_water.to(dtype=torch.float32)
        + float(obstruction_weight)
        * obstruction
        * concrete.to(dtype=torch.float32)
        * wet_or_water.to(dtype=torch.float32)
    )
    mechanism_weights = {
        "friction": friction_mechanism,
        "material": material_mechanism,
        "roughness": roughness_mechanism,
    }

    axis_specs = (
        ("friction", 0, 3, float(friction_weight)),
        ("material", 1, 2, float(material_weight)),
        ("roughness", 2, 3, float(roughness_weight)),
    )
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    base_weight = (uncertainty_gate.to(device=logits.device) * class_gate.float()).float()
    for axis_name, axis_idx, axis_size, axis_weight in axis_specs:
        if axis_weight <= 0.0:
            continue
        score_grid = axis_scores[axis_name]
        true_score = _gather_core_tensor_cells(score_grid, cells)
        neg_scores: list[torch.Tensor] = []
        for value in range(axis_size):
            candidate = cells.clone()
            candidate[:, axis_idx] = int(value)
            candidate_score = _gather_core_tensor_cells(score_grid, candidate)
            is_negative = cells[:, axis_idx].ne(int(value))
            neg_scores.append(candidate_score.masked_fill(~is_negative, -1.0e4))
        hardest_neighbor = torch.stack(neg_scores, dim=1).max(dim=1).values
        per_sample = F.relu(float(margin) - (true_score - hardest_neighbor))
        sample_weight = base_weight * mechanism_weights[axis_name].to(device=logits.device)
        active = sample_weight > 1e-8
        if not bool(active.any()):
            continue
        losses.append((per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6))
        weights.append(logits.new_tensor(axis_weight, dtype=torch.float32))
    if not losses:
        return logits.new_zeros(())
    stacked = torch.stack(losses)
    axis_weight_tensor = torch.stack(weights).to(device=stacked.device, dtype=stacked.dtype)
    return ((stacked * axis_weight_tensor).sum() / axis_weight_tensor.sum().clamp_min(1e-6)).to(dtype=logits.dtype)


@torch.no_grad()
def _concrete_optical_obstruction_score(image: torch.Tensor) -> torch.Tensor:
    """Return a [0,1]-like film/erasure score from normalized road patches."""

    device = image.device
    dtype = torch.float32
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
    rgb = (image.float() * std + mean).clamp(0.0, 1.0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    maxc = rgb.max(dim=1, keepdim=True).values
    minc = rgb.min(dim=1, keepdim=True).values
    value = maxc
    saturation = (maxc - minc) / maxc.clamp_min(1e-4)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    laplace = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
    lap = F.conv2d(gray, laplace, padding=1).abs()
    local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    local_contrast = F.avg_pool2d((gray - local_mean).abs(), kernel_size=9, stride=1, padding=4)
    low_texture = torch.sigmoid((0.045 - grad) * 35.0)
    low_contrast = torch.sigmoid((0.030 - local_contrast) * 45.0)
    specular = torch.sigmoid((value - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
    dark_water = torch.sigmoid((0.42 - value) * 10.0) * torch.sigmoid((0.30 - saturation) * 12.0) * low_texture
    thin_film = torch.clamp(specular + 0.6 * dark_water, 0.0, 1.0) * torch.sigmoid((0.08 - lap) * 22.0)
    texture_erasure = low_texture * low_contrast * torch.sigmoid((0.24 - saturation) * 10.0)
    score = torch.stack(
        [
            specular.mean(dim=(2, 3)).squeeze(1),
            dark_water.mean(dim=(2, 3)).squeeze(1),
            thin_film.mean(dim=(2, 3)).squeeze(1),
            texture_erasure.mean(dim=(2, 3)).squeeze(1),
        ],
        dim=1,
    ).mean(dim=1)
    return score.clamp(0.0, 1.0)


def factor_neighbor_contrastive_loss(
    feature: torch.Tensor,
    batch: dict[str, Any],
    *,
    margin: float,
    roughness_weight: float,
    friction_weight: float,
    material_weight: float,
    wet_water_weight: float,
    concrete_weight: float,
    hard_class_weight: float,
) -> torch.Tensor:
    """Separate factor-neighbor samples in representation space.

    This is deliberately not a logit residual. RSCD factor neighbors are often
    visually similar but label-different, so the feature geometry should reserve
    angular margin for roughness/friction/material boundaries without smoothing
    the class logits.
    """

    if feature.shape[0] < 2:
        return feature.new_zeros(())
    z = F.normalize(feature.float(), dim=1)
    cosine = z @ z.T
    labels_text = [canonical_class_label(x) for x in batch["class_label"]]
    factors = [_factor_text(x) for x in labels_text]
    weights = feature.new_zeros((feature.shape[0], feature.shape[0]), dtype=torch.float32)
    hard_classes = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "dry_concrete_slight",
        "wet_concrete_severe",
    }
    for i, a in enumerate(factors):
        for j, b in enumerate(factors):
            if i == j:
                continue
            same_material = a["material"] is not None and a["material"] == b["material"]
            same_uneven = a["unevenness"] is not None and a["unevenness"] == b["unevenness"]
            same_friction = a["friction"] is not None and a["friction"] == b["friction"]
            edge_weight = 0.0
            if same_material and same_uneven and _friction_neighbors(a["friction"], b["friction"]):
                edge_weight = max(edge_weight, float(friction_weight))
                if {a["friction"], b["friction"]} == {"wet", "water"}:
                    edge_weight *= float(wet_water_weight)
            if same_material and same_friction and _unevenness_neighbors(a["unevenness"], b["unevenness"]):
                edge_weight = max(edge_weight, float(roughness_weight))
            if same_friction and same_uneven and _material_neighbors(a["material"], b["material"]):
                edge_weight = max(edge_weight, float(material_weight))
            if edge_weight <= 0.0:
                continue
            if a["material"] == "concrete" or b["material"] == "concrete":
                edge_weight *= float(concrete_weight)
            if labels_text[i] in hard_classes or labels_text[j] in hard_classes:
                edge_weight *= float(hard_class_weight)
            weights[i, j] = edge_weight
    valid = weights > 0
    if not bool(valid.any()):
        return feature.new_zeros(())
    penalty = F.relu(cosine - float(margin)).square() * weights.to(device=feature.device, dtype=cosine.dtype)
    return penalty[valid].sum() / weights.to(device=feature.device, dtype=cosine.dtype)[valid].sum().clamp_min(1e-6)


def controlled_factor_tournament_loss(
    feature: torch.Tensor,
    batch: dict[str, Any],
    *,
    temperature: float,
    margin: float,
    neg_weight: float,
    friction_weight: float,
    material_weight: float,
    unevenness_weight: float,
    focus: str,
    axis_name: str | None = None,
) -> torch.Tensor:
    """Tournament-style comparison for RSCD compositional labels.

    For one factor axis, an anchor compares two teams in the mini-batch:
    positives that share the same factor value, and controlled opponents that
    share the other two factors but differ only on this axis. The loss asks the
    same-factor team to beat the controlled-difference team, so the feature
    learns both commonality and difference without adding inference-time heads.
    """

    if feature.shape[0] < 3:
        return feature.new_zeros(())
    focus = str(focus)
    if focus not in {"core", "all"}:
        raise ValueError(f"unknown controlled factor tournament focus: {focus}")
    device = feature.device
    dtype = feature.dtype
    z = torch.nan_to_num(feature.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    z = F.normalize(z, dim=1)
    sim = torch.nan_to_num(z @ z.T, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    bsz = int(feature.shape[0])
    eye = torch.eye(bsz, device=device, dtype=torch.bool)

    factor_tensors = {
        "friction": batch["friction_factor"].to(device=device, dtype=torch.long),
        "material": batch["material_factor"].to(device=device, dtype=torch.long),
        "unevenness": batch["unevenness_factor"].to(device=device, dtype=torch.long),
    }
    axis_specs = (
        ("friction", float(friction_weight)),
        ("material", float(material_weight)),
        ("unevenness", float(unevenness_weight)),
    )
    if axis_name is not None:
        axis_name = str(axis_name)
        if axis_name not in {"friction", "material", "unevenness"}:
            raise ValueError(f"unknown controlled factor tournament axis: {axis_name}")
        axis_specs = tuple((name, weight) for name, weight in axis_specs if name == axis_name)
    valid_all = torch.ones(bsz, device=device, dtype=torch.bool)
    for values in factor_tensors.values():
        valid_all = valid_all & values.ge(0)
    if focus == "core":
        valid_all = (
            valid_all
            & factor_tensors["friction"].le(2)
            & factor_tensors["material"].le(1)
            & factor_tensors["unevenness"].le(2)
        )
    if not bool(valid_all.any()):
        return feature.new_zeros(())

    temp = max(float(temperature), 1e-4)
    neg_weight = max(float(neg_weight), 1e-6)
    losses = []
    weights = []
    for axis, axis_weight in axis_specs:
        if axis_weight <= 0.0:
            continue
        axis_values = factor_tensors[axis]
        same_axis = axis_values[:, None].eq(axis_values[None, :])
        valid_pair = valid_all[:, None] & valid_all[None, :] & ~eye
        pos_mask = valid_pair & same_axis
        controlled_neg = valid_pair & ~same_axis
        for other_axis, other_values in factor_tensors.items():
            if other_axis == axis:
                continue
            controlled_neg = controlled_neg & other_values[:, None].eq(other_values[None, :])
        active = pos_mask.any(dim=1) & controlled_neg.any(dim=1)
        if not bool(active.any()):
            continue

        scaled = torch.nan_to_num(sim / temp, nan=0.0, posinf=50.0, neginf=-50.0)
        pos_logits = scaled.masked_fill(~pos_mask, -1.0e4)
        neg_logits = scaled.masked_fill(~controlled_neg, -1.0e4)
        pos_score = torch.logsumexp(pos_logits, dim=1) - pos_mask.sum(dim=1).clamp_min(1).float().log()
        neg_score = torch.logsumexp(neg_logits, dim=1) - controlled_neg.sum(dim=1).clamp_min(1).float().log()
        score_delta = torch.nan_to_num(neg_score - pos_score, nan=0.0, posinf=50.0, neginf=-50.0)
        tournament = F.relu(float(margin) + score_delta)
        ratio = F.softplus(math.log(float(neg_weight)) + score_delta)
        axis_loss = torch.nan_to_num((tournament + ratio)[active], nan=0.0, posinf=50.0, neginf=0.0).mean()
        losses.append(axis_loss)
        weights.append(feature.new_tensor(axis_weight, dtype=torch.float32))
    if not losses:
        return feature.new_zeros(())
    stacked = torch.stack(losses)
    weight_tensor = torch.stack(weights).to(device=stacked.device, dtype=stacked.dtype)
    return ((stacked * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)).to(dtype=dtype)


def mechanism_controlled_factor_tournament_loss(
    model_out: dict[str, Any],
    batch: dict[str, Any],
    *,
    temperature: float,
    margin: float,
    neg_weight: float,
    friction_weight: float,
    material_weight: float,
    unevenness_weight: float,
    focus: str,
) -> torch.Tensor:
    """Mechanism-adapted CFT over existing FAF evidence branches.

    Generic CFT on the fused classifier feature did not move the current RSCD
    anchor. This version binds each comparison axis to the branch that carries
    the relevant evidence: PhysicsTexture for friction/water-film state,
    concatenated low-level texture/semantic-physics evidence for material, and
    LocalPhysicsField for visible/hidden roughness.
    """

    fallback = model_out.get("feature")
    if not isinstance(fallback, torch.Tensor):
        raise ValueError("mechanism CFT requires model_out['feature']")
    def first_tensor(*names: str) -> torch.Tensor:
        for name in names:
            value = model_out.get(name)
            if isinstance(value, torch.Tensor):
                return value
        return fallback

    specs = [
        ("friction", first_tensor("physics_feature", "low_level_feature"), friction_weight),
        ("material", first_tensor("low_level_feature"), material_weight),
        ("unevenness", first_tensor("local_physics_feature", "low_level_feature"), unevenness_weight),
    ]
    losses = []
    weights = []
    for axis, feature, axis_weight in specs:
        if float(axis_weight) <= 0.0 or not isinstance(feature, torch.Tensor):
            continue
        value = controlled_factor_tournament_loss(
            feature,
            batch,
            temperature=float(temperature),
            margin=float(margin),
            neg_weight=float(neg_weight),
            focus=str(focus),
            friction_weight=1.0,
            material_weight=1.0,
            unevenness_weight=1.0,
            axis_name=axis,
        )
        if value.numel() == 0:
            continue
        losses.append(value)
        weights.append(fallback.new_tensor(float(axis_weight), dtype=torch.float32))
    if not losses:
        return fallback.new_zeros(())
    stacked = torch.stack(losses)
    weight_tensor = torch.stack(weights).to(device=stacked.device, dtype=stacked.dtype)
    return (stacked * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)


def factor_neighbor_prototype_contrastive_loss(
    model: nn.Module,
    feature: torch.Tensor,
    labels: torch.Tensor,
    negative_weights: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Separate each feature from its factor-neighbor class prototypes.

    In-batch contrastive loss is sparse with small GPU batches: the current
    mini-batch may not contain the wet/water or slight/severe neighbor needed
    for a useful hard-negative signal. This prototype form uses classifier rows
    as class prototypes, so every image is contrasted against audited RSCD
    factor-neighbor classes on every step.
    """

    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        return feature.new_zeros(())
    weights = negative_weights.index_select(0, labels).to(device=feature.device, dtype=torch.float32)
    valid = weights.sum(dim=1) > 0
    if not bool(valid.any()):
        return feature.new_zeros(())
    z = F.normalize(feature.float(), dim=1)
    prototypes = F.normalize(classifier.weight.float(), dim=1)
    cosine = z @ prototypes.T
    pos = cosine.gather(1, labels.view(-1, 1))
    pair_loss = F.relu(cosine - pos + float(margin)).square() * weights
    sample_loss = pair_loss.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    return sample_loss[valid].mean()


def relation_conditioned_prototype_contrastive_loss(
    model: nn.Module,
    feature: torch.Tensor,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    masks: dict[str, dict[str, torch.Tensor]],
    focus_mask: torch.Tensor,
    *,
    axis_weights: dict[str, float],
    class_axis_weights: torch.Tensor | None,
    temperature: float,
    uncertainty_margin: float,
    gate_temperature: float,
) -> torch.Tensor:
    """Prototype-NCA loss for RSCD conditional factor boundaries.

    This is a feature-space companion to relation_conditional_factor_loss. It
    compares each image feature with final classifier prototypes, but only within
    the physically meaningful conditional set: P(friction|material,roughness),
    P(material|friction,roughness), or P(roughness|friction,material). This gives
    every mini-batch a relation-conditioned contrastive signal without relying
    on scarce in-batch positives.
    """

    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        return feature.new_zeros(())
    if feature.shape[1] != classifier.weight.shape[1]:
        return feature.new_zeros(())
    if feature.numel() == 0:
        return feature.new_zeros(())
    temp = max(float(temperature), 1e-4)
    query = F.normalize(feature.float(), dim=1)
    prototypes = F.normalize(classifier.weight.float(), dim=1)
    scores = (query @ prototypes.T) / temp
    class_gate = focus_mask.to(device=feature.device, dtype=scores.dtype).index_select(0, labels)
    uncertainty_gate = torch.ones_like(class_gate)
    if logits is not None and logits.ndim == 2 and logits.shape[0] == feature.shape[0]:
        topk = logits.detach().topk(k=min(2, logits.shape[1]), dim=1).values
        if topk.shape[1] >= 2:
            pred_margin = topk[:, 0] - topk[:, 1]
            uncertainty_gate = torch.sigmoid(
                (float(uncertainty_margin) - pred_margin) * float(gate_temperature)
            ).to(device=feature.device, dtype=scores.dtype)
        else:
            uncertainty_gate = torch.ones_like(class_gate)
    class_axis_weights_for_batch = None
    if class_axis_weights is not None:
        class_axis_weights_for_batch = class_axis_weights.to(device=feature.device, dtype=scores.dtype).index_select(0, labels)
    losses = []
    weights = []
    for axis_idx, (axis, mask_pack) in enumerate(masks.items()):
        axis_weight = float(axis_weights.get(axis, 1.0))
        if axis_weight <= 0.0:
            continue
        denominator = mask_pack["denominator"].to(device=feature.device).index_select(0, labels)
        numerator = mask_pack["numerator"].to(device=feature.device).index_select(0, labels)
        valid = mask_pack["valid"].to(device=feature.device, dtype=scores.dtype).index_select(0, labels)
        valid = valid * class_gate * uncertainty_gate
        if not bool((valid > 0).any()):
            continue
        denom_score = scores.masked_fill(~denominator, -1.0e4)
        numer_score = scores.masked_fill(~numerator, -1.0e4)
        per_sample = torch.logsumexp(denom_score, dim=1) - torch.logsumexp(numer_score, dim=1)
        sample_weight = valid
        if class_axis_weights_for_batch is not None:
            sample_weight = sample_weight * class_axis_weights_for_batch[:, int(axis_idx)]
        if float(sample_weight.sum().detach().cpu()) <= 1e-8:
            continue
        losses.append((per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6))
        weights.append(scores.new_tensor(axis_weight))
    if not losses:
        return feature.new_zeros(())
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights).to(device=loss_tensor.device, dtype=loss_tensor.dtype)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)


def weighted_factor_neighbor_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    negative_weights: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Weighted hinge loss over all coupled factor-neighbor negatives."""

    weights = negative_weights.index_select(0, labels).to(device=logits.device, dtype=logits.dtype)
    valid = weights.sum(dim=1) > 0
    if not bool(valid.any()):
        return logits.new_zeros(())
    true_logits = logits.gather(1, labels.view(-1, 1))
    pair_loss = F.relu(float(margin) - (true_logits - logits)) * weights
    sample_loss = pair_loss.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    return sample_loss[valid].mean()


def local_factor_graph_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    neighbor_weights: torch.Tensor,
    *,
    margin: float,
    uncertainty_margin: float,
    gate_temperature: float,
    topk: int,
) -> torch.Tensor:
    """Uncertainty-gated margin only for active top-k factor-graph confusions.

    RSCD labels are compositions of friction, material, and unevenness factors.
    Broadly smoothing every graph neighbor previously helped a few hard cells
    but weakened wet/water slices. This loss is narrower: it only fires when the
    model's current top-k alternatives contain a valid factor-neighbor of the
    ground-truth class, so easy samples and unrelated classes are left alone.
    """

    if logits.numel() == 0:
        return logits.new_zeros(())
    weights = neighbor_weights.index_select(0, labels).to(device=logits.device, dtype=logits.dtype)
    k = max(2, min(int(topk), int(logits.shape[1])))
    top_idx = logits.topk(k=k, dim=1).indices
    top_mask = torch.zeros_like(logits, dtype=torch.bool)
    top_mask.scatter_(1, top_idx, True)
    true_mask = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=torch.bool, device=logits.device)
    active_weights = weights * (top_mask & ~true_mask).to(dtype=weights.dtype)
    valid = active_weights.sum(dim=1) > 0
    if not bool(valid.any()):
        return logits.new_zeros(())

    true_logits = logits.gather(1, labels.view(-1, 1))
    pair_loss = F.relu(float(margin) - (true_logits - logits)) * active_weights
    sample_loss = pair_loss.sum(dim=1) / active_weights.sum(dim=1).clamp_min(1e-6)

    top2 = logits.topk(k=2, dim=1).values
    pred_margin = top2[:, 0] - top2[:, 1]
    uncertainty_gate = torch.sigmoid((float(uncertainty_margin) - pred_margin) * float(gate_temperature))
    weighted = sample_loss * uncertainty_gate
    return weighted[valid].sum() / uncertainty_gate[valid].sum().clamp_min(1e-6)


def build_relation_specific_margin_matrices(
    class_to_idx: dict[str, int],
    device: torch.device,
    *,
    scope: str = "core",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build RSCD relation ids, hard-class weights, and no-harm protected mask."""

    if str(scope) not in {"all", "core", "hard"}:
        raise ValueError(f"unknown relation-specific margin scope: {scope}")
    num_classes = len(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    relation = torch.zeros((num_classes, num_classes), dtype=torch.long)
    class_weight = torch.ones(num_classes, dtype=torch.float32)
    protected = torch.zeros(num_classes, dtype=torch.bool)
    hard_classes = {
        "dry_concrete_slight",
        "dry_concrete_severe",
        "wet_concrete_smooth",
        "water_concrete_smooth",
        "wet_asphalt_slight",
        "dry_asphalt_slight",
        "water_concrete_slight",
        "water_concrete_severe",
        "wet_concrete_slight",
        "wet_concrete_severe",
        "water_asphalt_slight",
        "water_asphalt_severe",
    }

    for idx in range(num_classes):
        name = canonical_class_label(idx_to_class[idx])
        factors = _factor_text(name)
        if factors["friction"] in {"wet", "water"}:
            protected[idx] = True
        if name in hard_classes:
            class_weight[idx] = 1.50
        if factors["material"] == "concrete" and factors["friction"] in {"wet", "water"}:
            class_weight[idx] = max(float(class_weight[idx]), 1.80)
        if factors["material"] == "concrete" and factors["unevenness"] in {"slight", "severe"}:
            class_weight[idx] = max(float(class_weight[idx]), 1.35)

    for i in range(num_classes):
        name_i = canonical_class_label(idx_to_class[i])
        if str(scope) == "hard" and name_i not in hard_classes:
            continue
        factors_i = _factor_text(name_i)
        is_core_i = (
            factors_i["material"] in {"asphalt", "concrete"}
            and factors_i["friction"] in {"dry", "wet", "water"}
            and factors_i["unevenness"] in {"smooth", "slight", "severe"}
        )
        if str(scope) == "core" and not is_core_i:
            continue
        for j in range(num_classes):
            if i == j:
                continue
            name_j = canonical_class_label(idx_to_class[j])
            if str(scope) == "hard" and name_j not in hard_classes:
                continue
            factors_j = _factor_text(name_j)
            same_material = factors_i["material"] is not None and factors_i["material"] == factors_j["material"]
            same_uneven = factors_i["unevenness"] is not None and factors_i["unevenness"] == factors_j["unevenness"]
            same_friction = factors_i["friction"] is not None and factors_i["friction"] == factors_j["friction"]
            if same_material and same_uneven and _friction_neighbors(factors_i["friction"], factors_j["friction"]):
                relation[i, j] = 1
            elif same_material and same_friction and _unevenness_neighbors(factors_i["unevenness"], factors_j["unevenness"]):
                relation[i, j] = 2
            elif same_friction and same_uneven and _material_neighbors(factors_i["material"], factors_j["material"]):
                relation[i, j] = 3
    return relation.to(device=device), class_weight.to(device=device), protected.to(device=device)


def relation_specific_clean_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    relation_ids: torch.Tensor,
    class_weights: torch.Tensor,
    protected_mask: torch.Tensor,
    *,
    friction_margin: float,
    roughness_margin: float,
    material_margin: float,
    friction_weight: float,
    roughness_weight: float,
    material_weight: float,
    protected_negative_scale: float,
    uncertainty_margin: float,
    gate_temperature: float,
    topk: int,
) -> torch.Tensor:
    """Relation-specific clean-logit margin for active RSCD hard neighbors."""

    if logits.numel() == 0:
        return logits.new_zeros(())
    rel = relation_ids.to(device=logits.device).index_select(0, labels)
    rel = rel.to(dtype=torch.long)
    k = max(2, min(int(topk), int(logits.shape[1])))
    top_idx = logits.topk(k=k, dim=1).indices
    top_mask = torch.zeros_like(logits, dtype=torch.bool)
    top_mask.scatter_(1, top_idx, True)
    true_mask = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=torch.bool, device=logits.device)
    active = (rel > 0) & top_mask & ~true_mask
    if not bool(active.any()):
        return logits.new_zeros(())

    margin_values = logits.new_tensor([0.0, friction_margin, roughness_margin, material_margin])
    relation_weight_values = logits.new_tensor([0.0, friction_weight, roughness_weight, material_weight])
    relation_index = rel.clamp(0, 3)
    margin = margin_values.index_select(0, relation_index.flatten()).view_as(logits)
    weight = relation_weight_values.index_select(0, relation_index.flatten()).view_as(logits)
    sample_weight = class_weights.to(device=logits.device, dtype=logits.dtype).index_select(0, labels).view(-1, 1)
    weight = weight * sample_weight

    protected = protected_mask.to(device=logits.device)
    negative_protected = protected.view(1, -1) & (~protected.index_select(0, labels).view(-1, 1))
    weight = torch.where(
        negative_protected,
        weight * float(protected_negative_scale),
        weight,
    )
    weight = weight * active.to(dtype=logits.dtype)
    if float(weight.sum().detach().cpu()) <= 1e-8:
        return logits.new_zeros(())

    true_logits = logits.gather(1, labels.view(-1, 1))
    pair_loss = F.relu(margin - (true_logits - logits)).square() * weight
    per_sample = pair_loss.sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)

    top_values = logits.topk(k=2, dim=1).values
    pred_margin = top_values[:, 0] - top_values[:, 1]
    gate = torch.sigmoid((float(uncertainty_margin) - pred_margin) * float(gate_temperature))
    valid = weight.sum(dim=1) > 0
    return (per_sample[valid] * gate[valid]).sum() / gate[valid].sum().clamp_min(1e-6)


def build_directed_confusion_weight_matrix(
    class_to_idx: dict[str, int],
    device: torch.device,
    *,
    preset: str = "none",
) -> torch.Tensor:
    """Build directed hard-confusion penalties from audited RSCD errors."""

    weights = torch.zeros((len(class_to_idx), len(class_to_idx)), dtype=torch.float32)
    if str(preset) == "none":
        return weights.to(device=device)
    if str(preset) not in {"rscd_hard_v1", "rscd_protected_v2", "rscd_smooth_gap_v3"}:
        raise ValueError(f"unknown directed confusion preset: {preset}")

    if str(preset) == "rscd_smooth_gap_v3":
        pairs = {
            # Full-test audit vs RSPNet-L shows that the Top-1 gap is dominated
            # by high-support smooth cells. These directions are the observed
            # false-positive sinks of the current single-model anchor.
            ("wet_concrete_smooth", "water_concrete_smooth"): 1.70,
            ("wet_concrete_smooth", "dry_concrete_smooth"): 0.70,
            ("water_concrete_smooth", "wet_concrete_smooth"): 1.45,
            ("water_concrete_smooth", "water_concrete_slight"): 0.70,
            ("dry_concrete_smooth", "wet_concrete_smooth"): 1.00,
            ("dry_concrete_smooth", "dry_concrete_slight"): 0.85,
            ("dry_concrete_smooth", "dry_asphalt_smooth"): 0.60,
            ("dry_asphalt_smooth", "wet_asphalt_smooth"): 1.10,
            ("dry_asphalt_smooth", "dry_asphalt_slight"): 0.85,
            ("dry_asphalt_smooth", "dry_concrete_smooth"): 0.55,
            ("wet_asphalt_smooth", "wet_asphalt_slight"): 0.95,
            ("wet_asphalt_smooth", "water_asphalt_smooth"): 0.75,
            ("wet_asphalt_smooth", "dry_asphalt_smooth"): 0.55,
            ("water_asphalt_smooth", "water_asphalt_slight"): 0.95,
            ("water_asphalt_smooth", "wet_asphalt_smooth"): 0.65,
            # Granular confusions are the second-largest factor gap. Keep them
            # weaker than smooth penalties so the loss does not turn granular
            # classes into new confusion sinks.
            ("dry_gravel", "dry_mud"): 0.80,
            ("dry_mud", "dry_gravel"): 0.70,
            ("wet_gravel", "wet_mud"): 0.65,
            ("wet_mud", "wet_gravel"): 0.70,
            ("water_gravel", "water_mud"): 0.55,
            ("water_mud", "water_gravel"): 0.45,
        }
        missing = []
        for (true_name, pred_name), weight in pairs.items():
            if true_name not in class_to_idx or pred_name not in class_to_idx:
                missing.append((true_name, pred_name))
                continue
            weights[class_to_idx[true_name], class_to_idx[pred_name]] = float(weight)
        if missing:
            raise ValueError(f"directed confusion preset has missing classes: {missing}")
        return weights.to(device=device)

    pairs = {
        # Worst wet/water concrete cells: protect recall by pushing down their
        # most common wrong neighbors, instead of smoothing all factor neighbors.
        ("water_concrete_slight", "water_concrete_severe"): 1.60,
        ("water_concrete_slight", "wet_concrete_slight"): 1.40,
        ("water_concrete_severe", "water_concrete_slight"): 1.20,
        ("wet_concrete_slight", "wet_concrete_severe"): 1.15,
        ("wet_concrete_slight", "water_concrete_slight"): 1.15,
        ("wet_concrete_severe", "wet_concrete_slight"): 1.10,
        # Concrete roughness is the dominant error axis.
        ("dry_concrete_slight", "dry_concrete_severe"): 1.35,
        ("dry_concrete_slight", "dry_concrete_smooth"): 0.85,
        ("dry_concrete_severe", "dry_concrete_slight"): 1.00,
        ("dry_concrete_smooth", "dry_concrete_slight"): 0.65,
        # Asphalt/water hard cells are weaker but still among the lowest F1.
        ("water_asphalt_slight", "water_asphalt_severe"): 1.10,
        ("water_asphalt_slight", "wet_asphalt_slight"): 0.90,
        ("water_asphalt_severe", "water_asphalt_slight"): 0.85,
        ("wet_asphalt_severe", "wet_asphalt_slight"): 0.75,
        # Smooth wet/water concrete is a high-count friction confusion.
        ("water_concrete_smooth", "wet_concrete_smooth"): 0.75,
        ("wet_concrete_smooth", "water_concrete_smooth"): 0.65,
    }
    if str(preset) == "rscd_protected_v2":
        # Full-graph audit showed that the v1 asymmetric margin slightly
        # improved concrete wet/water cells but made water-asphalt and
        # granular classes more attractive confusion sinks. This preset keeps
        # the high-value concrete boundaries and protects those fragile groups.
        pairs = {
            key: value
            for key, value in pairs.items()
            if "asphalt" not in key[0]
            and "asphalt" not in key[1]
            and "gravel" not in key[0]
            and "gravel" not in key[1]
            and "mud" not in key[0]
            and "mud" not in key[1]
        }
    missing = []
    for (true_name, pred_name), weight in pairs.items():
        if true_name not in class_to_idx or pred_name not in class_to_idx:
            missing.append((true_name, pred_name))
            continue
        weights[class_to_idx[true_name], class_to_idx[pred_name]] = float(weight)
    if missing:
        raise ValueError(f"directed confusion preset has missing classes: {missing}")
    return weights.to(device=device)


def directed_confusion_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    directed_weights: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Asymmetric hinge loss for audited high-frequency confusion directions."""

    weights = directed_weights.index_select(0, labels).to(device=logits.device, dtype=logits.dtype)
    valid = weights.sum(dim=1) > 0
    if not bool(valid.any()):
        return logits.new_zeros(())
    true_logits = logits.gather(1, labels.view(-1, 1))
    pair_loss = F.relu(float(margin) - (true_logits - logits)) * weights
    sample_loss = pair_loss.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    return sample_loss[valid].mean()


def build_graph_angular_weight_matrix(
    class_to_idx: dict[str, int],
    device: torch.device,
    *,
    preset: str = "none",
) -> torch.Tensor:
    """Build an undirected hard-confusion graph for classifier prototype angles."""

    weights = torch.zeros((len(class_to_idx), len(class_to_idx)), dtype=torch.float32)
    if str(preset) == "none":
        return weights.to(device=device)
    if str(preset) == "rscd_dry_roughness_v1":
        pairs = {
            ("dry_concrete_slight", "dry_concrete_severe"): 1.40,
            ("dry_concrete_slight", "dry_concrete_smooth"): 0.90,
            ("dry_concrete_severe", "dry_concrete_smooth"): 0.70,
            ("dry_asphalt_slight", "dry_asphalt_severe"): 1.00,
            ("dry_asphalt_slight", "dry_asphalt_smooth"): 0.75,
            ("dry_asphalt_severe", "dry_asphalt_smooth"): 0.55,
        }
        missing = []
        for (a, b), weight in pairs.items():
            if a not in class_to_idx or b not in class_to_idx:
                missing.append((a, b))
                continue
            i = class_to_idx[a]
            j = class_to_idx[b]
            weights[i, j] = float(weight)
            weights[j, i] = float(weight)
        if missing:
            raise ValueError(f"graph angular preset has missing classes: {missing}")
        return weights.to(device=device)
    directed = build_directed_confusion_weight_matrix(class_to_idx, device=torch.device("cpu"), preset=preset)
    weights = torch.maximum(directed, directed.T)
    weights.fill_diagonal_(0.0)
    return weights.to(device=device)


def graph_angular_regularization_loss(
    model: nn.Module,
    graph_weights: torch.Tensor,
    *,
    max_cosine: float,
) -> torch.Tensor:
    """Separate hard-confusion classifier prototypes on the unit hypersphere."""

    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        return graph_weights.new_zeros(())
    weight = F.normalize(classifier.weight.float(), dim=1)
    graph = graph_weights.to(device=weight.device, dtype=weight.dtype)
    valid = graph > 0
    if not bool(valid.any()):
        return weight.new_zeros(())
    cosine = weight @ weight.T
    penalty = F.relu(cosine - float(max_cosine)).square() * graph
    return penalty[valid].sum() / graph[valid].sum().clamp_min(1e-6)


def factor_marginal_loss(
    logits: torch.Tensor,
    batch: dict[str, Any],
    masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Supervise factor marginals derived from the final class distribution.

    Unlike an independent auxiliary head, this loss acts directly on the 27-class
    logits: for each factor value, it sums the probability mass of all compatible
    classes. This makes the fine classifier respect the RSCD physical label
    graph while preserving the original benchmark prediction space.
    """

    log_prob = F.log_softmax(logits, dim=1)
    losses = []
    for name, mask in masks.items():
        target = batch[f"{name}_factor"].to(device=logits.device)
        valid = target >= 0
        if not bool(valid.any()):
            continue
        masked = log_prob.unsqueeze(1).masked_fill(~mask.unsqueeze(0), -1.0e4)
        marginal_log_prob = torch.logsumexp(masked, dim=2)
        losses.append(F.nll_loss(marginal_log_prob.index_select(0, valid.nonzero(as_tuple=False).flatten()), target[valid]))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def roughness_ordinal_energy_loss(
    logits: torch.Tensor,
    batch: dict[str, Any],
    masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Ordinal roughness supervision from the 27-class energy landscape.

    RSCD's smooth/slight/severe labels are ordered. This loss projects the
    class logits into unevenness energies and supervises two cumulative
    decisions: y >= slight and y >= severe. It targets the dominant audited
    error mode without adding any inference-time head.
    """

    if "unevenness" not in masks:
        return logits.new_zeros(())
    target = batch["unevenness_factor"].to(device=logits.device)
    valid = (target >= 0) & (target < 3)
    if not bool(valid.any()):
        return logits.new_zeros(())
    rough_mask = masks["unevenness"].to(device=logits.device)
    rough_logits = torch.logsumexp(
        logits.unsqueeze(1).masked_fill(~rough_mask.unsqueeze(0), -1.0e4),
        dim=2,
    )
    selected = rough_logits.index_select(0, valid.nonzero(as_tuple=False).flatten())
    y = target[valid]
    at_least_slight = torch.logsumexp(selected[:, 1:3], dim=1) - selected[:, 0]
    at_least_severe = selected[:, 2] - torch.logsumexp(selected[:, 0:2], dim=1)
    ordinal_logits = torch.stack([at_least_slight, at_least_severe], dim=1)
    ordinal_targets = torch.stack([(y >= 1), (y >= 2)], dim=1).to(dtype=ordinal_logits.dtype)
    return F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets)


def hierarchical_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor | None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if targets is None:
        loss = F.cross_entropy(logits, labels, reduction="none")
    else:
        soft_target = targets.index_select(0, labels)
        loss = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1)
    if class_weights is None:
        return loss.mean()
    sample_weight = class_weights.index_select(0, labels).to(device=logits.device, dtype=loss.dtype)
    return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


def factor_neighbor_prior_utility_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prior_bias: torch.Tensor,
    neighbor_mask: torch.Tensor,
    hard_mask: torch.Tensor,
    *,
    topk: int,
    margin: float,
    lam: float,
    mode: str,
) -> torch.Tensor:
    """Local Bayes-utility calibration loss for RSCD factor-neighbor decisions.

    This is not a global prior correction. It only simulates the natural RSCD
    prior inside uncertain top-k candidates that are factor-neighbors of the
    current prediction. The target remains the true class, so low-prior classes
    must learn a larger margin when they are visually valid.
    """

    if float(lam) == 0.0 or int(topk) < 2:
        return logits.new_zeros(())
    num_classes = int(logits.shape[1])
    k = min(max(int(topk), 2), num_classes)
    with torch.no_grad():
        top_values, top_indices = logits.detach().topk(k, dim=1)
        anchor = top_indices[:, 0]
        uncertain = (top_values[:, 0] - top_values[:, -1]) <= float(margin)
        candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
        candidate_mask.scatter_(1, top_indices, True)
        allowed = neighbor_mask.index_select(0, anchor).to(device=logits.device) & candidate_mask
        if str(mode) == "neighbor_hard":
            anchor_hard = hard_mask.index_select(0, anchor).view(-1, 1)
            class_hard = hard_mask.view(1, -1).to(device=logits.device)
            allowed = allowed & (anchor_hard | class_hard)
        elif str(mode) != "neighbor":
            raise ValueError(f"unknown factor-neighbor prior utility mode: {mode}")
        allowed.scatter_(1, anchor.view(-1, 1), True)
        label_allowed = allowed.gather(1, labels.view(-1, 1)).squeeze(1)
        active = uncertain & label_allowed
    if not bool(active.any()):
        return logits.new_zeros(())
    adjustment = torch.zeros_like(logits)
    adjustment = adjustment.masked_fill(allowed, 1.0) * prior_bias.to(device=logits.device, dtype=logits.dtype).view(1, -1)
    adjusted_logits = logits + float(lam) * adjustment
    per_sample = F.cross_entropy(adjusted_logits, labels, reduction="none")
    active_f = active.to(dtype=per_sample.dtype)
    return (per_sample * active_f).sum() / active_f.sum().clamp_min(1.0)


def load_teacher_probs(path: Path | None, num_classes: int) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Teacher probability file not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "image_path" not in data.files or "probs" not in data.files:
        raise ValueError(f"{path} must contain image_path and probs arrays.")
    paths = [str(x) for x in data["image_path"].tolist()]
    probs = np.asarray(data["probs"], dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != int(num_classes):
        raise ValueError(f"Teacher probs must have shape [N,{num_classes}], got {probs.shape}.")
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.clip(row_sums, 1e-8, None)
    out = {path_text: probs[idx] for idx, path_text in enumerate(paths)}
    print(f"Loaded teacher probabilities: {len(out)} rows from {path}", flush=True)
    return out


def teacher_batch_probs(
    teacher_probs: dict[str, np.ndarray],
    image_paths: list[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
    missing_policy: str = "error",
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    present = []
    missing = []
    for idx, path in enumerate(image_paths):
        value = teacher_probs.get(str(path))
        if value is None:
            missing.append(str(path))
        else:
            rows.append(value)
            present.append(idx)
    if missing and missing_policy == "error":
        preview = "; ".join(missing[:3])
        raise KeyError(f"Teacher probabilities missing for {len(missing)} batch paths, first: {preview}")
    if not rows:
        return (
            torch.empty((0, 0), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.long),
        )
    arr = np.stack(rows, axis=0).astype(np.float32)
    return (
        torch.as_tensor(arr, device=device, dtype=dtype),
        torch.as_tensor(present, device=device, dtype=torch.long),
    )


def _weighted_sample_mean(values: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return values.mean()
    weights = sample_weight.to(device=values.device, dtype=values.dtype).flatten()
    if weights.numel() != values.numel():
        raise ValueError("sample_weight must have one value per sample.")
    if float(weights.sum().detach().cpu()) <= 1e-8:
        return values.new_zeros(())
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def distillation_kl_loss(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-3)
    log_prob = F.log_softmax(logits / temperature, dim=1)
    teacher = teacher.clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    per_sample = F.kl_div(log_prob, teacher, reduction="none").sum(dim=1)
    return _weighted_sample_mean(per_sample, sample_weight) * (temperature * temperature)


def positive_congruent_distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    *,
    beta: float = 4.0,
    min_confidence: float = 0.0,
) -> torch.Tensor:
    """Preserve teacher-correct examples to reduce negative flips.

    Ordinary KD averages over all teacher rows. For an already strong anchor,
    the damaging update is specifically old-correct/new-wrong. This loss
    therefore filters to teacher-correct samples, then applies confidence-focal
    KL so high-certainty anchor decisions are harder for a specialist update to
    overwrite.
    """

    if logits.numel() == 0 or teacher.numel() == 0:
        return logits.new_zeros(())
    teacher = teacher.clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_conf, teacher_pred = teacher.max(dim=1)
    keep = teacher_pred.eq(labels)
    if float(min_confidence) > 0.0:
        keep = keep & teacher_conf.ge(float(min_confidence))
    if not bool(keep.any()):
        return logits.new_zeros(())
    selected_logits = logits[keep]
    selected_teacher = teacher[keep]
    selected_conf = teacher_conf[keep].to(dtype=logits.dtype)
    temperature = max(float(temperature), 1e-3)
    log_prob = F.log_softmax(selected_logits / temperature, dim=1)
    per_sample = F.kl_div(log_prob, selected_teacher, reduction="none").sum(dim=1)
    focal = 1.0 + max(float(beta), 0.0) * selected_conf
    return (per_sample * focal).sum() / focal.sum().clamp_min(1e-6) * (temperature * temperature)


def teacher_error_replay_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher: torch.Tensor,
    class_mask: torch.Tensor,
    *,
    beta: float = 1.0,
    min_confidence: float = 0.0,
) -> torch.Tensor:
    """Replay only anchor failures in audited RSCD hard cells.

    Top-1 is currently limited by sample-weighted hard boundaries, but broad
    class reweighting harms the macro-friendly wet/water/severe behavior.
    This loss uses the frozen anchor as a selector: if the anchor is correct,
    no extra CE is added; if it is wrong on a permitted hard-cell label, the
    student receives a small supervised correction.
    """

    if logits.numel() == 0 or teacher.numel() == 0:
        return logits.new_zeros(())
    teacher = teacher.clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_conf, teacher_pred = teacher.max(dim=1)
    allowed = class_mask.to(device=labels.device).index_select(0, labels).bool()
    keep = teacher_pred.ne(labels) & allowed
    if float(min_confidence) > 0.0:
        keep = keep & teacher_conf.ge(float(min_confidence))
    if not bool(keep.any()):
        return logits.new_zeros(())
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    selected_conf = teacher_conf[keep].to(dtype=logits.dtype).clamp(0.0, 1.0)
    weight = selected_conf.pow(max(float(beta), 0.0))
    if float(beta) <= 0.0:
        weight = torch.ones_like(selected_conf)
    return (per_sample[keep] * weight).sum() / weight.sum().clamp_min(1e-6)


def factor_decoupled_distillation_loss(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    masks: dict[str, torch.Tensor],
    temperature: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distill RSCD teacher probabilities after factor-wise marginalization.

    Direct 27-class distillation transfers teacher uncertainty across all
    coupled labels. This loss first projects teacher and student distributions
    onto friction/material/unevenness factors, then matches those lower-order
    distributions. It is a conservative way to transfer a TTA/ensemble teacher
    without smoothing heterophilic hard-neighbor classes together.
    """

    if logits.numel() == 0 or teacher.numel() == 0:
        return logits.new_zeros(())
    temperature = max(float(temperature), 1e-3)
    log_prob = F.log_softmax(logits / temperature, dim=1)
    teacher = teacher.clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    losses = []
    for mask in masks.values():
        factor_mask = mask.to(device=logits.device)
        student_masked = log_prob.unsqueeze(1).masked_fill(~factor_mask.unsqueeze(0), -1.0e4)
        student_log_marginal = torch.logsumexp(student_masked, dim=2)
        student_valid = student_log_marginal > -9999.0
        student_log_marginal = student_log_marginal.masked_fill(~student_valid, -1.0e4)
        student_log_marginal = student_log_marginal - torch.logsumexp(student_log_marginal, dim=1, keepdim=True)

        teacher_marginal = torch.matmul(teacher.to(dtype=logits.dtype), factor_mask.to(dtype=logits.dtype).T)
        valid = teacher_marginal.sum(dim=1, keepdim=True) > 1e-8
        teacher_marginal = teacher_marginal / teacher_marginal.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if not bool(valid.any()):
            continue
        per_sample = F.kl_div(student_log_marginal, teacher_marginal, reduction="none").sum(dim=1)
        valid_sample_weight = sample_weight
        if valid_sample_weight is not None:
            valid_sample_weight = valid_sample_weight.to(device=logits.device, dtype=per_sample.dtype) * valid.flatten().to(
                dtype=per_sample.dtype
            )
            if float(valid_sample_weight.sum().detach().cpu()) <= 1e-8:
                continue
        else:
            per_sample = per_sample[valid.flatten()]
        losses.append(_weighted_sample_mean(per_sample, valid_sample_weight))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean() * (temperature * temperature)


def symmetric_logit_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric KL consistency between two augmented views."""

    temperature = max(float(temperature), 1e-3)
    log_pa = F.log_softmax(logits_a / temperature, dim=1)
    log_pb = F.log_softmax(logits_b / temperature, dim=1)
    pa = log_pa.exp().detach()
    pb = log_pb.exp().detach()
    loss_ab = F.kl_div(log_pa, pb, reduction="batchmean")
    loss_ba = F.kl_div(log_pb, pa, reduction="batchmean")
    return 0.5 * (loss_ab + loss_ba) * (temperature * temperature)


def symmetric_feature_consistency_loss(
    feature_a: torch.Tensor | None,
    feature_b: torch.Tensor | None,
) -> torch.Tensor | None:
    """Scale-invariant consistency between two physics evidence feature views."""

    if feature_a is None or feature_b is None:
        return None
    if feature_a.shape != feature_b.shape or feature_a.numel() == 0:
        return None
    a = F.normalize(feature_a.float(), dim=1)
    b = F.normalize(feature_b.float(), dim=1)
    return F.mse_loss(a, b)


def random_normalized_block_mask(
    image: torch.Tensor,
    *,
    mask_ratio: float,
    block_frac: float,
    max_blocks: int,
    mask_value: str,
) -> torch.Tensor:
    """Cutout-style masking in normalized image space for consistency training."""

    if image.ndim != 4 or image.numel() == 0:
        return image
    ratio = max(0.0, min(float(mask_ratio), 0.75))
    if ratio <= 0.0:
        return image
    b, c, h, w = image.shape
    block_frac = max(0.04, min(float(block_frac), 0.75))
    block_h = max(1, min(h, int(round(h * block_frac))))
    block_w = max(1, min(w, int(round(w * block_frac))))
    target_area = max(1, int(round(ratio * h * w)))
    max_blocks = max(int(max_blocks), 1)
    mode = str(mask_value or "mean").lower()
    out = image.clone()
    for i in range(b):
        if mode == "zero":
            fill = out.new_zeros((c, 1, 1))
        elif mode == "random":
            fill = torch.randn((c, 1, 1), device=out.device, dtype=out.dtype)
        else:
            fill = out[i].mean(dim=(1, 2), keepdim=True)
        masked_area = 0
        for _ in range(max_blocks):
            y0 = int(torch.randint(0, max(h - block_h + 1, 1), (1,), device=out.device).item())
            x0 = int(torch.randint(0, max(w - block_w + 1, 1), (1,), device=out.device).item())
            out[i, :, y0 : y0 + block_h, x0 : x0 + block_w] = fill
            masked_area += block_h * block_w
            if masked_area >= target_area:
                break
    return out


def physics_protected_normalized_block_mask(
    image: torch.Tensor,
    *,
    mask_ratio: float,
    block_frac: float,
    max_blocks: int,
    mask_value: str,
    protect_gamma: float = 2.0,
    nuisance_weight: float = 0.65,
) -> torch.Tensor:
    """Mask nuisance-like regions while preserving road-surface evidence.

    This is a task-adapted masked consistency view for RSCD. Random blocks are
    softened where PhysicsTexture-style evidence suggests wet film, texture
    erasure, granular particles, snow/ice, or visible roughness. Bright
    low-saturation line-like artifacts remain maskable unless they look like
    snow/ice evidence.
    """

    if image.ndim != 4 or image.numel() == 0 or image.size(1) != 3:
        return random_normalized_block_mask(
            image,
            mask_ratio=mask_ratio,
            block_frac=block_frac,
            max_blocks=max_blocks,
            mask_value=mask_value,
        )
    ratio = max(0.0, min(float(mask_ratio), 0.75))
    if ratio <= 0.0:
        return image
    bsz, channels, height, width = image.shape
    block_frac = max(0.04, min(float(block_frac), 0.75))
    block_h = max(1, min(height, int(round(height * block_frac))))
    block_w = max(1, min(width, int(round(width * block_frac))))
    target_area = max(1, int(round(ratio * height * width)))
    max_blocks = max(int(max_blocks), 1)

    base_mask = image.new_zeros((bsz, 1, height, width))
    for i in range(bsz):
        masked_area = 0
        for _ in range(max_blocks):
            y0 = int(torch.randint(0, max(height - block_h + 1, 1), (1,), device=image.device).item())
            x0 = int(torch.randint(0, max(width - block_w + 1, 1), (1,), device=image.device).item())
            base_mask[i, :, y0 : y0 + block_h, x0 : x0 + block_w] = 1.0
            masked_area += block_h * block_w
            if masked_area >= target_area:
                break

    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = (image * std + mean).clamp(0.0, 1.0)
    value = rgb.max(dim=1, keepdim=True).values
    min_value = rgb.min(dim=1, keepdim=True).values
    saturation = ((value - min_value) / value.clamp_min(1e-4)).clamp(0.0, 1.0)
    low_sat = (1.0 - saturation).clamp(0.0, 1.0)
    gray = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]).clamp(0.0, 1.0)

    local_mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
    local_abs = F.avg_pool2d((gray - local_mean).abs(), kernel_size=7, stride=1, padding=3)
    grad_x = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)

    rough_evidence = ((local_abs - 0.030) / 0.120).clamp(0.0, 1.0)
    granular_evidence = (rough_evidence * ((grad - 0.025) / 0.120).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    snow_evidence = (((value - 0.62) / 0.28).clamp(0.0, 1.0) * low_sat * ((0.120 - local_abs) / 0.120).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    specular_evidence = (((value - 0.74) / 0.24).clamp(0.0, 1.0) * low_sat * ((0.090 - local_abs) / 0.090).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    dark_water_evidence = (((0.42 - value) / 0.42).clamp(0.0, 1.0) * ((0.100 - local_abs) / 0.100).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    texture_erasure = (((0.070 - local_abs) / 0.070).clamp(0.0, 1.0) * ((0.080 - grad) / 0.080).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    thin_film = (0.55 * specular_evidence + 0.45 * texture_erasure).clamp(0.0, 1.0)

    protect = (
        0.50 * rough_evidence
        + 0.55 * granular_evidence
        + 0.65 * snow_evidence
        + 0.70 * thin_film
        + 0.65 * dark_water_evidence
        + 0.45 * texture_erasure
    ).clamp(0.0, 1.0)
    line_like_nuisance = (
        ((value - 0.68) / 0.28).clamp(0.0, 1.0)
        * low_sat
        * ((grad - 0.035) / 0.120).clamp(0.0, 1.0)
        * (1.0 - snow_evidence)
        * (1.0 - rough_evidence)
    ).clamp(0.0, 1.0)

    gamma = max(float(protect_gamma), 0.25)
    mask = base_mask * (1.0 - protect).clamp(0.0, 1.0).pow(gamma)
    mask = torch.maximum(mask, float(nuisance_weight) * base_mask * line_like_nuisance)
    mask = mask.clamp(0.0, 1.0)

    mode = str(mask_value or "mean").lower()
    if mode == "zero":
        fill = image.new_zeros((bsz, channels, 1, 1))
    elif mode == "random":
        fill = torch.randn((bsz, channels, 1, 1), device=image.device, dtype=image.dtype)
    else:
        fill = image.mean(dim=(2, 3), keepdim=True)
    return image * (1.0 - mask) + fill * mask


def masked_view_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    temperature: float,
    class_weight: float,
    factor_weight: float,
    confidence_threshold: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match masked-view predictions to clean-view class and factor marginals."""

    if class_weight <= 0.0 and factor_weight <= 0.0:
        return student_logits.new_zeros(())
    temperature = max(float(temperature), 1e-3)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    if confidence_threshold > 0.0:
        keep = teacher_prob.max(dim=1).values >= float(confidence_threshold)
    else:
        keep = torch.ones(teacher_prob.shape[0], device=teacher_prob.device, dtype=torch.bool)
    if not bool(keep.any()):
        return student_logits.new_zeros(())
    selected_student = student_logits.index_select(0, keep.nonzero(as_tuple=False).flatten())
    selected_teacher = teacher_prob.index_select(0, keep.nonzero(as_tuple=False).flatten())
    selected_weight = None
    if sample_weight is not None:
        selected_weight = sample_weight.to(device=student_logits.device, dtype=student_logits.dtype).index_select(
            0, keep.nonzero(as_tuple=False).flatten()
        )
        if float(selected_weight.sum().detach().cpu()) <= 1e-8:
            return student_logits.new_zeros(())
    terms = []
    if class_weight > 0.0:
        terms.append(
            float(class_weight)
            * distillation_kl_loss(selected_student, selected_teacher, temperature, sample_weight=selected_weight)
        )
    if factor_weight > 0.0:
        terms.append(
            float(factor_weight)
            * factor_decoupled_distillation_loss(
                selected_student,
                selected_teacher,
                masks,
                temperature,
                sample_weight=selected_weight,
            )
        )
    if not terms:
        return student_logits.new_zeros(())
    return torch.stack(terms).sum()


def random_observer_disturbance(
    image: torch.Tensor,
    *,
    mode: str,
    strength: float,
    max_lines: int,
    block_ratio: float,
) -> torch.Tensor:
    """Create control-theory-style nuisance disturbances in normalized RGB space.

    The perturbations approximate exogenous inputs that should not change the
    RSCD latent road state: line-like markings, exposure/color drift, and small
    local occlusions. The function denormalizes first so the disturbance remains
    physically interpretable, then returns an ImageNet-normalized tensor.
    """

    if image.ndim != 4 or image.size(1) != 3 or image.numel() == 0:
        return image
    mode = str(mode).lower()
    if mode not in {"mixed", "line", "illumination", "mask"}:
        raise ValueError(f"unknown observer disturbance mode: {mode}")
    strength = max(float(strength), 0.0)
    if strength <= 0.0:
        return image
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = (image * std + mean).clamp(0.0, 1.0)
    out = rgb.clone()
    bsz, _, height, width = out.shape

    use_illum = mode in {"mixed", "illumination"}
    use_line = mode in {"mixed", "line"}
    use_mask = mode in {"mixed", "mask"}
    if use_illum:
        gain = 1.0 + (torch.rand((bsz, 1, 1, 1), device=out.device, dtype=out.dtype) * 2.0 - 1.0) * (0.22 * strength)
        bias = (torch.rand((bsz, 1, 1, 1), device=out.device, dtype=out.dtype) * 2.0 - 1.0) * (0.08 * strength)
        channel = 1.0 + (torch.rand((bsz, 3, 1, 1), device=out.device, dtype=out.dtype) * 2.0 - 1.0) * (0.06 * strength)
        out = (out * gain * channel + bias).clamp(0.0, 1.0)

    if use_line:
        line_count = max(int(max_lines), 1)
        for i in range(bsz):
            n_lines = int(torch.randint(1, line_count + 1, (1,), device=out.device).item())
            fill_mean = out[i].mean(dim=(1, 2), keepdim=True)
            for _ in range(n_lines):
                horizontal = bool(torch.rand((), device=out.device) < 0.5)
                if horizontal:
                    line_h = max(1, int(round(height * float(torch.empty((), device=out.device).uniform_(0.010, 0.040).item()) * strength)))
                    line_w = max(1, int(round(width * float(torch.empty((), device=out.device).uniform_(0.35, 0.95).item()))))
                    top = int(torch.randint(0, max(height - line_h + 1, 1), (1,), device=out.device).item())
                    left = int(torch.randint(0, max(width - line_w + 1, 1), (1,), device=out.device).item())
                    y_slice = slice(top, top + line_h)
                    x_slice = slice(left, left + line_w)
                else:
                    line_h = max(1, int(round(height * float(torch.empty((), device=out.device).uniform_(0.35, 0.95).item()))))
                    line_w = max(1, int(round(width * float(torch.empty((), device=out.device).uniform_(0.010, 0.040).item()) * strength)))
                    top = int(torch.randint(0, max(height - line_h + 1, 1), (1,), device=out.device).item())
                    left = int(torch.randint(0, max(width - line_w + 1, 1), (1,), device=out.device).item())
                    y_slice = slice(top, top + line_h)
                    x_slice = slice(left, left + line_w)
                if bool(torch.rand((), device=out.device) < 0.55):
                    fill = out.new_full((3, 1, 1), 0.82 + 0.14 * float(torch.rand((), device=out.device).item()))
                else:
                    fill = fill_mean
                alpha = min(0.85, 0.35 + 0.35 * strength)
                out[i, :, y_slice, x_slice] = (1.0 - alpha) * out[i, :, y_slice, x_slice] + alpha * fill

    if use_mask and block_ratio > 0.0:
        ratio = max(0.0, min(float(block_ratio) * strength, 0.55))
        block_h = max(1, int(round(height * max(0.05, min(0.32, ratio)))))
        block_w = max(1, int(round(width * max(0.05, min(0.32, ratio)))))
        for i in range(bsz):
            top = int(torch.randint(0, max(height - block_h + 1, 1), (1,), device=out.device).item())
            left = int(torch.randint(0, max(width - block_w + 1, 1), (1,), device=out.device).item())
            fill = out[i].mean(dim=(1, 2), keepdim=True)
            out[i, :, top : top + block_h, left : left + block_w] = fill
    return ((out.clamp(0.0, 1.0) - mean) / std).to(dtype=image.dtype)


def observer_disturbance_gain_loss(
    clean_feature: torch.Tensor | None,
    disturbed_feature: torch.Tensor | None,
    clean_image: torch.Tensor,
    disturbed_image: torch.Tensor,
    *,
    rho: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Bound representation gain from nuisance image disturbance to feature drift."""

    if clean_feature is None or disturbed_feature is None:
        return None
    if clean_feature.shape != disturbed_feature.shape or clean_feature.numel() == 0:
        return None
    clean = clean_feature.float().flatten(1)
    disturbed = disturbed_feature.float().flatten(1)
    clean = F.normalize(clean.detach(), dim=1)
    disturbed = F.normalize(disturbed, dim=1)
    feature_energy = (disturbed - clean).square().mean(dim=1)
    disturbance_energy = (
        disturbed_image.float() - clean_image.float()
    ).square().mean(dim=(1, 2, 3)).detach().clamp_min(1e-4)
    gain = torch.nan_to_num(feature_energy / disturbance_energy, nan=0.0, posinf=50.0, neginf=0.0)
    return _weighted_sample_mean(F.relu(gain - float(rho)).square(), sample_weight)


def observer_margin_barrier_loss(
    clean_logits: torch.Tensor,
    disturbed_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin_drop: float,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Control-barrier-style no-harm margin preservation under disturbance."""

    if clean_logits.numel() == 0 or disturbed_logits.shape != clean_logits.shape:
        return clean_logits.new_zeros(())
    labels = labels.to(device=clean_logits.device)
    clean_true = clean_logits.gather(1, labels.view(-1, 1)).squeeze(1)
    disturbed_true = disturbed_logits.gather(1, labels.view(-1, 1)).squeeze(1)
    one_hot = F.one_hot(labels, num_classes=clean_logits.shape[1]).to(dtype=torch.bool, device=clean_logits.device)
    clean_other = clean_logits.masked_fill(one_hot, -1.0e4).max(dim=1).values
    disturbed_other = disturbed_logits.masked_fill(one_hot, -1.0e4).max(dim=1).values
    clean_margin = (clean_true - clean_other).detach()
    disturbed_margin = disturbed_true - disturbed_other
    barrier_violation = F.relu(clean_margin - float(margin_drop) - disturbed_margin).square()
    return _weighted_sample_mean(barrier_violation, sample_weight)


def observer_hinf_robustness_loss(
    clean_out: dict[str, Any],
    disturbed_out: dict[str, Any],
    clean_image: torch.Tensor,
    disturbed_image: torch.Tensor,
    labels: torch.Tensor,
    factor_marginal_masks: dict[str, torch.Tensor],
    criterion: nn.Module,
    *,
    rho: float,
    feature_weight: float,
    physics_feature_weight: float,
    local_physics_feature_weight: float,
    low_level_feature_weight: float,
    class_consistency_weight: float,
    factor_consistency_weight: float,
    disturbed_ce_weight: float,
    barrier_weight: float,
    barrier_margin_drop: float,
    temperature: float,
    confidence_threshold: float,
    sample_weight: torch.Tensor | None = None,
    barrier_sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Observer/H-infinity robust training for RSCD physical state estimates."""

    logits = clean_out["logits"]
    disturbed_logits = disturbed_out["logits"]
    terms: list[torch.Tensor] = []
    feature_specs = (
        ("feature", float(feature_weight)),
        ("physics_feature", float(physics_feature_weight)),
        ("local_physics_feature", float(local_physics_feature_weight)),
        ("low_level_feature", float(low_level_feature_weight)),
    )
    for key, weight in feature_specs:
        if weight <= 0.0:
            continue
        value = observer_disturbance_gain_loss(
            clean_out.get(key),
            disturbed_out.get(key),
            clean_image,
            disturbed_image,
            rho=float(rho),
            sample_weight=sample_weight,
        )
        if value is not None:
            terms.append(float(weight) * value)
    if class_consistency_weight > 0.0 or factor_consistency_weight > 0.0:
        terms.append(
            masked_view_consistency_loss(
                disturbed_logits,
                logits,
                factor_marginal_masks,
                temperature=float(temperature),
                class_weight=float(class_consistency_weight),
                factor_weight=float(factor_consistency_weight),
                confidence_threshold=float(confidence_threshold),
                sample_weight=sample_weight,
            )
        )
    if disturbed_ce_weight > 0.0:
        per_sample_ce = F.cross_entropy(disturbed_logits, labels, reduction="none")
        terms.append(float(disturbed_ce_weight) * _weighted_sample_mean(per_sample_ce, sample_weight))
    if barrier_weight > 0.0:
        terms.append(
            float(barrier_weight)
            * observer_margin_barrier_loss(
                logits,
                disturbed_logits,
                labels,
                margin_drop=float(barrier_margin_drop),
                sample_weight=barrier_sample_weight,
            )
        )
    if not terms:
        return logits.new_zeros(())
    return torch.stack(terms).sum()


def _factor_aux_sample_weights(batch: dict[str, Any], device: torch.device, focus: str) -> torch.Tensor | None:
    focus = str(focus)
    if focus == "all":
        return None
    friction = batch["friction_factor"].to(device)
    material = batch["material_factor"].to(device)
    unevenness = batch["unevenness_factor"].to(device)
    weights = torch.ones_like(friction, dtype=torch.float32)
    f_idx = {name: FACTOR_LABELS["friction"].index(name) for name in FACTOR_LABELS["friction"]}
    m_idx = {name: FACTOR_LABELS["material"].index(name) for name in FACTOR_LABELS["material"]}
    if focus == "dry_only":
        weights = (friction == f_idx["dry"]).to(dtype=torch.float32)
    elif focus == "non_wet_water":
        weights = ((friction != f_idx["wet"]) & (friction != f_idx["water"]) & (friction >= 0)).to(dtype=torch.float32)
    elif focus == "dry_paved":
        paved = (material == m_idx["asphalt"]) | (material == m_idx["concrete"])
        weights = ((friction == f_idx["dry"]) & paved & (unevenness >= 0)).to(dtype=torch.float32)
    elif focus == "wet_water_concrete":
        weights = (
            ((friction == f_idx["wet"]) | (friction == f_idx["water"]))
            & (material == m_idx["concrete"])
            & (unevenness >= 0)
        ).to(dtype=torch.float32)
    elif focus == "water_concrete":
        weights = ((friction == f_idx["water"]) & (material == m_idx["concrete"]) & (unevenness >= 0)).to(
            dtype=torch.float32
        )
    elif focus == "wet_concrete":
        weights = ((friction == f_idx["wet"]) & (material == m_idx["concrete"]) & (unevenness >= 0)).to(
            dtype=torch.float32
        )
    elif focus == "water_asphalt":
        weights = ((friction == f_idx["water"]) & (material == m_idx["asphalt"]) & (unevenness >= 0)).to(
            dtype=torch.float32
        )
    elif focus == "wet_asphalt":
        weights = ((friction == f_idx["wet"]) & (material == m_idx["asphalt"]) & (unevenness >= 0)).to(
            dtype=torch.float32
        )
    elif focus == "granular":
        weights = ((material == m_idx["mud"]) | (material == m_idx["gravel"])).to(dtype=torch.float32)
    elif focus == "winter":
        weights = (
            (friction == f_idx["fresh_snow"])
            | (friction == f_idx["melted_snow"])
            | (friction == f_idx["ice"])
        ).to(dtype=torch.float32)
    else:
        raise ValueError(f"unknown factor aux focus: {focus}")
    return weights


def _factor_aux_axis_names(axis_focus: str) -> tuple[str, ...]:
    axis_focus = str(axis_focus)
    mapping = {
        "all": ("friction", "material", "unevenness"),
        "friction": ("friction",),
        "material": ("material",),
        "unevenness": ("unevenness",),
        "roughness": ("unevenness",),
        "friction_material": ("friction", "material"),
        "friction_unevenness": ("friction", "unevenness"),
        "friction_roughness": ("friction", "unevenness"),
        "material_unevenness": ("material", "unevenness"),
        "material_roughness": ("material", "unevenness"),
    }
    if axis_focus not in mapping:
        raise ValueError(f"unknown factor aux axis focus: {axis_focus}")
    return mapping[axis_focus]


def factor_auxiliary_loss(
    logits_by_factor: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    device: torch.device,
    focus: str = "all",
    axis_focus: str = "all",
) -> torch.Tensor | None:
    sample_weights = _factor_aux_sample_weights(batch, device, focus)
    losses = []
    for name in _factor_aux_axis_names(axis_focus):
        target = batch[f"{name}_factor"].to(device)
        valid = target != -1
        if not bool(valid.any()):
            continue
        per_sample = F.cross_entropy(logits_by_factor[name], target, ignore_index=-1, reduction="none")
        if sample_weights is not None:
            weights = sample_weights.to(device=device, dtype=per_sample.dtype) * valid.to(dtype=per_sample.dtype)
            if float(weights.sum().detach().cpu()) <= 1e-8:
                continue
            losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1e-6))
        else:
            losses.append(per_sample[valid].mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def mechanism_targets_from_batch(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Map RSCD factor labels to visual coupling mechanisms.

    Mechanism ids:
    0 dry_visible, 1 wet_film, 2 water_obstruction, 3 granular, 4 winter.
    """

    friction = batch["friction_factor"].to(device)
    material = batch["material_factor"].to(device)
    target = torch.full_like(friction, -1)
    f_idx = {name: FACTOR_LABELS["friction"].index(name) for name in FACTOR_LABELS["friction"]}
    m_idx = {name: FACTOR_LABELS["material"].index(name) for name in FACTOR_LABELS["material"]}

    target[friction == f_idx["dry"]] = 0
    target[friction == f_idx["wet"]] = 1
    target[friction == f_idx["water"]] = 2
    granular = (material == m_idx["mud"]) | (material == m_idx["gravel"])
    target[granular] = 3
    winter = (
        (friction == f_idx["fresh_snow"])
        | (friction == f_idx["melted_snow"])
        | (friction == f_idx["ice"])
    )
    target[winter] = 4
    return target


def physics_evidence_field_weights(mode: str, device: torch.device) -> torch.Tensor | None:
    """Per-field weights for task-specific dense physics evidence supervision.

    Field order follows PhysicsEvidenceTarget:
    obstruction, visible_rough, hidden_rough, thin_film, texture_erasure,
    dry_rough, masked_concrete_rough, film_rough_coupling, granular_wet.
    """

    mode = str(mode)
    if mode == "all":
        return None
    if mode == "roughness_coupling":
        weights = [0.25, 1.15, 1.10, 0.35, 0.45, 1.00, 1.20, 1.20, 0.20]
    elif mode == "wet_concrete_hidden_roughness":
        weights = [0.30, 0.75, 1.45, 0.30, 0.55, 0.30, 1.55, 1.35, 0.05]
    elif mode == "granular_guard":
        weights = [0.20, 0.45, 0.25, 0.25, 0.30, 0.20, 0.20, 0.35, 1.80]
    else:
        raise ValueError(f"unknown physics evidence field mode: {mode}")
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def observer_hinf_scope_weights(batch: dict[str, Any], device: torch.device, scope: str) -> torch.Tensor | None:
    """Training-only sample weights for mechanism-scoped Observer-Hinf control.

    The global disturbance observer was too blunt: it improved dry concrete
    roughness but suppressed wet/water film evidence. These scopes translate
    RSCD's factorized label state into where a disturbance-attenuation
    constraint should act.
    """

    scope = str(scope)
    if scope == "all":
        return None
    if scope not in OBSERVER_HINF_SCOPES:
        raise ValueError(f"unknown Observer-Hinf scope: {scope}")
    friction = batch["friction_factor"].to(device)
    material = batch["material_factor"].to(device)
    unevenness = batch["unevenness_factor"].to(device)
    f_idx = {name: FACTOR_LABELS["friction"].index(name) for name in FACTOR_LABELS["friction"]}
    m_idx = {name: FACTOR_LABELS["material"].index(name) for name in FACTOR_LABELS["material"]}
    valid_friction = friction >= 0
    paved = ((material == m_idx["asphalt"]) | (material == m_idx["concrete"])) & (unevenness >= 0)
    dry = friction == f_idx["dry"]
    wet_water = (friction == f_idx["wet"]) | (friction == f_idx["water"])
    granular = (material == m_idx["mud"]) | (material == m_idx["gravel"])
    winter = (
        (friction == f_idx["fresh_snow"])
        | (friction == f_idx["melted_snow"])
        | (friction == f_idx["ice"])
    )

    if scope == "core_paved":
        keep = ((dry | wet_water) & paved)
        return keep.to(dtype=torch.float32)
    if scope == "dry_visible":
        return dry.to(dtype=torch.float32)
    if scope == "dry_paved_roughness":
        return (dry & paved).to(dtype=torch.float32)
    if scope == "wet_water_paved":
        return (wet_water & paved).to(dtype=torch.float32)
    if scope == "wet_water_concrete":
        return (wet_water & (material == m_idx["concrete"]) & (unevenness >= 0)).to(dtype=torch.float32)
    if scope == "granular":
        return granular.to(dtype=torch.float32)
    if scope == "winter":
        return winter.to(dtype=torch.float32)
    if scope == "non_wet_water":
        return (valid_friction & (~wet_water)).to(dtype=torch.float32)
    if scope == "wet_water_guarded":
        weights = torch.ones_like(friction, dtype=torch.float32)
        weights[wet_water] = 0.15
        weights[~valid_friction] = 0.0
        return weights
    if scope == "hard_audited":
        hard_classes = {
            "water_concrete_slight",
            "water_asphalt_slight",
            "wet_concrete_slight",
            "water_concrete_severe",
            "dry_concrete_slight",
            "wet_concrete_severe",
            "water_asphalt_severe",
            "water_gravel",
        }
        labels = [canonical_class_label(str(item)) in hard_classes for item in batch.get("class_label", [])]
        return torch.as_tensor(labels, device=device, dtype=torch.float32)
    raise ValueError(f"unknown Observer-Hinf scope: {scope}")


def _cross_covariance_orthogonal_loss(embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
    names = ("friction", "material", "unevenness", "coupling")
    loss: torch.Tensor | None = None
    count = 0
    for i, left in enumerate(names):
        z_left = F.normalize(embeddings[left], dim=1)
        z_left = z_left - z_left.mean(dim=0, keepdim=True)
        for right in names[i + 1 :]:
            z_right = F.normalize(embeddings[right], dim=1)
            z_right = z_right - z_right.mean(dim=0, keepdim=True)
            denom = max(int(z_left.shape[0]) - 1, 1)
            cov = z_left.transpose(0, 1) @ z_right / float(denom)
            value = cov.square().mean()
            loss = value if loss is None else loss + value
            count += 1
    if loss is None:
        return embeddings[names[0]].new_zeros(())
    return loss / float(max(count, 1))


def mechanism_orthogonal_coupling_auxiliary_loss(
    aux: dict[str, Any],
    batch: dict[str, Any],
    *,
    device: torch.device,
    factor_weight: float,
    mechanism_weight: float,
    cov_weight: float,
) -> torch.Tensor | None:
    """MOCA training-only loss for mechanism-conditioned factor disentangling."""

    if not aux:
        return None
    factor_logits = aux.get("factor_logits")
    embeddings = aux.get("embeddings")
    mechanism_logits = aux.get("mechanism_logits")
    if not isinstance(factor_logits, dict) or not isinstance(embeddings, dict):
        return None

    terms: list[torch.Tensor] = []
    if factor_weight > 0.0:
        factor_terms = []
        for name in ("friction", "material", "unevenness"):
            logits = factor_logits.get(name)
            if logits is None:
                continue
            target = batch[f"{name}_factor"].to(device=device)
            valid = target >= 0
            if valid.any():
                factor_terms.append(
                    F.cross_entropy(
                        logits.index_select(0, valid.nonzero(as_tuple=False).flatten()),
                        target[valid],
                    )
                )
        if factor_terms:
            terms.append(float(factor_weight) * torch.stack(factor_terms).mean())

    if mechanism_weight > 0.0 and isinstance(mechanism_logits, torch.Tensor):
        mechanism_target = mechanism_targets_from_batch(batch, device)
        valid = mechanism_target >= 0
        if valid.any():
            terms.append(
                float(mechanism_weight)
                * F.cross_entropy(
                    mechanism_logits.index_select(0, valid.nonzero(as_tuple=False).flatten()),
                    mechanism_target[valid],
                )
            )

    if cov_weight > 0.0:
        terms.append(float(cov_weight) * _cross_covariance_orthogonal_loss(embeddings))

    if not terms:
        return None
    return torch.stack(terms).sum()


def uncertainty_gated_graph_diffusion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    gate_mask: torch.Tensor,
    *,
    uncertainty_margin: float,
    gate_temperature: float,
) -> torch.Tensor:
    """Apply graph-smoothed targets only to uncertain hard-neighborhood samples."""

    if targets is None:
        return logits.new_zeros(())
    soft_target = targets.index_select(0, labels)
    per_sample = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1)
    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
    if top2.shape[1] < 2:
        margin = logits.new_zeros((logits.shape[0],))
    else:
        margin = top2[:, 0] - top2[:, 1]
    uncertainty_gate = torch.sigmoid((float(uncertainty_margin) - margin) * float(gate_temperature))
    class_gate = gate_mask.to(device=logits.device, dtype=logits.dtype).index_select(0, labels)
    weights = uncertainty_gate * class_gate
    if float(weights.sum().detach().cpu()) <= 1e-8:
        return logits.new_zeros(())
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "friction_factor": torch.stack([item["friction_factor"] for item in batch], dim=0),
        "material_factor": torch.stack([item["material_factor"] for item in batch], dim=0),
        "unevenness_factor": torch.stack([item["unevenness_factor"] for item in batch], dim=0),
        "class_label": [str(item["class_label"]) for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
    }


def set_frozen_modules_eval(module: nn.Module) -> None:
    """Keep frozen backbones deterministic during head-only training."""

    for child in module.modules():
        if any(parameter.requires_grad for parameter in child.parameters(recurse=True)):
            continue
        child.eval()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    factor_criterion: nn.Module,
    *,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_accum_steps: int,
    log_every_steps: int,
    class_loss_weight: float,
    factor_aux_weight: float,
    factor_aux_friction_focus: str,
    factor_aux_axis_focus: str,
    hard_pair_aux_weight: float,
    hard_pair_aux_left: torch.Tensor,
    hard_pair_aux_right: torch.Tensor,
    mechanism_orthogonal_aux_weight: float,
    mechanism_orthogonal_factor_weight: float,
    mechanism_orthogonal_mechanism_weight: float,
    mechanism_orthogonal_cov_weight: float,
    local_physics_factor_aux_weight: float,
    factor_marginal_weight: float,
    relation_conditional_weight: float,
    relation_conditional_masks: dict[str, dict[str, torch.Tensor]],
    relation_conditional_focus_mask: torch.Tensor,
    relation_conditional_axis_weights: dict[str, float],
    relation_conditional_class_axis_weights: torch.Tensor | None,
    relation_prototype_contrastive_weight: float,
    relation_prototype_contrastive_temperature: float,
    relation_conditional_uncertainty_margin: float,
    relation_conditional_gate_temperature: float,
    roughness_ordinal_weight: float,
    roughness_boundary_margin_weight: float,
    roughness_boundary_mask: torch.Tensor,
    roughness_boundary_margin: float,
    roughness_boundary_uncertainty_margin: float,
    roughness_boundary_gate_temperature: float,
    concrete_masked_roughness_weight: float,
    concrete_masked_roughness_masks: torch.Tensor,
    concrete_masked_roughness_obstruction_weight: float,
    concrete_masked_roughness_uncertainty_margin: float,
    concrete_masked_roughness_gate_temperature: float,
    tensor_anova_boundary_weight: float,
    tensor_anova_boundary_spec: dict[str, torch.Tensor],
    tensor_anova_boundary_focus: str,
    tensor_anova_boundary_margin: float,
    tensor_anova_boundary_uncertainty_margin: float,
    tensor_anova_boundary_gate_temperature: float,
    tensor_anova_boundary_friction_weight: float,
    tensor_anova_boundary_material_weight: float,
    tensor_anova_boundary_roughness_weight: float,
    tensor_anova_boundary_obstruction_weight: float,
    factor_marginal_masks: dict[str, torch.Tensor],
    factor_neighbor_margin_weight: float,
    factor_neighbor_loss_mode: str,
    factor_neighbor_negative_mask: torch.Tensor,
    factor_neighbor_weight_matrix: torch.Tensor,
    factor_neighbor_margin: float,
    factor_neighbor_contrastive_weight: float,
    factor_neighbor_contrastive_margin: float,
    factor_prototype_contrastive_weight: float,
    factor_prototype_contrastive_margin: float,
    factor_neighbor_roughness_weight: float,
    factor_neighbor_friction_weight: float,
    factor_neighbor_material_weight: float,
    factor_neighbor_wet_water_weight: float,
    factor_neighbor_concrete_weight: float,
    factor_neighbor_hard_class_weight: float,
    controlled_factor_tournament_weight: float,
    controlled_factor_tournament_temperature: float,
    controlled_factor_tournament_margin: float,
    controlled_factor_tournament_neg_weight: float,
    controlled_factor_tournament_focus: str,
    controlled_factor_tournament_friction_weight: float,
    controlled_factor_tournament_material_weight: float,
    controlled_factor_tournament_unevenness_weight: float,
    mechanism_controlled_factor_tournament_weight: float,
    mechanism_controlled_factor_tournament_temperature: float,
    mechanism_controlled_factor_tournament_margin: float,
    mechanism_controlled_factor_tournament_neg_weight: float,
    mechanism_controlled_factor_tournament_focus: str,
    mechanism_controlled_factor_tournament_friction_weight: float,
    mechanism_controlled_factor_tournament_material_weight: float,
    mechanism_controlled_factor_tournament_unevenness_weight: float,
    local_factor_graph_weight: float,
    local_factor_graph_margin: float,
    local_factor_graph_uncertainty_margin: float,
    local_factor_graph_gate_temperature: float,
    local_factor_graph_topk: int,
    relation_specific_clean_margin_weight: float,
    relation_specific_clean_margin_relation_ids: torch.Tensor,
    relation_specific_clean_margin_class_weights: torch.Tensor,
    relation_specific_clean_margin_protected_mask: torch.Tensor,
    relation_specific_clean_margin_friction_margin: float,
    relation_specific_clean_margin_roughness_margin: float,
    relation_specific_clean_margin_material_margin: float,
    relation_specific_clean_margin_friction_weight: float,
    relation_specific_clean_margin_roughness_weight: float,
    relation_specific_clean_margin_material_weight: float,
    relation_specific_clean_margin_protected_negative_scale: float,
    relation_specific_clean_margin_uncertainty_margin: float,
    relation_specific_clean_margin_gate_temperature: float,
    relation_specific_clean_margin_topk: int,
    directed_confusion_weight: float,
    directed_confusion_weight_matrix: torch.Tensor,
    directed_confusion_margin: float,
    graph_angular_weight: float,
    graph_angular_weight_matrix: torch.Tensor,
    graph_angular_max_cosine: float,
    graph_diffusion_aux_weight: float,
    graph_diffusion_aux_targets: torch.Tensor | None,
    graph_diffusion_gate_mask: torch.Tensor | None,
    graph_diffusion_uncertainty_margin: float,
    graph_diffusion_gate_temperature: float,
    factor_neighbor_prior_utility_weight: float,
    factor_neighbor_prior_bias: torch.Tensor | None,
    factor_neighbor_prior_neighbor_mask: torch.Tensor | None,
    factor_neighbor_prior_hard_mask: torch.Tensor | None,
    factor_neighbor_prior_utility_lambda: float,
    factor_neighbor_prior_utility_topk: int,
    factor_neighbor_prior_utility_margin: float,
    factor_neighbor_prior_utility_mode: str,
    physics_aux_weight: float,
    physics_evidence_aux_weight: float,
    physics_evidence_target_builder: PhysicsEvidenceTarget | None,
    physics_evidence_aux_scope: str,
    physics_evidence_aux_field_mode: str,
    hierarchical_targets: torch.Tensor | None,
    class_weights: torch.Tensor | None,
    teacher_probs: dict[str, np.ndarray] | None,
    distill_weight: float,
    distill_factor_weight: float,
    distill_temperature: float,
    positive_congruent_weight: float,
    positive_congruent_beta: float,
    positive_congruent_min_confidence: float,
    distill_missing_policy: str,
    online_teacher_model: nn.Module | None,
    online_teacher_weight: float,
    online_teacher_temperature: float,
    online_teacher_beta: float,
    online_teacher_min_confidence: float,
    teacher_error_replay_weight: float,
    teacher_error_replay_class_mask: torch.Tensor,
    teacher_error_replay_beta: float,
    teacher_error_replay_min_confidence: float,
    hflip_consistency_weight: float,
    hflip_consistency_temperature: float,
    hflip_physics_feature_consistency_weight: float,
    hflip_local_physics_feature_consistency_weight: float,
    hflip_low_level_feature_consistency_weight: float,
    masked_consistency_weight: float,
    masked_factor_consistency_weight: float,
    masked_consistency_temperature: float,
    masked_consistency_mode: str,
    masked_consistency_ratio: float,
    masked_consistency_block_frac: float,
    masked_consistency_max_blocks: int,
    masked_consistency_value: str,
    masked_consistency_confidence_threshold: float,
    observer_hinf_weight: float,
    observer_hinf_mode: str,
    observer_hinf_scope: str,
    observer_hinf_strength: float,
    observer_hinf_rho: float,
    observer_hinf_max_lines: int,
    observer_hinf_block_ratio: float,
    observer_hinf_temperature: float,
    observer_hinf_confidence_threshold: float,
    observer_hinf_feature_weight: float,
    observer_hinf_physics_feature_weight: float,
    observer_hinf_local_physics_feature_weight: float,
    observer_hinf_low_level_feature_weight: float,
    observer_hinf_class_consistency_weight: float,
    observer_hinf_factor_consistency_weight: float,
    observer_hinf_disturbed_ce_weight: float,
    observer_hinf_barrier_weight: float,
    observer_hinf_barrier_scope: str,
    observer_hinf_barrier_margin_drop: float,
    backbone_aux_weight: float,
) -> dict[str, float]:
    model.train()
    set_frozen_modules_eval(model)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    grad_accum_steps = max(int(grad_accum_steps), 1)
    physics_evidence_field_weight = physics_evidence_field_weights(str(physics_evidence_aux_field_mode), device)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False, ascii=True), start=1):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            need_aux = (
                factor_aux_weight > 0.0
                or hard_pair_aux_weight > 0.0
                or local_physics_factor_aux_weight > 0.0
                or physics_aux_weight > 0.0
                or backbone_aux_weight > 0.0
                or mechanism_orthogonal_aux_weight > 0.0
                or factor_neighbor_contrastive_weight > 0.0
                or factor_prototype_contrastive_weight > 0.0
                or controlled_factor_tournament_weight > 0.0
                or mechanism_controlled_factor_tournament_weight > 0.0
                or relation_prototype_contrastive_weight > 0.0
                or hflip_physics_feature_consistency_weight > 0.0
                or hflip_local_physics_feature_consistency_weight > 0.0
                or hflip_low_level_feature_consistency_weight > 0.0
                or observer_hinf_weight > 0.0
                or physics_evidence_aux_weight > 0.0
            )
            model_out = model(image, return_aux=need_aux)
            logits = model_out["logits"] if isinstance(model_out, dict) else model_out
            loss = float(class_loss_weight) * hierarchical_cross_entropy(
                logits,
                label,
                hierarchical_targets,
                class_weights,
            )
            hflip_feature_weight = (
                float(hflip_physics_feature_consistency_weight)
                + float(hflip_local_physics_feature_consistency_weight)
                + float(hflip_low_level_feature_consistency_weight)
            )
            if hflip_consistency_weight > 0.0 or hflip_feature_weight > 0.0:
                flipped_out = model(torch.flip(image, dims=[3]), return_aux=hflip_feature_weight > 0.0)
                flipped_logits = flipped_out["logits"] if isinstance(flipped_out, dict) else flipped_out
                if hflip_consistency_weight > 0.0:
                    loss = loss + float(hflip_consistency_weight) * symmetric_logit_consistency_loss(
                        logits,
                        flipped_logits,
                        float(hflip_consistency_temperature),
                    )
                if hflip_feature_weight > 0.0 and isinstance(model_out, dict) and isinstance(flipped_out, dict):
                    for key, weight in (
                        ("physics_feature", float(hflip_physics_feature_consistency_weight)),
                        ("local_physics_feature", float(hflip_local_physics_feature_consistency_weight)),
                        ("low_level_feature", float(hflip_low_level_feature_consistency_weight)),
                    ):
                        if weight <= 0.0:
                            continue
                        feature_loss = symmetric_feature_consistency_loss(model_out.get(key), flipped_out.get(key))
                        if feature_loss is not None:
                            loss = loss + weight * feature_loss
            if masked_consistency_weight > 0.0 or masked_factor_consistency_weight > 0.0:
                if str(masked_consistency_mode) == "physics_protected":
                    masked_image = physics_protected_normalized_block_mask(
                        image,
                        mask_ratio=float(masked_consistency_ratio),
                        block_frac=float(masked_consistency_block_frac),
                        max_blocks=int(masked_consistency_max_blocks),
                        mask_value=str(masked_consistency_value),
                    )
                else:
                    masked_image = random_normalized_block_mask(
                        image,
                        mask_ratio=float(masked_consistency_ratio),
                        block_frac=float(masked_consistency_block_frac),
                        max_blocks=int(masked_consistency_max_blocks),
                        mask_value=str(masked_consistency_value),
                    )
                masked_out = model(masked_image, return_aux=False)
                masked_logits = masked_out["logits"] if isinstance(masked_out, dict) else masked_out
                loss = loss + masked_view_consistency_loss(
                    masked_logits,
                    logits,
                    factor_marginal_masks,
                    temperature=float(masked_consistency_temperature),
                    class_weight=float(masked_consistency_weight),
                    factor_weight=float(masked_factor_consistency_weight),
                    confidence_threshold=float(masked_consistency_confidence_threshold),
                )
            if observer_hinf_weight > 0.0 and isinstance(model_out, dict):
                observer_sample_weight = observer_hinf_scope_weights(batch, device, str(observer_hinf_scope))
                observer_barrier_weight = observer_hinf_scope_weights(batch, device, str(observer_hinf_barrier_scope))
                disturbed_image = random_observer_disturbance(
                    image,
                    mode=str(observer_hinf_mode),
                    strength=float(observer_hinf_strength),
                    max_lines=int(observer_hinf_max_lines),
                    block_ratio=float(observer_hinf_block_ratio),
                )
                disturbed_out = model(disturbed_image, return_aux=True)
                if isinstance(disturbed_out, dict):
                    loss = loss + float(observer_hinf_weight) * observer_hinf_robustness_loss(
                        model_out,
                        disturbed_out,
                        image,
                        disturbed_image,
                        label,
                        factor_marginal_masks,
                        criterion,
                        rho=float(observer_hinf_rho),
                        feature_weight=float(observer_hinf_feature_weight),
                        physics_feature_weight=float(observer_hinf_physics_feature_weight),
                        local_physics_feature_weight=float(observer_hinf_local_physics_feature_weight),
                        low_level_feature_weight=float(observer_hinf_low_level_feature_weight),
                        class_consistency_weight=float(observer_hinf_class_consistency_weight),
                        factor_consistency_weight=float(observer_hinf_factor_consistency_weight),
                        disturbed_ce_weight=float(observer_hinf_disturbed_ce_weight),
                        barrier_weight=float(observer_hinf_barrier_weight),
                        barrier_margin_drop=float(observer_hinf_barrier_margin_drop),
                        temperature=float(observer_hinf_temperature),
                        confidence_threshold=float(observer_hinf_confidence_threshold),
                        sample_weight=observer_sample_weight,
                        barrier_sample_weight=observer_barrier_weight,
                    )
            if factor_marginal_weight > 0.0:
                loss = loss + float(factor_marginal_weight) * factor_marginal_loss(
                    logits,
                    batch,
                    factor_marginal_masks,
                )
            if relation_conditional_weight > 0.0:
                loss = loss + float(relation_conditional_weight) * relation_conditional_factor_loss(
                    logits,
                    label,
                    relation_conditional_masks,
                    relation_conditional_focus_mask,
                    axis_weights=relation_conditional_axis_weights,
                    class_axis_weights=relation_conditional_class_axis_weights,
                    uncertainty_margin=float(relation_conditional_uncertainty_margin),
                    gate_temperature=float(relation_conditional_gate_temperature),
                )
            if (
                relation_prototype_contrastive_weight > 0.0
                and isinstance(model_out, dict)
                and "feature" in model_out
            ):
                loss = loss + float(relation_prototype_contrastive_weight) * relation_conditioned_prototype_contrastive_loss(
                    model,
                    model_out["feature"],
                    logits,
                    label,
                    relation_conditional_masks,
                    relation_conditional_focus_mask,
                    axis_weights=relation_conditional_axis_weights,
                    class_axis_weights=relation_conditional_class_axis_weights,
                    temperature=float(relation_prototype_contrastive_temperature),
                    uncertainty_margin=float(relation_conditional_uncertainty_margin),
                    gate_temperature=float(relation_conditional_gate_temperature),
                )
            if roughness_ordinal_weight > 0.0:
                loss = loss + float(roughness_ordinal_weight) * roughness_ordinal_energy_loss(
                    logits,
                    batch,
                    factor_marginal_masks,
                )
            if roughness_boundary_margin_weight > 0.0:
                loss = loss + float(roughness_boundary_margin_weight) * roughness_boundary_margin_loss(
                    logits,
                    label,
                    roughness_boundary_mask,
                    margin=float(roughness_boundary_margin),
                    uncertainty_margin=float(roughness_boundary_uncertainty_margin),
                    gate_temperature=float(roughness_boundary_gate_temperature),
                )
            if concrete_masked_roughness_weight > 0.0:
                loss = loss + float(concrete_masked_roughness_weight) * concrete_masked_roughness_ordinal_loss(
                    logits,
                    image,
                    batch,
                    concrete_masked_roughness_masks,
                    obstruction_weight=float(concrete_masked_roughness_obstruction_weight),
                    uncertainty_margin=float(concrete_masked_roughness_uncertainty_margin),
                    gate_temperature=float(concrete_masked_roughness_gate_temperature),
                )
            if tensor_anova_boundary_weight > 0.0:
                loss = loss + float(tensor_anova_boundary_weight) * tensor_anova_boundary_energy_loss(
                    logits,
                    image,
                    label,
                    tensor_anova_boundary_spec,
                    focus=str(tensor_anova_boundary_focus),
                    margin=float(tensor_anova_boundary_margin),
                    uncertainty_margin=float(tensor_anova_boundary_uncertainty_margin),
                    gate_temperature=float(tensor_anova_boundary_gate_temperature),
                    friction_weight=float(tensor_anova_boundary_friction_weight),
                    material_weight=float(tensor_anova_boundary_material_weight),
                    roughness_weight=float(tensor_anova_boundary_roughness_weight),
                    obstruction_weight=float(tensor_anova_boundary_obstruction_weight),
                )
            if factor_neighbor_margin_weight > 0.0:
                if str(factor_neighbor_loss_mode) == "weighted":
                    neighbor_loss = weighted_factor_neighbor_margin_loss(
                        logits,
                        label,
                        factor_neighbor_weight_matrix,
                        margin=float(factor_neighbor_margin),
                    )
                else:
                    neighbor_loss = factor_neighbor_margin_loss(
                        logits,
                        label,
                        factor_neighbor_negative_mask,
                        margin=float(factor_neighbor_margin),
                    )
                loss = loss + float(factor_neighbor_margin_weight) * neighbor_loss
            if (
                factor_neighbor_contrastive_weight > 0.0
                and isinstance(model_out, dict)
                and "feature" in model_out
            ):
                loss = loss + float(factor_neighbor_contrastive_weight) * factor_neighbor_contrastive_loss(
                    model_out["feature"],
                    batch,
                    margin=float(factor_neighbor_contrastive_margin),
                    roughness_weight=float(factor_neighbor_roughness_weight),
                    friction_weight=float(factor_neighbor_friction_weight),
                    material_weight=float(factor_neighbor_material_weight),
                    wet_water_weight=float(factor_neighbor_wet_water_weight),
                    concrete_weight=float(factor_neighbor_concrete_weight),
                    hard_class_weight=float(factor_neighbor_hard_class_weight),
                )
            if (
                factor_prototype_contrastive_weight > 0.0
                and isinstance(model_out, dict)
                and "feature" in model_out
            ):
                loss = loss + float(factor_prototype_contrastive_weight) * factor_neighbor_prototype_contrastive_loss(
                    model,
                    model_out["feature"],
                    label,
                    factor_neighbor_weight_matrix,
                    margin=float(factor_prototype_contrastive_margin),
                )
            if (
                controlled_factor_tournament_weight > 0.0
                and isinstance(model_out, dict)
                and "feature" in model_out
            ):
                loss = loss + float(controlled_factor_tournament_weight) * controlled_factor_tournament_loss(
                    model_out["feature"],
                    batch,
                    temperature=float(controlled_factor_tournament_temperature),
                    margin=float(controlled_factor_tournament_margin),
                    neg_weight=float(controlled_factor_tournament_neg_weight),
                    focus=str(controlled_factor_tournament_focus),
                    friction_weight=float(controlled_factor_tournament_friction_weight),
                    material_weight=float(controlled_factor_tournament_material_weight),
                    unevenness_weight=float(controlled_factor_tournament_unevenness_weight),
                )
            if (
                mechanism_controlled_factor_tournament_weight > 0.0
                and isinstance(model_out, dict)
                and "feature" in model_out
            ):
                loss = loss + float(mechanism_controlled_factor_tournament_weight) * mechanism_controlled_factor_tournament_loss(
                    model_out,
                    batch,
                    temperature=float(mechanism_controlled_factor_tournament_temperature),
                    margin=float(mechanism_controlled_factor_tournament_margin),
                    neg_weight=float(mechanism_controlled_factor_tournament_neg_weight),
                    focus=str(mechanism_controlled_factor_tournament_focus),
                    friction_weight=float(mechanism_controlled_factor_tournament_friction_weight),
                    material_weight=float(mechanism_controlled_factor_tournament_material_weight),
                    unevenness_weight=float(mechanism_controlled_factor_tournament_unevenness_weight),
                )
            if local_factor_graph_weight > 0.0:
                loss = loss + float(local_factor_graph_weight) * local_factor_graph_margin_loss(
                    logits,
                    label,
                    factor_neighbor_weight_matrix,
                    margin=float(local_factor_graph_margin),
                    uncertainty_margin=float(local_factor_graph_uncertainty_margin),
                    gate_temperature=float(local_factor_graph_gate_temperature),
                    topk=int(local_factor_graph_topk),
                )
            if relation_specific_clean_margin_weight > 0.0:
                loss = loss + float(relation_specific_clean_margin_weight) * relation_specific_clean_margin_loss(
                    logits,
                    label,
                    relation_specific_clean_margin_relation_ids,
                    relation_specific_clean_margin_class_weights,
                    relation_specific_clean_margin_protected_mask,
                    friction_margin=float(relation_specific_clean_margin_friction_margin),
                    roughness_margin=float(relation_specific_clean_margin_roughness_margin),
                    material_margin=float(relation_specific_clean_margin_material_margin),
                    friction_weight=float(relation_specific_clean_margin_friction_weight),
                    roughness_weight=float(relation_specific_clean_margin_roughness_weight),
                    material_weight=float(relation_specific_clean_margin_material_weight),
                    protected_negative_scale=float(relation_specific_clean_margin_protected_negative_scale),
                    uncertainty_margin=float(relation_specific_clean_margin_uncertainty_margin),
                    gate_temperature=float(relation_specific_clean_margin_gate_temperature),
                    topk=int(relation_specific_clean_margin_topk),
                )
            if directed_confusion_weight > 0.0:
                loss = loss + float(directed_confusion_weight) * directed_confusion_margin_loss(
                    logits,
                    label,
                    directed_confusion_weight_matrix,
                    margin=float(directed_confusion_margin),
                )
            if graph_angular_weight > 0.0:
                loss = loss + float(graph_angular_weight) * graph_angular_regularization_loss(
                    model,
                    graph_angular_weight_matrix,
                    max_cosine=float(graph_angular_max_cosine),
                )
            if (
                float(graph_diffusion_aux_weight) > 0.0
                and graph_diffusion_aux_targets is not None
                and graph_diffusion_gate_mask is not None
            ):
                loss = loss + float(graph_diffusion_aux_weight) * uncertainty_gated_graph_diffusion_loss(
                    logits,
                    label,
                    graph_diffusion_aux_targets,
                    graph_diffusion_gate_mask,
                    uncertainty_margin=float(graph_diffusion_uncertainty_margin),
                    gate_temperature=float(graph_diffusion_gate_temperature),
                )
            if (
                float(factor_neighbor_prior_utility_weight) > 0.0
                and factor_neighbor_prior_bias is not None
                and factor_neighbor_prior_neighbor_mask is not None
                and factor_neighbor_prior_hard_mask is not None
            ):
                loss = loss + float(factor_neighbor_prior_utility_weight) * factor_neighbor_prior_utility_loss(
                    logits,
                    label,
                    factor_neighbor_prior_bias,
                    factor_neighbor_prior_neighbor_mask,
                    factor_neighbor_prior_hard_mask,
                    topk=int(factor_neighbor_prior_utility_topk),
                    margin=float(factor_neighbor_prior_utility_margin),
                    lam=float(factor_neighbor_prior_utility_lambda),
                    mode=str(factor_neighbor_prior_utility_mode),
                )
            if teacher_probs is not None and float(distill_weight) > 0.0:
                teacher, teacher_index = teacher_batch_probs(
                    teacher_probs,
                    batch["image_path"],
                    device=device,
                    dtype=logits.dtype,
                    missing_policy=str(distill_missing_policy),
                )
                if teacher.numel() > 0:
                    loss = loss + float(distill_weight) * distillation_kl_loss(
                        logits.index_select(0, teacher_index),
                        teacher,
                        float(distill_temperature),
                    )
            if teacher_probs is not None and float(distill_factor_weight) > 0.0:
                teacher, teacher_index = teacher_batch_probs(
                    teacher_probs,
                    batch["image_path"],
                    device=device,
                    dtype=logits.dtype,
                    missing_policy=str(distill_missing_policy),
                )
                if teacher.numel() > 0:
                    loss = loss + float(distill_factor_weight) * factor_decoupled_distillation_loss(
                        logits.index_select(0, teacher_index),
                        teacher,
                        factor_marginal_masks,
                        float(distill_temperature),
                    )
            if teacher_probs is not None and float(positive_congruent_weight) > 0.0:
                teacher, teacher_index = teacher_batch_probs(
                    teacher_probs,
                    batch["image_path"],
                    device=device,
                    dtype=logits.dtype,
                    missing_policy=str(distill_missing_policy),
                )
                if teacher.numel() > 0:
                    selected_logits = logits.index_select(0, teacher_index)
                    selected_labels = label.index_select(0, teacher_index)
                    loss = loss + float(positive_congruent_weight) * positive_congruent_distillation_loss(
                        selected_logits,
                        selected_labels,
                        teacher,
                        float(distill_temperature),
                        beta=float(positive_congruent_beta),
                        min_confidence=float(positive_congruent_min_confidence),
                    )
            if online_teacher_model is not None and (
                float(online_teacher_weight) > 0.0 or float(teacher_error_replay_weight) > 0.0
            ):
                with torch.no_grad():
                    teacher_out = online_teacher_model(image, return_aux=False)
                    teacher_logits = teacher_out["logits"] if isinstance(teacher_out, dict) else teacher_out
                    teacher_plain_prob = F.softmax(teacher_logits.detach(), dim=1)
                    teacher_prob = F.softmax(
                        teacher_logits.detach() / max(float(online_teacher_temperature), 1e-3),
                        dim=1,
                    )
                if float(online_teacher_weight) > 0.0:
                    loss = loss + float(online_teacher_weight) * positive_congruent_distillation_loss(
                        logits,
                        label,
                        teacher_prob,
                        float(online_teacher_temperature),
                        beta=float(online_teacher_beta),
                        min_confidence=float(online_teacher_min_confidence),
                    )
                if float(teacher_error_replay_weight) > 0.0:
                    loss = loss + float(teacher_error_replay_weight) * teacher_error_replay_loss(
                        logits,
                        label,
                        teacher_plain_prob,
                        teacher_error_replay_class_mask,
                        beta=float(teacher_error_replay_beta),
                        min_confidence=float(teacher_error_replay_min_confidence),
                    )
            if isinstance(model_out, dict):
                if factor_aux_weight > 0.0 and "factor_logits" in model_out:
                    factor_loss = factor_auxiliary_loss(
                        model_out["factor_logits"],
                        batch,
                        device=device,
                        focus=str(factor_aux_friction_focus),
                        axis_focus=str(factor_aux_axis_focus),
                    )
                    if factor_loss is not None:
                        loss = loss + float(factor_aux_weight) * factor_loss
                if hard_pair_aux_weight > 0.0 and "hard_pair_logits" in model_out:
                    hard_pair_loss = hard_pair_auxiliary_loss(
                        model_out["hard_pair_logits"],
                        label,
                        hard_pair_aux_left,
                        hard_pair_aux_right,
                    )
                    if hard_pair_loss is not None:
                        loss = loss + float(hard_pair_aux_weight) * hard_pair_loss
                if mechanism_orthogonal_aux_weight > 0.0 and "mechanism_orthogonal_aux" in model_out:
                    moca_loss = mechanism_orthogonal_coupling_auxiliary_loss(
                        model_out["mechanism_orthogonal_aux"],
                        batch,
                        device=device,
                        factor_weight=float(mechanism_orthogonal_factor_weight),
                        mechanism_weight=float(mechanism_orthogonal_mechanism_weight),
                        cov_weight=float(mechanism_orthogonal_cov_weight),
                    )
                    if moca_loss is not None:
                        loss = loss + float(mechanism_orthogonal_aux_weight) * moca_loss
                if local_physics_factor_aux_weight > 0.0 and "local_physics_factor_logits" in model_out:
                    local_factor_loss = factor_auxiliary_loss(
                        model_out["local_physics_factor_logits"],
                        batch,
                        device=device,
                        focus="all",
                    )
                    if local_factor_loss is not None:
                        loss = loss + float(local_physics_factor_aux_weight) * local_factor_loss
                if physics_aux_weight > 0.0 and "physics_logits" in model_out:
                    loss = loss + float(physics_aux_weight) * criterion(model_out["physics_logits"], label)
                if (
                    physics_evidence_aux_weight > 0.0
                    and physics_evidence_target_builder is not None
                    and "physics_evidence_logits" in model_out
                ):
                    with torch.no_grad():
                        physics_evidence_target = physics_evidence_target_builder(image)
                        physics_evidence_sample_weight = torch.as_tensor(
                            [
                                1.0
                                if include_class_for_mechanism_scope(str(item), str(physics_evidence_aux_scope))
                                else 0.0
                                for item in batch.get("class_label", [])
                            ],
                            device=device,
                            dtype=physics_evidence_target.dtype,
                        )
                    loss = loss + float(physics_evidence_aux_weight) * physics_evidence_loss(
                        model_out["physics_evidence_logits"],
                        physics_evidence_target,
                        field_weights=physics_evidence_field_weight,
                        sample_weight=physics_evidence_sample_weight,
                    )
            if backbone_aux_weight > 0.0 and "backbone_logits" in model_out:
                loss = loss + float(backbone_aux_weight) * criterion(model_out["backbone_logits"], label)
            backward_loss = loss / float(grad_accum_steps)
        if not bool(torch.isfinite(loss.detach())):
            optimizer.zero_grad(set_to_none=True)
            continue
        if use_amp:
            scaler.scale(backward_loss).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if bool(torch.isfinite(grad_norm)):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            backward_loss.backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if bool(torch.isfinite(grad_norm)):
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        batch_size = int(label.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == label).sum().detach().cpu())
        total_seen += batch_size
        if log_every_steps > 0 and (step % log_every_steps == 0 or step == len(loader)):
            print(
                f"  train step {step}/{len(loader)} loss={total_loss / max(total_seen, 1):.4f} "
                f"top1={total_correct / max(total_seen, 1):.4f}",
                flush=True,
            )
    return {
        "loss": total_loss / max(total_seen, 1),
        "top1": total_correct / max(total_seen, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    idx_to_class: dict[int, str],
    save_predictions_path: Path | None = None,
    save_probabilities_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    probability_rows: list[np.ndarray] = []
    probability_labels: list[np.ndarray] = []
    probability_paths: list[str] = []
    for batch in tqdm(loader, desc="eval", leave=False, ascii=True):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = model(image)
        loss = criterion(logits, label)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        total_loss += float(loss.detach().cpu()) * int(label.numel())
        total_seen += int(label.numel())
        batch_true = label.detach().cpu().numpy().astype(int).tolist()
        batch_pred = pred.detach().cpu().numpy().astype(int).tolist()
        y_true.extend(batch_true)
        y_pred.extend(batch_pred)
        if save_predictions_path is not None:
            batch_conf = conf.detach().cpu().numpy().astype(float).tolist()
            for path, true_idx, pred_idx, confidence in zip(
                batch["image_path"],
                batch_true,
                batch_pred,
                batch_conf,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "image_path": str(path),
                        "true_label": idx_to_class[int(true_idx)],
                        "pred_label": idx_to_class[int(pred_idx)],
                        "confidence": float(confidence),
                    }
                )
        if save_probabilities_path is not None:
            probability_rows.append(probs.detach().float().cpu().numpy().astype(np.float32))
            probability_labels.append(label.detach().cpu().numpy().astype(np.int64))
            probability_paths.extend([str(path) for path in batch["image_path"]])
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[idx] for idx in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "loss": total_loss / max(total_seen, 1),
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "num_samples": int(total_seen),
        "num_classes": int(len(labels)),
    }
    confusion = confusion_rows(y_true, y_pred, idx_to_class)
    if save_predictions_path is not None:
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(save_predictions_path, index=False, encoding="utf-8")
    if save_probabilities_path is not None:
        save_probabilities_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_probabilities_path,
            image_path=np.asarray(probability_paths, dtype=object),
            label=np.concatenate(probability_labels, axis=0).astype(np.int64),
            probs=np.concatenate(probability_rows, axis=0).astype(np.float32),
        )
    return {
        "summary": summary,
        "classification_report": report,
        "confusion": confusion,
    }


def confusion_rows(y_true: list[int], y_pred: list[int], idx_to_class: dict[int, str]) -> list[dict[str, Any]]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    rows = []
    for idx, name in idx_to_class.items():
        mask = y_true_arr == idx
        if not mask.any():
            continue
        pred_counts = pd.Series(y_pred_arr[mask]).value_counts().sort_values(ascending=False)
        top = [
            {"pred": idx_to_class.get(int(pred_idx), str(pred_idx)), "count": int(count)}
            for pred_idx, count in pred_counts.head(5).items()
        ]
        rows.append({"class_label": name, "support": int(mask.sum()), "top_predictions": top})
    return rows


def write_protocol_manifest(args: argparse.Namespace, class_to_idx: dict[str, int]) -> None:
    payload = {
        "role": "RSCD original class-label classification protocol",
        "claim_boundary": (
            "This protocol is separate from the weak visual friction-affordance interval protocol. "
            "It can be used to discuss RSCD-style road-surface classification papers only when "
            "splits, labels, preprocessing, and metrics are explicitly matched."
        ),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "class_to_idx": class_to_idx,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_eval_outputs(output_dir: Path, metrics: dict[str, Any], *, split: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"evaluate_{split}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = metrics["summary"]
    lines = [
        f"# RSCD Surface Classification {split.title()} Result",
        "",
        "This is the original RSCD class-label protocol, not the friction/risk/interval protocol.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| top-1 accuracy | {summary['top1'] * 100:.2f} |",
        f"| mean precision | {summary.get('mean_precision', 0.0) * 100:.2f} |",
        f"| mean recall | {summary.get('mean_recall', summary.get('balanced_accuracy', 0.0)) * 100:.2f} |",
        f"| macro F1 | {summary['macro_f1'] * 100:.2f} |",
        f"| weighted F1 | {summary['weighted_f1'] * 100:.2f} |",
        f"| balanced accuracy | {summary['balanced_accuracy'] * 100:.2f} |",
        f"| samples | {summary['num_samples']} |",
        f"| classes | {summary['num_classes']} |",
        "",
    ]
    (output_dir / f"evaluate_{split}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

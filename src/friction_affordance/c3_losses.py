from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from friction_affordance.rscd_factors import (
    FACTOR_AXES,
    FACTOR_LABELS,
    RSCDFactorSpec,
    class_factor_targets,
)


def factor_ce_loss(
    factor_logits: dict[str, torch.Tensor],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    *,
    weights: dict[str, float],
    supervise_none: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = class_factor_targets(labels, spec, labels.device)
    terms: list[torch.Tensor] = []
    logs: dict[str, float] = {}
    for axis in FACTOR_AXES:
        logits = factor_logits.get(axis)
        if logits is None:
            continue
        target = targets[axis]
        valid = target.ge(0)
        if not supervise_none and axis in {"material", "roughness"}:
            valid = valid & target.ne(0)
        if not bool(valid.any()):
            logs[f"loss_{axis}"] = 0.0
            logs[f"acc_{axis}"] = 0.0
            continue
        idx = valid.nonzero(as_tuple=False).flatten()
        loss = F.cross_entropy(logits.index_select(0, idx), target.index_select(0, idx))
        terms.append(float(weights.get(axis, 1.0)) * loss)
        pred = logits.argmax(dim=1)
        logs[f"loss_{axis}"] = float(loss.detach().cpu())
        logs[f"acc_{axis}"] = float((pred[valid] == target[valid]).float().mean().detach().cpu())
    if not terms:
        return labels.new_zeros((), dtype=torch.float32), logs
    return torch.stack(terms).sum(), logs


def mechanism_routed_tournament_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    boundary_logits: dict[str, torch.Tensor],
    spec: RSCDFactorSpec,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    total = 0
    correct = 0
    by_axis_total = {axis: 0 for axis in FACTOR_AXES}
    by_axis_correct = {axis: 0 for axis in FACTOR_AXES}
    for pair in spec.hard_pairs:
        mask_left = labels.eq(int(pair.left))
        mask_right = labels.eq(int(pair.right))
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        sign = torch.where(mask_left[mask], logits.new_ones(mask.sum()), -logits.new_ones(mask.sum()))
        base = logits[mask, int(pair.left)] - logits[mask, int(pair.right)]
        expert = boundary_logits.get(pair.boundary)
        if expert is not None:
            base = base + expert[mask]
        losses.append(F.softplus(-sign * base).mean())
        pair_correct = (sign * base > 0).detach()
        total += int(pair_correct.numel())
        correct += int(pair_correct.sum().cpu())
        by_axis_total[pair.axis] += int(pair_correct.numel())
        by_axis_correct[pair.axis] += int(pair_correct.sum().cpu())
    logs = {
        "hard_pair_acc": float(correct / max(total, 1)),
        "hard_pair_count": float(total),
    }
    for axis in FACTOR_AXES:
        logs[f"{axis}_pair_acc"] = float(by_axis_correct[axis] / max(by_axis_total[axis], 1))
        logs[f"{axis}_pair_count"] = float(by_axis_total[axis])
    if not losses:
        return logits.new_zeros(()), logs
    return torch.stack(losses).mean(), logs


def counterfactual_factor_contrast_loss(
    tokens: dict[str, torch.Tensor],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    *,
    rho: torch.Tensor | None,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    bsz = int(labels.numel())
    if bsz < 2:
        return labels.new_zeros((), dtype=torch.float32), {
            "num_cf_friction_pairs": 0.0,
            "num_cf_material_pairs": 0.0,
            "num_cf_roughness_pairs": 0.0,
        }
    z = {
        "friction": F.normalize(tokens["friction"].float(), dim=1),
        "material": F.normalize(tokens["material"].float(), dim=1),
        "roughness": F.normalize(tokens["roughness"].float(), dim=1),
    }
    losses: list[torch.Tensor] = []
    counts = {axis: 0 for axis in FACTOR_AXES}
    for i in range(bsz):
        for j in range(i + 1, bsz):
            a = factors[i]
            b = factors[j]
            if bool((a < 0).any().item()) or bool((b < 0).any().item()):
                continue
            diff = [axis for axis_idx, axis in enumerate(FACTOR_AXES) if int(a[axis_idx]) != int(b[axis_idx])]
            if len(diff) != 1:
                continue
            axis = diff[0]
            same_axes = [name for name in FACTOR_AXES if name != axis]
            same_loss = sum((z[name][i] - z[name][j]).square().mean() for name in same_axes)
            dist = torch.norm(z[axis][i] - z[axis][j], p=2)
            sep_loss = F.relu(float(margin) - dist).square()
            if axis == "roughness" and rho is not None:
                sep_loss = sep_loss * rho[i].float().squeeze() * rho[j].float().squeeze()
            losses.append(same_loss + sep_loss)
            counts[axis] += 1
    logs = {
        "num_cf_friction_pairs": float(counts["friction"]),
        "num_cf_material_pairs": float(counts["material"]),
        "num_cf_roughness_pairs": float(counts["roughness"]),
    }
    if not losses:
        return labels.new_zeros((), dtype=torch.float32), logs
    return torch.stack(losses).mean().to(dtype=tokens["friction"].dtype), logs


def c3_reliability_loss(model_out: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    rho = model_out.get("rho_roughness")
    target = model_out.get("rho_target")
    if not isinstance(rho, torch.Tensor) or not isinstance(target, torch.Tensor):
        fallback = model_out["logits"].new_zeros(())
        return fallback, {"rho_R_mean": 0.0, "rho_target_mean": 0.0}
    with torch.amp.autocast(device_type=rho.device.type, enabled=False):
        loss = F.binary_cross_entropy(
            rho.float().clamp(1e-4, 1.0 - 1e-4),
            target.float().clamp(0.0, 1.0),
        )
    return loss.to(dtype=rho.dtype), {
        "rho_R_mean": float(rho.detach().mean().cpu()),
        "rho_target_mean": float(target.detach().mean().cpu()),
    }


def _supervised_contrastive_from_mask(
    z: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """SupCon term for a precomputed positive relation mask."""

    if z.ndim != 2 or z.size(0) < 2:
        return z.new_zeros(()), 0
    n = int(z.size(0))
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    positive_mask = positive_mask.to(device=z.device, dtype=torch.bool) & ~eye
    valid_anchor = positive_mask.any(dim=1)
    if not bool(valid_anchor.any()):
        return z.new_zeros(()), 0
    logits = (z @ z.t()) / max(float(temperature), 1e-4)
    logits = logits.masked_fill(eye, -1.0e4)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob * positive_mask.float()).sum(dim=1) / pos_count.float()
    return per_anchor[valid_anchor].mean(), int(valid_anchor.sum().detach().cpu())


def factor_graph_metric_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """RSCD factor-graph metric loss for representation shaping.

    This is task-adapted to the RSCD label algebra. It does not add another
    classifier head. Instead, it shapes the already exposed representation:

    - axis tokens learn their own factor through supervised contrastive masks;
    - the coupling token learns the exact composite class;
    - the fused feature is repelled from one-factor graph neighbors only when
      they become too close, with roughness repulsion softened by rho_R so
      hidden water-film roughness is not over-penalized.
    """

    weight = float(loss_cfg.get("factor_graph_metric_weight", 0.0))
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {"loss_factor_graph_metric": 0.0}
    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    valid_all = factors.ge(0).all(dim=1)
    if int(valid_all.sum().detach().cpu()) < 2:
        return model_out["logits"].new_zeros(()), {
            "loss_factor_graph_metric": 0.0,
            "factor_graph_metric_valid": float(valid_all.sum().detach().cpu()),
        }

    temperature = float(loss_cfg.get("factor_graph_metric_temperature", 0.12))
    axis_weight = float(loss_cfg.get("factor_graph_metric_axis_weight", 0.35))
    coupling_weight = float(loss_cfg.get("factor_graph_metric_coupling_weight", 0.20))
    neighbor_weight = float(loss_cfg.get("factor_graph_metric_neighbor_weight", 0.35))
    neighbor_margin = float(loss_cfg.get("factor_graph_metric_neighbor_margin", 0.62))
    roughness_neighbor_scale = float(loss_cfg.get("factor_graph_metric_roughness_neighbor_scale", 0.55))
    wet_concrete_focus_scale = float(loss_cfg.get("factor_graph_metric_wet_concrete_focus_scale", 1.25))
    supervise_none = bool(loss_cfg.get("factor_graph_metric_supervise_none", False))

    logits = model_out["logits"]
    total = logits.new_zeros(())
    logs: dict[str, float] = {"factor_graph_metric_valid": float(valid_all.sum().detach().cpu())}
    tokens = model_out.get("tokens", {})
    axis_terms: list[torch.Tensor] = []
    axis_anchor_count = 0
    for axis_idx, axis in enumerate(FACTOR_AXES):
        token = tokens.get(axis)
        if token is None and axis == "roughness":
            token = tokens.get("roughness_visible")
        if not isinstance(token, torch.Tensor):
            continue
        axis_target = factors[:, axis_idx]
        valid = valid_all & axis_target.ge(0)
        if not supervise_none and axis in {"material", "roughness"}:
            valid = valid & axis_target.ne(0)
        if int(valid.sum().detach().cpu()) < 2:
            continue
        z_axis = F.normalize(token.float(), dim=1)
        same_axis = axis_target[:, None].eq(axis_target[None, :]) & valid[:, None] & valid[None, :]
        loss_axis, count_axis = _supervised_contrastive_from_mask(
            z_axis,
            same_axis,
            temperature=temperature,
        )
        if count_axis > 0:
            axis_terms.append(loss_axis)
            axis_anchor_count += count_axis
            logs[f"factor_graph_metric_{axis}_anchors"] = float(count_axis)
    if axis_terms:
        axis_loss = torch.stack(axis_terms).mean().to(dtype=logits.dtype)
        total = total + axis_weight * axis_loss
        logs["loss_factor_graph_metric_axis"] = float(axis_loss.detach().cpu())
        logs["factor_graph_metric_axis_anchors"] = float(axis_anchor_count)
    else:
        logs["loss_factor_graph_metric_axis"] = 0.0
        logs["factor_graph_metric_axis_anchors"] = 0.0

    coupling = tokens.get("coupling")
    if coupling_weight > 0.0 and isinstance(coupling, torch.Tensor):
        z_c = F.normalize(coupling.float(), dim=1)
        same_class = labels[:, None].eq(labels[None, :]) & valid_all[:, None] & valid_all[None, :]
        coupling_loss, coupling_count = _supervised_contrastive_from_mask(
            z_c,
            same_class,
            temperature=temperature,
        )
        if coupling_count > 0:
            total = total + coupling_weight * coupling_loss.to(dtype=logits.dtype)
        logs["loss_factor_graph_metric_coupling"] = float(coupling_loss.detach().cpu())
        logs["factor_graph_metric_coupling_anchors"] = float(coupling_count)
    else:
        logs["loss_factor_graph_metric_coupling"] = 0.0
        logs["factor_graph_metric_coupling_anchors"] = 0.0

    feature = model_out.get("feature")
    if neighbor_weight > 0.0 and isinstance(feature, torch.Tensor):
        z = F.normalize(feature.float(), dim=1)
        sim = z @ z.t()
        diff = factors[:, None, :] != factors[None, :, :]
        one_axis_diff = diff.sum(dim=2).eq(1) & valid_all[:, None] & valid_all[None, :]
        eye = torch.eye(int(labels.numel()), device=labels.device, dtype=torch.bool)
        one_axis_diff = one_axis_diff & ~eye
        if bool(one_axis_diff.any()):
            changed_axis = diff.float().argmax(dim=2)
            pair_weight = torch.ones_like(sim)
            rough_mask = one_axis_diff & changed_axis.eq(FACTOR_AXES.index("roughness"))
            if bool(rough_mask.any()):
                rho = model_out.get("rho_roughness")
                if isinstance(rho, torch.Tensor):
                    rho_detached = rho.float().detach()
                    rho_pair = rho_detached.view(-1, 1) * rho_detached.view(1, -1)
                    pair_weight = torch.where(
                        rough_mask,
                        roughness_neighbor_scale + (1.0 - roughness_neighbor_scale) * rho_pair,
                        pair_weight,
                    )
                else:
                    pair_weight = torch.where(
                        rough_mask,
                        pair_weight.new_full(pair_weight.shape, roughness_neighbor_scale),
                        pair_weight,
                    )
            wet_idx = FACTOR_LABELS["friction"].index("wet")
            water_idx = FACTOR_LABELS["friction"].index("water")
            concrete_idx = FACTOR_LABELS["material"].index("concrete")
            friction = factors[:, 0]
            material = factors[:, 1]
            wc_sample = (friction.eq(wet_idx) | friction.eq(water_idx)) & material.eq(concrete_idx)
            wc_pair = one_axis_diff & wc_sample[:, None] & wc_sample[None, :]
            if bool(wc_pair.any()):
                pair_weight = torch.where(wc_pair, pair_weight * wet_concrete_focus_scale, pair_weight)
            neighbor_loss_values = F.relu(sim - float(neighbor_margin)).square() * pair_weight
            neighbor_loss = neighbor_loss_values[one_axis_diff].mean().to(dtype=logits.dtype)
            total = total + neighbor_weight * neighbor_loss
            logs["loss_factor_graph_metric_neighbor"] = float(neighbor_loss.detach().cpu())
            logs["factor_graph_metric_neighbor_pairs"] = float(one_axis_diff.sum().detach().cpu())
            logs["factor_graph_metric_neighbor_sim_mean"] = float(sim[one_axis_diff].detach().mean().cpu())
        else:
            logs["loss_factor_graph_metric_neighbor"] = 0.0
            logs["factor_graph_metric_neighbor_pairs"] = 0.0
            logs["factor_graph_metric_neighbor_sim_mean"] = 0.0
    else:
        logs["loss_factor_graph_metric_neighbor"] = 0.0
        logs["factor_graph_metric_neighbor_pairs"] = 0.0
        logs["factor_graph_metric_neighbor_sim_mean"] = 0.0

    total = weight * total
    logs["loss_factor_graph_metric"] = float(total.detach().cpu())
    return total, logs


def c3_total_aux_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    *,
    factor_weight: float = 0.3,
    factor_axis_weights: dict[str, float] | None = None,
    tournament_weight: float = 0.1,
    counterfactual_weight: float = 0.05,
    reliability_weight: float = 0.05,
    counterfactual_margin: float = 1.0,
    supervise_none: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model_out["logits"]
    total = logits.new_zeros(())
    logs: dict[str, float] = {}
    if factor_weight > 0.0:
        loss_factor, factor_logs = factor_ce_loss(
            model_out.get("factor_logits", {}),
            labels,
            spec,
            weights=factor_axis_weights or {"friction": 1.0, "material": 1.0, "roughness": 1.0},
            supervise_none=bool(supervise_none),
        )
        total = total + float(factor_weight) * loss_factor
        logs.update(factor_logs)
        logs["loss_factor_total"] = float(loss_factor.detach().cpu())
    if tournament_weight > 0.0:
        loss_tour, tour_logs = mechanism_routed_tournament_loss(
            logits,
            labels,
            model_out.get("boundary_logits", {}),
            spec,
        )
        total = total + float(tournament_weight) * loss_tour
        logs.update(tour_logs)
        logs["loss_tournament"] = float(loss_tour.detach().cpu())
    if counterfactual_weight > 0.0 and "tokens" in model_out:
        loss_cf, cf_logs = counterfactual_factor_contrast_loss(
            model_out["tokens"],
            labels,
            spec,
            rho=model_out.get("rho_roughness"),
            margin=float(counterfactual_margin),
        )
        total = total + float(counterfactual_weight) * loss_cf
        logs.update(cf_logs)
        logs["loss_counterfactual"] = float(loss_cf.detach().cpu())
    if reliability_weight > 0.0:
        loss_rel, rel_logs = c3_reliability_loss(model_out)
        total = total + float(reliability_weight) * loss_rel
        logs.update(rel_logs)
        logs["loss_reliability"] = float(loss_rel.detach().cpu())
    logs["loss_c3_aux"] = float(total.detach().cpu())
    return total, logs

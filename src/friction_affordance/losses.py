from __future__ import annotations

import torch
import torch.nn.functional as F

from friction_affordance.ontology import TASKS
from friction_affordance.models.friction_set import CORE_STATE_TASKS


def masked_multitask_ce(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    logs = {}
    for task in TASKS:
        mask = masks[task]
        if mask.any():
            loss = F.cross_entropy(logits[task][mask], labels[task][mask])
            losses.append(loss)
            logs[f"loss_{task}"] = float(loss.detach().cpu())
    if not losses:
        device = next(iter(logits.values())).device
        return torch.zeros((), device=device), logs
    return torch.stack(losses).mean(), logs


def ordinal_cdf_emd_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return torch.zeros((), device=logits.device)
    logits = logits[mask]
    labels = labels[mask]
    probs = logits.softmax(dim=1)
    pred_cdf = probs.cumsum(dim=1)[:, :-1]
    thresholds = torch.arange(logits.size(1) - 1, device=logits.device).view(1, -1)
    target_cdf = (labels.view(-1, 1) <= thresholds).float()
    return (pred_cdf - target_cdf).square().mean()


def ordinal_risk_emd_loss(
    risk_logits: torch.Tensor,
    risk_labels: torch.Tensor,
    risk_mask: torch.Tensor,
) -> torch.Tensor:
    return ordinal_cdf_emd_loss(risk_logits, risk_labels, risk_mask)


def friction_set_compatibility_nll(
    outputs: dict,
    labels: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    friction_set = outputs.get("friction_set")
    if not friction_set:
        device = next(iter(outputs["logits"].values())).device
        return torch.zeros(0, device=device)
    state_log_prob = friction_set["state_log_prob"]
    allowed = torch.ones_like(state_log_prob, dtype=torch.bool)
    for task in CORE_STATE_TASKS:
        if task not in labels or task not in masks:
            continue
        state_idx = friction_set[f"state_{task}_idx"].view(1, -1)
        known = masks[task].view(-1, 1)
        target = labels[task].view(-1, 1)
        allowed &= (~known) | (state_idx == target)
    neg_inf = torch.finfo(state_log_prob.dtype).min
    compatible_log_prob = torch.logsumexp(state_log_prob.masked_fill(~allowed, neg_inf), dim=1)
    return -compatible_log_prob.clamp_min(-30.0)


def group_dro_loss(sample_loss: torch.Tensor, group_idx: torch.Tensor | None, temperature: float = 0.0) -> torch.Tensor:
    if sample_loss.numel() == 0 or group_idx is None:
        device = sample_loss.device if sample_loss.numel() else torch.device("cpu")
        return torch.zeros((), device=device)
    means = []
    for group in torch.unique(group_idx.detach()):
        keep = group_idx == group
        if keep.any():
            means.append(sample_loss[keep].mean())
    if not means:
        return torch.zeros((), device=sample_loss.device)
    group_means = torch.stack(means)
    if temperature > 0:
        return temperature * torch.logsumexp(group_means / temperature, dim=0)
    return group_means.max()


def group_vrex_loss(sample_loss: torch.Tensor, group_idx: torch.Tensor | None) -> torch.Tensor:
    if sample_loss.numel() == 0 or group_idx is None:
        device = sample_loss.device if sample_loss.numel() else torch.device("cpu")
        return torch.zeros((), device=device)
    means = []
    for group in torch.unique(group_idx.detach()):
        keep = group_idx == group
        if keep.any():
            means.append(sample_loss[keep].mean())
    if len(means) < 2:
        return torch.zeros((), device=sample_loss.device)
    group_means = torch.stack(means)
    return group_means.var(unbiased=False)


def feature_coral_loss(
    features: torch.Tensor,
    domain_idx: torch.Tensor | None,
    condition_idx: torch.Tensor | None = None,
    condition_mask: torch.Tensor | None = None,
    min_samples_per_domain: int = 2,
) -> torch.Tensor:
    """Align feature first/second moments across domains.

    This is intentionally small and batch-local. The conditional variant aligns
    domains only inside the same known risk state, which reduces dataset style
    shortcuts without forcing genuinely different friction states to collapse.
    """
    if domain_idx is None or features.size(0) < 4:
        return torch.zeros((), device=features.device)
    if condition_idx is None:
        return _coral_over_domains(features, domain_idx, min_samples_per_domain)

    if condition_mask is None:
        condition_mask = torch.ones_like(condition_idx, dtype=torch.bool)
    losses = []
    for condition in torch.unique(condition_idx[condition_mask].detach()):
        keep = condition_mask & (condition_idx == condition)
        if keep.sum() < 4:
            continue
        loss = _coral_over_domains(features[keep], domain_idx[keep], min_samples_per_domain)
        if bool(torch.isfinite(loss).detach().cpu()) and float(loss.detach().abs().cpu()) > 0:
            losses.append(loss)
    if not losses:
        return torch.zeros((), device=features.device)
    return torch.stack(losses).mean()


def _coral_over_domains(
    features: torch.Tensor,
    domain_idx: torch.Tensor,
    min_samples_per_domain: int,
) -> torch.Tensor:
    domains = []
    for domain in torch.unique(domain_idx.detach()):
        keep = domain_idx == domain
        if int(keep.sum().detach().cpu()) >= int(min_samples_per_domain):
            domains.append(features[keep])
    if len(domains) < 2:
        return torch.zeros((), device=features.device)

    pooled_mean = features.mean(dim=0)
    pooled_cov = _covariance(features)
    dim = float(features.size(1))
    losses = []
    for domain_features in domains:
        mean_loss = (domain_features.mean(dim=0) - pooled_mean).square().mean()
        cov_loss = (_covariance(domain_features) - pooled_cov).square().sum() / (4.0 * dim * dim)
        losses.append(mean_loss + cov_loss)
    return torch.stack(losses).mean()


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    denom = max(int(features.size(0)) - 1, 1)
    return centered.t().matmul(centered) / float(denom)


def state_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor | None,
    mask: torch.Tensor | None,
    domain_idx: torch.Tensor | None,
    *,
    temperature: float = 0.2,
    cross_domain_only: bool = True,
) -> torch.Tensor:
    """Supervised cross-domain contrastive alignment for weak road states.

    Unlike CORAL, this does not collapse all domains together. It pulls features
    together only when two samples share a known road state, and the default
    cross-domain mode requires the positive pair to come from different datasets.
    """
    if labels is None or mask is None or features.ndim != 2 or features.size(0) < 2:
        return torch.zeros((), device=features.device)
    if cross_domain_only and domain_idx is None:
        return torch.zeros((), device=features.device)

    valid = mask.to(device=features.device, dtype=torch.bool).view(-1)
    if valid.sum() < 2:
        return torch.zeros((), device=features.device)

    z = F.normalize(features[valid].float(), dim=1)
    y = labels.to(device=features.device).view(-1)[valid]
    if domain_idx is not None:
        d = domain_idx.to(device=features.device).view(-1)[valid]
    else:
        d = None

    n = int(z.size(0))
    if n < 2:
        return torch.zeros((), device=features.device)
    same_state = y.view(-1, 1).eq(y.view(1, -1))
    eye = torch.eye(n, device=features.device, dtype=torch.bool)
    positives = same_state & ~eye
    if cross_domain_only:
        if d is None or torch.unique(d.detach()).numel() < 2:
            return torch.zeros((), device=features.device)
        positives = positives & ~d.view(-1, 1).eq(d.view(1, -1))

    anchors = positives.any(dim=1)
    if not anchors.any():
        return torch.zeros((), device=features.device)

    temp = max(float(temperature), 1e-6)
    logits = z.matmul(z.t()) / temp
    logits = logits.masked_fill(eye, -torch.inf)
    row_max = logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits - row_max).masked_fill(eye, 0.0)
    denom = exp_logits.sum(dim=1).clamp_min(1e-8)
    positive_mass = (exp_logits * positives.to(dtype=exp_logits.dtype)).sum(dim=1).clamp_min(1e-8)
    loss = -torch.log(positive_mass / denom)
    return loss[anchors].mean()


def interval_censored_logistic_loss(
    mu_mean: torch.Tensor,
    mu_scale: torch.Tensor,
    target_interval: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return torch.zeros((), device=mu_mean.device)
    low = target_interval[mask, 0]
    high = target_interval[mask, 1]
    mean = mu_mean[mask]
    scale = mu_scale[mask].clamp_min(1e-4)
    cdf_high = torch.sigmoid((high - mean) / scale)
    cdf_low = torch.sigmoid((low - mean) / scale)
    prob = (cdf_high - cdf_low).clamp_min(1e-6)
    return -torch.log(prob).mean()


def interval_coverage_loss(
    pred_interval: torch.Tensor,
    target_interval: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if not mask.any():
        return torch.zeros((), device=pred_interval.device)
    pred = pred_interval[mask]
    target = target_interval[mask]
    low_violation = F.relu(pred[:, 0] - target[:, 0] + margin)
    high_violation = F.relu(target[:, 1] - pred[:, 1] + margin)
    violation = low_violation + high_violation
    if sample_weight is not None:
        weight = sample_weight[mask].to(device=pred_interval.device, dtype=pred_interval.dtype).clamp_min(0.0)
        if weight.numel() and float(weight.sum().detach().cpu()) > 0.0:
            return (violation * weight).sum() / weight.sum().clamp_min(1e-8)
    return violation.mean()


def build_safety_coverage_weights(outputs: dict, batch: dict, cfg_loss: dict) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Upweight interval coverage errors on safety-critical road states.

    Public image datasets provide weak friction intervals rather than measured
    tire-road friction. This weighting keeps the same interval targets but tells
    the model that failing to cover high-risk/wet/snow examples is more costly.
    """
    risk_weight = float(cfg_loss.get("coverage_risk_weight", 0.0))
    wetness_weight = float(cfg_loss.get("coverage_wetness_weight", 0.0))
    snow_weight = float(cfg_loss.get("coverage_snow_weight", 0.0))
    near_white_weight = float(cfg_loss.get("coverage_near_white_weight", 0.0))
    low_texture_weight = float(cfg_loss.get("coverage_low_texture_weight", 0.0))
    specular_weight = float(cfg_loss.get("coverage_specular_weight", 0.0))
    if (
        risk_weight <= 0
        and wetness_weight <= 0
        and snow_weight <= 0
        and near_white_weight <= 0
        and low_texture_weight <= 0
        and specular_weight <= 0
    ):
        return None, {
            "coverage_weight_mean": 1.0,
            "coverage_weight_max": 1.0,
            "coverage_near_white_mean": 0.0,
            "coverage_low_texture_mean": 0.0,
            "coverage_specular_mean": 0.0,
            "coverage_visual_quality_weight_mean": 0.0,
        }

    device = outputs["mu_mean"].device
    weights = torch.ones_like(batch["mu_mask"], device=device, dtype=outputs["mu_mean"].dtype)

    def add_normalized(task: str, strength: float) -> None:
        if strength <= 0 or task not in batch["labels"] or task not in batch["masks"]:
            return
        mask = batch["masks"][task].to(device=device)
        if not mask.any():
            return
        labels = batch["labels"][task].to(device=device, dtype=weights.dtype)
        logits = outputs.get("logits", {}).get(task)
        denom = float(max((logits.size(1) - 1) if logits is not None and logits.ndim == 2 else int(labels.max().detach().cpu()), 1))
        weights[mask] = weights[mask] + float(strength) * (labels[mask] / denom).clamp(0.0, 1.0)

    add_normalized("risk", risk_weight)
    add_normalized("wetness", wetness_weight)
    add_normalized("snow", snow_weight)
    quality_weight, quality_logs = visual_quality_coverage_weight(
        batch,
        device=device,
        dtype=weights.dtype,
        near_white_weight=near_white_weight,
        low_texture_weight=low_texture_weight,
        specular_weight=specular_weight,
    )
    if quality_weight is not None:
        weights = weights + quality_weight.clamp_min(0.0)
    weights = weights.clamp_max(float(cfg_loss.get("coverage_weight_max", 2.0)))
    mu_mask = batch["mu_mask"].to(device=device)
    if mu_mask.any():
        mean_weight = float(weights[mu_mask].mean().detach().cpu())
        max_weight = float(weights[mu_mask].max().detach().cpu())
    else:
        mean_weight = float(weights.mean().detach().cpu())
        max_weight = float(weights.max().detach().cpu())
    return weights, {
        "coverage_weight_mean": mean_weight,
        "coverage_weight_max": max_weight,
        **quality_logs,
    }


def visual_quality_coverage_weight(
    batch: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
    near_white_weight: float = 0.0,
    low_texture_weight: float = 0.0,
    specular_weight: float = 0.0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Image-derived coverage weighting for weak visual friction intervals.

    Bright low-saturation patches, mirror-like wet highlights, and very low
    texture are exactly the conditions where public road-state proxy labels are
    visually ambiguous. The score is used only as a conservative interval
    coverage weight; it is not a friction label.
    """
    logs = {
        "coverage_near_white_mean": 0.0,
        "coverage_low_texture_mean": 0.0,
        "coverage_specular_mean": 0.0,
        "coverage_visual_quality_weight_mean": 0.0,
    }
    if near_white_weight <= 0 and low_texture_weight <= 0 and specular_weight <= 0:
        return None, logs
    image = batch.get("image")
    if image is None or image.ndim != 4 or image.size(1) < 3:
        return None, logs

    x = image.to(device=device, dtype=dtype)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = (x * std + mean).clamp(0.0, 1.0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    maxc = rgb.max(dim=1, keepdim=True).values
    minc = rgb.min(dim=1, keepdim=True).values
    value = maxc
    saturation = (maxc - minc) / maxc.clamp_min(1e-4)
    dx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    dy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    edge = 0.5 * (dx + dy)

    near_white = (
        torch.sigmoid((value - 0.92) * 18.0)
        * torch.sigmoid((0.22 - saturation) * 14.0)
    ).mean(dim=(1, 2, 3))
    specular = (
        torch.sigmoid((value - 0.82) * 14.0)
        * torch.sigmoid((0.24 - saturation) * 12.0)
    ).mean(dim=(1, 2, 3))
    low_texture = torch.sigmoid((0.045 - edge) * 35.0)
    quality_weight = (
        float(near_white_weight) * near_white
        + float(low_texture_weight) * low_texture
        + float(specular_weight) * specular
    )

    logs.update(
        {
            "coverage_near_white_mean": float(near_white.mean().detach().cpu()),
            "coverage_low_texture_mean": float(low_texture.mean().detach().cpu()),
            "coverage_specular_mean": float(specular.mean().detach().cpu()),
            "coverage_visual_quality_weight_mean": float(quality_weight.mean().detach().cpu()),
        }
    )
    return quality_weight, logs


def interval_width_loss(pred_interval: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return torch.zeros((), device=pred_interval.device)
    pred = pred_interval[mask]
    return (pred[:, 1] - pred[:, 0]).clamp_min(0.0).mean()


def interval_endpoint_loss(
    pred_interval: torch.Tensor,
    mu_mean: torch.Tensor,
    target_interval: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return torch.zeros((), device=pred_interval.device)
    pred = pred_interval[mask]
    target = target_interval[mask]
    pred_mid = mu_mean[mask]
    target_mid = target.mean(dim=1)
    endpoint = F.smooth_l1_loss(pred, target)
    center = F.smooth_l1_loss(pred_mid, target_mid)
    return endpoint + 0.5 * center


def interval_target_width_loss(
    pred_interval: torch.Tensor,
    target_interval: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return torch.zeros((), device=pred_interval.device)
    pred = pred_interval[mask]
    target = target_interval[mask]
    pred_width = (pred[:, 1] - pred[:, 0]).clamp_min(0.0)
    target_width = (target[:, 1] - target[:, 0]).clamp_min(0.0)
    return F.smooth_l1_loss(pred_width, target_width)


def interval_order_consistency_loss(
    mu_mean: torch.Tensor,
    target_interval: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin_scale: float = 0.35,
    min_gap: float = 0.02,
) -> torch.Tensor:
    """Pairwise order loss for weak friction intervals.

    Public road-condition datasets provide interval anchors rather than measured
    friction. This loss uses only non-overlapping target intervals: if sample i
    is definitely lower-friction than sample j, the predicted means should keep
    the same order with a margin proportional to the interval gap.
    """
    if not mask.any():
        return torch.zeros((), device=mu_mean.device)
    mu = mu_mean[mask].float()
    target = target_interval[mask].float()
    if mu.numel() < 2:
        return torch.zeros((), device=mu_mean.device)

    low = target[:, 0]
    high = target[:, 1]
    gap = low.view(1, -1) - high.view(-1, 1)
    ordered = gap >= float(min_gap)
    if not ordered.any():
        return torch.zeros((), device=mu_mean.device)

    margin = (float(margin_scale) * gap).clamp_min(float(min_gap))
    violation = F.relu(mu.view(-1, 1) - mu.view(1, -1) + margin)
    return violation[ordered].mean()


def soft_monotonic_wetness_loss(
    logits: dict[str, torch.Tensor],
    mu_mean: torch.Tensor,
    masks: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Local soft constraint: higher wetness should not imply higher mu on average.

    This is intentionally weak. It is only applied within a mini-batch where both
    wetness labels are available.
    """
    wet_mask = masks["wetness"]
    if wet_mask.sum() < 2:
        return torch.zeros((), device=mu_mean.device)
    wet = labels["wetness"][wet_mask].float()
    mu = mu_mean[wet_mask]
    delta_w = wet[:, None] - wet[None, :]
    delta_mu = mu[:, None] - mu[None, :]
    violation = (delta_w > 0).float() * F.relu(delta_mu)
    denom = (delta_w > 0).float().sum().clamp_min(1.0)
    return violation.sum() / denom


def soft_monotonic_risk_mu_loss(
    mu_mean: torch.Tensor,
    masks: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> torch.Tensor:
    risk_mask = masks["risk"]
    if risk_mask.sum() < 2:
        return torch.zeros((), device=mu_mean.device)
    risk = labels["risk"][risk_mask].float()
    mu = mu_mean[risk_mask]
    higher_risk = risk[:, None] > risk[None, :]
    violation = higher_risk.float() * F.relu(mu[:, None] - mu[None, :])
    denom = higher_risk.float().sum().clamp_min(1.0)
    return violation.sum() / denom


def attention_prior_kl_loss(attention: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    attn = attention.flatten(1).clamp_min(1e-8)
    prior = prior.flatten(1).clamp_min(1e-8)
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (attn * (attn.log() - prior.log())).sum(dim=1).mean()


def attention_smoothness_loss(attention: torch.Tensor) -> torch.Tensor:
    if attention.size(2) < 2 or attention.size(3) < 2:
        return torch.zeros((), device=attention.device)
    dy = (attention[:, :, 1:, :] - attention[:, :, :-1, :]).abs().mean()
    dx = (attention[:, :, :, 1:] - attention[:, :, :, :-1]).abs().mean()
    return dx + dy


def attention_query_diversity_loss(attention_queries: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, float]]:
    """Discourage multiple latent evidence masks from collapsing to one region.

    This is a segmentation-style regularizer for query masks. It is deliberately
    weak and optional: public datasets do not provide pixel labels, so the loss
    only penalizes redundant query masks and leaves the supervised friction/risk
    labels to decide whether multiple local evidence regions are useful.
    """
    if attention_queries is None or attention_queries.ndim != 4 or attention_queries.size(1) < 2:
        device = attention_queries.device if isinstance(attention_queries, torch.Tensor) else torch.device("cpu")
        return torch.zeros((), device=device), {
            "evidence_query_attention_overlap": 0.0,
            "loss_evidence_query_diversity": 0.0,
        }
    attn = attention_queries.flatten(2)
    attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(1e-8)
    norm = F.normalize(attn, dim=2, eps=1e-8)
    overlap = torch.bmm(norm, norm.transpose(1, 2))
    q = overlap.size(1)
    off_diag = ~torch.eye(q, device=overlap.device, dtype=torch.bool).view(1, q, q)
    loss = overlap.masked_select(off_diag).mean()
    return loss, {
        "evidence_query_attention_overlap": float(loss.detach().cpu()),
        "loss_evidence_query_diversity": float(loss.detach().cpu()),
    }


def attention_region_mass_losses(
    attention: torch.Tensor,
    *,
    bottom_target: float = 0.0,
    center_bottom_target: float = 0.0,
    top_max: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Softly keep weak evidence attention on plausible road-contact regions."""
    if attention.ndim != 4:
        return torch.zeros((), device=attention.device), {}
    _, _, h, w = attention.shape
    if h <= 0 or w <= 0:
        return torch.zeros((), device=attention.device), {}
    yy = torch.linspace(0.0, 1.0, h, device=attention.device, dtype=attention.dtype).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=attention.device, dtype=attention.dtype).view(1, 1, 1, w)
    bottom_half = (yy >= 0.5).to(dtype=attention.dtype)
    center = ((xx >= 0.25) & (xx <= 0.75)).to(dtype=attention.dtype)
    top_half = (yy < 0.5).to(dtype=attention.dtype)
    attn = attention / attention.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    bottom_mass = (attn * bottom_half).sum(dim=(2, 3)).squeeze(1)
    center_bottom_mass = (attn * bottom_half * center).sum(dim=(2, 3)).squeeze(1)
    top_mass = (attn * top_half).sum(dim=(2, 3)).squeeze(1)

    losses = []
    if bottom_target > 0:
        losses.append(F.relu(float(bottom_target) - bottom_mass).mean())
    if center_bottom_target > 0:
        losses.append(F.relu(float(center_bottom_target) - center_bottom_mass).mean())
    if top_max < 1.0:
        losses.append(F.relu(top_mass - float(top_max)).mean())
    if not losses:
        total = torch.zeros((), device=attention.device)
    else:
        total = torch.stack(losses).mean()
    return total, {
        "attention_bottom_half_mass": float(bottom_mass.mean().detach().cpu()),
        "attention_center_bottom_mass": float(center_bottom_mass.mean().detach().cpu()),
        "attention_top_half_mass": float(top_mass.mean().detach().cpu()),
        "loss_evidence_attention_region": float(total.detach().cpu()),
    }


def attention_soft_mask_mass_loss(
    attention: torch.Tensor,
    soft_mask: torch.Tensor | None,
    *,
    min_mass: float = 0.0,
    threshold: float = 0.0,
    sharpness: float = 12.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Encourage attention to remain inside a soft pseudo-road mask.

    The current public road-condition datasets do not include pixel-level road
    masks. This loss accepts any soft road-likelihood map, including the
    built-in heuristic mask now and external SegFormer/SAM pseudo masks later.
    It penalizes attention mass outside the pseudo-road support without forcing
    a brittle one-hot attention target.
    """
    if attention.ndim != 4 or soft_mask is None:
        device = attention.device
        return torch.zeros((), device=device), {
            "attention_pseudo_road_mass": 0.0,
            "attention_pseudo_nonroad_mass": 0.0,
            "loss_evidence_attention_pseudo_road": 0.0,
        }
    if soft_mask.shape[-2:] != attention.shape[-2:]:
        soft_mask = F.interpolate(
            soft_mask,
            size=attention.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if soft_mask.size(1) != 1:
        soft_mask = soft_mask.mean(dim=1, keepdim=True)
    mask = soft_mask.to(device=attention.device, dtype=attention.dtype).clamp(0.0, 1.0)
    if threshold > 0:
        mask = torch.sigmoid((mask - float(threshold)) * float(sharpness))
    attn = attention / attention.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    road_mass = (attn * mask).sum(dim=(2, 3)).squeeze(1)
    nonroad_mass = (attn * (1.0 - mask)).sum(dim=(2, 3)).squeeze(1)
    if min_mass > 0:
        loss = F.relu(float(min_mass) - road_mass).mean()
    else:
        loss = nonroad_mass.mean()
    return loss, {
        "attention_pseudo_road_mass": float(road_mass.mean().detach().cpu()),
        "attention_pseudo_nonroad_mass": float(nonroad_mass.mean().detach().cpu()),
        "loss_evidence_attention_pseudo_road": float(loss.detach().cpu()),
    }


def risk_distribution_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    return 0.5 * (
        F.kl_div(log_a, prob_b, reduction="batchmean")
        + F.kl_div(log_b, prob_a, reduction="batchmean")
    )


def prediction_consistency_loss(
    student_outputs: dict,
    teacher_outputs: dict,
    *,
    logit_tasks: tuple[str, ...] = ("friction", "risk", "wetness", "snow"),
    interval_weight: float = 1.0,
    attention_weight: float = 0.0,
    attention_mask: torch.Tensor | None = None,
    attention_mask_mode: str = "none",
    attention_mask_threshold: float = 0.0,
    attention_mask_sharpness: float = 12.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Consistency between a clean view and a weakly perturbed view.

    The teacher side is detached. This is a lightweight robustness regularizer:
    it encourages the friction/risk affordance prediction to be invariant to
    small camera-style changes while preserving gradients for the perturbed
    student view.
    """
    device = student_outputs["mu_mean"].device
    losses: list[torch.Tensor] = []
    logs: dict[str, float] = {}

    for task in logit_tasks:
        student_logits = student_outputs.get("logits", {}).get(task)
        teacher_logits = teacher_outputs.get("logits", {}).get(task)
        if student_logits is None or teacher_logits is None:
            continue
        n = min(student_logits.size(0), teacher_logits.size(0))
        if n <= 0:
            continue
        student_log_prob = F.log_softmax(student_logits[:n], dim=1)
        teacher_prob = F.softmax(teacher_logits[:n].detach(), dim=1)
        task_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        losses.append(task_loss)
        logs[f"loss_aug_consistency_{task}"] = float(task_loss.detach().cpu())

    if interval_weight > 0:
        n = min(student_outputs["mu_interval"].size(0), teacher_outputs["mu_interval"].size(0))
        if n > 0:
            interval_loss = F.smooth_l1_loss(
                student_outputs["mu_interval"][:n],
                teacher_outputs["mu_interval"][:n].detach(),
            )
            mean_loss = F.smooth_l1_loss(
                student_outputs["mu_mean"][:n],
                teacher_outputs["mu_mean"][:n].detach(),
            )
            interval_loss = interval_loss + 0.5 * mean_loss
            losses.append(float(interval_weight) * interval_loss)
            logs["loss_aug_consistency_interval"] = float(interval_loss.detach().cpu())

    if attention_weight > 0:
        student_ev = student_outputs.get("evidence_field")
        teacher_ev = teacher_outputs.get("evidence_field")
        if student_ev and teacher_ev:
            student_attn = student_ev.get("attention")
            teacher_attn = teacher_ev.get("attention")
            if student_attn is not None and teacher_attn is not None:
                n = min(student_attn.size(0), teacher_attn.size(0))
                if n > 0:
                    mask = _consistency_attention_mask(
                        attention_mask[:n] if attention_mask is not None else None,
                        student_ev,
                        teacher_ev,
                        n=n,
                        target_size=student_attn.shape[-2:],
                        mode=attention_mask_mode,
                        threshold=attention_mask_threshold,
                        sharpness=attention_mask_sharpness,
                    )
                    if mask is None:
                        attention_loss = F.smooth_l1_loss(
                            student_attn[:n],
                            teacher_attn[:n].detach(),
                        )
                        logs["aug_consistency_attention_mask_mean"] = 1.0
                    else:
                        element = F.smooth_l1_loss(
                            student_attn[:n],
                            teacher_attn[:n].detach(),
                            reduction="none",
                        )
                        denom = mask.sum().clamp_min(1e-8)
                        attention_loss = (element * mask).sum() / denom
                        logs["aug_consistency_attention_mask_mean"] = float(mask.mean().detach().cpu())
                    losses.append(float(attention_weight) * attention_loss)
                    logs["loss_aug_consistency_attention"] = float(attention_loss.detach().cpu())

    if not losses:
        zero = torch.zeros((), device=device)
        return zero, {"loss_aug_consistency": 0.0}
    total = torch.stack(losses).mean()
    logs["loss_aug_consistency"] = float(total.detach().cpu())
    return total, logs


def _consistency_attention_mask(
    explicit_mask: torch.Tensor | None,
    student_ev: dict,
    teacher_ev: dict,
    *,
    n: int,
    target_size: tuple[int, int],
    mode: str,
    threshold: float,
    sharpness: float,
) -> torch.Tensor | None:
    mode = str(mode or "none").lower()
    if mode in {"", "none", "full", "global"}:
        return None
    mask = None
    if mode in {"batch_road_mask", "external_road_mask"}:
        mask = explicit_mask
    elif mode in {"road_likelihood", "teacher_road_likelihood"}:
        mask = teacher_ev.get("road_likelihood")
    elif mode == "student_road_likelihood":
        mask = student_ev.get("road_likelihood")
    elif mode in {"attention_prior", "teacher_attention_prior"}:
        mask = teacher_ev.get("attention_prior")
    elif mode == "contact_prior":
        mask = teacher_ev.get("contact_prior")
    if mask is None:
        return None
    mask = mask[:n].detach()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        return None
    if mask.shape[-2:] != target_size:
        mask = F.interpolate(mask, size=target_size, mode="bilinear", align_corners=False)
    if mask.size(1) != 1:
        mask = mask.mean(dim=1, keepdim=True)
    mask = mask.clamp(0.0, 1.0)
    if float(threshold) > 0:
        mask = torch.sigmoid((mask - float(threshold)) * float(sharpness))
    if float(mask.sum().detach().cpu()) <= 1e-8:
        return None
    return mask


def compute_total_loss(outputs, batch, cfg_loss: dict) -> tuple[torch.Tensor, dict[str, float]]:
    task_loss, logs = masked_multitask_ce(outputs["logits"], batch["labels"], batch["masks"])
    compatibility_nll = friction_set_compatibility_nll(outputs, batch["labels"], batch["masks"])
    if compatibility_nll.numel():
        compatibility_loss = compatibility_nll.mean()
    else:
        compatibility_loss = torch.zeros((), device=outputs["mu_mean"].device)
    group_loss = group_dro_loss(
        compatibility_nll,
        batch.get("group_idx"),
        temperature=float(cfg_loss.get("group_dro_temperature", 0.0)),
    )
    vrex_loss = group_vrex_loss(compatibility_nll, batch.get("group_idx"))
    risk_ordinal_loss = ordinal_risk_emd_loss(
        outputs["logits"]["risk"], batch["labels"]["risk"], batch["masks"]["risk"]
    )
    wetness_ordinal_loss = ordinal_cdf_emd_loss(
        outputs["logits"]["wetness"], batch["labels"]["wetness"], batch["masks"]["wetness"]
    )
    interval_loss = interval_censored_logistic_loss(
        outputs["mu_mean"], outputs["mu_scale"], batch["mu_interval"], batch["mu_mask"]
    )
    coverage_sample_weight, coverage_weight_logs = build_safety_coverage_weights(outputs, batch, cfg_loss)
    coverage_loss = interval_coverage_loss(
        outputs["mu_interval"],
        batch["mu_interval"],
        batch["mu_mask"],
        margin=float(cfg_loss.get("coverage_margin", 0.0)),
        sample_weight=coverage_sample_weight,
    )
    width_loss = interval_width_loss(outputs["mu_interval"], batch["mu_mask"])
    endpoint_loss = interval_endpoint_loss(
        outputs["mu_interval"], outputs["mu_mean"], batch["mu_interval"], batch["mu_mask"]
    )
    target_width_loss = interval_target_width_loss(
        outputs["mu_interval"], batch["mu_interval"], batch["mu_mask"]
    )
    interval_order_loss = interval_order_consistency_loss(
        outputs["mu_mean"],
        batch["mu_interval"],
        batch["mu_mask"],
        margin_scale=float(cfg_loss.get("interval_order_margin_scale", 0.35)),
        min_gap=float(cfg_loss.get("interval_order_min_gap", 0.02)),
    )
    monotonic_loss = soft_monotonic_wetness_loss(
        outputs["logits"], outputs["mu_mean"], batch["masks"], batch["labels"]
    )
    risk_mu_loss = soft_monotonic_risk_mu_loss(outputs["mu_mean"], batch["masks"], batch["labels"])
    if "domain_logits" in outputs and "domain_idx" in batch:
        domain_loss = F.cross_entropy(outputs["domain_logits"], batch["domain_idx"])
    else:
        domain_loss = torch.zeros((), device=outputs["mu_mean"].device)
    coral_loss = feature_coral_loss(
        outputs["features"],
        batch.get("domain_idx"),
        min_samples_per_domain=int(cfg_loss.get("coral_min_samples_per_domain", 2)),
    )
    risk_conditional_coral_loss = feature_coral_loss(
        outputs["features"],
        batch.get("domain_idx"),
        condition_idx=batch["labels"].get("risk"),
        condition_mask=batch["masks"].get("risk"),
        min_samples_per_domain=int(cfg_loss.get("coral_min_samples_per_domain", 2)),
    )
    wetness_conditional_coral_loss = feature_coral_loss(
        outputs["features"],
        batch.get("domain_idx"),
        condition_idx=batch["labels"].get("wetness"),
        condition_mask=batch["masks"].get("wetness"),
        min_samples_per_domain=int(cfg_loss.get("coral_min_samples_per_domain", 2)),
    )
    state_contrastive_temperature = float(cfg_loss.get("state_contrastive_temperature", 0.2))
    state_contrastive_cross_domain_only = bool(cfg_loss.get("state_contrastive_cross_domain_only", True))
    risk_state_contrastive_loss = state_contrastive_loss(
        outputs["features"],
        batch["labels"].get("risk"),
        batch["masks"].get("risk"),
        batch.get("domain_idx"),
        temperature=state_contrastive_temperature,
        cross_domain_only=state_contrastive_cross_domain_only,
    )
    friction_state_contrastive_loss = state_contrastive_loss(
        outputs["features"],
        batch["labels"].get("friction"),
        batch["masks"].get("friction"),
        batch.get("domain_idx"),
        temperature=state_contrastive_temperature,
        cross_domain_only=state_contrastive_cross_domain_only,
    )
    wetness_state_contrastive_loss = state_contrastive_loss(
        outputs["features"],
        batch["labels"].get("wetness"),
        batch["masks"].get("wetness"),
        batch.get("domain_idx"),
        temperature=state_contrastive_temperature,
        cross_domain_only=state_contrastive_cross_domain_only,
    )
    evidence = outputs.get("evidence_field")
    if evidence:
        evidence_risk_loss = ordinal_risk_emd_loss(
            evidence["risk_logits"], batch["labels"]["risk"], batch["masks"]["risk"]
        )
        evidence_interval_loss = interval_censored_logistic_loss(
            evidence["mu_mean"], evidence["mu_scale"], batch["mu_interval"], batch["mu_mask"]
        )
        evidence_endpoint_loss = interval_endpoint_loss(
            evidence["mu_interval"], evidence["mu_mean"], batch["mu_interval"], batch["mu_mask"]
        )
        evidence_width_loss = interval_width_loss(evidence["mu_interval"], batch["mu_mask"])
        evidence_target_width_loss = interval_target_width_loss(
            evidence["mu_interval"], batch["mu_interval"], batch["mu_mask"]
        )
        evidence_attention_prior = attention_prior_kl_loss(
            evidence["attention"], evidence.get("attention_prior", evidence["contact_prior"])
        )
        evidence_attention_smooth = attention_smoothness_loss(evidence["attention"])
        evidence_query_diversity, query_diversity_logs = attention_query_diversity_loss(
            evidence.get("attention_queries")
        )
        evidence_attention_region, attention_region_logs = attention_region_mass_losses(
            evidence["attention"],
            bottom_target=float(cfg_loss.get("evidence_bottom_mass_target", 0.0)),
            center_bottom_target=float(cfg_loss.get("evidence_center_bottom_mass_target", 0.0)),
            top_max=float(cfg_loss.get("evidence_top_mass_max", 1.0)),
        )
        pseudo_road_mask = batch.get("road_mask", evidence.get("road_likelihood"))
        evidence_attention_pseudo_road, pseudo_road_logs = attention_soft_mask_mass_loss(
            evidence["attention"],
            pseudo_road_mask,
            min_mass=float(cfg_loss.get("evidence_pseudo_road_min_mass", 0.0)),
            threshold=float(cfg_loss.get("evidence_pseudo_road_threshold", 0.0)),
            sharpness=float(cfg_loss.get("evidence_pseudo_road_sharpness", 12.0)),
        )
        risk_consistency_loss = risk_distribution_consistency_loss(
            outputs["logits"]["risk"], evidence["risk_logits"]
        )
        reference_interval = outputs.get("mu_interval_before_evidence", outputs["mu_interval"])
        interval_consistency_loss = F.smooth_l1_loss(reference_interval, evidence["mu_interval"])
    else:
        evidence_risk_loss = torch.zeros((), device=outputs["mu_mean"].device)
        evidence_interval_loss = torch.zeros((), device=outputs["mu_mean"].device)
        evidence_endpoint_loss = torch.zeros((), device=outputs["mu_mean"].device)
        evidence_width_loss = torch.zeros((), device=outputs["mu_mean"].device)
        evidence_target_width_loss = torch.zeros((), device=outputs["mu_mean"].device)
        evidence_attention_prior = torch.zeros((), device=outputs["mu_mean"].device)
        evidence_attention_smooth = torch.zeros((), device=outputs["mu_mean"].device)
        evidence_query_diversity = torch.zeros((), device=outputs["mu_mean"].device)
        query_diversity_logs = {
            "evidence_query_attention_overlap": 0.0,
            "loss_evidence_query_diversity": 0.0,
        }
        evidence_attention_region = torch.zeros((), device=outputs["mu_mean"].device)
        evidence_attention_pseudo_road = torch.zeros((), device=outputs["mu_mean"].device)
        attention_region_logs = {
            "attention_bottom_half_mass": 0.0,
            "attention_center_bottom_mass": 0.0,
            "attention_top_half_mass": 0.0,
            "loss_evidence_attention_region": 0.0,
        }
        pseudo_road_logs = {
            "attention_pseudo_road_mass": 0.0,
            "attention_pseudo_nonroad_mass": 0.0,
            "loss_evidence_attention_pseudo_road": 0.0,
        }
        risk_consistency_loss = torch.zeros((), device=outputs["mu_mean"].device)
        interval_consistency_loss = torch.zeros((), device=outputs["mu_mean"].device)
    domain_adapter_loss = outputs.get("domain_adapter_penalty")
    if domain_adapter_loss is None:
        domain_adapter_loss = torch.zeros((), device=outputs["mu_mean"].device)
    total = (
        cfg_loss.get("task_weight", 1.0) * task_loss
        + cfg_loss.get("compatibility_weight", 0.0) * compatibility_loss
        + cfg_loss.get("group_dro_weight", 0.0) * group_loss
        + cfg_loss.get("group_vrex_weight", 0.0) * vrex_loss
        + cfg_loss.get("risk_ordinal_weight", 0.0) * risk_ordinal_loss
        + cfg_loss.get("wetness_ordinal_weight", 0.0) * wetness_ordinal_loss
        + cfg_loss.get("interval_weight", 0.25) * interval_loss
        + cfg_loss.get("coverage_weight", 0.0) * coverage_loss
        + cfg_loss.get("width_weight", 0.0) * width_loss
        + cfg_loss.get("endpoint_weight", 0.0) * endpoint_loss
        + cfg_loss.get("target_width_weight", 0.0) * target_width_loss
        + cfg_loss.get("interval_order_weight", 0.0) * interval_order_loss
        + cfg_loss.get("monotonic_weight", 0.0) * monotonic_loss
        + cfg_loss.get("risk_mu_monotonic_weight", 0.0) * risk_mu_loss
        + cfg_loss.get("domain_weight", 0.0) * domain_loss
        + cfg_loss.get("feature_coral_weight", 0.0) * coral_loss
        + cfg_loss.get("risk_conditional_coral_weight", 0.0) * risk_conditional_coral_loss
        + cfg_loss.get("wetness_conditional_coral_weight", 0.0) * wetness_conditional_coral_loss
        + cfg_loss.get("risk_state_contrastive_weight", 0.0) * risk_state_contrastive_loss
        + cfg_loss.get("friction_state_contrastive_weight", 0.0) * friction_state_contrastive_loss
        + cfg_loss.get("wetness_state_contrastive_weight", 0.0) * wetness_state_contrastive_loss
        + cfg_loss.get("evidence_risk_weight", 0.0) * evidence_risk_loss
        + cfg_loss.get("evidence_interval_weight", 0.0) * evidence_interval_loss
        + cfg_loss.get("evidence_endpoint_weight", 0.0) * evidence_endpoint_loss
        + cfg_loss.get("evidence_width_weight", 0.0) * evidence_width_loss
        + cfg_loss.get("evidence_target_width_weight", 0.0) * evidence_target_width_loss
        + cfg_loss.get("evidence_attention_prior_weight", 0.0) * evidence_attention_prior
        + cfg_loss.get("evidence_attention_smooth_weight", 0.0) * evidence_attention_smooth
        + cfg_loss.get("evidence_query_diversity_weight", 0.0) * evidence_query_diversity
        + cfg_loss.get("evidence_attention_region_weight", 0.0) * evidence_attention_region
        + cfg_loss.get("evidence_attention_pseudo_road_weight", 0.0) * evidence_attention_pseudo_road
        + cfg_loss.get("evidence_risk_consistency_weight", 0.0) * risk_consistency_loss
        + cfg_loss.get("evidence_interval_consistency_weight", 0.0) * interval_consistency_loss
        + cfg_loss.get("domain_adapter_weight", 0.0) * domain_adapter_loss
    )
    logs.update(
        {
            "loss_task": float(task_loss.detach().cpu()),
            "loss_compatibility": float(compatibility_loss.detach().cpu()),
            "loss_group_dro": float(group_loss.detach().cpu()),
            "loss_group_vrex": float(vrex_loss.detach().cpu()),
            "loss_risk_ordinal": float(risk_ordinal_loss.detach().cpu()),
            "loss_wetness_ordinal": float(wetness_ordinal_loss.detach().cpu()),
            "loss_interval": float(interval_loss.detach().cpu()),
            "loss_coverage": float(coverage_loss.detach().cpu()),
            "loss_width": float(width_loss.detach().cpu()),
            "loss_endpoint": float(endpoint_loss.detach().cpu()),
            "loss_target_width": float(target_width_loss.detach().cpu()),
            "loss_interval_order": float(interval_order_loss.detach().cpu()),
            "loss_monotonic": float(monotonic_loss.detach().cpu()),
            "loss_risk_mu_monotonic": float(risk_mu_loss.detach().cpu()),
            "loss_domain": float(domain_loss.detach().cpu()),
            "loss_feature_coral": float(coral_loss.detach().cpu()),
            "loss_risk_conditional_coral": float(risk_conditional_coral_loss.detach().cpu()),
            "loss_wetness_conditional_coral": float(wetness_conditional_coral_loss.detach().cpu()),
            "loss_risk_state_contrastive": float(risk_state_contrastive_loss.detach().cpu()),
            "loss_friction_state_contrastive": float(friction_state_contrastive_loss.detach().cpu()),
            "loss_wetness_state_contrastive": float(wetness_state_contrastive_loss.detach().cpu()),
            "loss_evidence_risk": float(evidence_risk_loss.detach().cpu()),
            "loss_evidence_interval": float(evidence_interval_loss.detach().cpu()),
            "loss_evidence_endpoint": float(evidence_endpoint_loss.detach().cpu()),
            "loss_evidence_width": float(evidence_width_loss.detach().cpu()),
            "loss_evidence_target_width": float(evidence_target_width_loss.detach().cpu()),
            "loss_evidence_attention_prior": float(evidence_attention_prior.detach().cpu()),
            "loss_evidence_attention_smooth": float(evidence_attention_smooth.detach().cpu()),
            "loss_evidence_query_diversity": float(evidence_query_diversity.detach().cpu()),
            "loss_evidence_attention_region": float(evidence_attention_region.detach().cpu()),
            "loss_evidence_attention_pseudo_road": float(evidence_attention_pseudo_road.detach().cpu()),
            "loss_evidence_risk_consistency": float(risk_consistency_loss.detach().cpu()),
            "loss_evidence_interval_consistency": float(interval_consistency_loss.detach().cpu()),
            "loss_domain_adapter": float(domain_adapter_loss.detach().cpu()),
            "loss_total": float(total.detach().cpu()),
            **coverage_weight_logs,
            **attention_region_logs,
            **pseudo_road_logs,
            **query_diversity_logs,
        }
    )
    return total, logs

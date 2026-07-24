from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import (  # noqa: E402
    anchor_consistency_loss,
    anchor_nonregression_barrier_loss,
    build_class_map,
    build_loaders,
    cached_teacher_logits_for_batch,
    factor_confusion_summary,
    load_teacher_logit_cache,
    load_config,
    pareto_safe_distillation_loss,
    _atomic_torch_save,
    _move_optimizer_state_to_device,
)
from friction_affordance.models.coupled_factor_backbone import (  # noqa: E402
    CoupledFactorBackboneConfig,
    RSCDCoupledFactorFactorizedClassifier,
    count_parameters,
)
from friction_affordance.rscd_factors import FACTOR_AXES, FACTOR_LABELS, sanity_summary  # noqa: E402
from friction_affordance.utils import set_seed  # noqa: E402


def build_s136_model(cfg: dict[str, Any], class_to_idx: dict[str, int]) -> RSCDCoupledFactorFactorizedClassifier:
    m = cfg.get("model", {})
    stage_dims = tuple(int(v) for v in m.get("stage_dims", [96, 192, 384]))
    if len(stage_dims) != 3:
        raise ValueError(f"model.stage_dims must contain exactly 3 values, got {stage_dims}")
    backbone_cfg = CoupledFactorBackboneConfig(
        in_channels=int(m.get("in_channels", 3)),
        stem_dim=int(m.get("stem_dim", 48)),
        stage_dims=stage_dims,  # type: ignore[arg-type]
        dropout=float(m.get("dropout", 0.15)),
        coupling_gate_mode=str(m.get("coupling_gate_mode", "learned")),
        use_concrete_roughness_scale_space=bool(m.get("use_concrete_roughness_scale_space", False)),
        concrete_roughness_scale_space_mode=str(m.get("concrete_roughness_scale_space_mode", "learned")),
        concrete_roughness_scale_space_scale=float(m.get("concrete_roughness_scale_space_scale", 0.18)),
        use_dual_film_texture_coupling=bool(m.get("use_dual_film_texture_coupling", False)),
        dual_film_texture_coupling_mode=str(m.get("dual_film_texture_coupling_mode", "learned")),
        dual_film_texture_coupling_scale=float(m.get("dual_film_texture_coupling_scale", 0.16)),
    )
    return RSCDCoupledFactorFactorizedClassifier(class_to_idx=class_to_idx, cfg=backbone_cfg)


def load_s136_checkpoint(model: torch.nn.Module, checkpoint: str | None, device: torch.device) -> dict[str, Any]:
    if not checkpoint:
        return {"loaded": False, "path": None}
    path = Path(str(checkpoint))
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=False)
    raw = state.get("model", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(raw, strict=False)
    print(f"Loaded S136 checkpoint: {path}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    return {
        "loaded": True,
        "path": str(path),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def resolve_training_device(device_name: str) -> torch.device:
    requested = str(device_name).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def coupling_gate_targets(friction: torch.Tensor, material: torch.Tensor) -> torch.Tensor:
    friction_labels = list(FACTOR_LABELS["friction"])
    material_labels = list(FACTOR_LABELS["material"])
    dry = friction_labels.index("dry")
    wet = friction_labels.index("wet")
    water = friction_labels.index("water")
    asphalt = material_labels.index("asphalt")
    concrete = material_labels.index("concrete")
    target = torch.full_like(friction, 4)
    target[(material == concrete) & (friction == water)] = 0
    target[(material == concrete) & (friction == wet)] = 1
    target[(material == concrete) & (friction == dry)] = 2
    target[(material == asphalt)] = 3
    return target


def concrete_roughness_route_targets(friction: torch.Tensor, material: torch.Tensor) -> torch.Tensor:
    friction_labels = list(FACTOR_LABELS["friction"])
    material_labels = list(FACTOR_LABELS["material"])
    dry = friction_labels.index("dry")
    wet = friction_labels.index("wet")
    water = friction_labels.index("water")
    concrete = material_labels.index("concrete")
    target = torch.full_like(friction, 2)
    target[(material == concrete) & (friction == dry)] = 0
    target[(material == concrete) & ((friction == wet) | (friction == water))] = 1
    return target


def dual_film_texture_route_targets(friction: torch.Tensor, material: torch.Tensor) -> torch.Tensor:
    friction_labels = list(FACTOR_LABELS["friction"])
    material_labels = list(FACTOR_LABELS["material"])
    dry = friction_labels.index("dry")
    wet = friction_labels.index("wet")
    water = friction_labels.index("water")
    concrete = material_labels.index("concrete")
    asphalt = material_labels.index("asphalt")
    target = torch.full_like(friction, 3)
    target[(material == concrete) & (friction == dry)] = 0
    target[(material == concrete) & ((friction == wet) | (friction == water))] = 1
    target[(material == asphalt)] = 2
    return target


def safe_distill_schedule_scale(loss_cfg: dict[str, Any], epoch: int | None) -> float:
    """Ramp teacher no-harm guard after the custom RSCD factors have started learning."""

    if epoch is None:
        return 1.0
    def _cfg_float(key: str, default: float) -> float:
        value = loss_cfg.get(key, default)
        if value is None or value == "":
            return float(default)
        return float(value)

    warmup = max(_cfg_float("safe_distill_warmup_epochs", 0.0), 0.0)
    ramp = max(_cfg_float("safe_distill_ramp_epochs", 0.0), 0.0)
    initial = min(max(_cfg_float("safe_distill_initial_scale", 0.0), 0.0), 1.0)
    final = min(max(_cfg_float("safe_distill_final_scale", 1.0), 0.0), 1.0)
    epoch_value = float(epoch)
    if epoch_value <= warmup:
        return initial
    if ramp <= 0.0:
        return final
    progress = min(max((epoch_value - warmup) / ramp, 0.0), 1.0)
    return initial + (final - initial) * progress


def _factor_marginal_probs(
    class_probs: torch.Tensor,
    class_to_factor: torch.Tensor,
    *,
    axis_idx: int,
    num_values: int,
) -> torch.Tensor:
    """Marginalize 27-class probabilities into one RSCD factor axis."""

    factor_ids = class_to_factor[:, axis_idx]
    valid = factor_ids.ge(0)
    out = class_probs.new_zeros((class_probs.shape[0], int(num_values)))
    if bool(valid.any()):
        idx = factor_ids[valid].view(1, -1).expand(class_probs.shape[0], -1)
        out.scatter_add_(1, idx, class_probs[:, valid])
    return out.clamp_min(1e-8)


def rscd_factor_relational_teacher_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    spec: Any,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Transfer teacher knowledge as class, factor, and hard-pair relations.

    This is RSCD-specific: the teacher's 27-class distribution is decomposed
    into friction/material/roughness marginals, then local hard-pair ratios are
    matched on class pairs that differ in exactly one physical factor.
    """

    class_weight = float(loss_cfg.get("teacher_class_kd_weight", 0.0))
    factor_weight = float(loss_cfg.get("teacher_factor_marginal_kd_weight", 0.0))
    pair_weight = float(loss_cfg.get("teacher_hardpair_relation_kd_weight", 0.0))
    if class_weight <= 0.0 and factor_weight <= 0.0 and pair_weight <= 0.0:
        return student_logits.new_zeros(()), {
            "loss_teacher_class_kd": 0.0,
            "loss_teacher_factor_marginal_kd": 0.0,
            "loss_teacher_hardpair_relation_kd": 0.0,
            "loss_teacher_factor_relational_distill": 0.0,
        }

    temperature = max(float(loss_cfg.get("teacher_kd_temperature", 2.5)), 1e-3)
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        s = student_logits.float() / temperature
        t = teacher_logits.detach().float() / temperature
        total = student_logits.new_zeros((), dtype=torch.float32)

        if class_weight > 0.0:
            class_kd = F.kl_div(
                F.log_softmax(s, dim=1),
                F.softmax(t, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            total = total + float(class_weight) * class_kd
        else:
            class_kd = s.new_zeros(())

        if factor_weight > 0.0:
            axis_weights = loss_cfg.get("teacher_factor_marginal_axis_weights", {}) or {}
            class_to_factor = spec.class_to_factor.to(device=student_logits.device)
            s_prob = F.softmax(s, dim=1)
            t_prob = F.softmax(t, dim=1)
            factor_terms: list[torch.Tensor] = []
            factor_weight_sum = 0.0
            for axis_idx, axis in enumerate(FACTOR_AXES):
                axis_w = float(axis_weights.get(axis, 1.0))
                if axis_w <= 0.0:
                    continue
                s_axis = _factor_marginal_probs(
                    s_prob,
                    class_to_factor,
                    axis_idx=axis_idx,
                    num_values=len(FACTOR_LABELS[axis]),
                )
                t_axis = _factor_marginal_probs(
                    t_prob,
                    class_to_factor,
                    axis_idx=axis_idx,
                    num_values=len(FACTOR_LABELS[axis]),
                )
                axis_kd = F.kl_div(s_axis.log(), t_axis, reduction="batchmean") * (temperature * temperature)
                factor_terms.append(float(axis_w) * axis_kd)
                factor_weight_sum += float(axis_w)
            factor_kd = torch.stack(factor_terms).sum() / max(factor_weight_sum, 1e-8) if factor_terms else s.new_zeros(())
            total = total + float(factor_weight) * factor_kd
        else:
            factor_kd = s.new_zeros(())

        if pair_weight > 0.0 and getattr(spec, "hard_pairs", None):
            axis_weights = loss_cfg.get("teacher_hardpair_axis_weights", {}) or {}
            pair_terms: list[torch.Tensor] = []
            pair_weight_sum = 0.0
            for pair in spec.hard_pairs:
                axis_w = float(axis_weights.get(pair.axis, 1.0))
                if axis_w <= 0.0:
                    continue
                pair_idx = torch.as_tensor([int(pair.left), int(pair.right)], device=student_logits.device)
                s_pair = s.index_select(1, pair_idx)
                t_pair = t.index_select(1, pair_idx)
                pair_kd = F.kl_div(
                    F.log_softmax(s_pair, dim=1),
                    F.softmax(t_pair, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                pair_terms.append(float(axis_w) * pair_kd)
                pair_weight_sum += float(axis_w)
            hardpair_kd = torch.stack(pair_terms).sum() / max(pair_weight_sum, 1e-8) if pair_terms else s.new_zeros(())
            total = total + float(pair_weight) * hardpair_kd
        else:
            hardpair_kd = s.new_zeros(())

    total = total.to(dtype=student_logits.dtype)
    return total, {
        "loss_teacher_class_kd": float(class_kd.detach().cpu()),
        "loss_teacher_factor_marginal_kd": float(factor_kd.detach().cpu()),
        "loss_teacher_hardpair_relation_kd": float(hardpair_kd.detach().cpu()),
        "loss_teacher_factor_relational_distill": float(total.detach().cpu()),
    }


def s136_loss(
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
    idx_to_class: dict[int, str],
    *,
    teacher_logits: torch.Tensor | None = None,
    spec: Any | None = None,
    teacher_guard_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = out["logits"]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("out['logits'] must be a tensor")
    label = batch["label"].to(logits.device)
    class_loss = F.cross_entropy(logits, label)
    factor_logits = out.get("factor_logits")
    if not isinstance(factor_logits, dict):
        raise TypeError("out['factor_logits'] must be a dict")
    factor_terms: list[torch.Tensor] = []
    factor_axis_weights = loss_cfg.get("factor_axis_weights", {}) or {}
    for axis in FACTOR_AXES:
        target = batch[f"{axis}_factor"].to(logits.device)
        axis_logits = factor_logits[axis]
        axis_loss = F.cross_entropy(axis_logits, target, ignore_index=-100)
        factor_terms.append(float(factor_axis_weights.get(axis, 1.0)) * axis_loss)
    factor_loss = torch.stack(factor_terms).sum()
    coupling_weights = out["coupling_weights"]
    if not isinstance(coupling_weights, torch.Tensor):
        raise TypeError("out['coupling_weights'] must be a tensor")
    gate_target = coupling_gate_targets(
        batch["friction_factor"].to(logits.device),
        batch["material_factor"].to(logits.device),
    )
    gate_loss = F.nll_loss(coupling_weights.clamp_min(1e-6).log(), gate_target)
    total = (
        float(loss_cfg.get("class_weight", 1.0)) * class_loss
        + float(loss_cfg.get("factor_weight", 0.25)) * factor_loss
        + float(loss_cfg.get("coupling_gate_weight", 0.06)) * gate_loss
    )
    logs = {
        "loss_class": float(class_loss.detach().cpu()),
        "loss_factor": float(factor_loss.detach().cpu()),
        "loss_gate": float(gate_loss.detach().cpu()),
    }
    route_weight = float(loss_cfg.get("concrete_roughness_route_weight", 0.0))
    concrete_route_weights = out.get("concrete_roughness_route_weights")
    if route_weight > 0.0 and isinstance(concrete_route_weights, torch.Tensor):
        route_target = concrete_roughness_route_targets(
            batch["friction_factor"].to(logits.device),
            batch["material_factor"].to(logits.device),
        )
        route_loss = F.nll_loss(concrete_route_weights.clamp_min(1e-6).log(), route_target)
        total = total + route_weight * route_loss
        logs["loss_concrete_roughness_route"] = float(route_loss.detach().cpu())
    else:
        logs["loss_concrete_roughness_route"] = 0.0
    dual_route_weight = float(loss_cfg.get("dual_film_texture_route_weight", 0.0))
    dual_route_weights = out.get("dual_film_texture_route_weights")
    if dual_route_weight > 0.0 and isinstance(dual_route_weights, torch.Tensor):
        dual_target = dual_film_texture_route_targets(
            batch["friction_factor"].to(logits.device),
            batch["material_factor"].to(logits.device),
        )
        dual_route_loss = F.nll_loss(dual_route_weights.clamp_min(1e-6).log(), dual_target)
        total = total + dual_route_weight * dual_route_loss
        logs["loss_dual_film_texture_route"] = float(dual_route_loss.detach().cpu())
    else:
        logs["loss_dual_film_texture_route"] = 0.0
    if teacher_logits is not None:
        if spec is not None:
            relational_loss, relational_logs = rscd_factor_relational_teacher_distill_loss(
                logits,
                teacher_logits,
                spec,
                loss_cfg,
            )
            total = total + relational_loss
            logs.update(relational_logs)
        else:
            logs.update(
                {
                    "loss_teacher_class_kd": 0.0,
                    "loss_teacher_factor_marginal_kd": 0.0,
                    "loss_teacher_hardpair_relation_kd": 0.0,
                    "loss_teacher_factor_relational_distill": 0.0,
                }
            )
        scale = min(max(float(teacher_guard_scale), 0.0), 1.0)
        logs["safe_distill_scale"] = scale
        if scale > 0.0:
            anchor_loss, anchor_logs = anchor_consistency_loss(
                logits,
                teacher_logits,
                label,
                idx_to_class,
                loss_cfg,
            )
            nonregression_loss, nonregression_logs = anchor_nonregression_barrier_loss(
                logits,
                teacher_logits,
                label,
                idx_to_class,
                loss_cfg,
            )
            if spec is not None:
                pareto_loss, pareto_logs = pareto_safe_distillation_loss(
                    logits,
                    teacher_logits,
                    label,
                    idx_to_class,
                    spec,
                    loss_cfg,
                )
            else:
                pareto_loss = logits.new_zeros(())
                pareto_logs = {}
            total = total + scale * (anchor_loss + nonregression_loss + pareto_loss)
            logs.update(anchor_logs)
            logs.update(nonregression_logs)
            logs.update(pareto_logs)
            logs["loss_safe_distill_scaled"] = float((scale * (anchor_loss + nonregression_loss + pareto_loss)).detach().cpu())
        else:
            logs.update(
                {
                    "loss_anchor_consistency": 0.0,
                    "loss_anchor_nonregression": 0.0,
                    "loss_pareto_safe_distill": 0.0,
                    "loss_safe_distill_scaled": 0.0,
                }
            )
    return total, logs


def train_one_epoch_s136(
    model: RSCDCoupledFactorFactorizedClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    scaler: torch.amp.GradScaler,
    idx_to_class: dict[int, str],
    teacher_logit_cache: dict[str, torch.Tensor] | None = None,
    teacher_cache_strict: bool = False,
    epoch: int | None = None,
    out_dir: Path | None = None,
    class_to_idx: dict[str, int] | None = None,
) -> dict[str, float]:
    model.train()
    train_cfg = cfg["train"]
    loss_cfg = cfg.get("loss", {})
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    accum = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    log_every = int(train_cfg.get("log_every_steps", 80))
    teacher_guard_scale = safe_distill_schedule_scale(loss_cfg, epoch)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    log_sum: dict[str, float] = {}
    log_count = 0
    resume_start_step = int(train_cfg.get("_resume_start_step", train_cfg.get("resume_start_step", 0)) or 0)
    total_steps = resume_start_step + len(loader)
    step_checkpoint_every = int(train_cfg.get("save_step_checkpoint_every", 0) or 0)
    for local_step, batch in enumerate(tqdm(loader, desc="train", leave=False, ascii=True), 1):
        step = resume_start_step + local_step
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        batch_on_device = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(image, return_aux=True)
            logits = out["logits"]
            if not isinstance(logits, torch.Tensor):
                raise TypeError("out['logits'] must be a tensor")
            teacher_logits, teacher_logs = cached_teacher_logits_for_batch(
                teacher_logit_cache,
                [str(v) for v in batch["image_path"]],
                device=device,
                dtype=logits.dtype,
                strict=teacher_cache_strict,
                cache_name="s136_anchor",
            )
            loss, loss_logs = s136_loss(
                out,
                batch_on_device,
                loss_cfg,
                idx_to_class,
                teacher_logits=teacher_logits,
                spec=model.spec,
                teacher_guard_scale=teacher_guard_scale,
            )
            loss_logs.update(teacher_logs)
            backward = loss / float(accum)
        if not bool(torch.isfinite(loss.detach())):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(backward).backward()
        if step % accum == 0 or local_step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip_norm", 5.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        logits = out["logits"]
        pred = logits.detach().argmax(dim=1)
        total_correct += int(pred.eq(label).sum().detach().cpu())
        total_seen += int(label.numel())
        total_loss += float(loss.detach().cpu()) * int(label.numel())
        log_count += 1
        for key, value in loss_logs.items():
            log_sum[key] = log_sum.get(key, 0.0) + float(value)
        if step_checkpoint_every > 0 and out_dir is not None and (step % step_checkpoint_every == 0 or local_step == len(loader)):
            step_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": int(epoch or 1),
                "step": int(step),
                "total_steps": int(total_steps),
                "class_to_idx": class_to_idx or {},
                "config": cfg,
                "train_partial": {
                    "loss": total_loss / max(total_seen, 1),
                    "top1": total_correct / max(total_seen, 1),
                    "seen": int(total_seen),
                },
            }
            _atomic_torch_save(step_state, out_dir / "last_step_checkpoint.pth")
            print(f"  saved step checkpoint: {out_dir / 'last_step_checkpoint.pth'} step={step}/{total_steps}")
        if log_every > 0 and (step % log_every == 0 or local_step == len(loader)):
            print(
                f"  train step {step}/{total_steps} "
                f"loss={total_loss / max(total_seen, 1):.4f} "
                f"top1={total_correct / max(total_seen, 1):.4f}"
            )
    logs = {key: value / max(log_count, 1) for key, value in log_sum.items()}
    logs.update(
        {
            "loss": total_loss / max(total_seen, 1),
            "top1": total_correct / max(total_seen, 1),
        }
    )
    return logs


@torch.no_grad()
def evaluate_s136(
    model: RSCDCoupledFactorFactorizedClassifier,
    loader,
    device: torch.device,
    idx_to_class: dict[int, str],
    *,
    save_predictions_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="eval", leave=False, ascii=True):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        out = model(image, return_aux=True)
        logits = out["logits"]
        loss = F.cross_entropy(logits, label)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        y_true.extend(label.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
        losses.append(float(loss.detach().cpu()) * int(label.numel()))
        if save_predictions_path is not None:
            for path, true_idx, pred_idx, confidence in zip(
                batch["image_path"],
                label.detach().cpu().numpy().astype(int).tolist(),
                pred.detach().cpu().numpy().astype(int).tolist(),
                conf.detach().cpu().tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "image_path": str(path),
                        "true_label": idx_to_class[int(true_idx)],
                        "pred_label": idx_to_class[int(pred_idx)],
                        "confidence": float(confidence),
                    }
                )
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[i] for i in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    total_seen = len(y_true)
    factor_summary = factor_confusion_summary(y_true, y_pred, model.spec, idx_to_class)
    hard_class_names = [
        "water_concrete_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "wet_concrete_severe",
        "water_asphalt_slight",
        "dry_concrete_slight",
    ]
    hard_scores = [float(report[name]["f1-score"]) for name in hard_class_names if name in report]
    wcs_report = report.get("water_concrete_slight", {})
    summary = {
        "loss": float(sum(losses) / max(total_seen, 1)),
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "num_samples": int(total_seen),
        "num_classes": int(len(labels)),
        "param_count": int(count_parameters(model)),
        "hard_class_mean_f1": float(np.mean(hard_scores)) if hard_scores else 0.0,
        "water_concrete_slight_precision": float(wcs_report.get("precision", 0.0)),
        "water_concrete_slight_recall": float(wcs_report.get("recall", 0.0)),
        "water_concrete_slight_f1": float(wcs_report.get("f1-score", 0.0)),
    }
    summary.update(factor_summary["summary"])
    if save_predictions_path is not None:
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(save_predictions_path, index=False, encoding="utf-8")
    return {
        "summary": summary,
        "classification_report": report,
        "factor_confusion_summary": factor_summary,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def write_s136_outputs(out_dir: Path, metrics: dict[str, Any], idx_to_class: dict[int, str], split: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in metrics.items() if key not in {"y_true", "y_pred"}}
    (out_dir / f"{split}_metrics.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = metrics["classification_report"]
    with (out_dir / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for idx in range(len(idx_to_class)):
            name = idx_to_class[idx]
            item = report.get(name, {})
            writer.writerow(
                [
                    name,
                    item.get("precision", 0.0),
                    item.get("recall", 0.0),
                    item.get("f1-score", 0.0),
                    item.get("support", 0),
                ]
            )
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(idx_to_class))))
    pd.DataFrame(
        cm,
        index=[idx_to_class[i] for i in range(len(idx_to_class))],
        columns=[idx_to_class[i] for i in range(len(idx_to_class))],
    ).to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    (out_dir / "factor_confusion_summary.json").write_text(
        json.dumps(metrics["factor_confusion_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_train(config_path: Path, *, device_override: str = "auto") -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 136)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg["train"]
    step_resume_state: dict[str, Any] | None = None
    step_resume_path = Path(str(train_cfg.get("resume_step_checkpoint_from") or (out_dir / "last_step_checkpoint.pth")))
    if bool(train_cfg.get("resume_step_checkpoint", False)) and step_resume_path.exists():
        step_resume_state = torch.load(step_resume_path, map_location="cpu", weights_only=False)
        resume_step = int(step_resume_state.get("step", 0) or 0)
        resume_total_steps = int(step_resume_state.get("total_steps", 0) or 0)
        cfg["train"]["_resume_start_step"] = resume_step
        cfg["train"]["_resume_epoch_training_complete"] = bool(resume_total_steps > 0 and resume_step >= resume_total_steps)
        cfg["train"]["_resume_step_checkpoint_path"] = str(step_resume_path)
        print(
            "Resuming S136 in-epoch checkpoint: "
            f"{step_resume_path} step={resume_step}/{resume_total_steps} "
            f"epoch_training_complete={cfg['train']['_resume_epoch_training_complete']}"
        )
    else:
        cfg["train"]["_resume_start_step"] = 0
        cfg["train"]["_resume_epoch_training_complete"] = False
    data = cfg["data"]
    manifests = [Path(data["train_manifest"]), Path(data["val_manifest"]), Path(data["test_manifest"])]
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    (out_dir / "label_factor_sanity.txt").write_text(sanity_summary(class_to_idx), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    train_loader, val_loader, test_loader = build_loaders(cfg, class_to_idx)
    device = resolve_training_device(device_override)
    model = build_s136_model(cfg, class_to_idx).to(device)
    if step_resume_state is not None:
        model.load_state_dict(step_resume_state["model"], strict=True)
        load_info = {"loaded": True, "path": str(step_resume_path), "in_epoch_resume": True}
    else:
        load_info = load_s136_checkpoint(model, cfg["train"].get("resume_from"), device)
    print(f"S136 parameter count: {count_parameters(model)}")
    teacher_logit_cache = load_teacher_logit_cache(train_cfg.get("teacher_logits_cache"))
    teacher_cache_strict = bool(train_cfg.get("teacher_logits_cache_strict", False))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(train_cfg.get("amp", True)))
    if step_resume_state is not None:
        if "optimizer" in step_resume_state:
            optimizer.load_state_dict(step_resume_state["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
        if "scaler" in step_resume_state:
            scaler.load_state_dict(step_resume_state["scaler"])
    best_key = (-1.0, -1.0)
    history: list[dict[str, Any]] = []
    if bool(train_cfg.get("evaluate_initial", False)) and step_resume_state is None:
        print("Evaluating initial S136 checkpoint before training")
        val_metrics = evaluate_s136(model, val_loader, device, idx_to_class)
        val_summary = val_metrics["summary"]
        print(f"  initial val top1={val_summary['top1']:.4f} macro_f1={val_summary['macro_f1']:.4f}")
        history.append({"epoch": 0, "train": {}, "val": val_summary, "resume": load_info})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        state = {
            "model": model.state_dict(),
            "epoch": 0,
            "class_to_idx": class_to_idx,
            "config": cfg,
            "val_summary": val_summary,
        }
        torch.save(state, out_dir / "best_checkpoint.pth")
        torch.save(state, out_dir / "best.pt")
        best_key = (float(val_summary["macro_f1"]), float(val_summary["top1"]))
        print(f"  saved initial best: {out_dir / 'best_checkpoint.pth'}")
    start_epoch = int(step_resume_state.get("epoch", 1) if step_resume_state is not None else 1)
    for epoch in range(start_epoch, int(train_cfg.get("epochs", 1)) + 1):
        if epoch > start_epoch and int(cfg["train"].get("_resume_start_step", 0) or 0) != 0:
            cfg["train"]["_resume_start_step"] = 0
            cfg["train"]["_resume_epoch_training_complete"] = False
            train_loader, _, _ = build_loaders(cfg, class_to_idx)
        print(f"Epoch {epoch}/{train_cfg.get('epochs', 1)}")
        if bool(cfg["train"].get("_resume_epoch_training_complete", False)) and epoch == start_epoch and step_resume_state is not None:
            train_metrics = dict(step_resume_state.get("train_partial", {}) or {})
            train_metrics.setdefault("loss", 0.0)
            train_metrics.setdefault("top1", 0.0)
            print(
                "  skipped S136 training epoch from completed step checkpoint "
                f"step={int(step_resume_state.get('step', 0) or 0)}/"
                f"{int(step_resume_state.get('total_steps', 0) or 0)}"
            )
        else:
            train_metrics = train_one_epoch_s136(
                model,
                train_loader,
                optimizer,
                device,
                cfg,
                scaler,
                idx_to_class,
                teacher_logit_cache=teacher_logit_cache,
                teacher_cache_strict=teacher_cache_strict,
                epoch=epoch,
                out_dir=out_dir,
                class_to_idx=class_to_idx,
            )
        val_metrics = evaluate_s136(model, val_loader, device, idx_to_class)
        val_summary = val_metrics["summary"]
        print(f"  train loss={train_metrics['loss']:.4f} top1={train_metrics['top1']:.4f}")
        print(f"  val top1={val_summary['top1']:.4f} macro_f1={val_summary['macro_f1']:.4f}")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_summary})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "class_to_idx": class_to_idx,
            "config": cfg,
            "val_summary": val_summary,
        }
        torch.save(state, out_dir / "last_checkpoint.pth")
        torch.save(state, out_dir / "last.pt")
        key = (float(val_summary["macro_f1"]), float(val_summary["top1"]))
        if key > best_key:
            best_key = key
            torch.save(state, out_dir / "best_checkpoint.pth")
            torch.save(state, out_dir / "best.pt")
            print(f"  saved best: {out_dir / 'best_checkpoint.pth'}")
    state = torch.load(out_dir / "best_checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    test_metrics = evaluate_s136(
        model,
        test_loader,
        device,
        idx_to_class,
        save_predictions_path=out_dir / "predictions_test.csv",
    )
    write_s136_outputs(out_dir, test_metrics, idx_to_class, split="test")
    print(json.dumps(test_metrics["summary"], indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the S136 coupled-factor custom RSCD backbone.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    run_train(args.config, device_override=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

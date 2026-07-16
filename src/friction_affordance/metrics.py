from __future__ import annotations

import numpy as np
import torch

from friction_affordance.ontology import TASKS


@torch.no_grad()
def batch_metrics(outputs, batch) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task in TASKS:
        mask = batch["masks"][task]
        if mask.any():
            pred = outputs["logits"][task].argmax(dim=1)
            acc = (pred[mask] == batch["labels"][task][mask]).float().mean()
            metrics[f"acc_{task}"] = float(acc.cpu())
    mu_mask = batch["mu_mask"]
    if mu_mask.any():
        pred_int = outputs["mu_interval"][mu_mask]
        tgt = batch["mu_interval"][mu_mask]
        covers = (pred_int[:, 0] <= tgt[:, 0]) & (pred_int[:, 1] >= tgt[:, 1])
        widths = pred_int[:, 1] - pred_int[:, 0]
        metrics["mu_interval_coverage"] = float(covers.float().mean().cpu())
        metrics["mu_interval_width"] = float(widths.mean().cpu())
        metrics["mu_mean_mae_to_interval_mid"] = float(
            torch.abs(outputs["mu_mean"][mu_mask] - tgt.mean(dim=1)).mean().cpu()
        )
    if "friction_set" in outputs:
        metrics["state_entropy"] = float(outputs["friction_set"]["state_entropy"].mean().cpu())
    if "mu_interval_parametric" in outputs:
        raw_width = outputs["mu_interval_parametric"][:, 1] - outputs["mu_interval_parametric"][:, 0]
        metrics["mu_parametric_width"] = float(raw_width.mean().cpu())
    return metrics


def average_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({k for item in items for k in item})
    out = {}
    for key in keys:
        vals = [item[key] for item in items if key in item and np.isfinite(item[key])]
        if vals:
            out[key] = float(np.mean(vals))
    return out

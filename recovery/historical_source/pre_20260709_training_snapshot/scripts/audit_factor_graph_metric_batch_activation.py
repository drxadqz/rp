from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from friction_affordance.c3_experiment import build_class_map, load_config
from friction_affordance.rscd_factors import FACTOR_AXES, FACTOR_LABELS, build_rscd_factor_spec, canonical_class_label


def _axis_anchor_count(axis_target: np.ndarray, valid: np.ndarray) -> int:
    count = 0
    for idx, value in enumerate(axis_target.tolist()):
        if not bool(valid[idx]):
            continue
        positive = valid & (axis_target == int(value))
        positive[idx] = False
        if bool(positive.any()):
            count += 1
    return int(count)


def _batch_stats(labels: np.ndarray, class_to_factor: np.ndarray, *, supervise_none: bool) -> dict[str, float]:
    factors = class_to_factor[labels]
    valid_all = np.all(factors >= 0, axis=1)
    out: dict[str, float] = {
        "valid_all": float(valid_all.sum()),
        "coupling_anchors": 0.0,
        "neighbor_pairs": 0.0,
        "roughness_neighbor_pairs": 0.0,
        "wet_concrete_neighbor_pairs": 0.0,
    }
    for axis_idx, axis in enumerate(FACTOR_AXES):
        axis_target = factors[:, axis_idx]
        valid = valid_all & (axis_target >= 0)
        if not supervise_none and axis in {"material", "roughness"}:
            valid = valid & (axis_target != 0)
        anchors = _axis_anchor_count(axis_target, valid)
        out[f"{axis}_anchors"] = float(anchors)

    coupling_anchors = 0
    for idx, value in enumerate(labels.tolist()):
        if not bool(valid_all[idx]):
            continue
        positive = valid_all & (labels == int(value))
        positive[idx] = False
        if bool(positive.any()):
            coupling_anchors += 1
    out["coupling_anchors"] = float(coupling_anchors)

    diff = factors[:, None, :] != factors[None, :, :]
    one_axis_diff = (diff.sum(axis=2) == 1) & valid_all[:, None] & valid_all[None, :]
    np.fill_diagonal(one_axis_diff, False)
    out["neighbor_pairs"] = float(one_axis_diff.sum())
    changed_axis = diff.astype(np.int64).argmax(axis=2)
    rough_axis_idx = FACTOR_AXES.index("roughness")
    rough_mask = one_axis_diff & (changed_axis == rough_axis_idx)
    out["roughness_neighbor_pairs"] = float(rough_mask.sum())

    wet_idx = FACTOR_LABELS["friction"].index("wet")
    water_idx = FACTOR_LABELS["friction"].index("water")
    concrete_idx = FACTOR_LABELS["material"].index("concrete")
    friction = factors[:, 0]
    material = factors[:, 1]
    wc_sample = ((friction == wet_idx) | (friction == water_idx)) & (material == concrete_idx)
    wc_pair = one_axis_diff & wc_sample[:, None] & wc_sample[None, :]
    out["wet_concrete_neighbor_pairs"] = float(wc_pair.sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-batches", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    manifests = [
        Path(data_cfg["train_manifest"]),
        Path(data_cfg["val_manifest"]),
        Path(data_cfg["test_manifest"]),
    ]
    class_to_idx = build_class_map(manifests)
    spec = build_rscd_factor_spec(class_to_idx)
    idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
    train_df = pd.read_csv(Path(data_cfg["train_manifest"]), usecols=["class_label"], dtype=str, low_memory=False)
    train_labels = train_df["class_label"].map(canonical_class_label)
    train_counts = train_labels.value_counts().to_dict()
    class_indices = np.array([class_to_idx[name] for name in sorted(class_to_idx)], dtype=np.int64)
    if bool(cfg.get("train", {}).get("balanced_sampling", False)):
        # WeightedRandomSampler uses inverse class frequency per sample, so the
        # induced class distribution is approximately uniform over present classes.
        probs = np.ones_like(class_indices, dtype=np.float64)
        probs /= probs.sum()
        sampler_mode = "balanced_class_uniform"
    else:
        counts = np.array([float(train_counts.get(name, 0.0)) for name in sorted(class_to_idx)], dtype=np.float64)
        probs = counts / counts.sum()
        sampler_mode = "manifest_frequency"

    batch_size = int(cfg.get("train", {}).get("batch_size", 8))
    supervise_none = bool(cfg.get("loss", {}).get("factor_graph_metric_supervise_none", False))
    rng = np.random.default_rng(int(args.seed))
    class_to_factor = spec.class_to_factor.numpy()
    accum: dict[str, list[float]] = {}
    active: dict[str, int] = {}
    for _ in range(int(args.num_batches)):
        labels = rng.choice(class_indices, size=batch_size, replace=True, p=probs)
        stats = _batch_stats(labels, class_to_factor, supervise_none=supervise_none)
        for key, value in stats.items():
            accum.setdefault(key, []).append(float(value))
            if value > 0:
                active[key] = active.get(key, 0) + 1

    summary: dict[str, Any] = {
        "config": str(args.config),
        "sampler_mode": sampler_mode,
        "num_classes": int(len(class_to_idx)),
        "batch_size": int(batch_size),
        "num_batches": int(args.num_batches),
        "supervise_none": bool(supervise_none),
        "class_order": [idx_to_class[int(idx)] for idx in sorted(idx_to_class)],
        "metrics": {},
    }
    for key, values in sorted(accum.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary["metrics"][key] = {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "active_batch_rate": float(active.get(key, 0) / max(int(args.num_batches), 1)),
        }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Factor Graph Metric Batch Activation Audit",
            "",
            f"Config: `{args.config}`",
            f"Sampler mode: `{sampler_mode}`",
            f"Batch size: {batch_size}",
            f"Simulated batches: {args.num_batches}",
            "",
            "| signal | mean count | p50 | p90 | active batch rate |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, metric in sorted(summary["metrics"].items()):
            lines.append(
                f"| {key} | {metric['mean']:.3f} | {metric['p50']:.1f} | {metric['p90']:.1f} | {100.0 * metric['active_batch_rate']:.2f}% |"
            )
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

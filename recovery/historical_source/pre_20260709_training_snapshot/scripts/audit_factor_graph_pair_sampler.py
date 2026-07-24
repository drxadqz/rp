from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from friction_affordance.c3_experiment import (
    RSCDSurfaceDataset,
    RSCDFactorGraphPairBatchSampler,
    build_class_map,
    load_config,
)
from friction_affordance.rscd_factors import FACTOR_AXES, FACTOR_LABELS, build_rscd_factor_spec
from friction_affordance.transforms import build_transforms


def _batch_stats(labels: np.ndarray, class_to_factor: np.ndarray) -> dict[str, float]:
    factors = class_to_factor[labels]
    valid_all = np.all(factors >= 0, axis=1)
    diff = factors[:, None, :] != factors[None, :, :]
    one_axis_diff = (diff.sum(axis=2) == 1) & valid_all[:, None] & valid_all[None, :]
    np.fill_diagonal(one_axis_diff, False)
    changed_axis = diff.astype(np.int64).argmax(axis=2)
    rough_axis_idx = FACTOR_AXES.index("roughness")
    friction_axis_idx = FACTOR_AXES.index("friction")

    wet_idx = FACTOR_LABELS["friction"].index("wet")
    water_idx = FACTOR_LABELS["friction"].index("water")
    concrete_idx = FACTOR_LABELS["material"].index("concrete")
    friction = factors[:, 0]
    material = factors[:, 1]
    wc_sample = ((friction == wet_idx) | (friction == water_idx)) & (material == concrete_idx)
    wc_pair = one_axis_diff & wc_sample[:, None] & wc_sample[None, :]
    wet_water_pair = wc_pair & (changed_axis == friction_axis_idx)

    coupling_anchors = 0
    for idx, value in enumerate(labels.tolist()):
        if not bool(valid_all[idx]):
            continue
        positive = valid_all & (labels == int(value))
        positive[idx] = False
        if bool(positive.any()):
            coupling_anchors += 1

    return {
        "coupling_anchors": float(coupling_anchors),
        "neighbor_pairs": float(one_axis_diff.sum()),
        "roughness_neighbor_pairs": float((one_axis_diff & (changed_axis == rough_axis_idx)).sum()),
        "wet_concrete_neighbor_pairs": float(wc_pair.sum()),
        "wet_water_concrete_pairs": float(wet_water_pair.sum()),
    }


def _summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(rows[0]) if rows else []
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "active_batch_rate": float((arr > 0).mean()),
        }
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
    manifests = [
        Path(cfg["data"]["train_manifest"]),
        Path(cfg["data"]["val_manifest"]),
        Path(cfg["data"]["test_manifest"]),
    ]
    class_to_idx = build_class_map(manifests)
    spec = build_rscd_factor_spec(class_to_idx)
    image_size = int(cfg["data"].get("image_size", 192))
    train_ds = RSCDSurfaceDataset(
        Path(cfg["data"]["train_manifest"]),
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size, train=False, aug_cfg={"resize_mode": str(cfg["data"].get("eval_resize_mode", "letterbox"))}),
        max_samples=cfg["train"].get("max_train_samples"),
        max_samples_per_class=cfg["train"].get("max_train_samples_per_class"),
        seed=int(cfg.get("seed", 79)),
    )
    batch_size = int(cfg["train"].get("batch_size", 8))
    num_samples = int(cfg["train"].get("samples_per_epoch", 0)) or len(train_ds)
    rng = np.random.default_rng(int(args.seed))
    class_names = sorted(class_to_idx)
    class_indices = np.array([class_to_idx[name] for name in class_names], dtype=np.int64)
    uniform_probs = np.ones_like(class_indices, dtype=np.float64) / float(len(class_indices))
    factor = spec.class_to_factor.numpy()

    balanced_rows = []
    for _ in range(int(args.num_batches)):
        labels = rng.choice(class_indices, size=batch_size, replace=True, p=uniform_probs)
        balanced_rows.append(_batch_stats(labels, factor))

    sampler = RSCDFactorGraphPairBatchSampler(
        train_ds,
        class_to_idx=class_to_idx,
        batch_size=batch_size,
        num_samples=batch_size * int(args.num_batches),
        seed=int(args.seed),
        pair_slots=int(cfg["train"].get("factor_graph_pair_sampling_pair_slots", 2)),
        positive_slots=int(cfg["train"].get("factor_graph_pair_sampling_positive_slots", 1)),
        wet_concrete_focus_scale=float(cfg["train"].get("factor_graph_pair_sampling_wet_concrete_focus_scale", 3.0)),
        roughness_focus_scale=float(cfg["train"].get("factor_graph_pair_sampling_roughness_focus_scale", 1.5)),
        wet_water_focus_scale=float(cfg["train"].get("factor_graph_pair_sampling_wet_water_focus_scale", 1.5)),
    )
    row_to_label = train_ds.df["class_label_canonical"].map(lambda x: class_to_idx[str(x)]).to_numpy(dtype=np.int64)
    graph_rows = []
    for count, batch in enumerate(sampler):
        if count >= int(args.num_batches):
            break
        labels = row_to_label[np.asarray(batch, dtype=np.int64)]
        graph_rows.append(_batch_stats(labels, factor))

    summary: dict[str, Any] = {
        "config": str(args.config),
        "num_batches": int(args.num_batches),
        "batch_size": int(batch_size),
        "balanced_class_uniform": _summarize(balanced_rows),
        "factor_graph_pair_sampler": _summarize(graph_rows),
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Factor Graph Pair Sampler Audit",
            "",
            f"Config: `{args.config}`",
            f"Batch size: {batch_size}",
            f"Simulated batches: {args.num_batches}",
            "",
            "| signal | balanced active | graph active | balanced mean | graph mean |",
            "|---|---:|---:|---:|---:|",
        ]
        for key in sorted(summary["balanced_class_uniform"]):
            b = summary["balanced_class_uniform"][key]
            g = summary["factor_graph_pair_sampler"][key]
            lines.append(
                f"| {key} | {100*b['active_batch_rate']:.2f}% | {100*g['active_batch_rate']:.2f}% | {b['mean']:.3f} | {g['mean']:.3f} |"
            )
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

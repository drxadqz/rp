from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from run_rscd_factor_product_projection import load_npz
from run_rscd_topology_logit_calibration import evaluate_payload, metric_bundle, pct

DEFAULT_CACHE = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\calibration_cache_current_best_physics_texture"
)
DEFAULT_SOURCE = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\screen_physics_texture_hardboost025_lr1e5_s36k_e1_seed101_from_best"
)
DEFAULT_OUT = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\posthoc_top2_pair_prior_calibration_current_best"
)


def main() -> None:
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((DEFAULT_SOURCE / "protocol.json").read_text(encoding="utf-8"))
    class_to_idx = {str(k): int(v) for k, v in protocol["class_to_idx"].items()}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    val = load_npz(DEFAULT_CACHE / "val_logits_topology.npz")
    test = load_npz(DEFAULT_CACHE / "test_logits_topology.npz")
    result = run_calibration(val, test, idx_to_class)
    result["protocol"] = {
        "source_run": str(DEFAULT_SOURCE),
        "cache": str(DEFAULT_CACHE),
        "claim_boundary": (
            "Post-hoc top-2 pair-prior calibration. The neural model is frozen; "
            "only a validation-selected pairwise prior is applied when the model "
            "is uncertain between two labels."
        ),
        "formula": (
            "For top-2 pair S={i,j}, z'_c = z_c + lambda * 1[c in S] * "
            "log((n_c(S)+alpha)/(sum_k n_k(S)+alpha|S|)), gated by top-2 margin <= tau."
        ),
    }
    (out_dir / "top2_pair_prior_calibration.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "evaluate_test.json").write_text(
        json.dumps(result["evaluate_test"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = to_markdown(result)
    (out_dir / "top2_pair_prior_calibration.md").write_text(md, encoding="utf-8")
    mirror = Path("reports/paper_protocol_summary/rscd_top2_pair_prior_calibration_current_best.md")
    mirror.write_text(md, encoding="utf-8")
    print(mirror)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)


def top2_pairs(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argpartition(logits, -2, axis=1)[:, -2:]
    sorted_order = np.take_along_axis(order, np.argsort(np.take_along_axis(logits, order, axis=1), axis=1), axis=1)
    low = sorted_order[:, 0]
    high = sorted_order[:, 1]
    margins = logits[np.arange(len(logits)), high] - logits[np.arange(len(logits)), low]
    pairs = np.stack([np.minimum(low, high), np.maximum(low, high)], axis=1)
    return pairs.astype(np.int64), margins.astype(np.float32)


def fit_pair_priors(logits: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, *, alpha: float, min_count: int) -> dict[tuple[int, int], np.ndarray]:
    pairs, _ = top2_pairs(logits)
    counts: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(logits.shape[1], dtype=np.float32))
    seen: dict[tuple[int, int], int] = defaultdict(int)
    for idx in train_idx:
        pair = tuple(int(x) for x in pairs[idx])
        label = int(labels[idx])
        if label not in pair:
            continue
        counts[pair][label] += 1.0
        seen[pair] += 1
    priors = {}
    for pair, count_vec in counts.items():
        if seen[pair] < int(min_count):
            continue
        active = np.asarray(pair, dtype=np.int64)
        pair_counts = count_vec[active] + float(alpha)
        pair_prior = np.log(pair_counts / np.clip(pair_counts.sum(), 1e-8, None)).astype(np.float32)
        bias = np.zeros(logits.shape[1], dtype=np.float32)
        bias[active] = pair_prior
        priors[pair] = bias
    return priors


def apply_pair_priors(logits: np.ndarray, priors: dict[tuple[int, int], np.ndarray], *, lam: float, tau: float) -> np.ndarray:
    out = logits.astype(np.float32).copy()
    pairs, margins = top2_pairs(out)
    for idx, pair_arr in enumerate(pairs):
        if float(margins[idx]) > float(tau):
            continue
        pair = tuple(int(x) for x in pair_arr)
        bias = priors.get(pair)
        if bias is None:
            continue
        out[idx] += float(lam) * bias
    return out


def run_calibration(val: dict[str, np.ndarray], test: dict[str, np.ndarray], idx_to_class: dict[int, str]) -> dict[str, Any]:
    y_val = val["label"].astype(np.int64)
    y_test = test["label"].astype(np.int64)
    val_logits = val["logits"].astype(np.float32)
    test_logits = test["logits"].astype(np.float32)
    base_test_pred = test_logits.argmax(axis=1)
    baseline = metric_bundle(y_test, base_test_pred, idx_to_class)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=20260628)
    train_idx, select_idx = next(splitter.split(val_logits, y_val))
    candidates: list[dict[str, Any]] = []
    for alpha in [0.5, 1.0, 2.0, 4.0]:
        for min_count in [4, 8, 12, 20]:
            priors = fit_pair_priors(val_logits, y_val, train_idx, alpha=alpha, min_count=min_count)
            for lam in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
                for tau in [0.20, 0.35, 0.50, 0.75, 1.00, 1.50]:
                    adjusted = apply_pair_priors(val_logits[select_idx], priors, lam=lam, tau=tau)
                    metrics = metric_bundle(y_val[select_idx], adjusted.argmax(axis=1), idx_to_class)
                    candidates.append(
                        {
                            "alpha": float(alpha),
                            "min_count": int(min_count),
                            "lambda": float(lam),
                            "tau": float(tau),
                            "num_pairs": int(len(priors)),
                            "select": metrics,
                        }
                    )
    candidates.sort(key=selection_key, reverse=True)
    selected = candidates[0]
    priors = fit_pair_priors(
        val_logits,
        y_val,
        np.arange(len(y_val)),
        alpha=float(selected["alpha"]),
        min_count=int(selected["min_count"]),
    )
    adjusted_test = apply_pair_priors(
        test_logits,
        priors,
        lam=float(selected["lambda"]),
        tau=float(selected["tau"]),
    )
    y_pred = adjusted_test.argmax(axis=1)
    calibrated = metric_bundle(y_test, y_pred, idx_to_class)
    return {
        "baseline_fixed_physics_texture": baseline,
        "selected_by_validation": selected,
        "calibrated_test": calibrated,
        "delta_test": {
            key: float(calibrated[key]) - float(baseline[key])
            for key in ["top1", "macro_f1", "wet_water_f1", "water_f1", "ice_f1", "low_friction_f1"]
        },
        "top_validation_candidates": candidates[:20],
        "evaluate_test": evaluate_payload(
            y_test,
            y_pred,
            idx_to_class,
            claim_boundary="Post-hoc top-2 pair-prior calibration; neural weights are unchanged.",
        ),
    }


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = row["select"]
    return (
        float(metrics["macro_f1"]),
        float(metrics["wet_water_f1"]),
        float(metrics["water_f1"]),
        float(metrics["top1"]),
    )


def to_markdown(result: dict[str, Any]) -> str:
    baseline = result["baseline_fixed_physics_texture"]
    calibrated = result["calibrated_test"]
    delta = result["delta_test"]
    selected = result["selected_by_validation"]
    return "\n".join(
        [
            "# RSCD Top-2 Pair-Prior Calibration",
            "",
            "This is a post-hoc graph/decoupling diagnostic. It does not replace the strict single neural model.",
            "",
            "## Selected Rule",
            "",
            f"- alpha: {selected['alpha']}",
            f"- min_count: {selected['min_count']}",
            f"- lambda: {selected['lambda']}",
            f"- tau: {selected['tau']}",
            f"- learned top-2 pairs: {selected['num_pairs']}",
            "",
            "## Test Metrics",
            "",
            "| row | Top-1 | Macro-F1 | wet/water F1 | water F1 | low-friction F1 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| fixed PhysicsTexture | {pct(baseline['top1'])} | {pct(baseline['macro_f1'])} | {pct(baseline['wet_water_f1'])} | {pct(baseline['water_f1'])} | {pct(baseline['low_friction_f1'])} |",
            f"| top-2 pair-prior | {pct(calibrated['top1'])} | {pct(calibrated['macro_f1'])} | {pct(calibrated['wet_water_f1'])} | {pct(calibrated['water_f1'])} | {pct(calibrated['low_friction_f1'])} |",
            f"| delta | {pct(delta['top1'])} | {pct(delta['macro_f1'])} | {pct(delta['wet_water_f1'])} | {pct(delta['water_f1'])} | {pct(delta['low_friction_f1'])} |",
            "",
            "## Decision",
            "",
            "Keep only if it beats the strict model without wet/water regression. Otherwise use the result as evidence that pairwise graph structure is better for diagnosis than for direct correction.",
        ]
    )


if __name__ == "__main__":
    main()

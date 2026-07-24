from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_rscd_topology_logit_calibration import (
    DEFAULT_OUT as DEFAULT_CACHE_DIR,
    DEFAULT_RUN,
    DEFAULT_TEST,
    DEFAULT_VAL,
    collect_or_load,
    evaluate_payload,
    load_protocol,
    metric_bundle,
    pp,
    pct,
    build_model_from_protocol,
)
from run_rscd_surface_classification import build_class_map
from friction_affordance.utils import resolve_device, set_seed


DEFAULT_OUT = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\conditional_topology_calibration_physics_texture"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Conditioned RSCD logit calibration. It keeps the trained PhysicsTexture "
            "network fixed, learns a validation-only topology/logit calibrator, and "
            "only applies residual corrections to physically hard boundary classes."
        )
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(args.source_run / "protocol.json")
    train_args = protocol["args"]
    class_to_idx = {str(k): int(v) for k, v in protocol.get("class_to_idx", {}).items()}
    if not class_to_idx:
        class_to_idx = build_class_map([args.val_manifest, args.test_manifest])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    val_cache = args.cache_dir / "val_logits_topology.npz"
    test_cache = args.cache_dir / "test_logits_topology.npz"
    if val_cache.exists() and test_cache.exists() and not args.force_cache:
        val = load_npz(val_cache)
        test = load_npz(test_cache)
    else:
        import torch

        device = resolve_device(str(args.device))
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        model = build_model_from_protocol(train_args, class_to_idx).to(device)
        checkpoint = args.checkpoint or (args.source_run / "best.pt")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        val = collect_or_load(
            split="val",
            cache_path=val_cache,
            manifest=args.val_manifest,
            class_to_idx=class_to_idx,
            image_size=int(train_args.get("image_size", 192)),
            model=model,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            force_cache=bool(args.force_cache),
            max_samples=train_args.get("max_val_samples"),
            max_samples_per_class=train_args.get("max_val_samples_per_class"),
            seed=int(train_args.get("seed", args.seed)),
        )
        test = collect_or_load(
            split="test",
            cache_path=test_cache,
            manifest=args.test_manifest,
            class_to_idx=class_to_idx,
            image_size=int(train_args.get("image_size", 192)),
            model=model,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            force_cache=bool(args.force_cache),
            max_samples=train_args.get("max_test_samples"),
            max_samples_per_class=train_args.get("max_test_samples_per_class"),
            seed=int(train_args.get("seed", args.seed)),
        )

    result = run_conditioned_calibration(val, test, idx_to_class, seed=int(args.seed))
    result["protocol"] = {
        "source_run": str(args.source_run),
        "cache_dir": str(args.cache_dir),
        "claim_boundary": (
            "Post-hoc conditioned topology/logit calibration. The neural checkpoint is fixed; "
            "only validation data select the feature set, class mask, C, and residual blend. "
            "Report separately from a pure end-to-end single-model result."
        ),
    }
    (args.output_dir / "conditional_topology_calibration.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "evaluate_test.json").write_text(
        json.dumps(result["evaluate_test"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "conditional_topology_calibration.md").write_text(to_markdown(result), encoding="utf-8")
    mirror = Path("reports/paper_protocol_summary/rscd_conditional_topology_calibration.md")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(to_markdown(result), encoding="utf-8")
    print(mirror)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def run_conditioned_calibration(
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    idx_to_class: dict[int, str],
    *,
    seed: int,
) -> dict[str, Any]:
    y_val = val["label"].astype(np.int64)
    y_test = test["label"].astype(np.int64)
    val_logits = val["logits"].astype(np.float32)
    test_logits = test["logits"].astype(np.float32)
    val_topo = val["topology"].astype(np.float32)
    test_topo = test["topology"].astype(np.float32)
    val_probs = softmax_np(val_logits)
    test_probs = softmax_np(test_logits)
    baseline = metric_bundle(y_test, test_probs.argmax(axis=1), idx_to_class)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    fit_idx, select_idx = next(splitter.split(val_logits, y_val))

    feature_sets = build_feature_sets(val_logits, test_logits, val_topo, test_topo)
    class_masks = build_class_masks(idx_to_class)
    candidates: list[dict[str, Any]] = []
    for feature_name, (x_val, _x_test) in feature_sets.items():
        for mask_name, mask in class_masks.items():
            if not mask.any():
                continue
            for c_value in [0.03, 0.10, 0.30, 1.00, 3.00]:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=float(c_value), max_iter=900, solver="lbfgs"),
                )
                model.fit(x_val[fit_idx], y_val[fit_idx])
                cal_select = model.predict_proba(x_val[select_idx])
                for alpha in [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]:
                    select_probs = apply_residual(
                        val_logits[select_idx],
                        val_probs[select_idx],
                        cal_select,
                        class_mask=mask,
                        alpha=float(alpha),
                    )
                    metrics = metric_bundle(y_val[select_idx], select_probs.argmax(axis=1), idx_to_class)
                    candidates.append(
                        {
                            "feature_set": feature_name,
                            "class_mask": mask_name,
                            "c": float(c_value),
                            "alpha": float(alpha),
                            "select": metrics,
                        }
                    )
    candidates.sort(key=selection_key, reverse=True)
    selected = candidates[0]

    selected_mask = class_masks[str(selected["class_mask"])]
    x_val, x_test = feature_sets[str(selected["feature_set"])]
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(selected["c"]), max_iter=1200, solver="lbfgs"),
    )
    final_model.fit(x_val, y_val)
    cal_test = final_model.predict_proba(x_test)
    final_probs = apply_residual(
        test_logits,
        test_probs,
        cal_test,
        class_mask=selected_mask,
        alpha=float(selected["alpha"]),
    )
    y_pred = final_probs.argmax(axis=1)
    calibrated = metric_bundle(y_test, y_pred, idx_to_class)
    return {
        "baseline_fixed_physics_texture": baseline,
        "selected_by_validation": selected,
        "calibrated_test": calibrated,
        "delta_test": {
            key: float(calibrated[key]) - float(baseline[key])
            for key in ["top1", "macro_f1", "wet_water_f1", "water_f1", "ice_f1", "low_friction_f1"]
        },
        "top_validation_candidates": candidates[:15],
        "evaluate_test": evaluate_payload(
            y_test,
            y_pred,
            idx_to_class,
            claim_boundary=(
                "Post-hoc conditioned topology/logit calibration. The trained PhysicsTexture checkpoint "
                "is fixed; residual corrections are applied only to validation-selected hard boundary classes."
            ),
        ),
    }


def build_feature_sets(
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    val_topo: np.ndarray,
    test_topo: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    val_probs = softmax_np(val_logits)
    test_probs = softmax_np(test_logits)
    val_entropy = entropy_np(val_probs)
    test_entropy = entropy_np(test_probs)
    val_margin = top2_margin(val_probs)
    test_margin = top2_margin(test_probs)
    return {
        "logits": (val_logits, test_logits),
        "logits_topology": (
            np.concatenate([val_logits, val_topo], axis=1),
            np.concatenate([test_logits, test_topo], axis=1),
        ),
        "logits_topology_uncertainty": (
            np.concatenate([val_logits, val_topo, val_entropy, val_margin], axis=1),
            np.concatenate([test_logits, test_topo, test_entropy, test_margin], axis=1),
        ),
    }


def build_class_masks(idx_to_class: dict[int, str]) -> dict[str, np.ndarray]:
    names = [idx_to_class[idx] for idx in sorted(idx_to_class)]
    masks: dict[str, np.ndarray] = {}
    masks["all"] = np.ones(len(names), dtype=bool)
    masks["wet_water"] = np.asarray([friction(name) in {"wet", "water"} for name in names], dtype=bool)
    masks["water_only"] = np.asarray([friction(name) == "water" for name in names], dtype=bool)
    masks["wet_water_concrete_asphalt"] = np.asarray(
        [friction(name) in {"wet", "water"} and material(name) in {"concrete", "asphalt"} for name in names],
        dtype=bool,
    )
    masks["hard_boundary"] = np.asarray(
        [
            (
                friction(name) in {"wet", "water"}
                and material(name) in {"concrete", "asphalt"}
                and roughness(name) in {"slight", "severe"}
            )
            or name in {"dry_concrete_slight", "dry_asphalt_severe"}
            for name in names
        ],
        dtype=bool,
    )
    return masks


def apply_residual(
    base_logits: np.ndarray,
    base_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    *,
    class_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    eps = 1e-8
    residual = np.log(np.clip(calibrated_probs, eps, 1.0)) - np.log(np.clip(base_probs, eps, 1.0))
    masked = np.zeros_like(residual, dtype=np.float32)
    masked[:, class_mask] = residual[:, class_mask]
    corrected = base_logits + float(alpha) * masked
    return softmax_np(corrected)


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = row["select"]
    return (
        float(metrics["macro_f1"]),
        float(metrics["top1"]),
        float(metrics["wet_water_f1"]),
        float(metrics["water_f1"]),
    )


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def entropy_np(probs: np.ndarray) -> np.ndarray:
    return (-probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1, keepdims=True).astype(np.float32)


def top2_margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return (top2[:, 1:2] - top2[:, 0:1]).astype(np.float32)


def friction(name: str) -> str:
    name = str(name)
    if name in {"ice", "fresh_snow", "melted_snow"}:
        return name
    return name.split("_")[0]


def material(name: str) -> str:
    parts = str(name).split("_")
    return parts[1] if len(parts) >= 3 else "unknown"


def roughness(name: str) -> str:
    parts = str(name).split("_")
    return "_".join(parts[2:]) if len(parts) >= 3 else "unknown"


def to_markdown(result: dict[str, Any]) -> str:
    baseline = result["baseline_fixed_physics_texture"]
    calibrated = result["calibrated_test"]
    delta = result["delta_test"]
    selected = result["selected_by_validation"]
    lines = [
        "# RSCD Conditional Topology Calibration",
        "",
        result["protocol"]["claim_boundary"],
        "",
        "## Selected By Validation",
        "",
        f"- Feature set: `{selected['feature_set']}`",
        f"- Class mask: `{selected['class_mask']}`",
        f"- Logistic C: `{selected['c']}`",
        f"- Residual alpha: `{selected['alpha']}`",
        "",
        "## Test Result",
        "",
        "| method | Top-1 | Macro-F1 | wet/water F1 | water F1 | ice F1 | low-friction F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            "| fixed PhysicsTexture | "
            f"{pct(baseline['top1'])} | {pct(baseline['macro_f1'])} | {pct(baseline['wet_water_f1'])} | "
            f"{pct(baseline['water_f1'])} | {pct(baseline['ice_f1'])} | {pct(baseline['low_friction_f1'])} |"
        ),
        (
            "| conditioned calibrated | "
            f"{pct(calibrated['top1'])} | {pct(calibrated['macro_f1'])} | {pct(calibrated['wet_water_f1'])} | "
            f"{pct(calibrated['water_f1'])} | {pct(calibrated['ice_f1'])} | {pct(calibrated['low_friction_f1'])} |"
        ),
        (
            "| delta | "
            f"{pp(delta['top1'])} | {pp(delta['macro_f1'])} | {pp(delta['wet_water_f1'])} | "
            f"{pp(delta['water_f1'])} | {pp(delta['ice_f1'])} | {pp(delta['low_friction_f1'])} |"
        ),
        "",
        "## Top Validation Candidates",
        "",
        "| rank | feature set | class mask | C | alpha | select Top-1 | select Macro-F1 | select wet/water F1 | select water F1 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(result["top_validation_candidates"], start=1):
        metrics = row["select"]
        lines.append(
            f"| {idx} | `{row['feature_set']}` | `{row['class_mask']}` | {row['c']:.2f} | {row['alpha']:.2f} | "
            f"{pct(metrics['top1'])} | {pct(metrics['macro_f1'])} | {pct(metrics['wet_water_f1'])} | "
            f"{pct(metrics['water_f1'])} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

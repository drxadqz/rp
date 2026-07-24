from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from run_rscd_topology_logit_calibration import evaluate_payload, metric_bundle, pp, pct
from run_rscd_surface_classification import FACTOR_LABELS, build_class_map, parse_rscd_factors


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
    r"\posthoc_factor_product_projection_current_best"
)
DEFAULT_TRAIN = Path("data/manifests_full/rscd_prepared_train.csv")
DEFAULT_VAL = Path("data/manifests_full/rscd_prepared_val.csv")
DEFAULT_TEST = Path("data/manifests_full/rscd_prepared_test.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc RSCD factor product projection. It keeps the network fixed and "
            "projects class probabilities toward a friction/material/roughness "
            "factor-product geometry selected only on validation data."
        )
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.source_run / "protocol.json").read_text(encoding="utf-8"))
    class_to_idx = {str(k): int(v) for k, v in protocol.get("class_to_idx", {}).items()}
    if not class_to_idx:
        class_to_idx = build_class_map([args.train_manifest, args.val_manifest, args.test_manifest])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    val = load_npz(args.cache_dir / "val_logits_topology.npz")
    test = load_npz(args.cache_dir / "test_logits_topology.npz")
    result = run_projection(val, test, idx_to_class, seed=int(args.seed))
    result["protocol"] = {
        "source_run": str(args.source_run),
        "cache_dir": str(args.cache_dir),
        "claim_boundary": (
            "Post-hoc factor product projection. The neural model is unchanged; "
            "validation data select only the factor-product exponents and class mask. "
            "Report separately from strict single-model training."
        ),
        "formula": (
            "q_c proportional to p_c * P(f_c)^a_f * P(m_c)^a_m * P(u_c)^a_u "
            "* P(f_c,m_c)^a_fm * P(f_c,u_c)^a_fu * P(m_c,u_c)^a_mu."
        ),
    }
    (args.output_dir / "factor_product_projection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "evaluate_test.json").write_text(
        json.dumps(result["evaluate_test"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = to_markdown(result)
    (args.output_dir / "factor_product_projection.md").write_text(md, encoding="utf-8")
    mirror = Path("reports/paper_protocol_summary/rscd_factor_product_projection_current_best.md")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(md, encoding="utf-8")
    print(mirror)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cache: {path}")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def run_projection(
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
    factor_table = build_factor_table(idx_to_class)

    base_val_probs = softmax_np(val_logits)
    base_test_probs = softmax_np(test_logits)
    baseline = metric_bundle(y_test, base_test_probs.argmax(axis=1), idx_to_class)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    _, select_idx = next(splitter.split(val_logits, y_val))
    candidates: list[dict[str, Any]] = []
    masks = build_class_masks(idx_to_class)
    scalar_grid = [-0.20, -0.10, 0.0, 0.05, 0.10, 0.20]
    pair_grid = [0.0, 0.05, 0.10]
    for mask_name, class_mask in masks.items():
        for af, am, au in product(scalar_grid, scalar_grid, scalar_grid):
            if af == am == au == 0.0:
                continue
            select_probs = project_probs(
                base_val_probs[select_idx],
                factor_table,
                class_mask=class_mask,
                exponents=(af, am, au, 0.0, 0.0, 0.0),
            )
            metrics = metric_bundle(y_val[select_idx], select_probs.argmax(axis=1), idx_to_class)
            candidates.append(
                {
                    "mask": mask_name,
                    "exponents": [float(af), float(am), float(au), 0.0, 0.0, 0.0],
                    "select": metrics,
                }
            )
        for shared_pair in pair_grid:
            if shared_pair <= 0:
                continue
            for af, am, au in [(0.05, 0.05, 0.05), (0.10, 0.05, 0.05), (0.05, 0.05, 0.10)]:
                exponents = (af, am, au, shared_pair, shared_pair, shared_pair)
                select_probs = project_probs(
                    base_val_probs[select_idx],
                    factor_table,
                    class_mask=class_mask,
                    exponents=exponents,
                )
                metrics = metric_bundle(y_val[select_idx], select_probs.argmax(axis=1), idx_to_class)
                candidates.append(
                    {
                        "mask": mask_name,
                        "exponents": [float(x) for x in exponents],
                        "select": metrics,
                    }
                )
    candidates.sort(key=selection_key, reverse=True)
    selected = candidates[0]
    test_probs = project_probs(
        base_test_probs,
        factor_table,
        class_mask=masks[str(selected["mask"])],
        exponents=tuple(float(x) for x in selected["exponents"]),
    )
    y_pred = test_probs.argmax(axis=1)
    projected = metric_bundle(y_test, y_pred, idx_to_class)
    return {
        "baseline_fixed_physics_texture": baseline,
        "selected_by_validation": selected,
        "projected_test": projected,
        "delta_test": {
            key: float(projected[key]) - float(baseline[key])
            for key in ["top1", "macro_f1", "wet_water_f1", "water_f1", "ice_f1", "low_friction_f1"]
        },
        "top_validation_candidates": candidates[:20],
        "evaluate_test": evaluate_payload(
            y_test,
            y_pred,
            idx_to_class,
            claim_boundary=(
                "Post-hoc factor-product probability projection. The trained PhysicsTexture "
                "checkpoint is fixed; no neural weights are updated."
            ),
        ),
    }


def build_factor_table(idx_to_class: dict[int, str]) -> dict[str, np.ndarray]:
    n = len(idx_to_class)
    table = {
        "friction": np.full(n, -1, dtype=np.int64),
        "material": np.full(n, -1, dtype=np.int64),
        "unevenness": np.full(n, -1, dtype=np.int64),
    }
    for idx in range(n):
        factors = parse_rscd_factors(idx_to_class[idx])
        for name in table:
            table[name][idx] = int(factors[name])
    return table


def build_class_masks(idx_to_class: dict[int, str]) -> dict[str, np.ndarray]:
    names = [idx_to_class[idx] for idx in sorted(idx_to_class)]
    masks = {
        "all": np.ones(len(names), dtype=bool),
        "core": np.asarray(
            [
                friction(name) in {"dry", "wet", "water"}
                and material(name) in {"asphalt", "concrete"}
                and roughness(name) in {"smooth", "slight", "severe"}
                for name in names
            ],
            dtype=bool,
        ),
        "wet_water": np.asarray([friction(name) in {"wet", "water"} for name in names], dtype=bool),
        "hard_boundary": np.asarray(
            [
                (
                    friction(name) in {"wet", "water"}
                    and material(name) in {"asphalt", "concrete"}
                    and roughness(name) in {"slight", "severe"}
                )
                or name == "dry_concrete_slight"
                for name in names
            ],
            dtype=bool,
        ),
    }
    return masks


def project_probs(
    probs: np.ndarray,
    factor_table: dict[str, np.ndarray],
    *,
    class_mask: np.ndarray,
    exponents: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    eps = 1e-8
    logq = np.log(np.clip(probs, eps, 1.0)).astype(np.float32)
    af, am, au, afm, afu, amu = exponents
    f = factor_table["friction"]
    m = factor_table["material"]
    u = factor_table["unevenness"]
    if af:
        logq += af * class_marginal_log(probs, f)
    if am:
        logq += am * class_marginal_log(probs, m)
    if au:
        logq += au * class_marginal_log(probs, u)
    if afm:
        logq += afm * pair_marginal_log(probs, f, m)
    if afu:
        logq += afu * pair_marginal_log(probs, f, u)
    if amu:
        logq += amu * pair_marginal_log(probs, m, u)
    corrected = np.log(np.clip(probs, eps, 1.0)).astype(np.float32)
    corrected[:, class_mask] = logq[:, class_mask]
    return softmax_np(corrected)


def class_marginal_log(probs: np.ndarray, ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs, dtype=np.float32)
    for value in sorted(int(x) for x in np.unique(ids) if int(x) >= 0):
        mask = ids == value
        mass = probs[:, mask].sum(axis=1, keepdims=True)
        out[:, mask] = np.log(np.clip(mass, 1e-8, 1.0))
    return out


def pair_marginal_log(probs: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs, dtype=np.float32)
    valid_pairs = sorted({(int(a), int(b)) for a, b in zip(ids_a, ids_b) if int(a) >= 0 and int(b) >= 0})
    for a, b in valid_pairs:
        mask = (ids_a == a) & (ids_b == b)
        mass = probs[:, mask].sum(axis=1, keepdims=True)
        out[:, mask] = np.log(np.clip(mass, 1e-8, 1.0))
    return out


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


def friction(name: str) -> str:
    if name in {"ice", "fresh_snow", "melted_snow"}:
        return name
    return str(name).split("_")[0]


def material(name: str) -> str:
    parts = str(name).split("_")
    return parts[1] if len(parts) >= 3 else "unknown"


def roughness(name: str) -> str:
    parts = str(name).split("_")
    return "_".join(parts[2:]) if len(parts) >= 3 else "unknown"


def to_markdown(result: dict[str, Any]) -> str:
    baseline = result["baseline_fixed_physics_texture"]
    projected = result["projected_test"]
    delta = result["delta_test"]
    selected = result["selected_by_validation"]
    lines = [
        "# RSCD Factor Product Projection",
        "",
        result["protocol"]["claim_boundary"],
        "",
        "## Mathematical Form",
        "",
        "`q_c` is proportional to `p_c * P(f_c)^a_f * P(m_c)^a_m * P(u_c)^a_u "
        "* P(f_c,m_c)^a_fm * P(f_c,u_c)^a_fu * P(m_c,u_c)^a_mu`.",
        "",
        "This is a probability-space factor-graph projection over the RSCD composite labels.",
        "",
        "## Selected By Validation",
        "",
        f"- Class mask: `{selected['mask']}`",
        f"- Exponents `[a_f, a_m, a_u, a_fm, a_fu, a_mu]`: `{selected['exponents']}`",
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
            "| factor-product projected | "
            f"{pct(projected['top1'])} | {pct(projected['macro_f1'])} | {pct(projected['wet_water_f1'])} | "
            f"{pct(projected['water_f1'])} | {pct(projected['ice_f1'])} | {pct(projected['low_friction_f1'])} |"
        ),
        (
            "| delta | "
            f"{pp(delta['top1'])} | {pp(delta['macro_f1'])} | {pp(delta['wet_water_f1'])} | "
            f"{pp(delta['water_f1'])} | {pp(delta['ice_f1'])} | {pp(delta['low_friction_f1'])} |"
        ),
        "",
        "## Top Validation Candidates",
        "",
        "| rank | mask | exponents | select Top-1 | select Macro-F1 | select wet/water F1 | select water F1 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(result["top_validation_candidates"], start=1):
        metrics = row["select"]
        lines.append(
            f"| {idx} | `{row['mask']}` | `{row['exponents']}` | "
            f"{pct(metrics['top1'])} | {pct(metrics['macro_f1'])} | "
            f"{pct(metrics['wet_water_f1'])} | {pct(metrics['water_f1'])} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

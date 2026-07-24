from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments" / "paper_protocol"
FAST_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments" / "fast_screen"
FAST_RUN_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\fast_screen")

PAPER_SOURCE_RUNS = [
    "v1_physics_texture",
    "v6_full_faf_fourier",
    "v7_full_faf_fourier_dann",
    "v8_full_faf_fourier_roadprior",
    "v9_full_faf_roadsaw_hard_sampling",
    "v10_full_faf_consistency",
    "v11_full_faf_domain_adapter",
    "v12_full_faf_roi_interval_safety",
    "v13_lean_physics_evidence",
    "v14_lean_road_roi_safety",
    "v15_lean_bottom_square_style_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
]

ROADSAW_STRESS_RUNS = [
    "lodo_roadsaw_full_faf",
    "final_lodo_roadsaw_lean_road_roi_safety",
    "single_roadsaw_full_faf",
    "baseline_single_roadsaw_global_convnext",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["candidates", "roadsaw", "all"],
        default="all",
        help="Which paper-protocol configs to mirror into the fast-screen protocol.",
    )
    parser.add_argument("--out-dir", type=Path, default=FAST_CONFIG_DIR)
    parser.add_argument("--run-root", type=Path, default=FAST_RUN_ROOT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=4800)
    parser.add_argument("--max-train-per-class", type=int, default=300)
    parser.add_argument("--max-val-per-class", type=int, default=80)
    parser.add_argument("--max-test-per-class", type=int, default=120)
    parser.add_argument("--max-val-samples", type=int, default=6000)
    parser.add_argument("--max-test-samples", type=int, default=9000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)

    selected = _selected_runs(args.scope)
    written = []
    for name in selected:
        src = PAPER_CONFIG_DIR / f"{name}.yaml"
        if not src.exists():
            raise FileNotFoundError(
                f"Missing paper config: {src}. Run scripts/make_paper_protocol_configs.py first."
            )
        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        screen_name = f"screen_{name}"
        screen_cfg = _to_fast_screen_config(
            cfg,
            source_run=name,
            screen_name=screen_name,
            run_root=args.run_root,
            epochs=int(args.epochs),
            samples_per_epoch=int(args.samples_per_epoch),
            max_train_per_class=int(args.max_train_per_class),
            max_val_per_class=int(args.max_val_per_class),
            max_test_per_class=int(args.max_test_per_class),
            max_val_samples=int(args.max_val_samples),
            max_test_samples=int(args.max_test_samples),
        )
        out = args.out_dir / f"{screen_name}.yaml"
        out.write_text(yaml.safe_dump(screen_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append({"source_run": name, "screen_run": screen_name, "config": str(out)})
        print(f"wrote: {out}")

    manifest = {
        "protocol": "fast_screen",
        "claim_boundary": (
            "Fast-screen runs are for direction finding only. They must not be used "
            "as final paper evidence or as fair SOTA comparisons."
        ),
        "source_config_dir": str(PAPER_CONFIG_DIR),
        "config_dir": str(args.out_dir),
        "run_root": str(args.run_root),
        "runs": written,
        "budget": {
            "epochs": int(args.epochs),
            "balanced_num_samples_per_epoch": int(args.samples_per_epoch),
            "max_train_samples_per_class": int(args.max_train_per_class),
            "max_val_samples_per_class": int(args.max_val_per_class),
            "max_test_samples_per_class": int(args.max_test_per_class),
            "max_val_samples": int(args.max_val_samples),
            "max_test_samples": int(args.max_test_samples),
        },
    }
    manifest_path = args.out_dir / "fast_screen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote: {manifest_path}")


def _selected_runs(scope: str) -> list[str]:
    if scope == "candidates":
        return list(PAPER_SOURCE_RUNS)
    if scope == "roadsaw":
        return list(ROADSAW_STRESS_RUNS)
    return list(PAPER_SOURCE_RUNS) + list(ROADSAW_STRESS_RUNS)


def _to_fast_screen_config(
    cfg: dict[str, Any],
    *,
    source_run: str,
    screen_name: str,
    run_root: Path,
    epochs: int,
    samples_per_epoch: int,
    max_train_per_class: int,
    max_val_per_class: int,
    max_test_per_class: int,
    max_val_samples: int,
    max_test_samples: int,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["output_dir"] = str(run_root / screen_name)
    out["screen_parent_run"] = source_run
    out["screen_protocol"] = {
        "role": "fast_direction_finding",
        "not_final_claim_evidence": True,
        "formal_follow_up": source_run,
    }
    out["experiment_note"] = (
        "FAST-SCREEN ONLY: small-budget proxy for rapid candidate ranking; "
        "not a final paper result. Formal claims require the corresponding "
        f"full paper-protocol run `{source_run}`. "
        + str(out.get("experiment_note", ""))
    ).strip()

    data = out.setdefault("data", {})
    data["balanced_num_samples_per_epoch"] = max(1, int(samples_per_epoch))
    data["max_train_samples"] = None
    data["max_train_samples_per_class"] = max(1, int(max_train_per_class))
    data["max_train_samples_per_dataset"] = None
    data["max_val_samples"] = max(1, int(max_val_samples))
    data["max_val_samples_per_class"] = max(1, int(max_val_per_class))
    data["max_val_samples_per_dataset"] = None
    data["max_test_samples"] = max(1, int(max_test_samples))
    data["max_test_samples_per_class"] = max(1, int(max_test_per_class))
    data["max_test_samples_per_dataset"] = None
    data["num_workers"] = min(int(data.get("num_workers", 0) or 0), 2)
    data["prefetch_factor"] = min(int(data.get("prefetch_factor", 2) or 2), 2)

    optim = out.setdefault("optim", {})
    optim["epochs"] = max(1, int(epochs))
    optim["early_stop_patience"] = min(int(optim.get("early_stop_patience", 5) or 5), 2)
    optim["early_stop_min_delta"] = float(optim.get("early_stop_min_delta", 0.0005) or 0.0005)
    optim["log_every_steps"] = 50
    # Preserve the full-run microbatch/effective-batch policy for OOM safety.
    optim["amp"] = True
    if int(data.get("batch_size", 16) or 16) <= 8:
        optim["grad_accum_steps"] = max(int(optim.get("grad_accum_steps", 1) or 1), 4)
    else:
        optim["grad_accum_steps"] = max(int(optim.get("grad_accum_steps", 1) or 1), 2)
    return out


if __name__ == "__main__":
    main()

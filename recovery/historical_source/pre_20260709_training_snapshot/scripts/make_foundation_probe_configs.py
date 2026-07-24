from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "experiments" / "foundation_probe"
RUN_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\foundation_probe")
FAF_SOURCE = ROOT / "configs" / "experiments" / "paper_protocol" / "v17_lean_quality_physics_safety.yaml"
BASELINE_SOURCE = (
    ROOT
    / "configs"
    / "experiments"
    / "paper_protocol"
    / "baseline_single_rscd_global_convnext.yaml"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faf = _load_yaml(FAF_SOURCE)
    baseline = _load_yaml(BASELINE_SOURCE)

    configs = {
        "foundation_dinov2_global_probe": _make_global_probe(baseline),
        "foundation_dinov2_quality_faf_probe": _make_faf_probe(faf),
    }
    for name, cfg in configs.items():
        path = OUT_DIR / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"wrote: {path}")


def _make_common(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["seed"] = 79
    out["output_dir"] = str(RUN_ROOT / name)
    data = out.setdefault("data", {})
    # Keep this as a feasibility probe. A formal paper row must use the full
    # matched split/budget after ConvNeXt and FAF evidence is available.
    data["image_size"] = 196
    data["batch_size"] = 2
    data["num_workers"] = 2
    data["prefetch_factor"] = 2
    data["balanced_num_samples_per_epoch"] = 12000
    data["max_train_samples"] = None
    data["max_val_samples"] = None
    data["max_test_samples"] = None

    model = out.setdefault("model", {})
    model["backbone"] = "timm:vit_small_patch14_dinov2"
    model["embedding_dim"] = 384
    model["pretrained"] = True
    model["freeze_backbone"] = False

    optim = out.setdefault("optim", {})
    optim["epochs"] = 8
    optim["lr"] = 5.0e-5
    optim["grad_accum_steps"] = 16
    optim["amp"] = True
    optim["early_stop_patience"] = 3
    optim["log_every_steps"] = 100

    out["screen_parent_run"] = None
    out["not_final_claim_evidence"] = True
    return out


def _make_global_probe(source: dict[str, Any]) -> dict[str, Any]:
    cfg = _make_common(source, "foundation_dinov2_global_probe")
    # Use the same multi-dataset manifests as the FAF probe so the two rows can
    # be compared as a feasibility pair.
    faf_source = _load_yaml(FAF_SOURCE)
    cfg["data"]["train_manifests"] = faf_source["data"]["train_manifests"]
    cfg["data"]["val_manifests"] = faf_source["data"]["val_manifests"]
    cfg["data"]["test_manifests"] = faf_source["data"]["test_manifests"]
    cfg["data"]["balanced_dataset_first"] = True
    cfg["data"]["balanced_group_columns"] = ["dataset", "class_label"]
    model = cfg.setdefault("model", {})
    model.update(
        {
            "use_physics_branch": False,
            "use_friction_set": False,
            "use_evidence_field": False,
            "num_domains": 3,
        }
    )
    cfg["loss"] = _global_loss()
    cfg["experiment_note"] = (
        "Feasibility probe only: DINOv2-small/timm global image-level baseline "
        "on the same public manifests as the quality FAF probe. Not final claim evidence."
    )
    return cfg


def _make_faf_probe(source: dict[str, Any]) -> dict[str, Any]:
    cfg = _make_common(source, "foundation_dinov2_quality_faf_probe")
    cfg["experiment_note"] = (
        "Feasibility probe only: v17 quality-aware lean PhysicsTexture/EvidenceField route "
        "with a timm DINOv2-small backbone. Not final claim evidence."
    )
    return cfg


def _global_loss() -> dict[str, Any]:
    return {
        "task_weight": 1.0,
        "compatibility_weight": 0.0,
        "group_dro_weight": 0.0,
        "group_vrex_weight": 0.0,
        "risk_ordinal_weight": 0.12,
        "interval_weight": 0.06,
        "coverage_weight": 1.0,
        "endpoint_weight": 1.0,
        "target_width_weight": 0.25,
        "width_weight": 0.0,
        "coverage_margin": 0.04,
        "monotonic_weight": 0.04,
        "risk_mu_monotonic_weight": 0.06,
        "domain_weight": 0.0,
        "domain_grl_lambda": 0.0,
        "feature_coral_weight": 0.0,
        "risk_conditional_coral_weight": 0.0,
        "wetness_conditional_coral_weight": 0.0,
        "wetness_ordinal_weight": 0.08,
        "coverage_risk_weight": 0.25,
        "coverage_wetness_weight": 0.20,
        "coverage_snow_weight": 0.15,
        "coverage_weight_max": 1.6,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

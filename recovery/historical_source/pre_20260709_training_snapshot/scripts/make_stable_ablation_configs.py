from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("configs/experiments/ablations_stable"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("D:/NMI_SPWFM_datasets/friction_affordance_outputs"),
    )
    args = parser.parse_args()

    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "global_only": _global_only,
        "friction_set": _friction_set,
        "heterogeneous_dg": _heterogeneous_dg,
        "evidence_field_full": _evidence_field_full,
    }
    for name, fn in variants.items():
        cfg = copy.deepcopy(base)
        cfg["output_dir"] = str(args.output_root / f"ablation_{name}_rtx3050_stable")
        cfg["seed"] = int(base.get("seed", 79))
        cfg = fn(cfg)
        path = args.out_dir / f"ablation_{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(path)


def _global_only(cfg: dict) -> dict:
    model = cfg["model"]
    model["use_physics_branch"] = False
    model["use_friction_set"] = False
    model["use_evidence_field"] = False
    loss = cfg["loss"]
    _zero(loss, "feature_coral_weight", "risk_conditional_coral_weight")
    _zero_evidence_losses(loss)
    loss["task_weight"] = 1.0
    cfg["experiment_note"] = "Ablation: image-level global backbone and task heads only."
    return cfg


def _friction_set(cfg: dict) -> dict:
    model = cfg["model"]
    model["use_physics_branch"] = False
    model["use_friction_set"] = True
    model["friction_set_interval_mix"] = 0.80
    model["use_evidence_field"] = False
    loss = cfg["loss"]
    _zero(loss, "feature_coral_weight", "risk_conditional_coral_weight")
    _zero_evidence_losses(loss)
    loss["task_weight"] = 0.45
    cfg["experiment_note"] = "Ablation: global backbone with FrictionSet interval prior."
    return cfg


def _heterogeneous_dg(cfg: dict) -> dict:
    model = cfg["model"]
    model["use_physics_branch"] = True
    model["use_friction_set"] = True
    model["use_evidence_field"] = False
    loss = cfg["loss"]
    _zero_evidence_losses(loss)
    loss["feature_coral_weight"] = 0.01
    loss["risk_conditional_coral_weight"] = 0.02
    loss["task_weight"] = 0.35
    cfg["experiment_note"] = "Ablation: heterogeneous weak supervision with physics branch and DG losses, no local evidence field."
    return cfg


def _evidence_field_full(cfg: dict) -> dict:
    cfg["experiment_note"] = "Full method: local evidence-field friction affordance learning."
    return cfg


def _zero(items: dict, *keys: str) -> None:
    for key in keys:
        if key in items:
            items[key] = 0.0


def _zero_evidence_losses(loss: dict) -> None:
    for key in list(loss):
        if key.startswith("evidence_"):
            loss[key] = 0.0


if __name__ == "__main__":
    main()

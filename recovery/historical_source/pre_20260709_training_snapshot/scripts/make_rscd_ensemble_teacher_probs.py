from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed
from run_rscd_surface_classification import RSCDSurfaceDataset, SurfaceClassifier, collate


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_OUT = DEFAULT_ROOT / "teacher_probs_ensemble_hflip_consistency_plus_rcd"


def _load_protocol(run_dir: Path) -> dict[str, Any]:
    protocol_path = run_dir / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"Missing protocol file: {protocol_path}")
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def _build_model(run_dir: Path, device: torch.device) -> tuple[nn.Module, dict[str, int], dict[str, Any]]:
    protocol = _load_protocol(run_dir)
    train_args = protocol["args"]
    class_to_idx = {str(k): int(v) for k, v in protocol["class_to_idx"].items()}
    model = SurfaceClassifier(
        backbone=str(train_args.get("backbone", "convnext_tiny")),
        embedding_dim=int(train_args.get("embedding_dim", 768)),
        num_classes=len(class_to_idx),
        pretrained=False,
        dropout=float(train_args.get("dropout", 0.2)),
        use_physics_branch=bool(train_args.get("use_physics_branch", False)),
        physics_dim=int(train_args.get("physics_dim", 96)),
        physics_quality_cues=bool(train_args.get("physics_quality_cues", False)),
        physics_quality_region_cues=bool(train_args.get("physics_quality_region_cues", True)),
        use_directional_texture_branch=bool(train_args.get("use_directional_texture_branch", False)),
        directional_texture_dim=int(train_args.get("directional_texture_dim", 64)),
        use_wavelet_texture_branch=bool(train_args.get("use_wavelet_texture_branch", False)),
        wavelet_texture_dim=int(train_args.get("wavelet_texture_dim", 64)),
        use_retinex_texture_branch=bool(train_args.get("use_retinex_texture_branch", False)),
        retinex_texture_dim=int(train_args.get("retinex_texture_dim", 48)),
        retinex_region_cues=bool(train_args.get("retinex_region_cues", True)),
        use_physics_attention_branch=bool(train_args.get("use_physics_attention_branch", False)),
        physics_attention_dim=int(train_args.get("physics_attention_dim", 64)),
        use_semantic_physics_attention_branch=bool(train_args.get("use_semantic_physics_attention_branch", False)),
        semantic_physics_attention_dim=int(train_args.get("semantic_physics_attention_dim", 64)),
        use_local_physics_field_branch=bool(train_args.get("use_local_physics_field_branch", False)),
        local_physics_field_dim=int(train_args.get("local_physics_field_dim", 64)),
        local_physics_field_scale=float(train_args.get("local_physics_field_scale", 0.10)),
        use_topological_texture_branch=bool(train_args.get("use_topological_texture_branch", False)),
        topological_texture_dim=int(train_args.get("topological_texture_dim", 48)),
        use_anti_human_texture_branch=bool(train_args.get("use_anti_human_texture_branch", False)),
        anti_human_texture_dim=int(train_args.get("anti_human_texture_dim", 64)),
        use_texture_gate=bool(train_args.get("use_texture_gate", False)),
        use_texture_residual_adapter=bool(train_args.get("use_texture_residual_adapter", False)),
        texture_residual_scale=float(train_args.get("texture_residual_scale", 0.25)),
        use_texture_film=bool(train_args.get("use_texture_film", False)),
        texture_film_scale=float(train_args.get("texture_film_scale", 0.20)),
        use_material_conditioned_texture_gate=bool(train_args.get("use_material_conditioned_texture_gate", False)),
        material_conditioned_gate_scale=float(train_args.get("material_conditioned_gate_scale", 0.25)),
        use_artifact_aware_texture_gate=bool(train_args.get("use_artifact_aware_texture_gate", False)),
        artifact_aware_gate_scale=float(train_args.get("artifact_aware_gate_scale", 0.20)),
        use_factor_logit_adjustment=bool(train_args.get("use_factor_logit_adjustment", False)),
        factor_logit_adjustment_scale=float(train_args.get("factor_logit_adjustment_scale", 0.30)),
        use_factorized_low_rank_head=bool(train_args.get("use_factorized_low_rank_head", False)),
        factorized_rank=int(train_args.get("factorized_rank", 64)),
        factorized_scale=float(train_args.get("factorized_scale", 0.25)),
        factorized_normalize=bool(train_args.get("factorized_normalize", True)),
        factorized_zero_init=bool(train_args.get("factorized_zero_init", False)),
        use_safe_factorized_low_rank_head=bool(train_args.get("use_safe_factorized_low_rank_head", False)),
        safe_factorized_rank=int(train_args.get("safe_factorized_rank", 64)),
        safe_factorized_scale=float(train_args.get("safe_factorized_scale", 0.25)),
        safe_factorized_gate_threshold=float(train_args.get("safe_factorized_gate_threshold", 0.55)),
        safe_factorized_gate_temperature=float(train_args.get("safe_factorized_gate_temperature", 8.0)),
        safe_factorized_protected_negative_limit=float(train_args.get("safe_factorized_protected_negative_limit", 0.0)),
        use_factor_interaction_low_rank_head=bool(train_args.get("use_factor_interaction_low_rank_head", False)),
        factor_interaction_rank=int(train_args.get("factor_interaction_rank", 64)),
        factor_interaction_scale=float(train_args.get("factor_interaction_scale", 0.20)),
        factor_interaction_gate_threshold=float(train_args.get("factor_interaction_gate_threshold", 0.55)),
        factor_interaction_gate_temperature=float(train_args.get("factor_interaction_gate_temperature", 8.0)),
        factor_interaction_protected_negative_limit=float(
            train_args.get("factor_interaction_protected_negative_limit", 0.0)
        ),
        use_water_evidence_logit_gate=bool(train_args.get("use_water_evidence_logit_gate", False)),
        water_evidence_gate_scale=float(train_args.get("water_evidence_gate_scale", 0.20)),
        water_evidence_gate_zero_init=bool(train_args.get("water_evidence_gate_zero_init", True)),
        use_coupled_optical_roughness_residual=bool(train_args.get("use_coupled_optical_roughness_residual", False)),
        coupled_residual_hidden_dim=int(train_args.get("coupled_residual_hidden_dim", 96)),
        coupled_residual_scale=float(train_args.get("coupled_residual_scale", 0.12)),
        coupled_residual_gate_threshold=float(train_args.get("coupled_residual_gate_threshold", 0.35)),
        coupled_residual_gate_temperature=float(train_args.get("coupled_residual_gate_temperature", 8.0)),
        coupled_residual_protected_negative_limit=float(train_args.get("coupled_residual_protected_negative_limit", 0.0)),
        use_roughness_neighbor_residual=bool(train_args.get("use_roughness_neighbor_residual", False)),
        roughness_neighbor_hidden_dim=int(train_args.get("roughness_neighbor_hidden_dim", 96)),
        roughness_neighbor_scale=float(train_args.get("roughness_neighbor_scale", 0.10)),
        roughness_neighbor_gate_threshold=float(train_args.get("roughness_neighbor_gate_threshold", 0.42)),
        roughness_neighbor_gate_temperature=float(train_args.get("roughness_neighbor_gate_temperature", 10.0)),
        roughness_neighbor_protected_negative_limit=float(train_args.get("roughness_neighbor_protected_negative_limit", 0.0)),
        roughness_neighbor_gate_floor=float(train_args.get("roughness_neighbor_gate_floor", 0.15)),
        use_relation_signed_graph_expert=bool(train_args.get("use_relation_signed_graph_expert", False)),
        relation_signed_hidden_dim=int(train_args.get("relation_signed_hidden_dim", 96)),
        relation_signed_scale=float(train_args.get("relation_signed_scale", 0.06)),
        relation_signed_gate_threshold=float(train_args.get("relation_signed_gate_threshold", 0.35)),
        relation_signed_gate_temperature=float(train_args.get("relation_signed_gate_temperature", 12.0)),
        relation_signed_protected_negative_limit=float(
            train_args.get("relation_signed_protected_negative_limit", 0.0)
        ),
        relation_signed_neighbor_gate_floor=float(train_args.get("relation_signed_neighbor_gate_floor", 0.0)),
        use_heterophilic_logit_boundary_expert=bool(
            train_args.get("use_heterophilic_logit_boundary_expert", False)
        ),
        heterophilic_boundary_scale=float(train_args.get("heterophilic_boundary_scale", 0.10)),
        heterophilic_boundary_gate_threshold=float(train_args.get("heterophilic_boundary_gate_threshold", 0.0)),
        heterophilic_boundary_gate_temperature=float(train_args.get("heterophilic_boundary_gate_temperature", 8.0)),
        heterophilic_boundary_protected_negative_limit=float(
            train_args.get("heterophilic_boundary_protected_negative_limit", 0.0)
        ),
        use_heterophilic_feature_boundary_expert=bool(
            train_args.get("use_heterophilic_feature_boundary_expert", False)
        ),
        heterophilic_feature_boundary_hidden_dim=int(train_args.get("heterophilic_feature_boundary_hidden_dim", 96)),
        heterophilic_feature_boundary_scale=float(train_args.get("heterophilic_feature_boundary_scale", 0.08)),
        heterophilic_feature_boundary_gate_threshold=float(
            train_args.get("heterophilic_feature_boundary_gate_threshold", 0.12)
        ),
        heterophilic_feature_boundary_gate_temperature=float(
            train_args.get("heterophilic_feature_boundary_gate_temperature", 10.0)
        ),
        heterophilic_feature_boundary_protected_negative_limit=float(
            train_args.get("heterophilic_feature_boundary_protected_negative_limit", 0.0)
        ),
        use_protected_heterophilic_factor_boundary_field=bool(
            train_args.get("use_protected_heterophilic_factor_boundary_field", False)
        ),
        protected_factor_boundary_hidden_dim=int(train_args.get("protected_factor_boundary_hidden_dim", 64)),
        protected_factor_boundary_pair_dim=int(train_args.get("protected_factor_boundary_pair_dim", 12)),
        protected_factor_boundary_relation_dim=int(train_args.get("protected_factor_boundary_relation_dim", 8)),
        protected_factor_boundary_scale=float(train_args.get("protected_factor_boundary_scale", 0.08)),
        protected_factor_boundary_gate_threshold=float(
            train_args.get("protected_factor_boundary_gate_threshold", 0.10)
        ),
        protected_factor_boundary_gate_temperature=float(
            train_args.get("protected_factor_boundary_gate_temperature", 10.0)
        ),
        protected_factor_boundary_protected_negative_limit=float(
            train_args.get("protected_factor_boundary_protected_negative_limit", 0.0)
        ),
        use_relation_specific_hard_edge_refiner=bool(train_args.get("use_relation_specific_hard_edge_refiner", False)),
        relation_specific_refiner_hidden_dim=int(train_args.get("relation_specific_refiner_hidden_dim", 64)),
        relation_specific_refiner_pair_dim=int(train_args.get("relation_specific_refiner_pair_dim", 12)),
        relation_specific_refiner_scale=float(train_args.get("relation_specific_refiner_scale", 0.08)),
        relation_specific_refiner_gate_threshold=float(train_args.get("relation_specific_refiner_gate_threshold", 0.10)),
        relation_specific_refiner_gate_temperature=float(
            train_args.get("relation_specific_refiner_gate_temperature", 10.0)
        ),
        relation_specific_refiner_protected_negative_limit=float(
            train_args.get("relation_specific_refiner_protected_negative_limit", 0.0)
        ),
        class_to_idx=class_to_idx,
        use_factor_aux=float(train_args.get("factor_aux_weight", 0.0)) > 0.0,
        use_local_physics_factor_aux=float(train_args.get("local_physics_factor_aux_weight", 0.0)) > 0.0,
        use_physics_aux=float(train_args.get("physics_aux_weight", 0.0)) > 0.0,
        use_backbone_aux=float(train_args.get("backbone_aux_weight", 0.0)) > 0.0,
    ).to(device)
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    weights = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(weights)
    model.eval()
    return model, class_to_idx, train_args


def _model_logits(model: nn.Module, image: torch.Tensor, tta: str) -> torch.Tensor:
    logits = model(image)
    if tta == "hflip":
        logits = 0.5 * (logits + model(torch.flip(image, dims=[3])))
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export train-subset soft targets from an hflip-TTA logits ensemble. "
            "The output can be used by run_rscd_surface_classification.py via "
            "--distill-teacher-probs."
        )
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_train.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-train-samples-per-class", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    args = parser.parse_args()

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    models: list[nn.Module] = []
    reference_class_to_idx: dict[str, int] | None = None
    image_size: int | None = None
    protocol_rows: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        model, class_to_idx, train_args = _build_model(run_dir, device)
        if reference_class_to_idx is None:
            reference_class_to_idx = class_to_idx
            image_size = int(train_args.get("image_size", 192))
        elif class_to_idx != reference_class_to_idx:
            raise ValueError(f"Class map mismatch in {run_dir}")
        elif int(train_args.get("image_size", image_size)) != image_size:
            raise ValueError(f"Image-size mismatch in {run_dir}")
        models.append(model)
        protocol_rows.append({"run_dir": str(run_dir), "args": train_args})

    if reference_class_to_idx is None or image_size is None:
        raise ValueError("At least one run directory is required.")

    transform = build_transforms(int(image_size), train=False, aug_cfg={"resize_mode": "letterbox"})
    max_per_class = int(args.max_train_samples_per_class)
    dataset = RSCDSurfaceDataset(
        args.train_manifest,
        class_to_idx=reference_class_to_idx,
        transform=transform,
        max_samples_per_class=max_per_class if max_per_class > 0 else None,
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    probs_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    temperature = max(float(args.temperature), 1e-3)
    with torch.no_grad():
        for batch in tqdm(loader, desc="export-ensemble-teacher", leave=False, ascii=True):
            image = batch["image"].to(device, non_blocking=True)
            logits_sum = None
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                for model in models:
                    logits = _model_logits(model, image, str(args.tta))
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                logits_mean = logits_sum / max(len(models), 1)
                probs = torch.softmax(logits_mean / temperature, dim=1)
            probs_rows.append(probs.detach().float().cpu().numpy())
            labels.append(batch["label"].detach().cpu().numpy().astype(np.int64))
            paths.extend([str(x) for x in batch["image_path"]])

    subset_tag = "all" if max_per_class <= 0 else f"mpc{max_per_class}"
    temp_tag = f"t{temperature:g}".replace(".", "p")
    out_npz = args.output_dir / f"teacher_probs_ensemble_{args.tta}_{subset_tag}_{temp_tag}_seed{int(args.seed)}.npz"
    np.savez_compressed(
        out_npz,
        image_path=np.asarray(paths, dtype=object),
        label=np.concatenate(labels, axis=0),
        probs=np.concatenate(probs_rows, axis=0).astype(np.float32),
    )
    meta = {
        "claim_boundary": (
            "Training-subset soft targets from a fixed hflip-TTA logits ensemble. "
            "This file is used only for distillation; strict test inference remains "
            "a single model unless separately reported as an ensemble diagnostic."
        ),
        "run_dirs": [str(p) for p in args.run_dirs],
        "train_manifest": str(args.train_manifest),
        "rows": int(len(paths)),
        "max_train_samples_per_class": max_per_class if max_per_class > 0 else None,
        "seed": int(args.seed),
        "temperature": float(temperature),
        "tta": str(args.tta),
        "output": str(out_npz),
        "protocols": protocol_rows,
    }
    out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_npz)


if __name__ == "__main__":
    main()

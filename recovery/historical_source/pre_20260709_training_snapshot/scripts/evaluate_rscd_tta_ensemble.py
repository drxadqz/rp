from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed
from run_rscd_surface_classification import RSCDSurfaceDataset, SurfaceClassifier, confusion_rows


DEFAULT_RUN_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_OUT = DEFAULT_RUN_ROOT / "tta_ensemble_physics_texture_formal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more trained RSCD surface classifiers with optional "
            "horizontal-flip TTA and logits averaging. This is an inference-only "
            "protocol; it does not change the training result."
        )
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Training output directories containing protocol.json and best.pt.",
    )
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument(
        "--tta",
        choices=["none", "hflip"],
        default="hflip",
        help="Use original image only, or average original plus horizontal flip.",
    )
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-test-samples-per-class", type=int, default=None)
    return parser.parse_args()


def _load_protocol(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing protocol file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model(run_dir: Path, device: torch.device) -> tuple[nn.Module, dict[str, int], dict[str, Any]]:
    protocol = _load_protocol(run_dir)
    args = protocol["args"]
    class_to_idx = {str(k): int(v) for k, v in protocol["class_to_idx"].items()}
    model = SurfaceClassifier(
        backbone=str(args.get("backbone", "convnext_tiny")),
        embedding_dim=int(args.get("embedding_dim", 768)),
        num_classes=len(class_to_idx),
        pretrained=False,
        dropout=float(args.get("dropout", 0.2)),
        use_physics_branch=bool(args.get("use_physics_branch", False)),
        physics_dim=int(args.get("physics_dim", 96)),
        physics_quality_cues=bool(args.get("physics_quality_cues", False)),
        physics_quality_region_cues=bool(args.get("physics_quality_region_cues", True)),
        use_directional_texture_branch=bool(args.get("use_directional_texture_branch", False)),
        directional_texture_dim=int(args.get("directional_texture_dim", 64)),
        use_wavelet_texture_branch=bool(args.get("use_wavelet_texture_branch", False)),
        wavelet_texture_dim=int(args.get("wavelet_texture_dim", 64)),
        use_retinex_texture_branch=bool(args.get("use_retinex_texture_branch", False)),
        retinex_texture_dim=int(args.get("retinex_texture_dim", 48)),
        retinex_region_cues=bool(args.get("retinex_region_cues", True)),
        use_physics_attention_branch=bool(args.get("use_physics_attention_branch", False)),
        physics_attention_dim=int(args.get("physics_attention_dim", 64)),
        use_semantic_physics_attention_branch=bool(args.get("use_semantic_physics_attention_branch", False)),
        semantic_physics_attention_dim=int(args.get("semantic_physics_attention_dim", 64)),
        use_local_physics_field_branch=bool(args.get("use_local_physics_field_branch", False)),
        local_physics_field_dim=int(args.get("local_physics_field_dim", 64)),
        local_physics_field_scale=float(args.get("local_physics_field_scale", 0.10)),
        use_topological_texture_branch=bool(args.get("use_topological_texture_branch", False)),
        topological_texture_dim=int(args.get("topological_texture_dim", 48)),
        use_anti_human_texture_branch=bool(args.get("use_anti_human_texture_branch", False)),
        anti_human_texture_dim=int(args.get("anti_human_texture_dim", 64)),
        use_texture_gate=bool(args.get("use_texture_gate", False)),
        use_texture_residual_adapter=bool(args.get("use_texture_residual_adapter", False)),
        texture_residual_scale=float(args.get("texture_residual_scale", 0.25)),
        use_texture_film=bool(args.get("use_texture_film", False)),
        texture_film_scale=float(args.get("texture_film_scale", 0.20)),
        use_material_conditioned_texture_gate=bool(args.get("use_material_conditioned_texture_gate", False)),
        material_conditioned_gate_scale=float(args.get("material_conditioned_gate_scale", 0.25)),
        use_artifact_aware_texture_gate=bool(args.get("use_artifact_aware_texture_gate", False)),
        artifact_aware_gate_scale=float(args.get("artifact_aware_gate_scale", 0.20)),
        use_factor_logit_adjustment=bool(args.get("use_factor_logit_adjustment", False)),
        factor_logit_adjustment_scale=float(args.get("factor_logit_adjustment_scale", 0.30)),
        use_factorized_low_rank_head=bool(args.get("use_factorized_low_rank_head", False)),
        factorized_rank=int(args.get("factorized_rank", 64)),
        factorized_scale=float(args.get("factorized_scale", 0.25)),
        factorized_normalize=bool(args.get("factorized_normalize", True)),
        factorized_zero_init=bool(args.get("factorized_zero_init", False)),
        use_safe_factorized_low_rank_head=bool(args.get("use_safe_factorized_low_rank_head", False)),
        safe_factorized_rank=int(args.get("safe_factorized_rank", 64)),
        safe_factorized_scale=float(args.get("safe_factorized_scale", 0.25)),
        safe_factorized_gate_threshold=float(args.get("safe_factorized_gate_threshold", 0.55)),
        safe_factorized_gate_temperature=float(args.get("safe_factorized_gate_temperature", 8.0)),
        safe_factorized_protected_negative_limit=float(args.get("safe_factorized_protected_negative_limit", 0.0)),
        use_factor_interaction_low_rank_head=bool(args.get("use_factor_interaction_low_rank_head", False)),
        factor_interaction_rank=int(args.get("factor_interaction_rank", 64)),
        factor_interaction_scale=float(args.get("factor_interaction_scale", 0.20)),
        factor_interaction_gate_threshold=float(args.get("factor_interaction_gate_threshold", 0.55)),
        factor_interaction_gate_temperature=float(args.get("factor_interaction_gate_temperature", 8.0)),
        factor_interaction_protected_negative_limit=float(args.get("factor_interaction_protected_negative_limit", 0.0)),
        use_water_evidence_logit_gate=bool(args.get("use_water_evidence_logit_gate", False)),
        water_evidence_gate_scale=float(args.get("water_evidence_gate_scale", 0.20)),
        water_evidence_gate_zero_init=bool(args.get("water_evidence_gate_zero_init", True)),
        use_coupled_optical_roughness_residual=bool(args.get("use_coupled_optical_roughness_residual", False)),
        coupled_residual_hidden_dim=int(args.get("coupled_residual_hidden_dim", 96)),
        coupled_residual_scale=float(args.get("coupled_residual_scale", 0.12)),
        coupled_residual_gate_threshold=float(args.get("coupled_residual_gate_threshold", 0.35)),
        coupled_residual_gate_temperature=float(args.get("coupled_residual_gate_temperature", 8.0)),
        coupled_residual_protected_negative_limit=float(args.get("coupled_residual_protected_negative_limit", 0.0)),
        use_roughness_neighbor_residual=bool(args.get("use_roughness_neighbor_residual", False)),
        roughness_neighbor_hidden_dim=int(args.get("roughness_neighbor_hidden_dim", 96)),
        roughness_neighbor_scale=float(args.get("roughness_neighbor_scale", 0.10)),
        roughness_neighbor_gate_threshold=float(args.get("roughness_neighbor_gate_threshold", 0.42)),
        roughness_neighbor_gate_temperature=float(args.get("roughness_neighbor_gate_temperature", 10.0)),
        roughness_neighbor_protected_negative_limit=float(args.get("roughness_neighbor_protected_negative_limit", 0.0)),
        roughness_neighbor_gate_floor=float(args.get("roughness_neighbor_gate_floor", 0.15)),
        class_to_idx=class_to_idx,
        use_factor_aux=float(args.get("factor_aux_weight", 0.0)) > 0.0,
        use_local_physics_factor_aux=float(args.get("local_physics_factor_aux_weight", 0.0)) > 0.0,
        use_physics_aux=float(args.get("physics_aux_weight", 0.0)) > 0.0,
    ).to(device)

    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    weights = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(weights)
    model.eval()
    return model, class_to_idx, args


def _predict_logits(model: nn.Module, image: torch.Tensor, tta: str) -> torch.Tensor:
    logits = model(image)
    if tta == "hflip":
        logits = logits + model(torch.flip(image, dims=[3]))
        logits = logits / 2.0
    return logits


@torch.no_grad()
def evaluate(
    models: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
    *,
    tta: str,
    idx_to_class: dict[int, str],
) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="ensemble-eval", leave=False, ascii=True):
        image = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits_sum = None
        for model in models:
            logits = _predict_logits(model, image, tta)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_mean = logits_sum / max(len(models), 1)
        pred = logits_mean.argmax(dim=1)
        true_batch = labels.detach().cpu().numpy().astype(int).tolist()
        pred_batch = pred.detach().cpu().numpy().astype(int).tolist()
        y_true.extend(true_batch)
        y_pred.extend(pred_batch)
        image_paths = [str(item) for item in batch.get("image_path", [""] * len(true_batch))]
        true_names = [str(item) for item in batch.get("class_label", [idx_to_class[idx] for idx in true_batch])]
        for image_path, true_idx, pred_idx, true_name in zip(image_paths, true_batch, pred_batch, true_names):
            prediction_rows.append(
                {
                    "image_path": image_path,
                    "true_idx": int(true_idx),
                    "pred_idx": int(pred_idx),
                    "true_label": true_name,
                    "pred_label": idx_to_class[int(pred_idx)],
                }
            )

    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[idx] for idx in labels]
    summary = {
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "num_samples": int(len(y_true)),
        "num_classes": int(len(labels)),
    }
    return {
        "summary": summary,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion": confusion_rows(y_true, y_pred, idx_to_class),
        "predictions": prediction_rows,
    }


def _write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluate_test.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions = payload.get("predictions")
    if predictions:
        pd.DataFrame(predictions).to_csv(output_dir / "predictions_test.csv", index=False, encoding="utf-8")
    summary = payload["summary"]
    lines = [
        "# RSCD TTA/Ensemble Evaluation",
        "",
        "Inference-only evaluation on the original RSCD 27-class protocol.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| top-1 accuracy | {summary['top1'] * 100:.2f} |",
        f"| mean precision | {summary['mean_precision'] * 100:.2f} |",
        f"| mean recall | {summary['mean_recall'] * 100:.2f} |",
        f"| macro F1 | {summary['macro_f1'] * 100:.2f} |",
        f"| weighted F1 | {summary['weighted_f1'] * 100:.2f} |",
        f"| balanced accuracy | {summary['balanced_accuracy'] * 100:.2f} |",
        f"| samples | {summary['num_samples']} |",
        f"| classes | {summary['num_classes']} |",
        "",
    ]
    (output_dir / "evaluate_test.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    models = []
    protocols = []
    reference_class_to_idx = None
    image_size = None
    for run_dir in args.run_dirs:
        model, class_to_idx, model_args = _load_model(run_dir, device)
        if reference_class_to_idx is None:
            reference_class_to_idx = class_to_idx
            image_size = int(model_args.get("image_size", 192))
        elif class_to_idx != reference_class_to_idx:
            raise ValueError(f"Class map mismatch in {run_dir}")
        elif int(model_args.get("image_size", image_size)) != image_size:
            raise ValueError(f"Image-size mismatch in {run_dir}; use separate evaluations.")
        models.append(model)
        protocols.append({"run_dir": str(run_dir), "args": model_args})

    if reference_class_to_idx is None or image_size is None:
        raise ValueError("At least one run directory is required.")
    idx_to_class = {idx: name for name, idx in reference_class_to_idx.items()}
    eval_tf = build_transforms(int(image_size), train=False, aug_cfg={"resize_mode": "letterbox"})
    dataset = RSCDSurfaceDataset(
        args.test_manifest,
        class_to_idx=reference_class_to_idx,
        transform=eval_tf,
        max_samples=args.max_test_samples,
        max_samples_per_class=args.max_test_samples_per_class,
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(models, loader, device, tta=str(args.tta), idx_to_class=idx_to_class)
    payload = {
        "protocol": {
            "role": "RSCD 27-class inference-only TTA/ensemble evaluation",
            "run_dirs": [str(p) for p in args.run_dirs],
            "tta": str(args.tta),
            "image_size": int(image_size),
            "test_manifest": str(args.test_manifest),
            "protocols": protocols,
        },
        **metrics,
    }
    _write_outputs(args.output_dir, payload)
    print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

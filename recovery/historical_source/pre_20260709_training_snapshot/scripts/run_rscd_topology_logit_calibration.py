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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed
from run_rscd_surface_classification import RSCDSurfaceDataset, SurfaceClassifier, build_class_map, collate, confusion_rows


DEFAULT_RUN = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\formal_physics_texture_quality_b12e20_resume"
)
DEFAULT_OUT = Path(
    r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
    r"\topology_logit_calibration_physics_texture"
)
DEFAULT_TRAIN = Path("data/manifests_full/rscd_prepared_train.csv")
DEFAULT_VAL = Path("data/manifests_full/rscd_prepared_val.csv")
DEFAULT_TEST = Path("data/manifests_full/rscd_prepared_test.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc RSCD topology/logit calibration. This is a lightweight "
            "exploratory test: it keeps the trained PhysicsTexture checkpoint fixed, "
            "adds explicit Euler-curve topology features, and evaluates whether a "
            "validation-fitted calibrator can improve hard wet/water confusions."
        )
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(args.source_run / "protocol.json")
    train_args = protocol["args"]
    class_to_idx = {str(k): int(v) for k, v in protocol.get("class_to_idx", {}).items()}
    if not class_to_idx:
        class_to_idx = build_class_map([args.train_manifest, args.val_manifest, args.test_manifest])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    image_size = int(train_args.get("image_size", 192))

    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = build_model_from_protocol(train_args, class_to_idx).to(device)
    checkpoint = args.checkpoint or (args.source_run / "best.pt")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    val_cache = args.output_dir / "val_logits_topology.npz"
    test_cache = args.output_dir / "test_logits_topology.npz"
    val = collect_or_load(
        split="val",
        cache_path=val_cache,
        manifest=args.val_manifest,
        class_to_idx=class_to_idx,
        image_size=image_size,
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
        image_size=image_size,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        force_cache=bool(args.force_cache),
        max_samples=train_args.get("max_test_samples"),
        max_samples_per_class=train_args.get("max_test_samples_per_class"),
        seed=int(train_args.get("seed", args.seed)),
    )

    result = run_calibration(val, test, idx_to_class, seed=int(args.seed))
    result["protocol"] = {
        "claim_boundary": (
            "Topology/logit calibration is a post-hoc exploratory candidate. "
            "It is not a replacement for a retrained single neural model unless "
            "it beats the fixed PhysicsTexture test result without hard-slice regressions."
        ),
        "source_run": str(args.source_run),
        "checkpoint": str(checkpoint),
        "val_cache": str(val_cache),
        "test_cache": str(test_cache),
        "calibration_selection": (
            "Validation is split stratified 50/50 into calibration-train and "
            "calibration-select. Test is used once after selecting feature set, C, and blend alpha."
        ),
    }
    (args.output_dir / "topology_logit_calibration.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if "calibrated_evaluate_test" in result:
        (args.output_dir / "evaluate_test.json").write_text(
            json.dumps(result["calibrated_evaluate_test"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_eval_md(args.output_dir / "evaluate_test.md", result["calibrated_evaluate_test"])
    (args.output_dir / "topology_logit_calibration.md").write_text(
        to_markdown(result),
        encoding="utf-8",
    )
    mirror_report = Path("reports/paper_protocol_summary/rscd_topology_logit_calibration.md")
    mirror_report.parent.mkdir(parents=True, exist_ok=True)
    mirror_report.write_text(to_markdown(result), encoding="utf-8")
    print(mirror_report)


def load_protocol(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Protocol not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_from_protocol(train_args: dict[str, Any], class_to_idx: dict[str, int]) -> SurfaceClassifier:
    return SurfaceClassifier(
        backbone=str(train_args.get("backbone", "convnext_tiny")),
        embedding_dim=int(train_args.get("embedding_dim", 768)),
        num_classes=len(class_to_idx),
        pretrained=bool(train_args.get("pretrained", True)),
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
        local_physics_field_scale=float(train_args.get("local_physics_field_scale", 0.15)),
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
        factorized_factors=tuple(
            item.strip()
            for item in str(train_args.get("factorized_factors", "friction,material,unevenness")).split(",")
            if item.strip()
        ),
        factorized_class_embedding=bool(train_args.get("factorized_class_embedding", True)),
        use_safe_factorized_low_rank_head=bool(train_args.get("use_safe_factorized_low_rank_head", False)),
        safe_factorized_rank=int(train_args.get("safe_factorized_rank", 64)),
        safe_factorized_scale=float(train_args.get("safe_factorized_scale", 0.25)),
        safe_factorized_gate_threshold=float(train_args.get("safe_factorized_gate_threshold", 0.55)),
        safe_factorized_gate_temperature=float(train_args.get("safe_factorized_gate_temperature", 8.0)),
        safe_factorized_protected_negative_limit=float(
            train_args.get("safe_factorized_protected_negative_limit", 0.0)
        ),
        use_factor_interaction_low_rank_head=bool(train_args.get("use_factor_interaction_low_rank_head", False)),
        factor_interaction_rank=int(train_args.get("factor_interaction_rank", 64)),
        factor_interaction_scale=float(train_args.get("factor_interaction_scale", 0.20)),
        factor_interaction_gate_threshold=float(train_args.get("factor_interaction_gate_threshold", 0.55)),
        factor_interaction_gate_temperature=float(train_args.get("factor_interaction_gate_temperature", 8.0)),
        factor_interaction_protected_negative_limit=float(
            train_args.get("factor_interaction_protected_negative_limit", 0.0)
        ),
        use_local_global_factor_attention=bool(train_args.get("use_local_global_factor_attention", False)),
        local_global_factor_rank=int(train_args.get("local_global_factor_rank", 48)),
        local_global_factor_scale=float(train_args.get("local_global_factor_scale", 0.08)),
        local_global_factor_gate_threshold=float(train_args.get("local_global_factor_gate_threshold", 0.35)),
        local_global_factor_gate_temperature=float(train_args.get("local_global_factor_gate_temperature", 10.0)),
        local_global_factor_neighbor_gate_floor=float(train_args.get("local_global_factor_neighbor_gate_floor", 0.15)),
        local_global_factor_protected_negative_limit=float(
            train_args.get("local_global_factor_protected_negative_limit", 0.0)
        ),
        use_factor_aux=float(train_args.get("factor_aux_weight", 0.0)) > 0.0,
        use_backbone_aux=float(train_args.get("backbone_aux_weight", 0.0)) > 0.0,
        use_physics_aux=float(train_args.get("physics_aux_weight", 0.0)) > 0.0,
        class_to_idx=class_to_idx,
    )


def collect_or_load(
    *,
    split: str,
    cache_path: Path,
    manifest: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    model: SurfaceClassifier,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force_cache: bool,
    max_samples: int | None = None,
    max_samples_per_class: int | None = None,
    seed: int = 79,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not force_cache:
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    transform = build_transforms(int(image_size), train=False, aug_cfg={"resize_mode": "letterbox"})
    dataset = RSCDSurfaceDataset(
        manifest,
        class_to_idx=class_to_idx,
        transform=transform,
        max_samples=max_samples,
        max_samples_per_class=max_samples_per_class,
        seed=int(seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    logits_list: list[np.ndarray] = []
    topology_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    path_list: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"collect {split}", leave=False, ascii=True):
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(image)
            topology = extract_topology_features(image)
            logits_list.append(logits.detach().float().cpu().numpy())
            topology_list.append(topology.detach().float().cpu().numpy())
            label_list.append(batch["label"].detach().cpu().numpy().astype(np.int64))
            path_list.extend([str(x) for x in batch["image_path"]])
    payload = {
        "logits": np.concatenate(logits_list, axis=0),
        "topology": np.concatenate(topology_list, axis=0),
        "label": np.concatenate(label_list, axis=0),
        "image_path": np.asarray(path_list, dtype=object),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def extract_topology_features(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3) / 8.0
    rgb = (image * std + mean).clamp(0.0, 1.0)
    if max(rgb.shape[-2:]) != 96:
        rgb = F.interpolate(rgb, size=(96, 96), mode="bilinear", align_corners=False)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    maxc = rgb.max(dim=1, keepdim=True).values
    minc = rgb.min(dim=1, keepdim=True).values
    saturation = (maxc - minc) / maxc.clamp_min(1e-4)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
    grad_norm = normalize_map(grad)
    snow_like = torch.sigmoid((maxc - 0.72) * 12.0) * torch.sigmoid((0.28 - saturation) * 12.0)
    specular = torch.sigmoid((maxc - 0.82) * 14.0) * torch.sigmoid((0.24 - saturation) * 12.0)
    dark_water = torch.sigmoid((0.38 - maxc) * 10.0) * torch.sigmoid((0.12 - grad) * 30.0)
    wet_proxy = torch.clamp(specular + 0.5 * dark_water, 0.0, 1.0)
    low_texture = torch.sigmoid((0.045 - grad) * 35.0)
    fields = [gray, grad_norm, snow_like, wet_proxy, low_texture]
    stats = [soft_euler_stats(field) for field in fields]
    stats.extend([tail_stats(wet_proxy), tail_stats(low_texture), tail_stats(specular), tail_stats(dark_water)])
    return torch.cat(stats, dim=1)


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(2)
    low = flat.amin(dim=2).view(x.size(0), x.size(1), 1, 1)
    high = flat.amax(dim=2).view(x.size(0), x.size(1), 1, 1)
    return (x - low) / (high - low).clamp_min(1e-6)


def soft_euler_stats(field: torch.Tensor) -> torch.Tensor:
    thresholds = torch.linspace(0.15, 0.85, 8, device=field.device, dtype=field.dtype).view(1, 1, 8, 1, 1)
    masks = torch.sigmoid((field.unsqueeze(2) - thresholds) * 18.0)
    a = masks[..., :-1, :-1]
    b = masks[..., :-1, 1:]
    c = masks[..., 1:, :-1]
    d = masks[..., 1:, 1:]
    q1 = (
        a * (1.0 - b) * (1.0 - c) * (1.0 - d)
        + (1.0 - a) * b * (1.0 - c) * (1.0 - d)
        + (1.0 - a) * (1.0 - b) * c * (1.0 - d)
        + (1.0 - a) * (1.0 - b) * (1.0 - c) * d
    )
    q3 = (
        (1.0 - a) * b * c * d
        + a * (1.0 - b) * c * d
        + a * b * (1.0 - c) * d
        + a * b * c * (1.0 - d)
    )
    qd = a * d * (1.0 - b) * (1.0 - c) + b * c * (1.0 - a) * (1.0 - d)
    euler = (q1.sum(dim=(-1, -2)) - q3.sum(dim=(-1, -2)) + 2.0 * qd.sum(dim=(-1, -2))) * 0.25
    area = masks.mean(dim=(-1, -2)).squeeze(1)
    euler = euler.squeeze(1) / max(float(field.size(-1) * field.size(-2)), 1.0)
    return torch.cat(
        [
            euler.mean(dim=1, keepdim=True),
            euler.std(dim=1, keepdim=True),
            (euler.amax(dim=1) - euler.amin(dim=1)).unsqueeze(1),
            area.mean(dim=1, keepdim=True),
            area.std(dim=1, keepdim=True),
            (area.amax(dim=1) - area.amin(dim=1)).unsqueeze(1),
        ],
        dim=1,
    )


def tail_stats(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    top10 = max(int(flat.size(1) * 0.10), 1)
    top25 = max(int(flat.size(1) * 0.25), 1)
    return torch.cat(
        [
            flat.mean(dim=1, keepdim=True),
            flat.std(dim=1, keepdim=True),
            flat.topk(top10, dim=1).values.mean(dim=1, keepdim=True),
            flat.topk(top25, dim=1).values.mean(dim=1, keepdim=True),
            (flat > 0.50).to(dtype=x.dtype).mean(dim=1, keepdim=True),
            (flat > 0.75).to(dtype=x.dtype).mean(dim=1, keepdim=True),
        ],
        dim=1,
    )


def run_calibration(
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

    base_val_probs = softmax_np(val_logits)
    base_test_probs = softmax_np(test_logits)
    baseline = metric_bundle(y_test, base_test_probs.argmax(axis=1), idx_to_class)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    cal_idx, select_idx = next(splitter.split(val_logits, y_val))
    feature_sets = {
        "logits_only": (val_logits, test_logits),
        "logits_plus_topology": (np.concatenate([val_logits, val_topo], axis=1), np.concatenate([test_logits, test_topo], axis=1)),
    }
    candidates = []
    for feature_name, (x_val, x_test) in feature_sets.items():
        for c_value in [0.03, 0.10, 0.30, 1.00]:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=float(c_value), max_iter=700, solver="lbfgs"),
            )
            model.fit(x_val[cal_idx], y_val[cal_idx])
            select_probs = model.predict_proba(x_val[select_idx])
            select_base = base_val_probs[select_idx]
            for alpha in [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]:
                blended = (1.0 - alpha) * select_base + alpha * select_probs
                metrics = metric_bundle(y_val[select_idx], blended.argmax(axis=1), idx_to_class)
                candidates.append(
                    {
                        "feature_set": feature_name,
                        "c": float(c_value),
                        "alpha": float(alpha),
                        "select": metrics,
                    }
                )
    candidates.sort(
        key=lambda row: (
            float(row["select"]["macro_f1"]),
            float(row["select"]["top1"]),
            float(row["select"]["wet_water_f1"]),
        ),
        reverse=True,
    )
    selected = candidates[0]

    x_val, x_test = feature_sets[str(selected["feature_set"])]
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(selected["c"]), max_iter=900, solver="lbfgs"),
    )
    final_model.fit(x_val, y_val)
    cal_test_probs = final_model.predict_proba(x_test)
    test_probs = (1.0 - float(selected["alpha"])) * base_test_probs + float(selected["alpha"]) * cal_test_probs
    test_pred = test_probs.argmax(axis=1)
    calibrated = metric_bundle(y_test, test_pred, idx_to_class)
    calibrated_evaluate_test = evaluate_payload(
        y_test,
        test_pred,
        idx_to_class,
        claim_boundary=(
            "Post-hoc calibrated PhysicsTexture result. The ConvNeXt/PhysicsTexture checkpoint is fixed; "
            "a validation-fitted logistic calibrator is blended with the original logits. This is not a "
            "pure retrained single neural model."
        ),
    )
    test_by_feature_set: dict[str, Any] = {}
    for feature_name in feature_sets:
        best_for_feature = next(row for row in candidates if row["feature_set"] == feature_name)
        feature_x_val, feature_x_test = feature_sets[feature_name]
        feature_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(best_for_feature["c"]), max_iter=900, solver="lbfgs"),
        )
        feature_model.fit(feature_x_val, y_val)
        feature_cal_probs = feature_model.predict_proba(feature_x_test)
        feature_probs = (
            (1.0 - float(best_for_feature["alpha"])) * base_test_probs
            + float(best_for_feature["alpha"]) * feature_cal_probs
        )
        feature_metrics = metric_bundle(y_test, feature_probs.argmax(axis=1), idx_to_class)
        test_by_feature_set[feature_name] = {
            "selected_config": {
                "feature_set": feature_name,
                "c": float(best_for_feature["c"]),
                "alpha": float(best_for_feature["alpha"]),
                "select": best_for_feature["select"],
            },
            "test": feature_metrics,
            "delta_vs_baseline": {
                key: float(feature_metrics[key]) - float(baseline[key])
                for key in ["top1", "macro_f1", "wet_water_f1", "water_f1", "ice_f1", "low_friction_f1"]
            },
        }

    return {
        "baseline_fixed_physics_texture": baseline,
        "selected_by_validation": selected,
        "calibrated_test": calibrated,
        "delta_test": {
            key: float(calibrated[key]) - float(baseline[key])
            for key in ["top1", "macro_f1", "wet_water_f1", "water_f1", "ice_f1", "low_friction_f1"]
        },
        "test_by_feature_set": test_by_feature_set,
        "calibrated_evaluate_test": calibrated_evaluate_test,
        "top_validation_candidates": candidates[:12],
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, float]:
    labels = sorted(idx_to_class)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[idx_to_class[idx] for idx in labels],
        output_dict=True,
        zero_division=0,
    )
    slices = weighted_class_slices(report)
    return {
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "wet_water_f1": slices["wet_water_f1"],
        "water_f1": slices["water_f1"],
        "ice_f1": slices["ice_f1"],
        "low_friction_f1": slices["low_friction_f1"],
        "wet_water_macro_f1": grouped_f1(y_true, y_pred, idx_to_class, lambda name: name.startswith("wet_") or name.startswith("water_")),
        "water_macro_f1": grouped_f1(y_true, y_pred, idx_to_class, lambda name: name.startswith("water_")),
        "ice_macro_f1": grouped_f1(y_true, y_pred, idx_to_class, lambda name: name == "ice"),
        "low_friction_macro_f1": grouped_f1(
            y_true,
            y_pred,
            idx_to_class,
            lambda name: name.startswith("wet_") or name.startswith("water_") or name in {"fresh_snow", "melted_snow", "ice"},
        ),
    }


def evaluate_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: dict[int, str],
    *,
    claim_boundary: str,
) -> dict[str, Any]:
    labels = sorted(idx_to_class)
    target_names = [idx_to_class[idx] for idx in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
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
        "classification_report": report,
        "confusion": confusion_rows(y_true.astype(int).tolist(), y_pred.astype(int).tolist(), idx_to_class),
        "claim_boundary": claim_boundary,
    }


def write_eval_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# RSCD Surface Classification Test Result",
        "",
        payload.get("claim_boundary", ""),
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
    path.write_text("\n".join(lines), encoding="utf-8")


def grouped_f1(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict[int, str], keep) -> float:
    chosen = [idx for idx, name in idx_to_class.items() if keep(name)]
    if not chosen:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=chosen, average="macro", zero_division=0))


def weighted_class_slices(report: dict[str, Any]) -> dict[str, float]:
    rows = []
    for label, item in report.items():
        if not isinstance(item, dict) or "f1-score" not in item:
            continue
        friction = friction_state(label)
        rows.append(
            {
                "label": label,
                "friction": friction,
                "f1": float(item.get("f1-score") or 0.0),
                "support": float(item.get("support") or 0.0),
            }
        )
    return {
        "wet_water_f1": weighted_slice(rows, lambda row: row["friction"] in {"wet", "water"}),
        "water_f1": weighted_slice(rows, lambda row: row["friction"] == "water"),
        "ice_f1": weighted_slice(rows, lambda row: row["friction"] == "ice"),
        "low_friction_f1": weighted_slice(
            rows,
            lambda row: row["friction"] in {"wet", "water", "fresh_snow", "melted_snow", "ice"},
        ),
    }


def weighted_slice(rows: list[dict[str, Any]], keep) -> float:
    chosen = [row for row in rows if keep(row)]
    support = sum(float(row["support"]) for row in chosen)
    if support <= 0:
        return 0.0
    return float(sum(float(row["f1"]) * float(row["support"]) for row in chosen) / support)


def friction_state(label: str) -> str:
    label = str(label).strip().lower().replace("-", "_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return label
    parts = label.split("_")
    return parts[0] if parts else "unknown"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def to_markdown(result: dict[str, Any]) -> str:
    baseline = result["baseline_fixed_physics_texture"]
    calibrated = result["calibrated_test"]
    delta = result["delta_test"]
    selected = result["selected_by_validation"]
    lines = [
        "# RSCD Topology Logit Calibration",
        "",
        result["protocol"]["claim_boundary"],
        "",
        "## Selected Calibrator",
        "",
        f"- Feature set: `{selected['feature_set']}`",
        f"- Logistic C: `{selected['c']}`",
        f"- Blend alpha: `{selected['alpha']}`",
        "- Selection split: stratified half of validation; test is not used for selection.",
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
            "| topology/logit calibrated | "
            f"{pct(calibrated['top1'])} | {pct(calibrated['macro_f1'])} | {pct(calibrated['wet_water_f1'])} | "
            f"{pct(calibrated['water_f1'])} | {pct(calibrated['ice_f1'])} | {pct(calibrated['low_friction_f1'])} |"
        ),
        (
            "| delta | "
            f"{pp(delta['top1'])} | {pp(delta['macro_f1'])} | {pp(delta['wet_water_f1'])} | "
            f"{pp(delta['water_f1'])} | {pp(delta['ice_f1'])} | {pp(delta['low_friction_f1'])} |"
        ),
        "",
        "## Decision Rule",
        "",
        (
            "Promote only if the calibrated result improves Top-1 or Macro-F1 and does not regress "
            "wet/water safety slices. Otherwise keep topology as an analysis/calibration idea rather "
            "than a main model component."
        ),
        "",
        "## Feature-Set Ablation On Test",
        "",
        "| feature set | C | alpha | Top-1 | Macro-F1 | wet/water F1 | water F1 | d Top-1 | d Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature_name, item in result.get("test_by_feature_set", {}).items():
        cfg = item["selected_config"]
        metrics = item["test"]
        delta_item = item["delta_vs_baseline"]
        lines.append(
            f"| `{feature_name}` | {cfg['c']:.2f} | {cfg['alpha']:.2f} | "
            f"{pct(metrics['top1'])} | {pct(metrics['macro_f1'])} | {pct(metrics['wet_water_f1'])} | "
            f"{pct(metrics['water_f1'])} | {pp(delta_item['top1'])} | {pp(delta_item['macro_f1'])} |"
        )
    lines.extend(
        [
        "",
        "## Top Validation Candidates",
        "",
        "| rank | feature set | C | alpha | select Top-1 | select Macro-F1 | select wet/water F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(result["top_validation_candidates"], start=1):
        metrics = row["select"]
        lines.append(
            f"| {idx} | `{row['feature_set']}` | {row['c']:.2f} | {row['alpha']:.2f} | "
            f"{pct(metrics['top1'])} | {pct(metrics['macro_f1'])} | {pct(metrics['wet_water_f1'])} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

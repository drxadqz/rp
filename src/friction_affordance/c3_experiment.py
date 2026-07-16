from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from tqdm import tqdm
import yaml

from friction_affordance.c3_losses import (
    c3_total_aux_loss,
    factor_graph_metric_loss,
    mechanism_routed_tournament_loss,
)
from friction_affordance.models.c3_farnet import C3FaRNetSurfaceClassifier, C3PhysicsEvidenceStats
from friction_affordance.rscd_factors import (
    FACTOR_AXES,
    FACTOR_LABELS,
    RSCDFactorSpec,
    build_rscd_factor_spec,
    canonical_class_label,
    class_factor_targets,
    parse_rscd_label,
    sanity_summary,
)
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import resolve_device, set_seed


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _deep_update_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_update_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config_file(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"config extends cycle detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.pop("extends", None)
    if extends:
        base_path = Path(str(extends))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_cfg = _load_config_file(base_path, seen)
        cfg = _deep_update_config(base_cfg, cfg)
    return cfg


def load_config(path: Path) -> dict[str, Any]:
    cfg = _load_config_file(path)
    cfg.setdefault("seed", 79)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("output_dir", "outputs/c3_farnet")
    _normalize_model_config(cfg["model"])
    _normalize_loss_config(cfg["loss"])
    return cfg


def _normalize_model_config(model_cfg: dict[str, Any]) -> None:
    """Accept whitepaper-style config aliases while keeping old configs valid."""

    head_cfg = model_cfg.get("head", {}) if isinstance(model_cfg.get("head"), dict) else {}
    if "head_type" not in model_cfg and "type" in head_cfg:
        model_cfg["head_type"] = head_cfg["type"]
    if "use_dry_vor" in model_cfg and "use_dry_concrete_roughness_vor_residual" not in model_cfg:
        model_cfg["use_dry_concrete_roughness_vor_residual"] = bool(model_cfg["use_dry_vor"])


def _normalize_loss_config(loss_cfg: dict[str, Any]) -> None:
    """Map C3-FaRNet paper notation to the implementation's loss weights."""

    axis_weights = dict(loss_cfg.get("factor_axis_weights", {}) or {})
    alias_to_axis = {
        "lambda_factor_f": "friction",
        "lambda_factor_m": "material",
        "lambda_factor_r": "roughness",
    }
    for alias, axis in alias_to_axis.items():
        if alias in loss_cfg:
            axis_weights[axis] = float(loss_cfg[alias])
    if axis_weights:
        loss_cfg["factor_axis_weights"] = {
            "friction": float(axis_weights.get("friction", 1.0)),
            "material": float(axis_weights.get("material", 1.0)),
            "roughness": float(axis_weights.get("roughness", 1.0)),
        }

    if "lambda_factor" in loss_cfg and "factor_weight" not in loss_cfg:
        loss_cfg["factor_weight"] = float(loss_cfg["lambda_factor"])
    if "lambda_tournament" in loss_cfg and "tournament_weight" not in loss_cfg:
        loss_cfg["tournament_weight"] = float(loss_cfg["lambda_tournament"])
    if "lambda_counterfactual" in loss_cfg and "counterfactual_weight" not in loss_cfg:
        loss_cfg["counterfactual_weight"] = float(loss_cfg["lambda_counterfactual"])
    if "lambda_reliability" in loss_cfg and "reliability_weight" not in loss_cfg:
        loss_cfg["reliability_weight"] = float(loss_cfg["lambda_reliability"])

    use_to_weight = {
        "use_factor_ce": ("factor_weight", 0.3),
        "use_tournament": ("tournament_weight", 0.1),
        "use_counterfactual": ("counterfactual_weight", 0.05),
        "use_reliability": ("reliability_weight", 0.05),
    }
    for flag, (weight_name, default_weight) in use_to_weight.items():
        if flag not in loss_cfg:
            continue
        if not bool(loss_cfg[flag]):
            loss_cfg[weight_name] = 0.0
        elif weight_name not in loss_cfg:
            loss_cfg[weight_name] = float(default_weight)


def build_class_map(manifests: list[Path]) -> dict[str, int]:
    labels: set[str] = set()
    for manifest in manifests:
        df = pd.read_csv(manifest, usecols=["class_label"], dtype=str, low_memory=False)
        labels.update(canonical_class_label(v) for v in df["class_label"].dropna().astype(str).unique().tolist())
    return {name: idx for idx, name in enumerate(sorted(labels))}


class RSCDSurfaceDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        *,
        class_to_idx: dict[str, int],
        transform,
        max_samples: int | None = None,
        max_samples_per_class: int | None = None,
        seed: int = 79,
    ) -> None:
        df = pd.read_csv(manifest, dtype=str, low_memory=False)
        df["class_label_canonical"] = df["class_label"].map(canonical_class_label)
        df = df[df["class_label_canonical"].isin(class_to_idx)].copy()
        if max_samples_per_class:
            parts = []
            for _, group in df.groupby("class_label_canonical", sort=True):
                n = min(int(max_samples_per_class), len(group))
                parts.append(group.sample(n=n, random_state=int(seed)))
            df = pd.concat(parts, ignore_index=True)
        if max_samples:
            df = df.sample(n=min(int(max_samples), len(df)), random_state=int(seed)).reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self._warned: set[str] = set()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for offset in range(min(50, len(self.df))):
            row = self.df.iloc[(int(idx) + offset) % len(self.df)]
            path = Path(str(row["image_path"]))
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.load()
                    tensor = self.transform(image)
                label_name = canonical_class_label(row["class_label_canonical"])
                triple = parse_rscd_label(label_name)
                return {
                    "image": tensor,
                    "label": torch.tensor(self.class_to_idx[label_name], dtype=torch.long),
                    "friction_factor": torch.tensor(triple.friction, dtype=torch.long),
                    "material_factor": torch.tensor(triple.material, dtype=torch.long),
                    "roughness_factor": torch.tensor(triple.roughness, dtype=torch.long),
                    "class_label": label_name,
                    "image_path": str(path),
                }
            except (OSError, SyntaxError, ValueError) as exc:
                last_error = exc
                path_text = str(path)
                if path_text not in self._warned:
                    self._warned.add(path_text)
                    print(f"WARNING: skipped unreadable image: {path_text} ({type(exc).__name__}: {exc})")
        raise RuntimeError(f"Could not load image near index {idx}: {last_error}")


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "friction_factor": torch.stack([item["friction_factor"] for item in batch]),
        "material_factor": torch.stack([item["material_factor"] for item in batch]),
        "roughness_factor": torch.stack([item["roughness_factor"] for item in batch]),
        "class_label": [str(item["class_label"]) for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
    }


class RSCDFactorGraphPairBatchSampler(Sampler[list[int]]):
    """Build batches that contain RSCD factor-graph neighbors and positives.

    Factor-graph metric learning needs same-batch structure: exact-class
    positives for the coupling token and one-axis-different neighbors for the
    friction/material/roughness graph. Plain class-balanced sampling is fair at
    epoch level but often misses the wet/water concrete hard subgraph inside a
    small batch. This sampler is therefore a task-adapted batch constructor, not
    a generic weak-class oversampler.
    """

    def __init__(
        self,
        dataset: RSCDSurfaceDataset,
        *,
        class_to_idx: dict[str, int],
        batch_size: int,
        num_samples: int,
        seed: int,
        pair_slots: int = 2,
        positive_slots: int = 1,
        wet_concrete_focus_scale: float = 3.0,
        roughness_focus_scale: float = 1.5,
        wet_water_focus_scale: float = 1.5,
        focus_pairs: list[str] | tuple[str, ...] | None = None,
        focus_pairs_only: bool = False,
        start_batch: int = 0,
    ) -> None:
        if batch_size < 2:
            raise ValueError("factor_graph_pair_sampling requires batch_size >= 2")
        self.dataset = dataset
        self.class_to_idx = {canonical_class_label(k): int(v) for k, v in class_to_idx.items()}
        self.idx_to_class = {int(v): canonical_class_label(k) for k, v in self.class_to_idx.items()}
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.num_batches = max(int(np.ceil(max(self.num_samples, 1) / float(self.batch_size))), 1)
        self.seed = int(seed)
        self.pair_slots = max(int(pair_slots), 0)
        self.positive_slots = max(int(positive_slots), 0)
        self.start_batch = min(max(int(start_batch), 0), self.num_batches)

        self.class_to_rows: dict[int, np.ndarray] = {}
        for label_name, group in dataset.df.groupby("class_label_canonical", sort=True):
            idx = self.class_to_idx.get(canonical_class_label(label_name))
            if idx is None:
                continue
            rows = group.index.to_numpy(dtype=np.int64)
            if rows.size > 0:
                self.class_to_rows[int(idx)] = rows
        self.present_classes = np.array(sorted(self.class_to_rows), dtype=np.int64)
        if self.present_classes.size == 0:
            raise ValueError("factor_graph_pair_sampling found no present classes in dataset")

        spec = build_rscd_factor_spec(self.class_to_idx)
        factors = spec.class_to_factor.numpy()
        wet_idx = FACTOR_LABELS["friction"].index("wet")
        water_idx = FACTOR_LABELS["friction"].index("water")
        concrete_idx = FACTOR_LABELS["material"].index("concrete")
        requested_pairs: set[frozenset[str]] = set()
        for item in focus_pairs or []:
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) == 2:
                requested_pairs.add(
                    frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))
                )

        pairs: list[tuple[int, int]] = []
        weights: list[float] = []
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            if left not in self.class_to_rows or right not in self.class_to_rows:
                continue
            pair_names = frozenset((self.idx_to_class[left], self.idx_to_class[right]))
            if requested_pairs and bool(focus_pairs_only) and pair_names not in requested_pairs:
                continue
            w = 1.0
            if requested_pairs and pair_names in requested_pairs:
                w *= 8.0
            left_f, left_m, _ = factors[left].tolist()
            right_f, right_m, _ = factors[right].tolist()
            both_wet_concrete = (
                left_f in {wet_idx, water_idx}
                and right_f in {wet_idx, water_idx}
                and left_m == concrete_idx
                and right_m == concrete_idx
            )
            if both_wet_concrete:
                w *= max(float(wet_concrete_focus_scale), 1.0)
            if pair.axis == "roughness":
                w *= max(float(roughness_focus_scale), 1.0)
            if pair.boundary == "wet_water":
                w *= max(float(wet_water_focus_scale), 1.0)
            pairs.append((left, right))
            weights.append(float(w))
        if not pairs:
            raise ValueError("factor_graph_pair_sampling found no valid factor graph pairs")
        self.pairs = np.asarray(pairs, dtype=np.int64)
        pair_weights = np.asarray(weights, dtype=np.float64)
        self.pair_probs = pair_weights / pair_weights.sum()

    def __len__(self) -> int:
        return int(max(self.num_batches - self.start_batch, 0))

    def _sample_row(self, cls_idx: int, rng: np.random.Generator) -> int:
        rows = self.class_to_rows[int(cls_idx)]
        return int(rows[int(rng.integers(0, len(rows)))])

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        for batch_i in range(self.num_batches):
            batch: list[int] = []
            max_pair_slots = min(self.pair_slots, self.batch_size // 2)
            for _pair_slot in range(max_pair_slots):
                pair_idx = int(rng.choice(len(self.pairs), p=self.pair_probs))
                left, right = self.pairs[pair_idx].tolist()
                batch.append(self._sample_row(left, rng))
                batch.append(self._sample_row(right, rng))
            remaining_after_pairs = self.batch_size - len(batch)
            max_positive_slots = min(self.positive_slots, remaining_after_pairs // 2)
            for _positive_slot in range(max_positive_slots):
                cls_idx = int(rng.choice(self.present_classes))
                batch.append(self._sample_row(cls_idx, rng))
                batch.append(self._sample_row(cls_idx, rng))
            while len(batch) < self.batch_size:
                cls_idx = int(rng.choice(self.present_classes))
                batch.append(self._sample_row(cls_idx, rng))
            rng.shuffle(batch)
            if batch_i < self.start_batch:
                continue
            yield batch


def _apply_anchor_error_sampler_weights(
    *,
    weights: np.ndarray,
    train_ds: RSCDSurfaceDataset,
    train_cfg: dict[str, Any],
    class_to_idx: dict[str, int],
) -> np.ndarray:
    """Boost cached anchor-error boundary samples for RSCD no-harm repair.

    This is a task-adapted JTT/GroupDRO sampler: it does not simply oversample
    a class. It oversamples samples in specific RSCD composite classes where a
    frozen anchor already makes mistakes, so PCGrad/no-harm losses see enough
    repair cases in each random batch.
    """

    cache_path_text = train_cfg.get("anchor_error_sampler_cache")
    if not cache_path_text:
        return weights
    cache_path = Path(str(cache_path_text))
    if not cache_path.exists():
        raise FileNotFoundError(f"anchor error sampler cache does not exist: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    image_paths = payload.get("image_paths")
    labels_obj = payload.get("labels")
    logits_obj = payload.get("logits")
    cache_class_to_idx = payload.get("class_to_idx", class_to_idx)
    if image_paths is None or labels_obj is None or logits_obj is None:
        raise ValueError(f"anchor error sampler cache must contain image_paths, labels, logits: {cache_path}")
    labels = torch.as_tensor(labels_obj, dtype=torch.long)
    logits = torch.as_tensor(logits_obj).float()
    if len(image_paths) != int(labels.numel()) or len(image_paths) != int(logits.shape[0]):
        raise ValueError(
            f"anchor error sampler cache length mismatch: paths={len(image_paths)} "
            f"labels={int(labels.numel())} logits={tuple(logits.shape)}"
        )

    cache_idx_to_class = {int(idx): canonical_class_label(name) for name, idx in dict(cache_class_to_idx).items()}
    focus_names = {
        canonical_class_label(name)
        for name in train_cfg.get("anchor_error_sampler_focus_classes", [])
    }
    if not focus_names:
        focus_names = {
            "water_concrete_slight",
            "water_concrete_severe",
            "water_concrete_smooth",
        }
    error_boost = max(float(train_cfg.get("anchor_error_sampler_error_boost", 1.0)), 1.0)
    focus_boost = max(float(train_cfg.get("anchor_error_sampler_focus_boost", 1.0)), 1.0)
    include_correct_focus = bool(train_cfg.get("anchor_error_sampler_include_correct_focus", True))

    boosts: dict[str, float] = {}
    stats = {
        "focus_cached": 0,
        "focus_errors": 0,
        "focus_correct": 0,
        "matched_dataset_rows": 0,
    }
    pred = logits.argmax(dim=1)
    for i, path_text in enumerate(image_paths):
        label_idx = int(labels[i])
        true_name = cache_idx_to_class.get(label_idx)
        if true_name is None or true_name not in focus_names:
            continue
        stats["focus_cached"] += 1
        is_error = int(pred[i]) != label_idx
        if is_error:
            stats["focus_errors"] += 1
            boost = error_boost
        else:
            stats["focus_correct"] += 1
            if not include_correct_focus:
                continue
            boost = focus_boost
        key = teacher_cache_key(str(path_text))
        boosts[key] = max(boosts.get(key, 1.0), float(boost))

    if not boosts:
        print(f"Anchor-error sampler cache loaded but no focus paths matched: {cache_path}")
        return weights

    boosted = weights.copy()
    image_path_series = train_ds.df["image_path"].astype(str)
    for row_idx, path_text in enumerate(image_path_series.tolist()):
        boost = boosts.get(teacher_cache_key(path_text))
        if boost is None:
            continue
        boosted[int(row_idx)] *= float(boost)
        stats["matched_dataset_rows"] += 1
    print(
        "Anchor-error sampler enabled: "
        f"cache={cache_path} focus_cached={stats['focus_cached']} "
        f"errors={stats['focus_errors']} correct={stats['focus_correct']} "
        f"matched_rows={stats['matched_dataset_rows']} "
        f"error_boost={error_boost:.1f} focus_boost={focus_boost:.1f}"
    )
    return boosted


def build_loaders(cfg: dict[str, Any], class_to_idx: dict[str, int]) -> tuple[DataLoader, DataLoader, DataLoader]:
    data = cfg["data"]
    train_cfg = cfg["train"]
    image_size = int(data.get("image_size", 192))
    train_tf = build_transforms(
        image_size,
        train=bool(train_cfg.get("augmentation", True)),
        aug_cfg={"resize_mode": str(data.get("train_resize_mode", "letterbox"))},
    )
    eval_tf = build_transforms(image_size, train=False, aug_cfg={"resize_mode": str(data.get("eval_resize_mode", "letterbox"))})
    train_ds = RSCDSurfaceDataset(
        Path(data["train_manifest"]),
        class_to_idx=class_to_idx,
        transform=train_tf,
        max_samples=train_cfg.get("max_train_samples"),
        max_samples_per_class=train_cfg.get("max_train_samples_per_class"),
        seed=int(cfg.get("seed", 79)),
    )
    val_ds = RSCDSurfaceDataset(
        Path(data["val_manifest"]),
        class_to_idx=class_to_idx,
        transform=eval_tf,
        max_samples=cfg["eval"].get("max_val_samples"),
        max_samples_per_class=cfg["eval"].get("max_val_samples_per_class"),
        seed=int(cfg.get("seed", 79)) + 1,
    )
    test_ds = RSCDSurfaceDataset(
        Path(data["test_manifest"]),
        class_to_idx=class_to_idx,
        transform=eval_tf,
        max_samples=cfg["eval"].get("max_test_samples"),
        max_samples_per_class=cfg["eval"].get("max_test_samples_per_class"),
        seed=int(cfg.get("seed", 79)) + 2,
    )
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
        "collate_fn": collate,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", False))
    sampler = None
    batch_sampler = None
    shuffle = True
    resume_start_batch = int(train_cfg.get("_resume_start_step", train_cfg.get("resume_start_step", 0)) or 0)
    if bool(train_cfg.get("factor_graph_pair_sampling", False)):
        num_samples = int(train_cfg.get("samples_per_epoch", 0)) or len(train_ds)
        batch_sampler = RSCDFactorGraphPairBatchSampler(
            train_ds,
            class_to_idx=class_to_idx,
            batch_size=batch_size,
            num_samples=num_samples,
            seed=int(train_cfg.get("factor_graph_pair_sampling_seed", int(cfg.get("seed", 79)))),
            pair_slots=int(train_cfg.get("factor_graph_pair_sampling_pair_slots", 2)),
            positive_slots=int(train_cfg.get("factor_graph_pair_sampling_positive_slots", 1)),
            wet_concrete_focus_scale=float(train_cfg.get("factor_graph_pair_sampling_wet_concrete_focus_scale", 3.0)),
            roughness_focus_scale=float(train_cfg.get("factor_graph_pair_sampling_roughness_focus_scale", 1.5)),
            wet_water_focus_scale=float(train_cfg.get("factor_graph_pair_sampling_wet_water_focus_scale", 1.5)),
            focus_pairs=train_cfg.get("factor_graph_pair_sampling_focus_pairs"),
            focus_pairs_only=bool(train_cfg.get("factor_graph_pair_sampling_focus_pairs_only", False)),
            start_batch=resume_start_batch,
        )
        shuffle = False
        print(
            "Factor-graph pair batch sampler enabled: "
            f"batches={len(batch_sampler)} batch_size={batch_size} "
            f"start_batch={batch_sampler.start_batch}/{batch_sampler.num_batches} "
            f"pair_slots={batch_sampler.pair_slots} positive_slots={batch_sampler.positive_slots}"
        )
    elif bool(train_cfg.get("balanced_sampling", True)):
        sizes = train_ds.df.groupby("class_label_canonical")["class_label_canonical"].transform("size").astype(float)
        weights = (1.0 / sizes.clip(lower=1.0)).to_numpy(dtype=np.float64).copy()
        weights = _apply_anchor_error_sampler_weights(
            weights=weights,
            train_ds=train_ds,
            train_cfg=train_cfg,
            class_to_idx=class_to_idx,
        )
        num_samples = int(train_cfg.get("samples_per_epoch", 0)) or len(train_ds)
        if resume_start_batch > 0:
            consumed = int(resume_start_batch) * int(batch_size)
            remaining = max(int(num_samples) - consumed, 1)
            print(
                "Balanced sampler resume enabled: "
                f"start_batch={resume_start_batch} consumed={consumed} "
                f"remaining_samples={remaining}/{num_samples}"
            )
            num_samples = remaining
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)
        shuffle = False
    if batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["eval"].get("batch_size", batch_size)), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["eval"].get("batch_size", batch_size)), shuffle=False, **loader_kwargs)
    print(f"Dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_loader, val_loader, test_loader


def build_model(cfg: dict[str, Any], class_to_idx: dict[str, int]) -> C3FaRNetSurfaceClassifier:
    m = cfg["model"]
    head_cfg = m.get("head", {}) if isinstance(m.get("head"), dict) else {}
    head_type = str(m.get("head_type", head_cfg.get("type", "coupled_tensor")))
    pareto_edge_expert_rules = m.get("pareto_edge_expert_rules")
    if m.get("pareto_edge_expert_rules_path"):
        pareto_edge_expert_rules = load_pareto_safe_logit_patch_rules(Path(str(m["pareto_edge_expert_rules_path"])))
    return C3FaRNetSurfaceClassifier(
        class_to_idx=class_to_idx,
        backbone=str(m.get("backbone", "convnext_tiny")),
        embedding_dim=int(m.get("embedding_dim", 768)),
        pretrained=bool(m.get("pretrained", False)),
        dropout=float(m.get("dropout", 0.2)),
        token_dim=int(m.get("token_dim", 256)),
        pair_rank=int(m.get("pair_rank", 8)),
        triple_rank=int(m.get("triple_rank", 8)),
        head_type=head_type,
        hybrid_coupled_scale=float(m.get("hybrid_coupled_scale", 0.10)),
        hardpair_correction_scale=float(m.get("hardpair_correction_scale", 0.08)),
        hardpair_margin_scale=float(m.get("hardpair_margin_scale", 0.18)),
        hardpair_gate_margin=float(m.get("hardpair_gate_margin", 1.00)),
        hardpair_gate_temperature=float(m.get("hardpair_gate_temperature", 4.00)),
        hardpair_error_gate_bias_init=float(m.get("hardpair_error_gate_bias_init", -3.5)),
        hardpair_error_gate_floor=float(m.get("hardpair_error_gate_floor", 0.0)),
        hardpair_physics_gate=str(m.get("hardpair_physics_gate", "none")),
        hardpair_physics_gate_floor=float(m.get("hardpair_physics_gate_floor", 0.0)),
        hardpair_physics_gate_power=float(m.get("hardpair_physics_gate_power", 1.0)),
        use_hardpair_value_signed_adapter=bool(m.get("use_hardpair_value_signed_adapter", False)),
        hardpair_value_adapter_pairs=m.get("hardpair_value_adapter_pairs"),
        hardpair_value_adapter_pair_scales=m.get("hardpair_value_adapter_pair_scales"),
        hardpair_value_adapter_hidden_dim=int(m.get("hardpair_value_adapter_hidden_dim", 48)),
        hardpair_value_adapter_scale=float(m.get("hardpair_value_adapter_scale", 0.10)),
        hardpair_value_adapter_gate_floor=float(m.get("hardpair_value_adapter_gate_floor", 0.0)),
        hardpair_value_adapter_value_aug_std=float(m.get("hardpair_value_adapter_value_aug_std", 0.0)),
        hardpair_value_adapter_dropout=float(m.get("hardpair_value_adapter_dropout", 0.0)),
        use_hardpair_value_rough_tail_guard=bool(m.get("use_hardpair_value_rough_tail_guard", False)),
        hardpair_value_rough_tail_guard_pairs=m.get("hardpair_value_rough_tail_guard_pairs"),
        hardpair_value_rough_tail_guard_threshold=float(m.get("hardpair_value_rough_tail_guard_threshold", 0.52)),
        hardpair_value_rough_tail_guard_temperature=float(m.get("hardpair_value_rough_tail_guard_temperature", 10.0)),
        hardpair_value_rough_tail_guard_strength=float(m.get("hardpair_value_rough_tail_guard_strength", 0.85)),
        hardpair_focus_classes=m.get("hardpair_focus_classes"),
        hardpair_focus_boundaries=m.get("hardpair_focus_boundaries"),
        hardpair_disabled_class_pairs=m.get("hardpair_disabled_class_pairs"),
        hardpair_pair_scales=m.get("hardpair_pair_scales"),
        hardpair_protected_classes=m.get("hardpair_protected_classes"),
        hardpair_sample_protect_classes=m.get("hardpair_sample_protect_classes"),
        hardpair_sample_protect_threshold=float(m.get("hardpair_sample_protect_threshold", 0.08)),
        hardpair_sample_protect_temperature=float(m.get("hardpair_sample_protect_temperature", 30.0)),
        boundary_use_physics_feature=bool(m.get("boundary_use_physics_feature", False)),
        use_physics_branch=bool(m.get("use_physics_branch", True)),
        physics_dim=int(m.get("physics_dim", 96)),
        physics_quality_cues=bool(m.get("physics_quality_cues", True)),
        physics_quality_region_cues=bool(m.get("physics_quality_region_cues", False)),
        use_semantic_physics_attention_branch=bool(m.get("use_semantic_physics_attention_branch", True)),
        semantic_physics_attention_dim=int(m.get("semantic_physics_attention_dim", 64)),
        use_local_physics_field_branch=bool(m.get("use_local_physics_field_branch", True)),
        local_physics_field_dim=int(m.get("local_physics_field_dim", 64)),
        local_physics_field_scale=float(m.get("local_physics_field_scale", 0.08)),
        use_physics_texture_stem_adapter=bool(m.get("use_physics_texture_stem_adapter", False)),
        physics_texture_stem_hidden_dim=int(m.get("physics_texture_stem_hidden_dim", 32)),
        physics_texture_stem_scale=float(m.get("physics_texture_stem_scale", 0.035)),
        physics_texture_stem_gate_floor=float(m.get("physics_texture_stem_gate_floor", 0.18)),
        use_scale_space_roughness_stem_adapter=bool(m.get("use_scale_space_roughness_stem_adapter", False)),
        scale_space_roughness_stem_hidden_dim=int(m.get("scale_space_roughness_stem_hidden_dim", 32)),
        scale_space_roughness_stem_scale=float(m.get("scale_space_roughness_stem_scale", 0.020)),
        scale_space_roughness_stem_gate_floor=float(m.get("scale_space_roughness_stem_gate_floor", 0.10)),
        scale_space_roughness_stem_gate_mode=str(m.get("scale_space_roughness_stem_gate_mode", "concrete_tail")),
        scale_space_roughness_stem_dry_tail_weight=float(m.get("scale_space_roughness_stem_dry_tail_weight", 1.0)),
        scale_space_roughness_stem_wet_hidden_tail_weight=float(
            m.get("scale_space_roughness_stem_wet_hidden_tail_weight", 1.0)
        ),
        use_pair_value_stem_conditioner=bool(m.get("use_pair_value_stem_conditioner", False)),
        pair_value_stem_hidden_dim=int(m.get("pair_value_stem_hidden_dim", 32)),
        pair_value_stem_scale=float(m.get("pair_value_stem_scale", 0.018)),
        pair_value_stem_gate_floor=float(m.get("pair_value_stem_gate_floor", 0.0)),
        pair_value_stem_value_aug_std=float(m.get("pair_value_stem_value_aug_std", 0.0)),
        pair_value_stem_learned_gate_bias=float(m.get("pair_value_stem_learned_gate_bias", -1.6)),
        use_wet_water_concrete_film_depth_stem_conditioner=bool(
            m.get("use_wet_water_concrete_film_depth_stem_conditioner", False)
        ),
        wet_water_concrete_film_depth_stem_hidden_dim=int(
            m.get("wet_water_concrete_film_depth_stem_hidden_dim", 36)
        ),
        wet_water_concrete_film_depth_stem_scale=float(
            m.get("wet_water_concrete_film_depth_stem_scale", 0.030)
        ),
        wet_water_concrete_film_depth_stem_gate_floor=float(
            m.get("wet_water_concrete_film_depth_stem_gate_floor", 0.04)
        ),
        wet_water_concrete_film_depth_stem_learned_gate_bias=float(
            m.get("wet_water_concrete_film_depth_stem_learned_gate_bias", -1.2)
        ),
        use_water_concrete_topology_texture_stem_conditioner=bool(
            m.get("use_water_concrete_topology_texture_stem_conditioner", False)
        ),
        water_concrete_topology_texture_stem_hidden_dim=int(
            m.get("water_concrete_topology_texture_stem_hidden_dim", 36)
        ),
        water_concrete_topology_texture_stem_scale=float(
            m.get("water_concrete_topology_texture_stem_scale", 0.026)
        ),
        water_concrete_topology_texture_stem_gate_floor=float(
            m.get("water_concrete_topology_texture_stem_gate_floor", 0.03)
        ),
        water_concrete_topology_texture_stem_learned_gate_bias=float(
            m.get("water_concrete_topology_texture_stem_learned_gate_bias", -1.25)
        ),
        use_scale_space_roughness_token_conditioner=bool(
            m.get("use_scale_space_roughness_token_conditioner", False)
        ),
        scale_space_roughness_token_hidden_dim=int(m.get("scale_space_roughness_token_hidden_dim", 64)),
        scale_space_roughness_token_scale=float(m.get("scale_space_roughness_token_scale", 0.10)),
        scale_space_roughness_token_gate_floor=float(m.get("scale_space_roughness_token_gate_floor", 0.0)),
        scale_space_roughness_token_dry_tail_weight=float(
            m.get("scale_space_roughness_token_dry_tail_weight", 1.0)
        ),
        scale_space_roughness_token_wet_hidden_tail_weight=float(
            m.get("scale_space_roughness_token_wet_hidden_tail_weight", 0.75)
        ),
        use_local_global_scale_token_conditioner=bool(
            m.get("use_local_global_scale_token_conditioner", False)
        ),
        local_global_scale_token_hidden_dim=int(m.get("local_global_scale_token_hidden_dim", 96)),
        local_global_scale_token_scale=float(m.get("local_global_scale_token_scale", 0.050)),
        local_global_scale_token_feature_scale=float(m.get("local_global_scale_token_feature_scale", 0.010)),
        local_global_scale_token_gate_floor=float(m.get("local_global_scale_token_gate_floor", 0.0)),
        local_global_scale_token_dropout=float(m.get("local_global_scale_token_dropout", 0.0)),
        local_global_scale_token_detach_context=bool(
            m.get("local_global_scale_token_detach_context", False)
        ),
        use_water_film_roughness_feature_film=bool(
            m.get("use_water_film_roughness_feature_film", False)
        ),
        water_film_roughness_feature_film_hidden_dim=int(
            m.get("water_film_roughness_feature_film_hidden_dim", 128)
        ),
        water_film_roughness_feature_film_scale=float(
            m.get("water_film_roughness_feature_film_scale", 0.080)
        ),
        water_film_roughness_feature_film_gate_floor=float(
            m.get("water_film_roughness_feature_film_gate_floor", 0.0)
        ),
        water_film_roughness_feature_film_max_gamma=float(
            m.get("water_film_roughness_feature_film_max_gamma", 0.18)
        ),
        water_film_roughness_feature_film_dropout=float(
            m.get("water_film_roughness_feature_film_dropout", 0.0)
        ),
        water_film_roughness_feature_film_detach_context=bool(
            m.get("water_film_roughness_feature_film_detach_context", False)
        ),
        use_pseudo_roughness_aware_reliability=bool(
            m.get("use_pseudo_roughness_aware_reliability", False)
        ),
        roughness_reliability_use_coupling_context=bool(
            m.get("roughness_reliability_use_coupling_context", False)
        ),
        pseudo_roughness_aware_reliability_hidden_dim=int(
            m.get("pseudo_roughness_aware_reliability_hidden_dim", 128)
        ),
        pseudo_roughness_aware_reliability_scale=float(
            m.get("pseudo_roughness_aware_reliability_scale", 0.060)
        ),
        pseudo_roughness_aware_reliability_rho_scale=float(
            m.get("pseudo_roughness_aware_reliability_rho_scale", 0.100)
        ),
        pseudo_roughness_aware_reliability_gate_floor=float(
            m.get("pseudo_roughness_aware_reliability_gate_floor", 0.0)
        ),
        pseudo_roughness_aware_reliability_dropout=float(
            m.get("pseudo_roughness_aware_reliability_dropout", 0.0)
        ),
        pseudo_roughness_aware_reliability_detach_context=bool(
            m.get("pseudo_roughness_aware_reliability_detach_context", False)
        ),
        use_spatial_factor_queries=bool(m.get("use_spatial_factor_queries", False)),
        spatial_factor_query_map_dim=int(m.get("spatial_factor_query_map_dim", 768)),
        spatial_factor_query_heads=int(m.get("spatial_factor_query_heads", 4)),
        spatial_factor_query_scale=float(m.get("spatial_factor_query_scale", 0.25)),
        use_dry_concrete_roughness_vor_residual=bool(m.get("use_dry_concrete_roughness_vor_residual", False)),
        dry_concrete_roughness_hidden_dim=int(m.get("dry_concrete_roughness_hidden_dim", 48)),
        dry_concrete_roughness_scale=float(m.get("dry_concrete_roughness_scale", 0.12)),
        dry_concrete_roughness_gate_threshold=float(m.get("dry_concrete_roughness_gate_threshold", 0.12)),
        dry_concrete_roughness_gate_temperature=float(m.get("dry_concrete_roughness_gate_temperature", 14.0)),
        use_dry_concrete_ordinal_chart_residual=bool(m.get("use_dry_concrete_ordinal_chart_residual", False)),
        dry_concrete_ordinal_chart_hidden_dim=int(m.get("dry_concrete_ordinal_chart_hidden_dim", 48)),
        dry_concrete_ordinal_chart_scale=float(m.get("dry_concrete_ordinal_chart_scale", 0.06)),
        dry_concrete_ordinal_chart_gate_threshold=float(m.get("dry_concrete_ordinal_chart_gate_threshold", 0.12)),
        dry_concrete_ordinal_chart_gate_temperature=float(m.get("dry_concrete_ordinal_chart_gate_temperature", 14.0)),
        dry_concrete_ordinal_chart_protect_confidence=float(
            m.get("dry_concrete_ordinal_chart_protect_confidence", 0.72)
        ),
        dry_concrete_ordinal_chart_protect_temperature=float(
            m.get("dry_concrete_ordinal_chart_protect_temperature", 18.0)
        ),
        use_dry_concrete_validation_transition=bool(m.get("use_dry_concrete_validation_transition", False)),
        dry_concrete_validation_transition_source=str(
            m.get("dry_concrete_validation_transition_source", "dry_concrete_severe")
        ),
        dry_concrete_validation_transition_target=str(
            m.get("dry_concrete_validation_transition_target", "dry_concrete_slight")
        ),
        dry_concrete_validation_transition_topk=int(m.get("dry_concrete_validation_transition_topk", 2)),
        dry_concrete_validation_transition_margin=float(m.get("dry_concrete_validation_transition_margin", 0.20)),
        dry_concrete_validation_transition_delta=float(m.get("dry_concrete_validation_transition_delta", 0.20)),
        use_backbone_isolated_dry_concrete_adapter=bool(
            m.get("use_backbone_isolated_dry_concrete_adapter", False)
        ),
        backbone_isolated_dry_concrete_branch_dim=int(
            m.get("backbone_isolated_dry_concrete_branch_dim", 96)
        ),
        backbone_isolated_dry_concrete_hidden_dim=int(
            m.get("backbone_isolated_dry_concrete_hidden_dim", 64)
        ),
        backbone_isolated_dry_concrete_scale=float(
            m.get("backbone_isolated_dry_concrete_scale", 0.18)
        ),
        backbone_isolated_dry_concrete_gate_threshold=float(
            m.get("backbone_isolated_dry_concrete_gate_threshold", 0.10)
        ),
        backbone_isolated_dry_concrete_gate_temperature=float(
            m.get("backbone_isolated_dry_concrete_gate_temperature", 14.0)
        ),
        backbone_isolated_dry_concrete_dropout=float(
            m.get("backbone_isolated_dry_concrete_dropout", 0.02)
        ),
        backbone_isolated_dry_concrete_output_mode=str(
            m.get("backbone_isolated_dry_concrete_output_mode", "free")
        ),
        use_dry_concrete_pair_signed_selector=bool(m.get("use_dry_concrete_pair_signed_selector", False)),
        dry_concrete_pair_selector_pairs=m.get("dry_concrete_pair_selector_pairs"),
        dry_concrete_pair_selector_hidden_dim=int(m.get("dry_concrete_pair_selector_hidden_dim", 48)),
        dry_concrete_pair_selector_shift_scale=float(m.get("dry_concrete_pair_selector_shift_scale", 0.65)),
        dry_concrete_pair_selector_gain_scale=float(m.get("dry_concrete_pair_selector_gain_scale", 0.50)),
        dry_concrete_pair_selector_direct_delta_scale=float(
            m.get("dry_concrete_pair_selector_direct_delta_scale", 0.0)
        ),
        dry_concrete_pair_selector_safe_margin=float(m.get("dry_concrete_pair_selector_safe_margin", 0.20)),
        dry_concrete_pair_selector_safe_temperature=float(
            m.get("dry_concrete_pair_selector_safe_temperature", 28.0)
        ),
        protected_factor_adapter_rank=int(m.get("protected_factor_adapter_rank", 6)),
        protected_factor_adapter_hidden_dim=int(m.get("protected_factor_adapter_hidden_dim", 96)),
        protected_factor_adapter_scale=float(m.get("protected_factor_adapter_scale", 0.08)),
        protected_factor_adapter_gate_margin=float(m.get("protected_factor_adapter_gate_margin", 0.18)),
        protected_factor_adapter_gate_temperature=float(m.get("protected_factor_adapter_gate_temperature", 10.0)),
        protected_factor_adapter_active_classes=m.get("protected_factor_adapter_active_classes"),
        protected_factor_adapter_protected_classes=m.get("protected_factor_adapter_protected_classes"),
        use_feature_value_boundary_corrector=bool(m.get("use_feature_value_boundary_corrector", False)),
        feature_value_boundary_pairs=m.get("feature_value_boundary_pairs"),
        feature_value_boundary_hidden_dim=int(m.get("feature_value_boundary_hidden_dim", 64)),
        feature_value_boundary_scale=float(m.get("feature_value_boundary_scale", 0.22)),
        feature_value_boundary_gate_margin=float(m.get("feature_value_boundary_gate_margin", 1.05)),
        feature_value_boundary_gate_temperature=float(m.get("feature_value_boundary_gate_temperature", 4.5)),
        feature_value_boundary_gate_floor=float(m.get("feature_value_boundary_gate_floor", 0.0)),
        feature_value_boundary_value_aug_std=float(m.get("feature_value_boundary_value_aug_std", 0.0)),
        feature_value_boundary_dropout=float(m.get("feature_value_boundary_dropout", 0.0)),
        feature_value_boundary_severe_tail_protect=bool(
            m.get("feature_value_boundary_severe_tail_protect", False)
        ),
        feature_value_boundary_severe_tail_protect_pairs=m.get(
            "feature_value_boundary_severe_tail_protect_pairs"
        ),
        feature_value_boundary_severe_tail_protect_strength=float(
            m.get("feature_value_boundary_severe_tail_protect_strength", 0.85)
        ),
        feature_value_boundary_severe_tail_protect_prob=float(
            m.get("feature_value_boundary_severe_tail_protect_prob", 0.34)
        ),
        feature_value_boundary_severe_tail_protect_tail_threshold=float(
            m.get("feature_value_boundary_severe_tail_protect_tail_threshold", 0.115)
        ),
        feature_value_boundary_severe_tail_protect_temperature=float(
            m.get("feature_value_boundary_severe_tail_protect_temperature", 16.0)
        ),
        use_water_concrete_opponent_feature_conditioner=bool(
            m.get("use_water_concrete_opponent_feature_conditioner", False)
        ),
        water_concrete_opponent_pairs=m.get("water_concrete_opponent_pairs"),
        water_concrete_opponent_hidden_dim=int(m.get("water_concrete_opponent_hidden_dim", 64)),
        water_concrete_opponent_scale=float(m.get("water_concrete_opponent_scale", 0.018)),
        water_concrete_opponent_gate_margin=float(m.get("water_concrete_opponent_gate_margin", 1.08)),
        water_concrete_opponent_gate_temperature=float(
            m.get("water_concrete_opponent_gate_temperature", 4.5)
        ),
        water_concrete_opponent_gate_floor=float(m.get("water_concrete_opponent_gate_floor", 0.03)),
        water_concrete_opponent_value_aug_std=float(m.get("water_concrete_opponent_value_aug_std", 0.0)),
        water_concrete_opponent_dropout=float(m.get("water_concrete_opponent_dropout", 0.0)),
        use_factor_graph_edge_flow_corrector=bool(m.get("use_factor_graph_edge_flow_corrector", False)),
        factor_graph_edge_flow_pairs=m.get("factor_graph_edge_flow_pairs"),
        factor_graph_edge_flow_hidden_dim=int(m.get("factor_graph_edge_flow_hidden_dim", 64)),
        factor_graph_edge_flow_scale=float(m.get("factor_graph_edge_flow_scale", 0.10)),
        factor_graph_edge_flow_gate_margin=float(m.get("factor_graph_edge_flow_gate_margin", 0.90)),
        factor_graph_edge_flow_gate_temperature=float(m.get("factor_graph_edge_flow_gate_temperature", 4.0)),
        factor_graph_edge_flow_gate_floor=float(m.get("factor_graph_edge_flow_gate_floor", 0.0)),
        factor_graph_edge_flow_confidence_protect=float(m.get("factor_graph_edge_flow_confidence_protect", 0.74)),
        factor_graph_edge_flow_confidence_temperature=float(
            m.get("factor_graph_edge_flow_confidence_temperature", 16.0)
        ),
        factor_graph_edge_flow_dropout=float(m.get("factor_graph_edge_flow_dropout", 0.0)),
        use_tristate_wet_concrete_boundary_expert=bool(
            m.get("use_tristate_wet_concrete_boundary_expert", False)
        ),
        tristate_wet_concrete_boundary_pairs=m.get("tristate_wet_concrete_boundary_pairs"),
        tristate_wet_concrete_boundary_hidden_dim=int(
            m.get("tristate_wet_concrete_boundary_hidden_dim", 64)
        ),
        tristate_wet_concrete_boundary_scale=float(
            m.get("tristate_wet_concrete_boundary_scale", 0.08)
        ),
        tristate_wet_concrete_boundary_gate_margin=float(
            m.get("tristate_wet_concrete_boundary_gate_margin", 0.85)
        ),
        tristate_wet_concrete_boundary_gate_temperature=float(
            m.get("tristate_wet_concrete_boundary_gate_temperature", 5.0)
        ),
        tristate_wet_concrete_boundary_gate_floor=float(
            m.get("tristate_wet_concrete_boundary_gate_floor", 0.0)
        ),
        tristate_wet_concrete_boundary_confidence_protect=float(
            m.get("tristate_wet_concrete_boundary_confidence_protect", 0.78)
        ),
        tristate_wet_concrete_boundary_confidence_temperature=float(
            m.get("tristate_wet_concrete_boundary_confidence_temperature", 16.0)
        ),
        tristate_wet_concrete_boundary_dropout=float(
            m.get("tristate_wet_concrete_boundary_dropout", 0.0)
        ),
        tristate_wet_concrete_boundary_severe_protect=bool(
            m.get("tristate_wet_concrete_boundary_severe_protect", False)
        ),
        tristate_wet_concrete_boundary_severe_protect_prob=float(
            m.get("tristate_wet_concrete_boundary_severe_protect_prob", 0.30)
        ),
        tristate_wet_concrete_boundary_severe_protect_raw_margin=float(
            m.get("tristate_wet_concrete_boundary_severe_protect_raw_margin", 0.0)
        ),
        tristate_wet_concrete_boundary_severe_protect_temperature=float(
            m.get("tristate_wet_concrete_boundary_severe_protect_temperature", 12.0)
        ),
        tristate_wet_concrete_boundary_severe_protect_strength=float(
            m.get("tristate_wet_concrete_boundary_severe_protect_strength", 1.0)
        ),
        use_closed_set_factor_redistributor=bool(m.get("use_closed_set_factor_redistributor", False)),
        closed_set_factor_redistributor_sets=m.get("closed_set_factor_redistributor_sets"),
        closed_set_factor_redistributor_hidden_dim=int(m.get("closed_set_factor_redistributor_hidden_dim", 96)),
        closed_set_factor_redistributor_scale=float(m.get("closed_set_factor_redistributor_scale", 0.06)),
        closed_set_factor_redistributor_gate_floor=float(m.get("closed_set_factor_redistributor_gate_floor", 0.0)),
        closed_set_factor_redistributor_mass_threshold=float(
            m.get("closed_set_factor_redistributor_mass_threshold", 0.08)
        ),
        closed_set_factor_redistributor_margin_threshold=float(
            m.get("closed_set_factor_redistributor_margin_threshold", 0.25)
        ),
        closed_set_factor_redistributor_temperature=float(m.get("closed_set_factor_redistributor_temperature", 8.0)),
        closed_set_factor_redistributor_dropout=float(m.get("closed_set_factor_redistributor_dropout", 0.0)),
        closed_set_factor_redistributor_gate_bias_init=float(
            m.get("closed_set_factor_redistributor_gate_bias_init", -2.5)
        ),
        closed_set_factor_redistributor_use_graph_locality_guard=bool(
            m.get("closed_set_factor_redistributor_use_graph_locality_guard", False)
        ),
        closed_set_factor_redistributor_graph_max_distance=float(
            m.get("closed_set_factor_redistributor_graph_max_distance", 2.0)
        ),
        closed_set_factor_redistributor_graph_guard_floor=float(
            m.get("closed_set_factor_redistributor_graph_guard_floor", 0.0)
        ),
        closed_set_factor_redistributor_graph_guard_temperature=float(
            m.get("closed_set_factor_redistributor_graph_guard_temperature", 12.0)
        ),
        use_backbone_family_ordinal_no_spill_adapter=bool(
            m.get("use_backbone_family_ordinal_no_spill_adapter", False)
        ),
        backbone_family_ordinal_no_spill_hidden_dim=int(
            m.get("backbone_family_ordinal_no_spill_hidden_dim", 96)
        ),
        backbone_family_ordinal_no_spill_family_embed_dim=int(
            m.get("backbone_family_ordinal_no_spill_family_embed_dim", 12)
        ),
        backbone_family_ordinal_no_spill_scale=float(
            m.get("backbone_family_ordinal_no_spill_scale", 0.18)
        ),
        backbone_family_ordinal_no_spill_gate_threshold=float(
            m.get("backbone_family_ordinal_no_spill_gate_threshold", 0.055)
        ),
        backbone_family_ordinal_no_spill_gate_temperature=float(
            m.get("backbone_family_ordinal_no_spill_gate_temperature", 10.0)
        ),
        backbone_family_ordinal_no_spill_dropout=float(
            m.get("backbone_family_ordinal_no_spill_dropout", 0.02)
        ),
        backbone_family_ordinal_no_spill_families=m.get("backbone_family_ordinal_no_spill_families"),
        use_pair_value_mechanism_conditioner=bool(m.get("use_pair_value_mechanism_conditioner", False)),
        pair_value_mechanism_hidden_dim=int(m.get("pair_value_mechanism_hidden_dim", 64)),
        pair_value_mechanism_feature_scale=float(m.get("pair_value_mechanism_feature_scale", 0.010)),
        pair_value_mechanism_token_scale=float(m.get("pair_value_mechanism_token_scale", 0.060)),
        pair_value_mechanism_gate_floor=float(m.get("pair_value_mechanism_gate_floor", 0.0)),
        pair_value_mechanism_value_aug_std=float(m.get("pair_value_mechanism_value_aug_std", 0.0)),
        pair_value_mechanism_protect_classes=m.get("pair_value_mechanism_protect_classes"),
        pair_value_mechanism_protect_threshold=float(m.get("pair_value_mechanism_protect_threshold", 0.18)),
        pair_value_mechanism_protect_temperature=float(m.get("pair_value_mechanism_protect_temperature", 24.0)),
        use_coupled_form_expert_conditioner=bool(m.get("use_coupled_form_expert_conditioner", False)),
        coupled_form_expert_hidden_dim=int(m.get("coupled_form_expert_hidden_dim", 64)),
        coupled_form_expert_feature_scale=float(m.get("coupled_form_expert_feature_scale", 0.010)),
        coupled_form_expert_token_scale=float(m.get("coupled_form_expert_token_scale", 0.060)),
        coupled_form_expert_gate_floor=float(m.get("coupled_form_expert_gate_floor", 0.0)),
        coupled_form_expert_value_aug_std=float(m.get("coupled_form_expert_value_aug_std", 0.0)),
        coupled_form_expert_learned_gate_bias=float(m.get("coupled_form_expert_learned_gate_bias", -1.5)),
        coupled_form_expert_detach_context=bool(m.get("coupled_form_expert_detach_context", False)),
        coupled_form_expert_protect_classes=m.get("coupled_form_expert_protect_classes"),
        coupled_form_expert_protect_threshold=float(m.get("coupled_form_expert_protect_threshold", 0.18)),
        coupled_form_expert_protect_temperature=float(m.get("coupled_form_expert_protect_temperature", 24.0)),
        use_pareto_edge_expert=bool(m.get("use_pareto_edge_expert", False)),
        pareto_edge_expert_rules=pareto_edge_expert_rules,
        pareto_edge_expert_hidden_dim=int(m.get("pareto_edge_expert_hidden_dim", 48)),
        pareto_edge_expert_scale=float(m.get("pareto_edge_expert_scale", 1.0)),
        pareto_edge_expert_gate_temperature=float(m.get("pareto_edge_expert_gate_temperature", 18.0)),
        pareto_edge_expert_gate_floor=float(m.get("pareto_edge_expert_gate_floor", 0.0)),
        pareto_edge_expert_learned_gate_bias=float(m.get("pareto_edge_expert_learned_gate_bias", -1.6)),
        pareto_edge_expert_dropout=float(m.get("pareto_edge_expert_dropout", 0.0)),
        use_source_reliable_boundary_router=bool(m.get("use_source_reliable_boundary_router", False)),
        source_reliable_boundary_routes=m.get("source_reliable_boundary_routes"),
        source_reliable_boundary_hidden_dim=int(m.get("source_reliable_boundary_hidden_dim", 32)),
        source_reliable_boundary_scale=float(m.get("source_reliable_boundary_scale", 0.012)),
        source_reliable_boundary_gate_temperature=float(m.get("source_reliable_boundary_gate_temperature", 6.0)),
        source_reliable_boundary_physics_gate_floor=float(
            m.get("source_reliable_boundary_physics_gate_floor", 0.0)
        ),
        source_reliable_boundary_base_strength=float(m.get("source_reliable_boundary_base_strength", 0.0)),
        source_reliable_boundary_source_temperature=float(
            m.get("source_reliable_boundary_source_temperature", 28.0)
        ),
        source_reliable_boundary_learned_gate_bias=float(
            m.get("source_reliable_boundary_learned_gate_bias", -2.0)
        ),
        source_reliable_boundary_dropout=float(m.get("source_reliable_boundary_dropout", 0.0)),
        expose_hardpair_pair_value_evidence=bool(m.get("expose_hardpair_pair_value_evidence", False)),
    )


def flexible_load(
    model: nn.Module,
    checkpoint: str | None,
    *,
    skip_prefixes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not checkpoint:
        return {"loaded": 0, "skipped": 0, "path": None}
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    skip_prefixes = tuple(str(prefix) for prefix in (skip_prefixes or ()))
    state = torch.load(path, map_location="cpu", weights_only=False)
    raw = state.get("model", state.get("state_dict", state))
    target = model.state_dict()
    loadable = {}
    skipped = []
    aliases = {
        "classifier.weight": "linear_head.weight",
        "classifier.bias": "linear_head.bias",
        "backbone.proj.weight": "backbone.global_proj.weight",
        "backbone.proj.bias": "backbone.global_proj.bias",
    }
    for name, tensor in raw.items():
        if skip_prefixes and any(name.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(name)
            continue
        if name.endswith(("cell_mask", "chart_mask", "active_mask")):
            skipped.append(name)
            continue
        load_name = name
        if load_name in target and tuple(target[load_name].shape) == tuple(tensor.shape):
            loadable[load_name] = tensor
            continue
        alias_name = aliases.get(name)
        if alias_name and alias_name in target and tuple(target[alias_name].shape) == tuple(tensor.shape):
            loadable[alias_name] = tensor
        else:
            skipped.append(name)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print(f"Loaded flexible checkpoint: {path}")
    if skip_prefixes:
        print(f"  skip_prefixes={list(skip_prefixes)}")
    print(f"  loaded={len(loadable)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
    return {"loaded": len(loadable), "skipped": len(skipped), "missing": len(missing), "path": str(path)}


def apply_trainable_prefixes(model: nn.Module, prefixes: list[str] | None) -> None:
    if not prefixes:
        return
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    matched = {prefix: 0 for prefix in prefixes}
    for name, param in model.named_parameters():
        for prefix in prefixes:
            if name.startswith(prefix):
                param.requires_grad_(True)
                matched[prefix] += int(param.numel())
    empty = [prefix for prefix, count in matched.items() if count == 0]
    if empty:
        raise ValueError(f"trainable prefixes matched no parameters: {empty}")
    print("Trainable prefixes:", matched)


def build_anchor_teacher(
    cfg: dict[str, Any],
    class_to_idx: dict[str, int],
    device: torch.device,
) -> C3FaRNetSurfaceClassifier | None:
    train_cfg = cfg["train"]
    checkpoint = train_cfg.get("teacher_checkpoint")
    if not checkpoint:
        return None
    teacher_cfg = copy.deepcopy(cfg)
    teacher_model_cfg = teacher_cfg["model"]
    teacher_model_cfg["backbone"] = str(train_cfg.get("teacher_backbone", "convnext_tiny"))
    teacher_model_cfg["head_type"] = str(train_cfg.get("teacher_head_type", "linear"))
    teacher_overrides = train_cfg.get("teacher_model_overrides")
    if isinstance(teacher_overrides, dict):
        teacher_cfg["model"] = _deep_update_config(teacher_model_cfg, teacher_overrides)
        teacher_model_cfg = teacher_cfg["model"]
    if not bool(train_cfg.get("teacher_preserve_hardpair_focus", False)):
        teacher_model_cfg["hardpair_focus_classes"] = []
        teacher_model_cfg["hardpair_focus_boundaries"] = []
    teacher = build_model(teacher_cfg, class_to_idx).to(device)
    flexible_load(teacher, str(checkpoint))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print(f"Frozen anchor teacher enabled: {checkpoint}")
    return teacher


def build_specialist_teacher(
    cfg: dict[str, Any],
    class_to_idx: dict[str, int],
    device: torch.device,
) -> C3FaRNetSurfaceClassifier | None:
    train_cfg = cfg["train"]
    checkpoint = train_cfg.get("expert_teacher_checkpoint")
    if not checkpoint:
        return None
    teacher_cfg = copy.deepcopy(cfg)
    teacher_model_cfg = teacher_cfg["model"]
    teacher_model_cfg["backbone"] = str(
        train_cfg.get("expert_teacher_backbone", train_cfg.get("teacher_backbone", teacher_model_cfg.get("backbone", "convnext_tiny")))
    )
    teacher_model_cfg["head_type"] = str(
        train_cfg.get("expert_teacher_head_type", train_cfg.get("teacher_head_type", teacher_model_cfg.get("head_type", "linear")))
    )
    expert_teacher_overrides = train_cfg.get("expert_teacher_model_overrides", train_cfg.get("teacher_model_overrides"))
    if isinstance(expert_teacher_overrides, dict):
        teacher_cfg["model"] = _deep_update_config(teacher_model_cfg, expert_teacher_overrides)
        teacher_model_cfg = teacher_cfg["model"]
    teacher = build_model(teacher_cfg, class_to_idx).to(device)
    flexible_load(teacher, str(checkpoint))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print(f"Frozen specialist teacher enabled: {checkpoint}")
    return teacher


def teacher_cache_key(path: str | os.PathLike[str]) -> str:
    """Stable key for cached teacher logits indexed by image path."""

    return os.path.normcase(os.path.abspath(str(path)))


def load_teacher_logit_cache(path: str | os.PathLike[str] | None) -> dict[str, torch.Tensor] | None:
    """Load an image-path -> logits cache created by `scripts/cache_teacher_logits.py`."""

    if not path:
        return None
    cache_path = Path(str(path))
    if not cache_path.exists():
        raise FileNotFoundError(f"teacher logits cache does not exist: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    image_paths = payload.get("image_paths")
    logits = payload.get("logits")
    if image_paths is None or logits is None:
        raise ValueError(f"teacher logits cache must contain image_paths and logits: {cache_path}")
    if len(image_paths) != int(logits.shape[0]):
        raise ValueError(
            f"teacher logits cache length mismatch: paths={len(image_paths)} logits={tuple(logits.shape)} at {cache_path}"
        )
    logits = logits.float().cpu()
    cache = {teacher_cache_key(path_text): logits[int(i)] for i, path_text in enumerate(image_paths)}
    print(f"Loaded teacher logits cache: {cache_path} ({len(cache)} images)")
    return cache


def cached_teacher_logits_for_batch(
    cache: dict[str, torch.Tensor] | None,
    image_paths: list[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool,
    cache_name: str,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if cache is None:
        return None, {}
    rows: list[torch.Tensor] = []
    missing: list[str] = []
    for path_text in image_paths:
        row = cache.get(teacher_cache_key(path_text))
        if row is None:
            missing.append(str(path_text))
        else:
            rows.append(row)
    if missing:
        if strict:
            preview = ", ".join(missing[:3])
            raise KeyError(f"{cache_name} teacher logits cache missing {len(missing)} paths, first: {preview}")
        return None, {f"{cache_name}_teacher_cache_miss": float(len(missing))}
    logits = torch.stack(rows).to(device=device, dtype=dtype, non_blocking=True)
    return logits, {f"{cache_name}_teacher_cache_hit": float(len(rows))}


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def anchor_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = float(loss_cfg.get("anchor_consistency_weight", 0.0))
    focus_weight = float(loss_cfg.get("anchor_consistency_focus_weight", 0.0))
    if weight <= 0.0 and focus_weight <= 0.0:
        return student_logits.new_zeros(()), {"loss_anchor_consistency": 0.0, "anchor_consistency_count": 0.0}
    temperature = max(float(loss_cfg.get("anchor_consistency_temperature", 2.0)), 1e-3)
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        s = student_logits.float() / temperature
        t = teacher_logits.float() / temperature
        per_sample = F.kl_div(
            F.log_softmax(s, dim=1),
            F.softmax(t, dim=1),
            reduction="none",
        ).sum(dim=1) * (temperature * temperature)
        teacher_prob = F.softmax(teacher_logits.float(), dim=1)
        teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
        teacher_conf = teacher_top2.values[:, 0]
        if teacher_top2.values.size(1) > 1:
            teacher_margin = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_conf)
        teacher_pred = teacher_top2.indices[:, 0]
        teacher_correct = teacher_pred.eq(labels)
    focus_classes = {canonical_class_label(name) for name in loss_cfg.get("anchor_consistency_exempt_classes", [])}
    if focus_classes:
        focus_idx = {
            int(idx)
            for idx, name in idx_to_class.items()
            if canonical_class_label(name) in focus_classes
        }
        focus_mask = torch.zeros_like(labels, dtype=torch.bool)
        for idx in focus_idx:
            focus_mask |= labels.eq(int(idx))
    else:
        focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    nonfocus_mask = ~focus_mask
    terms: list[torch.Tensor] = []
    weighted_count = 0.0
    if weight > 0.0 and bool(nonfocus_mask.any()):
        terms.append(float(weight) * per_sample[nonfocus_mask].mean())
        weighted_count += float(nonfocus_mask.sum().detach().cpu())
    if focus_weight > 0.0 and bool(focus_mask.any()):
        focus_loss_mask = focus_mask
        low_margin_threshold = float(loss_cfg.get("anchor_consistency_focus_low_margin_threshold", -1.0))
        if low_margin_threshold >= 0.0:
            focus_loss_mask = focus_loss_mask & teacher_margin.le(low_margin_threshold)
        if bool(focus_loss_mask.any()):
            terms.append(float(focus_weight) * per_sample[focus_loss_mask].mean())
            weighted_count += float(focus_loss_mask.sum().detach().cpu())
    protect_weight = float(loss_cfg.get("anchor_consistency_protect_weight", 0.0))
    protect_conf = float(loss_cfg.get("anchor_consistency_protect_confidence", 0.0))
    protect_margin = float(loss_cfg.get("anchor_consistency_protect_margin", 0.0))
    protect_mask = teacher_correct
    if protect_conf > 0.0:
        protect_mask = protect_mask & teacher_conf.ge(protect_conf)
    if protect_margin > 0.0:
        protect_mask = protect_mask & teacher_margin.ge(protect_margin)
    if protect_weight > 0.0 and bool(protect_mask.any()):
        terms.append(float(protect_weight) * per_sample[protect_mask].mean())
        weighted_count += float(protect_mask.sum().detach().cpu())
    no_flip_weight = float(loss_cfg.get("anchor_no_flip_weight", 0.0))
    no_flip_mask = protect_mask
    if bool(loss_cfg.get("anchor_no_flip_nonfocus_only", True)):
        no_flip_mask = no_flip_mask & nonfocus_mask
    if no_flip_weight > 0.0 and bool(no_flip_mask.any()):
        no_flip = F.cross_entropy(student_logits.float().index_select(0, no_flip_mask.nonzero(as_tuple=False).flatten()), teacher_pred.index_select(0, no_flip_mask.nonzero(as_tuple=False).flatten()))
        terms.append(float(no_flip_weight) * no_flip.to(dtype=student_logits.dtype))
    if not terms:
        return student_logits.new_zeros(()), {"loss_anchor_consistency": 0.0, "anchor_consistency_count": 0.0}
    loss = torch.stack(terms).sum().to(dtype=student_logits.dtype)
    return loss, {
        "loss_anchor_consistency": float(loss.detach().cpu()),
        "anchor_consistency_count": weighted_count,
        "anchor_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
        "anchor_teacher_margin_mean": float(teacher_margin.detach().mean().cpu()),
        "anchor_protect_count": float(protect_mask.sum().detach().cpu()),
        "anchor_no_flip_count": float(no_flip_mask.sum().detach().cpu()),
    }


def anchor_nonregression_barrier_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Class-balanced anchor barrier for RSCD negative-transfer control.

    This is a task-adapted GEM/LwF-style constraint. Focused RSCD mechanisms may
    update weak concrete/water boundaries, but samples that the anchor teacher
    already classifies correctly should not receive a higher true-label CE loss
    beyond a small tolerance. The penalty is averaged per class first, so large
    asphalt/snow groups cannot hide regressions in smaller hard classes.
    """

    weight = float(loss_cfg.get("anchor_nonregression_weight", 0.0))
    focus_weight = float(loss_cfg.get("anchor_nonregression_focus_weight", weight))
    if weight <= 0.0 and focus_weight <= 0.0:
        return student_logits.new_zeros(()), {
            "loss_anchor_nonregression": 0.0,
            "anchor_nonregression_count": 0.0,
        }
    tolerance = max(float(loss_cfg.get("anchor_nonregression_margin", 0.0)), 0.0)
    protect_conf = float(loss_cfg.get("anchor_nonregression_confidence", 0.0))
    protect_margin = float(loss_cfg.get("anchor_nonregression_teacher_margin", 0.0))
    use_squared = bool(loss_cfg.get("anchor_nonregression_squared", True))
    focus_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("anchor_nonregression_focus_classes", loss_cfg.get("focus_ce_classes", []))
    }
    focus_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in focus_classes
    }
    focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in focus_idx:
        focus_mask |= labels.eq(int(idx))
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        student_ce = F.cross_entropy(student_logits.float(), labels, reduction="none")
        teacher_ce = F.cross_entropy(teacher_logits.float(), labels, reduction="none")
        teacher_prob = F.softmax(teacher_logits.float(), dim=1)
        teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
        teacher_conf = teacher_top2.values[:, 0]
        if teacher_top2.values.size(1) > 1:
            teacher_gap = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
        else:
            teacher_gap = torch.ones_like(teacher_conf)
        teacher_pred = teacher_top2.indices[:, 0]
        protect_mask = teacher_pred.eq(labels)
        if protect_conf > 0.0:
            protect_mask = protect_mask & teacher_conf.ge(protect_conf)
        if protect_margin > 0.0:
            protect_mask = protect_mask & teacher_gap.ge(protect_margin)
        excess = F.relu(student_ce - teacher_ce.detach() - tolerance)
        if use_squared:
            excess = excess.pow(2)
    terms: list[torch.Tensor] = []
    weighted_count = 0.0
    active_classes = 0
    active_excess_sum = 0.0
    for class_idx in labels.detach().unique().tolist():
        class_mask = labels.eq(int(class_idx)) & protect_mask
        if not bool(class_mask.any()):
            continue
        class_is_focus = int(class_idx) in focus_idx
        class_weight = focus_weight if class_is_focus else weight
        if class_weight <= 0.0:
            continue
        class_loss = excess[class_mask].mean()
        terms.append(float(class_weight) * class_loss)
        active_classes += 1
        weighted_count += float(class_mask.sum().detach().cpu())
        active_excess_sum += float(excess[class_mask].detach().mean().cpu())
    if not terms:
        return student_logits.new_zeros(()), {
            "loss_anchor_nonregression": 0.0,
            "anchor_nonregression_count": 0.0,
            "anchor_nonregression_active_classes": 0.0,
        }
    loss = torch.stack(terms).mean().to(dtype=student_logits.dtype)
    return loss, {
        "loss_anchor_nonregression": float(loss.detach().cpu()),
        "anchor_nonregression_count": weighted_count,
        "anchor_nonregression_active_classes": float(active_classes),
        "anchor_nonregression_excess_mean": active_excess_sum / max(active_classes, 1),
        "anchor_nonregression_protect_rate": float(protect_mask.float().detach().mean().cpu()),
        "anchor_nonregression_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
        "anchor_nonregression_teacher_margin_mean": float(teacher_gap.detach().mean().cpu()),
    }


def pareto_safe_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Classwise no-harm distillation for RSCD hard-boundary fine tuning.

    This loss is a task-adapted LwF/GEM-style guard. It lets the new RSCD
    hard-class mechanism optimize weak wet/water/concrete boundaries, but it
    keeps teacher-correct samples from losing either their full probability
    distribution or their true-vs-nearest-boundary margin. The averaging is
    class-balanced so a gain on one large class cannot hide regressions on a
    smaller protected class.
    """

    weight = float(loss_cfg.get("pareto_safe_distill_weight", 0.0))
    margin_weight = float(loss_cfg.get("pareto_safe_margin_weight", 0.0))
    hardpair_weight = float(loss_cfg.get("pareto_safe_hardpair_margin_weight", 0.0))
    if weight <= 0.0 and margin_weight <= 0.0 and hardpair_weight <= 0.0:
        return student_logits.new_zeros(()), {
            "loss_pareto_safe_distill": 0.0,
            "pareto_safe_protect_count": 0.0,
            "pareto_safe_hardpair_count": 0.0,
        }

    temperature = max(float(loss_cfg.get("pareto_safe_temperature", 2.0)), 1e-3)
    protect_conf = float(loss_cfg.get("pareto_safe_confidence", 0.0))
    protect_margin = float(loss_cfg.get("pareto_safe_teacher_margin", 0.0))
    tolerance = max(float(loss_cfg.get("pareto_safe_margin_tolerance", 0.0)), 0.0)
    focus_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("pareto_safe_focus_classes", loss_cfg.get("focus_ce_classes", []))
    }
    protected_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("pareto_safe_protected_classes", [])
    }
    exempt_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("pareto_safe_exempt_classes", [])
    }
    focus_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in focus_classes
    }
    protected_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in protected_classes
    }
    exempt_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in exempt_classes
    }

    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        s_logits = student_logits.float()
        t_logits = teacher_logits.float()
        teacher_prob = F.softmax(t_logits, dim=1)
        teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
        teacher_pred = teacher_top2.indices[:, 0]
        teacher_conf = teacher_top2.values[:, 0]
        if teacher_top2.values.size(1) > 1:
            teacher_gap = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
        else:
            teacher_gap = torch.ones_like(teacher_conf)
        protect_mask = teacher_pred.eq(labels)
        if protect_conf > 0.0:
            protect_mask = protect_mask & teacher_conf.ge(protect_conf)
        if protect_margin > 0.0:
            protect_mask = protect_mask & teacher_gap.ge(protect_margin)
        if protected_idx:
            class_mask = torch.zeros_like(labels, dtype=torch.bool)
            for idx in protected_idx:
                class_mask |= labels.eq(int(idx))
            protect_mask = protect_mask & class_mask
        if exempt_idx:
            exempt_mask = torch.zeros_like(labels, dtype=torch.bool)
            for idx in exempt_idx:
                exempt_mask |= labels.eq(int(idx))
            protect_mask = protect_mask & ~exempt_mask

        focus_mask = torch.zeros_like(labels, dtype=torch.bool)
        for idx in focus_idx:
            focus_mask |= labels.eq(int(idx))
        focus_protect_scale = float(loss_cfg.get("pareto_safe_focus_protect_scale", 0.25))
        focus_protect_scale = min(max(focus_protect_scale, 0.0), 1.0)

        kl_values = F.kl_div(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1),
            reduction="none",
        ).sum(dim=1).clamp_min(0.0) * (temperature * temperature)

        true_student = s_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        true_teacher = t_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        inf = torch.finfo(s_logits.dtype).max
        one_hot = F.one_hot(labels, num_classes=s_logits.size(1)).bool()
        student_other = s_logits.masked_fill(one_hot, -inf).max(dim=1).values
        teacher_other = t_logits.masked_fill(one_hot, -inf).max(dim=1).values
        student_true_margin = true_student - student_other
        teacher_true_margin = true_teacher - teacher_other
        margin_barrier = F.relu(teacher_true_margin.detach() - tolerance - student_true_margin)
        if bool(loss_cfg.get("pareto_safe_margin_squared", True)):
            margin_barrier = margin_barrier.pow(2)

    terms: list[torch.Tensor] = []
    active_classes = 0
    protect_count = 0.0
    kl_sum = 0.0
    margin_sum = 0.0
    for class_idx in labels.detach().unique().tolist():
        class_idx = int(class_idx)
        class_mask = labels.eq(class_idx) & protect_mask
        if not bool(class_mask.any()):
            continue
        class_scale = focus_protect_scale if class_idx in focus_idx else 1.0
        if class_scale <= 0.0:
            continue
        class_terms: list[torch.Tensor] = []
        if weight > 0.0:
            class_kl = kl_values[class_mask].mean()
            class_terms.append(float(weight) * class_kl)
            kl_sum += float(class_kl.detach().cpu())
        if margin_weight > 0.0:
            class_margin = margin_barrier[class_mask].mean()
            class_terms.append(float(margin_weight) * class_margin)
            margin_sum += float(class_margin.detach().cpu())
        if class_terms:
            terms.append(float(class_scale) * torch.stack(class_terms).sum())
            active_classes += 1
            protect_count += float(class_mask.sum().detach().cpu())

    hardpair_terms: list[torch.Tensor] = []
    hardpair_count = 0
    hardpair_violation_sum = 0.0
    if hardpair_weight > 0.0:
        idx_to_name = {int(idx): canonical_class_label(name) for idx, name in idx_to_class.items()}
        requested_pairs: set[frozenset[str]] = set()
        for item in loss_cfg.get("pareto_safe_hardpair_pairs", []):
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) == 2:
                requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            if requested_pairs:
                names = frozenset((idx_to_name.get(left, str(left)), idx_to_name.get(right, str(right))))
                if names not in requested_pairs:
                    continue
            pair_mask = (labels.eq(left) | labels.eq(right)) & protect_mask
            if not bool(pair_mask.any()):
                continue
            sign = torch.where(labels.eq(left), 1.0, -1.0).to(device=s_logits.device, dtype=s_logits.dtype)
            teacher_pair_margin = sign * (t_logits[:, left] - t_logits[:, right])
            student_pair_margin = sign * (s_logits[:, left] - s_logits[:, right])
            violation = F.relu(teacher_pair_margin.detach() - tolerance - student_pair_margin)
            if bool(loss_cfg.get("pareto_safe_margin_squared", True)):
                violation = violation.pow(2)
            selected = violation[pair_mask]
            if bool(selected.numel()):
                hardpair_terms.append(selected.mean())
                hardpair_count += int(pair_mask.sum().detach().cpu())
                hardpair_violation_sum += float(selected.detach().mean().cpu())

    if hardpair_terms:
        terms.append(float(hardpair_weight) * torch.stack(hardpair_terms).mean())
    if not terms:
        return student_logits.new_zeros(()), {
            "loss_pareto_safe_distill": 0.0,
            "pareto_safe_protect_count": 0.0,
            "pareto_safe_hardpair_count": 0.0,
            "pareto_safe_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
            "pareto_safe_teacher_margin_mean": float(teacher_gap.detach().mean().cpu()),
        }
    loss = torch.stack(terms).mean().to(dtype=student_logits.dtype)
    return loss, {
        "loss_pareto_safe_distill": float(loss.detach().cpu()),
        "pareto_safe_protect_count": protect_count,
        "pareto_safe_active_classes": float(active_classes),
        "pareto_safe_hardpair_count": float(hardpair_count),
        "pareto_safe_kl_mean": kl_sum / max(active_classes, 1),
        "pareto_safe_margin_excess_mean": margin_sum / max(active_classes, 1),
        "pareto_safe_hardpair_excess_mean": hardpair_violation_sum / max(len(hardpair_terms), 1),
        "pareto_safe_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
        "pareto_safe_teacher_margin_mean": float(teacher_gap.detach().mean().cpu()),
        "pareto_safe_focus_count": float(focus_mask.sum().detach().cpu()),
    }


def dual_teacher_noharm_loss(
    student_logits: torch.Tensor,
    anchor_logits: torch.Tensor | None,
    expert_logits: torch.Tensor | None,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single-student distillation that separates weak-class repair from no-harm protection.

    RSCD improvements often help wet/water/concrete/slight boundaries while
    hurting already-stable classes. This loss routes each training sample to one
    frozen teacher: a specialist teacher for explicitly improved weak classes,
    and the anchor teacher for protected or non-focus classes. Both routes are
    class-balanced and gated by teacher correctness/confidence, so a local repair
    cannot silently buy gains by damaging another RSCD composite class.
    """

    total_weight = float(loss_cfg.get("dual_teacher_noharm_weight", 0.0))
    expert_weight = float(loss_cfg.get("dual_teacher_expert_weight", 1.0))
    anchor_weight = float(loss_cfg.get("dual_teacher_anchor_weight", 1.0))
    if total_weight <= 0.0 or (expert_logits is None and anchor_logits is None):
        return student_logits.new_zeros(()), {
            "loss_dual_teacher_noharm": 0.0,
            "dual_teacher_expert_count": 0.0,
            "dual_teacher_anchor_count": 0.0,
        }

    temperature = max(float(loss_cfg.get("dual_teacher_temperature", 2.0)), 1e-3)
    expert_conf = float(loss_cfg.get("dual_teacher_expert_confidence", 0.0))
    expert_margin = float(loss_cfg.get("dual_teacher_expert_margin", 0.0))
    anchor_conf = float(loss_cfg.get("dual_teacher_anchor_confidence", 0.0))
    anchor_margin = float(loss_cfg.get("dual_teacher_anchor_margin", 0.0))
    margin_weight = float(loss_cfg.get("dual_teacher_margin_weight", 0.0))
    margin_tolerance = max(float(loss_cfg.get("dual_teacher_margin_tolerance", 0.0)), 0.0)
    require_teacher_correct = bool(loss_cfg.get("dual_teacher_require_teacher_correct", True))
    expert_requires_advantage = bool(loss_cfg.get("dual_teacher_expert_requires_anchor_advantage", True))
    advantage_margin = float(loss_cfg.get("dual_teacher_expert_advantage_margin", 0.0))
    protect_nonexpert = bool(loss_cfg.get("dual_teacher_anchor_protect_nonexpert", True))
    protect_expert_classes = bool(loss_cfg.get("dual_teacher_anchor_protect_expert_classes", False))

    expert_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("dual_teacher_expert_classes", loss_cfg.get("focus_ce_classes", []))
    }
    protected_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("dual_teacher_protected_classes", [])
    }
    expert_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in expert_classes
    }
    protected_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in protected_classes
    }

    def _class_mask(class_indices: set[int]) -> torch.Tensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_idx in class_indices:
            mask |= labels.eq(int(class_idx))
        return mask

    def _teacher_stats(teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prob = F.softmax(teacher.float(), dim=1)
        top2 = prob.topk(k=min(2, prob.size(1)), dim=1)
        pred = top2.indices[:, 0]
        conf = top2.values[:, 0]
        if top2.values.size(1) > 1:
            gap = top2.values[:, 0] - top2.values[:, 1]
        else:
            gap = torch.ones_like(conf)
        true_logit = teacher.float().gather(1, labels.view(-1, 1)).squeeze(1)
        one_hot = F.one_hot(labels, num_classes=teacher.size(1)).bool()
        other_logit = teacher.float().masked_fill(one_hot, -torch.finfo(teacher.float().dtype).max).max(dim=1).values
        true_margin = true_logit - other_logit
        return pred, conf, gap, true_margin

    def _distill_values(teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_logits = student_logits.float()
        t_logits = teacher.float()
        kl_values = F.kl_div(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1),
            reduction="none",
        ).sum(dim=1).clamp_min(0.0) * (temperature * temperature)
        true_student = s_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        true_teacher = t_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        one_hot = F.one_hot(labels, num_classes=s_logits.size(1)).bool()
        student_other = s_logits.masked_fill(one_hot, -torch.finfo(s_logits.dtype).max).max(dim=1).values
        teacher_other = t_logits.masked_fill(one_hot, -torch.finfo(t_logits.dtype).max).max(dim=1).values
        margin_barrier = F.relu((true_teacher - teacher_other).detach() - margin_tolerance - (true_student - student_other))
        if bool(loss_cfg.get("dual_teacher_margin_squared", True)):
            margin_barrier = margin_barrier.pow(2)
        return kl_values, margin_barrier

    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        expert_mask = _class_mask(expert_idx) if expert_idx else torch.zeros_like(labels, dtype=torch.bool)
        anchor_mask = torch.zeros_like(labels, dtype=torch.bool)
        if protected_idx:
            anchor_mask |= _class_mask(protected_idx)
        if protect_nonexpert:
            anchor_mask |= ~expert_mask
        if not protect_expert_classes:
            anchor_mask &= ~expert_mask

        expert_count = 0.0
        anchor_count = 0.0
        expert_active_classes = 0
        anchor_active_classes = 0
        expert_terms: list[torch.Tensor] = []
        anchor_terms: list[torch.Tensor] = []
        expert_conf_mean = 0.0
        anchor_conf_mean = 0.0

        anchor_true_margin: torch.Tensor | None = None
        if anchor_logits is not None:
            anchor_pred, anchor_teacher_conf, anchor_gap, anchor_true_margin = _teacher_stats(anchor_logits)
            if require_teacher_correct:
                anchor_mask &= anchor_pred.eq(labels)
            if anchor_conf > 0.0:
                anchor_mask &= anchor_teacher_conf.ge(anchor_conf)
            if anchor_margin > 0.0:
                anchor_mask &= anchor_gap.ge(anchor_margin)
            anchor_conf_mean = float(anchor_teacher_conf.detach().mean().cpu())
            anchor_kl, anchor_barrier = _distill_values(anchor_logits)
            for class_idx in labels.detach().unique().tolist():
                class_mask = labels.eq(int(class_idx)) & anchor_mask
                if not bool(class_mask.any()):
                    continue
                class_terms = [anchor_kl[class_mask].mean()]
                if margin_weight > 0.0:
                    class_terms.append(float(margin_weight) * anchor_barrier[class_mask].mean())
                anchor_terms.append(torch.stack(class_terms).sum())
                anchor_count += float(class_mask.sum().detach().cpu())
                anchor_active_classes += 1

        if expert_logits is not None:
            expert_pred, expert_teacher_conf, expert_gap, expert_true_margin = _teacher_stats(expert_logits)
            if require_teacher_correct:
                expert_mask &= expert_pred.eq(labels)
            if expert_conf > 0.0:
                expert_mask &= expert_teacher_conf.ge(expert_conf)
            if expert_margin > 0.0:
                expert_mask &= expert_gap.ge(expert_margin)
            if expert_requires_advantage and anchor_true_margin is not None:
                expert_mask &= expert_true_margin.ge(anchor_true_margin + advantage_margin)
            expert_conf_mean = float(expert_teacher_conf.detach().mean().cpu())
            expert_kl, expert_barrier = _distill_values(expert_logits)
            for class_idx in labels.detach().unique().tolist():
                class_mask = labels.eq(int(class_idx)) & expert_mask
                if not bool(class_mask.any()):
                    continue
                class_terms = [expert_kl[class_mask].mean()]
                if margin_weight > 0.0:
                    class_terms.append(float(margin_weight) * expert_barrier[class_mask].mean())
                expert_terms.append(torch.stack(class_terms).sum())
                expert_count += float(class_mask.sum().detach().cpu())
                expert_active_classes += 1

        terms: list[torch.Tensor] = []
        if expert_terms and expert_weight > 0.0:
            terms.append(float(expert_weight) * torch.stack(expert_terms).mean())
        if anchor_terms and anchor_weight > 0.0:
            terms.append(float(anchor_weight) * torch.stack(anchor_terms).mean())

    if not terms:
        return student_logits.new_zeros(()), {
            "loss_dual_teacher_noharm": 0.0,
            "dual_teacher_expert_count": float(expert_count),
            "dual_teacher_anchor_count": float(anchor_count),
            "dual_teacher_expert_active_classes": float(expert_active_classes),
            "dual_teacher_anchor_active_classes": float(anchor_active_classes),
            "dual_teacher_expert_conf_mean": expert_conf_mean,
            "dual_teacher_anchor_conf_mean": anchor_conf_mean,
        }
    loss = float(total_weight) * torch.stack(terms).sum().to(dtype=student_logits.dtype)
    return loss, {
        "loss_dual_teacher_noharm": float(loss.detach().cpu()),
        "dual_teacher_expert_count": float(expert_count),
        "dual_teacher_anchor_count": float(anchor_count),
        "dual_teacher_expert_active_classes": float(expert_active_classes),
        "dual_teacher_anchor_active_classes": float(anchor_active_classes),
        "dual_teacher_expert_conf_mean": expert_conf_mean,
        "dual_teacher_anchor_conf_mean": anchor_conf_mean,
    }


def classwise_pareto_groupdro_loss(
    model_out: dict[str, Any],
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """RSCD classwise hard-group training with teacher-protected no-harm terms.

    This is the task-adapted combination of group DRO/JTT and LwF/GEM-style
    protection. The focus side gives more gradient to currently hard RSCD
    composite classes or anchor-teacher error groups. The protection side keeps
    teacher-correct, high-confidence classes from losing their distribution or
    true-vs-neighbor margin. Both sides are averaged by class first, so gains on
    one water/concrete boundary cannot hide regressions on another class.
    """

    focus_weight = float(loss_cfg.get("classwise_pareto_groupdro_weight", 0.0))
    protect_weight = float(loss_cfg.get("classwise_pareto_groupdro_protect_weight", 0.0))
    if focus_weight <= 0.0 and protect_weight <= 0.0:
        return model_out["logits"].new_zeros(()), {
            "loss_classwise_pareto_groupdro": 0.0,
            "classwise_pareto_focus_count": 0.0,
            "classwise_pareto_protect_count": 0.0,
        }

    logits = model_out["logits"]
    focus_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("classwise_pareto_groupdro_focus_classes", loss_cfg.get("focus_ce_classes", []))
    }
    focus_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in focus_classes
    }
    exempt_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("classwise_pareto_groupdro_exempt_classes", [])
    }
    exempt_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in exempt_classes
    }

    requested_pairs: set[frozenset[str]] = set()
    pair_specs = loss_cfg.get("classwise_pareto_groupdro_focus_pairs", [])
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        s_logits = logits.float()
        t_logits = teacher_logits.float()
        per_ce = F.cross_entropy(s_logits, labels, reduction="none")
        teacher_ce = F.cross_entropy(t_logits, labels, reduction="none")
        teacher_prob = F.softmax(t_logits, dim=1)
        teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
        teacher_pred = teacher_top2.indices[:, 0]
        teacher_conf = teacher_top2.values[:, 0]
        if teacher_top2.values.size(1) > 1:
            teacher_gap = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
        else:
            teacher_gap = torch.ones_like(teacher_conf)
        teacher_error = teacher_pred.ne(labels)

        temperature = max(float(loss_cfg.get("classwise_pareto_groupdro_temperature", 2.0)), 1e-3)
        kl_values = F.kl_div(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1),
            reduction="none",
        ).sum(dim=1).clamp_min(0.0) * (temperature * temperature)

        one_hot = F.one_hot(labels, num_classes=s_logits.size(1)).bool()
        neg_inf = -torch.finfo(s_logits.dtype).max
        student_other = s_logits.masked_fill(one_hot, neg_inf).max(dim=1).values
        teacher_other = t_logits.masked_fill(one_hot, neg_inf).max(dim=1).values
        student_true_margin = s_logits.gather(1, labels.view(-1, 1)).squeeze(1) - student_other
        teacher_true_margin = t_logits.gather(1, labels.view(-1, 1)).squeeze(1) - teacher_other
        margin_tolerance = max(float(loss_cfg.get("classwise_pareto_groupdro_margin_tolerance", 0.02)), 0.0)
        margin_barrier = F.relu(teacher_true_margin.detach() - margin_tolerance - student_true_margin)
        if bool(loss_cfg.get("classwise_pareto_groupdro_margin_squared", True)):
            margin_barrier = margin_barrier.pow(2)

    focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in focus_idx:
        focus_mask |= labels.eq(int(idx))
    if bool(loss_cfg.get("classwise_pareto_groupdro_include_teacher_errors", True)):
        focus_mask = focus_mask | teacher_error
    if requested_pairs:
        idx_to_name = {int(idx): canonical_class_label(name) for idx, name in idx_to_class.items()}
        pair_mask = torch.zeros_like(labels, dtype=torch.bool)
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            names = frozenset((idx_to_name.get(left, str(left)), idx_to_name.get(right, str(right))))
            if names in requested_pairs:
                pair_mask |= labels.eq(left) | labels.eq(right)
        focus_mask = focus_mask & pair_mask
    if exempt_idx:
        exempt_mask = torch.zeros_like(labels, dtype=torch.bool)
        for idx in exempt_idx:
            exempt_mask |= labels.eq(int(idx))
        focus_mask = focus_mask & ~exempt_mask

    sample_priority = logits.new_ones(labels.shape, dtype=torch.float32)
    physics_extra = max(float(loss_cfg.get("classwise_pareto_groupdro_physics_extra", 0.0)), 0.0)
    evidence = model_out.get("evidence_stats")
    if physics_extra > 0.0 and isinstance(evidence, torch.Tensor):
        stats = evidence.float().to(device=logits.device)
        grad_std = stats[:, 5].clamp(0.0, 1.0)
        lap_mean = stats[:, 6].clamp(0.0, 1.0)
        contrast_mean = stats[:, 7].clamp(0.0, 1.0)
        specular = stats[:, 8].clamp(0.0, 1.0)
        dark_water = stats[:, 9].clamp(0.0, 1.0)
        wet = stats[:, 10].clamp(0.0, 1.0)
        rough = stats[:, 11].clamp(0.0, 1.0)
        erasure = stats[:, 12].clamp(0.0, 1.0)
        wet_film = torch.clamp(0.45 * wet + 0.25 * dark_water + 0.15 * specular + 0.15 * erasure, 0.0, 1.0)
        hidden_rough = wet_film * torch.sigmoid((0.085 - rough) * 28.0)
        visible_rough = torch.clamp(0.35 * rough + 0.25 * grad_std + 0.22 * lap_mean + 0.18 * contrast_mean, 0.0, 1.0)
        granular = torch.clamp(0.35 * grad_std + 0.35 * lap_mean + 0.20 * contrast_mean + 0.10 * rough, 0.0, 1.0)
        factor_table = spec.class_to_factor.to(device=labels.device)
        factors = factor_table.index_select(0, labels)
        friction = factors[:, 0]
        material = factors[:, 1]
        roughness = factors[:, 2]
        wet_or_water = friction.eq(1) | friction.eq(2)
        concrete = material.eq(FACTOR_LABELS["material"].index("concrete"))
        loose = material.ge(FACTOR_LABELS["material"].index("mud"))
        has_roughness = roughness.ge(1)
        mechanism_score = torch.zeros_like(sample_priority)
        mechanism_score = torch.where(wet_or_water & concrete & has_roughness, hidden_rough, mechanism_score)
        mechanism_score = torch.where((friction.eq(0) & concrete & has_roughness), visible_rough, mechanism_score)
        mechanism_score = torch.where(loose, granular, mechanism_score)
        sample_priority = (sample_priority + physics_extra * mechanism_score).clamp(1.0, 1.0 + physics_extra)

    terms: list[torch.Tensor] = []
    focus_count = 0.0
    focus_group_count = 0
    focus_loss_sum = 0.0
    focus_softmax_temp = max(float(loss_cfg.get("classwise_pareto_groupdro_group_temperature", 5.0)), 0.0)
    group_losses: list[torch.Tensor] = []
    for class_idx in labels.detach().unique().tolist():
        class_idx = int(class_idx)
        class_mask = labels.eq(class_idx) & focus_mask
        if not bool(class_mask.any()):
            continue
        idx = class_mask.nonzero(as_tuple=False).flatten()
        weights = sample_priority.index_select(0, idx).to(device=per_ce.device, dtype=per_ce.dtype)
        group_loss = (per_ce.index_select(0, idx) * weights).sum() / weights.sum().clamp_min(1e-6)
        group_losses.append(group_loss)
        focus_count += float(idx.numel())
        focus_group_count += 1
        focus_loss_sum += float(group_loss.detach().cpu())
    if focus_weight > 0.0 and group_losses:
        grouped = torch.stack(group_losses)
        if focus_softmax_temp > 0.0 and grouped.numel() > 1:
            dro_weights = F.softmax(focus_softmax_temp * grouped.detach(), dim=0)
            focus_loss = (dro_weights * grouped).sum()
        else:
            focus_loss = grouped.mean()
        terms.append(float(focus_weight) * focus_loss)

    protect_conf = float(loss_cfg.get("classwise_pareto_groupdro_protect_confidence", 0.70))
    protect_margin = float(loss_cfg.get("classwise_pareto_groupdro_protect_teacher_margin", 0.12))
    protect_mask = teacher_pred.eq(labels)
    if protect_conf > 0.0:
        protect_mask = protect_mask & teacher_conf.ge(protect_conf)
    if protect_margin > 0.0:
        protect_mask = protect_mask & teacher_gap.ge(protect_margin)
    if bool(loss_cfg.get("classwise_pareto_groupdro_protect_nonfocus_only", True)):
        protect_mask = protect_mask & ~focus_mask
    if exempt_idx:
        exempt_mask = torch.zeros_like(labels, dtype=torch.bool)
        for idx in exempt_idx:
            exempt_mask |= labels.eq(int(idx))
        protect_mask = protect_mask & ~exempt_mask

    protect_terms: list[torch.Tensor] = []
    protect_count = 0.0
    protect_group_count = 0
    protect_kl_sum = 0.0
    protect_margin_sum = 0.0
    kl_weight = float(loss_cfg.get("classwise_pareto_groupdro_kl_weight", 1.0))
    margin_weight = float(loss_cfg.get("classwise_pareto_groupdro_margin_weight", 1.0))
    ce_barrier_weight = float(loss_cfg.get("classwise_pareto_groupdro_ce_barrier_weight", 0.0))
    ce_tolerance = max(float(loss_cfg.get("classwise_pareto_groupdro_ce_tolerance", 0.02)), 0.0)
    no_flip_weight = float(loss_cfg.get("classwise_pareto_groupdro_no_flip_weight", 0.0))
    for class_idx in labels.detach().unique().tolist():
        class_idx = int(class_idx)
        class_mask = labels.eq(class_idx) & protect_mask
        if not bool(class_mask.any()):
            continue
        class_parts: list[torch.Tensor] = []
        if kl_weight > 0.0:
            class_kl = kl_values[class_mask].mean()
            class_parts.append(float(kl_weight) * class_kl)
            protect_kl_sum += float(class_kl.detach().cpu())
        if margin_weight > 0.0:
            class_margin = margin_barrier[class_mask].mean()
            class_parts.append(float(margin_weight) * class_margin)
            protect_margin_sum += float(class_margin.detach().cpu())
        if ce_barrier_weight > 0.0:
            ce_excess = F.relu(per_ce[class_mask] - teacher_ce[class_mask].detach() - ce_tolerance)
            class_parts.append(float(ce_barrier_weight) * ce_excess.pow(2).mean())
        if no_flip_weight > 0.0:
            idx = class_mask.nonzero(as_tuple=False).flatten()
            class_parts.append(
                float(no_flip_weight)
                * F.cross_entropy(s_logits.index_select(0, idx), teacher_pred.index_select(0, idx))
            )
        if class_parts:
            protect_terms.append(torch.stack(class_parts).sum())
            protect_count += float(class_mask.sum().detach().cpu())
            protect_group_count += 1
    if protect_weight > 0.0 and protect_terms:
        protect_group_losses = torch.stack(protect_terms)
        protect_loss = protect_group_losses.mean()
        terms.append(float(protect_weight) * protect_loss)

    if not terms:
        return logits.new_zeros(()), {
            "loss_classwise_pareto_groupdro": 0.0,
            "classwise_pareto_focus_count": focus_count,
            "classwise_pareto_protect_count": protect_count,
            "classwise_pareto_teacher_error_rate": float(teacher_error.float().detach().mean().cpu()),
        }

    loss = torch.stack(terms).sum().to(dtype=logits.dtype)
    return loss, {
        "loss_classwise_pareto_groupdro": float(loss.detach().cpu()),
        "classwise_pareto_focus_count": focus_count,
        "classwise_pareto_focus_groups": float(focus_group_count),
        "classwise_pareto_focus_loss_mean": focus_loss_sum / max(focus_group_count, 1),
        "classwise_pareto_protect_count": protect_count,
        "classwise_pareto_protect_groups": float(protect_group_count),
        "classwise_pareto_protect_kl_mean": protect_kl_sum / max(protect_group_count, 1),
        "classwise_pareto_protect_margin_mean": protect_margin_sum / max(protect_group_count, 1),
        "classwise_pareto_teacher_error_rate": float(teacher_error.float().detach().mean().cpu()),
        "classwise_pareto_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
        "classwise_pareto_teacher_margin_mean": float(teacher_gap.detach().mean().cpu()),
        "classwise_pareto_priority_mean": float(sample_priority.detach().mean().cpu()),
    }


def _class_mask_from_names(
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    class_names: list[str] | tuple[str, ...] | set[str],
) -> torch.Tensor:
    target_names = {canonical_class_label(name) for name in class_names}
    mask = torch.zeros_like(labels, dtype=torch.bool)
    if not target_names:
        return mask
    for idx, name in idx_to_class.items():
        if canonical_class_label(name) in target_names:
            mask |= labels.eq(int(idx))
    return mask


def rscd_focus_protect_objectives(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    """Build RSCD-specific weak-class and protected-class objectives.

    The focus objective targets the RSCD composite classes where split concrete
    film/roughness mechanisms have shown possible gains. The protection
    objective is computed on non-focus samples, optionally restricted to anchor
    teacher-correct, high-confidence regions. It is used only as a gradient
    constraint by the surgery step below.
    """

    weight = float(loss_cfg.get("rscd_pcgrad_focus_weight", 0.0))
    if weight <= 0.0:
        return None, None, {"rscd_pcgrad_active": 0.0}
    focus_classes = loss_cfg.get("rscd_pcgrad_focus_classes", loss_cfg.get("focus_ce_classes", []))
    focus_mask = _class_mask_from_names(labels, idx_to_class, focus_classes)
    if bool(loss_cfg.get("rscd_pcgrad_protect_focus_teacher_correct", False)):
        protect_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        protect_mask = ~focus_mask
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        logits = student_logits.float()
        per_ce = F.cross_entropy(logits, labels, reduction="none")
        teacher_prob = None
        teacher_conf = None
        teacher_margin = None
        if teacher_logits is not None:
            teacher_prob = F.softmax(teacher_logits.float(), dim=1)
            teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
            teacher_pred = teacher_top2.indices[:, 0]
            teacher_conf = teacher_top2.values[:, 0]
            if teacher_top2.values.size(1) > 1:
                teacher_margin = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
            else:
                teacher_margin = torch.ones_like(teacher_conf)
            if bool(loss_cfg.get("rscd_pcgrad_protect_teacher_correct", True)):
                protect_mask = protect_mask & teacher_pred.eq(labels)
            if bool(loss_cfg.get("rscd_pcgrad_focus_teacher_errors_only", False)):
                focus_mask = focus_mask & teacher_pred.ne(labels)
            protect_conf = float(loss_cfg.get("rscd_pcgrad_protect_confidence", 0.0))
            protect_margin = float(loss_cfg.get("rscd_pcgrad_protect_margin", 0.0))
            if protect_conf > 0.0:
                protect_mask = protect_mask & teacher_conf.ge(protect_conf)
            if protect_margin > 0.0:
                protect_mask = protect_mask & teacher_margin.ge(protect_margin)
        if not bool(focus_mask.any()) or not bool(protect_mask.any()):
            return None, None, {
                "rscd_pcgrad_active": 0.0,
                "rscd_pcgrad_focus_count": float(focus_mask.sum().detach().cpu()),
                "rscd_pcgrad_protect_count": float(protect_mask.sum().detach().cpu()),
            }
        focus_loss = per_ce[focus_mask].mean()
        protect_loss = per_ce[protect_mask].mean()
        protect_kl_weight = float(loss_cfg.get("rscd_pcgrad_protect_kl_weight", 0.0))
        if teacher_logits is not None and teacher_prob is not None and protect_kl_weight > 0.0:
            temperature = max(float(loss_cfg.get("rscd_pcgrad_protect_temperature", 2.0)), 1e-3)
            protect_kl = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits.float() / temperature, dim=1),
                reduction="none",
            ).sum(dim=1) * (temperature * temperature)
            protect_loss = protect_loss + protect_kl_weight * protect_kl[protect_mask].mean()
    logs = {
        "rscd_pcgrad_active": 1.0,
        "rscd_pcgrad_focus_count": float(focus_mask.sum().detach().cpu()),
        "rscd_pcgrad_protect_count": float(protect_mask.sum().detach().cpu()),
        "loss_rscd_pcgrad_focus": float(focus_loss.detach().cpu()),
        "loss_rscd_pcgrad_protect": float(protect_loss.detach().cpu()),
    }
    if teacher_conf is not None and teacher_margin is not None:
        logs.update(
            {
                "rscd_pcgrad_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
                "rscd_pcgrad_teacher_margin_mean": float(teacher_margin.detach().mean().cpu()),
            }
        )
    return focus_loss, protect_loss, logs


def _indices_from_class_names(
    idx_to_class: dict[int, str],
    class_names: list[str] | tuple[str, ...] | set[str],
) -> set[int]:
    target_names = {canonical_class_label(name) for name in class_names}
    return {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in target_names
    }


def _group_mask_from_indices(labels: torch.Tensor, indices: set[int]) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in indices:
        mask |= labels.eq(int(idx))
    return mask


def _build_factor_group_masks(
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    spec: RSCDFactorSpec,
    base_mask: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    """Build RSCD factor-aware protection groups for no-harm gradient surgery."""

    min_count = max(int(loss_cfg.get("rscd_pcgrad_group_min_count", 1)), 1)
    explicit_groups = loss_cfg.get("rscd_pcgrad_protect_groups", [])
    groups: list[tuple[str, torch.Tensor]] = []
    if explicit_groups:
        for group in explicit_groups:
            if isinstance(group, dict):
                name = str(group.get("name", f"group_{len(groups)}"))
                class_names = group.get("classes", [])
            else:
                name = str(group)
                class_names = [str(group)]
            indices = _indices_from_class_names(idx_to_class, class_names)
            if not indices:
                continue
            mask = _group_mask_from_indices(labels, indices) & base_mask
            if int(mask.sum().detach().cpu()) >= min_count:
                groups.append((name, mask))
        return groups

    mode = str(loss_cfg.get("rscd_pcgrad_protect_group_mode", "factor")).lower()
    class_to_factor = spec.class_to_factor.to(device=labels.device)
    if mode in {"class", "classes", "per_class"}:
        for class_idx in labels[base_mask].detach().unique().tolist():
            class_idx = int(class_idx)
            name = canonical_class_label(idx_to_class.get(class_idx, str(class_idx)))
            mask = labels.eq(class_idx) & base_mask
            if int(mask.sum().detach().cpu()) >= min_count:
                groups.append((f"class:{name}", mask))
        return groups

    if mode in {"coarse", "road_state"}:
        coarse_defs = {
            "dry_asphalt": ("dry_asphalt_smooth", "dry_asphalt_slight", "dry_asphalt_severe"),
            "dry_concrete": ("dry_concrete_smooth", "dry_concrete_slight", "dry_concrete_severe"),
            "wet_water_asphalt": (
                "wet_asphalt_smooth",
                "wet_asphalt_slight",
                "wet_asphalt_severe",
                "water_asphalt_smooth",
                "water_asphalt_slight",
                "water_asphalt_severe",
            ),
            "wet_water_concrete": (
                "wet_concrete_smooth",
                "wet_concrete_slight",
                "wet_concrete_severe",
                "water_concrete_smooth",
                "water_concrete_slight",
                "water_concrete_severe",
            ),
            "snow_ice": ("fresh_snow", "melted_snow", "ice"),
            "granular": ("dry_mud", "wet_mud", "water_mud", "dry_gravel", "wet_gravel", "water_gravel"),
        }
        for group_name, names in coarse_defs.items():
            indices = _indices_from_class_names(idx_to_class, names)
            mask = _group_mask_from_indices(labels, indices) & base_mask
            if int(mask.sum().detach().cpu()) >= min_count:
                groups.append((f"coarse:{group_name}", mask))
        return groups

    factors = class_to_factor.index_select(0, labels)
    axes = loss_cfg.get("rscd_pcgrad_protect_factor_axes", FACTOR_AXES)
    for axis in axes:
        axis = str(axis)
        if axis not in FACTOR_AXES:
            continue
        axis_i = FACTOR_AXES.index(axis)
        valid = base_mask & factors[:, axis_i].ge(0)
        for value_idx in factors[valid, axis_i].detach().unique().tolist():
            value_idx = int(value_idx)
            if value_idx < 0:
                continue
            value_name = FACTOR_LABELS[axis][value_idx]
            mask = valid & factors[:, axis_i].eq(value_idx)
            if int(mask.sum().detach().cpu()) >= min_count:
                groups.append((f"{axis}:{value_name}", mask))
    return groups


def rscd_physics_focus_priority(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """PhysicsTexture-guided focus weights for RSCD no-harm gradient surgery.

    Generic PCGrad treats all focus samples equally. RSCD errors are more
    structured: wet/water concrete roughness is often hidden by film, asphalt
    water severity depends on dark/specular water evidence, and dry-concrete
    roughness depends on visible high-frequency texture. This detached priority
    only scales the focus objective; protected groups still constrain the final
    gradient direction.
    """

    extra = max(float(loss_cfg.get("rscd_pcgrad_focus_physics_extra", 0.0)), 0.0)
    evidence = model_out.get("evidence_stats")
    if extra <= 0.0 or not isinstance(evidence, torch.Tensor):
        return None, {
            "rscd_pcgrad_focus_physics_weight_mean": 1.0,
            "rscd_pcgrad_focus_physics_active_rate": 0.0,
        }

    stats = evidence.detach().float().to(device=labels.device)
    grad_mean = stats[:, 4].clamp(0.0, 1.0)
    grad_std = stats[:, 5].clamp(0.0, 1.0)
    lap_mean = stats[:, 6].clamp(0.0, 1.0)
    contrast_mean = stats[:, 7].clamp(0.0, 1.0)
    specular = stats[:, 8].clamp(0.0, 1.0)
    dark_water = stats[:, 9].clamp(0.0, 1.0)
    wet = stats[:, 10].clamp(0.0, 1.0)
    rough = stats[:, 11].clamp(0.0, 1.0)
    erasure = stats[:, 12].clamp(0.0, 1.0)
    snow_ice = torch.maximum(stats[:, 13], stats[:, 14]).clamp(0.0, 1.0)

    wet_film = torch.clamp(0.48 * wet + 0.24 * dark_water + 0.16 * specular + 0.12 * erasure, 0.0, 1.0)
    hidden_concrete_roughness = wet_film * torch.sigmoid((0.080 - rough) * 32.0) * torch.sigmoid(
        (0.34 - snow_ice) * 18.0
    )
    visible_concrete_roughness = torch.clamp(
        0.34 * rough + 0.25 * grad_std + 0.22 * lap_mean + 0.19 * contrast_mean,
        0.0,
        1.0,
    )
    asphalt_water_severity = torch.clamp(
        0.38 * dark_water + 0.24 * wet + 0.20 * specular + 0.18 * erasure,
        0.0,
        1.0,
    )
    granular_texture = torch.clamp(
        0.34 * grad_std + 0.30 * lap_mean + 0.22 * contrast_mean + 0.14 * grad_mean,
        0.0,
        1.0,
    )

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    friction = factors[:, 0]
    material = factors[:, 1]
    roughness = factors[:, 2]
    wet_or_water = friction.eq(1) | friction.eq(2)
    concrete = material.eq(FACTOR_LABELS["material"].index("concrete"))
    asphalt = material.eq(FACTOR_LABELS["material"].index("asphalt"))
    loose = material.ge(FACTOR_LABELS["material"].index("mud"))
    rough_family = roughness.ge(1)

    priority = torch.zeros_like(wet_film)
    priority = torch.where(wet_or_water & concrete & rough_family, hidden_concrete_roughness, priority)
    priority = torch.where(friction.eq(0) & concrete & rough_family, visible_concrete_roughness, priority)
    priority = torch.where(wet_or_water & asphalt & rough_family, asphalt_water_severity, priority)
    priority = torch.where(loose, granular_texture, priority)

    weights = (1.0 + float(extra) * priority).clamp(
        1.0,
        max(float(loss_cfg.get("rscd_pcgrad_focus_physics_max_weight", 1.65)), 1.0),
    )
    logs = {
        "rscd_pcgrad_focus_physics_weight_mean": float(weights.detach().mean().cpu()),
        "rscd_pcgrad_focus_physics_active_rate": float(weights.detach().gt(1.001).float().mean().cpu()),
        "rscd_pcgrad_focus_hidden_concrete_mean": float(hidden_concrete_roughness.detach().mean().cpu()),
        "rscd_pcgrad_focus_asphalt_water_mean": float(asphalt_water_severity.detach().mean().cpu()),
    }
    return weights, logs


def family_mechanism_router_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise the early family router with RSCD mechanism families.

    Route 0 is a protection route. Routes 1-3 correspond to the three hard
    mechanisms that previously traded off against each other: visible dry
    concrete roughness, wet/water asphalt film severity, and wet/water concrete
    hidden roughness / smooth-bridge ambiguity.
    """

    weight = float(loss_cfg.get("family_mechanism_router_weight", 0.0))
    router_logits_obj = model_out.get("family_route_logits")
    logits_ref = model_out["logits"]
    if weight <= 0.0 or router_logits_obj is None:
        return logits_ref.new_zeros(()), {"loss_family_mechanism_router": 0.0, "family_router_active": 0.0}

    if isinstance(router_logits_obj, dict):
        router_values = [value for value in router_logits_obj.values() if isinstance(value, torch.Tensor)]
        if not router_values:
            return logits_ref.new_zeros(()), {"loss_family_mechanism_router": 0.0, "family_router_active": 0.0}
        router_logits = torch.stack([value.float() for value in router_values], dim=0).mean(dim=0)
    elif isinstance(router_logits_obj, torch.Tensor):
        router_logits = router_logits_obj.float()
    else:
        return logits_ref.new_zeros(()), {"loss_family_mechanism_router": 0.0, "family_router_active": 0.0}

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    friction = factors[:, 0]
    material = factors[:, 1]
    roughness = factors[:, 2]
    dry = friction.eq(FACTOR_LABELS["friction"].index("dry"))
    wet_or_water = friction.eq(FACTOR_LABELS["friction"].index("wet")) | friction.eq(
        FACTOR_LABELS["friction"].index("water")
    )
    asphalt = material.eq(FACTOR_LABELS["material"].index("asphalt"))
    concrete = material.eq(FACTOR_LABELS["material"].index("concrete"))
    smooth = roughness.eq(FACTOR_LABELS["roughness"].index("smooth"))
    slight = roughness.eq(FACTOR_LABELS["roughness"].index("slight"))
    severe = roughness.eq(FACTOR_LABELS["roughness"].index("severe"))
    rough_visible = roughness.eq(FACTOR_LABELS["roughness"].index("slight")) | roughness.eq(
        FACTOR_LABELS["roughness"].index("severe")
    )
    paved_rough = roughness.ge(FACTOR_LABELS["roughness"].index("smooth"))
    split_concrete_routes = bool(loss_cfg.get("family_router_split_concrete_film_routes", False)) and int(
        router_logits.shape[1]
    ) >= 6

    target = torch.zeros_like(labels)
    dry_concrete_route = dry & concrete & rough_visible
    asphalt_film_route = wet_or_water & asphalt & rough_visible
    concrete_bridge_route = wet_or_water & concrete & paved_rough
    concrete_smooth_route = wet_or_water & concrete & smooth
    concrete_slight_route = wet_or_water & concrete & slight
    concrete_severe_route = wet_or_water & concrete & severe
    if bool(loss_cfg.get("family_router_dry_as_protect", False)):
        dry_concrete_route = torch.zeros_like(dry_concrete_route)
    if bool(loss_cfg.get("family_router_asphalt_as_protect", False)):
        asphalt_film_route = torch.zeros_like(asphalt_film_route)
    if bool(loss_cfg.get("family_router_concrete_as_protect", False)):
        concrete_bridge_route = torch.zeros_like(concrete_bridge_route)
        concrete_smooth_route = torch.zeros_like(concrete_smooth_route)
        concrete_slight_route = torch.zeros_like(concrete_slight_route)
        concrete_severe_route = torch.zeros_like(concrete_severe_route)
    target = torch.where(dry_concrete_route, torch.ones_like(target), target)
    target = torch.where(asphalt_film_route, torch.full_like(target, 2), target)
    if split_concrete_routes:
        target = torch.where(concrete_smooth_route, torch.full_like(target, 3), target)
        target = torch.where(concrete_slight_route, torch.full_like(target, 4), target)
        target = torch.where(concrete_severe_route, torch.full_like(target, 5), target)
    else:
        target = torch.where(concrete_bridge_route, torch.full_like(target, 3), target)

    route_weights = torch.full_like(router_logits[:, 0], float(loss_cfg.get("family_router_protect_sample_weight", 0.35)))
    route_weights = torch.where(target.gt(0), torch.ones_like(route_weights), route_weights)
    route_weights = torch.where(dry_concrete_route, route_weights * float(loss_cfg.get("family_router_dry_weight", 1.10)), route_weights)
    route_weights = torch.where(
        asphalt_film_route,
        route_weights * float(loss_cfg.get("family_router_asphalt_weight", 1.35)),
        route_weights,
    )
    route_weights = torch.where(
        concrete_smooth_route | concrete_slight_route | concrete_severe_route if split_concrete_routes else concrete_bridge_route,
        route_weights * float(loss_cfg.get("family_router_concrete_weight", 1.45)),
        route_weights,
    )
    if split_concrete_routes:
        route_weights = torch.where(
            concrete_slight_route,
            route_weights * float(loss_cfg.get("family_router_concrete_slight_weight", 1.0)),
            route_weights,
        )
        route_weights = torch.where(
            concrete_severe_route,
            route_weights * float(loss_cfg.get("family_router_concrete_severe_weight", 1.0)),
            route_weights,
        )

    per_sample = F.cross_entropy(router_logits, target, reduction="none")
    loss = (per_sample * route_weights).sum() / route_weights.sum().clamp_min(1e-6)

    probs_for_loss = F.softmax(router_logits, dim=1)
    probs = probs_for_loss.detach()
    route_entropy = -(
        probs_for_loss.clamp_min(1e-8) * probs_for_loss.clamp_min(1e-8).log()
    ).sum(dim=1) / math.log(float(router_logits.shape[1]))
    entropy_weight = float(loss_cfg.get("family_router_entropy_weight", 0.0))
    if entropy_weight > 0.0:
        entropy_mask = target.gt(0) if bool(loss_cfg.get("family_router_entropy_active_only", True)) else torch.ones_like(target, dtype=torch.bool)
        if bool(entropy_mask.any()):
            entropy_values = route_entropy[entropy_mask]
            entropy_sample_weights = route_weights[entropy_mask]
            entropy_loss = (entropy_values * entropy_sample_weights).sum() / entropy_sample_weights.sum().clamp_min(1e-6)
            loss = loss + entropy_weight * entropy_loss.to(device=loss.device, dtype=loss.dtype)
        else:
            entropy_loss = route_entropy.new_zeros(())
    else:
        entropy_loss = route_entropy.new_zeros(())

    margin_weight = float(loss_cfg.get("family_router_target_margin_weight", 0.0))
    margin_target = float(loss_cfg.get("family_router_target_margin", 0.28))
    if margin_weight > 0.0:
        one_hot = F.one_hot(target.clamp_min(0), num_classes=router_logits.shape[1]).bool()
        target_prob = probs_for_loss.gather(1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
        other_prob = probs_for_loss.masked_fill(one_hot, -1.0).amax(dim=1)
        margin_mask = target.gt(0) if bool(loss_cfg.get("family_router_margin_active_only", True)) else torch.ones_like(target, dtype=torch.bool)
        if bool(margin_mask.any()):
            margin_violation = F.relu(float(margin_target) - (target_prob - other_prob)).pow(2)
            margin_loss = (
                margin_violation[margin_mask] * route_weights[margin_mask]
            ).sum() / route_weights[margin_mask].sum().clamp_min(1e-6)
            loss = loss + margin_weight * margin_loss.to(device=loss.device, dtype=loss.dtype)
        else:
            margin_loss = route_entropy.new_zeros(())
    else:
        margin_loss = route_entropy.new_zeros(())

    pred = router_logits.detach().argmax(dim=1)
    logs = {
        "loss_family_mechanism_router": float(loss.detach().cpu()),
        "family_router_active": 1.0,
        "family_router_acc": float(pred.eq(target).float().mean().cpu()),
        "family_router_protect_count": float(target.eq(0).sum().detach().cpu()),
        "family_router_dry_count": float(target.eq(1).sum().detach().cpu()),
        "family_router_asphalt_count": float(target.eq(2).sum().detach().cpu()),
        "family_router_concrete_count": float(target.ge(3).sum().detach().cpu() if split_concrete_routes else target.eq(3).sum().detach().cpu()),
        "family_router_protect_prob_mean": float(probs[:, 0].mean().cpu()),
        "family_router_active_prob_mean": float(probs[:, 1:].sum(dim=1).mean().cpu()),
        "family_router_entropy_mean": float(route_entropy.detach().mean().cpu()),
        "family_router_entropy_loss": float(entropy_loss.detach().cpu()),
        "family_router_margin_loss": float(margin_loss.detach().cpu()),
    }
    if split_concrete_routes:
        logs.update(
            {
                "family_router_concrete_smooth_count": float(target.eq(3).sum().detach().cpu()),
                "family_router_concrete_slight_count": float(target.eq(4).sum().detach().cpu()),
                "family_router_concrete_severe_count": float(target.eq(5).sum().detach().cpu()),
                "family_router_concrete_smooth_prob_mean": float(probs[:, 3].mean().cpu()),
                "family_router_concrete_slight_prob_mean": float(probs[:, 4].mean().cpu()),
                "family_router_concrete_severe_prob_mean": float(probs[:, 5].mean().cpu()),
            }
        )
    leak_weight = float(loss_cfg.get("family_router_protect_leak_weight", 0.0))
    if leak_weight > 0.0 and bool(target.eq(0).any()):
        leak = probs_for_loss[:, 1:].sum(dim=1)
        protect_leak = leak[target.eq(0)]
        leak_loss = protect_leak.mean().to(device=router_logits.device, dtype=router_logits.dtype)
        loss = loss + float(leak_weight) * leak_loss
        logs["family_router_protect_leak"] = float(protect_leak.detach().mean().cpu())
    return float(weight) * loss.to(dtype=logits_ref.dtype), logs


def rscd_focus_grouped_protect_objectives(
    model_out: dict[str, Any],
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, list[tuple[str, torch.Tensor]], dict[str, float]]:
    """Focus-vs-many-protect objectives for RSCD factor-group no-harm updates.

    A single non-focus protection loss can hide local regressions: a gain on a
    wet-concrete boundary may hurt only dry-asphalt severe, while the averaged
    non-focus gradient still looks harmless. This RSCD-specific variant splits
    protected samples by class, coarse road state, or factor value and performs
    gradient conflict checks against each group separately.
    """

    weight = float(loss_cfg.get("rscd_pcgrad_focus_weight", 0.0))
    if weight <= 0.0:
        return None, [], {"rscd_pcgrad_grouped_active": 0.0}
    student_logits = model_out["logits"]
    focus_classes = loss_cfg.get("rscd_pcgrad_focus_classes", loss_cfg.get("focus_ce_classes", []))
    focus_mask = _class_mask_from_names(labels, idx_to_class, focus_classes)
    if bool(loss_cfg.get("rscd_pcgrad_protect_focus_teacher_correct", False)):
        protect_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        protect_mask = ~focus_mask
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        logits = student_logits.float()
        per_ce = F.cross_entropy(logits, labels, reduction="none")
        teacher_prob = None
        teacher_conf = None
        teacher_margin = None
        if teacher_logits is not None:
            teacher_prob = F.softmax(teacher_logits.float(), dim=1)
            teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
            teacher_pred = teacher_top2.indices[:, 0]
            teacher_conf = teacher_top2.values[:, 0]
            if teacher_top2.values.size(1) > 1:
                teacher_margin = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
            else:
                teacher_margin = torch.ones_like(teacher_conf)
            if bool(loss_cfg.get("rscd_pcgrad_protect_teacher_correct", True)):
                protect_mask = protect_mask & teacher_pred.eq(labels)
            if bool(loss_cfg.get("rscd_pcgrad_focus_teacher_errors_only", False)):
                focus_mask = focus_mask & teacher_pred.ne(labels)
            protect_conf = float(loss_cfg.get("rscd_pcgrad_protect_confidence", 0.0))
            protect_margin = float(loss_cfg.get("rscd_pcgrad_protect_margin", 0.0))
            if protect_conf > 0.0:
                protect_mask = protect_mask & teacher_conf.ge(protect_conf)
            if protect_margin > 0.0:
                protect_mask = protect_mask & teacher_margin.ge(protect_margin)
        if not bool(focus_mask.any()) or not bool(protect_mask.any()):
            return None, [], {
                "rscd_pcgrad_grouped_active": 0.0,
                "rscd_pcgrad_focus_count": float(focus_mask.sum().detach().cpu()),
                "rscd_pcgrad_protect_count": float(protect_mask.sum().detach().cpu()),
            }
        physics_weights, physics_logs = rscd_physics_focus_priority(model_out, labels, spec, loss_cfg)
        if physics_weights is not None:
            selected_weights = physics_weights[focus_mask].to(device=per_ce.device, dtype=per_ce.dtype)
            focus_loss = (per_ce[focus_mask] * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)
        else:
            focus_loss = per_ce[focus_mask].mean()
        group_masks = _build_factor_group_masks(labels, idx_to_class, spec, protect_mask, loss_cfg)
        protect_losses: list[tuple[str, torch.Tensor]] = []
        protect_kl_weight = float(loss_cfg.get("rscd_pcgrad_protect_kl_weight", 0.0))
        temperature = max(float(loss_cfg.get("rscd_pcgrad_protect_temperature", 2.0)), 1e-3)
        protect_kl = None
        if teacher_logits is not None and teacher_prob is not None and protect_kl_weight > 0.0:
            protect_kl = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits.float() / temperature, dim=1),
                reduction="none",
            ).sum(dim=1) * (temperature * temperature)
        for group_name, group_mask in group_masks:
            if not bool(group_mask.any()):
                continue
            protect_loss = per_ce[group_mask].mean()
            if protect_kl is not None:
                protect_loss = protect_loss + protect_kl_weight * protect_kl[group_mask].mean()
            protect_losses.append((group_name, protect_loss))
        max_groups = int(loss_cfg.get("rscd_pcgrad_max_protect_groups", 0))
        if max_groups > 0 and len(protect_losses) > max_groups:
            mode = str(loss_cfg.get("rscd_pcgrad_protect_group_select", "hardest")).lower()
            if mode in {"hard", "hardest", "loss"}:
                protect_losses = sorted(
                    protect_losses,
                    key=lambda item: float(item[1].detach().cpu()),
                    reverse=True,
                )[:max_groups]
            else:
                protect_losses = protect_losses[:max_groups]
    logs = {
        "rscd_pcgrad_grouped_active": 1.0 if protect_losses else 0.0,
        "rscd_pcgrad_focus_count": float(focus_mask.sum().detach().cpu()),
        "rscd_pcgrad_protect_count": float(protect_mask.sum().detach().cpu()),
        "rscd_pcgrad_protect_group_count": float(len(protect_losses)),
        "loss_rscd_pcgrad_focus": float(focus_loss.detach().cpu()),
    }
    if "physics_logs" in locals():
        logs.update(physics_logs)
    if protect_losses:
        logs["loss_rscd_pcgrad_protect_mean"] = float(
            torch.stack([loss.detach() for _, loss in protect_losses]).mean().cpu()
        )
    if teacher_conf is not None and teacher_margin is not None:
        logs.update(
            {
                "rscd_pcgrad_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
                "rscd_pcgrad_teacher_margin_mean": float(teacher_margin.detach().mean().cpu()),
            }
        )
    return focus_loss, protect_losses, logs


def rscd_focus_protect_gradient_surgery(
    params: list[torch.nn.Parameter],
    focus_loss: torch.Tensor,
    protect_loss: torch.Tensor,
    *,
    focus_weight: float,
    accum: int,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Project weak-class gradients away from protected-class conflicts.

    This is a task-adapted PCGrad/GEM-style step. Let g_f minimize hard RSCD
    focus classes and g_p minimize protected non-focus samples. If
    <g_f, g_p> < 0, applying -g_f would increase the protected loss to first
    order, so the conflicting component of g_f is removed before it is added as
    an extra update on top of the normal training loss.
    """

    focus_grads = torch.autograd.grad(focus_loss, params, retain_graph=True, allow_unused=True)
    protect_grads = torch.autograd.grad(protect_loss, params, retain_graph=True, allow_unused=True)
    dot = focus_loss.new_zeros(())
    focus_norm = focus_loss.new_zeros(())
    protect_norm = focus_loss.new_zeros(())
    for focus_grad, protect_grad in zip(focus_grads, protect_grads):
        if focus_grad is not None:
            focus_norm = focus_norm + focus_grad.detach().pow(2).sum()
        if protect_grad is not None:
            protect_norm = protect_norm + protect_grad.detach().pow(2).sum()
        if focus_grad is not None and protect_grad is not None:
            dot = dot + (focus_grad.detach() * protect_grad.detach()).sum()
    protect_norm = protect_norm.clamp_min(1e-12)
    conflict = bool((dot < 0).detach().cpu())
    coeff = dot / protect_norm if conflict else dot.new_zeros(())
    scale = float(focus_weight) / max(int(accum), 1)
    adjusted: list[torch.Tensor | None] = []
    with torch.no_grad():
        for focus_grad, protect_grad in zip(focus_grads, protect_grads):
            if focus_grad is None:
                adjusted.append(None)
                continue
            if conflict and protect_grad is not None:
                update = focus_grad - coeff.to(dtype=focus_grad.dtype, device=focus_grad.device) * protect_grad
            else:
                update = focus_grad
            adjusted.append((float(scale) * update).detach())
    return adjusted, {
        "rscd_pcgrad_conflict": 1.0 if conflict else 0.0,
        "rscd_pcgrad_dot": float(dot.detach().cpu()),
        "rscd_pcgrad_focus_grad_norm": float(torch.sqrt(focus_norm.detach().clamp_min(0.0)).cpu()),
        "rscd_pcgrad_protect_grad_norm": float(torch.sqrt(protect_norm.detach().clamp_min(0.0)).cpu()),
        "rscd_pcgrad_projection_coeff": float(coeff.detach().cpu()),
        "rscd_pcgrad_focus_weight": float(focus_weight),
    }


def rscd_focus_grouped_protect_gradient_surgery(
    params: list[torch.nn.Parameter],
    focus_loss: torch.Tensor,
    protect_losses: list[tuple[str, torch.Tensor]],
    *,
    focus_weight: float,
    accum: int,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Sequential PCGrad against multiple RSCD protection groups."""

    focus_grads = torch.autograd.grad(focus_loss, params, retain_graph=True, allow_unused=True)
    current: list[torch.Tensor | None] = [
        grad.detach().clone() if grad is not None else None for grad in focus_grads
    ]
    focus_norm = focus_loss.new_zeros(())
    for grad in current:
        if grad is not None:
            focus_norm = focus_norm + grad.pow(2).sum()

    conflicts = 0
    protect_count = 0
    dot_sum = 0.0
    min_dot = None
    max_projection = 0.0
    for _, protect_loss in protect_losses:
        protect_grads = torch.autograd.grad(protect_loss, params, retain_graph=True, allow_unused=True)
        dot = protect_loss.new_zeros(())
        protect_norm = protect_loss.new_zeros(())
        for focus_grad, protect_grad in zip(current, protect_grads):
            if protect_grad is not None:
                protect_norm = protect_norm + protect_grad.detach().pow(2).sum()
            if focus_grad is not None and protect_grad is not None:
                dot = dot + (focus_grad * protect_grad.detach()).sum()
        protect_norm = protect_norm.clamp_min(1e-12)
        dot_value = float(dot.detach().cpu())
        dot_sum += dot_value
        min_dot = dot_value if min_dot is None else min(float(min_dot), dot_value)
        protect_count += 1
        if bool((dot < 0).detach().cpu()):
            conflicts += 1
            coeff = dot / protect_norm
            max_projection = max(max_projection, abs(float(coeff.detach().cpu())))
            next_current: list[torch.Tensor | None] = []
            with torch.no_grad():
                for focus_grad, protect_grad in zip(current, protect_grads):
                    if focus_grad is None:
                        next_current.append(None)
                    elif protect_grad is None:
                        next_current.append(focus_grad)
                    else:
                        next_current.append(
                            focus_grad - coeff.to(dtype=focus_grad.dtype, device=focus_grad.device) * protect_grad.detach()
                        )
            current = next_current

    scale = float(focus_weight) / max(int(accum), 1)
    adjusted = [
        None if grad is None else (float(scale) * grad).detach()
        for grad in current
    ]
    return adjusted, {
        "rscd_pcgrad_grouped_conflicts": float(conflicts),
        "rscd_pcgrad_grouped_protect_losses": float(protect_count),
        "rscd_pcgrad_grouped_conflict_rate": float(conflicts / max(protect_count, 1)),
        "rscd_pcgrad_grouped_dot_mean": float(dot_sum / max(protect_count, 1)),
        "rscd_pcgrad_grouped_dot_min": float(min_dot if min_dot is not None else 0.0),
        "rscd_pcgrad_grouped_focus_grad_norm": float(torch.sqrt(focus_norm.detach().clamp_min(0.0)).cpu()),
        "rscd_pcgrad_grouped_max_projection_coeff": float(max_projection),
        "rscd_pcgrad_focus_weight": float(focus_weight),
    }


def rscd_collect_protect_memory_gradient(
    params: list[torch.nn.Parameter],
    protect_loss: torch.Tensor | None,
    protect_losses: list[tuple[str, torch.Tensor]] | None = None,
) -> list[torch.Tensor | None] | None:
    """Collect a detached protected-memory gradient for RSCD A-GEM projection."""

    objective: torch.Tensor | None = None
    if protect_losses:
        objective = torch.stack([loss for _, loss in protect_losses]).mean()
    elif protect_loss is not None:
        objective = protect_loss
    if objective is None:
        return None
    grads = torch.autograd.grad(objective, params, retain_graph=True, allow_unused=True)
    return [None if grad is None else grad.detach().clone() for grad in grads]


def rscd_project_total_gradient_against_memory(
    params: list[torch.nn.Parameter],
    memory_grads: list[torch.Tensor | None] | None,
) -> dict[str, float]:
    """Project the total update so protected-sample loss does not increase."""

    if not memory_grads:
        return {
            "rscd_agem_total_projection_active": 0.0,
            "rscd_agem_total_projection_conflict": 0.0,
        }
    ref_grad = next((param.grad for param in params if param.grad is not None), None)
    if ref_grad is None:
        return {
            "rscd_agem_total_projection_active": 0.0,
            "rscd_agem_total_projection_conflict": 0.0,
        }
    dot = ref_grad.new_zeros(())
    grad_norm = ref_grad.new_zeros(())
    memory_norm = ref_grad.new_zeros(())
    with torch.no_grad():
        for param, memory_grad in zip(params, memory_grads):
            if param.grad is not None:
                grad_norm = grad_norm + param.grad.detach().pow(2).sum()
            if memory_grad is not None:
                memory_norm = memory_norm + memory_grad.detach().pow(2).sum()
            if param.grad is not None and memory_grad is not None:
                mem = memory_grad.to(device=param.grad.device, dtype=param.grad.dtype)
                dot = dot + (param.grad.detach() * mem).sum()
        memory_norm = memory_norm.clamp_min(1e-12)
        conflict = bool((dot < 0).detach().cpu())
        coeff = dot / memory_norm if conflict else dot.new_zeros(())
        if conflict:
            for param, memory_grad in zip(params, memory_grads):
                if param.grad is None or memory_grad is None:
                    continue
                mem = memory_grad.to(device=param.grad.device, dtype=param.grad.dtype)
                param.grad.sub_(coeff.to(device=param.grad.device, dtype=param.grad.dtype) * mem)
    return {
        "rscd_agem_total_projection_active": 1.0,
        "rscd_agem_total_projection_conflict": 1.0 if conflict else 0.0,
        "rscd_agem_total_projection_dot": float(dot.detach().cpu()),
        "rscd_agem_total_projection_coeff": float(coeff.detach().cpu()),
        "rscd_agem_total_projection_grad_norm": float(torch.sqrt(grad_norm.detach().clamp_min(0.0)).cpu()),
        "rscd_agem_total_projection_memory_norm": float(torch.sqrt(memory_norm.detach().clamp_min(0.0)).cpu()),
    }


def focus_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    extra = float(loss_cfg.get("focus_ce_extra_weight", 0.0))
    focus_classes = {canonical_class_label(name) for name in loss_cfg.get("focus_ce_classes", [])}
    if extra <= 0.0 or not focus_classes:
        return F.cross_entropy(logits, labels)
    focus_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in focus_classes
    }
    if not focus_idx:
        return F.cross_entropy(logits, labels)
    focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in focus_idx:
        focus_mask |= labels.eq(int(idx))
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    weights = 1.0 + float(extra) * focus_mask.to(dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def mechanism_feature_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
    model_out: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """CE with RSCD mechanism-aware weights derived from PhysicsTexture values.

    The weights are not a separate classifier. They emphasize known RSCD hard
    boundaries when the image evidence indicates that the relevant factor is
    visually ambiguous: hidden wet/water roughness, visible dry-concrete
    roughness, asphalt water-film brightness, and granular mud/gravel texture.
    """

    extra = float(loss_cfg.get("feature_mechanism_ce_extra_weight", 0.0))
    evidence = model_out.get("evidence_stats")
    if extra <= 0.0 or not isinstance(evidence, torch.Tensor):
        return focus_weighted_cross_entropy(logits, labels, idx_to_class, loss_cfg), {
            "feature_mechanism_ce_weight_mean": 1.0,
            "feature_mechanism_ce_active_rate": 0.0,
        }

    stats = evidence.float()
    jitter_std = float(loss_cfg.get("feature_mechanism_ce_jitter_std", 0.0))
    if logits.requires_grad and jitter_std > 0.0:
        stats = (stats + torch.randn_like(stats) * jitter_std).clamp(0.0, 1.0)
    gray_std = stats[:, 1].clamp(0.0, 1.0)
    sat_std = stats[:, 3].clamp(0.0, 1.0)
    grad_mean = stats[:, 4].clamp(0.0, 1.0)
    grad_std = stats[:, 5].clamp(0.0, 1.0)
    lap_mean = stats[:, 6].clamp(0.0, 1.0)
    contrast_mean = stats[:, 7].clamp(0.0, 1.0)
    specular = stats[:, 8].clamp(0.0, 1.0)
    dark_water = stats[:, 9].clamp(0.0, 1.0)
    wet = stats[:, 10].clamp(0.0, 1.0)
    rough = stats[:, 11].clamp(0.0, 1.0)
    erasure = stats[:, 12].clamp(0.0, 1.0)
    snow_ice = torch.maximum(stats[:, 13], stats[:, 14]).clamp(0.0, 1.0)

    wet_film = torch.clamp(0.50 * wet + 0.25 * dark_water + 0.15 * specular + 0.10 * erasure, 0.0, 1.0)
    hidden_roughness = wet_film * torch.sigmoid((0.075 - rough) * 35.0) * torch.sigmoid((0.32 - snow_ice) * 18.0)
    visible_roughness = torch.clamp(0.35 * rough + 0.25 * grad_std + 0.20 * lap_mean + 0.20 * contrast_mean, 0.0, 1.0)
    dry_rough_ambiguity = (4.0 * visible_roughness * (1.0 - visible_roughness)).clamp(0.0, 1.0)
    asphalt_water_film = torch.clamp(0.45 * dark_water + 0.25 * wet + 0.20 * gray_std + 0.10 * sat_std, 0.0, 1.0)
    granular_texture = torch.clamp(0.35 * grad_std + 0.35 * lap_mean + 0.20 * contrast_mean + 0.10 * grad_mean, 0.0, 1.0)

    class_to_idx = {canonical_class_label(name): int(idx) for idx, name in idx_to_class.items()}

    def class_mask(names: tuple[str, ...]) -> torch.Tensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for name in names:
            idx = class_to_idx.get(canonical_class_label(name))
            if idx is not None:
                mask |= labels.eq(int(idx))
        return mask

    wet_water_concrete = class_mask(
        (
            "water_concrete_slight",
            "water_concrete_severe",
            "water_concrete_smooth",
            "wet_concrete_slight",
            "wet_concrete_severe",
            "wet_concrete_smooth",
        )
    )
    dry_concrete = class_mask(("dry_concrete_slight", "dry_concrete_severe", "dry_concrete_smooth"))
    water_asphalt = class_mask(
        (
            "water_asphalt_slight",
            "water_asphalt_severe",
            "water_asphalt_smooth",
            "wet_asphalt_slight",
            "wet_asphalt_severe",
        )
    )
    granular = class_mask(("water_mud", "water_gravel", "dry_mud", "dry_gravel", "wet_mud", "wet_gravel"))

    weights = logits.new_ones(labels.shape, dtype=torch.float32)
    weights = weights + float(extra) * hidden_roughness.to(device=labels.device) * wet_water_concrete.float()
    weights = weights + float(extra) * dry_rough_ambiguity.to(device=labels.device) * dry_concrete.float()
    weights = weights + float(extra) * asphalt_water_film.to(device=labels.device) * water_asphalt.float()
    weights = weights + float(extra) * granular_texture.to(device=labels.device) * granular.float()

    max_weight = max(float(loss_cfg.get("feature_mechanism_ce_max_weight", 1.8)), 1.0)
    weights = weights.clamp(1.0, max_weight).to(device=logits.device, dtype=logits.dtype)
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
    active = weights.detach().gt(1.001)
    logs = {
        "feature_mechanism_ce_weight_mean": float(weights.detach().mean().cpu()),
        "feature_mechanism_ce_active_rate": float(active.float().mean().cpu()),
        "feature_mechanism_ce_hidden_rough_mean": float(hidden_roughness.detach().mean().cpu()),
        "feature_mechanism_ce_visible_rough_mean": float(visible_roughness.detach().mean().cpu()),
    }
    return loss, logs


def anchor_error_gate_loss(
    model_out: dict[str, Any],
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise learned pair gates with anchor pair-confusion targets."""

    weight = float(loss_cfg.get("anchor_error_gate_weight", 0.0))
    gate_logits = model_out.get("hardpair_error_gate_logits", {})
    if weight <= 0.0 or not isinstance(gate_logits, dict) or not gate_logits:
        return model_out["logits"].new_zeros(()), {"loss_anchor_error_gate": 0.0, "anchor_error_gate_count": 0.0}
    teacher_pred = teacher_logits.detach().argmax(dim=1)
    pos_weight_value = max(float(loss_cfg.get("anchor_error_gate_pos_weight", 8.0)), 1.0)
    pair_error_only = bool(loss_cfg.get("anchor_error_gate_pair_error_only", True))
    losses: list[torch.Tensor] = []
    gate_means: list[torch.Tensor] = []
    pos_count = 0
    total_count = 0
    for pair in spec.hard_pairs:
        key = f"p{int(pair.left)}_{int(pair.right)}"
        logit = gate_logits.get(key)
        if not isinstance(logit, torch.Tensor):
            continue
        left = int(pair.left)
        right = int(pair.right)
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        if pair_error_only:
            target = (mask_left & teacher_pred.eq(right)) | (mask_right & teacher_pred.eq(left))
        else:
            target = (mask_left | mask_right) & teacher_pred.ne(labels)
        idx = mask.nonzero(as_tuple=False).flatten()
        target_slice = target.index_select(0, idx).float()
        logit_slice = logit.index_select(0, idx).float()
        pos_weight = torch.as_tensor(pos_weight_value, device=logit_slice.device, dtype=logit_slice.dtype)
        losses.append(F.binary_cross_entropy_with_logits(logit_slice, target_slice, pos_weight=pos_weight))
        gate_means.append(torch.sigmoid(logit_slice.detach()).mean())
        pos_count += int(target_slice.detach().sum().cpu())
        total_count += int(target_slice.numel())
    if not losses:
        return model_out["logits"].new_zeros(()), {"loss_anchor_error_gate": 0.0, "anchor_error_gate_count": 0.0}
    loss = torch.stack(losses).mean().to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_anchor_error_gate": float(loss.detach().cpu()),
        "anchor_error_gate_count": float(total_count),
        "anchor_error_gate_pos_count": float(pos_count),
        "anchor_error_gate_pos_rate": float(pos_count / max(total_count, 1)),
    }
    if gate_means:
        logs["anchor_error_gate_mean"] = float(torch.stack(gate_means).mean().cpu())
    return float(weight) * loss, logs


def hardpair_margin_directed_loss(
    model_out: dict[str, Any],
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Push true-vs-confused hard-pair logits with explicit signed margins."""

    weight = float(loss_cfg.get("hardpair_margin_loss_weight", 0.0))
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {"loss_hardpair_margin": 0.0, "hardpair_margin_count": 0.0}
    logits = model_out["logits"].float()
    teacher_pred = teacher_logits.detach().argmax(dim=1)
    target_margin = float(loss_cfg.get("hardpair_margin_target", 0.75))
    pair_error_only = bool(loss_cfg.get("hardpair_margin_teacher_pair_only", True))
    direction_weight = float(loss_cfg.get("hardpair_margin_direction_weight", 0.0))
    keep_weight = float(loss_cfg.get("hardpair_margin_keep_weight", 0.0))
    margin_delta = model_out.get("hardpair_margin_delta", {})
    losses: list[torch.Tensor] = []
    direction_losses: list[torch.Tensor] = []
    keep_losses: list[torch.Tensor] = []
    pos_count = 0
    total_count = 0
    margin_sum = 0.0
    delta_sum = 0.0
    delta_count = 0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        pair_mask = mask_left | mask_right
        if not bool(pair_mask.any()):
            continue
        if pair_error_only:
            focus_mask = (mask_left & teacher_pred.eq(right)) | (mask_right & teacher_pred.eq(left))
        else:
            focus_mask = pair_mask & teacher_pred.ne(labels)
        if bool(focus_mask.any()):
            sign = torch.where(labels.eq(left), 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
            signed_margin = sign * (logits[:, left] - logits[:, right])
            selected_margin = signed_margin.index_select(0, focus_mask.nonzero(as_tuple=False).flatten())
            losses.append(F.relu(float(target_margin) - selected_margin).pow(2).mean())
            pos_count += int(focus_mask.sum().detach().cpu())
            margin_sum += float(selected_margin.detach().sum().cpu())
            key = f"p{left}_{right}"
            delta = margin_delta.get(key) if isinstance(margin_delta, dict) else None
            if direction_weight > 0.0 and isinstance(delta, torch.Tensor):
                selected_sign = sign.index_select(0, focus_mask.nonzero(as_tuple=False).flatten())
                selected_delta = delta.float().index_select(0, focus_mask.nonzero(as_tuple=False).flatten())
                direction_losses.append(F.relu(0.02 - selected_sign * selected_delta).pow(2).mean())
                delta_sum += float(selected_delta.detach().abs().sum().cpu())
                delta_count += int(selected_delta.numel())
        if keep_weight > 0.0:
            key = f"p{left}_{right}"
            delta = margin_delta.get(key) if isinstance(margin_delta, dict) else None
            keep_mask = pair_mask & teacher_pred.eq(labels)
            if isinstance(delta, torch.Tensor) and bool(keep_mask.any()):
                keep_losses.append(delta.float().index_select(0, keep_mask.nonzero(as_tuple=False).flatten()).pow(2).mean())
        total_count += int(pair_mask.sum().detach().cpu())
    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_hardpair_margin": 0.0,
            "hardpair_margin_count": float(total_count),
            "hardpair_margin_pos_count": 0.0,
        }
    loss = torch.stack(losses).mean()
    if direction_losses:
        loss = loss + float(direction_weight) * torch.stack(direction_losses).mean()
    if keep_losses:
        loss = loss + float(keep_weight) * torch.stack(keep_losses).mean()
    loss = loss.to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_hardpair_margin": float(loss.detach().cpu()),
        "hardpair_margin_count": float(total_count),
        "hardpair_margin_pos_count": float(pos_count),
        "hardpair_margin_pos_rate": float(pos_count / max(total_count, 1)),
        "hardpair_margin_selected_mean": float(margin_sum / max(pos_count, 1)),
    }
    if delta_count > 0:
        logs["hardpair_margin_abs_delta_mean"] = float(delta_sum / max(delta_count, 1))
    return float(weight) * loss, logs


def dry_concrete_bidirectional_ordinal_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced two-sided ordinal loss for dry-concrete roughness boundaries.

    The feature-value screens showed a one-sided failure: correcting
    `dry_concrete_slight` from `dry_concrete_severe` can also push true severe
    samples into slight. This loss treats each configured dry-concrete hard
    pair as a bidirectional comparator and averages the two class-side losses,
    so both roughness directions must keep a margin.
    """

    weight = float(loss_cfg.get("dry_concrete_bidirectional_ordinal_weight", 0.0))
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {
            "loss_dry_concrete_bidirectional": 0.0,
            "dry_concrete_bidirectional_count": 0.0,
        }

    logits = model_out["logits"].float()
    margin = float(loss_cfg.get("dry_concrete_bidirectional_margin", 0.55))
    low_margin = float(loss_cfg.get("dry_concrete_bidirectional_low_margin_threshold", -1.0))
    delta_weight = float(loss_cfg.get("dry_concrete_bidirectional_delta_weight", 0.0))
    delta_margin = float(loss_cfg.get("dry_concrete_bidirectional_delta_margin", 0.015))
    pair_specs = loss_cfg.get(
        "dry_concrete_bidirectional_pairs",
        ["dry_concrete_severe|dry_concrete_slight"],
    )
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]

    requested_pairs: set[frozenset[str]] = set()
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) != 2:
            continue
        requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
    if not requested_pairs:
        return model_out["logits"].new_zeros(()), {
            "loss_dry_concrete_bidirectional": 0.0,
            "dry_concrete_bidirectional_count": 0.0,
        }

    idx_to_class = {idx: name for name, idx in spec.class_to_idx.items()}
    deltas = model_out.get("hardpair_margin_delta", {})
    losses: list[torch.Tensor] = []
    delta_losses: list[torch.Tensor] = []
    total = 0
    correct = 0
    side_terms = 0
    signed_gap_sum = 0.0
    selected_pair_count = 0

    for pair in spec.hard_pairs:
        if pair.axis != "roughness":
            continue
        left = int(pair.left)
        right = int(pair.right)
        left_name = canonical_class_label(idx_to_class[left])
        right_name = canonical_class_label(idx_to_class[right])
        if frozenset((left_name, right_name)) not in requested_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        pair_mask = mask_left | mask_right
        if not bool(pair_mask.any()):
            continue
        sign = torch.where(mask_left, 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
        signed_gap = sign * (logits[:, left] - logits[:, right])
        focus_mask = pair_mask
        if low_margin >= 0.0:
            focus_mask = focus_mask & signed_gap.detach().le(low_margin)
        if not bool(focus_mask.any()):
            continue
        side_losses: list[torch.Tensor] = []
        for side_mask in (mask_left & focus_mask, mask_right & focus_mask):
            if bool(side_mask.any()):
                side_gap = signed_gap.index_select(0, side_mask.nonzero(as_tuple=False).flatten())
                side_losses.append(F.softplus(float(margin) - side_gap).mean())
                signed_gap_sum += float(side_gap.detach().sum().cpu())
                side_terms += int(side_gap.numel())
        if side_losses:
            losses.append(torch.stack(side_losses).mean())
            selected_pair_count += 1
        if delta_weight > 0.0 and isinstance(deltas, dict):
            key = f"p{left}_{right}"
            delta = deltas.get(key)
            if isinstance(delta, torch.Tensor):
                idx = focus_mask.nonzero(as_tuple=False).flatten()
                delta_slice = delta.float().index_select(0, idx)
                sign_slice = sign.index_select(0, idx)
                delta_losses.append(F.softplus(float(delta_margin) - sign_slice * delta_slice).mean())
        pair_idx = pair_mask.nonzero(as_tuple=False).flatten()
        pred_left = (logits[:, left] - logits[:, right]).index_select(0, pair_idx).ge(0.0)
        true_left = mask_left.index_select(0, pair_idx)
        correct += int(pred_left.eq(true_left).sum().detach().cpu())
        total += int(pair_idx.numel())

    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_dry_concrete_bidirectional": 0.0,
            "dry_concrete_bidirectional_count": float(total),
        }
    loss = torch.stack(losses).mean()
    if delta_losses:
        loss = loss + float(delta_weight) * torch.stack(delta_losses).mean()
    loss = loss.to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_dry_concrete_bidirectional": float(loss.detach().cpu()),
        "dry_concrete_bidirectional_count": float(total),
        "dry_concrete_bidirectional_focus_count": float(side_terms),
        "dry_concrete_bidirectional_pair_count": float(selected_pair_count),
        "dry_concrete_bidirectional_pair_acc": float(correct / max(total, 1)),
        "dry_concrete_bidirectional_signed_gap_mean": float(signed_gap_sum / max(side_terms, 1)),
    }
    return float(weight) * loss, logs


def hardpair_binary_tournament_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train active hard-pair heads as RSCD left/right binary comparators."""

    weight = float(loss_cfg.get("hardpair_binary_loss_weight", 0.0))
    raw_logits = model_out.get("hardpair_margin_raw", {})
    if weight <= 0.0 or not isinstance(raw_logits, dict) or not raw_logits:
        return model_out["logits"].new_zeros(()), {"loss_hardpair_binary": 0.0, "hardpair_binary_count": 0.0}
    losses: list[torch.Tensor] = []
    total = 0
    correct = 0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        key = f"p{left}_{right}"
        raw = raw_logits.get(key)
        if not isinstance(raw, torch.Tensor):
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).flatten()
        raw_slice = raw.float().index_select(0, idx)
        target = mask_left.float().index_select(0, idx)
        losses.append(F.binary_cross_entropy_with_logits(raw_slice, target))
        pred_left = raw_slice.ge(0.0)
        correct += int(pred_left.eq(target.bool()).sum().detach().cpu())
        total += int(target.numel())
    if not losses:
        return model_out["logits"].new_zeros(()), {"loss_hardpair_binary": 0.0, "hardpair_binary_count": 0.0}
    loss = torch.stack(losses).mean().to(dtype=model_out["logits"].dtype)
    return float(weight) * loss, {
        "loss_hardpair_binary": float(loss.detach().cpu()),
        "hardpair_binary_count": float(total),
        "hardpair_binary_acc": float(correct / max(total, 1)),
    }


def hardpair_value_adapter_pairwise_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise value adapters as pair-specific left/right classifiers."""

    weight = float(loss_cfg.get("hardpair_value_pairwise_loss_weight", 0.0))
    raw_logits = model_out.get("hardpair_value_adapter_logits", {})
    if weight <= 0.0 or not isinstance(raw_logits, dict) or not raw_logits:
        return model_out["logits"].new_zeros(()), {
            "loss_hardpair_value_pairwise": 0.0,
            "hardpair_value_pairwise_count": 0.0,
        }

    idx_to_class = {int(idx): canonical_class_label(name) for name, idx in spec.class_to_idx.items()}
    pair_weights_cfg = loss_cfg.get("hardpair_value_pairwise_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(pair_weight)
    loss_pairs_cfg = loss_cfg.get("hardpair_value_pairwise_loss_pairs", None)
    loss_pairs: set[frozenset[str]] = set()
    for item in loss_pairs_cfg or []:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            loss_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

    losses: list[torch.Tensor] = []
    weighted_terms: list[torch.Tensor] = []
    total_weight = 0.0
    total = 0
    correct = 0
    active_pairs = 0
    logit_abs_sum = 0.0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        key = f"p{left}_{right}"
        raw = raw_logits.get(key)
        if not isinstance(raw, torch.Tensor):
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).flatten()
        raw_slice = raw.float().index_select(0, idx)
        target = mask_left.float().index_select(0, idx)
        term = F.binary_cross_entropy_with_logits(raw_slice, target)
        left_name = idx_to_class.get(left, str(left))
        right_name = idx_to_class.get(right, str(right))
        pair_names = frozenset((left_name, right_name))
        if loss_pairs and pair_names not in loss_pairs:
            continue
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        losses.append(term)
        weighted_terms.append(term * pair_weight)
        total_weight += pair_weight
        pred_left = raw_slice.ge(0.0)
        correct += int(pred_left.eq(target.bool()).sum().detach().cpu())
        total += int(target.numel())
        active_pairs += 1
        logit_abs_sum += float(raw_slice.detach().abs().sum().cpu())

    if not weighted_terms:
        return model_out["logits"].new_zeros(()), {
            "loss_hardpair_value_pairwise": 0.0,
            "hardpair_value_pairwise_count": 0.0,
        }
    loss = (torch.stack(weighted_terms).sum() / max(total_weight, 1e-6)).to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_hardpair_value_pairwise": float(torch.stack(losses).mean().detach().cpu()),
        "loss_hardpair_value_pairwise_weighted": float(loss.detach().cpu()),
        "hardpair_value_pairwise_count": float(total),
        "hardpair_value_pairwise_pair_count": float(active_pairs),
        "hardpair_value_pairwise_acc": float(correct / max(total, 1)),
        "hardpair_value_pairwise_abs_logit_mean": float(logit_abs_sum / max(total, 1)),
    }
    return float(weight) * loss, logs


def feature_value_boundary_pairwise_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise feature-value boundary correctors on diagnosed hard pairs.

    The corrector is intentionally pair-local: a positive raw logit means the
    sample should move toward the left class in `spec.hard_pairs`, while a
    negative raw logit means it should move toward the right class.
    """

    weight = float(loss_cfg.get("feature_value_boundary_pairwise_loss_weight", 0.0))
    raw_logits = model_out.get("feature_value_boundary_logits", {})
    if weight <= 0.0 or not isinstance(raw_logits, dict) or not raw_logits:
        return model_out["logits"].new_zeros(()), {
            "loss_feature_value_boundary_pairwise": 0.0,
            "feature_value_boundary_pairwise_count": 0.0,
        }

    idx_to_class = {int(idx): canonical_class_label(name) for name, idx in spec.class_to_idx.items()}
    pair_weights_cfg = loss_cfg.get("feature_value_boundary_pairwise_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(pair_weight)
    loss_pairs_cfg = loss_cfg.get("feature_value_boundary_pairwise_loss_pairs", None)
    loss_pairs: set[frozenset[str]] = set()
    for item in loss_pairs_cfg or []:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            loss_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

    losses: list[torch.Tensor] = []
    weighted_terms: list[torch.Tensor] = []
    total_weight = 0.0
    total = 0
    correct = 0
    active_pairs = 0
    logit_abs_sum = 0.0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        key = f"p{left}_{right}"
        raw = raw_logits.get(key)
        if not isinstance(raw, torch.Tensor):
            continue
        left_name = idx_to_class.get(left, str(left))
        right_name = idx_to_class.get(right, str(right))
        pair_names = frozenset((left_name, right_name))
        if loss_pairs and pair_names not in loss_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).flatten()
        raw_slice = raw.float().index_select(0, idx)
        target = mask_left.float().index_select(0, idx)
        term = F.binary_cross_entropy_with_logits(raw_slice, target)
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        losses.append(term)
        weighted_terms.append(term * pair_weight)
        total_weight += pair_weight
        pred_left = raw_slice.ge(0.0)
        correct += int(pred_left.eq(target.bool()).sum().detach().cpu())
        total += int(target.numel())
        active_pairs += 1
        logit_abs_sum += float(raw_slice.detach().abs().sum().cpu())

    if not weighted_terms:
        return model_out["logits"].new_zeros(()), {
            "loss_feature_value_boundary_pairwise": 0.0,
            "feature_value_boundary_pairwise_count": 0.0,
        }
    loss = (torch.stack(weighted_terms).sum() / max(total_weight, 1e-6)).to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_feature_value_boundary_pairwise": float(torch.stack(losses).mean().detach().cpu()),
        "loss_feature_value_boundary_pairwise_weighted": float(loss.detach().cpu()),
        "feature_value_boundary_pairwise_count": float(total),
        "feature_value_boundary_pairwise_pair_count": float(active_pairs),
        "feature_value_boundary_pairwise_acc": float(correct / max(total, 1)),
        "feature_value_boundary_pairwise_abs_logit_mean": float(logit_abs_sum / max(total, 1)),
    }
    return float(weight) * loss, logs


def water_concrete_opponent_feature_pairwise_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise feature-space opponent axes before the RSCD decoder.

    This loss is intentionally paired with `WaterConcreteOpponentFeatureConditioner`.
    A positive raw logit means the sample should move toward the left class of a
    hard pair; a negative raw logit means it should move toward the right class.
    Unlike S96's final-logit corrector, the supervised signal shapes the fused
    feature that feeds both factor tokens and the calibrated class head.
    """

    weight = float(loss_cfg.get("water_concrete_opponent_pairwise_loss_weight", 0.0))
    raw_logits = model_out.get("water_concrete_opponent_feature_logits", {})
    if weight <= 0.0 or not isinstance(raw_logits, dict) or not raw_logits:
        return model_out["logits"].new_zeros(()), {
            "loss_water_concrete_opponent_pairwise": 0.0,
            "water_concrete_opponent_pairwise_count": 0.0,
        }

    idx_to_class = {int(idx): canonical_class_label(name) for name, idx in spec.class_to_idx.items()}
    pair_specs = loss_cfg.get("water_concrete_opponent_pairwise_loss_pairs", [])
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]
    requested_pairs: set[frozenset[str]] = set()
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

    pair_weights_cfg = loss_cfg.get("water_concrete_opponent_pairwise_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(
                pair_weight
            )

    weighted_terms: list[torch.Tensor] = []
    raw_terms: list[torch.Tensor] = []
    total_weight = 0.0
    total = 0
    correct = 0
    active_pairs = 0
    logit_abs_sum = 0.0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        key = f"p{left}_{right}"
        raw = raw_logits.get(key)
        if not isinstance(raw, torch.Tensor):
            continue
        pair_names = frozenset((idx_to_class.get(left, str(left)), idx_to_class.get(right, str(right))))
        if requested_pairs and pair_names not in requested_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).flatten()
        raw_slice = raw.float().index_select(0, idx)
        target = mask_left.float().index_select(0, idx)
        term = F.binary_cross_entropy_with_logits(raw_slice, target)
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        weighted_terms.append(term * pair_weight)
        raw_terms.append(term)
        total_weight += pair_weight
        pred_left = raw_slice.ge(0.0)
        correct += int(pred_left.eq(target.bool()).sum().detach().cpu())
        total += int(target.numel())
        active_pairs += 1
        logit_abs_sum += float(raw_slice.detach().abs().sum().cpu())

    if not weighted_terms:
        return model_out["logits"].new_zeros(()), {
            "loss_water_concrete_opponent_pairwise": 0.0,
            "water_concrete_opponent_pairwise_count": 0.0,
        }
    loss = (torch.stack(weighted_terms).sum() / max(total_weight, 1e-6)).to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_water_concrete_opponent_pairwise": float(torch.stack(raw_terms).mean().detach().cpu()),
        "loss_water_concrete_opponent_pairwise_weighted": float(loss.detach().cpu()),
        "water_concrete_opponent_pairwise_count": float(total),
        "water_concrete_opponent_pairwise_pair_count": float(active_pairs),
        "water_concrete_opponent_pairwise_acc": float(correct / max(total, 1)),
        "water_concrete_opponent_pairwise_abs_logit_mean": float(logit_abs_sum / max(total, 1)),
    }
    return float(weight) * loss, logs


def value_guided_roughness_order_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
    idx_to_class: dict[int, str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Guide roughness factor ordering using physically visible texture evidence.

    This is deliberately not a post-logit correction. It supervises the
    intermediate roughness factor score so token-level roughness conditioning
    learns the RSCD ordinal relation smooth < slight < severe. The per-sample
    weight is lowered when wet-film/texture-erasure evidence suggests that
    roughness is visually unreliable.
    """

    weight = float(loss_cfg.get("value_guided_roughness_order_weight", 0.0))
    factor_logits = model_out.get("factor_logits", {})
    rough_logits = factor_logits.get("roughness") if isinstance(factor_logits, dict) else None
    if weight <= 0.0 or not isinstance(rough_logits, torch.Tensor):
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_roughness_order": 0.0,
            "value_guided_roughness_order_count": 0.0,
        }

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    roughness = factors[:, 2]
    valid = torch.isin(roughness, torch.as_tensor([1, 2, 3], device=labels.device, dtype=roughness.dtype))
    focus_classes = {canonical_class_label(name) for name in loss_cfg.get("value_guided_roughness_order_classes", [])}
    if focus_classes:
        focus_idx = {
            int(idx)
            for idx, name in idx_to_class.items()
            if canonical_class_label(name) in focus_classes
        }
        focus_mask = torch.zeros_like(valid)
        for idx in focus_idx:
            focus_mask |= labels.eq(int(idx))
        valid = valid & focus_mask
    focus_friction = {str(item) for item in loss_cfg.get("value_guided_roughness_order_friction", [])}
    if focus_friction:
        friction_labels = FACTOR_LABELS["friction"]
        allowed = {
            idx
            for idx, name in enumerate(friction_labels)
            if str(name) in focus_friction
        }
        friction = factors[:, 0]
        friction_mask = torch.zeros_like(valid)
        for idx in allowed:
            friction_mask |= friction.eq(int(idx))
        valid = valid & friction_mask
    focus_material = {str(item) for item in loss_cfg.get("value_guided_roughness_order_material", [])}
    if focus_material:
        material_labels = FACTOR_LABELS["material"]
        allowed = {
            idx
            for idx, name in enumerate(material_labels)
            if str(name) in focus_material
        }
        material = factors[:, 1]
        material_mask = torch.zeros_like(valid)
        for idx in allowed:
            material_mask |= material.eq(int(idx))
        valid = valid & material_mask
    if not bool(valid.any()):
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_roughness_order": 0.0,
            "value_guided_roughness_order_count": 0.0,
        }

    idx = valid.nonzero(as_tuple=False).flatten()
    logits = rough_logits.float().index_select(0, idx)
    target = roughness.index_select(0, idx)
    evidence = model_out.get("evidence_stats")
    if isinstance(evidence, torch.Tensor):
        ev = evidence.float().index_select(0, idx)
        rough = ev[:, 11].clamp(0.0, 1.0)
        wet = ev[:, 10].clamp(0.0, 1.0)
        dark_water = ev[:, 9].clamp(0.0, 1.0)
        specular = ev[:, 8].clamp(0.0, 1.0)
        erasure = ev[:, 12].clamp(0.0, 1.0)
        visible = torch.sigmoid((rough - 0.018) * 130.0)
        occlusion_guard = torch.sigmoid((0.58 - erasure - 0.35 * wet - 0.25 * dark_water - 0.20 * specular) * 7.0)
        phys_weight = (visible * occlusion_guard).clamp(0.0, 1.0)
    else:
        phys_weight = logits.new_ones((idx.numel(),))
    rho = model_out.get("rho_roughness")
    if isinstance(rho, torch.Tensor):
        rho_weight = rho.float().view(-1).index_select(0, idx).clamp(0.0, 1.0)
        phys_weight = torch.maximum(phys_weight, 0.35 * rho_weight)
    min_weight = float(loss_cfg.get("value_guided_roughness_order_min_weight", 0.12))
    phys_weight = (min_weight + (1.0 - min_weight) * phys_weight).clamp(min_weight, 1.0)

    margin = float(loss_cfg.get("value_guided_roughness_order_margin", 0.55))
    slight_margin_scale = float(loss_cfg.get("value_guided_roughness_order_slight_margin_scale", 0.65))
    smooth = logits[:, 1]
    slight = logits[:, 2]
    severe = logits[:, 3]
    sample_losses: list[torch.Tensor] = []
    sample_weights: list[torch.Tensor] = []

    mask = target.eq(1)
    if bool(mask.any()):
        local_margin = margin * phys_weight[mask]
        sample_losses.append(
            0.5
            * (
                F.softplus(local_margin - (smooth[mask] - slight[mask]))
                + F.softplus(local_margin - (smooth[mask] - severe[mask]))
            )
        )
        sample_weights.append(phys_weight[mask])

    mask = target.eq(2)
    if bool(mask.any()):
        local_margin = margin * slight_margin_scale * phys_weight[mask]
        sample_losses.append(
            0.5
            * (
                F.softplus(local_margin - (slight[mask] - smooth[mask]))
                + F.softplus(local_margin - (slight[mask] - severe[mask]))
            )
        )
        sample_weights.append(phys_weight[mask])

    mask = target.eq(3)
    if bool(mask.any()):
        local_margin = margin * phys_weight[mask]
        sample_losses.append(
            0.5
            * (
                F.softplus(local_margin - (severe[mask] - slight[mask]))
                + F.softplus(local_margin - (severe[mask] - smooth[mask]))
            )
        )
        sample_weights.append(phys_weight[mask])

    if not sample_losses:
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_roughness_order": 0.0,
            "value_guided_roughness_order_count": 0.0,
        }
    losses = torch.cat(sample_losses)
    weights = torch.cat(sample_weights).to(device=losses.device, dtype=losses.dtype)
    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    pred = logits[:, 1:4].argmax(dim=1) + 1
    logs = {
        "loss_value_guided_roughness_order": float(loss.detach().cpu()),
        "value_guided_roughness_order_count": float(idx.numel()),
        "value_guided_roughness_order_weight_mean": float(phys_weight.detach().mean().cpu()),
        "value_guided_roughness_order_acc": float(pred.eq(target).float().mean().detach().cpu()),
    }
    return float(weight) * loss.to(dtype=model_out["logits"].dtype), logs


def protected_tristate_roughness_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Ordinal class-logit loss for RSCD smooth/slight/severe triplets.

    The previous pair-value margin can help local boundaries, but it can also
    push severe concrete samples into the adjacent slight class. This loss
    treats roughness as a three-state ordered variable inside one fixed
    friction/material group and uses physical value evidence only as a
    reliability signal, not as a direct classifier.
    """

    weight = float(loss_cfg.get("protected_tristate_roughness_weight", 0.0))
    logits = model_out["logits"].float()
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {
            "loss_protected_tristate_roughness": 0.0,
            "protected_tristate_roughness_count": 0.0,
        }

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    friction = factors[:, 0]
    material = factors[:, 1]
    roughness = factors[:, 2]
    valid = torch.isin(roughness, torch.as_tensor([1, 2, 3], device=labels.device, dtype=roughness.dtype))

    group_specs = loss_cfg.get("protected_tristate_roughness_groups", [])
    if isinstance(group_specs, str):
        group_specs = [group_specs]
    if group_specs:
        friction_labels = FACTOR_LABELS["friction"]
        material_labels = FACTOR_LABELS["material"]
        allowed: set[tuple[int, int]] = set()
        for item in group_specs:
            parts = str(item).replace("/", "|").replace(",", "|").split("|")
            if len(parts) != 2:
                continue
            f_name = canonical_class_label(parts[0])
            m_name = canonical_class_label(parts[1])
            if f_name in friction_labels and m_name in material_labels:
                allowed.add((friction_labels.index(f_name), material_labels.index(m_name)))
        if allowed:
            group_mask = torch.zeros_like(valid)
            for f_idx, m_idx in allowed:
                group_mask |= friction.eq(int(f_idx)) & material.eq(int(m_idx))
            valid = valid & group_mask

    grid = spec.class_index_grid.to(device=labels.device)
    smooth_idx = grid[friction.clamp_min(0), material.clamp_min(0), torch.ones_like(roughness)]
    slight_idx = grid[friction.clamp_min(0), material.clamp_min(0), torch.full_like(roughness, 2)]
    severe_idx = grid[friction.clamp_min(0), material.clamp_min(0), torch.full_like(roughness, 3)]
    valid = valid & smooth_idx.ge(0) & slight_idx.ge(0) & severe_idx.ge(0)
    if not bool(valid.any()):
        return model_out["logits"].new_zeros(()), {
            "loss_protected_tristate_roughness": 0.0,
            "protected_tristate_roughness_count": 0.0,
        }

    idx = valid.nonzero(as_tuple=False).flatten()
    group_indices = torch.stack(
        [
            smooth_idx.index_select(0, idx),
            slight_idx.index_select(0, idx),
            severe_idx.index_select(0, idx),
        ],
        dim=1,
    )
    group_logits = logits.gather(1, group_indices)
    target_rough = roughness.index_select(0, idx)
    target_rank = target_rough - 1

    values = model_out.get("hardpair_pair_value_evidence_vector")
    if isinstance(values, torch.Tensor):
        v = values.float().index_select(0, idx).clamp(0.0, 1.0)
        macro_rough = v[:, 0]
        micro_rough = v[:, 1]
        film = v[:, 2]
        artifact = v[:, 3]
        saturation = v[:, 4]
        macro_mean = v[:, 5]
        macro_std = v[:, 6]
        meso_std = v[:, 7]
        micro_std = v[:, 8]
        lap_std = v[:, 9]
        grad_std = v[:, 10]
        dark_water = v[:, 12]
        dark_water_top = v[:, 13]
        texture_erasure = v[:, 16]
        texture_erasure_top = v[:, 17]
        visible_rough = (
            0.30 * macro_rough
            + 0.18 * macro_std
            + 0.16 * meso_std
            + 0.12 * micro_rough
            + 0.10 * micro_std
            + 0.09 * lap_std
            + 0.05 * grad_std
        ).clamp(0.0, 1.0)
        film_occlusion = (
            0.34 * film
            + 0.24 * dark_water_top
            + 0.18 * dark_water
            + 0.14 * texture_erasure_top
            + 0.10 * texture_erasure
        ).clamp(0.0, 1.0)
        concrete_visibility = (0.55 * macro_mean + 0.25 * saturation + 0.20 * (1.0 - film)).clamp(0.0, 1.0)
        artifact_guard = (1.0 - 0.72 * artifact).clamp(0.10, 1.0)
    else:
        visible_rough = group_logits.new_full((idx.numel(),), 0.5)
        film_occlusion = group_logits.new_zeros((idx.numel(),))
        concrete_visibility = group_logits.new_full((idx.numel(),), 0.5)
        artifact_guard = group_logits.new_ones((idx.numel(),))

    selected_friction = friction.index_select(0, idx)
    wet_or_water = selected_friction.eq(1) | selected_friction.eq(2)
    dry = selected_friction.eq(0)
    wet_visibility_guard = torch.where(
        wet_or_water,
        (0.42 + 0.58 * (1.0 - film_occlusion)).clamp(0.12, 1.0),
        torch.ones_like(film_occlusion),
    )
    dry_visibility_boost = torch.where(dry, 1.0 + 0.20 * concrete_visibility, torch.ones_like(concrete_visibility))
    reliability = (artifact_guard * wet_visibility_guard * dry_visibility_boost).clamp(0.08, 1.0)
    rho = model_out.get("rho_roughness")
    if isinstance(rho, torch.Tensor):
        rho_weight = rho.float().view(-1).index_select(0, idx).clamp(0.0, 1.0)
        reliability = torch.maximum(reliability, 0.30 * rho_weight)

    min_weight = float(loss_cfg.get("protected_tristate_roughness_min_weight", 0.10))
    margin = float(loss_cfg.get("protected_tristate_roughness_margin", 0.48))
    severe_boost = float(loss_cfg.get("protected_tristate_roughness_severe_boost", 0.55))
    slight_protect = float(loss_cfg.get("protected_tristate_roughness_slight_protect", 0.55))
    smooth_score = ((1.0 - visible_rough) * (0.60 + 0.40 * film_occlusion)).clamp(0.0, 1.0)
    severe_score = (visible_rough * (1.0 - 0.45 * film_occlusion) * artifact_guard).clamp(0.0, 1.0)
    slight_score = (4.0 * visible_rough * (1.0 - visible_rough)).clamp(0.0, 1.0)

    smooth_logit = group_logits[:, 0]
    slight_logit = group_logits[:, 1]
    severe_logit = group_logits[:, 2]
    context_gate_enabled = bool(loss_cfg.get("protected_tristate_roughness_use_context_gate", False))
    if context_gate_enabled:
        with torch.no_grad():
            full_prob = F.softmax(logits, dim=1)
            group_mass = full_prob.gather(1, group_indices).sum(dim=1).clamp(0.0, 1.0)
            target_group_logit = group_logits.gather(1, target_rank.view(-1, 1)).squeeze(1)
            other_group_logits = group_logits.masked_fill(
                F.one_hot(target_rank, num_classes=3).bool(),
                torch.finfo(group_logits.dtype).min,
            )
            target_gap = target_group_logit - other_group_logits.max(dim=1).values
            mass_threshold = float(loss_cfg.get("protected_tristate_roughness_context_mass_threshold", 0.34))
            mass_temperature = float(loss_cfg.get("protected_tristate_roughness_context_mass_temperature", 12.0))
            gap_threshold = float(loss_cfg.get("protected_tristate_roughness_context_gap_threshold", 0.55))
            gap_temperature = float(loss_cfg.get("protected_tristate_roughness_context_gap_temperature", 4.0))
            floor = float(loss_cfg.get("protected_tristate_roughness_context_floor", 0.22))
            physics_context = (0.55 * concrete_visibility + 0.45 * (1.0 - film_occlusion)).clamp(0.0, 1.0)
            mass_gate = torch.sigmoid((group_mass - mass_threshold) * mass_temperature)
            hard_gate = torch.sigmoid((gap_threshold - target_gap) * gap_temperature)
            context_gate = (floor + (1.0 - floor) * mass_gate * hard_gate * physics_context).clamp(floor, 1.0)
    else:
        group_mass = group_logits.new_ones((idx.numel(),))
        target_gap = group_logits.gather(1, target_rank.view(-1, 1)).squeeze(1).detach()
        context_gate = group_logits.new_ones((idx.numel(),))
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    smooth_count = int(target_rough.eq(1).sum().detach().cpu())
    slight_count = int(target_rough.eq(2).sum().detach().cpu())
    severe_count = int(target_rough.eq(3).sum().detach().cpu())

    mask = target_rough.eq(1)
    if bool(mask.any()):
        local_weight = (
            min_weight + (1.0 - min_weight) * reliability[mask] * smooth_score[mask] * context_gate[mask]
        ).clamp(min_weight, 1.0)
        local_margin = margin * (0.55 + 0.45 * smooth_score[mask])
        losses.append(
            0.5
            * (
                F.softplus(local_margin - (smooth_logit[mask] - slight_logit[mask]))
                + F.softplus(0.65 * local_margin - (smooth_logit[mask] - severe_logit[mask]))
            )
        )
        weights.append(local_weight)

    mask = target_rough.eq(2)
    if bool(mask.any()):
        local_weight = (
            min_weight + (1.0 - min_weight) * reliability[mask] * slight_score[mask] * context_gate[mask]
        ).clamp(min_weight, 1.0)
        margin_vs_smooth = margin * (0.40 + 0.60 * slight_score[mask]) * (1.0 - 0.35 * smooth_score[mask])
        margin_vs_severe = margin * (0.40 + 0.60 * slight_score[mask]) * (1.0 - slight_protect * severe_score[mask])
        margin_vs_smooth = margin_vs_smooth.clamp_min(0.12 * margin)
        margin_vs_severe = margin_vs_severe.clamp_min(0.12 * margin)
        losses.append(
            0.5
            * (
                F.softplus(margin_vs_smooth - (slight_logit[mask] - smooth_logit[mask]))
                + F.softplus(margin_vs_severe - (slight_logit[mask] - severe_logit[mask]))
            )
        )
        weights.append(local_weight)

    mask = target_rough.eq(3)
    if bool(mask.any()):
        local_weight = (
            min_weight + (1.0 - min_weight) * reliability[mask] * severe_score[mask] * context_gate[mask]
        ).clamp(min_weight, 1.0)
        local_margin = margin * (0.60 + severe_boost * severe_score[mask])
        losses.append(
            0.5
            * (
                F.softplus(local_margin - (severe_logit[mask] - slight_logit[mask]))
                + F.softplus(0.75 * local_margin - (severe_logit[mask] - smooth_logit[mask]))
            )
        )
        weights.append(local_weight)

    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_protected_tristate_roughness": 0.0,
            "protected_tristate_roughness_count": 0.0,
        }
    loss_values = torch.cat(losses)
    loss_weights = torch.cat(weights).to(device=loss_values.device, dtype=loss_values.dtype)
    loss = (loss_values * loss_weights).sum() / loss_weights.sum().clamp_min(1e-6)
    pred_rank = group_logits.argmax(dim=1)
    severe_margin = (severe_logit - slight_logit).detach()
    slight_vs_severe = (slight_logit - severe_logit).detach()
    logs = {
        "loss_protected_tristate_roughness": float(loss.detach().cpu()),
        "protected_tristate_roughness_count": float(idx.numel()),
        "protected_tristate_roughness_acc": float(pred_rank.eq(target_rank).float().mean().detach().cpu()),
        "protected_tristate_roughness_weight_mean": float(loss_weights.detach().mean().cpu()),
        "protected_tristate_roughness_visible_mean": float(visible_rough.detach().mean().cpu()),
        "protected_tristate_roughness_occlusion_mean": float(film_occlusion.detach().mean().cpu()),
        "protected_tristate_roughness_smooth_count": float(smooth_count),
        "protected_tristate_roughness_slight_count": float(slight_count),
        "protected_tristate_roughness_severe_count": float(severe_count),
        "protected_tristate_roughness_severe_margin_mean": float(severe_margin.mean().cpu()),
        "protected_tristate_roughness_slight_vs_severe_mean": float(slight_vs_severe.mean().cpu()),
        "protected_tristate_roughness_context_gate_mean": float(context_gate.detach().mean().cpu()),
        "protected_tristate_roughness_group_mass_mean": float(group_mass.detach().mean().cpu()),
        "protected_tristate_roughness_target_gap_mean": float(target_gap.detach().mean().cpu()),
    }
    return float(weight) * loss.to(dtype=model_out["logits"].dtype), logs


def value_guided_hardpair_margin_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train selected hard-pair margins with physics-weighted visibility.

    Unlike feature-value boundary correction, this loss does not add a new
    inference-time residual. It only tells the currently trainable middle
    mechanism to increase the true-vs-neighbor margin on selected RSCD hard
    pairs when physical roughness evidence is reliable enough.
    """

    weight = float(loss_cfg.get("value_guided_hardpair_margin_weight", 0.0))
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_hardpair_margin": 0.0,
            "value_guided_hardpair_margin_count": 0.0,
        }
    pair_specs = loss_cfg.get("value_guided_hardpair_margin_pairs", [])
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]
    requested_pairs: set[frozenset[str]] = set()
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
    if not requested_pairs:
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_hardpair_margin": 0.0,
            "value_guided_hardpair_margin_count": 0.0,
        }

    pair_weights_cfg = loss_cfg.get("value_guided_hardpair_margin_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(pair_weight)

    logits = model_out["logits"].float()
    idx_to_class = {idx: name for name, idx in spec.class_to_idx.items()}
    evidence = model_out.get("evidence_stats")
    rho = model_out.get("rho_roughness")
    margin = float(loss_cfg.get("value_guided_hardpair_margin_target", 0.48))
    low_margin = float(loss_cfg.get("value_guided_hardpair_margin_low_margin_threshold", 1.10))
    min_weight = float(loss_cfg.get("value_guided_hardpair_margin_min_weight", 0.10))
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    total = 0
    correct = 0
    selected_pairs = 0
    signed_margin_sum = 0.0
    physics_weight_sum = 0.0

    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        left_name = canonical_class_label(idx_to_class[left])
        right_name = canonical_class_label(idx_to_class[right])
        pair_names = frozenset((left_name, right_name))
        if pair_names not in requested_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        pair_mask = mask_left | mask_right
        if not bool(pair_mask.any()):
            continue
        sign = torch.where(mask_left, 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
        signed_margin = sign * (logits[:, left] - logits[:, right])
        focus_mask = pair_mask
        if low_margin >= 0.0:
            focus_mask = focus_mask & signed_margin.detach().le(low_margin)
        if not bool(focus_mask.any()):
            continue
        idx = focus_mask.nonzero(as_tuple=False).flatten()
        selected_margin = signed_margin.index_select(0, idx)
        phys_weight = selected_margin.new_ones(selected_margin.shape)
        if isinstance(evidence, torch.Tensor):
            ev = evidence.float().index_select(0, idx)
            rough = ev[:, 11].clamp(0.0, 1.0)
            wet = ev[:, 10].clamp(0.0, 1.0)
            dark_water = ev[:, 9].clamp(0.0, 1.0)
            specular = ev[:, 8].clamp(0.0, 1.0)
            erasure = ev[:, 12].clamp(0.0, 1.0)
            visible = torch.sigmoid((rough - 0.016) * 150.0)
            occlusion_guard = torch.sigmoid((0.60 - erasure - 0.30 * wet - 0.25 * dark_water - 0.18 * specular) * 7.5)
            phys_weight = (visible * occlusion_guard).to(device=selected_margin.device, dtype=selected_margin.dtype)
        if isinstance(rho, torch.Tensor):
            rho_weight = rho.float().view(-1).index_select(0, idx).to(dtype=selected_margin.dtype)
            phys_weight = torch.maximum(phys_weight, 0.30 * rho_weight.clamp(0.0, 1.0))
        phys_weight = (min_weight + (1.0 - min_weight) * phys_weight).clamp(min_weight, 1.0)
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        losses.append(F.softplus(float(margin) - selected_margin))
        weights.append(phys_weight * pair_weight)
        pred_left = (logits[:, left] - logits[:, right]).index_select(0, idx).ge(0.0)
        true_left = mask_left.index_select(0, idx)
        correct += int(pred_left.eq(true_left).sum().detach().cpu())
        total += int(idx.numel())
        selected_pairs += 1
        signed_margin_sum += float(selected_margin.detach().sum().cpu())
        physics_weight_sum += float(phys_weight.detach().sum().cpu())

    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_hardpair_margin": 0.0,
            "value_guided_hardpair_margin_count": 0.0,
        }
    loss_values = torch.cat(losses)
    loss_weights = torch.cat(weights).to(device=loss_values.device, dtype=loss_values.dtype)
    loss = (loss_values * loss_weights).sum() / loss_weights.sum().clamp_min(1e-6)
    logs = {
        "loss_value_guided_hardpair_margin": float(loss.detach().cpu()),
        "value_guided_hardpair_margin_count": float(total),
        "value_guided_hardpair_margin_pair_count": float(selected_pairs),
        "value_guided_hardpair_margin_acc": float(correct / max(total, 1)),
        "value_guided_hardpair_margin_signed_mean": float(signed_margin_sum / max(total, 1)),
        "value_guided_hardpair_margin_weight_mean": float(physics_weight_sum / max(total, 1)),
    }
    return float(weight) * loss.to(dtype=model_out["logits"].dtype), logs


def pair_value_selective_margin_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use diagnosed pair-value evidence only as a training-time margin gate.

    This loss is the conservative follow-up to the value-augmentation audit:
    the physical/color/texture values are too weak as a classifier and can hurt
    if injected as an always-on residual. Here they only decide which samples
    are reliable enough to emphasize for selected hard-pair margins. Inference
    logits are unchanged.
    """

    weight = float(loss_cfg.get("pair_value_selective_margin_weight", 0.0))
    logits = model_out["logits"].float()
    value_vector = model_out.get("hardpair_pair_value_evidence_vector")
    if weight <= 0.0 or not isinstance(value_vector, torch.Tensor):
        return model_out["logits"].new_zeros(()), {
            "loss_pair_value_selective_margin": 0.0,
            "pair_value_selective_margin_count": 0.0,
        }

    pair_specs = loss_cfg.get("pair_value_selective_margin_pairs", [])
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]
    requested_pairs: set[frozenset[str]] = set()
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
    if not requested_pairs:
        return model_out["logits"].new_zeros(()), {
            "loss_pair_value_selective_margin": 0.0,
            "pair_value_selective_margin_count": 0.0,
        }

    pair_weights_cfg = loss_cfg.get("pair_value_selective_margin_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(pair_weight)

    value_aug_std = max(float(loss_cfg.get("pair_value_selective_margin_value_aug_std", 0.0)), 0.0)
    values = value_vector.float().clamp(0.0, 1.0)
    if value_aug_std > 0.0:
        values = (values + torch.randn_like(values) * value_aug_std).clamp(0.0, 1.0)

    idx_to_class = {idx: name for name, idx in spec.class_to_idx.items()}
    margin = float(loss_cfg.get("pair_value_selective_margin_target", 0.48))
    low_margin = float(loss_cfg.get("pair_value_selective_margin_low_margin_threshold", 1.05))
    min_weight = float(loss_cfg.get("pair_value_selective_margin_min_weight", 0.08))
    threshold = float(loss_cfg.get("pair_value_selective_margin_gate_threshold", 0.38))
    temperature = float(loss_cfg.get("pair_value_selective_margin_gate_temperature", 8.0))
    uncertainty_temperature = float(loss_cfg.get("pair_value_selective_margin_uncertainty_temperature", 2.0))

    def gate_for(pair_names: frozenset[str], local_values: torch.Tensor) -> torch.Tensor:
        macro_rough = local_values[:, 0].clamp(0.0, 1.0)
        micro_rough = local_values[:, 1].clamp(0.0, 1.0)
        film = local_values[:, 2].clamp(0.0, 1.0)
        artifact = local_values[:, 3].clamp(0.0, 1.0)
        saturation = local_values[:, 4].clamp(0.0, 1.0)
        macro_mean = local_values[:, 5].clamp(0.0, 1.0)
        macro_std = local_values[:, 6].clamp(0.0, 1.0)
        meso_std = local_values[:, 7].clamp(0.0, 1.0)
        micro_std = local_values[:, 8].clamp(0.0, 1.0)
        lap_std = local_values[:, 9].clamp(0.0, 1.0)
        grad_std = local_values[:, 10].clamp(0.0, 1.0)
        anisotropy = local_values[:, 11].clamp(0.0, 1.0)
        dark_water = local_values[:, 12].clamp(0.0, 1.0)
        dark_water_top = local_values[:, 13].clamp(0.0, 1.0)
        specular = local_values[:, 14].clamp(0.0, 1.0)
        specular_top = local_values[:, 15].clamp(0.0, 1.0)
        texture_erasure = local_values[:, 16].clamp(0.0, 1.0)
        texture_erasure_top = local_values[:, 17].clamp(0.0, 1.0)
        value_mean = local_values[:, 18].clamp(0.0, 1.0)
        value_std = local_values[:, 19].clamp(0.0, 1.0)
        if pair_names == frozenset(("dry_asphalt_slight", "dry_asphalt_severe")):
            score = 0.34 * macro_std + 0.24 * macro_mean + 0.18 * macro_rough + 0.14 * anisotropy + 0.10 * value_std
        elif pair_names == frozenset(("wet_asphalt_slight", "wet_asphalt_severe")):
            score = 0.36 * macro_std + 0.25 * macro_mean + 0.17 * macro_rough + 0.12 * anisotropy + 0.10 * saturation
        elif pair_names == frozenset(("water_asphalt_slight", "water_asphalt_severe")):
            score = 0.28 * macro_std + 0.22 * macro_mean + 0.18 * dark_water + 0.16 * meso_std + 0.16 * texture_erasure_top
        elif pair_names == frozenset(("dry_concrete_smooth", "dry_concrete_slight")):
            score = 0.28 * texture_erasure_top + 0.25 * meso_std + 0.19 * macro_mean + 0.16 * macro_std + 0.12 * value_std
        elif pair_names == frozenset(("dry_concrete_slight", "dry_concrete_severe")):
            score = 0.32 * macro_rough + 0.24 * macro_std + 0.18 * meso_std + 0.16 * lap_std + 0.10 * grad_std
        elif pair_names == frozenset(("water_asphalt_smooth", "water_asphalt_slight")):
            score = 0.31 * film + 0.24 * texture_erasure_top + 0.19 * micro_std + 0.16 * lap_std + 0.10 * dark_water_top
        elif pair_names == frozenset(("water_concrete_smooth", "water_concrete_slight")):
            score = 0.34 * dark_water_top + 0.26 * dark_water + 0.18 * film + 0.14 * texture_erasure + 0.08 * (1.0 - saturation)
        elif pair_names == frozenset(("wet_concrete_slight", "wet_concrete_severe")):
            score = 0.30 * anisotropy + 0.24 * macro_rough + 0.20 * texture_erasure_top + 0.16 * meso_std + 0.10 * saturation
        elif pair_names == frozenset(("water_gravel", "water_mud")):
            score = 0.30 * micro_rough + 0.23 * micro_std + 0.21 * lap_std + 0.16 * grad_std + 0.10 * saturation
        elif pair_names in {
            frozenset(("water_asphalt_slight", "wet_asphalt_slight")),
            frozenset(("water_asphalt_severe", "wet_asphalt_severe")),
            frozenset(("water_concrete_slight", "wet_concrete_slight")),
        }:
            score = 0.33 * dark_water_top + 0.25 * dark_water + 0.20 * film + 0.12 * specular_top + 0.10 * specular
        else:
            score = 0.25 * macro_rough + 0.25 * micro_rough + 0.20 * film + 0.15 * texture_erasure_top + 0.15 * lap_std
        artifact_guard = (1.0 - 0.70 * artifact).clamp(0.12, 1.0)
        gate = torch.sigmoid((score.clamp(0.0, 1.0) - threshold) * temperature) * artifact_guard
        return (min_weight + (1.0 - min_weight) * gate).clamp(min_weight, 1.0)

    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    total = 0
    correct = 0
    active_pairs = 0
    selected_margin_sum = 0.0
    gate_sum = 0.0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        left_name = canonical_class_label(idx_to_class[left])
        right_name = canonical_class_label(idx_to_class[right])
        pair_names = frozenset((left_name, right_name))
        if pair_names not in requested_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        pair_mask = mask_left | mask_right
        if not bool(pair_mask.any()):
            continue
        sign = torch.where(mask_left, 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
        signed_margin = sign * (logits[:, left] - logits[:, right])
        focus_mask = pair_mask
        if low_margin >= 0.0:
            focus_mask = focus_mask & signed_margin.detach().le(low_margin)
        if not bool(focus_mask.any()):
            continue
        idx = focus_mask.nonzero(as_tuple=False).flatten()
        selected_margin = signed_margin.index_select(0, idx)
        local_values = values.index_select(0, idx).to(device=selected_margin.device, dtype=selected_margin.dtype)
        sample_gate = gate_for(pair_names, local_values)
        if uncertainty_temperature > 0.0 and low_margin >= 0.0:
            uncertainty = torch.sigmoid((low_margin - selected_margin.detach()) * uncertainty_temperature)
            sample_gate = sample_gate * (0.35 + 0.65 * uncertainty.to(dtype=sample_gate.dtype))
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        losses.append(F.softplus(float(margin) - selected_margin))
        weights.append(sample_gate * pair_weight)
        pred_left = (logits[:, left] - logits[:, right]).index_select(0, idx).ge(0.0)
        true_left = mask_left.index_select(0, idx)
        correct += int(pred_left.eq(true_left).sum().detach().cpu())
        total += int(idx.numel())
        active_pairs += 1
        selected_margin_sum += float(selected_margin.detach().sum().cpu())
        gate_sum += float(sample_gate.detach().sum().cpu())

    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_pair_value_selective_margin": 0.0,
            "pair_value_selective_margin_count": 0.0,
        }
    loss_values = torch.cat(losses)
    loss_weights = torch.cat(weights).to(device=loss_values.device, dtype=loss_values.dtype)
    loss = (loss_values * loss_weights).sum() / loss_weights.sum().clamp_min(1e-6)
    logs = {
        "loss_pair_value_selective_margin": float(loss.detach().cpu()),
        "pair_value_selective_margin_count": float(total),
        "pair_value_selective_margin_pair_count": float(active_pairs),
        "pair_value_selective_margin_acc": float(correct / max(total, 1)),
        "pair_value_selective_margin_signed_mean": float(selected_margin_sum / max(total, 1)),
        "pair_value_selective_margin_gate_mean": float(gate_sum / max(total, 1)),
    }
    return float(weight) * loss.to(dtype=model_out["logits"].dtype), logs


def factor_marginal_consistency_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align 27-class probabilities with explicit factor probabilities.

    RSCD labels are compositional: every class is a friction/material/roughness
    triple. If the class posterior assigns mass to water-concrete classes, its
    friction and material marginals should agree with the factor heads. This
    loss ties the flat class distribution and the factorized tensor head without
    adding another classifier or post-hoc residual.
    """

    weight = float(loss_cfg.get("factor_marginal_consistency_weight", 0.0))
    factor_logits = model_out.get("factor_logits", {})
    if weight <= 0.0 or not isinstance(factor_logits, dict):
        return model_out["logits"].new_zeros(()), {
            "loss_factor_marginal_consistency": 0.0,
            "factor_marginal_consistency_count": 0.0,
        }

    logits = model_out["logits"].float()
    temperature = max(float(loss_cfg.get("factor_marginal_consistency_temperature", 1.0)), 1.0e-3)
    class_probs = F.softmax(logits / temperature, dim=1)
    class_to_factor = spec.class_to_factor.to(device=logits.device)
    axis_weights_cfg = loss_cfg.get(
        "factor_marginal_consistency_axis_weights",
        {"friction": 1.0, "material": 1.0, "roughness": 1.0},
    )
    sample_weight = logits.new_ones((logits.shape[0],), dtype=torch.float32)
    downweight_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("factor_marginal_consistency_downweight_classes", [])
    }
    if downweight_classes:
        downweight_value = min(
            max(float(loss_cfg.get("factor_marginal_consistency_downweight_value", 0.25)), 0.0),
            1.0,
        )
        downweight_idx = {
            int(idx)
            for name, idx in spec.class_to_idx.items()
            if canonical_class_label(name) in downweight_classes
        }
        if downweight_idx:
            mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
            for idx in downweight_idx:
                mask |= labels.eq(int(idx))
            sample_weight = torch.where(
                mask.to(device=logits.device),
                sample_weight.new_full(sample_weight.shape, downweight_value),
                sample_weight,
            )
    roughness_axis_weight = sample_weight
    use_roughness_reliability_gate = bool(
        loss_cfg.get("factor_marginal_consistency_roughness_reliability_gate", False)
    )
    roughness_gate_logs: dict[str, float] = {}
    if use_roughness_reliability_gate:
        evidence = model_out.get("evidence_stats")
        gate_classes = {
            canonical_class_label(name)
            for name in loss_cfg.get("factor_marginal_consistency_roughness_gate_classes", [])
        }
        if isinstance(evidence, torch.Tensor) and gate_classes:
            gate_idx = {
                int(idx)
                for name, idx in spec.class_to_idx.items()
                if canonical_class_label(name) in gate_classes
            }
            gate_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
            for idx in gate_idx:
                gate_mask |= labels.eq(int(idx))
            stats = evidence.to(device=logits.device, dtype=torch.float32)
            wet = stats[:, 10].clamp(0.0, 1.0)
            dark_water = stats[:, 9].clamp(0.0, 1.0)
            specular = stats[:, 8].clamp(0.0, 1.0)
            erasure = stats[:, 12].clamp(0.0, 1.0)
            wet_film = torch.clamp(
                0.45 * wet + 0.25 * dark_water + 0.15 * specular + 0.15 * erasure,
                0.0,
                1.0,
            )
            rho_target = C3PhysicsEvidenceStats.roughness_reliability_target(stats).view(-1).clamp(0.0, 1.0)
            film_occlusion = (wet_film * (1.0 - rho_target)).clamp(0.0, 1.0)
            strength = min(
                max(float(loss_cfg.get("factor_marginal_consistency_roughness_gate_strength", 0.85)), 0.0),
                1.0,
            )
            floor = min(
                max(float(loss_cfg.get("factor_marginal_consistency_roughness_gate_floor", 0.30)), 0.0),
                1.0,
            )
            reliability_gate = torch.clamp(1.0 - strength * film_occlusion, min=floor, max=1.0)
            roughness_axis_weight = torch.where(
                gate_mask.to(device=logits.device),
                sample_weight * reliability_gate.to(device=logits.device, dtype=sample_weight.dtype),
                sample_weight,
            )
            active = gate_mask.to(device=logits.device)
            if bool(active.any()):
                roughness_gate_logs = {
                    "factor_marginal_consistency_roughness_gate_active_rate": float(
                        active.float().mean().detach().cpu()
                    ),
                    "factor_marginal_consistency_roughness_gate_active_mean": float(
                        reliability_gate[active].detach().mean().cpu()
                    ),
                    "factor_marginal_consistency_roughness_gate_film_occlusion_active_mean": float(
                        film_occlusion[active].detach().mean().cpu()
                    ),
                    "factor_marginal_consistency_roughness_gate_rho_active_mean": float(
                        rho_target[active].detach().mean().cpu()
                    ),
                }
            else:
                roughness_gate_logs = {
                    "factor_marginal_consistency_roughness_gate_active_rate": 0.0,
                    "factor_marginal_consistency_roughness_gate_active_mean": 1.0,
                    "factor_marginal_consistency_roughness_gate_film_occlusion_active_mean": 0.0,
                    "factor_marginal_consistency_roughness_gate_rho_active_mean": 0.0,
                }
        else:
            use_roughness_reliability_gate = False
    eps = 1.0e-6
    terms: list[torch.Tensor] = []
    logs: dict[str, float] = {}

    for axis_idx, axis in enumerate(FACTOR_AXES):
        axis_logits = factor_logits.get(axis)
        if not isinstance(axis_logits, torch.Tensor):
            continue
        axis_logits = axis_logits.float() / temperature
        num_factor_classes = int(axis_logits.shape[1])
        factor_index = class_to_factor[:, axis_idx]
        valid = (factor_index >= 0) & (factor_index < num_factor_classes)
        if not bool(valid.any()):
            continue
        valid_index = factor_index[valid].long()
        valid_class_probs = class_probs[:, valid]
        marginal = logits.new_zeros((logits.shape[0], num_factor_classes))
        marginal.scatter_add_(
            1,
            valid_index.unsqueeze(0).expand(logits.shape[0], -1),
            valid_class_probs,
        )
        marginal = marginal.clamp_min(eps)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(eps)
        factor_prob = F.softmax(axis_logits, dim=1).clamp_min(eps)
        factor_prob = factor_prob / factor_prob.sum(dim=1, keepdim=True).clamp_min(eps)

        # Two one-way KL terms update both sides while using a stable detached target.
        factor_to_class = F.kl_div(factor_prob.log(), marginal.detach(), reduction="none").sum(dim=1)
        class_to_factor_term = F.kl_div(marginal.log(), factor_prob.detach(), reduction="none").sum(dim=1)
        axis_loss_values = 0.5 * (factor_to_class + class_to_factor_term)
        axis_sample_weight = roughness_axis_weight if axis == "roughness" else sample_weight
        axis_loss = (axis_loss_values * axis_sample_weight).sum() / axis_sample_weight.sum().clamp_min(eps)
        axis_weight = float(axis_weights_cfg.get(axis, 1.0)) if isinstance(axis_weights_cfg, dict) else 1.0
        terms.append(float(axis_weight) * axis_loss)
        logs[f"factor_marginal_consistency_{axis}"] = float(axis_loss.detach().cpu())
        logs[f"factor_marginal_consistency_{axis}_sample_weight_mean"] = float(
            axis_sample_weight.detach().mean().cpu()
        )
        logs[f"factor_marginal_consistency_{axis}_l1"] = float(
            (marginal.detach() - factor_prob.detach()).abs().mean().cpu()
        )

    if not terms:
        return model_out["logits"].new_zeros(()), {
            "loss_factor_marginal_consistency": 0.0,
            "factor_marginal_consistency_count": 0.0,
        }
    loss = torch.stack(terms).mean().to(dtype=model_out["logits"].dtype)
    logs["loss_factor_marginal_consistency"] = float(loss.detach().cpu())
    logs["factor_marginal_consistency_count"] = float(len(terms))
    logs["factor_marginal_consistency_sample_weight_mean"] = float(sample_weight.detach().mean().cpu())
    logs["factor_marginal_consistency_roughness_reliability_gate"] = float(use_roughness_reliability_gate)
    if use_roughness_reliability_gate:
        logs["factor_marginal_consistency_roughness_gate_mean"] = float(
            roughness_axis_weight.detach().mean().cpu()
        )
        logs.update(roughness_gate_logs)
    return float(weight) * loss, logs


def prepare_pareto_selected_edge_rules(loss_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load split-validation accepted hard-edge rules for train-time imitation."""

    rules: list[dict[str, Any]] = []
    path_value = loss_cfg.get("pareto_selected_edge_rules_path")
    if path_value:
        rules.extend(load_pareto_safe_logit_patch_rules(Path(str(path_value))))
    for item in loss_cfg.get("pareto_selected_edge_rules", []) or []:
        if not isinstance(item, dict):
            continue
        rule = item.get("rule_raw", item)
        if isinstance(rule, dict) and {"source", "target", "topk", "margin", "delta"}.issubset(rule):
            rules.append(
                {
                    "source": str(rule["source"]),
                    "target": str(rule["target"]),
                    "topk": int(rule["topk"]),
                    "margin": float(rule["margin"]),
                    "delta": float(rule["delta"]),
                }
            )
    seen: set[tuple[str, str, int, float, float]] = set()
    unique: list[dict[str, Any]] = []
    for rule in rules:
        key = (
            canonical_class_label(str(rule["source"])),
            canonical_class_label(str(rule["target"])),
            int(rule["topk"]),
            round(float(rule["margin"]), 6),
            round(float(rule["delta"]), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(rule)
    return unique


def pareto_selected_edge_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    rules: list[dict[str, Any]],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train only validation-accepted RSCD hard edges with source protection.

    A selected rule means that split validation allowed a local source->target
    boundary correction without Top-1/Macro-F1/protected-class regression. During
    training, the target class is pulled back only when it is in a close
    source-vs-target ambiguity, while true source samples receive a symmetric
    protection margin. The gates are detached so this remains a local boundary
    objective rather than a new global classifier.
    """

    weight = float(loss_cfg.get("pareto_selected_edge_loss_weight", 0.0))
    if weight <= 0.0 or not rules:
        return logits.new_zeros(()), {
            "loss_pareto_selected_edge": 0.0,
            "pareto_selected_edge_target_count": 0.0,
            "pareto_selected_edge_source_protect_count": 0.0,
        }
    class_to_idx = {canonical_class_label(name): int(idx) for idx, name in idx_to_class.items()}
    if not class_to_idx:
        return logits.new_zeros(()), {
            "loss_pareto_selected_edge": 0.0,
            "pareto_selected_edge_target_count": 0.0,
            "pareto_selected_edge_source_protect_count": 0.0,
        }

    target_margin = float(loss_cfg.get("pareto_selected_edge_target_margin", 0.10))
    source_margin = float(loss_cfg.get("pareto_selected_edge_source_margin", target_margin))
    source_weight = float(loss_cfg.get("pareto_selected_edge_source_protect_weight", 1.0))
    gate_temperature = float(loss_cfg.get("pareto_selected_edge_gate_temperature", 8.0))
    min_gate = float(loss_cfg.get("pareto_selected_edge_min_gate", 0.0))
    eps = 1e-6

    logits_f = logits.float()
    with torch.no_grad():
        probs = F.softmax(logits_f, dim=1)
        order = torch.argsort(logits_f, dim=1, descending=True)

    terms: list[torch.Tensor] = []
    target_count = 0.0
    source_count = 0.0
    active_rule_count = 0
    gate_sum = 0.0
    pair_correct = 0
    pair_total = 0
    for rule in rules:
        source_name = canonical_class_label(str(rule["source"]))
        target_name = canonical_class_label(str(rule["target"]))
        if source_name not in class_to_idx or target_name not in class_to_idx:
            continue
        source = int(class_to_idx[source_name])
        target = int(class_to_idx[target_name])
        topk = max(1, min(int(rule.get("topk", 2)), int(logits_f.shape[1])))
        gate_margin = float(rule.get("margin", loss_cfg.get("pareto_selected_edge_gate_margin", 0.35)))
        source_logit = logits_f[:, source]
        target_logit = logits_f[:, target]
        with torch.no_grad():
            pair_mass = (probs[:, source] + probs[:, target]).clamp(0.0, 1.0)
            abs_gap = (source_logit - target_logit).abs()
            boundary_gate = torch.sigmoid((gate_margin - abs_gap) * gate_temperature) * pair_mass
            if min_gate > 0.0:
                boundary_gate = min_gate + (1.0 - min_gate) * boundary_gate
            source_in_topk = order[:, :topk].eq(source).any(dim=1)
            target_in_topk = order[:, :topk].eq(target).any(dim=1)

        target_mask = labels.eq(target) & source_in_topk
        if bool(target_mask.any()):
            gate = boundary_gate[target_mask]
            margin = (target_logit - source_logit)[target_mask]
            loss_values = F.relu(target_margin - margin).pow(2)
            terms.append((loss_values * gate).sum() / gate.sum().clamp_min(eps))
            target_count += float(target_mask.sum().detach().cpu())
            gate_sum += float(gate.detach().sum().cpu())
            active_rule_count += 1

        source_mask = labels.eq(source) & target_in_topk
        if source_weight > 0.0 and bool(source_mask.any()):
            gate = boundary_gate[source_mask]
            margin = (source_logit - target_logit)[source_mask]
            loss_values = F.relu(source_margin - margin).pow(2)
            terms.append(float(source_weight) * (loss_values * gate).sum() / gate.sum().clamp_min(eps))
            source_count += float(source_mask.sum().detach().cpu())
            gate_sum += float(gate.detach().sum().cpu())
            active_rule_count += 1

        pair_mask = labels.eq(source) | labels.eq(target)
        if bool(pair_mask.any()):
            pred_target = target_logit[pair_mask].ge(source_logit[pair_mask])
            true_target = labels[pair_mask].eq(target)
            pair_correct += int(pred_target.eq(true_target).sum().detach().cpu())
            pair_total += int(pair_mask.sum().detach().cpu())

    if not terms:
        return logits.new_zeros(()), {
            "loss_pareto_selected_edge": 0.0,
            "pareto_selected_edge_target_count": float(target_count),
            "pareto_selected_edge_source_protect_count": float(source_count),
            "pareto_selected_edge_pair_acc": float(pair_correct / max(pair_total, 1)),
        }
    loss = torch.stack(terms).mean().to(dtype=logits.dtype)
    logs = {
        "loss_pareto_selected_edge": float(loss.detach().cpu()),
        "pareto_selected_edge_target_count": float(target_count),
        "pareto_selected_edge_source_protect_count": float(source_count),
        "pareto_selected_edge_active_rule_terms": float(active_rule_count),
        "pareto_selected_edge_gate_mean": float(gate_sum / max(target_count + source_count, 1.0)),
        "pareto_selected_edge_pair_acc": float(pair_correct / max(pair_total, 1)),
    }
    return float(weight) * loss, logs


def teacher_feature_distillation_loss(
    student_out: dict[str, Any],
    teacher_out: dict[str, Any] | None,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align a custom RSCD backbone to the frozen teacher's fused representation."""

    logits = student_out["logits"]
    weight = float(loss_cfg.get("teacher_feature_distill_weight", 0.0))
    if weight <= 0.0 or teacher_out is None:
        return logits.new_zeros(()), {
            "loss_teacher_feature_distill": 0.0,
            "teacher_feature_distill_cosine": 0.0,
            "teacher_feature_distill_count": 0.0,
        }
    key = str(loss_cfg.get("teacher_feature_distill_key", "feature"))
    student_feature = student_out.get(key)
    teacher_feature = teacher_out.get(key)
    if not isinstance(student_feature, torch.Tensor) or not isinstance(teacher_feature, torch.Tensor):
        return logits.new_zeros(()), {
            "loss_teacher_feature_distill": 0.0,
            "teacher_feature_distill_cosine": 0.0,
            "teacher_feature_distill_count": 0.0,
        }
    student_feature = student_feature.float()
    teacher_feature = teacher_feature.detach().float().to(device=student_feature.device)
    if student_feature.ndim > 2:
        student_feature = student_feature.flatten(1)
    if teacher_feature.ndim > 2:
        teacher_feature = teacher_feature.flatten(1)
    count = min(int(student_feature.shape[0]), int(teacher_feature.shape[0]))
    dim = min(int(student_feature.shape[1]), int(teacher_feature.shape[1]))
    if count <= 0 or dim <= 0:
        return logits.new_zeros(()), {
            "loss_teacher_feature_distill": 0.0,
            "teacher_feature_distill_cosine": 0.0,
            "teacher_feature_distill_count": 0.0,
        }
    student_feature = student_feature[:count, :dim]
    teacher_feature = teacher_feature[:count, :dim]
    student_norm = F.normalize(student_feature, dim=1)
    teacher_norm = F.normalize(teacher_feature, dim=1)
    cosine = (student_norm * teacher_norm).sum(dim=1).clamp(-1.0, 1.0)
    mode = str(loss_cfg.get("teacher_feature_distill_mode", "cosine")).lower()
    if mode == "mse":
        raw_loss = F.mse_loss(student_norm, teacher_norm)
    elif mode == "cosine_mse":
        raw_loss = (1.0 - cosine).mean() + 0.25 * F.mse_loss(student_norm, teacher_norm)
    else:
        raw_loss = (1.0 - cosine).mean()
    loss = float(weight) * raw_loss.to(dtype=logits.dtype)
    return loss, {
        "loss_teacher_feature_distill": float(raw_loss.detach().cpu()),
        "teacher_feature_distill_cosine": float(cosine.detach().mean().cpu()),
        "teacher_feature_distill_count": float(count),
    }


def train_one_epoch(
    model: C3FaRNetSurfaceClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    scaler: torch.amp.GradScaler,
    *,
    teacher_model: C3FaRNetSurfaceClassifier | None = None,
    expert_teacher_model: C3FaRNetSurfaceClassifier | None = None,
    anchor_teacher_logit_cache: dict[str, torch.Tensor] | None = None,
    expert_teacher_logit_cache: dict[str, torch.Tensor] | None = None,
    teacher_cache_strict: bool = False,
    idx_to_class: dict[int, str] | None = None,
    out_dir: Path | None = None,
    epoch: int = 1,
    class_to_idx: dict[str, int] | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    pcgrad_enabled = bool(loss_cfg.get("rscd_pcgrad_enabled", train_cfg.get("rscd_pcgrad_enabled", False)))
    grouped_pcgrad_enabled = bool(loss_cfg.get("rscd_pcgrad_grouped_protect_enabled", False))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda" and not pcgrad_enabled
    accum = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    pareto_selected_edge_rules = prepare_pareto_selected_edge_rules(loss_cfg)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    log_every = int(train_cfg.get("log_every_steps", 80))
    aux_log_sum: dict[str, float] = {}
    aux_log_count = 0
    resume_start_step = int(train_cfg.get("_resume_start_step", train_cfg.get("resume_start_step", 0)) or 0)
    total_steps = resume_start_step + len(loader)
    step_checkpoint_every = int(train_cfg.get("save_step_checkpoint_every", 0) or 0)
    for local_step, batch in enumerate(tqdm(loader, desc="train", leave=False, ascii=True), 1):
        step = resume_start_step + local_step
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(image, return_aux=True)
            logits = out["logits"]
            ce_logs: dict[str, float] = {}
            if float(loss_cfg.get("feature_mechanism_ce_extra_weight", 0.0)) > 0.0:
                main_loss, ce_logs = mechanism_feature_weighted_cross_entropy(
                    logits,
                    label,
                    idx_to_class or {},
                    loss_cfg,
                    out,
                )
            else:
                main_loss = focus_weighted_cross_entropy(logits, label, idx_to_class or {}, loss_cfg)
            aux_loss, aux_logs = c3_total_aux_loss(
                out,
                label,
                model.spec,
                factor_weight=float(loss_cfg.get("factor_weight", 0.3)),
                factor_axis_weights=loss_cfg.get("factor_axis_weights", {"friction": 1.0, "material": 1.0, "roughness": 1.0}),
                tournament_weight=float(loss_cfg.get("tournament_weight", 0.1)),
                counterfactual_weight=float(loss_cfg.get("counterfactual_weight", 0.05)),
                reliability_weight=float(loss_cfg.get("reliability_weight", 0.05)),
                counterfactual_margin=float(loss_cfg.get("counterfactual_margin", 1.0)),
                supervise_none=bool(loss_cfg.get("supervise_none", False)),
            )
            binary_loss, binary_logs = hardpair_binary_tournament_loss(out, label, model.spec, loss_cfg)
            value_pair_loss, value_pair_logs = hardpair_value_adapter_pairwise_loss(out, label, model.spec, loss_cfg)
            feature_value_pair_loss, feature_value_pair_logs = feature_value_boundary_pairwise_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            opponent_feature_pair_loss, opponent_feature_pair_logs = water_concrete_opponent_feature_pairwise_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            rough_order_loss, rough_order_logs = value_guided_roughness_order_loss(
                out,
                label,
                model.spec,
                loss_cfg,
                idx_to_class or {},
            )
            protected_tristate_loss, protected_tristate_logs = protected_tristate_roughness_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            value_margin_loss, value_margin_logs = value_guided_hardpair_margin_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            pair_value_selective_loss, pair_value_selective_logs = pair_value_selective_margin_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            marginal_consistency_loss, marginal_consistency_logs = factor_marginal_consistency_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            dry_ordinal_loss, dry_ordinal_logs = dry_concrete_bidirectional_ordinal_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            family_router_loss, family_router_logs = family_mechanism_router_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            selected_edge_loss, selected_edge_logs = pareto_selected_edge_margin_loss(
                logits,
                label,
                idx_to_class or {},
                pareto_selected_edge_rules,
                loss_cfg,
            )
            graph_metric_loss, graph_metric_logs = factor_graph_metric_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            teacher_logits: torch.Tensor | None = None
            anchor_loss = logits.new_zeros(())
            nonregression_loss = logits.new_zeros(())
            pareto_safe_loss = logits.new_zeros(())
            classwise_pareto_loss = logits.new_zeros(())
            dual_teacher_loss = logits.new_zeros(())
            gate_loss = logits.new_zeros(())
            margin_loss = logits.new_zeros(())
            feature_distill_loss = logits.new_zeros(())
            anchor_logs: dict[str, float] = {}
            nonregression_logs: dict[str, float] = {}
            pareto_safe_logs: dict[str, float] = {}
            classwise_pareto_logs: dict[str, float] = {}
            dual_teacher_logs: dict[str, float] = {}
            gate_logs: dict[str, float] = {}
            margin_logs: dict[str, float] = {}
            feature_distill_logs: dict[str, float] = {}
            cache_logs: dict[str, float] = {}
            expert_teacher_logits: torch.Tensor | None = None
            teacher_aux_out: dict[str, Any] | None = None
            teacher_logits, anchor_cache_logs = cached_teacher_logits_for_batch(
                anchor_teacher_logit_cache,
                batch["image_path"],
                device=device,
                dtype=logits.dtype,
                strict=teacher_cache_strict,
                cache_name="anchor",
            )
            cache_logs.update(anchor_cache_logs)
            if teacher_logits is None and teacher_model is not None:
                with torch.no_grad():
                    if float(loss_cfg.get("teacher_feature_distill_weight", 0.0)) > 0.0:
                        teacher_raw = teacher_model(image, return_aux=True)
                        if isinstance(teacher_raw, dict):
                            teacher_aux_out = teacher_raw
                            teacher_logits = teacher_raw["logits"]
                        else:
                            teacher_logits = teacher_raw
                    else:
                        teacher_logits = teacher_model(image, return_aux=False)
            if teacher_logits is not None:
                anchor_loss, anchor_logs = anchor_consistency_loss(
                    logits,
                    teacher_logits,
                    label,
                    idx_to_class or {},
                    loss_cfg,
                )
                nonregression_loss, nonregression_logs = anchor_nonregression_barrier_loss(
                    logits,
                    teacher_logits,
                    label,
                    idx_to_class or {},
                    loss_cfg,
                )
                pareto_safe_loss, pareto_safe_logs = pareto_safe_distillation_loss(
                    logits,
                    teacher_logits,
                    label,
                    idx_to_class or {},
                    model.spec,
                    loss_cfg,
                )
                classwise_pareto_loss, classwise_pareto_logs = classwise_pareto_groupdro_loss(
                    out,
                    teacher_logits,
                    label,
                    idx_to_class or {},
                    model.spec,
                    loss_cfg,
                )
                gate_loss, gate_logs = anchor_error_gate_loss(out, teacher_logits, label, model.spec, loss_cfg)
                margin_loss, margin_logs = hardpair_margin_directed_loss(out, teacher_logits, label, model.spec, loss_cfg)
            feature_distill_loss, feature_distill_logs = teacher_feature_distillation_loss(
                out,
                teacher_aux_out,
                loss_cfg,
            )
            expert_teacher_logits, expert_cache_logs = cached_teacher_logits_for_batch(
                expert_teacher_logit_cache,
                batch["image_path"],
                device=device,
                dtype=logits.dtype,
                strict=teacher_cache_strict,
                cache_name="expert",
            )
            cache_logs.update(expert_cache_logs)
            if expert_teacher_logits is None and expert_teacher_model is not None:
                with torch.no_grad():
                    expert_teacher_logits = expert_teacher_model(image, return_aux=False)
            if teacher_logits is not None or expert_teacher_logits is not None:
                dual_teacher_loss, dual_teacher_logs = dual_teacher_noharm_loss(
                    logits,
                    teacher_logits,
                    expert_teacher_logits,
                    label,
                    idx_to_class or {},
                    loss_cfg,
                )
            pcgrad_focus_loss: torch.Tensor | None = None
            pcgrad_protect_loss: torch.Tensor | None = None
            pcgrad_protect_group_losses: list[tuple[str, torch.Tensor]] = []
            pcgrad_logs: dict[str, float] = {}
            pcgrad_focus_gradient_weight = float(loss_cfg.get("rscd_pcgrad_focus_weight", 0.0))
            if pcgrad_enabled:
                if grouped_pcgrad_enabled:
                    pcgrad_focus_loss, pcgrad_protect_group_losses, pcgrad_logs = rscd_focus_grouped_protect_objectives(
                        out,
                        teacher_logits,
                        label,
                        idx_to_class or {},
                        model.spec,
                        loss_cfg,
                    )
                    if pcgrad_protect_group_losses:
                        pcgrad_protect_loss = torch.stack([item[1] for item in pcgrad_protect_group_losses]).mean()
                else:
                    pcgrad_focus_loss, pcgrad_protect_loss, pcgrad_logs = rscd_focus_protect_objectives(
                        logits,
                        teacher_logits,
                        label,
                        idx_to_class or {},
                        loss_cfg,
                    )
            agem_memory_grads: list[torch.Tensor | None] | None = None
            if bool(loss_cfg.get("rscd_agem_total_projection_enabled", False)) and (
                pcgrad_protect_loss is not None or pcgrad_protect_group_losses
            ):
                agem_memory_grads = rscd_collect_protect_memory_gradient(
                    trainable_params,
                    pcgrad_protect_loss,
                    pcgrad_protect_group_losses,
                )
            main_loss_for_total = main_loss
            if bool(loss_cfg.get("rscd_pcgrad_decompose_main_ce", False)) and pcgrad_protect_loss is not None:
                protect_loss_weight = float(loss_cfg.get("rscd_pcgrad_protect_loss_weight", 1.0))
                if bool(loss_cfg.get("rscd_pcgrad_preserve_batch_ce_scale", True)):
                    protect_loss_weight *= float(pcgrad_logs.get("rscd_pcgrad_protect_count", 0.0)) / max(float(label.numel()), 1.0)
                    pcgrad_focus_gradient_weight *= float(pcgrad_logs.get("rscd_pcgrad_focus_count", 0.0)) / max(
                        float(label.numel()), 1.0
                    )
                main_loss_for_total = protect_loss_weight * pcgrad_protect_loss.to(dtype=logits.dtype)
                pcgrad_logs["loss_main_decomposed_protect_ce"] = float(main_loss_for_total.detach().cpu())
                pcgrad_logs["rscd_pcgrad_effective_focus_weight"] = float(pcgrad_focus_gradient_weight)
            loss = (
                main_loss_for_total
                + aux_loss
                + binary_loss
                + value_pair_loss
                + feature_value_pair_loss
                + opponent_feature_pair_loss
                + rough_order_loss
                + protected_tristate_loss
                + value_margin_loss
                + pair_value_selective_loss
                + marginal_consistency_loss
                + dry_ordinal_loss
                + family_router_loss
                + selected_edge_loss
                + graph_metric_loss
                + anchor_loss
                + nonregression_loss
                + pareto_safe_loss
                + classwise_pareto_loss
                + dual_teacher_loss
                + gate_loss
                + margin_loss
                + feature_distill_loss
            )
            backward = loss / float(accum)
        if not bool(torch.isfinite(loss.detach())):
            optimizer.zero_grad(set_to_none=True)
            continue
        if pcgrad_enabled:
            pcgrad_adjusted_grads: list[torch.Tensor | None] | None = None
            if pcgrad_focus_loss is not None and pcgrad_protect_loss is not None:
                if grouped_pcgrad_enabled and pcgrad_protect_group_losses:
                    pcgrad_adjusted_grads, pcgrad_surgery_logs = rscd_focus_grouped_protect_gradient_surgery(
                        trainable_params,
                        pcgrad_focus_loss,
                        pcgrad_protect_group_losses,
                        focus_weight=float(pcgrad_focus_gradient_weight),
                        accum=accum,
                    )
                else:
                    pcgrad_adjusted_grads, pcgrad_surgery_logs = rscd_focus_protect_gradient_surgery(
                        trainable_params,
                        pcgrad_focus_loss,
                        pcgrad_protect_loss,
                        focus_weight=float(pcgrad_focus_gradient_weight),
                        accum=accum,
                    )
                pcgrad_logs.update(pcgrad_surgery_logs)
            backward.backward()
            if pcgrad_adjusted_grads is not None:
                with torch.no_grad():
                    for param, grad in zip(trainable_params, pcgrad_adjusted_grads):
                        if grad is None:
                            continue
                        if param.grad is None:
                            param.grad = grad.detach().clone()
                        else:
                            param.grad.add_(grad.to(device=param.grad.device, dtype=param.grad.dtype))
            if agem_memory_grads is not None:
                pcgrad_logs.update(rscd_project_total_gradient_against_memory(trainable_params, agem_memory_grads))
            if step % accum == 0 or local_step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip_norm", 5.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            scaler.scale(backward).backward()
            if step % accum == 0 or local_step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip_norm", 5.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        batch_size = int(label.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == label).sum().detach().cpu())
        total_seen += batch_size
        aux_log_count += 1
        for key, value in aux_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in ce_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in binary_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in value_pair_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in feature_value_pair_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in opponent_feature_pair_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in rough_order_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in protected_tristate_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in value_margin_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in pair_value_selective_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in marginal_consistency_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in dry_ordinal_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in family_router_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in selected_edge_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in graph_metric_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in anchor_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in nonregression_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in pareto_safe_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in classwise_pareto_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in dual_teacher_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in cache_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in gate_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in margin_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in feature_distill_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in pcgrad_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        if step_checkpoint_every > 0 and out_dir is not None and (step % step_checkpoint_every == 0 or local_step == len(loader)):
            step_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": int(epoch),
                "step": int(step),
                "total_steps": int(total_steps),
                "class_to_idx": class_to_idx or {},
                "config": cfg,
                "train_partial": {
                    "loss": total_loss / max(total_seen, 1),
                    "top1": total_correct / max(total_seen, 1),
                    "seen": int(total_seen),
                },
            }
            _atomic_torch_save(step_state, out_dir / "last_step_checkpoint.pth")
            print(f"  saved step checkpoint: {out_dir / 'last_step_checkpoint.pth'} step={step}/{total_steps}")
        if log_every > 0 and (step % log_every == 0 or local_step == len(loader)):
            print(f"  train step {step}/{total_steps} loss={total_loss/max(total_seen,1):.4f} top1={total_correct/max(total_seen,1):.4f}")
    logs = {key: value / max(aux_log_count, 1) for key, value in aux_log_sum.items()}
    logs.update({"loss": total_loss / max(total_seen, 1), "top1": total_correct / max(total_seen, 1)})
    return logs


@torch.no_grad()
def evaluate(
    model: C3FaRNetSurfaceClassifier,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: dict[int, str],
    *,
    save_predictions_path: Path | None = None,
    logit_patch_rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses = []
    rows = []
    rho_values = []
    rho_label_indices: list[int] = []
    logit_patch_count = 0
    logit_patch_rule_hits: dict[str, int] = {}
    tournament_logs_sum: dict[str, float] = {}
    tournament_batches = 0
    for batch in tqdm(loader, desc="eval", leave=False, ascii=True):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        out = model(image, return_aux=True)
        logits = out["logits"]
        if logit_patch_rules:
            logits, patch_logs = apply_pareto_safe_logit_patch(
                logits,
                logit_patch_rules,
                idx_to_class,
            )
            logit_patch_count += int(patch_logs.get("count", 0))
            for key, value in patch_logs.get("rule_hits", {}).items():
                logit_patch_rule_hits[key] = logit_patch_rule_hits.get(key, 0) + int(value)
        loss = F.cross_entropy(logits, label)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        y_true.extend(label.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
        losses.append(float(loss.detach().cpu()) * int(label.numel()))
        if isinstance(out.get("rho_roughness"), torch.Tensor):
            rho_values.extend(out["rho_roughness"].detach().float().cpu().view(-1).numpy().tolist())
            rho_label_indices.extend(label.detach().cpu().numpy().astype(int).tolist())
        _, tour_logs = mechanism_routed_tournament_loss(logits, label, out.get("boundary_logits", {}), model.spec)
        tournament_batches += 1
        for key, value in tour_logs.items():
            tournament_logs_sum[key] = tournament_logs_sum.get(key, 0.0) + float(value)
        if save_predictions_path is not None:
            for path, true_idx, pred_idx, confidence in zip(batch["image_path"], y_true[-len(label):], y_pred[-len(label):], conf.detach().cpu().tolist(), strict=True):
                rows.append(
                    {
                        "image_path": str(path),
                        "true_label": idx_to_class[int(true_idx)],
                        "pred_label": idx_to_class[int(pred_idx)],
                        "confidence": float(confidence),
                    }
                )
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[i] for i in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    total_seen = len(y_true)
    factor_summary = factor_confusion_summary(y_true, y_pred, model.spec, idx_to_class)
    hard_class_names = [
        "water_concrete_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "wet_concrete_severe",
        "water_asphalt_slight",
        "dry_concrete_slight",
    ]
    hard_scores = [float(report[name]["f1-score"]) for name in hard_class_names if name in report]
    wcs_report = report.get("water_concrete_slight", {})
    rho_slice = rho_group_summary(rho_values, rho_label_indices, model.spec, idx_to_class)
    summary = {
        "loss": float(sum(losses) / max(total_seen, 1)),
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "num_samples": int(total_seen),
        "num_classes": int(len(labels)),
        "hard_class_mean_f1": float(np.mean(hard_scores)) if hard_scores else 0.0,
        "water_concrete_slight_precision": float(wcs_report.get("precision", 0.0)),
        "water_concrete_slight_recall": float(wcs_report.get("recall", 0.0)),
        "water_concrete_slight_f1": float(report.get("water_concrete_slight", {}).get("f1-score", 0.0)),
        "rho_R_mean": float(np.mean(rho_values)) if rho_values else 0.0,
        "head_type": str(getattr(model, "head_type", "")),
        "dryvor_enabled": bool(getattr(model, "dry_concrete_roughness_vor_residual", None) is not None),
        "logits_after_dryvor": bool(getattr(model, "dry_concrete_roughness_vor_residual", None) is not None),
        "boundary_use_physics_feature": bool(getattr(model, "boundary_use_physics_feature", False)),
        "pareto_safe_logit_patch_enabled": bool(logit_patch_rules),
        "pareto_safe_logit_patch_count": int(logit_patch_count),
    }
    for key, value in sorted(logit_patch_rule_hits.items()):
        summary[f"pareto_safe_logit_patch_hits/{key}"] = int(value)
    summary.update(rho_slice)
    summary.update({key: value / max(tournament_batches, 1) for key, value in tournament_logs_sum.items()})
    summary.update(factor_summary["summary"])
    if save_predictions_path is not None:
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(save_predictions_path, index=False, encoding="utf-8")
    return {
        "summary": summary,
        "classification_report": report,
        "factor_confusion_summary": factor_summary,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def load_pareto_safe_logit_patch_rules(path: Path | None) -> list[dict[str, Any]]:
    """Load validation-accepted RSCD hard-edge logit patch rules."""

    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"logit patch rules file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_rules = payload.get("accepted_rules", [])
    elif isinstance(payload, list):
        raw_rules = payload
    else:
        raise ValueError(f"unsupported logit patch rule payload in {path}")
    rules: list[dict[str, Any]] = []
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        rule = item.get("rule_raw", item)
        if isinstance(rule, dict) and {"source", "target", "topk", "margin", "delta"}.issubset(rule):
            rules.append(
                {
                    "source": str(rule["source"]),
                    "target": str(rule["target"]),
                    "topk": int(rule["topk"]),
                    "margin": float(rule["margin"]),
                    "delta": float(rule["delta"]),
                }
            )
    return rules


def apply_pareto_safe_logit_patch(
    logits: torch.Tensor,
    rules: list[dict[str, Any]],
    idx_to_class: dict[int, str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply split-validation-accepted RSCD hard-edge logit corrections."""

    if not rules:
        return logits, {"count": 0, "rule_hits": {}}
    class_to_idx = {canonical_class_label(name): int(idx) for idx, name in idx_to_class.items()}
    out = logits
    total_hits = 0
    rule_hits: dict[str, int] = {}
    for rule in rules:
        source_name = canonical_class_label(str(rule["source"]))
        target_name = canonical_class_label(str(rule["target"]))
        if source_name not in class_to_idx or target_name not in class_to_idx:
            continue
        source = int(class_to_idx[source_name])
        target = int(class_to_idx[target_name])
        topk = max(1, min(int(rule.get("topk", 2)), int(out.shape[1])))
        margin = float(rule.get("margin", 0.0))
        delta = float(rule.get("delta", 0.0))
        pred = out.argmax(dim=1)
        order = torch.argsort(out, dim=1, descending=True)
        in_topk = order[:, :topk].eq(int(target)).any(dim=1)
        close = (out[:, source] - out[:, target]) <= margin
        mask = pred.eq(int(source)) & in_topk & close
        hit_count = int(mask.detach().sum().cpu())
        if hit_count <= 0:
            continue
        if out is logits:
            out = logits.clone()
        out[mask, target] = out[mask, target] + delta
        out[mask, source] = out[mask, source] - delta * 0.25
        key = f"{source_name}->{target_name}"
        rule_hits[key] = rule_hits.get(key, 0) + hit_count
        total_hits += hit_count
    return out, {"count": int(total_hits), "rule_hits": rule_hits}


def rho_group_summary(
    rho_values: list[float],
    labels: list[int],
    spec: RSCDFactorSpec,
    idx_to_class: dict[int, str],
) -> dict[str, float]:
    """Summarize roughness visibility reliability by friction state and hard class."""

    if not rho_values or not labels:
        empty = {"rho_R_mean_water": 0.0, "rho_R_mean_wet": 0.0, "rho_R_mean_dry": 0.0}
        for name in ("water_concrete_slight", "water_concrete_severe", "wet_concrete_slight", "dry_concrete_slight"):
            empty[f"rho_R_mean_{name}"] = 0.0
            empty[f"rho_R_gap_dry_minus_{name}"] = 0.0
        return empty
    rho = np.asarray(rho_values, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    factors = spec.class_to_factor.numpy()
    friction = factors[label_arr, 0]
    friction_names = list(FACTOR_LABELS["friction"])

    def mean_for(name: str) -> float:
        if name not in friction_names:
            return 0.0
        idx = friction_names.index(name)
        mask = friction == idx
        return float(rho[mask].mean()) if bool(mask.any()) else 0.0

    return {
        "rho_R_mean_water": mean_for("water"),
        "rho_R_mean_wet": mean_for("wet"),
        "rho_R_mean_dry": mean_for("dry"),
        **rho_hard_class_summary(rho, label_arr, idx_to_class, dry_reference=mean_for("dry")),
    }


def rho_hard_class_summary(
    rho: np.ndarray,
    labels: np.ndarray,
    idx_to_class: dict[int, str],
    *,
    dry_reference: float,
) -> dict[str, float]:
    """Return rho_R means for the RSCD hard classes named in the goal."""

    class_to_idx = {str(name): int(idx) for idx, name in idx_to_class.items()}
    target_names = (
        "water_concrete_slight",
        "water_concrete_severe",
        "wet_concrete_slight",
        "dry_concrete_slight",
    )
    out: dict[str, float] = {}
    for name in target_names:
        idx = class_to_idx.get(name)
        if idx is None:
            mean_value = 0.0
        else:
            mask = labels == int(idx)
            mean_value = float(rho[mask].mean()) if bool(mask.any()) else 0.0
        out[f"rho_R_mean_{name}"] = mean_value
        out[f"rho_R_gap_dry_minus_{name}"] = float(dry_reference - mean_value)
    return out


def factor_confusion_summary(
    y_true: list[int],
    y_pred: list[int],
    spec: RSCDFactorSpec,
    idx_to_class: dict[int, str],
) -> dict[str, Any]:
    factors = spec.class_to_factor.numpy()
    errors = 0
    friction_error = 0
    material_error = 0
    roughness_error = 0
    axis_valid = {axis: 0 for axis in FACTOR_AXES}
    axis_correct = {axis: 0 for axis in FACTOR_AXES}
    rows = []
    for true_idx, pred_idx in zip(y_true, y_pred, strict=True):
        t = factors[int(true_idx)]
        p = factors[int(pred_idx)]
        axis_diff = {}
        for axis_i, axis in enumerate(FACTOR_AXES):
            valid = int(t[axis_i]) >= 0 and int(p[axis_i]) >= 0
            if valid:
                axis_valid[axis] += 1
                if int(t[axis_i]) == int(p[axis_i]):
                    axis_correct[axis] += 1
            axis_diff[axis] = bool(valid and int(t[axis_i]) != int(p[axis_i]))
        if int(true_idx) != int(pred_idx):
            errors += 1
            friction_error += int(axis_diff["friction"])
            material_error += int(axis_diff["material"])
            roughness_error += int(axis_diff["roughness"])
            rows.append(
                {
                    "true": idx_to_class[int(true_idx)],
                    "pred": idx_to_class[int(pred_idx)],
                    **{f"{axis}_error": axis_diff[axis] for axis in FACTOR_AXES},
                }
            )
    summary = {
        "friction_acc": axis_correct["friction"] / max(axis_valid["friction"], 1),
        "material_acc": axis_correct["material"] / max(axis_valid["material"], 1),
        "roughness_acc": axis_correct["roughness"] / max(axis_valid["roughness"], 1),
        "friction_error_share": friction_error / max(errors, 1),
        "material_error_share": material_error / max(errors, 1),
        "roughness_error_share": roughness_error / max(errors, 1),
        "num_errors": int(errors),
    }
    return {"summary": summary, "error_rows": rows}


def write_outputs(out_dir: Path, metrics: dict[str, Any], idx_to_class: dict[int, str], split: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k not in {"y_true", "y_pred"}}
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    report = metrics["classification_report"]
    with (out_dir / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for idx in range(len(idx_to_class)):
            name = idx_to_class[idx]
            item = report.get(name, {})
            w.writerow([name, item.get("precision", 0.0), item.get("recall", 0.0), item.get("f1-score", 0.0), item.get("support", 0)])
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(idx_to_class))))
    pd.DataFrame(cm, index=[idx_to_class[i] for i in range(len(idx_to_class))], columns=[idx_to_class[i] for i in range(len(idx_to_class))]).to_csv(
        out_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    (out_dir / "factor_confusion_summary.json").write_text(
        json.dumps(metrics["factor_confusion_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_hard_pair_metrics(out_dir / "hard_pair_metrics.csv", metrics, idx_to_class)
    write_wcs_diagnosis(out_dir / "water_concrete_slight_diagnosis.json", metrics, idx_to_class)


def materialize_eval_checkpoint_aliases(out_dir: Path, checkpoint: Path, *, split: str) -> None:
    """Expose the evaluated checkpoint under the standard train/eval artifact names.

    Training naturally writes best/last checkpoints. Evaluation consumes an
    existing checkpoint, so the honest equivalent is an alias to the source
    checkpoint plus a small provenance file. On the same volume this uses a
    hard link and costs no extra checkpoint storage.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(checkpoint).resolve()
    provenance = {
        "split": str(split),
        "source_checkpoint": str(source),
        "artifact_role": "evaluation alias of the checkpoint passed to validate.py/test.py",
    }
    (out_dir / "checkpoint_used.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ("best_checkpoint.pth", "last_checkpoint.pth"):
        target = out_dir / name
        if target.exists():
            continue
        try:
            os.link(source, target)
            continue
        except OSError as link_error:
            try:
                shutil.copy2(source, target)
                continue
            except OSError as copy_error:
                torch.save(
                    {
                        **provenance,
                        "warning": "source checkpoint could not be hard-linked or copied",
                        "link_error": repr(link_error),
                        "copy_error": repr(copy_error),
                    },
                    target,
                )


def write_hard_pair_metrics(path: Path, metrics: dict[str, Any], idx_to_class: dict[int, str]) -> None:
    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}
    spec = build_rscd_factor_spec(class_to_idx)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["left", "right", "axis", "boundary", "pair_samples", "pair_acc"])
        for pair in spec.hard_pairs:
            mask = (y_true == pair.left) | (y_true == pair.right)
            if not mask.any():
                continue
            ok = y_true[mask] == y_pred[mask]
            w.writerow([idx_to_class[pair.left], idx_to_class[pair.right], pair.axis, pair.boundary, int(mask.sum()), float(ok.mean())])


def write_wcs_diagnosis(path: Path, metrics: dict[str, Any], idx_to_class: dict[int, str]) -> None:
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}
    spec = build_rscd_factor_spec(class_to_idx)
    target = class_to_idx.get("water_concrete_slight")
    payload: dict[str, Any] = {"class": "water_concrete_slight", "present": target is not None}
    if target is not None:
        y_true = np.asarray(metrics["y_true"], dtype=int)
        y_pred = np.asarray(metrics["y_pred"], dtype=int)
        mask = y_true == int(target)
        report = metrics["classification_report"].get("water_concrete_slight", {})
        counts = pd.Series(y_pred[mask]).value_counts().head(8)
        factor_error_counts = _target_factor_error_counts(y_true, y_pred, int(target), spec)
        payload.update(
            {
                "precision": float(report.get("precision", 0.0)),
                "recall": float(report.get("recall", 0.0)),
                "f1": float(report.get("f1-score", 0.0)),
                "support": int(mask.sum()),
                "top_confused_classes": [
                    {"pred": idx_to_class[int(idx)], "count": int(count)}
                    for idx, count in counts.items()
                    if int(idx) != int(target)
                ],
                "roughness_error_count": int(factor_error_counts["roughness"]),
                "friction_error_count": int(factor_error_counts["friction"]),
                "material_error_count": int(factor_error_counts["material"]),
            }
        )
        factor_summary = metrics["factor_confusion_summary"]["summary"]
        payload.update(
            {
                "roughness_error_share": factor_summary.get("roughness_error_share", 0.0),
                "friction_error_share": factor_summary.get("friction_error_share", 0.0),
                "material_error_share": factor_summary.get("material_error_share", 0.0),
                "rho_R_mean": metrics["summary"].get("rho_R_mean", 0.0),
                "rho_R_mean_water": metrics["summary"].get("rho_R_mean_water", 0.0),
                "rho_R_mean_wet": metrics["summary"].get("rho_R_mean_wet", 0.0),
                "rho_R_mean_dry": metrics["summary"].get("rho_R_mean_dry", 0.0),
                "rho_R_mean_water_concrete_slight": metrics["summary"].get("rho_R_mean_water_concrete_slight", 0.0),
                "rho_R_mean_water_concrete_severe": metrics["summary"].get("rho_R_mean_water_concrete_severe", 0.0),
                "rho_R_mean_wet_concrete_slight": metrics["summary"].get("rho_R_mean_wet_concrete_slight", 0.0),
                "rho_R_mean_dry_concrete_slight": metrics["summary"].get("rho_R_mean_dry_concrete_slight", 0.0),
                "rho_R_gap_dry_minus_water_concrete_slight": metrics["summary"].get(
                    "rho_R_gap_dry_minus_water_concrete_slight",
                    0.0,
                ),
                "rho_R_gap_dry_minus_water_concrete_severe": metrics["summary"].get(
                    "rho_R_gap_dry_minus_water_concrete_severe",
                    0.0,
                ),
                "rho_R_gap_dry_minus_wet_concrete_slight": metrics["summary"].get(
                    "rho_R_gap_dry_minus_wet_concrete_slight",
                    0.0,
                ),
                "tournament_pair_accuracy_involving_water_concrete_slight": _wcs_pair_acc(metrics, class_to_idx, idx_to_class),
            }
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _target_factor_error_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_idx: int,
    spec: RSCDFactorSpec,
) -> dict[str, int]:
    factors = spec.class_to_factor.numpy()
    mask = (y_true == int(target_idx)) & (y_pred != int(target_idx))
    counts = {axis: 0 for axis in FACTOR_AXES}
    if not bool(mask.any()):
        return counts
    target_factors = factors[y_true[mask]]
    pred_factors = factors[y_pred[mask]]
    for axis_i, axis in enumerate(FACTOR_AXES):
        valid = (target_factors[:, axis_i] >= 0) & (pred_factors[:, axis_i] >= 0)
        counts[axis] = int(((target_factors[:, axis_i] != pred_factors[:, axis_i]) & valid).sum())
    return counts


def _wcs_pair_acc(metrics: dict[str, Any], class_to_idx: dict[str, int], idx_to_class: dict[int, str]) -> float:
    target = class_to_idx.get("water_concrete_slight")
    if target is None:
        return 0.0
    spec = build_rscd_factor_spec(class_to_idx)
    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    accs = []
    for pair in spec.hard_pairs:
        if pair.left != target and pair.right != target:
            continue
        mask = (y_true == pair.left) | (y_true == pair.right)
        if mask.any():
            accs.append(float((y_true[mask] == y_pred[mask]).mean()))
    return float(np.mean(accs)) if accs else 0.0


def run_train(config_path: Path) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 79)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg["train"]
    step_resume_state: dict[str, Any] | None = None
    step_resume_path = Path(str(train_cfg.get("resume_step_checkpoint_from") or (out_dir / "last_step_checkpoint.pth")))
    if bool(train_cfg.get("resume_step_checkpoint", False)) and step_resume_path.exists():
        step_resume_state = torch.load(step_resume_path, map_location="cpu", weights_only=False)
        resume_step = int(step_resume_state.get("step", 0) or 0)
        resume_total_steps = int(step_resume_state.get("total_steps", 0) or 0)
        cfg["train"]["_resume_start_step"] = resume_step
        cfg["train"]["_resume_epoch_training_complete"] = bool(resume_total_steps > 0 and resume_step >= resume_total_steps)
        cfg["train"]["_resume_step_checkpoint_path"] = str(step_resume_path)
        print(
            "Resuming in-epoch training checkpoint: "
            f"{step_resume_path} step={resume_step}/{resume_total_steps} "
            f"epoch_training_complete={cfg['train']['_resume_epoch_training_complete']}"
        )
    else:
        cfg["train"]["_resume_start_step"] = 0
        cfg["train"]["_resume_epoch_training_complete"] = False
    data = cfg["data"]
    manifests = [Path(data["train_manifest"]), Path(data["val_manifest"]), Path(data["test_manifest"])]
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    (out_dir / "label_factor_sanity.txt").write_text(sanity_summary(class_to_idx), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    train_loader, val_loader, test_loader = build_loaders(cfg, class_to_idx)
    device = resolve_device()
    model = build_model(cfg, class_to_idx).to(device)
    if step_resume_state is not None:
        model.load_state_dict(step_resume_state["model"], strict=True)
    else:
        flexible_load(model, cfg["train"].get("resume_from"), skip_prefixes=cfg["train"].get("resume_skip_prefixes"))
    apply_trainable_prefixes(model, cfg["train"].get("trainable_prefixes"))
    anchor_teacher_logit_cache = load_teacher_logit_cache(train_cfg.get("teacher_logits_cache"))
    expert_teacher_logit_cache = load_teacher_logit_cache(train_cfg.get("expert_teacher_logits_cache"))
    use_teacher_cache_fallback = bool(train_cfg.get("teacher_cache_online_fallback", False))
    teacher_model = None if anchor_teacher_logit_cache is not None and not use_teacher_cache_fallback else build_anchor_teacher(cfg, class_to_idx, device)
    expert_teacher_model = (
        None
        if expert_teacher_logit_cache is not None and not use_teacher_cache_fallback
        else build_specialist_teacher(cfg, class_to_idx, device)
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(train_cfg.get("lr", 3.5e-5)), weight_decay=float(train_cfg.get("weight_decay", 0.003)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(train_cfg.get("amp", True)))
    if step_resume_state is not None:
        if "optimizer" in step_resume_state:
            optimizer.load_state_dict(step_resume_state["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
        if "scaler" in step_resume_state:
            scaler.load_state_dict(step_resume_state["scaler"])
    best_key = (-1.0, -1.0)
    history = []
    if bool(train_cfg.get("evaluate_initial", False)) and step_resume_state is None:
        print("Evaluating initial checkpoint before fine-tuning")
        val_metrics = evaluate(model, val_loader, device, idx_to_class)
        val_summary = val_metrics["summary"]
        print(f"  initial val top1={val_summary['top1']:.4f} macro_f1={val_summary['macro_f1']:.4f} wcs={val_summary['water_concrete_slight_f1']:.4f}")
        history.append({"epoch": 0, "train": {}, "val": val_summary})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        state = {"model": model.state_dict(), "epoch": 0, "class_to_idx": class_to_idx, "config": cfg, "val_summary": val_summary}
        torch.save(state, out_dir / "best_checkpoint.pth")
        torch.save(state, out_dir / "best.pt")
        best_key = (float(val_summary["macro_f1"]), float(val_summary["top1"]))
        print(f"  saved initial best: {out_dir / 'best_checkpoint.pth'}")
    start_epoch = int(step_resume_state.get("epoch", 1) if step_resume_state is not None else 1)
    for epoch in range(start_epoch, int(cfg["train"].get("epochs", 1)) + 1):
        if epoch > start_epoch and int(cfg["train"].get("_resume_start_step", 0) or 0) != 0:
            cfg["train"]["_resume_start_step"] = 0
            cfg["train"]["_resume_epoch_training_complete"] = False
            train_loader, _, _ = build_loaders(cfg, class_to_idx)
        print(f"Epoch {epoch}/{cfg['train'].get('epochs', 1)}")
        if bool(cfg["train"].get("_resume_epoch_training_complete", False)) and epoch == start_epoch and step_resume_state is not None:
            train_metrics = dict(step_resume_state.get("train_partial", {}) or {})
            train_metrics.setdefault("loss", 0.0)
            train_metrics.setdefault("top1", 0.0)
            print(
                "  skipped training epoch from completed step checkpoint "
                f"step={int(step_resume_state.get('step', 0) or 0)}/"
                f"{int(step_resume_state.get('total_steps', 0) or 0)}"
            )
        else:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                cfg,
                scaler,
                teacher_model=teacher_model,
                expert_teacher_model=expert_teacher_model,
                anchor_teacher_logit_cache=anchor_teacher_logit_cache,
                expert_teacher_logit_cache=expert_teacher_logit_cache,
                teacher_cache_strict=bool(train_cfg.get("teacher_logits_cache_strict", False)),
                idx_to_class=idx_to_class,
                out_dir=out_dir,
                epoch=epoch,
                class_to_idx=class_to_idx,
            )
        val_metrics = evaluate(model, val_loader, device, idx_to_class)
        val_summary = val_metrics["summary"]
        print(f"  train loss={train_metrics['loss']:.4f} top1={train_metrics['top1']:.4f}")
        print(f"  val top1={val_summary['top1']:.4f} macro_f1={val_summary['macro_f1']:.4f} wcs={val_summary['water_concrete_slight_f1']:.4f}")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_summary})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        state = {"model": model.state_dict(), "epoch": epoch, "class_to_idx": class_to_idx, "config": cfg, "val_summary": val_summary}
        torch.save(state, out_dir / "last_checkpoint.pth")
        torch.save(state, out_dir / "last.pt")
        key = (float(val_summary["macro_f1"]), float(val_summary["top1"]))
        if key > best_key:
            best_key = key
            torch.save(state, out_dir / "best_checkpoint.pth")
            torch.save(state, out_dir / "best.pt")
            print(f"  saved best: {out_dir / 'best_checkpoint.pth'}")
    state = torch.load(out_dir / "best_checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    test_metrics = evaluate(model, test_loader, device, idx_to_class, save_predictions_path=out_dir / "predictions_test.csv")
    write_outputs(out_dir, test_metrics, idx_to_class, split="test")
    print(json.dumps(test_metrics["summary"], indent=2, ensure_ascii=False))


def run_eval(
    config_path: Path,
    checkpoint: Path,
    split: str = "test",
    *,
    seed_override: int | None = None,
    output_dir_override: Path | None = None,
    logit_patch_rules_path: Path | None = None,
) -> None:
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    if output_dir_override is not None:
        cfg["output_dir"] = str(output_dir_override)
    eval_cfg = cfg.get("eval", {})
    if logit_patch_rules_path is None and eval_cfg.get("logit_patch_rules_path"):
        logit_patch_rules_path = Path(str(eval_cfg["logit_patch_rules_path"]))
    logit_patch_rules = load_pareto_safe_logit_patch_rules(logit_patch_rules_path)
    out_dir = Path(cfg["output_dir"]) / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = cfg["data"]
    manifests = [Path(data["train_manifest"]), Path(data["val_manifest"]), Path(data["test_manifest"])]
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    (out_dir / "label_factor_sanity.txt").write_text(sanity_summary(class_to_idx), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _, val_loader, test_loader = build_loaders(cfg, class_to_idx)
    loader = val_loader if split == "val" else test_loader
    device = resolve_device()
    model = build_model(cfg, class_to_idx).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    metrics = evaluate(
        model,
        loader,
        device,
        idx_to_class,
        save_predictions_path=out_dir / f"predictions_{split}.csv",
        logit_patch_rules=logit_patch_rules,
    )
    write_outputs(out_dir, metrics, idx_to_class, split=split)
    materialize_eval_checkpoint_aliases(out_dir, checkpoint, split=split)
    print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))

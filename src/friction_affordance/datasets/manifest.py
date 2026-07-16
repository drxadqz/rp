from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

from friction_affordance.ontology import IGNORE_INDEX, TASKS, label_to_index


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifests: list[str | Path],
        transform=None,
        max_samples: int | None = None,
        max_samples_per_dataset: int | None = None,
        max_samples_per_class: int | None = None,
        sample_seed: int = 17,
        mask_transform=None,
        load_road_masks: bool = False,
    ) -> None:
        frames = []
        for path in manifests:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Manifest not found: {p}")
            frames.append(pd.read_csv(p, dtype=str, low_memory=False))
        self.df = pd.concat(frames, ignore_index=True)
        if max_samples_per_class:
            self.df = _sample_grouped(
                self.df,
                ["dataset", "class_label"],
                int(max_samples_per_class),
                sample_seed,
            )
        if max_samples_per_dataset:
            self.df = _sample_grouped(
                self.df,
                ["dataset"],
                int(max_samples_per_dataset),
                sample_seed,
            )
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=sample_seed).reset_index(drop=True)
        domain_values = self.df.apply(_domain_key, axis=1)
        self.domain_to_idx = {name: idx for idx, name in enumerate(sorted(domain_values.unique().tolist()))}
        self.num_domains = len(self.domain_to_idx)
        group_values = self.df.apply(_group_key, axis=1)
        self.group_to_idx = {name: idx for idx, name in enumerate(sorted(group_values.unique().tolist()))}
        self.num_groups = len(self.group_to_idx)
        self.transform = transform
        self.mask_transform = mask_transform
        self.load_road_masks = bool(load_road_masks)
        self._warned_bad_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start_idx = int(idx)
        last_error: Exception | None = None
        for offset in range(min(50, len(self.df))):
            row_idx = (start_idx + offset) % len(self.df)
            row = self.df.iloc[row_idx].to_dict()
            try:
                return self._load_item(row)
            except (OSError, SyntaxError, ValueError) as exc:
                last_error = exc
                self._warn_bad_image(str(row.get("image_path", "")), exc)
                continue
        raise RuntimeError(f"Could not load a valid image after retries near index {start_idx}: {last_error}")

    def _load_item(self, row: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(row["image_path"]))
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.load()
            if self.transform is not None:
                img = self.transform(img)

        labels = {}
        masks = {}
        column_map = {
            "friction": "friction_label",
            "material": "material_label",
            "unevenness": "unevenness_label",
            "wetness": "wetness_label",
            "snow": "snow_label",
            "risk": "risk_label",
        }
        for task in TASKS:
            value = row.get(column_map[task], None)
            label_idx = label_to_index(task, value)
            labels[task] = label_idx
            masks[task] = label_idx != IGNORE_INDEX

        mu_low = _maybe_float(row.get("mu_low"))
        mu_high = _maybe_float(row.get("mu_high"))
        mu_mask = mu_low is not None and mu_high is not None
        if not mu_mask:
            mu_low, mu_high = 0.0, 1.0

        item = {
            "image": img,
            "labels": labels,
            "masks": masks,
            "mu_interval": torch.tensor([float(mu_low), float(mu_high)], dtype=torch.float32),
            "mu_mask": bool(mu_mask),
            "dataset": str(row.get("dataset", "")),
            "domain_id": str(row.get("domain_id", "")),
            "domain_idx": int(self.domain_to_idx.get(_domain_key(row), 0)),
            "group_key": _group_key(row),
            "group_idx": int(self.group_to_idx.get(_group_key(row), 0)),
            "image_path": str(path),
        }
        road_mask = self._load_road_mask(row)
        if road_mask is not None:
            item["road_mask"] = road_mask
        return item

    def _load_road_mask(self, row: dict[str, Any]) -> torch.Tensor | None:
        if not self.load_road_masks and "road_mask_path" not in row:
            return None
        raw_path = row.get("road_mask_path", None)
        if raw_path is None or pd.isna(raw_path):
            return None
        text = str(raw_path).strip()
        if not text or text.lower() in {"nan", "none", "null", "-1"}:
            return None
        path = Path(text)
        if not path.exists():
            return None
        with Image.open(path) as mask:
            mask = mask.convert("L")
            if self.mask_transform is not None:
                tensor = self.mask_transform(mask)
            else:
                array = np.asarray(mask, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor.to(dtype=torch.float32).clamp(0.0, 1.0)

    def _warn_bad_image(self, path: str, exc: Exception) -> None:
        if path in self._warned_bad_paths:
            return
        self._warned_bad_paths.add(path)
        print(f"WARNING: skipped unreadable image: {path} ({type(exc).__name__}: {exc})", flush=True)


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_grouped(df: pd.DataFrame, columns: list[str], n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or not all(col in df.columns for col in columns):
        return df
    parts = []
    for _, group in df.groupby(columns, dropna=False, sort=False):
        if len(group) > n:
            group = group.sample(n=n, random_state=seed)
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _domain_key(row) -> str:
    domain = row.get("domain_id", None)
    if domain is not None and not pd.isna(domain) and str(domain).strip():
        return str(domain)
    dataset = row.get("dataset", None)
    if dataset is not None and not pd.isna(dataset) and str(dataset).strip():
        return str(dataset)
    return "unknown"


def _label_text(value) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown", "-1"}:
        return None
    return text


def _group_key(row) -> str:
    dataset = _label_text(row.get("dataset", None)) or "unknown"
    snow = _label_text(row.get("snow_label", None))
    wetness = _label_text(row.get("wetness_label", None))
    friction = _label_text(row.get("friction_label", None))
    class_label = _label_text(row.get("class_label", None))
    if snow and snow != "none":
        core = snow
    elif wetness:
        core = wetness
    elif friction:
        core = friction
    else:
        core = class_label or "unknown"
    return f"{dataset}::{core}"


def collate_manifest_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = {
        task: torch.tensor([b["labels"][task] for b in batch], dtype=torch.long)
        for task in TASKS
    }
    masks = {
        task: torch.tensor([b["masks"][task] for b in batch], dtype=torch.bool)
        for task in TASKS
    }
    out = {
        "image": images,
        "labels": labels,
        "masks": masks,
        "mu_interval": torch.stack([b["mu_interval"] for b in batch], dim=0),
        "mu_mask": torch.tensor([b["mu_mask"] for b in batch], dtype=torch.bool),
        "dataset": [b["dataset"] for b in batch],
        "domain_id": [b["domain_id"] for b in batch],
        "domain_idx": torch.tensor([b["domain_idx"] for b in batch], dtype=torch.long),
        "group_key": [b["group_key"] for b in batch],
        "group_idx": torch.tensor([b["group_idx"] for b in batch], dtype=torch.long),
        "image_path": [b["image_path"] for b in batch],
    }
    road_masks = [b.get("road_mask") for b in batch]
    if road_masks and all(mask is not None for mask in road_masks):
        out["road_mask"] = torch.stack([mask for mask in road_masks if mask is not None], dim=0)
    return out

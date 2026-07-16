from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from friction_affordance.ontology import infer_record, record_to_manifest_fields

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_rscd_prepared(labels_csv: str | Path, max_per_class: int | None = None) -> pd.DataFrame:
    labels_csv = Path(labels_csv)
    df = pd.read_csv(labels_csv, low_memory=False)
    if "combined_label" not in df.columns:
        raise ValueError(f"Expected combined_label column in {labels_csv}")
    df["split"] = df["split"].map(_normalize_split)
    df = _limit_per_class(df, "combined_label", max_per_class, split_col="split").reset_index(drop=True)

    label_fields = {
        label: record_to_manifest_fields(infer_record("rscd", label))
        for label in df["combined_label"].dropna().unique().tolist()
    }
    fields = pd.DataFrame(df["combined_label"].map(label_fields).tolist())
    out = pd.DataFrame(
        {
            "image_path": _remap_rscd_image_paths(df["image_path"], labels_csv),
            "split": df["split"],
            "dataset": "rscd",
            "class_label": df["combined_label"],
            "domain_id": "rscd_official",
        }
    )
    out = pd.concat([out, fields], axis=1)
    if "friction_mean" in df.columns:
        mean = pd.to_numeric(df["friction_mean"], errors="coerce")
        if "friction_std" in df.columns:
            std = pd.to_numeric(df["friction_std"], errors="coerce").fillna(0.08)
        else:
            std = pd.Series(0.08, index=df.index)
        valid = mean.notna()
        out.loc[valid, "mu_low"] = (mean[valid] - 2.0 * std[valid]).clip(lower=0.0)
        out.loc[valid, "mu_high"] = (mean[valid] + 2.0 * std[valid]).clip(upper=1.2)
    return out


def scan_imagefolder(
    root: str | Path,
    dataset_name: str,
    max_per_class: int | None = None,
) -> pd.DataFrame:
    root = Path(root)
    rows = []
    split_dirs = _find_split_dirs(root)
    for split, split_path in split_dirs:
        direct_files = [
            p for p in split_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        for path in sorted(direct_files):
            class_label = _label_from_flat_filename(path)
            rec = infer_record(dataset_name, class_label)
            item = {
                "image_path": str(path),
                "split": split,
                "dataset": dataset_name,
                "class_label": class_label,
                "domain_id": dataset_name,
            }
            item.update(record_to_manifest_fields(rec))
            rows.append(item)
        for class_dir in sorted([p for p in split_path.iterdir() if p.is_dir()]):
            files = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            files = sorted(files)
            if max_per_class:
                files = files[:max_per_class]
            for path in files:
                rec = infer_record(dataset_name, class_dir.name)
                item = {
                    "image_path": str(path),
                    "split": split,
                    "dataset": dataset_name,
                    "class_label": class_dir.name,
                    "domain_id": dataset_name,
                }
                item.update(record_to_manifest_fields(rec))
                rows.append(item)
    df = pd.DataFrame(rows)
    df = _limit_per_class(df, "class_label", max_per_class, split_col="split").reset_index(drop=True)
    return _ensure_validation_split(df)


def _label_from_flat_filename(path: Path) -> str:
    stem = path.stem
    if "-" not in stem:
        return stem
    return stem.split("-", 1)[1]


def split_and_write_manifests(df: pd.DataFrame, out_prefix: str | Path) -> list[Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].copy()
        if part.empty:
            continue
        out = out_prefix.parent / f"{out_prefix.name}_{split}.csv"
        part.to_csv(out, index=False, encoding="utf-8")
        written.append(out)
    return written


def _limit_per_class(
    df: pd.DataFrame,
    class_col: str,
    max_per_class: int | None,
    split_col: str | None = None,
) -> pd.DataFrame:
    if not max_per_class:
        return df
    if split_col and split_col in df.columns:
        return (
            df.groupby([split_col, class_col], group_keys=False)
            .head(max_per_class)
            .reset_index(drop=True)
        )
    return (
        df.groupby(class_col, group_keys=False)
        .head(max_per_class)
        .reset_index(drop=True)
    )


def _ensure_validation_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "val" in set(df["split"]):
        return df
    if "train" not in set(df["split"]):
        return df
    train_idx = []
    for _, group in df[df["split"] == "train"].groupby("class_label"):
        n_val = max(1, int(round(len(group) * 0.1)))
        train_idx.extend(group.sort_values("image_path").head(n_val).index.tolist())
    df = df.copy()
    df.loc[train_idx, "split"] = "val"
    return df


def _find_split_dirs(root: Path) -> Iterable[tuple[str, Path]]:
    split_aliases = {
        "train": "train",
        "validation": "val",
        "vali": "val",
        "val": "val",
        "test": "test",
        "test_50k": "test",
        "vali_20k": "val",
    }
    found = []
    for child in root.iterdir():
        if child.is_dir():
            key = child.name.lower()
            if key in split_aliases:
                found.append((split_aliases[key], child))
    if found:
        return found
    return [("train", root)]


def _normalize_split(value) -> str:
    text = str(value).lower()
    if text in {"validation", "vali", "val"}:
        return "val"
    if text.startswith("test"):
        return "test"
    return "train"


def _remap_rscd_image_path(image_path: str | Path, labels_csv: Path) -> str:
    raw_root = _sibling_dataset_root(labels_csv, "RSCD_prepared", "RSCD_raw")
    text = str(image_path)
    if raw_root is None:
        return text
    normalized = text.replace("/", "\\")
    marker = "\\RSCD_raw\\"
    idx = normalized.lower().find(marker.lower())
    if idx < 0:
        return text
    rel = normalized[idx + len(marker) :]
    return str(raw_root / rel)


def _remap_rscd_image_paths(paths: pd.Series, labels_csv: Path) -> pd.Series:
    raw_root = _sibling_dataset_root(labels_csv, "RSCD_prepared", "RSCD_raw")
    if raw_root is None:
        return paths.astype(str)
    raw_root_text = str(raw_root)
    marker = "\\RSCD_raw\\"

    def remap(value: str | Path) -> str:
        normalized = str(value).replace("/", "\\")
        idx = normalized.lower().find(marker.lower())
        if idx < 0:
            return str(value)
        return raw_root_text + "\\" + normalized[idx + len(marker) :]

    return paths.map(remap)


def _sibling_dataset_root(path: Path, anchor_name: str, sibling_name: str) -> Path | None:
    for parent in path.parents:
        if parent.name.lower() == anchor_name.lower():
            sibling = parent.parent / sibling_name
            if sibling.exists():
                return sibling
    return None

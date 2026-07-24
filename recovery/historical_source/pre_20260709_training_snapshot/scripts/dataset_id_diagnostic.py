from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.engine import build_model, move_batch
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml, resolve_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    manifests = list(cfg["data"].get("train_manifests", [])) + list(cfg["data"].get("val_manifests", []))
    ds = ManifestDataset(
        manifests,
        transform=build_transforms(
            int(cfg["data"].get("image_size", 224)),
            train=False,
            aug_cfg=cfg["data"].get("augmentation"),
        ),
        max_samples=args.max_samples,
    )
    loader = DataLoader(ds, batch_size=int(cfg["data"].get("batch_size", 32)), shuffle=False, collate_fn=collate_manifest_batch)

    feats = []
    labels = []
    risk_conditions = []
    core_conditions = []
    dataset_to_idx = {}
    for batch in loader:
        moved = move_batch(batch, device)
        out = model(moved["image"], domain_idx=moved.get("domain_idx"))
        feats.append(out.get("shared_features", out["features"]).detach().cpu().numpy())
        for i, name in enumerate(batch["dataset"]):
            if name not in dataset_to_idx:
                dataset_to_idx[name] = len(dataset_to_idx)
            labels.append(dataset_to_idx[name])
            risk_conditions.append(str(int(batch["labels"]["risk"][i])))
            core_conditions.append(
                "|".join(
                    str(int(batch["labels"][task][i]))
                    for task in ["friction", "material", "wetness", "snow", "risk"]
                )
            )
    x = np.concatenate(feats, axis=0)
    y = np.asarray(labels)
    if len(set(y.tolist())) < 2:
        raise SystemExit("Need at least two datasets for dataset-ID diagnostic.")
    metrics = {
        "dataset_to_idx": dataset_to_idx,
        "num_samples": int(len(y)),
    }
    metrics.update(_probe(x, y, "overall"))
    metrics.update(
        _probe_common_condition(
            x,
            y,
            np.asarray(risk_conditions),
            "risk_conditioned_common",
            min_datasets=2,
            min_samples=50,
        )
    )
    metrics.update(
        _probe_common_condition(
            x,
            y,
            np.asarray(core_conditions),
            "core_state_conditioned_common",
            min_datasets=2,
            min_samples=20,
        )
    )
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    out_path = args.out
    if out_path is None:
        out_dir = Path(cfg.get("output_dir", args.checkpoint.parent))
        out_path = out_dir / "dataset_id_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote: {out_path}")


def _probe(x: np.ndarray, y: np.ndarray, prefix: str) -> dict[str, float | int]:
    if len(set(y.tolist())) < 2 or len(y) < 10:
        return {f"{prefix}_num_samples": int(len(y))}
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=17, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        f"{prefix}_num_samples": int(len(y)),
        f"{prefix}_dataset_id_accuracy": float(accuracy_score(y_test, pred)),
        f"{prefix}_dataset_id_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }


def _probe_common_condition(
    x: np.ndarray,
    y: np.ndarray,
    condition: np.ndarray,
    prefix: str,
    min_datasets: int,
    min_samples: int,
) -> dict[str, float | int]:
    keep = np.zeros(len(y), dtype=bool)
    for key in sorted(set(condition.tolist())):
        idx = condition == key
        if idx.sum() < min_samples:
            continue
        if len(set(y[idx].tolist())) >= min_datasets:
            keep |= idx
    if keep.sum() == 0:
        return {f"{prefix}_num_samples": 0}
    out = _probe(x[keep], y[keep], prefix)
    out[f"{prefix}_num_conditions"] = int(len(set(condition[keep].tolist())))
    return out


if __name__ == "__main__":
    main()

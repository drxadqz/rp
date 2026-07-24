from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.engine import build_model, dataloader_worker_settings, move_batch
from friction_affordance.ontology import RISK
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--config-b", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--num-bootstrap", type=int, default=500)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    cfg_a, model_a, device = _load_model(args.config_a, args.checkpoint_a)
    cfg_b, model_b, _ = _load_model(args.config_b, args.checkpoint_b, device=device)
    test = _collect_pair(model_a, model_b, cfg_a, device, args.split)
    calib = _collect_pair(model_a, model_b, cfg_a, device, "val")
    report = build_report(
        test,
        calib,
        name_a=args.name_a,
        name_b=args.name_b,
        num_bootstrap=max(1, int(args.num_bootstrap)),
        target_coverage=float(args.target_coverage),
        seed=int(args.seed),
    )
    report["config_a"] = str(args.config_a)
    report["config_b"] = str(args.config_b)
    report["checkpoint_a"] = str(args.checkpoint_a)
    report["checkpoint_b"] = str(args.checkpoint_b)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def _load_model(
    config_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], torch.nn.Module, torch.device]:
    cfg = load_yaml(config_path)
    device = device or resolve_device(cfg.get("device", "auto"))
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, model, device


@torch.no_grad()
def _collect_pair(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    split: str,
) -> dict[str, Any]:
    loader = _build_loader(cfg, split)
    out: dict[str, Any] = {
        "tasks": {
            task: {"y_true": [], "pred_a": [], "pred_b": [], "dataset": []}
            for task in ["friction", "risk"]
        },
        "mu": {
            "pred_a": [],
            "pred_b": [],
            "target": [],
        },
    }
    for batch in loader:
        moved = move_batch(batch, device)
        pred_a = model_a(moved["image"], domain_idx=moved.get("domain_idx"))
        pred_b = model_b(moved["image"], domain_idx=moved.get("domain_idx"))
        dataset = np.asarray(batch["dataset"])
        for task in ["friction", "risk"]:
            mask = moved["masks"][task].detach().cpu().numpy().astype(bool)
            if not mask.any():
                continue
            raw = out["tasks"][task]
            raw["y_true"].extend(moved["labels"][task].detach().cpu().numpy()[mask].tolist())
            raw["pred_a"].extend(pred_a["logits"][task].argmax(dim=1).detach().cpu().numpy()[mask].tolist())
            raw["pred_b"].extend(pred_b["logits"][task].argmax(dim=1).detach().cpu().numpy()[mask].tolist())
            raw["dataset"].extend(dataset[mask].tolist())

        mu_mask = moved["mu_mask"]
        if mu_mask.any():
            raw_mu = out["mu"]
            raw_mu["pred_a"].append(pred_a["mu_interval"][mu_mask].detach().cpu().numpy())
            raw_mu["pred_b"].append(pred_b["mu_interval"][mu_mask].detach().cpu().numpy())
            raw_mu["target"].append(moved["mu_interval"][mu_mask].detach().cpu().numpy())
    mu = out["mu"]
    for key in ["pred_a", "pred_b", "target"]:
        mu[key] = np.concatenate(mu[key], axis=0) if mu[key] else np.empty((0, 2), dtype=float)
    return out


def _build_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    data_cfg = cfg["data"]
    manifests = data_cfg.get(f"{split}_manifests", data_cfg["val_manifests"])
    ds = ManifestDataset(
        manifests,
        transform=build_transforms(
            int(data_cfg.get("image_size", 224)),
            train=False,
            aug_cfg=data_cfg.get("augmentation"),
        ),
        max_samples=data_cfg.get(f"max_{split}_samples"),
        max_samples_per_dataset=data_cfg.get(f"max_{split}_samples_per_dataset"),
        max_samples_per_class=data_cfg.get(f"max_{split}_samples_per_class"),
        sample_seed=int(data_cfg.get("sample_seed", 17)) + (1 if split == "val" else 2),
    )
    num_workers, loader_kwargs = dataloader_worker_settings(data_cfg)
    return DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )


def build_report(
    test: dict[str, Any],
    calib: dict[str, Any],
    *,
    name_a: str,
    name_b: str,
    num_bootstrap: int,
    target_coverage: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    report = {
        "name_a": name_a,
        "name_b": name_b,
        "delta_definition": f"{name_a} - {name_b}",
        "num_bootstrap": int(num_bootstrap),
        "target_coverage": float(target_coverage),
        "metrics": {},
    }
    for task in ["friction", "risk"]:
        raw = test["tasks"][task]
        y_true = np.asarray(raw["y_true"], dtype=int)
        pred_a = np.asarray(raw["pred_a"], dtype=int)
        pred_b = np.asarray(raw["pred_b"], dtype=int)
        dataset = np.asarray(raw["dataset"])
        if len(y_true) == 0:
            continue
        report["metrics"][f"{task}_macro_f1_delta"] = _bootstrap_delta(
            len(y_true),
            lambda idx, yt=y_true, a=pred_a, b=pred_b: _macro_f1(yt[idx], a[idx])
            - _macro_f1(yt[idx], b[idx]),
            lambda yt=y_true, a=pred_a, b=pred_b: _macro_f1(yt, a) - _macro_f1(yt, b),
            rng,
            num_bootstrap,
        )
        report["metrics"][f"{task}_worst_dataset_macro_f1_delta"] = _bootstrap_delta(
            len(y_true),
            lambda idx, yt=y_true, a=pred_a, b=pred_b, ds=dataset: _worst_dataset_f1(
                yt[idx], a[idx], ds[idx]
            )
            - _worst_dataset_f1(yt[idx], b[idx], ds[idx]),
            lambda yt=y_true, a=pred_a, b=pred_b, ds=dataset: _worst_dataset_f1(yt, a, ds)
            - _worst_dataset_f1(yt, b, ds),
            rng,
            num_bootstrap,
        )

    risk = test["tasks"]["risk"]
    y_true = np.asarray(risk["y_true"], dtype=int)
    pred_a = np.asarray(risk["pred_a"], dtype=int)
    pred_b = np.asarray(risk["pred_b"], dtype=int)
    if len(y_true):
        high_idx = RISK.index("high")
        low_true = y_true >= high_idx
        low_a = pred_a >= high_idx
        low_b = pred_b >= high_idx
        report["metrics"]["low_friction_recall_delta"] = _bootstrap_delta(
            len(y_true),
            lambda idx: float(recall_score(low_true[idx], low_a[idx], zero_division=0))
            - float(recall_score(low_true[idx], low_b[idx], zero_division=0)),
            lambda: float(recall_score(low_true, low_a, zero_division=0))
            - float(recall_score(low_true, low_b, zero_division=0)),
            rng,
            num_bootstrap,
        )

    mu_test = test["mu"]
    mu_calib = calib["mu"]
    if len(mu_test["target"]):
        radius_a = _conformal_radius(_scores(mu_calib["pred_a"], mu_calib["target"]), target_coverage)
        radius_b = _conformal_radius(_scores(mu_calib["pred_b"], mu_calib["target"]), target_coverage)
        for metric_name, fn in [
            ("raw_interval_coverage_delta", lambda idx=None: _coverage(mu_test["pred_a"], mu_test["target"], 0.0, idx)
            - _coverage(mu_test["pred_b"], mu_test["target"], 0.0, idx)),
            ("calibrated_interval_coverage_delta", lambda idx=None: _coverage(
                mu_test["pred_a"], mu_test["target"], radius_a, idx
            )
            - _coverage(mu_test["pred_b"], mu_test["target"], radius_b, idx)),
            ("calibrated_interval_width_delta", lambda idx=None: _width(mu_test["pred_a"], radius_a, idx)
            - _width(mu_test["pred_b"], radius_b, idx)),
        ]:
            report["metrics"][metric_name] = _bootstrap_delta(
                len(mu_test["target"]),
                lambda idx, f=fn: f(idx),
                lambda f=fn: f(None),
                rng,
                num_bootstrap,
            )
        report["calibration"] = {
            "radius_a": float(radius_a),
            "radius_b": float(radius_b),
        }
    return report


def _bootstrap_delta(
    n: int,
    bootstrap_fn: Callable[[np.ndarray], float],
    point_fn: Callable[[], float],
    rng: np.random.Generator,
    num_bootstrap: int,
) -> dict[str, float | int]:
    point = float(point_fn())
    if n <= 1:
        return {"point": point, "ci_low": point, "ci_high": point, "num_samples": int(n)}
    values = np.empty(num_bootstrap, dtype=float)
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        values[i] = float(bootstrap_fn(idx))
    return {
        "point": point,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "num_samples": int(n),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _worst_dataset_f1(y_true: np.ndarray, y_pred: np.ndarray, dataset: np.ndarray) -> float:
    scores = []
    for name in sorted(set(dataset.tolist())):
        keep = dataset == name
        if keep.any():
            scores.append(_macro_f1(y_true[keep], y_pred[keep]))
    return float(min(scores)) if scores else 0.0


def _scores(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(target) == 0:
        return np.zeros(0, dtype=float)
    return np.maximum.reduce([pred[:, 0] - target[:, 0], target[:, 1] - pred[:, 1], np.zeros(len(target))])


def _conformal_radius(scores: np.ndarray, target_coverage: float) -> float:
    if len(scores) == 0:
        return 0.0
    sorted_scores = np.sort(scores)
    idx = int(math.ceil((len(sorted_scores) + 1) * float(np.clip(target_coverage, 0.0, 1.0))) - 1)
    idx = min(max(idx, 0), len(sorted_scores) - 1)
    return float(sorted_scores[idx])


def _coverage(pred: np.ndarray, target: np.ndarray, radius: float, idx: np.ndarray | None) -> float:
    pred = pred if idx is None else pred[idx]
    target = target if idx is None else target[idx]
    low = np.clip(pred[:, 0] - radius, 0.0, 1.2)
    high = np.clip(pred[:, 1] + radius, 0.0, 1.2)
    return float(((low <= target[:, 0]) & (high >= target[:, 1])).mean())


def _width(pred: np.ndarray, radius: float, idx: np.ndarray | None) -> float:
    pred = pred if idx is None else pred[idx]
    low = np.clip(pred[:, 0] - radius, 0.0, 1.2)
    high = np.clip(pred[:, 1] + radius, 0.0, 1.2)
    return float((high - low).mean())


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired Bootstrap Model Comparison",
        "",
        f"Delta: `{report['delta_definition']}`",
        f"Bootstrap samples: `{report['num_bootstrap']}`",
        "",
        "| Metric | Delta | 95% CI | N |",
        "|---|---:|---:|---:|",
    ]
    for name, item in report["metrics"].items():
        percent = "width" not in name
        lines.append(
            "| {name} | {point} | [{low}, {high}] | {n} |".format(
                name=name,
                point=_fmt(item["point"], percent),
                low=_fmt(item["ci_low"], percent),
                high=_fmt(item["ci_high"], percent),
                n=item.get("num_samples", "-"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float, percent: bool) -> str:
    return f"{100.0 * float(value):+.2f}" if percent else f"{float(value):+.4f}"


if __name__ == "__main__":
    main()

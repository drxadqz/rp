from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rscd_surface_classification as rscd  # noqa: E402


def summarize(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, Any]:
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[idx] for idx in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "summary": {
            "top1": float(accuracy_score(y_true, y_pred)),
            "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "num_samples": int(y_true.shape[0]),
            "num_classes": int(len(labels)),
        },
        "classification_report": report,
    }


def parse_group(class_name: str) -> int:
    if class_name in {"fresh_snow", "melted_snow", "ice"}:
        return 3
    parts = class_name.split("_")
    friction = parts[0] if len(parts) > 0 else ""
    material = parts[1] if len(parts) > 1 else ""
    if material in {"mud", "gravel"}:
        return 2
    if material in {"asphalt", "concrete"} and friction in {"wet", "water"}:
        return 1
    return 0


def factor_ids(class_to_idx: dict[str, int]) -> dict[str, torch.Tensor]:
    friction_vocab = {"dry": 0, "wet": 1, "water": 2, "fresh_snow": 3, "melted_snow": 4, "ice": 5}
    material_vocab = {"asphalt": 0, "concrete": 1, "mud": 2, "gravel": 3, "winter": 4}
    roughness_vocab = {"smooth": 0, "slight": 1, "severe": 2, "granular": 3, "winter": 4}
    n = len(class_to_idx)
    f = torch.zeros(n, dtype=torch.long)
    m = torch.zeros(n, dtype=torch.long)
    u = torch.zeros(n, dtype=torch.long)
    g = torch.zeros(n, dtype=torch.long)
    for class_name, idx in class_to_idx.items():
        item = rscd._factor_text(class_name)
        friction = item["friction"] if item["friction"] is not None else "dry"
        material = item["material"] if item["material"] is not None else "winter"
        roughness = item["unevenness"] if item["unevenness"] is not None else "winter"
        f[int(idx)] = friction_vocab[str(friction)]
        m[int(idx)] = material_vocab[str(material)]
        u[int(idx)] = roughness_vocab[str(roughness)]
        g[int(idx)] = parse_group(class_name)
    return {"friction": f, "material": m, "unevenness": u, "group": g}


class RelationTensorResidual(nn.Module):
    """Low-rank relation-specific residual on cached RSPNet logits and physics features."""

    def __init__(
        self,
        *,
        in_dim: int,
        class_to_idx: dict[str, int],
        rank: int = 16,
        hidden_dim: int = 128,
        scale: float = 0.12,
        dropout: float = 0.10,
        components: tuple[str, ...] = ("factor", "cp", "group", "boundary"),
    ) -> None:
        super().__init__()
        self.num_classes = len(class_to_idx)
        self.scale = float(scale)
        self.components = set(components)
        ids = factor_ids(class_to_idx)
        for name, value in ids.items():
            self.register_buffer(f"{name}_ids", value)
        self.norm = nn.LayerNorm(in_dim)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.factor_logits = nn.ModuleDict(
            {
                "friction": nn.Linear(hidden_dim, int(ids["friction"].max().item()) + 1),
                "material": nn.Linear(hidden_dim, int(ids["material"].max().item()) + 1),
                "unevenness": nn.Linear(hidden_dim, int(ids["unevenness"].max().item()) + 1),
            }
        )
        self.alpha = nn.Linear(hidden_dim, rank)
        self.fr = nn.Parameter(torch.empty(int(ids["friction"].max().item()) + 1, rank))
        self.ma = nn.Parameter(torch.empty(int(ids["material"].max().item()) + 1, rank))
        self.un = nn.Parameter(torch.empty(int(ids["unevenness"].max().item()) + 1, rank))
        self.group_gate = nn.Linear(hidden_dim, int(ids["group"].max().item()) + 1)
        self.group_bias = nn.Parameter(torch.zeros(int(ids["group"].max().item()) + 1, self.num_classes))
        self.boundary = nn.Sequential(
            nn.LayerNorm(in_dim + self.num_classes),
            nn.Linear(in_dim + self.num_classes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        nn.init.trunc_normal_(self.fr, std=0.02)
        nn.init.trunc_normal_(self.ma, std=0.02)
        nn.init.trunc_normal_(self.un, std=0.02)
        nn.init.zeros_(self.group_bias)
        nn.init.zeros_(self.boundary[-1].weight)
        nn.init.zeros_(self.boundary[-1].bias)

    def forward(self, x: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        h = self.shared(self.norm(x))
        residual = torch.zeros_like(base_logits)
        if "factor" in self.components:
            f_logits = self.factor_logits["friction"](h).index_select(1, self.friction_ids)
            m_logits = self.factor_logits["material"](h).index_select(1, self.material_ids)
            u_logits = self.factor_logits["unevenness"](h).index_select(1, self.unevenness_ids)
            residual = residual + f_logits + m_logits + u_logits

        if "cp" in self.components:
            alpha = torch.tanh(self.alpha(h))
            class_tensor = (
                self.fr.index_select(0, self.friction_ids)
                * self.ma.index_select(0, self.material_ids)
                * self.un.index_select(0, self.unevenness_ids)
            )
            residual = residual + alpha @ class_tensor.t()

        if "group" in self.components:
            group_prob = torch.softmax(self.group_gate(h), dim=1)
            residual = residual + group_prob @ self.group_bias

        if "boundary" in self.components:
            with torch.no_grad():
                probs = torch.softmax(base_logits, dim=1)
                top2 = torch.topk(probs, k=2, dim=1)
                margin = (top2.values[:, 0] - top2.values[:, 1]).view(-1, 1)
                uncertainty = (1.0 - top2.values[:, 0]).view(-1, 1)
                gate = torch.sigmoid((uncertainty - 0.04) * 20.0) * torch.sigmoid((0.18 - margin) * 20.0)
            residual = residual + self.boundary(torch.cat([x, base_logits], dim=1)) * gate
        return base_logits + torch.tanh(residual) * self.scale


def class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, base: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    pred = []
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        logits = model(x[start:end], base[start:end])
        pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(pred, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_val.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests_full/rscd_prepared_test.csv"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--scale", type=float, default=0.08)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--components",
        default="factor,cp,group,boundary",
        help="Comma-separated residual components to enable: factor,cp,group,boundary.",
    )
    parser.add_argument(
        "--selection-split",
        choices=("holdout", "test"),
        default="holdout",
        help="Select best epoch on a calibration holdout by default; use test only for diagnostic reproduction.",
    )
    parser.add_argument("--holdout-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_to_idx = rscd.build_class_map([args.train_manifest, args.val_manifest, args.test_manifest])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    data = np.load(args.cache, allow_pickle=True)
    x_cal = data["x_cal"].astype(np.float32)
    y_cal = data["y_cal"].astype(np.int64)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)
    num_classes = len(class_to_idx)

    # Keep logits in their original scale, but standardize auxiliary features.
    mean = x_cal[:, num_classes:].mean(axis=0, keepdims=True)
    std = x_cal[:, num_classes:].std(axis=0, keepdims=True) + 1e-6
    x_cal_scaled = x_cal.copy()
    x_test_scaled = x_test.copy()
    x_cal_scaled[:, num_classes:] = (x_cal_scaled[:, num_classes:] - mean) / std
    x_test_scaled[:, num_classes:] = (x_test_scaled[:, num_classes:] - mean) / std

    train_idx = np.arange(y_cal.shape[0])
    select_idx = np.arange(y_test.shape[0])
    selection_name = str(args.selection_split)
    if selection_name == "holdout":
        train_idx, select_idx = train_test_split(
            np.arange(y_cal.shape[0]),
            test_size=float(args.holdout_ratio),
            random_state=int(args.seed),
            stratify=y_cal,
        )
        x_select_np = x_cal_scaled[select_idx]
        base_select_np = x_cal[select_idx, :num_classes]
        y_select_np = y_cal[select_idx]
    else:
        x_select_np = x_test_scaled
        base_select_np = x_test[:, :num_classes]
        y_select_np = y_test

    x_train = torch.tensor(x_cal_scaled[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y_cal[train_idx], dtype=torch.long, device=device)
    base_train = torch.tensor(x_cal[train_idx, :num_classes], dtype=torch.float32, device=device)
    x_select = torch.tensor(x_select_np, dtype=torch.float32, device=device)
    base_select = torch.tensor(base_select_np, dtype=torch.float32, device=device)
    x_eval = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
    base_eval = torch.tensor(x_test[:, :num_classes], dtype=torch.float32, device=device)

    model = RelationTensorResidual(
        in_dim=x_train.shape[1],
        class_to_idx=class_to_idx,
        rank=args.rank,
        hidden_dim=args.hidden_dim,
        scale=args.scale,
        dropout=args.dropout,
        components=tuple(part.strip() for part in str(args.components).split(",") if part.strip()),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights(y_cal[train_idx], num_classes, device)
    loader = DataLoader(TensorDataset(x_train, base_train, y_train), batch_size=args.batch_size, shuffle=True)

    best = {"macro_f1": -1.0, "epoch": 0, "state": None, "pred": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, bb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb, bb)
            loss = F.cross_entropy(logits, yb, weight=weights, label_smoothing=0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += float(loss.item()) * int(yb.numel())
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            pred = predict(model, x_select, base_select, args.batch_size)
            macro = f1_score(y_select_np, pred, labels=list(range(num_classes)), average="macro", zero_division=0)
            top1 = accuracy_score(y_select_np, pred)
            print(json.dumps({"epoch": epoch, "loss": total_loss / len(train_idx), "selection": selection_name, "top1": top1, "macro_f1": macro}))
            if macro > best["macro_f1"]:
                best = {
                    "macro_f1": float(macro),
                    "epoch": int(epoch),
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "pred": pred.copy(),
                }

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    pred = predict(model, x_eval, base_eval, args.batch_size)
    base_pred = x_test[:, :num_classes].argmax(axis=1)
    selection_pred = predict(model, x_select, base_select, args.batch_size)
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({"device": str(device), "cache": str(args.cache)})
    result = {
        "selection": summarize(y_select_np, selection_pred, idx_to_class),
        "base": summarize(y_test, base_pred, idx_to_class),
        "relation_tensor_head": summarize(y_test, pred, idx_to_class),
        "best_epoch": int(best["epoch"]),
        "train_samples": int(len(train_idx)),
        "selection_samples": int(len(y_select_np)),
        "config": config,
    }
    (args.output_dir / "relation_tensor_head_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd_rows = [
        {"image_index": int(i), "true_label": idx_to_class[int(t)], "base_label": idx_to_class[int(b)], "pred_label": idx_to_class[int(p)]}
        for i, (t, b, p) in enumerate(zip(y_test, base_pred, pred, strict=True))
    ]
    try:
        import pandas as pd

        pd.DataFrame(pd_rows).to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8")
    except Exception:
        pass
    print(json.dumps({k: v["summary"] for k, v in result.items() if isinstance(v, dict) and "summary" in v}, indent=2))


if __name__ == "__main__":
    main()

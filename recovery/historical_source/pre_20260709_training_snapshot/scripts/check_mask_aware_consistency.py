from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.engine import build_loaders, build_model, train_one_epoch
from friction_affordance.utils import load_yaml, set_seed


DEFAULT_CONFIG = Path("configs/experiments/paper_protocol/v23_lean_region_mixture_evidence_safety.yaml")
DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/mask_aware_consistency_smoke.md")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/mask_aware_consistency_smoke.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU smoke-test the segmentation-style mask-aware attention "
            "consistency path without starting a competing GPU job."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    report = run_check(args.config, image_size=int(args.image_size), batch_size=int(args.batch_size))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "config": report["config"]}, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


def run_check(config: Path, *, image_size: int, batch_size: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config": config.stem,
        "path": str(config),
        "status": "ok",
        "claim_boundary": (
            "This is a wiring smoke test only. It proves that the mask-aware "
            "consistency loss is executable and logged; it does not prove metric "
            "improvement."
        ),
    }
    try:
        cfg = copy.deepcopy(load_yaml(config))
        set_seed(int(cfg.get("seed", 79)))
        data = cfg.setdefault("data", {})
        data["image_size"] = max(int(image_size), 32)
        data["batch_size"] = max(int(batch_size), 1)
        data["num_workers"] = 0
        data["balanced_num_samples_per_epoch"] = data["batch_size"]
        data["max_train_samples"] = max(data["batch_size"] * 2, 2)
        data["max_val_samples"] = max(data["batch_size"] * 2, 2)

        model_cfg = cfg.setdefault("model", {})
        model_cfg["pretrained"] = False
        model_cfg["embedding_dim"] = min(int(model_cfg.get("embedding_dim", 64)), 64)
        model_cfg["physics_dim"] = min(int(model_cfg.get("physics_dim", 16)), 16)
        model_cfg["evidence_dim"] = min(int(model_cfg.get("evidence_dim", 16)), 16)
        model_cfg["evidence_hidden_dim"] = min(int(model_cfg.get("evidence_hidden_dim", 16)), 16)

        loss_cfg = dict(cfg.get("loss", {}))
        loss_cfg["grad_accum_steps"] = 1
        loss_cfg["log_every_steps"] = 0

        train_loader, _ = build_loaders(cfg)
        model_cfg["num_domains"] = int(getattr(train_loader.dataset, "num_domains", 0))
        model = build_model(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            torch.device("cpu"),
            loss_cfg,
            use_amp=False,
        )
        keys = [
            "loss",
            "loss_aug_consistency",
            "loss_aug_consistency_attention",
            "aug_consistency_attention_mask_mean",
            "loss_aug_consistency_interval",
            "attention_pseudo_road_mass",
            "loss_evidence_attention_pseudo_road",
        ]
        row["logs"] = {key: logs.get(key) for key in keys}
        missing = [
            key
            for key in [
                "loss_aug_consistency",
                "loss_aug_consistency_attention",
                "aug_consistency_attention_mask_mean",
            ]
            if logs.get(key) is None
        ]
        if missing:
            row["status"] = "failed"
            row["error"] = f"Missing expected mask-aware consistency logs: {missing}"
    except Exception as exc:  # noqa: BLE001 - smoke report should catch all wiring failures.
        row.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    return row


def render_markdown(report: dict[str, Any]) -> str:
    logs = report.get("logs", {}) if isinstance(report.get("logs"), dict) else {}
    lines = [
        "# Mask-Aware Consistency Smoke Check",
        "",
        f"Config: `{report.get('path')}`",
        f"Status: `{report.get('status')}`",
        "",
        f"Claim boundary: {report.get('claim_boundary')}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in logs.items():
        lines.append(f"| `{key}` | {_fmt(value)} |")
    if report.get("error"):
        lines.extend(["", f"Error: `{report.get('error')}`"])
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()

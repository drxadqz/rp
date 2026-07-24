from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.engine import build_loaders, build_model, move_batch
from friction_affordance.losses import compute_total_loss
from friction_affordance.utils import load_yaml, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU smoke-test config model forward and loss composition."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config-dir", type=Path)
    source.add_argument("--config", type=Path, action="append")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--keep-pretrained", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    configs = _resolve_configs(args)
    rows = [
        _check_config(
            config=path,
            image_size=max(int(args.image_size), 32),
            batch_size=max(int(args.batch_size), 1),
            split=args.split,
            keep_pretrained=bool(args.keep_pretrained),
            backward=bool(args.backward),
        )
        for path in configs
    ]
    report = {
        "config_count": len(configs),
        "checks": len(rows),
        "failures": [row for row in rows if row["status"] != "ok"],
        "rows": rows,
        "note": (
            "Smoke test uses small CPU inputs and pretrained=False by default. "
            "It validates wiring/shape/loss logic, not final accuracy or GPU memory."
        ),
    }
    if args.out_json:
        save_json(report, args.out_json)
    if args.out_md:
        _write_md(report, args.out_md)
    print(json.dumps({"checks": report["checks"], "failures": len(report["failures"])}, indent=2))
    if report["failures"]:
        raise SystemExit(1)


def _resolve_configs(args: argparse.Namespace) -> list[Path]:
    if args.config_dir is not None:
        return sorted(args.config_dir.glob("*.yaml"))
    return sorted(Path(path) for path in (args.config or []))


def _check_config(
    *,
    config: Path,
    image_size: int,
    batch_size: int,
    split: str,
    keep_pretrained: bool,
    backward: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {"config": config.stem, "path": str(config), "status": "ok"}
    try:
        cfg = copy.deepcopy(load_yaml(config))
        set_seed(int(cfg.get("seed", 79)))
        data_cfg = cfg.setdefault("data", {})
        data_cfg["num_workers"] = 0
        data_cfg["image_size"] = int(image_size)
        data_cfg["batch_size"] = int(batch_size)
        data_cfg["balanced_num_samples_per_epoch"] = int(batch_size)
        data_cfg["max_train_samples"] = max(int(batch_size) * 2, 2)
        data_cfg["max_val_samples"] = max(int(batch_size) * 2, 2)
        model_cfg = cfg.setdefault("model", {})
        if not keep_pretrained:
            model_cfg["pretrained"] = False

        train_loader, val_loader = build_loaders(cfg)
        loader = train_loader if split == "train" else val_loader
        model_cfg["num_domains"] = int(getattr(loader.dataset, "num_domains", 0))
        data_cfg["num_groups"] = int(getattr(loader.dataset, "num_groups", 0))

        device = torch.device("cpu")
        model = build_model(cfg).to(device)
        model.train()
        batch = move_batch(next(iter(loader)), device)
        loss_cfg = dict(cfg.get("loss", {}))
        loss_cfg["grad_clip_norm"] = float(cfg.get("optim", {}).get("grad_clip_norm", 5.0))
        loss_cfg["grad_accum_steps"] = 1
        grl_lambda = float(loss_cfg.get("domain_grl_lambda", loss_cfg.get("domain_weight", 0.0)))
        outputs = model(batch["image"], grl_lambda=grl_lambda, domain_idx=batch.get("domain_idx"))
        loss, logs = compute_total_loss(outputs, batch, loss_cfg)
        if backward:
            loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {float(loss.detach())}")
        row.update(
            {
                "loss_total": float(loss.detach()),
                "batch_shape": list(batch["image"].shape),
                "num_domains": int(model_cfg["num_domains"]),
                "num_groups": int(data_cfg["num_groups"]),
                "has_evidence": bool(outputs.get("evidence_field")),
                "has_friction_set": bool(outputs.get("friction_set")),
                "has_domain_logits": bool(outputs.get("domain_logits") is not None),
                "has_domain_adapter_penalty": bool(outputs.get("domain_adapter_penalty") is not None),
                "loss_evidence_attention_pseudo_road": logs.get("loss_evidence_attention_pseudo_road"),
                "loss_evidence_attention_region": logs.get("loss_evidence_attention_region"),
                "loss_evidence_query_diversity": logs.get("loss_evidence_query_diversity"),
                "evidence_query_attention_overlap": logs.get("evidence_query_attention_overlap"),
                "loss_domain_adapter": logs.get("loss_domain_adapter"),
                "coverage_weight_mean": logs.get("coverage_weight_mean"),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report all smoke-test failures.
        row.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    return row


def _write_md(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Config Forward/Loss Smoke Check",
        "",
        report["note"],
        "",
        f"Checks: `{report['checks']}`; failures: `{len(report['failures'])}`.",
        "",
        "| Config | Status | Loss | Batch | Evidence | FrictionSet | Domain | Adapter | Pseudo-road | ROI | QueryDiv | QueryOverlap | Safety weight | Error |",
        "|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        batch = row.get("batch_shape", "-")
        if isinstance(batch, list):
            batch = "x".join(str(item) for item in batch)
        lines.append(
            "| {config} | {status} | {loss} | {batch} | {evidence} | {fset} | {domain} | {adapter} | {pseudo} | {roi} | {query_div} | {query_overlap} | {safety} | {error} |".format(
                config=row.get("config"),
                status=row.get("status"),
                loss=_fmt_float(row.get("loss_total")),
                batch=batch,
                evidence="yes" if row.get("has_evidence") else "-",
                fset="yes" if row.get("has_friction_set") else "-",
                domain="yes" if row.get("has_domain_logits") else "-",
                adapter="yes" if row.get("has_domain_adapter_penalty") else "-",
                pseudo=_fmt_float(row.get("loss_evidence_attention_pseudo_road")),
                roi=_fmt_float(row.get("loss_evidence_attention_region")),
                query_div=_fmt_float(row.get("loss_evidence_query_diversity")),
                query_overlap=_fmt_float(row.get("evidence_query_attention_overlap")),
                safety=_fmt_float(row.get("coverage_weight_mean")),
                error=str(row.get("error", "-")).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch
from friction_affordance.transforms import build_transforms
from friction_affordance.utils import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight batch-construction checks for experiment configs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config-dir", type=Path)
    source.add_argument("--config", type=Path, action="append")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    configs = _resolve_configs(args)
    rows: list[dict[str, Any]] = []
    for config in configs:
        cfg = load_yaml(config)
        data_cfg = cfg.get("data", {})
        image_size = int(data_cfg.get("image_size", 160))
        aug_cfg = data_cfg.get("augmentation", {})
        for split in args.splits:
            rows.append(
                _check_one(
                    config=config,
                    data_cfg=data_cfg,
                    image_size=image_size,
                    aug_cfg=aug_cfg,
                    split=split,
                    max_samples=max(1, int(args.max_samples)),
                    batch_size=max(1, int(args.batch_size)),
                )
            )

    report = {
        "config_count": len(configs),
        "split_count": len(args.splits),
        "checks": len(rows),
        "failures": [row for row in rows if row["status"] != "ok"],
        "rows": rows,
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
    configs = [Path(p) for p in args.config or []]
    return sorted(configs)


def _check_one(
    *,
    config: Path,
    data_cfg: dict[str, Any],
    image_size: int,
    aug_cfg: dict[str, Any],
    split: str,
    max_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config": config.stem,
        "path": str(config),
        "split": split,
        "status": "ok",
    }
    try:
        key = f"{split}_manifests"
        manifests = [Path(p) for p in data_cfg.get(key, [])]
        if not manifests:
            raise ValueError(f"No manifests configured for split '{split}'.")
        missing = [str(p) for p in manifests if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing manifest(s): {missing}")

        transform = build_transforms(image_size, train=split == "train", aug_cfg=aug_cfg)
        dataset = ManifestDataset(manifests, transform=transform, max_samples=max_samples)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), collate_fn=collate_manifest_batch)
        batch = next(iter(loader))
        labels = {
            task: int(mask.sum().item())
            for task, mask in batch["masks"].items()
        }
        row.update(
            {
                "manifests": [str(p) for p in manifests],
                "dataset_len": int(len(dataset)),
                "batch_shape": list(batch["image"].shape),
                "datasets": sorted(set(batch["dataset"])),
                "num_domains": int(dataset.num_domains),
                "num_groups": int(dataset.num_groups),
                "valid_label_counts": labels,
                "mu_mask_count": int(batch["mu_mask"].sum().item()),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report all config/data failures.
        row.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return row


def _write_md(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Config Batch Check",
        "",
        f"Configs: `{report['config_count']}`; checks: `{report['checks']}`; failures: `{len(report['failures'])}`.",
        "",
        "| Config | Split | Status | Batch | Datasets | Domains | Groups | mu labels | Error |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        batch = row.get("batch_shape", "-")
        if isinstance(batch, list):
            batch = "x".join(str(v) for v in batch)
        datasets = ", ".join(row.get("datasets", [])) if row.get("datasets") else "-"
        error = row.get("error", "-")
        lines.append(
            "| {config} | {split} | {status} | {batch} | {datasets} | {domains} | {groups} | {mu} | {error} |".format(
                config=row["config"],
                split=row["split"],
                status=row["status"],
                batch=batch,
                datasets=datasets,
                domains=row.get("num_domains", "-"),
                groups=row.get("num_groups", "-"),
                mu=row.get("mu_mask_count", "-"),
                error=str(error).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

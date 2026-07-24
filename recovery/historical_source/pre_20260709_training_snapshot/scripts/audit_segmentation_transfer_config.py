from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.datasets import ManifestDataset, collate_manifest_batch  # noqa: E402
from friction_affordance.losses import compute_total_loss  # noqa: E402
from friction_affordance.models import FrictionAffordanceModel  # noqa: E402
from friction_affordance.transforms import build_mask_transforms, build_transforms  # noqa: E402


DEFAULT_OUT = Path("reports/paper_protocol_summary/segmentation_transfer_config_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit optional mask-supervised EvidenceField configs and cached road-mask manifests."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT.with_suffix(".md"))
    parser.add_argument("--sample-rows-per-manifest", type=int, default=8)
    args = parser.parse_args()

    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = cfg.get("data", {})
    checks: list[dict[str, Any]] = []
    _add_check(checks, "config_exists", args.config.exists(), path=str(args.config))
    _add_check(checks, "load_road_masks_enabled", data.get("load_road_masks") is True)
    _add_check(checks, "road_mask_pretransformed", data.get("road_mask_pretransformed") is True)
    aug = data.get("augmentation", {}) if isinstance(data.get("augmentation"), dict) else {}
    _add_check(checks, "no_random_resized_crop", aug.get("random_resized_crop") is False)
    _add_check(checks, "no_horizontal_flip", float(aug.get("horizontal_flip_p", 1.0)) == 0.0)

    manifest_rows = []
    total_checked = 0
    total_existing = 0
    for split in ("train", "val", "test"):
        for manifest in data.get(f"{split}_manifests", []) or []:
            row = _audit_manifest(Path(manifest), split, int(args.sample_rows_per_manifest))
            manifest_rows.append(row)
            total_checked += int(row.get("sampled_masks", 0))
            total_existing += int(row.get("existing_masks", 0))
    _add_check(
        checks,
        "manifest_mask_paths_exist",
        total_checked > 0 and total_checked == total_existing,
        sampled_masks=total_checked,
        existing_masks=total_existing,
    )
    batch_report = _batch_and_loss_smoke(cfg)
    _add_check(checks, "batch_has_road_mask", bool(batch_report.get("batch_has_road_mask")), **batch_report)
    _add_check(
        checks,
        "pseudo_road_loss_active",
        bool(batch_report.get("pseudo_road_loss_active")),
        pseudo_loss=batch_report.get("loss_evidence_attention_pseudo_road"),
        road_mass=batch_report.get("attention_pseudo_road_mass"),
    )

    failures = [row for row in checks if row["level"] == "fail"]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(args.config),
        "verdict": "pass" if not failures else "fail",
        "claim_boundary": (
            "This audit proves that cached road/contact pseudo masks can be loaded "
            "and used by the EvidenceField attention loss. It does not prove that "
            "the pseudo masks are ground truth or that the candidate improves metrics."
        ),
        "checks": checks,
        "manifest_rows": manifest_rows,
        "batch_report": batch_report,
    }


def _audit_manifest(path: Path, split: str, sample_rows: int) -> dict[str, Any]:
    if not path.exists():
        return {"split": split, "manifest": str(path), "status": "missing", "rows": 0}
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    has_column = "road_mask_path" in frame.columns
    sample = frame.head(max(int(sample_rows), 1)) if has_column else frame.head(0)
    existing = 0
    for value in sample.get("road_mask_path", []):
        if Path(str(value)).exists():
            existing += 1
    return {
        "split": split,
        "manifest": str(path),
        "status": "ok" if has_column else "missing_road_mask_path",
        "rows": int(len(frame)),
        "sampled_masks": int(len(sample)),
        "existing_masks": int(existing),
    }


def _batch_and_loss_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg.get("data", {})
    aug = data.get("augmentation", {}) if isinstance(data.get("augmentation"), dict) else {}
    image_size = int(data.get("image_size", 96))
    ds = ManifestDataset(
        data.get("train_manifests", []),
        transform=build_transforms(image_size, train=True, aug_cfg=aug),
        mask_transform=build_mask_transforms(
            image_size,
            aug,
            pretransformed=bool(data.get("road_mask_pretransformed", False)),
        ),
        load_road_masks=bool(data.get("load_road_masks", False)),
        max_samples=2,
    )
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_manifest_batch)))
    model = FrictionAffordanceModel(
        backbone="simple_cnn",
        embedding_dim=64,
        use_physics_branch=True,
        physics_dim=32,
        use_evidence_field=True,
        evidence_dim=32,
        evidence_hidden_dim=24,
        evidence_road_likelihood_prior_strength=0.5,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image"])
        loss, logs = compute_total_loss(
            outputs,
            batch,
            {
                "task_weight": 1.0,
                "interval_weight": 0.1,
                "evidence_risk_weight": 0.1,
                "evidence_interval_weight": 0.05,
                "evidence_attention_pseudo_road_weight": 0.1,
                "evidence_pseudo_road_min_mass": 0.5,
                "evidence_pseudo_road_threshold": 0.35,
            },
        )
    pseudo_loss = float(logs.get("loss_evidence_attention_pseudo_road", 0.0))
    return {
        "dataset_len": int(len(ds)),
        "batch_has_road_mask": "road_mask" in batch,
        "image_shape": list(batch["image"].shape),
        "road_mask_shape": list(batch["road_mask"].shape) if "road_mask" in batch else None,
        "road_mask_min": float(batch["road_mask"].min()) if "road_mask" in batch else None,
        "road_mask_max": float(batch["road_mask"].max()) if "road_mask" in batch else None,
        "loss_total": float(loss.detach()),
        "loss_evidence_attention_pseudo_road": pseudo_loss,
        "attention_pseudo_road_mass": logs.get("attention_pseudo_road_mass"),
        "pseudo_road_loss_active": pseudo_loss > 0.0,
    }


def _add_check(checks: list[dict[str, Any]], name: str, ok: bool, **details: Any) -> None:
    checks.append({"level": "pass" if ok else "fail", "name": name, **details})


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Segmentation Transfer Config Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Checks",
        "",
        "| Level | Check | Details |",
        "|---|---|---|",
    ]
    for row in report.get("checks", []):
        details = {k: v for k, v in row.items() if k not in {"level", "name"}}
        lines.append(f"| {row['level']} | {row['name']} | `{json.dumps(details, ensure_ascii=False)}` |")
    lines.extend(["", "## Manifests", ""])
    lines.append("| Split | Manifest | Rows | Sampled masks | Existing masks | Status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in report.get("manifest_rows", []):
        lines.append(
            "| {split} | `{manifest}` | {rows} | {sampled} | {existing} | `{status}` |".format(
                split=row.get("split", "-"),
                manifest=row.get("manifest", "-"),
                rows=row.get("rows", 0),
                sampled=row.get("sampled_masks", 0),
                existing=row.get("existing_masks", 0),
                status=row.get("status", "-"),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

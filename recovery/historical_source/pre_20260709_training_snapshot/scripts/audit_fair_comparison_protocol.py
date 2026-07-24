from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DATASETS = ["roadsaw", "rscd", "roadsc"]
REQUIRED_COLUMNS = {
    "image_path",
    "split",
    "dataset",
    "class_label",
    "domain_id",
    "friction_label",
    "risk_label",
    "mu_low",
    "mu_high",
}
STRICT_DATA_KEYS = [
    "train_manifests",
    "val_manifests",
    "test_manifests",
    "image_size",
    "max_train_samples",
    "max_val_samples",
    "max_test_samples",
    "max_train_samples_per_class",
    "max_val_samples_per_class",
    "max_test_samples_per_class",
    "balanced_sampling",
    "balanced_dataset_first",
    "balanced_group_columns",
    "balanced_num_samples_per_epoch",
    "sample_seed",
]
STRICT_AUGMENTATION_KEYS = ["augmentation"]
STRICT_OPTIM_KEYS = [
    "epochs",
    "lr",
    "weight_decay",
    "grad_clip_norm",
    "amp",
    "early_stop_patience",
    "early_stop_min_delta",
]
STRICT_MODEL_KEYS = ["backbone", "embedding_dim", "dropout", "pretrained", "freeze_backbone"]
BASELINE_DISABLED_MODEL_FLAGS = ["use_physics_branch", "use_friction_set", "use_evidence_field"]
BASELINE_ZERO_LOSS_KEYS = [
    "compatibility_weight",
    "group_dro_weight",
    "group_vrex_weight",
    "domain_weight",
    "domain_grl_lambda",
    "feature_coral_weight",
    "risk_conditional_coral_weight",
    "evidence_risk_weight",
    "evidence_interval_weight",
    "evidence_endpoint_weight",
    "evidence_width_weight",
    "evidence_target_width_weight",
    "evidence_attention_prior_weight",
    "evidence_attention_smooth_weight",
    "evidence_risk_consistency_weight",
    "evidence_interval_consistency_weight",
    "evidence_attention_region_weight",
    "evidence_attention_pseudo_road_weight",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "fair_comparison_protocol_audit.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "fair_comparison_protocol_audit.json")
    args = parser.parse_args()

    report = audit(args.config_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def audit(config_dir: Path) -> dict[str, Any]:
    manifest_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    pair_specs = []
    for dataset in DATASETS:
        pair_specs.append(
            {
                "dataset": dataset,
                "scope": "single_dataset_full_faf_vs_convnext",
                "faf": f"single_{dataset}_full_faf",
                "baseline": f"baseline_single_{dataset}_global_convnext",
                "strict_augmentation": True,
            }
        )
        pair_specs.append(
            {
                "dataset": dataset,
                "scope": "final_lean_vs_convnext",
                "faf": f"final_single_{dataset}_lean_road_roi_safety",
                "baseline": f"baseline_single_{dataset}_global_convnext",
                "strict_augmentation": False,
            }
        )

    for spec in pair_specs:
        row_checks: list[dict[str, Any]] = []
        row = _audit_pair(config_dir, spec, manifest_cache, row_checks)
        row["status"] = _row_status(row_checks)
        row["checks"] = row_checks
        rows.append(row)
        checks.extend({"pair": row["pair"], **item} for item in row_checks)

    verdict = "pass"
    if any(item["level"] == "block" for item in checks):
        verdict = "fail"
    elif any(item["level"] == "warn" for item in checks):
        verdict = "pass_with_warnings"

    return {
        "config_dir": str(config_dir),
        "verdict": verdict,
        "num_pairs": len(rows),
        "num_blocks": sum(1 for item in checks if item["level"] == "block"),
        "num_warnings": sum(1 for item in checks if item["level"] == "warn"),
        "rows": rows,
        "checks": checks,
        "policy": [
            "Comparable single-dataset rows must use identical public train/val/test manifests, labels, metrics, and evaluation/calibration scripts.",
            "The matched ConvNeXt baseline must share the backbone, seed, optimizer budget, image size, sampling budget, and effective batch size.",
            "Full-FAF single-dataset rows should match the baseline augmentation policy; final-method rows may introduce declared method-specific augmentation or hard sampling.",
            "Reported advantages must come from completed paired runs with the same split and paired bootstrap deltas, not from unmatched published numbers.",
        ],
    }


def _audit_pair(
    config_dir: Path,
    spec: dict[str, Any],
    manifest_cache: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    faf_name = spec["faf"]
    baseline_name = spec["baseline"]
    faf_path = config_dir / f"{faf_name}.yaml"
    baseline_path = config_dir / f"{baseline_name}.yaml"
    faf = _load_config(faf_path, checks, "faf")
    baseline = _load_config(baseline_path, checks, "baseline")
    pair = f"{faf_name} vs {baseline_name}"
    row = {
        "pair": pair,
        "scope": spec["scope"],
        "dataset": spec["dataset"],
        "faf_config": str(faf_path),
        "baseline_config": str(baseline_path),
        "strict_augmentation": spec["strict_augmentation"],
    }
    if not faf or not baseline:
        return row

    _check_same_seed(checks, faf, baseline)
    _check_same_values(checks, "data", faf.get("data", {}), baseline.get("data", {}), STRICT_DATA_KEYS, "block")
    if spec["strict_augmentation"]:
        _check_same_values(
            checks,
            "augmentation",
            faf.get("data", {}),
            baseline.get("data", {}),
            STRICT_AUGMENTATION_KEYS,
            "block",
        )
    else:
        _check_declared_method_training_differences(checks, faf, baseline)
    _check_same_values(checks, "optim", faf.get("optim", {}), baseline.get("optim", {}), STRICT_OPTIM_KEYS, "block")
    _check_effective_batch(checks, faf, baseline)
    _check_same_values(checks, "model_backbone", faf.get("model", {}), baseline.get("model", {}), STRICT_MODEL_KEYS, "block")
    _check_baseline_is_global(checks, baseline)
    _check_faf_route(checks, faf, spec["scope"])

    manifest_summary = _audit_pair_manifests(
        checks,
        spec["dataset"],
        faf.get("data", {}),
        baseline.get("data", {}),
        manifest_cache,
    )
    row["manifest_summary"] = manifest_summary
    row["effective_batch_faf"] = _effective_batch(faf)
    row["effective_batch_baseline"] = _effective_batch(baseline)
    row["faf_output_dir"] = faf.get("output_dir")
    row["baseline_output_dir"] = baseline.get("output_dir")
    return row


def _load_config(path: Path, checks: list[dict[str, Any]], role: str) -> dict[str, Any]:
    if not path.exists():
        _add(checks, "block", "missing_config", f"Missing {role} config: {path}", path=str(path), role=role)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        _add(checks, "block", "invalid_config", f"{role} config is not a YAML mapping.", path=str(path), role=role)
        return {}
    return data


def _check_same_seed(checks: list[dict[str, Any]], faf: dict[str, Any], baseline: dict[str, Any]) -> None:
    if faf.get("seed") != baseline.get("seed"):
        _add(checks, "block", "seed_mismatch", "FAF and ConvNeXt baseline use different seeds.", faf=faf.get("seed"), baseline=baseline.get("seed"))


def _check_same_values(
    checks: list[dict[str, Any]],
    scope: str,
    left: dict[str, Any],
    right: dict[str, Any],
    keys: list[str],
    level: str,
) -> None:
    for key in keys:
        if _canonical(left.get(key)) != _canonical(right.get(key)):
            _add(
                checks,
                level,
                f"{scope}_{key}_mismatch",
                f"{scope}.{key} differs between FAF and ConvNeXt baseline.",
                faf=left.get(key),
                baseline=right.get(key),
            )


def _check_declared_method_training_differences(
    checks: list[dict[str, Any]],
    faf: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    faf_data = faf.get("data", {})
    baseline_data = baseline.get("data", {})
    diffs = []
    for key in ["augmentation", "balanced_weight_overrides"]:
        if _canonical(faf_data.get(key)) != _canonical(baseline_data.get(key)):
            diffs.append(key)
    if diffs:
        _add(
            checks,
            "warn",
            "declared_final_method_training_difference",
            "Final-method single-dataset row has declared method-specific training differences from the ConvNeXt baseline.",
            fields=diffs,
        )


def _check_effective_batch(checks: list[dict[str, Any]], faf: dict[str, Any], baseline: dict[str, Any]) -> None:
    left = _effective_batch(faf)
    right = _effective_batch(baseline)
    if left != right:
        _add(checks, "block", "effective_batch_mismatch", "FAF and baseline effective batch sizes differ.", faf=left, baseline=right)


def _effective_batch(cfg: dict[str, Any]) -> int | None:
    try:
        return int((cfg.get("data") or {}).get("batch_size")) * int((cfg.get("optim") or {}).get("grad_accum_steps", 1))
    except (TypeError, ValueError):
        return None


def _check_baseline_is_global(checks: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    model = baseline.get("model", {})
    loss = baseline.get("loss", {})
    bad_flags = [key for key in BASELINE_DISABLED_MODEL_FLAGS if bool(model.get(key))]
    bad_losses = [key for key in BASELINE_ZERO_LOSS_KEYS if _num(loss.get(key)) not in {None, 0.0}]
    if bad_flags:
        _add(checks, "block", "baseline_model_not_global", "ConvNeXt baseline has FAF modules enabled.", flags=bad_flags)
    if bad_losses:
        _add(checks, "block", "baseline_loss_not_global", "ConvNeXt baseline has FAF/domain/evidence losses enabled.", losses=bad_losses)
    if _num((baseline.get("loss") or {}).get("task_weight")) != 1.0:
        _add(checks, "block", "baseline_task_weight", "ConvNeXt baseline should use task_weight=1.0.", value=(baseline.get("loss") or {}).get("task_weight"))


def _check_faf_route(checks: list[dict[str, Any]], faf: dict[str, Any], scope: str) -> None:
    model = faf.get("model", {})
    if not bool(model.get("use_physics_branch")):
        _add(checks, "warn", "faf_physics_branch_disabled", "FAF comparison row does not use the PhysicsTexture branch.")
    if not bool(model.get("use_evidence_field")):
        _add(checks, "warn", "faf_evidence_field_disabled", "FAF comparison row does not use EvidenceField.")
    if scope == "final_lean_vs_convnext" and bool(model.get("use_friction_set")):
        _add(checks, "warn", "final_friction_set_enabled", "Final lean route should keep FrictionSet disabled unless later evidence rescues it.")


def _audit_pair_manifests(
    checks: list[dict[str, Any]],
    expected_dataset: str,
    faf_data: dict[str, Any],
    baseline_data: dict[str, Any],
    manifest_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    split_paths: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        key = f"{split}_manifests"
        faf_paths = [Path(item) for item in faf_data.get(key, [])]
        baseline_paths = [Path(item) for item in baseline_data.get(key, [])]
        if _canonical([str(path) for path in faf_paths]) != _canonical([str(path) for path in baseline_paths]):
            _add(checks, "block", f"{split}_manifest_mismatch", f"{split} manifests differ between FAF and baseline.", faf=[str(p) for p in faf_paths], baseline=[str(p) for p in baseline_paths])
        summaries = [_manifest_summary(path, manifest_cache) for path in faf_paths]
        combined = _combine_manifest_summaries(summaries)
        out[split] = {key: value for key, value in combined.items() if key != "_paths"}
        split_paths[split] = combined.get("_paths", set())
        missing_columns = sorted(REQUIRED_COLUMNS - set(combined.get("columns", [])))
        if missing_columns:
            _add(checks, "block", f"{split}_manifest_required_columns", f"{split} manifests miss required label columns.", missing=missing_columns)
        observed_datasets = set(combined.get("datasets", {}))
        if observed_datasets != {expected_dataset}:
            _add(checks, "block", f"{split}_dataset_mismatch", f"{split} split should contain only {expected_dataset}.", observed=sorted(observed_datasets))
        observed_split_values = set(combined.get("split_values", {}))
        if observed_split_values != {split}:
            _add(checks, "block", f"{split}_split_value_mismatch", f"{split} rows should have split={split}.", observed=sorted(observed_split_values))
        if combined.get("num_rows", 0) <= 0:
            _add(checks, "block", f"{split}_empty_manifest", f"{split} split has no rows.")
        if combined.get("invalid_mu_rows", 0):
            _add(checks, "block", f"{split}_invalid_mu_interval", f"{split} split has invalid weak friction intervals.", num_rows=combined.get("invalid_mu_rows"))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_paths[left] & split_paths[right]
        if overlap:
            _add(checks, "block", f"{left}_{right}_path_overlap", f"{left} and {right} image paths overlap.", num_overlap=len(overlap), examples=sorted(overlap)[:5])
    return out


def _manifest_summary(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path)
    if key in cache:
        return cache[key]
    if not path.exists():
        summary = {
            "path": key,
            "exists": False,
            "num_rows": 0,
            "columns": [],
            "datasets": {},
            "split_values": {},
            "class_top": {},
            "invalid_mu_rows": 0,
            "min_mu_low": None,
            "max_mu_high": None,
            "_paths": set(),
        }
        cache[key] = summary
        return summary
    datasets: Counter[str] = Counter()
    split_values: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    paths: set[str] = set()
    columns: list[str] = []
    invalid_mu = 0
    min_mu_low: float | None = None
    max_mu_high: float | None = None
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            datasets[_norm(row.get("dataset"))] += 1
            split_values[_norm(row.get("split"))] += 1
            classes[_norm(row.get("class_label"))] += 1
            image_path = _norm(row.get("image_path"))
            if image_path:
                paths.add(image_path)
            low = _num(row.get("mu_low"))
            high = _num(row.get("mu_high"))
            if low is None or high is None or low > high or low < 0.0 or high > 1.3:
                invalid_mu += 1
            else:
                min_mu_low = low if min_mu_low is None else min(min_mu_low, low)
                max_mu_high = high if max_mu_high is None else max(max_mu_high, high)
    summary = {
        "path": key,
        "exists": True,
        "num_rows": rows,
        "columns": columns,
        "datasets": dict(sorted(datasets.items())),
        "split_values": dict(sorted(split_values.items())),
        "class_top": dict(classes.most_common(6)),
        "invalid_mu_rows": invalid_mu,
        "min_mu_low": min_mu_low,
        "max_mu_high": max_mu_high,
        "_paths": paths,
    }
    cache[key] = summary
    return summary


def _combine_manifest_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: Counter[str] = Counter()
    split_values: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    paths: set[str] = set()
    columns: set[str] = set()
    num_rows = 0
    invalid_mu = 0
    min_mu_low: float | None = None
    max_mu_high: float | None = None
    for item in summaries:
        num_rows += int(item.get("num_rows", 0) or 0)
        invalid_mu += int(item.get("invalid_mu_rows", 0) or 0)
        datasets.update(item.get("datasets", {}))
        split_values.update(item.get("split_values", {}))
        classes.update(item.get("class_top", {}))
        columns.update(item.get("columns", []))
        paths.update(item.get("_paths", set()))
        low = item.get("min_mu_low")
        high = item.get("max_mu_high")
        if low is not None:
            min_mu_low = float(low) if min_mu_low is None else min(min_mu_low, float(low))
        if high is not None:
            max_mu_high = float(high) if max_mu_high is None else max(max_mu_high, float(high))
    return {
        "num_rows": num_rows,
        "columns": sorted(columns),
        "datasets": dict(sorted(datasets.items())),
        "split_values": dict(sorted(split_values.items())),
        "class_top": dict(classes.most_common(6)),
        "invalid_mu_rows": invalid_mu,
        "min_mu_low": min_mu_low,
        "max_mu_high": max_mu_high,
        "_paths": paths,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fair Comparison Protocol Audit",
        "",
        f"Config dir: `{report['config_dir']}`",
        f"Verdict: `{report['verdict']}` ({report['num_blocks']} blocks, {report['num_warnings']} warnings)",
        "",
        "## Pair Summary",
        "",
        "| Scope | Dataset | Pair | Status | Effective batch | Train rows | Val rows | Test rows |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        manifest = row.get("manifest_summary", {})
        lines.append(
            "| {scope} | {dataset} | `{pair}` | `{status}` | {batch} | {train} | {val} | {test} |".format(
                scope=row.get("scope"),
                dataset=row.get("dataset"),
                pair=row.get("pair"),
                status=row.get("status"),
                batch=row.get("effective_batch_faf"),
                train=(manifest.get("train") or {}).get("num_rows", "-"),
                val=(manifest.get("val") or {}).get("num_rows", "-"),
                test=(manifest.get("test") or {}).get("num_rows", "-"),
            )
        )
    lines.extend(["", "## Non-Pass Checks", "", "| Level | Pair | Check | Message |", "|---|---|---|---|"])
    non_pass = [item for item in report["checks"] if item.get("level") != "pass"]
    if not non_pass:
        lines.append("| pass | - | `all_checks` | All strict fair-comparison protocol checks passed. |")
    else:
        for item in non_pass:
            lines.append(f"| {item.get('level')} | `{item.get('pair')}` | `{item.get('name')}` | {item.get('message')} |")
    lines.extend(["", "## Policy", ""])
    for idx, item in enumerate(report.get("policy", []), start=1):
        lines.append(f"{idx}. {item}")
    lines.append("")
    return "\n".join(lines)


def _row_status(checks: list[dict[str, Any]]) -> str:
    if any(item["level"] == "block" for item in checks):
        return "fail"
    if any(item["level"] == "warn" for item in checks):
        return "pass_with_warnings"
    return "pass"


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\\", "/").lower()


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add(checks: list[dict[str, Any]], level: str, name: str, message: str, **details: Any) -> None:
    checks.append({"level": level, "name": name, "message": message, **details})


if __name__ == "__main__":
    main()

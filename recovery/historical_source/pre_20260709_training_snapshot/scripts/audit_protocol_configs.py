from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/protocol_config_audit.json")
DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/protocol_config_audit.md")

DATASETS = ("rscd", "roadsaw", "roadsc")
P0_RUNS = (
    "v0_global_only",
    "v1_physics_texture",
    "v2_friction_set",
    "v3_dg_losses",
    "v4_evidence_aux",
    "v5_full_faf",
)
CANDIDATE_RUNS = (
    "v6_full_faf_fourier",
    "v7_full_faf_fourier_dann",
    "v8_full_faf_fourier_roadprior",
    "v9_full_faf_roadsaw_hard_sampling",
    "v10_full_faf_consistency",
    "v11_full_faf_domain_adapter",
    "v12_full_faf_roi_interval_safety",
    "v13_lean_physics_evidence",
    "v14_lean_road_roi_safety",
    "v15_lean_bottom_square_style_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
)
FINAL_LODO_RUNS = tuple(f"final_lodo_{dataset}_lean_road_roi_safety" for dataset in DATASETS)
FINAL_SINGLE_RUNS = tuple(f"final_single_{dataset}_lean_road_roi_safety" for dataset in DATASETS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    report = audit(args.config_dir)
    md = render_markdown(report)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def audit(config_dir: Path) -> dict[str, Any]:
    configs = _load_configs(config_dir)
    checks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    _check_required_configs(checks, configs)
    _check_unique_outputs(checks, configs)
    for name, cfg in configs.items():
        row = _config_row(name, cfg)
        rows.append(row)
        _check_no_split_path_overlap(checks, row)

    _check_p0_protocol(checks, configs)
    _check_candidate_protocol(checks, configs)
    _check_lodo_protocol(checks, configs)
    _check_single_dataset_protocol(checks, configs)
    _check_final_method_protocol(checks, configs)

    verdict = "pass"
    if any(item["level"] == "block" for item in checks):
        verdict = "fail"
    elif any(item["level"] == "warn" for item in checks):
        verdict = "pass_with_warnings"
    return {
        "config_dir": str(config_dir),
        "verdict": verdict,
        "checks": checks,
        "runs": sorted(rows, key=lambda item: item["run"]),
    }


def _load_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        out[path.stem] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return out


def _check_required_configs(checks: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> None:
    required = set(P0_RUNS) | set(CANDIDATE_RUNS) | set(FINAL_LODO_RUNS) | set(FINAL_SINGLE_RUNS)
    required |= {
        f"lodo_{dataset}_full_faf" for dataset in DATASETS
    }
    required |= {
        f"single_{dataset}_full_faf" for dataset in DATASETS
    }
    required |= {
        f"baseline_single_{dataset}_global_convnext" for dataset in DATASETS
    }
    missing = sorted(required - set(configs))
    if missing:
        checks.append(
            {
                "level": "block",
                "name": "missing_required_configs",
                "message": "Some required paper-protocol configs are absent.",
                "missing": missing,
            }
        )


def _check_unique_outputs(checks: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> None:
    seen: dict[str, list[str]] = {}
    for name, cfg in configs.items():
        output_dir = str(cfg.get("output_dir", "")).strip().lower()
        if not output_dir:
            checks.append(
                {
                    "level": "block",
                    "name": "missing_output_dir",
                    "run": name,
                    "message": "Config has no output_dir.",
                }
            )
            continue
        seen.setdefault(output_dir, []).append(name)
    duplicates = {path: names for path, names in seen.items() if len(names) > 1}
    if duplicates:
        checks.append(
            {
                "level": "block",
                "name": "duplicate_output_dirs",
                "message": "Multiple configs write to the same output directory.",
                "duplicates": duplicates,
            }
        )


def _config_row(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    train = [str(item) for item in data.get("train_manifests", [])]
    val = [str(item) for item in data.get("val_manifests", [])]
    test = [str(item) for item in data.get("test_manifests", [])]
    return {
        "run": name,
        "train_manifests": train,
        "val_manifests": val,
        "test_manifests": test,
        "train_datasets": sorted(_datasets_from_paths(train)),
        "val_datasets": sorted(_datasets_from_paths(val)),
        "test_datasets": sorted(_datasets_from_paths(test)),
        "output_dir": str(cfg.get("output_dir", "")),
        "use_physics_branch": bool(model.get("use_physics_branch", False)),
        "use_friction_set": bool(model.get("use_friction_set", False)),
        "use_evidence_field": bool(model.get("use_evidence_field", False)),
    }


def _check_no_split_path_overlap(checks: list[dict[str, Any]], row: dict[str, Any]) -> None:
    splits = {
        "train": {_norm_path(item) for item in row["train_manifests"]},
        "val": {_norm_path(item) for item in row["val_manifests"]},
        "test": {_norm_path(item) for item in row["test_manifests"]},
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(splits[left] & splits[right])
        if overlap:
            checks.append(
                {
                    "level": "block",
                    "name": "split_manifest_overlap",
                    "run": row["run"],
                    "message": f"{left} and {right} use the same manifest path.",
                    "overlap": overlap,
                }
            )


def _check_p0_protocol(checks: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> None:
    rows = {name: _config_row(name, configs[name]) for name in P0_RUNS if name in configs}
    if len(rows) != len(P0_RUNS):
        return
    reference = rows["v0_global_only"]
    for name, row in rows.items():
        _require_dataset_set(checks, row, "train", set(DATASETS))
        _require_dataset_set(checks, row, "val", set(DATASETS))
        _require_dataset_set(checks, row, "test", set(DATASETS))
        for split in ("train_manifests", "val_manifests", "test_manifests"):
            if sorted(row[split]) != sorted(reference[split]):
                checks.append(
                    {
                        "level": "block",
                        "name": "p0_ablation_split_mismatch",
                        "run": name,
                        "message": "P0 ablations must use identical manifests for fair adjacent comparison.",
                        "split": split,
                    }
                )
    expected_flags = {
        "v0_global_only": (False, False, False),
        "v1_physics_texture": (True, False, False),
        "v2_friction_set": (True, True, False),
        "v3_dg_losses": (True, True, False),
        "v4_evidence_aux": (True, True, True),
        "v5_full_faf": (True, True, True),
    }
    for name, flags in expected_flags.items():
        row = rows.get(name)
        if not row:
            continue
        actual = (row["use_physics_branch"], row["use_friction_set"], row["use_evidence_field"])
        if actual != flags:
            checks.append(
                {
                    "level": "block",
                    "name": "p0_ablation_module_flag_mismatch",
                    "run": name,
                    "message": "P0 run does not match the expected cumulative module flags.",
                    "expected": flags,
                    "actual": actual,
                }
            )


def _check_candidate_protocol(checks: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> None:
    for name in CANDIDATE_RUNS:
        if name not in configs:
            continue
        row = _config_row(name, configs[name])
        _require_dataset_set(checks, row, "train", set(DATASETS))
        _require_dataset_set(checks, row, "val", set(DATASETS))
        _require_dataset_set(checks, row, "test", set(DATASETS))
        if "_lean_" in name:
            if not (row["use_physics_branch"] and row["use_evidence_field"]) or row["use_friction_set"]:
                checks.append(
                    {
                        "level": "block",
                        "name": "lean_candidate_module_mismatch",
                        "run": name,
                        "message": "Lean candidates must keep PhysicsTexture and EvidenceField while intentionally removing FrictionSet.",
                    }
                )
            continue
        if not (row["use_physics_branch"] and row["use_friction_set"] and row["use_evidence_field"]):
            checks.append(
                {
                    "level": "block",
                    "name": "candidate_not_full_faf",
                    "run": name,
                    "message": "Candidate-path methods must start from the full FAF model unless explicitly renamed.",
                }
            )


def _check_lodo_protocol(checks: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> None:
    for held_out in DATASETS:
        name = f"lodo_{held_out}_full_faf"
        if name not in configs:
            continue
        row = _config_row(name, configs[name])
        source = set(DATASETS) - {held_out}
        _require_dataset_set(checks, row, "train", source)
        _require_dataset_set(checks, row, "val", source)
        _require_dataset_set(checks, row, "test", {held_out})
        if held_out in row["train_datasets"] or held_out in row["val_datasets"]:
            checks.append(
                {
                    "level": "block",
                    "name": "lodo_heldout_leakage",
                    "run": name,
                    "message": "Held-out dataset appears in train/val manifests.",
                    "held_out": held_out,
                }
            )


def _check_single_dataset_protocol(checks: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> None:
    for dataset in DATASETS:
        faf_name = f"single_{dataset}_full_faf"
        base_name = f"baseline_single_{dataset}_global_convnext"
        if faf_name not in configs or base_name not in configs:
            continue
        faf = _config_row(faf_name, configs[faf_name])
        base = _config_row(base_name, configs[base_name])
        for row in (faf, base):
            _require_dataset_set(checks, row, "train", {dataset})
            _require_dataset_set(checks, row, "val", {dataset})
            _require_dataset_set(checks, row, "test", {dataset})
        for split in ("train_manifests", "val_manifests", "test_manifests"):
            if sorted(faf[split]) != sorted(base[split]):
                checks.append(
                    {
                        "level": "block",
                        "name": "single_dataset_baseline_split_mismatch",
                        "dataset": dataset,
                        "message": "Single-dataset FAF and ConvNeXt baseline must use identical manifests.",
                        "split": split,
                    }
                )
        if base["use_physics_branch"] or base["use_friction_set"] or base["use_evidence_field"]:
            checks.append(
                {
                    "level": "block",
                    "name": "baseline_has_faf_module_enabled",
                    "run": base_name,
                    "message": "Fair visual baseline should be global ConvNeXt without FAF modules.",
                }
            )


def _check_final_method_protocol(checks: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> None:
    for dataset in DATASETS:
        lodo_name = f"final_lodo_{dataset}_lean_road_roi_safety"
        if lodo_name in configs:
            row = _config_row(lodo_name, configs[lodo_name])
            source = set(DATASETS) - {dataset}
            _require_dataset_set(checks, row, "train", source)
            _require_dataset_set(checks, row, "val", source)
            _require_dataset_set(checks, row, "test", {dataset})
            _require_lean_final_modules(checks, row)

        single_name = f"final_single_{dataset}_lean_road_roi_safety"
        base_name = f"baseline_single_{dataset}_global_convnext"
        if single_name not in configs:
            continue
        row = _config_row(single_name, configs[single_name])
        _require_dataset_set(checks, row, "train", {dataset})
        _require_dataset_set(checks, row, "val", {dataset})
        _require_dataset_set(checks, row, "test", {dataset})
        _require_lean_final_modules(checks, row)
        if base_name not in configs:
            continue
        base = _config_row(base_name, configs[base_name])
        for split in ("train_manifests", "val_manifests", "test_manifests"):
            if sorted(row[split]) != sorted(base[split]):
                checks.append(
                    {
                        "level": "block",
                        "name": "final_single_baseline_split_mismatch",
                        "dataset": dataset,
                        "message": "Final single-dataset method and ConvNeXt baseline must use identical manifests.",
                        "split": split,
                    }
                )


def _require_lean_final_modules(checks: list[dict[str, Any]], row: dict[str, Any]) -> None:
    if row["use_physics_branch"] and not row["use_friction_set"] and row["use_evidence_field"]:
        return
    checks.append(
        {
            "level": "block",
            "name": "final_method_module_mismatch",
            "run": row["run"],
            "message": "Final lean method must keep PhysicsTexture and EvidenceField while removing FrictionSet.",
        }
    )


def _require_dataset_set(
    checks: list[dict[str, Any]],
    row: dict[str, Any],
    split: str,
    expected: set[str],
) -> None:
    key = f"{split}_datasets"
    actual = set(row[key])
    if actual != expected:
        checks.append(
            {
                "level": "block",
                "name": "dataset_set_mismatch",
                "run": row["run"],
                "split": split,
                "message": "Config uses the wrong dataset set for this protocol role.",
                "expected": sorted(expected),
                "actual": sorted(actual),
            }
        )


def _datasets_from_paths(paths: list[str]) -> set[str]:
    datasets = set()
    for raw in paths:
        text = str(raw).replace("\\", "/").lower()
        name = Path(text).name
        if "roadsaw" in name:
            datasets.add("roadsaw")
        elif "roadsc" in name:
            datasets.add("roadsc")
        elif "rscd" in name:
            datasets.add("rscd")
    return datasets


def _norm_path(path: str) -> str:
    return str(path).replace("\\", "/").lower()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Protocol Config Audit",
        "",
        f"Config dir: `{report['config_dir']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Checks",
        "",
    ]
    if report["checks"]:
        lines.append("| Level | Check | Run | Message |")
        lines.append("|---|---|---|---|")
        for item in report["checks"]:
            lines.append(
                "| {level} | {name} | {run} | {message} |".format(
                    level=item.get("level", "-"),
                    name=item.get("name", "-"),
                    run=item.get("run", item.get("dataset", "-")),
                    message=item.get("message", "-"),
                )
            )
    else:
        lines.append("No blocking issues or warnings found.")
    lines.extend(
        [
            "",
            "## Run Manifest Summary",
            "",
            "| Run | Train datasets | Val datasets | Test datasets | Modules |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["runs"]:
        modules = []
        if row["use_physics_branch"]:
            modules.append("PhysicsTexture")
        if row["use_friction_set"]:
            modules.append("FrictionSet")
        if row["use_evidence_field"]:
            modules.append("EvidenceField")
        lines.append(
            "| {run} | {train} | {val} | {test} | {modules} |".format(
                run=row["run"],
                train=", ".join(row["train_datasets"]) or "-",
                val=", ".join(row["val_datasets"]) or "-",
                test=", ".join(row["test_datasets"]) or "-",
                modules=", ".join(modules) or "Global",
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

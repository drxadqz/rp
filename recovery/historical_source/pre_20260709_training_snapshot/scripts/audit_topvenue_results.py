from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/topvenue_v4_evidencefield"))
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    report = build_report(output_dir)
    md = render_markdown(report)
    print(md)

    out_md = args.out_md or output_dir / "topvenue_result_audit.md"
    out_json = args.out_json or output_dir / "topvenue_result_audit.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote: {out_md}")
    print(f"wrote: {out_json}")


def build_report(output_dir: Path) -> dict[str, Any]:
    config_hint = _infer_config_hint(output_dir)
    artifacts = {
        "config": output_dir / "config.json",
        "manifest_stats": output_dir / "manifest_stats_train.json",
        "best_checkpoint": output_dir / "best.pt",
        "last_checkpoint": output_dir / "last.pt",
        "evaluate_test": output_dir / "evaluate_test.json",
        "detailed_test": output_dir / "detailed_test.json",
        "interval_calibration_90": output_dir / "interval_calibration_90.json",
        "bootstrap_metrics": output_dir / "bootstrap_metrics.json",
        "dataset_id_diagnostic": output_dir / "dataset_id_diagnostic.json",
    }
    loaded = {name: _load_json(path) for name, path in artifacts.items() if path.suffix == ".json"}
    existence = {name: path.exists() for name, path in artifacts.items()}

    checks: list[dict[str, Any]] = []
    require_dataset_diagnostic = _requires_dataset_diagnostic(loaded.get("config"))
    _check_required_artifacts(checks, existence, require_dataset_diagnostic=require_dataset_diagnostic)
    _check_protocol(checks, loaded)
    _check_metrics(checks, loaded)
    _check_paper_evidence(checks, output_dir)

    blocking = [item for item in checks if item["level"] == "block"]
    warnings = [item for item in checks if item["level"] == "warn"]
    verdict = "candidate_ready_for_paper_table"
    if blocking:
        verdict = "not_ready"
    elif warnings:
        verdict = "promising_but_needs_more_evidence"

    return {
        "output_dir": str(output_dir),
        "verdict": verdict,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "artifact_exists": existence,
        "metrics": _extract_key_metrics(loaded),
        "checks": checks,
        "next_actions": _next_actions(checks, output_dir, config_hint),
    }


def _check_required_artifacts(
    checks: list[dict[str, Any]],
    exists: dict[str, bool],
    require_dataset_diagnostic: bool,
) -> None:
    required = [
        "config",
        "manifest_stats",
        "best_checkpoint",
        "evaluate_test",
        "detailed_test",
        "interval_calibration_90",
        "bootstrap_metrics",
    ]
    if require_dataset_diagnostic:
        required.append("dataset_id_diagnostic")
    for name in required:
        if not exists.get(name, False):
            checks.append(
                {
                    "level": "block",
                    "name": f"missing_{name}",
                    "message": f"Missing required artifact: {name}",
                }
            )


def _requires_dataset_diagnostic(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return True
    num_domains = _dig(config, ["model", "num_domains"])
    if num_domains is None:
        train_manifests = _dig(config, ["data", "train_manifests"], [])
        return len(train_manifests) > 1
    return int(num_domains) > 1


def _check_protocol(checks: list[dict[str, Any]], loaded: dict[str, Any]) -> None:
    stats = loaded.get("manifest_stats")
    if not stats:
        return
    by_dataset = stats.get("by_dataset", {})
    cfg = loaded.get("config") or {}
    train_manifests = _dig(cfg, ["data", "train_manifests"], [])
    required = _datasets_from_manifest_paths(train_manifests)
    if not required:
        required = set(by_dataset)
    missing = sorted(required - set(by_dataset))
    if missing:
        checks.append(
            {
                "level": "block",
                "name": "missing_dataset_in_train_manifest",
                "message": f"Training manifest missing datasets: {missing}",
            }
        )
    if stats.get("num_samples", 0) < 100000:
        checks.append(
            {
                "level": "warn",
                "name": "small_training_pool",
                "message": "Training pool is small for a paper-scale result.",
                "value": stats.get("num_samples", 0),
            }
        )


def _datasets_from_manifest_paths(paths: Any) -> set[str]:
    if not isinstance(paths, list):
        return set()
    out: set[str] = set()
    for raw in paths:
        text = str(raw).replace("\\", "/").lower()
        name = Path(text).name
        if "roadsaw" in name:
            out.add("roadsaw")
        elif "roadsc" in name:
            out.add("roadsc")
        elif "rscd" in name:
            out.add("rscd")
    return out


def _check_metrics(checks: list[dict[str, Any]], loaded: dict[str, Any]) -> None:
    detailed = loaded.get("detailed_test")
    if detailed:
        risk = _dig(detailed, ["tasks", "risk"], {})
        risk_macro = risk.get("macro_f1")
        worst_risk = _dig(risk, ["by_dataset", "_worst_macro_f1", "value"])
        low_info = _low_friction_info(detailed)
        low_recall = low_info.get("recall")
        coverage = _dig(detailed, ["mu_interval", "coverage"])
        width = _dig(detailed, ["mu_interval", "width_mean"])
        _metric_gate(checks, "risk_macro_f1", risk_macro, minimum=0.70, warn_minimum=0.55)
        _metric_gate(checks, "worst_dataset_risk_macro_f1", worst_risk, minimum=0.55, warn_minimum=0.40)
        if low_info.get("applicable") is False:
            checks.append(
                {
                    "level": "info",
                    "name": "low_friction_recall_not_applicable",
                    "message": "No high/very_high risk positives exist in this split, so low_friction_recall is not a valid gate.",
                    "num_positive": low_info.get("num_positive", 0),
                }
            )
        else:
            _metric_gate(checks, "low_friction_recall", low_recall, minimum=0.85, warn_minimum=0.75)
        _metric_gate(checks, "raw_interval_coverage", coverage, minimum=0.55, warn_minimum=0.40)
        if width is not None and width > 0.90:
            checks.append(
                {
                    "level": "warn",
                    "name": "raw_interval_too_wide",
                    "message": "Raw friction interval is very wide; reviewers may see it as uninformative.",
                    "value": width,
                }
            )

    calib = loaded.get("interval_calibration_90")
    if calib:
        test = calib.get("test_split", {})
        cov = test.get("calibrated_coverage")
        width = test.get("calibrated_width")
        if cov is not None and not (0.88 <= float(cov) <= 0.94):
            checks.append(
                {
                    "level": "block" if float(cov) < 0.86 else "warn",
                    "name": "calibrated_coverage_off_target",
                    "message": "90% conformal interval coverage is outside the expected paper band.",
                    "value": cov,
                }
            )
        if width is not None and float(width) > 0.95:
            checks.append(
                {
                    "level": "warn",
                    "name": "calibrated_interval_too_wide",
                    "message": "Calibrated intervals may be too wide to be useful.",
                    "value": width,
                }
            )
        for name, key in [
            ("dataset_conditional_calibration", "dataset_conditional_test"),
            ("dataset_core_conditional_calibration", "dataset_core_conditional_test"),
            ("risk_conditional_calibration", "risk_conditional_test"),
        ]:
            pooled = calib.get(key, {}).get("_pooled", {})
            if not pooled or pooled.get("calibrated_coverage") is None or pooled.get("calibrated_width") is None:
                checks.append(
                    {
                        "level": "warn",
                        "name": f"missing_{name}",
                        "message": (
                            "Conditional conformal calibration coverage-width diagnostics are missing; "
                            "P3 interval-quality claims need dataset/core/risk conditional evidence."
                        ),
                    }
                )

    diag = loaded.get("dataset_id_diagnostic")
    if diag:
        bal = diag.get("overall_dataset_id_balanced_accuracy")
        if bal is not None and float(bal) > 0.80:
            checks.append(
                {
                    "level": "warn",
                    "name": "high_dataset_id_predictability",
                    "message": "Features still encode dataset identity strongly; domain shortcut claim needs caution.",
                    "value": bal,
                }
            )


def _check_paper_evidence(checks: list[dict[str, Any]], output_dir: Path) -> None:
    scope = _audit_scope(output_dir)
    if scope in {"single_dataset", "single_dataset_baseline", "lodo"}:
        return
    sibling_outputs = output_dir.parent
    evidence_roots = [sibling_outputs]
    external_root = Path("D:/NMI_SPWFM_datasets/friction_affordance_outputs")
    if external_root.exists() or output_dir.as_posix().lower().startswith(external_root.as_posix().lower()):
        evidence_roots.append(external_root)
    lodo_outputs = [
        "lodo_rscd*",
        "lodo_roadsaw*",
        "lodo_roadsc*",
    ]
    if not all(_has_completed_output(evidence_roots, pattern) for pattern in lodo_outputs):
        checks.append(
            {
                "level": "block",
                "name": "missing_lodo_evidence",
                "message": "Missing leave-one-dataset-out evidence, required for the top-venue generalization claim.",
            }
        )
    ablation_patterns = {
        "global_only": ["v0_global_only", "*global_only*"],
        "physics_texture": ["v1_physics_texture", "*physics_texture*"],
        "friction_set": ["v2_friction_set", "*friction_set*"],
        "dg_losses": ["v3_dg_losses", "*dg*"],
        "evidence_field_aux": ["v4_evidence_aux", "*evidence*"],
        "full_faf": ["v5_full_faf"],
    }
    missing_ablation = [
        name
        for name, patterns in ablation_patterns.items()
        if not _has_any_completed_output(evidence_roots, patterns)
    ]
    if missing_ablation:
        checks.append(
            {
                "level": "block",
                "name": "insufficient_ablation_suite",
                "message": "Need the complete core P0 ablation suite v0-v5 for a defensible paper table.",
                "missing": missing_ablation,
            }
        )
    uses_evidence = _output_uses_evidence(output_dir)
    if uses_evidence and not (output_dir / "evidence_maps").exists():
        checks.append(
            {
                "level": "warn",
                "name": "missing_qualitative_evidence_maps",
                "message": "Evidence maps are useful for interpretability figures and shortcut analysis.",
            }
        )
    if uses_evidence and not (output_dir / "evidence_field_audit.json").exists():
        checks.append(
            {
                "level": "warn",
                "name": "missing_evidence_field_audit",
                "message": "EvidenceField models should include quantitative attention-road diagnostics.",
            }
        )


def _audit_scope(output_dir: Path) -> str:
    name = output_dir.name.lower()
    if name.startswith("baseline_single_"):
        return "single_dataset_baseline"
    if name.startswith("single_"):
        return "single_dataset"
    if name.startswith("lodo_"):
        return "lodo"
    return "multi_dataset_method"


def _output_uses_evidence(output_dir: Path) -> bool:
    config = _load_json(output_dir / "config.json")
    if isinstance(config, dict):
        return bool(_dig(config, ["model", "use_evidence_field"], False))
    name = output_dir.name.lower()
    return "evidence" in name or "faf" in name


def _has_completed_output(roots: list[Path], pattern: str) -> bool:
    required = ["best.pt", "detailed_test.json", "interval_calibration_90.json", "bootstrap_metrics.json"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob(pattern):
            if path.is_dir() and all((path / name).exists() for name in required):
                return True
    return False


def _has_any_completed_output(roots: list[Path], patterns: list[str]) -> bool:
    return any(_has_completed_output(roots, pattern) for pattern in patterns)


def _extract_key_metrics(loaded: dict[str, Any]) -> dict[str, Any]:
    detailed = loaded.get("detailed_test", {})
    calib = loaded.get("interval_calibration_90", {})
    bootstrap = loaded.get("bootstrap_metrics", {})
    diag = loaded.get("dataset_id_diagnostic") or {}
    low_info = _low_friction_info(detailed)
    low_applicable = low_info.get("applicable")
    low_recall = None if low_applicable is False else low_info.get("recall")
    low_recall_ci = None if low_applicable is False else _ci(_dig(bootstrap, ["low_friction_detection", "recall"]))
    return {
        "risk_macro_f1": _dig(detailed, ["tasks", "risk", "macro_f1"]),
        "risk_macro_f1_ci": _ci(_dig(bootstrap, ["classification", "risk", "macro_f1"])),
        "risk_worst_dataset_macro_f1": _dig(
            detailed, ["tasks", "risk", "by_dataset", "_worst_macro_f1", "value"]
        ),
        "risk_worst_dataset_macro_f1_ci": _ci(
            _dig(bootstrap, ["classification", "risk", "worst_dataset_macro_f1"])
        ),
        "low_friction_recall": low_recall,
        "low_friction_recall_ci": low_recall_ci,
        "low_friction_recall_applicable": low_applicable,
        "low_friction_positive_count": low_info.get("num_positive"),
        "raw_interval_coverage": _dig(detailed, ["mu_interval", "coverage"]),
        "raw_interval_width": _dig(detailed, ["mu_interval", "width_mean"]),
        "calibrated_test_coverage": _dig(calib, ["test_split", "calibrated_coverage"]),
        "calibrated_test_coverage_ci": _ci(_dig(bootstrap, ["mu_interval", "calibrated_coverage"])),
        "calibrated_test_width": _dig(calib, ["test_split", "calibrated_width"]),
        "dataset_id_balanced_accuracy": diag.get("overall_dataset_id_balanced_accuracy"),
    }


def _ci(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    if item.get("ci_low") is None or item.get("ci_high") is None:
        return None
    return f"[{float(item['ci_low']):.4f}, {float(item['ci_high']):.4f}]"


def _next_actions(checks: list[dict[str, Any]], output_dir: Path, config_hint: str) -> list[str]:
    actions = []
    names = {item["name"] for item in checks}
    if "missing_evaluate_test" in names:
        actions.append(
            f"python scripts/evaluate.py --config {config_hint} "
            f"--checkpoint {output_dir / 'best.pt'} --split test"
        )
    if "missing_detailed_test" in names:
        actions.append(
            f"python scripts/evaluate_detailed.py --config {config_hint} "
            f"--checkpoint {output_dir / 'best.pt'} --split test --out {output_dir / 'detailed_test.json'}"
        )
    if "missing_interval_calibration_90" in names:
        actions.append(
            f"python scripts/calibrate_intervals.py --config {config_hint} "
            f"--checkpoint {output_dir / 'best.pt'} --target-coverage 0.90 "
            f"--out {output_dir / 'interval_calibration_90.json'}"
        )
    if "missing_bootstrap_metrics" in names:
        actions.append(
            f"python scripts/bootstrap_metrics.py --config {config_hint} "
            f"--checkpoint {output_dir / 'best.pt'} --split test --target-coverage 0.90 "
            f"--out-json {output_dir / 'bootstrap_metrics.json'} --out-md {output_dir / 'bootstrap_metrics.md'}"
        )
    if "missing_dataset_id_diagnostic" in names:
        actions.append(
            f"python scripts/dataset_id_diagnostic.py --config {config_hint} "
            f"--checkpoint {output_dir / 'best.pt'} --max-samples 5000"
        )
    if "missing_lodo_evidence" in names:
        actions.append("Run leave-one-dataset-out configs and evaluate each held-out dataset.")
    if "insufficient_ablation_suite" in names:
        actions.append("Run ablation suite: global-only, FrictionSet, DG/heterogeneous, full evidence-field.")
    if not actions:
        actions.append("Prepare paper tables, qualitative figures, and method limitations.")
    return actions


def _infer_config_hint(output_dir: Path) -> str:
    protocol_config = Path("configs/experiments/paper_protocol") / f"{output_dir.name}.yaml"
    if protocol_config.exists():
        return str(protocol_config)
    name = output_dir.name.lower()
    if "rtx3050_stable" in name:
        return "configs/experiments/topvenue_v4_evidencefield_rtx3050_stable.yaml"
    return "configs/experiments/topvenue_v4_evidencefield.yaml"


def _metric_gate(
    checks: list[dict[str, Any]],
    name: str,
    value: Any,
    minimum: float,
    warn_minimum: float,
) -> None:
    if value is None:
        return
    value = float(value)
    if value < warn_minimum:
        checks.append(
            {
                "level": "block",
                "name": f"weak_{name}",
                "message": f"{name} is below the minimum credible level.",
                "value": value,
                "minimum": minimum,
            }
        )
    elif value < minimum:
        checks.append(
            {
                "level": "warn",
                "name": f"borderline_{name}",
                "message": f"{name} is promising but below the target paper threshold.",
                "value": value,
                "target": minimum,
            }
        )


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dig(items: Any, keys: list[str], default: Any = None) -> Any:
    cur = items
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _low_friction_info(detailed: dict[str, Any]) -> dict[str, Any]:
    info = dict(detailed.get("low_friction_detection") or {})
    if info.get("applicable") is not None and info.get("num_positive") is not None:
        return info
    num_positive = _low_friction_positive_count_from_confusion(detailed)
    if num_positive is None:
        return info
    info["num_positive"] = int(num_positive)
    info["applicable"] = int(num_positive) > 0
    return info


def _low_friction_positive_count_from_confusion(detailed: dict[str, Any]) -> int | None:
    matrix = _dig(detailed, ["tasks", "risk", "confusion_matrix"])
    labels = _dig(detailed, ["tasks", "risk", "confusion_matrix_labels"])
    if not isinstance(matrix, list):
        return None
    if not isinstance(labels, list):
        labels = ["very_low", "low", "medium", "high", "very_high"]
    try:
        high_idx = labels.index("high")
    except ValueError:
        return None
    total = 0
    for idx, row in enumerate(matrix):
        if idx < high_idx:
            continue
        if isinstance(row, list):
            total += sum(int(value) for value in row)
    return total


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Top-Venue Result Audit",
        "",
        f"Output directory: `{report['output_dir']}`",
        f"Verdict: **{report['verdict']}**",
        "",
        "## Key Metrics",
        "",
    ]
    for key, value in report["metrics"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Checks", ""])
    for item in report["checks"]:
        suffix = ""
        if "value" in item:
            suffix = f" value={item['value']}"
        lines.append(f"- **{item['level']}** `{item['name']}`: {item['message']}{suffix}")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

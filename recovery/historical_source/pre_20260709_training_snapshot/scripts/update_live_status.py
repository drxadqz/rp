from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_OUT = Path("reports/p0_live_status.md")

P0_ORDER = [
    "v0_global_only",
    "v1_physics_texture",
    "v2_friction_set",
    "v3_dg_losses",
    "v4_evidence_aux",
    "v5_full_faf",
    "lodo_roadsaw_full_faf",
    "lodo_rscd_full_faf",
    "lodo_roadsc_full_faf",
]

FOLLOWUP_ORDER = [
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
    "single_roadsaw_full_faf",
    "single_rscd_full_faf",
    "single_roadsc_full_faf",
    "baseline_single_roadsaw_global_convnext",
    "baseline_single_rscd_global_convnext",
    "baseline_single_roadsc_global_convnext",
    "final_lodo_roadsaw_lean_road_roi_safety",
    "final_lodo_rscd_lean_road_roi_safety",
    "final_lodo_roadsc_lean_road_roi_safety",
    "final_single_roadsaw_lean_road_roi_safety",
    "final_single_rscd_lean_road_roi_safety",
    "final_single_roadsc_lean_road_roi_safety",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    current = first_active(args.root, P0_ORDER + FOLLOWUP_ORDER)
    lines = ["# P0 Live Status", "", f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.extend(render_current(args.root, current))
    lines.extend(render_open_issues(args.root, current))
    lines.extend(render_next_actions(args.root))
    lines.extend(render_pending(args.root, "Pending P0 Runs", P0_ORDER))
    lines.extend(render_pending(args.root, "Pending Follow-Up Runs", FOLLOWUP_ORDER))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def first_active(root: Path, names: list[str]) -> str | None:
    for name in names:
        if status_for(root / name) == "running/partial":
            return name
    for name in names:
        status = status_for(root / name)
        if status in {"partial_ci_missing", "missing"}:
            return name
    return None


def render_current(root: Path, name: str | None) -> list[str]:
    lines = ["## Current Run", ""]
    if name is None:
        return lines + ["- No pending protocol run detected.", ""]
    run_dir = root / name
    state = load_state(run_dir)
    metrics = state.get("val_metrics", {}) if state else {}
    lines.extend(
        [
            f"- Run: `{name}`",
            f"- Status: {status_for(run_dir)}",
            f"- Latest checkpoint epoch: {state.get('epoch') if state else '-'} / {state.get('epochs') if state else '-'}",
            f"- Best validation loss so far: {fmt_raw(state.get('best_metric') if state else None)}",
            f"- Stale epochs: {state.get('stale_epochs') if state else '-'}",
            f"- Latest validation loss: {fmt_raw(metrics.get('loss'))}",
            f"- Latest validation friction accuracy: {fmt_raw(metrics.get('acc_friction'))}",
            f"- Latest validation risk accuracy: {fmt_raw(metrics.get('acc_risk'))}",
            f"- Latest raw interval coverage: {fmt_raw(metrics.get('mu_interval_coverage'))}",
            f"- Latest raw interval width: {fmt_raw(metrics.get('mu_interval_width'))}",
            "",
        ]
    )
    return lines


def render_pending(root: Path, title: str, names: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    pending = [name for name in names if status_for(root / name) != "complete"]
    if not pending:
        lines.append("- None.")
    else:
        for name in pending:
            lines.append(f"- `{name}`: {status_for(root / name)}")
    lines.append("")
    return lines


def render_open_issues(root: Path, current: str | None) -> list[str]:
    issues: list[str] = []
    if any(status_for(root / name) != "complete" for name in P0_ORDER[:6]):
        issues.append("P0 ablation table is incomplete, so module-level claims are not yet defensible.")
    missing_lodo = [name for name in P0_ORDER[6:] if status_for(root / name) != "complete"]
    if missing_lodo:
        if (root / "lodo_roadsaw_full_faf" / "topvenue_result_audit.json").exists():
            issues.append(
                "LODO evidence is still incomplete; held-out RoadSaW is complete and weak, while "
                + ", ".join(f"`{name}`" for name in missing_lodo)
                + " remains open."
            )
        else:
            issues.append("LODO evidence is missing, especially held-out RoadSaW generalization.")
    if current:
        state = load_state(root / current)
        metrics = state.get("val_metrics", {}) if state else {}
        cov = metrics.get("mu_interval_coverage")
        if cov is not None and float(cov) < 0.70:
            issues.append(
                "Raw validation interval coverage is low; final judgment needs conformal calibration and width."
            )
    free_gb = disk_free_gb(root.anchor or str(root))
    if free_gb is not None and free_gb < 15.0:
        issues.append(f"Output disk free space is tight ({free_gb:.2f} GB); keep checkpoint slimming enabled.")
    if not issues:
        issues.append("No immediate protocol blocker beyond waiting for queued experiments.")

    lines = ["## Open Issues", ""]
    for issue in issues:
        lines.append(f"- {issue}")
    lines.append("")
    return lines


def render_next_actions(root: Path) -> list[str]:
    actions: list[str] = []
    if status_for(root / "v0_global_only") != "complete":
        actions.append("Let `v0_global_only` finish train, test evaluation, calibration, bootstrap, shortcut diagnostic, and audit.")
    elif any(status_for(root / name) != "complete" for name in P0_ORDER[1:6]):
        missing = [name for name in P0_ORDER[1:6] if status_for(root / name) != "complete"]
        actions.append(
            "Finish remaining P0 ablations: "
            + ", ".join(f"`{name}`" for name in missing)
            + ". Compare adjacent deltas before keeping/removing modules."
        )
    elif any(status_for(root / name) != "complete" for name in P0_ORDER[6:]):
        missing = [name for name in P0_ORDER[6:] if status_for(root / name) != "complete"]
        if (root / "lodo_roadsaw_full_faf" / "topvenue_result_audit.json").exists():
            actions.append(
                "Finish remaining LODO row(s): "
                + ", ".join(f"`{name}`" for name in missing)
                + ". Use the completed held-out RoadSaW failure as the key robustness stress-test evidence."
            )
        else:
            actions.append("Run LODO, with held-out RoadSaW as the key generalization gate.")
    else:
        actions.append("Re-run postprocess and top-venue audit, then decide P1 robustness candidates from evidence.")
    actions.append("Refresh `reports/paper_protocol_summary/paper_protocol_summary.md` only from fresh artifacts.")
    lines = ["## Next Actions", ""]
    for action in actions:
        lines.append(f"- {action}")
    lines.append("")
    return lines


def disk_free_gb(path: str) -> float | None:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except OSError:
        return None


def status_for(run_dir: Path) -> str:
    missing = missing_completion_artifacts(run_dir)
    if not missing:
        return "complete"
    core_complete = all(
        (run_dir / name).exists()
        for name in ["best.pt", "detailed_test.json", "interval_calibration_90.json"]
    )
    if core_complete:
        return "partial_ci_missing"
    if (run_dir / "last.pt").exists() or (run_dir / "best.pt").exists():
        return "running/partial"
    return "missing"


def missing_completion_artifacts(run_dir: Path) -> list[str]:
    required = [
        "best.pt",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "topvenue_result_audit.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    name = run_dir.name
    if not (name.startswith("single_") or name.startswith("baseline_single_")):
        if not (run_dir / "dataset_id_diagnostic.json").exists():
            missing.append("dataset_id_diagnostic.json")
    config = load_json(run_dir / "config.json")
    if isinstance(config, dict) and config.get("model", {}).get("use_evidence_field"):
        if not (run_dir / "evidence_maps").exists():
            missing.append("evidence_maps")
        for artifact in ["evidence_field_audit.json", "evidence_field_audit.md"]:
            if not (run_dir / artifact).exists():
                missing.append(artifact)
    return missing


def load_state(run_dir: Path) -> dict[str, Any] | None:
    state_path = run_dir / "training_state.json"
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        return None
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu")
        return {
            "epoch": ckpt.get("epoch"),
            "epochs": dig(ckpt, ["config", "optim", "epochs"]),
            "val_metrics": ckpt.get("metrics", {}),
        }
    except Exception:
        return None


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def dig(items: Any, keys: list[str]) -> Any:
    cur = items
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return "-"
        cur = cur[key]
    return cur


def fmt_raw(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()

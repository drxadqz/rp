from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")

ROWS = [
    "v0_global_only",
    "v1_physics_texture",
    "v2_friction_set",
    "v3_dg_losses",
    "v4_evidence_aux",
    "v5_full_faf",
    "lodo_roadsaw_full_faf",
    "lodo_rscd_full_faf",
    "lodo_roadsc_full_faf",
    "single_roadsaw_full_faf",
    "single_rscd_full_faf",
    "single_roadsc_full_faf",
    "baseline_single_roadsaw_global_convnext",
    "baseline_single_rscd_global_convnext",
    "baseline_single_roadsc_global_convnext",
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
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--load-checkpoints", action="store_true")
    args = parser.parse_args()

    rows = [
        inspect_run(args.root / name, load_checkpoints=args.load_checkpoints, log_dir=args.log_dir)
        for name in ROWS
    ]
    print("| run | status | epoch | active progress | last update | artifacts |")
    print("|---|---|---:|---|---|---|")
    for row in rows:
        print(
            "| {name} | {status} | {epoch} | {active} | {last_update} | {artifacts} |".format(
                name=row["name"],
                status=row["status"],
                epoch=row.get("epoch", "-"),
                active=_format_active_progress(row),
                last_update=row.get("last_update", "-"),
                artifacts=", ".join(row["artifacts"]) if row["artifacts"] else "-",
            )
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def inspect_run(path: Path, load_checkpoints: bool = False, log_dir: Path | None = None) -> dict[str, Any]:
    artifacts = []
    for name in [
        "config.json",
        "train.lock",
        "last.pt",
        "best.pt",
        "best_safety.pt",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "dataset_id_diagnostic.json",
        "topvenue_result_audit.json",
    ]:
        if (path / name).exists():
            artifacts.append(name)
    state = load_json(path / "training_state.json")
    ckpt_state = load_checkpoint_state(path / "last.pt") if load_checkpoints and state is None else None
    core_complete = all(
        (path / name).exists()
        for name in ["best.pt", "detailed_test.json", "interval_calibration_90.json"]
    )
    active = active_log_progress(path.name, log_dir) if log_dir is not None else {}
    complete = not missing_completion_artifacts(path)
    if complete:
        status = "complete"
        active = {}
    elif core_complete:
        status = "partial_ci_missing"
    elif artifacts or active:
        status = "running_or_partial"
    else:
        status = "missing"
    last_update = latest_mtime(path)
    return {
        "name": path.name,
        "path": str(path),
        "status": status,
        "epoch": _state_value(state, ckpt_state, "epoch"),
        "epochs": _state_value(state, ckpt_state, "epochs"),
        "best_metric": _state_value(state, ckpt_state, "best_metric"),
        "stale_epochs": _state_value(state, ckpt_state, "stale_epochs"),
        "last_update": last_update,
        "artifacts": artifacts,
        **active,
    }


def missing_completion_artifacts(path: Path) -> list[str]:
    required = [
        "best.pt",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "topvenue_result_audit.json",
    ]
    missing = [name for name in required if not (path / name).exists()]
    if not (
        path.name.startswith("single_")
        or path.name.startswith("baseline_single_")
        or path.name.startswith("final_single_")
    ):
        if not (path / "dataset_id_diagnostic.json").exists():
            missing.append("dataset_id_diagnostic.json")
    config = load_json(path / "config.json")
    if isinstance(config, dict) and config.get("model", {}).get("use_evidence_field"):
        if not (path / "evidence_maps").exists():
            missing.append("evidence_maps")
        for name in ["evidence_field_audit.json", "evidence_field_audit.md"]:
            if not (path / name).exists():
                missing.append(name)
    return missing


def active_log_progress(run_name: str, log_dir: Path) -> dict[str, Any]:
    if not log_dir.exists():
        return {}
    candidates = sorted(
        log_dir.glob(f"*{run_name}*.out.log"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}
    log_path = candidates[0]
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    epoch_matches = list(re.finditer(r"Epoch\s+(\d+)\s*/\s*(\d+)", text))
    step_matches = list(re.finditer(r"train step\s+(\d+)\s*/\s*(\d+)", text))
    out: dict[str, Any] = {
        "active_log": str(log_path),
        "active_log_mtime": latest_file_mtime(log_path),
    }
    if epoch_matches:
        match = epoch_matches[-1]
        out["active_epoch"] = int(match.group(1))
        out["active_epochs"] = int(match.group(2))
        scoped_text = text[match.end() :]
        step_matches = list(re.finditer(r"train step\s+(\d+)\s*/\s*(\d+)", scoped_text))
    if step_matches:
        match = step_matches[-1]
        out["active_step"] = int(match.group(1))
        out["active_steps"] = int(match.group(2))
    return out


def _format_active_progress(row: dict[str, Any]) -> str:
    epoch = row.get("active_epoch")
    epochs = row.get("active_epochs")
    step = row.get("active_step")
    steps = row.get("active_steps")
    if epoch is None:
        return "-"
    if step is not None and steps is not None:
        return f"epoch {epoch}/{epochs}, step {step}/{steps}"
    return f"epoch {epoch}/{epochs}"


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_checkpoint_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("config", {})
        return {
            "epoch": ckpt.get("epoch"),
            "epochs": _dig(cfg, ["optim", "epochs"]),
            "val_loss": _dig(ckpt, ["metrics", "loss"]),
        }
    except Exception:
        return None


def _state_value(state: Any, ckpt_state: Any, key: str) -> Any:
    if isinstance(state, dict) and state.get(key) is not None:
        return state.get(key)
    if isinstance(ckpt_state, dict):
        return ckpt_state.get(key)
    return None


def _dig(items: Any, keys: list[str]) -> Any:
    cur = items
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def latest_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    latest = max((item.stat().st_mtime for item in path.rglob("*") if item.is_file()), default=None)
    if latest is None:
        return None
    from datetime import datetime

    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S")


def latest_file_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    from datetime import datetime

    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from friction_affordance.c3_experiment import build_class_map, load_config  # noqa: E402
from train_coupled_factor_backbone import build_s136_model  # noqa: E402


FULL_TRAIN_SAMPLES = 958_941
FULL_VAL_SAMPLES = 19_860
FULL_TEST_SAMPLES = 49_500


@dataclass
class Check:
    name: str
    severity: str
    passed: bool
    message: str
    details: dict[str, Any]


def _add(checks: list[Check], name: str, severity: str, passed: bool, message: str, **details: Any) -> None:
    checks.append(Check(name=name, severity=severity, passed=bool(passed), message=message, details=details))


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _same_values(
    checks: list[Check],
    name: str,
    left: dict[str, Any],
    right: dict[str, Any],
    keys: list[str],
    *,
    severity: str = "block",
) -> None:
    diffs = {}
    for key in keys:
        lv = _get(left, key)
        rv = _get(right, key)
        if lv != rv:
            diffs[key] = {"screen": lv, "control": rv}
    _add(
        checks,
        name,
        severity,
        not diffs,
        "S137 learned screen and off-control must share the same budget except the isolated mechanism.",
        diffs=diffs,
    )


def _is_empty(value: Any) -> bool:
    return value is None or value == ""


def _count_manifest_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _param_count(cfg: dict[str, Any]) -> int:
    manifests = [Path(cfg["data"]["train_manifest"]), Path(cfg["data"]["val_manifest"]), Path(cfg["data"]["test_manifest"])]
    class_to_idx = build_class_map(manifests)
    model = build_s136_model(cfg, class_to_idx)
    return int(sum(param.numel() for param in model.parameters()))


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# S137 Queue Readiness Audit",
        "",
        f"- Overall: `{payload['overall']}`",
        f"- Screen config: `{payload['paths']['screen_config']}`",
        f"- Control config: `{payload['paths']['control_config']}`",
        f"- Full config: `{payload['paths']['full_config']}`",
        f"- Watcher: `{payload['paths']['watcher_script']}`",
        "",
        "## Checks",
        "",
        "| Severity | Pass | Check | Message |",
        "|---|---:|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(
            f"| {item['severity']} | {item['passed']} | `{item['name']}` | {item['message']} |"
        )
    lines.extend(
        [
            "",
            "## Parameter Counts",
            "",
            "| Variant | Params |",
            "|---|---:|",
        ]
    )
    for name, value in payload["parameter_counts"].items():
        lines.append(f"| {name} | {value} |")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in payload["notes"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the S137 queue before GPU execution.")
    parser.add_argument(
        "--screen-config",
        type=Path,
        default=Path("configs/c3_farnet/c3_farnet_s137_concrete_roughness_scalespace_screen_20260715.yaml"),
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        default=Path("configs/c3_farnet/c3_farnet_s137_concrete_roughness_scalespace_control_20260715.yaml"),
    )
    parser.add_argument(
        "--full-config",
        type=Path,
        default=Path("configs/c3_farnet/c3_farnet_s137_concrete_roughness_scalespace_full_20260715.yaml"),
    )
    parser.add_argument(
        "--smoke-metrics",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\s137_concrete_roughness_scalespace_smoke_20260715\test_metrics.json"),
    )
    parser.add_argument(
        "--control-smoke-metrics",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\s137_concrete_roughness_scalespace_control_smoke_20260715\test_metrics.json"),
    )
    parser.add_argument(
        "--watcher-script",
        type=Path,
        default=Path("scripts/run_s137_after_current_queue_if_needed.ps1"),
    )
    parser.add_argument(
        "--live-status-script",
        type=Path,
        default=Path("scripts/write_rscd_live_route_status.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\S137_queue_readiness_20260715"),
    )
    args = parser.parse_args()

    checks: list[Check] = []
    for name, path in [
        ("screen_config_exists", args.screen_config),
        ("control_config_exists", args.control_config),
        ("full_config_exists", args.full_config),
        ("watcher_script_exists", args.watcher_script),
    ]:
        _add(checks, name, "block", path.exists(), f"Required S137 artifact must exist: {path}", path=str(path))

    screen = load_config(args.screen_config)
    control = load_config(args.control_config)
    full = load_config(args.full_config)

    manifest_keys = ["data.train_manifest", "data.val_manifest", "data.test_manifest", "data.image_size", "data.train_resize_mode", "data.eval_resize_mode"]
    _same_values(checks, "screen_control_same_data", screen, control, manifest_keys)

    for key in ["data.train_manifest", "data.val_manifest", "data.test_manifest"]:
        path = Path(str(_get(screen, key)))
        _add(checks, f"manifest_exists_{key.split('.')[-1]}", "block", path.exists(), f"S137 manifest path must exist: {path}", path=str(path))

    same_budget_keys = [
        "seed",
        "model.stem_dim",
        "model.stage_dims",
        "model.dropout",
        "model.coupling_gate_mode",
        "train.batch_size",
        "train.grad_accum_steps",
        "train.num_workers",
        "train.prefetch_factor",
        "train.lr",
        "train.weight_decay",
        "train.epochs",
        "train.amp",
        "train.augmentation",
        "train.balanced_sampling",
        "train.max_train_samples_per_class",
        "train.samples_per_epoch",
        "train.grad_clip_norm",
        "eval.batch_size",
        "eval.max_val_samples_per_class",
        "eval.max_test_samples_per_class",
        "loss.class_weight",
        "loss.factor_weight",
        "loss.factor_axis_weights",
        "loss.coupling_gate_weight",
    ]
    _same_values(checks, "screen_control_same_training_budget", screen, control, same_budget_keys)

    _add(
        checks,
        "screen_learned_scale_space_enabled",
        "block",
        bool(_get(screen, "model.use_concrete_roughness_scale_space")) and str(_get(screen, "model.concrete_roughness_scale_space_mode")).lower() == "learned",
        "S137 screen must enable the learned concrete roughness scale-space route.",
        mode=_get(screen, "model.concrete_roughness_scale_space_mode"),
        enabled=_get(screen, "model.use_concrete_roughness_scale_space"),
    )
    _add(
        checks,
        "control_isolated_mechanism_off",
        "block",
        bool(_get(control, "model.use_concrete_roughness_scale_space")) and str(_get(control, "model.concrete_roughness_scale_space_mode")).lower() in {"off", "false", "none", "disabled"},
        "S137 control must keep the code path but disable the learned mechanism effect.",
        mode=_get(control, "model.concrete_roughness_scale_space_mode"),
        enabled=_get(control, "model.use_concrete_roughness_scale_space"),
    )
    _add(
        checks,
        "route_loss_isolated",
        "block",
        float(_get(screen, "loss.concrete_roughness_route_weight", 0.0)) > 0.0
        and float(_get(control, "loss.concrete_roughness_route_weight", 0.0)) == 0.0,
        "Only S137 learned screen should supervise the early concrete roughness route.",
        screen_weight=_get(screen, "loss.concrete_roughness_route_weight"),
        control_weight=_get(control, "loss.concrete_roughness_route_weight"),
    )
    _add(
        checks,
        "screen_protocol_cap250",
        "block",
        int(_get(screen, "eval.max_val_samples_per_class", -1)) == 250
        and int(_get(screen, "eval.max_test_samples_per_class", -1)) == 250
        and int(_get(screen, "train.max_train_samples_per_class", -1)) == 1000,
        "S137 screen must use the same cap1000/cap250 screening protocol as S96/S136.",
        train_cap=_get(screen, "train.max_train_samples_per_class"),
        val_cap=_get(screen, "eval.max_val_samples_per_class"),
        test_cap=_get(screen, "eval.max_test_samples_per_class"),
    )
    _add(
        checks,
        "full_protocol_uncapped",
        "block",
        _is_empty(_get(full, "train.max_train_samples_per_class"))
        and int(_get(full, "train.samples_per_epoch", -1)) == 0
        and _is_empty(_get(full, "eval.max_val_samples_per_class"))
        and _is_empty(_get(full, "eval.max_test_samples_per_class")),
        "S137 full must use uncapped train/val/test manifests for fair final evidence.",
        train_cap=_get(full, "train.max_train_samples_per_class"),
        samples_per_epoch=_get(full, "train.samples_per_epoch"),
        val_cap=_get(full, "eval.max_val_samples_per_class"),
        test_cap=_get(full, "eval.max_test_samples_per_class"),
    )
    full_manifest_paths = {
        "train": Path(str(_get(full, "data.train_manifest"))),
        "val": Path(str(_get(full, "data.val_manifest"))),
        "test": Path(str(_get(full, "data.test_manifest"))),
    }
    full_manifest_counts = {
        split: _count_manifest_rows(path)
        for split, path in full_manifest_paths.items()
    }
    _add(
        checks,
        "full_manifest_rows_complete",
        "block",
        full_manifest_counts["train"] == FULL_TRAIN_SAMPLES
        and full_manifest_counts["val"] == FULL_VAL_SAMPLES
        and full_manifest_counts["test"] == FULL_TEST_SAMPLES,
        "S137 full manifest row counts must match the complete RSCD train/val/test protocol.",
        expected={
            "train": FULL_TRAIN_SAMPLES,
            "val": FULL_VAL_SAMPLES,
            "test": FULL_TEST_SAMPLES,
        },
        observed=full_manifest_counts,
        paths={split: str(path) for split, path in full_manifest_paths.items()},
    )
    _add(
        checks,
        "full_resumes_from_screen_best",
        "warn",
        str(_get(full, "train.resume_from", "")).endswith("s137_concrete_roughness_scalespace_screen_20260715/best_checkpoint.pth"),
        "S137 full should resume from the promoted S137 screen checkpoint.",
        resume_from=_get(full, "train.resume_from"),
    )

    param_counts: dict[str, int] = {}
    try:
        param_counts["screen"] = _param_count(screen)
        param_counts["control"] = _param_count(control)
        _add(
            checks,
            "screen_control_same_param_count",
            "block",
            param_counts["screen"] == param_counts["control"],
            "S137 learned screen and off-control must have identical parameter count.",
            screen_params=param_counts["screen"],
            control_params=param_counts["control"],
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        _add(checks, "param_count_build", "block", False, "S137 model construction failed.", error=repr(exc))

    for name, path in [("learned_smoke_metrics", args.smoke_metrics), ("control_smoke_metrics", args.control_smoke_metrics)]:
        _add(checks, name, "block", path.exists(), f"S137 smoke metrics should exist before queue promotion: {path}", path=str(path))

    watcher_text = args.watcher_script.read_text(encoding="utf-8", errors="ignore") if args.watcher_script.exists() else ""
    watcher_needles = [
        "S137Config",
        "S137ControlConfig",
        "S137FullConfig",
        "S137_learned_scale_space_vs_off_control",
        "S137_screen_promotion_audit_vs_S96",
        "S137_full_promotion_audit_vs_S7",
        "--require-sota",
        "Any-FullSotaPass",
        "Get-ActiveRscdTraining",
    ]
    missing_watcher = [needle for needle in watcher_needles if needle not in watcher_text]
    _add(
        checks,
        "watcher_has_fair_gates",
        "block",
        not missing_watcher,
        "S137 watcher must include upstream SOTA exit, same-budget control, screen gate, and full SOTA gate.",
        missing=missing_watcher,
    )

    live_text = args.live_status_script.read_text(encoding="utf-8", errors="ignore") if args.live_status_script.exists() else ""
    missing_live = [needle for needle in ["DEFAULT_S137_DIR", "DEFAULT_S137_CONTROL_DIR", "DEFAULT_S137_FULL_DIR", "S137_concrete_roughness_screen"] if needle not in live_text]
    _add(
        checks,
        "live_status_tracks_s137",
        "warn",
        not missing_live,
        "Live route status should track S137 learned screen, off-control, and full rows.",
        missing=missing_live,
    )

    block_failures = [check for check in checks if check.severity == "block" and not check.passed]
    overall = "pass" if not block_failures else "fail"
    notes = [
        "S137 is a single task-adapted route targeting concrete roughness boundaries, not a generic late add-on.",
        "The off-control keeps the same code path and budget while disabling the learned route effect.",
        "The watcher may wait for a long time while S133c/S135/S136/S136d are active; that is expected.",
    ]
    payload = {
        "overall": overall,
        "checks": [asdict(check) for check in checks],
        "parameter_counts": param_counts,
        "full_manifest_counts": full_manifest_counts,
        "notes": notes,
        "paths": {
            "screen_config": str(args.screen_config),
            "control_config": str(args.control_config),
            "full_config": str(args.full_config),
            "watcher_script": str(args.watcher_script),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "s137_queue_readiness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "s137_queue_readiness.md")
    print(json.dumps({"overall": overall, "report": str(args.output_dir / "s137_queue_readiness.md")}, ensure_ascii=False))
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

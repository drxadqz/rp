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


def _is_empty(value: Any) -> bool:
    return value is None or value == ""


def _count_manifest_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(child, child_prefix))
        return out
    return {prefix: value}


def _read_summary(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        return None
    return summary


def _param_count(cfg: dict[str, Any]) -> int:
    manifests = [
        Path(str(cfg["data"]["train_manifest"])),
        Path(str(cfg["data"]["val_manifest"])),
        Path(str(cfg["data"]["test_manifest"])),
    ]
    class_to_idx = build_class_map(manifests)
    model = build_s136_model(cfg, class_to_idx)
    return int(sum(param.numel() for param in model.parameters()))


def _same_values(
    checks: list[Check],
    name: str,
    left: dict[str, Any],
    right: dict[str, Any],
    keys: list[str],
    *,
    severity: str = "block",
    message: str,
) -> None:
    diffs = {}
    for key in keys:
        left_value = _get(left, key)
        right_value = _get(right, key)
        if left_value != right_value:
            diffs[key] = {"screen": left_value, "control": right_value}
    _add(checks, name, severity, not diffs, message, diffs=diffs)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# S138 Queue Readiness Audit",
        "",
        f"- Overall: `{payload['overall']}`",
        f"- Screen config: `{payload['paths']['screen_config']}`",
        f"- Control config: `{payload['paths']['control_config']}`",
        f"- Full config: `{payload['paths']['full_config']}`",
        "",
        "## Checks",
        "",
        "| Severity | Pass | Check | Message |",
        "|---|---:|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['severity']} | {item['passed']} | `{item['name']}` | {item['message']} |")

    lines.extend(["", "## Parameter Counts", "", "| Variant | Params |", "|---|---:|"])
    for name, value in payload["parameter_counts"].items():
        lines.append(f"| {name} | {value} |")

    lines.extend(["", "## Smoke Summaries", "", "| Variant | Samples | Top-1 | Macro-F1 | Params |", "|---|---:|---:|---:|---:|"])
    for name, summary in payload["smoke_summaries"].items():
        if not summary:
            lines.append(f"| {name} | missing | missing | missing | missing |")
            continue
        lines.append(
            f"| {name} | {summary.get('num_samples')} | {summary.get('top1')} | "
            f"{summary.get('macro_f1')} | {summary.get('param_count')} |"
        )

    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in payload["notes"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the S138 fallback route before any GPU queue launch.")
    parser.add_argument(
        "--screen-config",
        type=Path,
        default=Path("configs/c3_farnet/c3_farnet_s138_dual_film_texture_roughness_screen_20260716.yaml"),
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        default=Path("configs/c3_farnet/c3_farnet_s138_dual_film_texture_roughness_control_20260716.yaml"),
    )
    parser.add_argument(
        "--full-config",
        type=Path,
        default=Path("configs/c3_farnet/c3_farnet_s138_dual_film_texture_roughness_full_20260716.yaml"),
    )
    parser.add_argument(
        "--smoke-metrics",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\s138_dual_film_texture_roughness_smoke_20260716\test_metrics.json"),
    )
    parser.add_argument(
        "--control-smoke-metrics",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\s138_dual_film_texture_roughness_control_smoke_20260716\test_metrics.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\S138_queue_readiness_20260716"),
    )
    args = parser.parse_args()

    checks: list[Check] = []
    for name, path in [
        ("screen_config_exists", args.screen_config),
        ("control_config_exists", args.control_config),
        ("full_config_exists", args.full_config),
    ]:
        _add(checks, name, "block", path.exists(), f"Required S138 artifact must exist: {path}", path=str(path))

    screen = load_config(args.screen_config)
    control = load_config(args.control_config)
    full = load_config(args.full_config)

    manifest_keys = [
        "data.train_manifest",
        "data.val_manifest",
        "data.test_manifest",
        "data.image_size",
        "data.train_resize_mode",
        "data.eval_resize_mode",
    ]
    _same_values(
        checks,
        "screen_control_same_data",
        screen,
        control,
        manifest_keys,
        message="S138 learned screen and off-control must use identical data, image size, and resize rules.",
    )
    for key in ["data.train_manifest", "data.val_manifest", "data.test_manifest"]:
        path = Path(str(_get(screen, key)))
        _add(checks, f"manifest_exists_{key.split('.')[-1]}", "block", path.exists(), f"S138 manifest path must exist: {path}", path=str(path))

    allowed_screen_control_diffs = {
        "output_dir",
        "model.dual_film_texture_coupling_mode",
        "loss.dual_film_texture_route_weight",
    }
    screen_flat = _flatten(screen)
    control_flat = _flatten(control)
    all_keys = set(screen_flat) | set(control_flat)
    unexpected_diffs = {}
    for key in sorted(all_keys):
        if key in allowed_screen_control_diffs:
            continue
        if screen_flat.get(key) != control_flat.get(key):
            unexpected_diffs[key] = {"screen": screen_flat.get(key), "control": control_flat.get(key)}
    _add(
        checks,
        "screen_control_only_intended_diffs",
        "block",
        not unexpected_diffs,
        "S138 same-budget control may differ only in output path, dual route mode, and dual route loss.",
        diffs=unexpected_diffs,
    )

    _add(
        checks,
        "screen_learned_dual_route_enabled",
        "block",
        bool(_get(screen, "model.use_dual_film_texture_coupling"))
        and str(_get(screen, "model.dual_film_texture_coupling_mode")).lower() == "learned"
        and float(_get(screen, "loss.dual_film_texture_route_weight", 0.0)) > 0.0,
        "S138 screen must enable the learned dry-concrete/film-concrete early route and supervise its gate.",
        mode=_get(screen, "model.dual_film_texture_coupling_mode"),
        route_weight=_get(screen, "loss.dual_film_texture_route_weight"),
    )
    _add(
        checks,
        "control_isolated_dual_route_off",
        "block",
        bool(_get(control, "model.use_dual_film_texture_coupling"))
        and str(_get(control, "model.dual_film_texture_coupling_mode")).lower() in {"off", "false", "none", "disabled"}
        and float(_get(control, "loss.dual_film_texture_route_weight", 0.0)) == 0.0,
        "S138 control must keep the module allocated but disable the learned effect and route supervision.",
        mode=_get(control, "model.dual_film_texture_coupling_mode"),
        route_weight=_get(control, "loss.dual_film_texture_route_weight"),
    )

    _add(
        checks,
        "screen_protocol_cap250",
        "block",
        int(_get(screen, "train.max_train_samples_per_class", -1)) == 1000
        and int(_get(screen, "train.samples_per_epoch", -1)) == 27000
        and int(_get(screen, "eval.max_val_samples_per_class", -1)) == 250
        and int(_get(screen, "eval.max_test_samples_per_class", -1)) == 250,
        "S138 screen must use the same cap1000/cap250 route-selection protocol as S96/S136/S137.",
        train_cap=_get(screen, "train.max_train_samples_per_class"),
        samples_per_epoch=_get(screen, "train.samples_per_epoch"),
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
        "S138 full must use uncapped train/val/test manifests for any final SOTA claim.",
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
        "S138 full manifest row counts must match the complete RSCD train/val/test protocol.",
        expected={
            "train": FULL_TRAIN_SAMPLES,
            "val": FULL_VAL_SAMPLES,
            "test": FULL_TEST_SAMPLES,
        },
        observed=full_manifest_counts,
        paths={split: str(path) for split, path in full_manifest_paths.items()},
    )
    expected_resume = str(Path(str(_get(screen, "output_dir"))) / "best_checkpoint.pth").replace("\\", "/")
    actual_resume = str(_get(full, "train.resume_from", "")).replace("\\", "/")
    _add(
        checks,
        "full_resumes_from_screen_best",
        "warn",
        actual_resume == expected_resume,
        "S138 full should resume from the promoted S138 screen checkpoint.",
        expected=expected_resume,
        actual=actual_resume,
    )

    param_counts: dict[str, int] = {}
    try:
        param_counts["screen"] = _param_count(screen)
        param_counts["control"] = _param_count(control)
        param_counts["full"] = _param_count(full)
        _add(
            checks,
            "screen_control_same_param_count",
            "block",
            param_counts["screen"] == param_counts["control"],
            "S138 learned screen and off-control must have identical parameter count.",
            screen_params=param_counts["screen"],
            control_params=param_counts["control"],
        )
    except Exception as exc:
        _add(checks, "param_count_build", "block", False, "S138 model construction failed.", error=repr(exc))

    smoke = _read_summary(args.smoke_metrics)
    control_smoke = _read_summary(args.control_smoke_metrics)
    _add(checks, "learned_smoke_metrics_exist", "block", smoke is not None, f"S138 learned smoke metrics should exist: {args.smoke_metrics}", path=str(args.smoke_metrics))
    _add(checks, "control_smoke_metrics_exist", "block", control_smoke is not None, f"S138 control smoke metrics should exist: {args.control_smoke_metrics}", path=str(args.control_smoke_metrics))
    if smoke is not None and control_smoke is not None:
        _add(
            checks,
            "smoke_same_param_count",
            "block",
            int(smoke.get("param_count", -1)) == int(control_smoke.get("param_count", -2)),
            "S138 learned/control smoke runs should report identical parameter counts.",
            learned_params=smoke.get("param_count"),
            control_params=control_smoke.get("param_count"),
        )
        _add(
            checks,
            "smoke_schema_has_required_metrics",
            "block",
            {"top1", "macro_f1", "num_samples", "param_count"}.issubset(smoke)
            and {"top1", "macro_f1", "num_samples", "param_count"}.issubset(control_smoke),
            "S138 smoke metrics should contain the minimal train/eval schema.",
            learned_keys=sorted(smoke),
            control_keys=sorted(control_smoke),
        )

    block_failures = [check for check in checks if check.severity == "block" and not check.passed]
    overall = "pass" if not block_failures else "fail"
    payload = {
        "overall": overall,
        "checks": [asdict(check) for check in checks],
        "parameter_counts": param_counts,
        "full_manifest_counts": full_manifest_counts,
        "smoke_summaries": {
            "learned": smoke,
            "control": control_smoke,
        },
        "notes": [
            "S138 targets an RSCD-specific coupling law: dry concrete roughness is visible texture, while wet/water concrete roughness can be hidden by film and must use film-erasure evidence.",
            "The off-control is intentionally parameter-matched; it disables the early dual route effect rather than deleting the module.",
            "Smoke results prove code execution and metric schema only. They are not performance evidence because they use 3 samples per class.",
            "S138 is a fallback candidate and should not be launched while the current S133c/S135/S136/S136d/S137 queue is active.",
        ],
        "paths": {
            "screen_config": str(args.screen_config),
            "control_config": str(args.control_config),
            "full_config": str(args.full_config),
            "smoke_metrics": str(args.smoke_metrics),
            "control_smoke_metrics": str(args.control_smoke_metrics),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "s138_queue_readiness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "s138_queue_readiness.md")
    print(json.dumps({"overall": overall, "report": str(args.output_dir / "s138_queue_readiness.md")}, ensure_ascii=False))
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

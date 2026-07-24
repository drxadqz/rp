from __future__ import annotations

import argparse
import csv
import json
import py_compile
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from friction_affordance.c3_experiment import apply_trainable_prefixes, build_class_map, build_model, load_config  # noqa: E402


STEM_PREFIX = "water_concrete_topology_texture_stem_conditioner"
ALLOWED_NONSTEM_MISSING_SUFFIXES = ("cell_mask", "chart_mask", "active_mask")


def _check(name: str, passed: bool, detail: str, *, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail, "severity": severity}


def _fmt_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _compile_sources() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for rel in [
        "src/friction_affordance/models/c3_farnet.py",
        "src/friction_affordance/c3_experiment.py",
        "scripts/diagnose_candidate_route.py",
        "scripts/snapshot_rscd_candidate.py",
        "scripts/verify_candidate_integrity.py",
        "scripts/analyze_sota_gap_budget.py",
    ]:
        path = ROOT / rel
        try:
            py_compile.compile(str(path), doraise=True)
            checks.append(_check(f"compile:{rel}", True, "py_compile ok"))
        except Exception as exc:
            checks.append(_check(f"compile:{rel}", False, repr(exc)))
    return checks


def _checkpoint_compatibility(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = state.get("model", state.get("state_dict", state))
    target = model.state_dict()
    aliases = {
        "classifier.weight": "linear_head.weight",
        "classifier.bias": "linear_head.bias",
        "backbone.proj.weight": "backbone.global_proj.weight",
        "backbone.proj.bias": "backbone.global_proj.bias",
    }
    loadable: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, tensor in raw.items():
        if name.endswith(ALLOWED_NONSTEM_MISSING_SUFFIXES):
            skipped.append(name)
            continue
        if name in target and tuple(target[name].shape) == tuple(tensor.shape):
            loadable[name] = tensor
            continue
        alias = aliases.get(name)
        if alias and alias in target and tuple(target[alias].shape) == tuple(tensor.shape):
            loadable[alias] = tensor
        else:
            skipped.append(name)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    stem_missing = [name for name in missing if name.startswith(STEM_PREFIX)]
    nonstem_missing = [name for name in missing if not name.startswith(STEM_PREFIX)]
    allowed_nonstem_missing = [
        name for name in nonstem_missing if name.endswith(ALLOWED_NONSTEM_MISSING_SUFFIXES)
    ]
    disallowed_nonstem_missing = [
        name for name in nonstem_missing if not name.endswith(ALLOWED_NONSTEM_MISSING_SUFFIXES)
    ]
    return {
        "checkpoint": str(checkpoint),
        "loaded": len(loadable),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "stem_missing": len(stem_missing),
        "allowed_nonstem_missing": allowed_nonstem_missing,
        "disallowed_nonstem_missing": disallowed_nonstem_missing,
        "stem_missing_names": stem_missing,
        "unexpected_names": unexpected,
    }


def _load_class_map(cfg: dict[str, Any]) -> dict[str, int]:
    data = cfg["data"]
    manifests = [Path(data["train_manifest"]), Path(data["val_manifest"]), Path(data["test_manifest"])]
    return build_class_map(manifests)


def _trainable_summary(model: torch.nn.Module, cfg: dict[str, Any]) -> dict[str, Any]:
    apply_trainable_prefixes(model, cfg["train"].get("trainable_prefixes"))
    rows = [(name, int(param.numel())) for name, param in model.named_parameters() if param.requires_grad]
    return {
        "num_params": int(sum(num for _, num in rows)),
        "names": [name for name, _ in rows],
    }


def _is_none(value: Any) -> bool:
    return value is None


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit S135c queue readiness before screen/full launch.")
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--full-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["queue", "full"], default="queue")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    checks.extend(_compile_sources())

    screen_cfg = load_config(args.screen_config)
    full_cfg = load_config(args.full_config)
    screen_out = Path(screen_cfg["output_dir"])
    full_out = Path(full_cfg["output_dir"])
    screen_train = screen_cfg["train"]
    full_train = full_cfg["train"]
    screen_eval = screen_cfg["eval"]
    full_eval = full_cfg["eval"]
    screen_model_cfg = screen_cfg["model"]
    full_model_cfg = full_cfg["model"]

    checks.append(_check("screen_config_exists", args.screen_config.exists(), str(args.screen_config)))
    checks.append(_check("full_config_exists", args.full_config.exists(), str(args.full_config)))
    checks.append(
        _check(
            "screen_output_not_full_output",
            screen_out != full_out,
            f"screen={screen_out}; full={full_out}",
        )
    )
    checks.append(
        _check(
            "screen_protocol_cap250",
            screen_train.get("max_train_samples_per_class") == 1000
            and int(screen_train.get("samples_per_epoch", -1)) == 27000
            and screen_eval.get("max_val_samples_per_class") == 250
            and screen_eval.get("max_test_samples_per_class") == 250,
            "screen must remain same-budget capped protocol",
        )
    )
    checks.append(
        _check(
            "full_protocol_uncapped",
            _is_none(full_train.get("max_train_samples_per_class"))
            and int(full_train.get("samples_per_epoch", -1)) == 0
            and _is_none(full_eval.get("max_val_samples_per_class"))
            and _is_none(full_eval.get("max_test_samples_per_class")),
            "full promotion must use complete train/val/test manifests",
        )
    )
    expected_prefixes = [STEM_PREFIX]
    checks.append(
        _check(
            "screen_trainable_prefix",
            screen_train.get("trainable_prefixes") == expected_prefixes,
            f"trainable_prefixes={screen_train.get('trainable_prefixes')}",
        )
    )
    checks.append(
        _check(
            "full_trainable_prefix",
            full_train.get("trainable_prefixes") == expected_prefixes,
            f"trainable_prefixes={full_train.get('trainable_prefixes')}",
        )
    )
    checks.append(
        _check(
            "stem_enabled",
            bool(screen_model_cfg.get("use_water_concrete_topology_texture_stem_conditioner"))
            and bool(full_model_cfg.get("use_water_concrete_topology_texture_stem_conditioner")),
            "S135c early stem must be enabled in screen and full config",
        )
    )

    screen_resume = Path(str(screen_train.get("resume_from", "")))
    screen_teacher = Path(str(screen_train.get("teacher_checkpoint", "")))
    full_resume = Path(str(full_train.get("resume_from", "")))
    full_teacher = Path(str(full_train.get("teacher_checkpoint", "")))
    checks.append(_check("screen_resume_exists", screen_resume.exists(), str(screen_resume)))
    checks.append(_check("screen_teacher_exists", screen_teacher.exists(), str(screen_teacher)))
    checks.append(_check("full_teacher_exists", full_teacher.exists(), str(full_teacher)))
    if args.mode == "full":
        checks.append(_check("full_resume_exists", full_resume.exists(), str(full_resume)))
    else:
        checks.append(
            _check(
                "full_resume_pending",
                str(full_resume).endswith("best_checkpoint.pth"),
                f"expected pending screen best checkpoint: {full_resume}",
                severity="warning",
            )
        )

    drive = Path(str(screen_out)).drive or Path(str(args.output_dir)).drive
    usage = shutil.disk_usage(drive + "\\")
    free_gb = usage.free / (1024**3)
    checks.append(
        _check(
            "disk_free_for_screen",
            free_gb >= float(args.min_free_gb),
            f"{drive} free={free_gb:.2f}GB required={args.min_free_gb:.2f}GB",
        )
    )

    model_info: dict[str, Any] = {}
    compatibility: dict[str, Any] = {}
    trainable: dict[str, Any] = {}
    try:
        class_to_idx = _load_class_map(screen_cfg)
        checks.append(_check("class_count_27", len(class_to_idx) == 27, f"class_count={len(class_to_idx)}"))
        model = build_model(screen_cfg, class_to_idx).cpu()
        cond = getattr(model, "water_concrete_topology_texture_stem_conditioner", None)
        model_info = {
            "conditioner_class": type(cond).__name__ if cond is not None else None,
            "mechanism_channels": getattr(cond, "mechanism_channels", None),
            "adapter_first_weight_shape": tuple(cond.adapter[0].weight.shape) if cond is not None else None,
            "global_gate_norm_shape": tuple(cond.global_gate[0].normalized_shape) if cond is not None else None,
        }
        checks.append(
            _check(
                "mechanism_channels_12",
                model_info["mechanism_channels"] == 12,
                f"mechanism_channels={model_info['mechanism_channels']}",
            )
        )
        if screen_resume.exists():
            compatibility = _checkpoint_compatibility(model, screen_resume)
            checks.append(
                _check(
                    "checkpoint_no_unexpected",
                    compatibility["unexpected"] == 0,
                    f"unexpected={compatibility['unexpected']}",
                )
            )
            checks.append(
                _check(
                    "checkpoint_no_disallowed_nonstem_missing",
                    not compatibility["disallowed_nonstem_missing"],
                    f"disallowed_nonstem_missing={compatibility['disallowed_nonstem_missing']}",
                )
            )
            checks.append(
                _check(
                    "checkpoint_loaded_backbone_head",
                    compatibility["loaded"] > 700,
                    f"loaded={compatibility['loaded']}",
                )
            )
        trainable = _trainable_summary(model, screen_cfg)
        checks.append(
            _check(
                "trainable_only_stem",
                trainable["num_params"] > 0
                and all(name.startswith(STEM_PREFIX) for name in trainable["names"]),
                f"trainable_params={trainable['num_params']}",
            )
        )
    except Exception as exc:
        checks.append(_check("model_build_and_load", False, repr(exc)))

    errors = [item for item in checks if item["severity"] == "error" and not item["pass"]]
    warnings = [item for item in checks if item["severity"] == "warning" and not item["pass"]]
    payload = {
        "ok": not errors,
        "mode": args.mode,
        "screen_config": str(args.screen_config),
        "full_config": str(args.full_config),
        "screen_output_dir": str(screen_out),
        "full_output_dir": str(full_out),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "model_info": model_info,
        "checkpoint_compatibility": compatibility,
        "trainable_summary": trainable,
    }
    (args.output_dir / "s135c_queue_readiness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# S135c Queue Readiness Audit",
        "",
        f"- Mode: `{args.mode}`",
        f"- Screen config: `{args.screen_config}`",
        f"- Full config: `{args.full_config}`",
        f"- Overall readiness: **{payload['ok']}**",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Detail |",
        "|---|---:|---|---|",
    ]
    for item in checks:
        md.append(f"| {item['name']} | {_fmt_bool(bool(item['pass']))} | {item['severity']} | {item['detail']} |")
    md.extend(["", "## Model", "", "```json", json.dumps(model_info, indent=2, ensure_ascii=False), "```"])
    md.extend(["", "## Checkpoint Compatibility", "", "```json", json.dumps(compatibility, indent=2, ensure_ascii=False), "```"])
    md.extend(["", "## Trainable Summary", "", "```json", json.dumps(trainable, indent=2, ensure_ascii=False), "```"])
    (args.output_dir / "s135c_queue_readiness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "report": str(args.output_dir / "s135c_queue_readiness.md")}, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

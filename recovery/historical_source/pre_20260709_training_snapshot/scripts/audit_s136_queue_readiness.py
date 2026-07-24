from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from friction_affordance.c3_experiment import load_config  # noqa: E402


FULL_TRAIN_SAMPLES = 958_941
FULL_VAL_SAMPLES = 19_860
FULL_TEST_SAMPLES = 49_500


@dataclass
class Check:
    name: str
    status: str
    detail: str


def check(name: str, passed: bool, detail: str) -> Check:
    return Check(name=name, status="pass" if passed else "fail", detail=detail)


def warn(name: str, detail: str) -> Check:
    return Check(name=name, status="warn", detail=detail)


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "test_metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("summary", payload))


def count_manifest_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def resolved_path(value: Any) -> Path:
    return Path(str(value or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit S136 custom-backbone queue readiness and fairness notes.")
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--full-config", required=True, type=Path)
    parser.add_argument("--control-config", required=True, type=Path)
    parser.add_argument("--s96-dir", required=True, type=Path)
    parser.add_argument("--s7-dir", required=True, type=Path)
    parser.add_argument("--smoke-dir", required=True, type=Path)
    parser.add_argument("--watcher-script", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[Check] = []
    checks.append(check("screen_config_exists", args.screen_config.exists(), str(args.screen_config)))
    checks.append(check("full_config_exists", args.full_config.exists(), str(args.full_config)))
    checks.append(check("control_config_exists", args.control_config.exists(), str(args.control_config)))
    checks.append(check("watcher_script_exists", args.watcher_script.exists(), str(args.watcher_script)))

    screen_cfg = load_config(args.screen_config)
    full_cfg = load_config(args.full_config)
    screen_out = Path(str(screen_cfg["output_dir"]))
    full_out = Path(str(full_cfg["output_dir"]))
    screen_train = screen_cfg["train"]
    screen_eval = screen_cfg["eval"]
    full_train = full_cfg["train"]
    full_eval = full_cfg["eval"]
    control_cfg = load_config(args.control_config)

    checks.append(
        check(
            "screen_protocol_cap250",
            screen_train.get("max_train_samples_per_class") == 1000
            and int(screen_train.get("samples_per_epoch", -1)) == 27000
            and screen_eval.get("max_val_samples_per_class") == 250
            and screen_eval.get("max_test_samples_per_class") == 250,
            "screen must remain capped route-selection protocol: train cap 1000/class, samples_per_epoch 27000, val/test cap250",
        )
    )
    checks.append(
        check(
            "screen_uses_custom_training_entry",
            "train_coupled_factor_backbone.py" in args.watcher_script.read_text(encoding="utf-8"),
            "fallback watcher should launch the S136 custom-backbone training entry",
        )
    )
    checks.append(
        check(
            "full_protocol_complete",
            full_train.get("max_train_samples_per_class") is None
            and int(full_train.get("samples_per_epoch", -1)) == 0
            and full_eval.get("max_val_samples_per_class") is None
            and full_eval.get("max_test_samples_per_class") is None,
            "full protocol must use complete train/val/test manifests",
        )
    )
    full_data = full_cfg.get("data") or {}
    manifest_paths = {
        "train": resolved_path(full_data.get("train_manifest")),
        "val": resolved_path(full_data.get("val_manifest")),
        "test": resolved_path(full_data.get("test_manifest")),
    }
    manifest_counts = {split: count_manifest_rows(path) for split, path in manifest_paths.items()}
    checks.append(
        check(
            "full_manifest_rows_complete",
            manifest_counts["train"] == FULL_TRAIN_SAMPLES
            and manifest_counts["val"] == FULL_VAL_SAMPLES
            and manifest_counts["test"] == FULL_TEST_SAMPLES,
            "expected train/val/test rows="
            f"{FULL_TRAIN_SAMPLES}/{FULL_VAL_SAMPLES}/{FULL_TEST_SAMPLES}; observed={manifest_counts}",
        )
    )
    expected_resume = str(screen_out / "best_checkpoint.pth").replace("\\", "/")
    actual_resume = str(full_train.get("resume_from", "")).replace("\\", "/")
    checks.append(
        check(
            "full_resumes_from_screen_checkpoint",
            actual_resume == expected_resume,
            f"resume_from={actual_resume}; expected={expected_resume}",
        )
    )

    smoke_summary = read_summary(args.smoke_dir)
    if smoke_summary is None:
        checks.append(check("smoke_metrics_exist", False, str(args.smoke_dir / "test_metrics.json")))
    else:
        required = {"top1", "macro_f1", "hard_class_mean_f1", "water_concrete_slight_f1", "param_count"}
        checks.append(
            check(
                "smoke_metrics_schema",
                required.issubset(smoke_summary),
                f"required={sorted(required)} present={sorted(set(smoke_summary) & required)}",
            )
        )

    s96_summary = read_summary(args.s96_dir)
    s7_summary = read_summary(args.s7_dir)
    checks.append(
        check(
            "s96_cap250_baseline_available",
            s96_summary is not None and int(float(s96_summary.get("num_samples", 0))) == 6750,
            str(args.s96_dir),
        )
    )
    checks.append(
        check(
            "s7_full_baseline_available",
            s7_summary is not None and int(float(s7_summary.get("num_samples", 0))) >= 40000,
            str(args.s7_dir),
        )
    )

    checks.append(
        warn(
            "screen_training_budget_interpretation",
            "S136 screen trains a custom backbone from scratch, so its screen result is route-selection evidence only. "
            "It must not be claimed as final fair SOTA. The final claim requires S136_full on complete RSCD train/test.",
        )
    )
    control_train = control_cfg["train"]
    control_eval = control_cfg["eval"]
    control_model = control_cfg["model"]
    control_loss = control_cfg["loss"]
    checks.append(
        check(
            "control_protocol_matches_screen",
            control_train.get("max_train_samples_per_class") == screen_train.get("max_train_samples_per_class")
            and int(control_train.get("samples_per_epoch", -1)) == int(screen_train.get("samples_per_epoch", -2))
            and int(control_train.get("epochs", -1)) == int(screen_train.get("epochs", -2))
            and int(control_train.get("batch_size", -1)) == int(screen_train.get("batch_size", -2))
            and control_eval.get("max_val_samples_per_class") == screen_eval.get("max_val_samples_per_class")
            and control_eval.get("max_test_samples_per_class") == screen_eval.get("max_test_samples_per_class"),
            "fixed-uniform control must use the same cap250 training/evaluation budget, epochs, and batch size as S136 screen",
        )
    )
    checks.append(
        check(
            "control_disables_adaptive_gate",
            str(control_model.get("coupling_gate_mode", "")) == "fixed_uniform"
            and float(control_loss.get("coupling_gate_weight", 1.0)) == 0.0,
            f"coupling_gate_mode={control_model.get('coupling_gate_mode')} coupling_gate_weight={control_loss.get('coupling_gate_weight')}",
        )
    )

    payload = {
        "ok": not any(item.status == "fail" for item in checks),
        "screen_config": str(args.screen_config),
        "full_config": str(args.full_config),
        "control_config": str(args.control_config) if args.control_config is not None else None,
        "screen_output_dir": str(screen_out),
        "full_output_dir": str(full_out),
        "full_manifest_counts": manifest_counts,
        "full_manifest_paths": {split: str(path) for split, path in manifest_paths.items()},
        "checks": [asdict(item) for item in checks],
        "screen_train": {
            "epochs": screen_train.get("epochs"),
            "samples_per_epoch": screen_train.get("samples_per_epoch"),
            "max_train_samples_per_class": screen_train.get("max_train_samples_per_class"),
            "batch_size": screen_train.get("batch_size"),
        },
        "screen_eval": {
            "max_val_samples_per_class": screen_eval.get("max_val_samples_per_class"),
            "max_test_samples_per_class": screen_eval.get("max_test_samples_per_class"),
        },
        "full_train": {
            "epochs": full_train.get("epochs"),
            "samples_per_epoch": full_train.get("samples_per_epoch"),
            "max_train_samples_per_class": full_train.get("max_train_samples_per_class"),
            "batch_size": full_train.get("batch_size"),
            "resume_from": full_train.get("resume_from"),
        },
        "full_eval": {
            "max_val_samples_per_class": full_eval.get("max_val_samples_per_class"),
            "max_test_samples_per_class": full_eval.get("max_test_samples_per_class"),
        },
    }
    (args.output_dir / "s136_queue_readiness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# S136 Queue Readiness And Fairness Audit",
        "",
        f"- Screen config: `{args.screen_config}`",
        f"- Full config: `{args.full_config}`",
        f"- Control config: `{args.control_config}`",
        f"- OK: `{payload['ok']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for item in checks:
        lines.append(f"| {item.name} | {item.status} | {item.detail} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "S136 is an early coupling custom backbone. The cap250 screen is a route-selection test under the same evaluation cap as S96/S135 screens. "
            "Because S136 is trained from scratch while S96/S135 are fine-tuning routes, the screen should not be used as a final SOTA claim.",
            "",
            "A final fair claim requires the full S136 protocol on complete RSCD manifests and the promotion audit with `--require-sota`. "
            "If S136 screen is strong, run a same-budget gate-disabled ablation before writing the mechanism claim so the improvement is attributable to early factor coupling rather than capacity or training length.",
        ]
    )
    (args.output_dir / "s136_queue_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "report": str(args.output_dir / "s136_queue_readiness.md")}, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

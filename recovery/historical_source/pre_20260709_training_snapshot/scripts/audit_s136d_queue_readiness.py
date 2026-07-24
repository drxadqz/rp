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


def norm(path_text: Any) -> str:
    return str(path_text).replace("\\", "/")


def is_empty(value: Any) -> bool:
    return value is None or value == ""


def cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    value = cfg.get(key, default)
    if value is None or value == "":
        return float(default)
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit S136d safe-distilled custom-backbone queue readiness.")
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--full-config", required=True, type=Path)
    parser.add_argument("--teacher-screen-config", required=True, type=Path)
    parser.add_argument("--teacher-full-config", required=True, type=Path)
    parser.add_argument("--s96-dir", required=True, type=Path)
    parser.add_argument("--s7-dir", required=True, type=Path)
    parser.add_argument("--distill-smoke-dir", required=True, type=Path)
    parser.add_argument("--watcher-script", required=True, type=Path)
    parser.add_argument("--nodistill-screen-config", type=Path)
    parser.add_argument("--nodistill-screen-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[Check] = []
    for name, path in [
        ("screen_config_exists", args.screen_config),
        ("full_config_exists", args.full_config),
        ("teacher_screen_config_exists", args.teacher_screen_config),
        ("teacher_full_config_exists", args.teacher_full_config),
        ("watcher_script_exists", args.watcher_script),
    ]:
        checks.append(check(name, path.exists(), str(path)))

    screen_cfg = load_config(args.screen_config)
    full_cfg = load_config(args.full_config)
    teacher_screen_cfg = load_config(args.teacher_screen_config)
    teacher_full_cfg = load_config(args.teacher_full_config)
    nodistill_cfg = load_config(args.nodistill_screen_config) if args.nodistill_screen_config else None

    screen_out = Path(str(screen_cfg["output_dir"]))
    full_out = Path(str(full_cfg["output_dir"]))
    screen_train = screen_cfg["train"]
    screen_eval = screen_cfg["eval"]
    full_train = full_cfg["train"]
    full_eval = full_cfg["eval"]
    teacher_screen_train = teacher_screen_cfg["train"]
    teacher_screen_eval = teacher_screen_cfg["eval"]
    teacher_full_train = teacher_full_cfg["train"]
    teacher_full_eval = teacher_full_cfg["eval"]
    loss_cfg = screen_cfg["loss"]
    nodistill_train = nodistill_cfg["train"] if nodistill_cfg is not None else None
    nodistill_eval = nodistill_cfg["eval"] if nodistill_cfg is not None else None
    nodistill_model = nodistill_cfg["model"] if nodistill_cfg is not None else None

    manifests = [
        Path(screen_cfg["data"]["train_manifest"]),
        Path(screen_cfg["data"]["val_manifest"]),
        Path(screen_cfg["data"]["test_manifest"]),
    ]
    class_to_idx = build_class_map(manifests)
    model = build_s136_model(screen_cfg, class_to_idx)
    param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)

    checks.append(
        check(
            "screen_protocol_cap250",
            screen_train.get("max_train_samples_per_class") == 1000
            and int(screen_train.get("samples_per_epoch", -1)) == 27000
            and screen_eval.get("max_val_samples_per_class") == 250
            and screen_eval.get("max_test_samples_per_class") == 250,
            "screen must use train cap 1000/class, samples_per_epoch 27000, val/test cap250",
        )
    )
    checks.append(
        check(
            "full_protocol_complete",
            is_empty(full_train.get("max_train_samples_per_class"))
            and int(full_train.get("samples_per_epoch", -1)) == 0
            and is_empty(full_eval.get("max_val_samples_per_class"))
            and is_empty(full_eval.get("max_test_samples_per_class")),
            "full must use complete train/val/test manifests with samples_per_epoch=0 and no eval caps",
        )
    )
    full_manifest_paths = {
        "train": Path(str(full_cfg["data"]["train_manifest"])),
        "val": Path(str(full_cfg["data"]["val_manifest"])),
        "test": Path(str(full_cfg["data"]["test_manifest"])),
    }
    full_manifest_counts = {split: count_manifest_rows(path) for split, path in full_manifest_paths.items()}
    checks.append(
        check(
            "full_manifest_rows_complete",
            full_manifest_counts["train"] == FULL_TRAIN_SAMPLES
            and full_manifest_counts["val"] == FULL_VAL_SAMPLES
            and full_manifest_counts["test"] == FULL_TEST_SAMPLES,
            "expected train/val/test rows="
            f"{FULL_TRAIN_SAMPLES}/{FULL_VAL_SAMPLES}/{FULL_TEST_SAMPLES}; observed={full_manifest_counts}",
        )
    )
    teacher_full_manifest_paths = {
        "train": Path(str(teacher_full_cfg["data"]["train_manifest"])),
        "val": Path(str(teacher_full_cfg["data"]["val_manifest"])),
        "test": Path(str(teacher_full_cfg["data"]["test_manifest"])),
    }
    teacher_full_manifest_counts = {
        split: count_manifest_rows(path)
        for split, path in teacher_full_manifest_paths.items()
    }
    checks.append(
        check(
            "teacher_full_manifest_rows_complete",
            teacher_full_manifest_counts["train"] == FULL_TRAIN_SAMPLES
            and teacher_full_manifest_counts["val"] == FULL_VAL_SAMPLES
            and teacher_full_manifest_counts["test"] == FULL_TEST_SAMPLES,
            "expected teacher train/val/test rows="
            f"{FULL_TRAIN_SAMPLES}/{FULL_VAL_SAMPLES}/{FULL_TEST_SAMPLES}; observed={teacher_full_manifest_counts}",
        )
    )
    checks.append(
        check(
            "full_resumes_from_screen_checkpoint",
            norm(full_train.get("resume_from", "")) == norm(screen_out / "best_checkpoint.pth"),
            f"resume_from={full_train.get('resume_from')} expected={screen_out / 'best_checkpoint.pth'}",
        )
    )
    checks.append(
        check(
            "screen_teacher_cache_path_matches_output",
            norm(screen_train.get("teacher_logits_cache", "")).startswith(norm(screen_out))
            and norm(screen_train.get("teacher_logits_cache", "")).endswith("s7_teacher_logits_train_cap1000.pt")
            and bool(screen_train.get("teacher_logits_cache_strict", False)),
            f"teacher_logits_cache={screen_train.get('teacher_logits_cache')} strict={screen_train.get('teacher_logits_cache_strict')}",
        )
    )
    checks.append(
        check(
            "full_teacher_cache_path_matches_output",
            norm(full_train.get("teacher_logits_cache", "")).startswith(norm(full_out))
            and norm(full_train.get("teacher_logits_cache", "")).endswith("s7_teacher_logits_train_full.pt")
            and bool(full_train.get("teacher_logits_cache_strict", False)),
            f"teacher_logits_cache={full_train.get('teacher_logits_cache')} strict={full_train.get('teacher_logits_cache_strict')}",
        )
    )
    checks.append(
        check(
            "teacher_screen_cache_protocol_matches_screen",
            teacher_screen_train.get("max_train_samples_per_class") == screen_train.get("max_train_samples_per_class")
            and teacher_screen_eval.get("max_val_samples_per_class") == screen_eval.get("max_val_samples_per_class")
            and teacher_screen_eval.get("max_test_samples_per_class") == screen_eval.get("max_test_samples_per_class"),
            "teacher screen cache must cover exactly the capped training sample universe used by S136d screen",
        )
    )
    checks.append(
        check(
            "teacher_full_cache_protocol_complete",
            is_empty(teacher_full_train.get("max_train_samples_per_class"))
            and is_empty(teacher_full_eval.get("max_val_samples_per_class"))
            and is_empty(teacher_full_eval.get("max_test_samples_per_class")),
            "teacher full cache must cover complete training data for S136d full",
        )
    )
    s7_teacher_ckpt = Path(str(teacher_screen_train.get("teacher_checkpoint", "")))
    checks.append(
        check(
            "s7_teacher_checkpoint_exists",
            s7_teacher_ckpt.exists() and norm(teacher_full_train.get("teacher_checkpoint", "")) == norm(s7_teacher_ckpt),
            f"screen={teacher_screen_train.get('teacher_checkpoint')} full={teacher_full_train.get('teacher_checkpoint')}",
        )
    )

    active_losses = {
        key: float(loss_cfg.get(key, 0.0) or 0.0)
        for key in [
            "anchor_consistency_weight",
            "anchor_consistency_protect_weight",
            "anchor_nonregression_weight",
            "pareto_safe_distill_weight",
            "pareto_safe_margin_weight",
            "pareto_safe_hardpair_margin_weight",
        ]
    }
    checks.append(
        check(
            "safe_distill_losses_enabled",
            all(value > 0.0 for value in active_losses.values()),
            json.dumps(active_losses, ensure_ascii=False),
        )
    )

    if args.nodistill_screen_config is None:
        checks.append(warn("nodistill_control_config_missing", "No no-distill S136 screen config was provided."))
    else:
        checks.append(check("nodistill_control_config_exists", args.nodistill_screen_config.exists(), str(args.nodistill_screen_config)))
        assert nodistill_train is not None
        assert nodistill_eval is not None
        assert nodistill_model is not None
        checks.append(
            check(
                "nodistill_control_protocol_matches_screen",
                nodistill_train.get("max_train_samples_per_class") == screen_train.get("max_train_samples_per_class")
                and int(nodistill_train.get("samples_per_epoch", -1)) == int(screen_train.get("samples_per_epoch", -2))
                and int(nodistill_train.get("epochs", -1)) == int(screen_train.get("epochs", -2))
                and int(nodistill_train.get("batch_size", -1)) == int(screen_train.get("batch_size", -2))
                and nodistill_eval.get("max_val_samples_per_class") == screen_eval.get("max_val_samples_per_class")
                and nodistill_eval.get("max_test_samples_per_class") == screen_eval.get("max_test_samples_per_class"),
                "S136 no-distill control must use the same cap250 training/evaluation budget, epochs, and batch size as S136d screen",
            )
        )
        checks.append(
            check(
                "nodistill_control_same_custom_backbone_shape",
                int(nodistill_model.get("stem_dim", -1)) == int(screen_cfg["model"].get("stem_dim", -2))
                and list(nodistill_model.get("stage_dims", [])) == list(screen_cfg["model"].get("stage_dims", []))
                and str(nodistill_model.get("coupling_gate_mode", "learned")) == str(screen_cfg["model"].get("coupling_gate_mode", "learned")),
                f"nodistill_model={nodistill_model} s136d_model={screen_cfg['model']}",
            )
        )
        nodistill_loss = nodistill_cfg["loss"]
        checks.append(
            check(
                "nodistill_control_has_no_teacher_guard",
                "teacher_logits_cache" not in nodistill_train
                and float(nodistill_loss.get("anchor_consistency_weight", 0.0) or 0.0) == 0.0
                and float(nodistill_loss.get("anchor_nonregression_weight", 0.0) or 0.0) == 0.0
                and float(nodistill_loss.get("pareto_safe_distill_weight", 0.0) or 0.0) == 0.0,
                "control should isolate the teacher no-harm guard by disabling teacher cache and safe-distill losses",
            )
        )
    if args.nodistill_screen_dir is None:
        checks.append(warn("nodistill_control_metrics_pending", "No no-distill S136 screen output dir was provided."))
    else:
        nodistill_summary = read_summary(args.nodistill_screen_dir)
        if nodistill_summary is None:
            checks.append(
                warn(
                    "nodistill_control_metrics_pending",
                    f"S136 no-distill metrics are not available yet: {args.nodistill_screen_dir / 'test_metrics.json'}",
                )
            )
        else:
            checks.append(
                check(
                    "nodistill_control_protocol_result_available",
                    int(float(nodistill_summary.get("num_samples", 0))) == 6750,
                    f"{args.nodistill_screen_dir} num_samples={nodistill_summary.get('num_samples')}",
                )
            )
    focus_classes = set(loss_cfg.get("pareto_safe_focus_classes", []))
    expected_focus = {
        "water_concrete_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "wet_concrete_severe",
        "dry_concrete_slight",
    }
    checks.append(
        check(
            "focus_classes_target_coupled_boundaries",
            expected_focus.issubset(focus_classes),
            f"focus={sorted(focus_classes)} expected_subset={sorted(expected_focus)}",
        )
    )
    checks.append(
        check(
            "screen_safe_distill_schedule_early_coupling_first",
            cfg_float(loss_cfg, "safe_distill_warmup_epochs", 0.0) >= 1.0
            and cfg_float(loss_cfg, "safe_distill_ramp_epochs", 0.0) >= 1.0
            and cfg_float(loss_cfg, "safe_distill_initial_scale", 1.0) == 0.0
            and cfg_float(loss_cfg, "safe_distill_final_scale", 0.0) == 1.0,
            "screen should let the custom factor/coupling backbone learn first, then ramp teacher no-harm protection",
        )
    )
    full_loss_cfg = full_cfg["loss"]
    checks.append(
        check(
            "full_safe_distill_schedule_full_strength",
            cfg_float(full_loss_cfg, "safe_distill_warmup_epochs", 0.0) == 0.0
            and cfg_float(full_loss_cfg, "safe_distill_ramp_epochs", 0.0) == 0.0
            and cfg_float(full_loss_cfg, "safe_distill_final_scale", 1.0) == 1.0,
            "full resumes from the screened backbone, so teacher no-harm guard should be active immediately",
        )
    )

    watcher_text = args.watcher_script.read_text(encoding="utf-8")
    for name, snippet in [
        ("watcher_runs_teacher_cache", "cache_teacher_logits.py"),
        ("watcher_runs_s136d_training", "train_coupled_factor_backbone.py"),
        ("watcher_audits_screen_vs_s96", "S136d_screen_promotion_audit_vs_S96"),
        ("watcher_compares_safe_distill_to_nodistill", "S136d_safe_distill_vs_S136_no_distill_screen"),
        ("watcher_runs_mechanism_diagnosis", "diagnose_s136d_mechanism_route.py"),
        ("watcher_requires_full_sota_audit", "S136d_full_promotion_audit_vs_S7"),
        ("watcher_blocks_while_active_training", "Get-ActiveRscdTraining"),
    ]:
        checks.append(check(name, snippet in watcher_text, snippet))

    smoke_summary = read_summary(args.distill_smoke_dir)
    if smoke_summary is None:
        checks.append(check("distill_smoke_metrics_exist", False, str(args.distill_smoke_dir / "test_metrics.json")))
    else:
        required = {"top1", "macro_f1", "water_concrete_slight_f1", "param_count"}
        checks.append(
            check(
                "distill_smoke_metrics_schema",
                required.issubset(smoke_summary),
                f"required={sorted(required)} present={sorted(set(smoke_summary) & required)}",
            )
        )
        history_path = args.distill_smoke_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            train_log = dict(history[-1].get("train", {})) if history else {}
            log_required = {
                "s136_anchor_teacher_cache_hit",
                "loss_anchor_consistency",
                "loss_anchor_nonregression",
                "loss_pareto_safe_distill",
            }
            checks.append(
                check(
                    "distill_smoke_logged_safe_losses",
                    log_required.issubset(train_log),
                    f"required={sorted(log_required)} present={sorted(set(train_log) & log_required)}",
                )
            )
        else:
            checks.append(check("distill_smoke_history_exists", False, str(history_path)))

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
            "method_claim_boundary",
            "S136d is still a custom backbone route-selection candidate until full complete-manifest training and final --require-sota audit pass.",
        )
    )
    checks.append(
        warn(
            "teacher_usage_interpretation",
            "The teacher is used only as an offline no-harm training guard. Final inference remains the S136d custom backbone, so the teacher must not be counted as an ensemble at test time.",
        )
    )

    payload = {
        "ok": not any(item.status == "fail" for item in checks),
        "screen_config": str(args.screen_config),
        "full_config": str(args.full_config),
        "teacher_screen_config": str(args.teacher_screen_config),
        "teacher_full_config": str(args.teacher_full_config),
        "watcher_script": str(args.watcher_script),
        "nodistill_screen_config": str(args.nodistill_screen_config) if args.nodistill_screen_config else None,
        "nodistill_screen_dir": str(args.nodistill_screen_dir) if args.nodistill_screen_dir else None,
        "screen_output_dir": str(screen_out),
        "full_output_dir": str(full_out),
        "full_manifest_counts": full_manifest_counts,
        "teacher_full_manifest_counts": teacher_full_manifest_counts,
        "param_count": int(param_count),
        "num_classes": int(len(class_to_idx)),
        "checks": [asdict(item) for item in checks],
        "active_losses": active_losses,
    }
    (args.output_dir / "s136d_queue_readiness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# S136d Queue Readiness And Fairness Audit",
        "",
        f"- Screen config: `{args.screen_config}`",
        f"- Full config: `{args.full_config}`",
        f"- Teacher screen config: `{args.teacher_screen_config}`",
        f"- Teacher full config: `{args.teacher_full_config}`",
        f"- Watcher: `{args.watcher_script}`",
        f"- No-distill control config: `{args.nodistill_screen_config}`",
        f"- No-distill control dir: `{args.nodistill_screen_dir}`",
        f"- OK: `{payload['ok']}`",
        f"- Trainable parameters: `{param_count}`",
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
            "S136d is a safe-distilled variant of the S136 custom early-coupling backbone. "
            "The extra teacher logits are not a late head and not an ensemble; they are an offline no-harm constraint that protects teacher-correct stable classes while leaving wet/water/concrete roughness focus boundaries less constrained.",
            "",
            "The screen remains a capped route-selection protocol against S96. A final paper-level claim requires full complete-manifest training, full comparison against S7, and the final promotion audit with public SOTA thresholds enabled.",
            "",
            "The no-distill S136 screen is treated as the same-budget mechanism control for S136d. If both results exist, the watcher compares S136d against S136 before full promotion so that any gain can be attributed to the task-adapted safe-distillation guard rather than only to the custom backbone capacity.",
            "",
            "The mechanism diagnosis report converts available S136/S136d screen/full results into an explicit next action: wait, promote full, weaken safe-distill, rebalance focus protection, revise early evidence experts, or prepare final evidence if full SOTA is reached.",
        ]
    )
    (args.output_dir / "s136d_queue_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "report": str(args.output_dir / "s136d_queue_readiness.md")}, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

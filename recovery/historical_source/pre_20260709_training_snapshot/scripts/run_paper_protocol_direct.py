from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "paper_protocol_queue"
SUMMARY_DIR = PROJECT_ROOT / "reports" / "paper_protocol_summary"

ABLATION = [
    "configs/experiments/paper_protocol/v0_global_only.yaml",
    "configs/experiments/paper_protocol/v1_physics_texture.yaml",
    "configs/experiments/paper_protocol/v2_friction_set.yaml",
    "configs/experiments/paper_protocol/v3_dg_losses.yaml",
    "configs/experiments/paper_protocol/v4_evidence_aux.yaml",
    "configs/experiments/paper_protocol/v5_full_faf.yaml",
]
CANDIDATES = [
    "configs/experiments/paper_protocol/v6_full_faf_fourier.yaml",
    "configs/experiments/paper_protocol/v7_full_faf_fourier_dann.yaml",
    "configs/experiments/paper_protocol/v8_full_faf_fourier_roadprior.yaml",
    "configs/experiments/paper_protocol/v9_full_faf_roadsaw_hard_sampling.yaml",
    "configs/experiments/paper_protocol/v10_full_faf_consistency.yaml",
    "configs/experiments/paper_protocol/v11_full_faf_domain_adapter.yaml",
    "configs/experiments/paper_protocol/v12_full_faf_roi_interval_safety.yaml",
    "configs/experiments/paper_protocol/v13_lean_physics_evidence.yaml",
    "configs/experiments/paper_protocol/v14_lean_road_roi_safety.yaml",
    "configs/experiments/paper_protocol/v15_lean_bottom_square_style_safety.yaml",
    "configs/experiments/paper_protocol/v16_lean_bottom_square_color_constancy_safety.yaml",
    "configs/experiments/paper_protocol/v17_lean_quality_physics_safety.yaml",
    "configs/experiments/paper_protocol/v18_lean_mixstyle_quality_safety.yaml",
    "configs/experiments/paper_protocol/v19_lean_state_contrast_quality_safety.yaml",
    "configs/experiments/paper_protocol/v20_lean_interval_order_quality_safety.yaml",
    "configs/experiments/paper_protocol/v21_lean_quality_uncertainty_safety.yaml",
    "configs/experiments/paper_protocol/v22_lean_quality_order_contrast_safety.yaml",
    "configs/experiments/paper_protocol/v23_lean_region_mixture_evidence_safety.yaml",
    "configs/experiments/paper_protocol/v24_lean_multi_query_region_evidence_safety.yaml",
    "configs/experiments/paper_protocol/v25_lean_masked_query_consistency_safety.yaml",
]
LODO = [
    "configs/experiments/paper_protocol/lodo_roadsaw_full_faf.yaml",
    "configs/experiments/paper_protocol/lodo_rscd_full_faf.yaml",
    "configs/experiments/paper_protocol/lodo_roadsc_full_faf.yaml",
]
SINGLE = [
    "configs/experiments/paper_protocol/single_roadsaw_full_faf.yaml",
    "configs/experiments/paper_protocol/single_rscd_full_faf.yaml",
    "configs/experiments/paper_protocol/single_roadsc_full_faf.yaml",
]
BASELINES = [
    "configs/experiments/paper_protocol/baseline_single_roadsaw_global_convnext.yaml",
    "configs/experiments/paper_protocol/baseline_single_rscd_global_convnext.yaml",
    "configs/experiments/paper_protocol/baseline_single_roadsc_global_convnext.yaml",
]
FINAL_LODO = [
    "configs/experiments/paper_protocol/final_lodo_roadsaw_lean_road_roi_safety.yaml",
    "configs/experiments/paper_protocol/final_lodo_rscd_lean_road_roi_safety.yaml",
    "configs/experiments/paper_protocol/final_lodo_roadsc_lean_road_roi_safety.yaml",
]
FINAL_SINGLE = [
    "configs/experiments/paper_protocol/final_single_roadsaw_lean_road_roi_safety.yaml",
    "configs/experiments/paper_protocol/final_single_rscd_lean_road_roi_safety.yaml",
    "configs/experiments/paper_protocol/final_single_roadsc_lean_road_roi_safety.yaml",
]

BASE_COMPLETE_ARTIFACTS = [
    "best.pt",
    "evaluate_test.json",
    "detailed_test.json",
    "interval_calibration_90.json",
    "bootstrap_metrics.json",
    "topvenue_result_audit.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["p0", "ablation", "lodo", "single", "baselines", "candidates", "final_lodo", "final_single", "final", "all"],
        default="p0",
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--keep-last-checkpoint", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--postprocess-each", action="store_true")
    parser.add_argument(
        "--candidate-policy",
        choices=["fail_fast", "legacy_all"],
        default="fail_fast",
        help=(
            "Candidate scheduling policy. fail_fast uses the latest fail-fast "
            "report to run only promoted candidates; legacy_all "
            "runs every configured v6-v25 candidate."
        ),
    )
    parser.add_argument(
        "--final-policy",
        choices=["defer_until_candidate_complete", "legacy_run"],
        default="defer_until_candidate_complete",
        help=(
            "For --phase all, defer final-method rows until at least one selected "
            "candidate has complete formal artifacts. Explicit --phase final* still runs."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional run/config stems to execute within the selected phase, e.g. v14_lean_road_roi_safety.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Optional run/config stems to skip within the selected phase.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    _prepare_env()

    if args.wait_pid:
        _wait_for_pid(args.wait_pid, args.log_dir)

    _run([str(args.python), "scripts/make_paper_protocol_configs.py"], args.log_dir, "make_configs")
    phase_configs = _phase_configs(
        args.phase,
        candidate_policy=args.candidate_policy,
        final_policy=args.final_policy,
        root=args.root,
        only=args.only,
    )
    if not phase_configs:
        _log(args.log_dir, f"{args.phase}: no configs selected by current policy; refreshing reports and exiting")
        if args.skip_postprocess:
            _refresh_lightweight_status(args.python, args.root, args.log_dir)
        else:
            _postprocess(args.python, args.root, args.log_dir)
        return
    for config in _filter_configs(phase_configs, only=args.only, exclude=args.exclude):
        _run_config(
            Path(config),
            python=args.python,
            log_dir=args.log_dir,
            force_train=args.force_train,
            keep_last_checkpoint=args.keep_last_checkpoint,
        )
        if args.skip_postprocess:
            _refresh_lightweight_status(args.python, args.root, args.log_dir)
        elif args.postprocess_each:
            _postprocess(args.python, args.root, args.log_dir)

    if not args.skip_postprocess:
        _postprocess(args.python, args.root, args.log_dir)


def _phase_configs(
    phase: str,
    *,
    candidate_policy: str = "fail_fast",
    final_policy: str = "defer_until_candidate_complete",
    root: Path = DEFAULT_ROOT,
    only: list[str] | None = None,
) -> list[str]:
    has_explicit_only = bool(_normalize_tokens(only))
    candidate_configs = CANDIDATES if has_explicit_only else _candidate_configs(candidate_policy)
    final_configs = FINAL_LODO + FINAL_SINGLE
    if phase == "ablation":
        return ABLATION
    if phase == "lodo":
        return LODO
    if phase == "single":
        return SINGLE
    if phase == "baselines":
        return BASELINES
    if phase == "candidates":
        return candidate_configs
    if phase == "final_lodo":
        return FINAL_LODO
    if phase == "final_single":
        return FINAL_SINGLE
    if phase == "final":
        return final_configs
    if phase == "all":
        # Evidence priority for a strict paper: first close the core P0 table,
        # then OOD LODO, then direct fair public-dataset comparisons. P1
        # candidates are important, but should not delay the proof that the
        # proposed method is better than matched ConvNeXt baselines.
        if (
            not has_explicit_only
            and final_policy == "defer_until_candidate_complete"
            and not _any_formal_candidate_complete(candidate_configs, root)
        ):
            final_configs = []
        return ABLATION + LODO + SINGLE + BASELINES + candidate_configs + final_configs
    return ABLATION + LODO


def _candidate_configs(candidate_policy: str) -> list[str]:
    if candidate_policy == "legacy_all":
        return CANDIDATES
    selected = _fail_fast_selected_candidate_stems()
    by_stem = {Path(config).stem: config for config in CANDIDATES}
    out = [by_stem[stem] for stem in selected if stem in by_stem]
    return out


def _fail_fast_selected_candidate_stems() -> list[str]:
    report_path = SUMMARY_DIR / "fail_fast_exploration_report.json"
    if not report_path.exists():
        return []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    policy = report.get("formal_policy") if isinstance(report, dict) else None
    selected = (policy or {}).get("promoted_or_fallback") if isinstance(policy, dict) else None
    if not isinstance(selected, list):
        return []
    return [str(item) for item in selected if str(item).strip()]


def _any_formal_candidate_complete(candidate_configs: list[str], root: Path) -> bool:
    for config in candidate_configs:
        if _run_complete(root / Path(config).stem):
            return True
    return False


def _filter_configs(configs: list[str], *, only: list[str] | None, exclude: list[str] | None) -> list[str]:
    out = list(configs)
    only_tokens = _normalize_tokens(only)
    exclude_tokens = _normalize_tokens(exclude)
    if only_tokens:
        selected = [config for config in out if _config_matches(config, only_tokens)]
        matched = {
            token
            for token in only_tokens
            if any(_single_token_matches(config, token) for config in out)
        }
        missing = sorted(only_tokens - matched)
        if missing:
            raise ValueError(f"--only did not match any configs in this phase: {', '.join(missing)}")
        out = selected
    if exclude_tokens:
        out = [config for config in out if not _config_matches(config, exclude_tokens)]
    if not out:
        raise ValueError("No configs selected after applying --only/--exclude")
    return out


def _normalize_tokens(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    out = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        path = Path(text)
        out.add(text)
        out.add(path.stem)
        out.add(path.name)
    return out


def _config_matches(config: str, tokens: set[str]) -> bool:
    return any(_single_token_matches(config, token) for token in tokens)


def _single_token_matches(config: str, token: str) -> bool:
    path = Path(config)
    return token in {config, path.stem, path.name, str(path)}


def _run_config(
    config: Path,
    *,
    python: Path,
    log_dir: Path,
    force_train: bool,
    keep_last_checkpoint: bool,
) -> None:
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")
    cfg = load_yaml(config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _safe_cleanup_completed_last(output_dir.parent)

    if force_train:
        _remove_stale_artifacts(output_dir)

    best = output_dir / "best.pt"
    last = output_dir / "last.pt"
    if not force_train and not _training_complete(output_dir, cfg):
        train_cmd = [str(python), "-u", "scripts/train.py", "--config", str(config)]
        if last.exists():
            train_cmd.extend(["--resume", str(last)])
        _run(train_cmd, log_dir, f"{config.stem}_train")
    elif not best.exists():
        raise RuntimeError(f"Training marked complete but best checkpoint missing: {best}")

    if not best.exists():
        raise RuntimeError(f"Checkpoint not found after training: {best}")

    if _run_complete(output_dir) and not force_train:
        _log(log_dir, f"{config.stem}: complete artifacts already present; skipping eval pipeline")
    else:
        _run_eval_pipeline(config, output_dir, best, python, log_dir, skip_dataset_diagnostic=_is_single_or_baseline(config))

    if not keep_last_checkpoint and _run_complete(output_dir) and last.exists():
        last.unlink()
        _log(log_dir, f"{config.stem}: removed completed-run resume checkpoint {last}")


def _run_eval_pipeline(
    config: Path,
    output_dir: Path,
    best: Path,
    python: Path,
    log_dir: Path,
    *,
    skip_dataset_diagnostic: bool,
) -> None:
    config_json = output_dir / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"Run config not found after training: {config_json}")
    run_cfg = json.loads(config_json.read_text(encoding="utf-8"))

    manifest_args = []
    for manifest in run_cfg["data"]["train_manifests"]:
        manifest_args.extend(["--manifest", manifest])
    _run(
        [
            str(python),
            "-u",
            "scripts/manifest_stats.py",
            *manifest_args,
            "--out",
            str(output_dir / "manifest_stats_train.json"),
        ],
        log_dir,
        f"{config.stem}_manifest_stats",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/evaluate.py",
            "--config",
            str(config),
            "--checkpoint",
            str(best),
            "--split",
            "test",
            "--out",
            str(output_dir / "evaluate_test.json"),
        ],
        log_dir,
        f"{config.stem}_evaluate_test",
    )
    detailed = output_dir / "detailed_test.json"
    _run(
        [
            str(python),
            "-u",
            "scripts/evaluate_detailed.py",
            "--config",
            str(config),
            "--checkpoint",
            str(best),
            "--split",
            "test",
            "--out",
            str(detailed),
        ],
        log_dir,
        f"{config.stem}_detailed_test",
    )
    _confusion(python, log_dir, detailed, output_dir, "friction")
    _confusion(python, log_dir, detailed, output_dir, "risk")
    if detailed.exists() and '"roadsaw"' in detailed.read_text(encoding="utf-8", errors="ignore"):
        _confusion(python, log_dir, detailed, output_dir, "friction", dataset="roadsaw")
        _confusion(python, log_dir, detailed, output_dir, "risk", dataset="roadsaw")

    _run(
        [
            str(python),
            "-u",
            "scripts/calibrate_intervals.py",
            "--config",
            str(config),
            "--checkpoint",
            str(best),
            "--target-coverage",
            "0.90",
            "--out",
            str(output_dir / "interval_calibration_90.json"),
        ],
        log_dir,
        f"{config.stem}_calibrate_intervals",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/bootstrap_metrics.py",
            "--config",
            str(config),
            "--checkpoint",
            str(best),
            "--split",
            "test",
            "--target-coverage",
            "0.90",
            "--num-bootstrap",
            "500",
            "--out-json",
            str(output_dir / "bootstrap_metrics.json"),
            "--out-md",
            str(output_dir / "bootstrap_metrics.md"),
        ],
        log_dir,
        f"{config.stem}_bootstrap_metrics",
    )
    if not skip_dataset_diagnostic:
        _run(
            [
                str(python),
                "-u",
                "scripts/dataset_id_diagnostic.py",
                "--config",
                str(config),
                "--checkpoint",
                str(best),
                "--max-samples",
                "5000",
                "--out",
                str(output_dir / "dataset_id_diagnostic.json"),
            ],
            log_dir,
            f"{config.stem}_dataset_id_diagnostic",
        )
    if run_cfg.get("model", {}).get("use_evidence_field"):
        _run(
            [
                str(python),
                "-u",
                "scripts/export_evidence_maps.py",
                "--config",
                str(config),
                "--checkpoint",
                str(best),
                "--split",
                "test",
                "--out-dir",
                str(output_dir / "evidence_maps"),
                "--max-samples",
                "24",
                "--selection",
                "mixed",
                "--clean",
            ],
            log_dir,
            f"{config.stem}_evidence_maps",
        )
        _run(
            [
                str(python),
                "-u",
                "scripts/analyze_evidence_field.py",
                "--config",
                str(config),
                "--checkpoint",
                str(best),
                "--split",
                "test",
                "--max-samples",
                "3000",
                "--out-json",
                str(output_dir / "evidence_field_audit.json"),
                "--out-md",
                str(output_dir / "evidence_field_audit.md"),
            ],
            log_dir,
            f"{config.stem}_evidence_field_audit",
        )
    _run(
        [
            str(python),
            "-u",
            "scripts/audit_topvenue_results.py",
            "--output-dir",
            str(output_dir),
            "--out-md",
            str(output_dir / "topvenue_result_audit.md"),
            "--out-json",
            str(output_dir / "topvenue_result_audit.json"),
        ],
        log_dir,
        f"{config.stem}_audit",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/slim_best_checkpoints.py",
            "--root",
            str(output_dir.parent),
            "--apply",
        ],
        log_dir,
        f"{config.stem}_slim_best_checkpoints",
    )


def _confusion(
    python: Path,
    log_dir: Path,
    detailed: Path,
    output_dir: Path,
    task: str,
    *,
    dataset: str | None = None,
) -> None:
    suffix = f"_{dataset}" if dataset else "_overall"
    cmd = [
        str(python),
        "-u",
        "scripts/summarize_confusions.py",
        "--detailed",
        str(detailed),
        "--task",
        task,
        "--out-csv",
        str(output_dir / f"confusion_{task}{suffix}.csv"),
        "--out-md",
        str(output_dir / f"confusion_{task}{suffix}.md"),
    ]
    if dataset:
        cmd.extend(["--dataset", dataset])
    _run(cmd, log_dir, f"{output_dir.name}_confusion_{task}{suffix}")


def _postprocess(python: Path, root: Path, log_dir: Path) -> None:
    _run(
        [
            str(python),
            "-u",
            "scripts/postprocess_protocol_outputs.py",
            "--root",
            str(root),
            "--summary-dir",
            "reports/paper_protocol_summary",
        ],
        log_dir,
        "postprocess_protocol_outputs",
    )


def _refresh_lightweight_status(python: Path, root: Path, log_dir: Path) -> None:
    summary = SUMMARY_DIR
    summary.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(python),
            "-u",
            "scripts/write_queue_recovery_report.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
            "--log-dir",
            str(log_dir),
            "--out-md",
            str(summary / "queue_recovery_report.md"),
            "--out-json",
            str(summary / "queue_recovery_report.json"),
        ],
        log_dir,
        "refresh_queue_recovery",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_runtime_guard_report.py",
            "--summary-dir",
            str(summary),
        ],
        log_dir,
        "refresh_runtime_guard",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_active_training_watch_report.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
            "--log-dir",
            str(log_dir),
        ],
        log_dir,
        "refresh_active_training_watch",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_active_live_training_reports.py",
            "--summary-dir",
            str(summary),
            "--log-dir",
            str(log_dir),
        ],
        log_dir,
        "refresh_active_live_training_reports",
    )
    _run(
        [str(python), "-u", "scripts/write_gpu_scheduling_guard_report.py"],
        log_dir,
        "refresh_gpu_scheduling_guard",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_followup_watcher_report.py",
            "--summary-dir",
            str(summary),
        ],
        log_dir,
        "refresh_followup_watcher_report",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/check_protocol_completeness.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
        ],
        log_dir,
        "refresh_protocol_completeness",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_artifact_contract_report.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
            "--config-dir",
            "configs/experiments/paper_protocol",
            "--log-dir",
            str(log_dir),
        ],
        log_dir,
        "refresh_artifact_contract",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_quality_mondrian_summary.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
        ],
        log_dir,
        "refresh_quality_mondrian_summary",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_asymmetric_mondrian_summary.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
        ],
        log_dir,
        "refresh_asymmetric_mondrian_summary",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_region_mixture_summary.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
        ],
        log_dir,
        "refresh_region_mixture_summary",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_checkpoint_divergence_report.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
        ],
        log_dir,
        "refresh_checkpoint_divergence",
    )
    _run(
        [str(python), "-u", "scripts/topvenue_readiness_gate.py", "--summary-dir", str(summary)],
        log_dir,
        "refresh_topvenue_readiness",
    )
    _run(
        [
            str(python),
            "-u",
            "scripts/write_experiment_dashboard.py",
            "--root",
            str(root),
            "--summary-dir",
            str(summary),
            "--log-dir",
            str(log_dir),
            "--out-json",
            str(summary / "experiment_status_dashboard.json"),
            "--out-md",
            str(summary / "experiment_status_dashboard.md"),
        ],
        log_dir,
        "refresh_experiment_dashboard",
    )
    shutil.copyfile(summary / "experiment_status_dashboard.json", summary / "experiment_dashboard.json")
    shutil.copyfile(summary / "experiment_status_dashboard.md", summary / "experiment_dashboard.md")
    _run(
        [str(python), "-u", "scripts/write_current_remaining_reports.py", "--summary-dir", str(summary)],
        log_dir,
        "refresh_current_remaining_reports",
    )
    _run(
        [str(python), "-u", "scripts/write_live_research_route_update.py", "--summary-dir", str(summary)],
        log_dir,
        "refresh_live_research_route_update",
    )
    _run(
        [str(python), "-u", "scripts/write_objective_completion_audit.py", "--summary-dir", str(summary)],
        log_dir,
        "refresh_objective_completion_audit",
    )
    _run(
        [str(python), "-u", "scripts/audit_live_route_update_automation.py", "--summary-dir", str(summary)],
        log_dir,
        "refresh_live_route_update_automation_audit",
    )


def _run(cmd: list[str], log_dir: Path, name: str) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = log_dir / f"{name}_{stamp}.out.log"
    err = log_dir / f"{name}_{stamp}.err.log"
    _log(log_dir, f"RUN {name}: {' '.join(cmd)}")
    with out.open("w", encoding="utf-8", errors="replace") as fout, err.open(
        "w", encoding="utf-8", errors="replace"
    ) as ferr:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fout, stderr=ferr)
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {name} exit={proc.returncode} out={out} err={err}")


def _prepare_env() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TORCH_HOME", r"D:\NMI_SPWFM_datasets\torch_cache")
    os.environ.setdefault("TEMP", r"D:\NMI_SPWFM_datasets\tmp")
    os.environ.setdefault("TMP", r"D:\NMI_SPWFM_datasets\tmp")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)


def _wait_for_pid(pid: int, log_dir: Path) -> None:
    _log(log_dir, f"waiting for pid={pid}")
    while _pid_exists(pid):
        time.sleep(30.0)
    _log(log_dir, f"pid={pid} finished or no longer exists")


def _pid_exists(pid: int) -> bool:
    if os.name != "nt":
        return Path(f"/proc/{pid}").exists()
    proc = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) in proc.stdout


def _training_complete(output_dir: Path, cfg: dict) -> bool:
    state_path = output_dir / "training_state.json"
    best = output_dir / "best.pt"
    if not state_path.exists() or not best.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    optim = cfg.get("optim", {})
    epochs = int(optim.get("epochs", state.get("epochs", 0) or 0))
    patience = optim.get("early_stop_patience")
    reached_epochs = int(state.get("epoch", 0)) >= epochs
    reached_patience = patience is not None and int(state.get("stale_epochs", 0)) >= int(patience)
    return reached_epochs or reached_patience


def _run_complete(output_dir: Path) -> bool:
    missing = _missing_completion_artifacts(output_dir)
    return not missing


def _missing_completion_artifacts(output_dir: Path) -> list[str]:
    missing = [artifact for artifact in BASE_COMPLETE_ARTIFACTS if not (output_dir / artifact).exists()]
    config = _load_run_config(output_dir)
    if not _is_single_or_baseline_name(output_dir.name) and not (output_dir / "dataset_id_diagnostic.json").exists():
        missing.append("dataset_id_diagnostic.json")
    if config is None:
        missing.append("config.json")
    elif config.get("model", {}).get("use_evidence_field"):
        if not (output_dir / "evidence_maps").exists():
            missing.append("evidence_maps")
        for artifact in ["evidence_field_audit.json", "evidence_field_audit.md"]:
            if not (output_dir / artifact).exists():
                missing.append(artifact)
    return missing


def _is_single_or_baseline(config: Path) -> bool:
    return _is_single_or_baseline_name(config.stem)


def _is_single_or_baseline_name(name: str) -> bool:
    return name.startswith("single_") or name.startswith("baseline_single_") or name.startswith("final_single_")


def _load_run_config(output_dir: Path) -> dict | None:
    path = output_dir / "config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _safe_cleanup_completed_last(root: Path) -> None:
    if not root.exists():
        return
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        if _run_complete(run_dir):
            last = run_dir / "last.pt"
            if last.exists():
                last.unlink()


def _remove_stale_artifacts(output_dir: Path) -> None:
    for name in [
        "best.pt",
        "best_safety.pt",
        "last.pt",
        "training_state.json",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "bootstrap_metrics.md",
        "dataset_id_diagnostic.json",
        "topvenue_result_audit.json",
        "topvenue_result_audit.md",
        "evidence_field_audit.json",
        "evidence_field_audit.md",
        "manifest_stats_train.json",
        "confusion_friction_overall.csv",
        "confusion_friction_overall.md",
        "confusion_risk_overall.csv",
        "confusion_risk_overall.md",
        "confusion_friction_roadsaw.csv",
        "confusion_friction_roadsaw.md",
        "confusion_risk_roadsaw.csv",
        "confusion_risk_roadsaw.md",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for dirname in ["tb", "evidence_maps"]:
        path = output_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def _log(log_dir: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (log_dir / "run_paper_protocol_direct.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()

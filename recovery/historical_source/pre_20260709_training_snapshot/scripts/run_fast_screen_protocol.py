from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments" / "fast_screen"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "fast_screen_queue"

CANDIDATE_RUNS = [
    "screen_v1_physics_texture",
    "screen_v6_full_faf_fourier",
    "screen_v7_full_faf_fourier_dann",
    "screen_v8_full_faf_fourier_roadprior",
    "screen_v9_full_faf_roadsaw_hard_sampling",
    "screen_v10_full_faf_consistency",
    "screen_v11_full_faf_domain_adapter",
    "screen_v12_full_faf_roi_interval_safety",
    "screen_v13_lean_physics_evidence",
    "screen_v14_lean_road_roi_safety",
    "screen_v15_lean_bottom_square_style_safety",
    "screen_v16_lean_bottom_square_color_constancy_safety",
    "screen_v17_lean_quality_physics_safety",
    "screen_v18_lean_mixstyle_quality_safety",
    "screen_v19_lean_state_contrast_quality_safety",
    "screen_v20_lean_interval_order_quality_safety",
    "screen_v21_lean_quality_uncertainty_safety",
    "screen_v22_lean_quality_order_contrast_safety",
    "screen_v23_lean_region_mixture_evidence_safety",
    "screen_v24_lean_multi_query_region_evidence_safety",
    "screen_v25_lean_masked_query_consistency_safety",
]

ROADSAW_RUNS = [
    "screen_lodo_roadsaw_full_faf",
    "screen_final_lodo_roadsaw_lean_road_roi_safety",
    "screen_single_roadsaw_full_faf",
    "screen_baseline_single_roadsaw_global_convnext",
]

LEAN_FIRST_WAVE = [
    "screen_v1_physics_texture",
    "screen_v14_lean_road_roi_safety",
    "screen_v17_lean_quality_physics_safety",
    "screen_v18_lean_mixstyle_quality_safety",
    "screen_v19_lean_state_contrast_quality_safety",
    "screen_v20_lean_interval_order_quality_safety",
    "screen_v21_lean_quality_uncertainty_safety",
    "screen_v22_lean_quality_order_contrast_safety",
    "screen_v23_lean_region_mixture_evidence_safety",
    "screen_v24_lean_multi_query_region_evidence_safety",
    "screen_v25_lean_masked_query_consistency_safety",
    "screen_v16_lean_bottom_square_color_constancy_safety",
    "screen_v15_lean_bottom_square_style_safety",
    "screen_v13_lean_physics_evidence",
]

COMPLETE_ARTIFACTS = [
    "best.pt",
    "evaluate_test.json",
    "detailed_test.json",
    "interval_calibration_90.json",
    "bootstrap_metrics.json",
    "topvenue_result_audit.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["candidates", "roadsaw", "all"], default="candidates")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--keep-last-checkpoint", action="store_true")
    parser.add_argument("--allow-concurrent-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "Optional screen/source run stems to execute, e.g. screen_v14_lean_road_roi_safety "
            "or v14_lean_road_roi_safety."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Optional screen/source run stems to skip within the selected scope.",
    )
    parser.add_argument(
        "--lean-first-wave",
        action="store_true",
        help="Shortcut for the fail-fast first wave: v1 PhysicsTexture anchor plus lean v14/v17/v18/v19/v20/v21/v22/v23/v16/v15/v13.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--dataset-diagnostic-samples", type=int, default=2000)
    parser.add_argument("--evidence-map-samples", type=int, default=12)
    parser.add_argument("--evidence-audit-samples", type=int, default=1000)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    _prepare_env()

    if args.wait_pid:
        _wait_for_pid(args.wait_pid, args.log_dir)
    if not args.allow_concurrent_train and not args.dry_run:
        _assert_no_other_train_process()

    _run(
        [str(args.python), "scripts/make_fast_screen_configs.py", "--scope", args.scope],
        args.log_dir,
        "make_fast_screen_configs",
        dry_run=args.dry_run,
    )

    for config in _filter_configs(
        _config_paths(args.scope, args.config_dir, lean_first_wave=args.lean_first_wave),
        only=args.only,
        exclude=args.exclude,
    ):
        _run_config(
            config,
            python=args.python,
            log_dir=args.log_dir,
            force_train=args.force_train,
            keep_last_checkpoint=args.keep_last_checkpoint,
            dry_run=args.dry_run,
            bootstrap_samples=max(1, int(args.bootstrap_samples)),
            dataset_diagnostic_samples=max(1, int(args.dataset_diagnostic_samples)),
            evidence_map_samples=max(1, int(args.evidence_map_samples)),
            evidence_audit_samples=max(1, int(args.evidence_audit_samples)),
        )
        _run(
            [
                str(args.python),
                "scripts/write_fast_screen_status_report.py",
                "--config-dir",
                str(args.config_dir),
                "--log-dir",
                str(args.log_dir),
            ],
            args.log_dir,
            "write_fast_screen_status_report",
            dry_run=args.dry_run,
        )


def _config_paths(scope: str, config_dir: Path, *, lean_first_wave: bool = False) -> list[Path]:
    if lean_first_wave:
        names = [name for name in LEAN_FIRST_WAVE if scope in {"candidates", "all"}]
    elif scope == "candidates":
        names = CANDIDATE_RUNS
    elif scope == "roadsaw":
        names = ROADSAW_RUNS
    else:
        names = CANDIDATE_RUNS + ROADSAW_RUNS
    return [config_dir / f"{name}.yaml" for name in names]


def _filter_configs(configs: list[Path], *, only: list[str] | None, exclude: list[str] | None) -> list[Path]:
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
            raise ValueError(f"--only did not match any fast-screen configs: {', '.join(missing)}")
        out = selected
    if exclude_tokens:
        out = [config for config in out if not _config_matches(config, exclude_tokens)]
    if not out:
        raise ValueError("No fast-screen configs selected after applying --only/--exclude")
    return out


def _normalize_tokens(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        path = Path(text)
        candidates = {text, path.stem, path.name}
        if path.suffix == ".yaml":
            candidates.add(path.stem)
        for item in list(candidates):
            if item.startswith("screen_"):
                candidates.add(item.removeprefix("screen_"))
            else:
                candidates.add(f"screen_{item}")
        out.update(candidates)
    return out


def _config_matches(config: Path, tokens: set[str]) -> bool:
    return any(_single_token_matches(config, token) for token in tokens)


def _single_token_matches(config: Path, token: str) -> bool:
    source = config.stem.removeprefix("screen_")
    return token in {
        str(config),
        config.name,
        config.stem,
        source,
        f"{source}.yaml",
        f"screen_{source}",
        f"screen_{source}.yaml",
    }


def _run_config(
    config: Path,
    *,
    python: Path,
    log_dir: Path,
    force_train: bool,
    keep_last_checkpoint: bool,
    dry_run: bool,
    bootstrap_samples: int,
    dataset_diagnostic_samples: int,
    evidence_map_samples: int,
    evidence_audit_samples: int,
) -> None:
    if not config.exists() and not dry_run:
        raise FileNotFoundError(f"Fast-screen config not found: {config}")
    cfg = load_yaml(config) if config.exists() else {"output_dir": "DRY_RUN"}
    output_dir = Path(cfg["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if force_train and not dry_run:
        _remove_stale_artifacts(output_dir)

    best = output_dir / "best.pt"
    last = output_dir / "last.pt"
    if force_train or not _training_complete(output_dir, cfg):
        cmd = [str(python), "-u", "scripts/train.py", "--config", str(config)]
        if last.exists() and not force_train:
            cmd.extend(["--resume", str(last)])
        _run(cmd, log_dir, f"{config.stem}_train", dry_run=dry_run)
    elif not best.exists() and not dry_run:
        raise RuntimeError(f"Training marked complete but best checkpoint missing: {best}")

    if dry_run:
        return
    if not best.exists():
        raise RuntimeError(f"Checkpoint not found after fast-screen training: {best}")
    if _run_complete(output_dir) and not force_train:
        _log(log_dir, f"{config.stem}: complete fast-screen artifacts already present; skipping eval")
    else:
        _run_eval_pipeline(
            config,
            output_dir,
            best,
            python,
            log_dir,
            bootstrap_samples=bootstrap_samples,
            dataset_diagnostic_samples=dataset_diagnostic_samples,
            evidence_map_samples=evidence_map_samples,
            evidence_audit_samples=evidence_audit_samples,
        )
    if not keep_last_checkpoint and _run_complete(output_dir) and last.exists():
        last.unlink()
        _log(log_dir, f"{config.stem}: removed completed fast-screen resume checkpoint {last}")


def _run_eval_pipeline(
    config: Path,
    output_dir: Path,
    best: Path,
    python: Path,
    log_dir: Path,
    *,
    bootstrap_samples: int,
    dataset_diagnostic_samples: int,
    evidence_map_samples: int,
    evidence_audit_samples: int,
) -> None:
    run_cfg = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
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
            str(bootstrap_samples),
            "--out-json",
            str(output_dir / "bootstrap_metrics.json"),
            "--out-md",
            str(output_dir / "bootstrap_metrics.md"),
        ],
        log_dir,
        f"{config.stem}_bootstrap_metrics",
    )
    if not _is_single_or_baseline_name(output_dir.name):
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
                str(dataset_diagnostic_samples),
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
                str(evidence_map_samples),
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
                str(evidence_audit_samples),
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


def _training_complete(output_dir: Path, cfg: dict) -> bool:
    state_path = output_dir / "training_state.json"
    best = output_dir / "best.pt"
    if not state_path.exists() or not best.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    epochs = int(cfg.get("optim", {}).get("epochs", state.get("epochs", 0) or 0))
    patience = cfg.get("optim", {}).get("early_stop_patience")
    reached_epochs = int(state.get("epoch", 0)) >= epochs
    reached_patience = patience is not None and int(state.get("stale_epochs", 0)) >= int(patience)
    return reached_epochs or reached_patience


def _run_complete(output_dir: Path) -> bool:
    missing = [name for name in COMPLETE_ARTIFACTS if not (output_dir / name).exists()]
    if not _is_single_or_baseline_name(output_dir.name) and not (output_dir / "dataset_id_diagnostic.json").exists():
        missing.append("dataset_id_diagnostic.json")
    config_json = output_dir / "config.json"
    if not config_json.exists():
        missing.append("config.json")
    else:
        cfg = json.loads(config_json.read_text(encoding="utf-8"))
        if cfg.get("model", {}).get("use_evidence_field"):
            for name in ["evidence_maps", "evidence_field_audit.json", "evidence_field_audit.md"]:
                if not (output_dir / name).exists():
                    missing.append(name)
    return not missing


def _remove_stale_artifacts(output_dir: Path) -> None:
    for name in [
        "best.pt",
        "best_safety.pt",
        "last.pt",
        "training_state.json",
        "metrics_history.json",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "bootstrap_metrics.md",
        "dataset_id_diagnostic.json",
        "topvenue_result_audit.json",
        "topvenue_result_audit.md",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def _is_single_or_baseline_name(name: str) -> bool:
    base = name.removeprefix("screen_")
    return base.startswith("single_") or base.startswith("baseline_single_") or base.startswith("final_single_")


def _assert_no_other_train_process() -> None:
    if os.name != "nt":
        return
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -like '*scripts/train.py*' -and $_.ProcessId -ne $PID } | "
            "Select-Object -ExpandProperty ProcessId",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = [line.strip() for line in proc.stdout.splitlines() if line.strip().isdigit()]
    if pids:
        raise RuntimeError(
            "Refusing to start fast-screen training while another train.py process is active: "
            + ", ".join(pids)
            + ". Re-run with --allow-concurrent-train only after manually verifying VRAM headroom."
        )


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


def _run(cmd: list[str], log_dir: Path, name: str, *, dry_run: bool = False) -> None:
    _log(log_dir, f"RUN {name}: {' '.join(cmd)}")
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = log_dir / f"{name}_{stamp}.out.log"
    err = log_dir / f"{name}_{stamp}.err.log"
    with out.open("w", encoding="utf-8", errors="replace") as fout, err.open(
        "w", encoding="utf-8", errors="replace"
    ) as ferr:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fout, stderr=ferr)
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {name} exit={proc.returncode} out={out} err={err}")


def _log(log_dir: Path, message: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with (log_dir / "fast_screen_protocol.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()

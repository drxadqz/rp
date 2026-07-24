from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from friction_affordance.c3_experiment import load_config  # noqa: E402


DEFAULT_FILES = [
    "train.py",
    "src/friction_affordance/c3_experiment.py",
    "src/friction_affordance/models/c3_farnet.py",
    "scripts/decide_s135c_screen_promotion.py",
    "scripts/audit_s135c_queue_readiness.py",
    "scripts/audit_s135c_stem_activation.py",
    "scripts/summarize_s135c_activation_risk.py",
    "scripts/compare_rscd_runs.py",
    "scripts/audit_rscd_run.py",
    "scripts/analyze_rscd_physics_cues.py",
    "scripts/synthesize_physics_cue_evidence.py",
    "scripts/summarize_rscd_results.py",
    "scripts/rscd_pipeline_status.py",
    "scripts/run_s135_after_s133c.ps1",
    "scripts/snapshot_rscd_candidate.py",
    "scripts/diagnose_candidate_route.py",
    "scripts/verify_candidate_integrity.py",
    "scripts/analyze_sota_gap_budget.py",
]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path = ROOT, timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {"ok": False, "error": repr(exc), "stdout": "", "stderr": ""}


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _file_record(path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists and path.is_file() else None
    return {
        "path": _relative(path),
        "absolute_path": str(path),
        "exists": exists,
        "size_bytes": int(stat.st_size) if stat else None,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds") if stat else None,
        "sha256": _sha256(path),
    }


def _resolve_config_chain(config_path: Path) -> list[Path]:
    chain: list[Path] = []
    seen: set[Path] = set()
    current = config_path.resolve()
    while current.exists() and current not in seen:
        seen.add(current)
        chain.append(current)
        parent: Path | None = None
        for line in current.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("extends:"):
                raw = stripped.split(":", 1)[1].strip().strip("'\"")
                parent = (current.parent / raw).resolve()
                break
        if parent is None:
            break
        current = parent
    return chain


def _config_summary(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    train = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    model = cfg.get("model", {})
    loss = cfg.get("loss", {})
    data = cfg.get("data", {})
    return {
        "path": str(config_path),
        "chain": [_file_record(p) for p in _resolve_config_chain(config_path)],
        "output_dir": cfg.get("output_dir"),
        "data": {
            "train_manifest": data.get("train_manifest"),
            "val_manifest": data.get("val_manifest"),
            "test_manifest": data.get("test_manifest"),
        },
        "train": {
            "resume_from": train.get("resume_from"),
            "teacher_checkpoint": train.get("teacher_checkpoint"),
            "trainable_prefixes": train.get("trainable_prefixes"),
            "batch_size": train.get("batch_size"),
            "grad_accum_steps": train.get("grad_accum_steps"),
            "num_workers": train.get("num_workers"),
            "max_train_samples_per_class": train.get("max_train_samples_per_class"),
            "samples_per_epoch": train.get("samples_per_epoch"),
            "factor_graph_pair_sampling_seed": train.get("factor_graph_pair_sampling_seed"),
            "evaluate_initial": train.get("evaluate_initial"),
        },
        "eval": {
            "batch_size": eval_cfg.get("batch_size"),
            "max_val_samples_per_class": eval_cfg.get("max_val_samples_per_class"),
            "max_test_samples_per_class": eval_cfg.get("max_test_samples_per_class"),
        },
        "model": {
            "backbone": model.get("backbone"),
            "head_type": model.get("head_type"),
            "use_physics_texture": model.get("use_physics_texture", model.get("use_physics_branch")),
            "use_physics_texture_stem_adapter": model.get("use_physics_texture_stem_adapter"),
            "use_scale_space_roughness_stem_adapter": model.get("use_scale_space_roughness_stem_adapter"),
            "use_pair_value_stem_conditioner": model.get("use_pair_value_stem_conditioner"),
            "use_feature_value_boundary_corrector": model.get("use_feature_value_boundary_corrector"),
            "use_local_physics_field": model.get(
                "use_local_physics_field", model.get("use_local_physics_field_branch")
            ),
            "use_semantic_physics_attention": model.get(
                "use_semantic_physics_attention", model.get("use_semantic_physics_attention_branch")
            ),
            "use_water_concrete_topology_texture_stem_conditioner": model.get(
                "use_water_concrete_topology_texture_stem_conditioner"
            ),
        },
        "loss": loss,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metric_value(metrics: dict[str, Any] | None, *names: str) -> float | None:
    if not metrics:
        return None
    if isinstance(metrics.get("summary"), dict):
        value = _metric_value(metrics["summary"], *names)
        if value is not None:
            return value
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _worst_class(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "per_class_metrics.csv"
    if not path.exists():
        return None
    candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f1_raw = row.get("f1") or row.get("F1") or row.get("macro_f1") or row.get("Macro-F1")
            if f1_raw is None:
                continue
            try:
                f1 = float(f1_raw)
            except ValueError:
                continue
            candidates.append(
                {
                    "class": row.get("class") or row.get("label") or row.get("name"),
                    "f1": f1,
                    "precision": row.get("precision") or row.get("Precision"),
                    "recall": row.get("recall") or row.get("Recall"),
                    "support": row.get("support") or row.get("Support"),
                }
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["f1"])


def _run_summary(name: str, run_dir: Path) -> dict[str, Any]:
    metrics = _read_json(run_dir / "test_metrics.json")
    summary = metrics.get("summary", metrics) if metrics else None
    return {
        "name": name,
        "run_dir": str(run_dir),
        "metrics_exists": metrics is not None,
        "top1": _metric_value(metrics, "top1", "accuracy", "acc", "top_1"),
        "macro_f1": _metric_value(metrics, "macro_f1", "mean_f1", "mean_f1/f1", "f1"),
        "num_samples": summary.get("num_samples") if summary else None,
        "worst_class": _worst_class(run_dir),
        "best_checkpoint": _file_record(run_dir / "best_checkpoint.pth"),
        "predictions_test": _file_record(run_dir / "predictions_test.csv"),
        "test_metrics": metrics,
    }


def _disk_summary(paths: list[Path]) -> dict[str, Any]:
    drives: dict[str, dict[str, Any]] = {}
    for path in paths:
        drive = path.drive
        if not drive or drive in drives:
            continue
        try:
            usage = os.statvfs(str(path))  # type: ignore[attr-defined]
            free = usage.f_bavail * usage.f_frsize
            total = usage.f_blocks * usage.f_frsize
        except Exception:
            try:
                import shutil

                usage2 = shutil.disk_usage(drive + "\\")
                free = usage2.free
                total = usage2.total
            except Exception:
                continue
        drives[drive] = {
            "free_gb": round(free / (1024**3), 3),
            "total_gb": round(total / (1024**3), 3),
            "used_gb": round((total - free) / (1024**3), 3),
        }
    return drives


def _environment_summary() -> dict[str, Any]:
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"error": repr(exc)}
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_info,
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        ),
    }


def _git_summary() -> dict[str, Any]:
    git_root = ROOT
    for candidate in [ROOT, *ROOT.parents]:
        if (candidate / ".git").exists():
            git_root = candidate
            break
    return {
        "git_root_candidate": str(git_root),
        "git_dir_exists": (git_root / ".git").exists(),
        "rev_parse_head": _run(["git", "rev-parse", "HEAD"], cwd=git_root),
        "status_short": _run(["git", "status", "--short"], cwd=git_root),
        "diff_stat": _run(["git", "diff", "--stat"], cwd=git_root),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    def pct(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value * 100:.3f}%"

    lines: list[str] = []
    lines.append("# RSCD Candidate Reproducibility Snapshot")
    lines.append("")
    lines.append(f"- Candidate: `{payload['candidate_name']}`")
    lines.append(f"- Created: `{payload['created_at']}`")
    lines.append(f"- Project root: `{payload['project_root']}`")
    lines.append("")
    lines.append("## Fair-Comparison Intent")
    lines.append("")
    lines.append(
        "This snapshot freezes the code, configs, run directories, baseline metrics, and environment used before launching or evaluating the candidate. "
        "It is meant to prevent accidental comparison against a moving implementation."
    )
    lines.append("")
    lines.append("## Config Protocol")
    for key in ["screen_config", "full_config"]:
        cfg = payload[key]
        train = cfg["train"]
        eval_cfg = cfg["eval"]
        lines.append("")
        lines.append(f"### {key}")
        lines.append(f"- Path: `{cfg['path']}`")
        lines.append(f"- Output: `{cfg['output_dir']}`")
        lines.append(f"- Resume: `{train.get('resume_from')}`")
        lines.append(f"- Teacher: `{train.get('teacher_checkpoint')}`")
        lines.append(f"- Trainable prefixes: `{train.get('trainable_prefixes')}`")
        lines.append(
            "- Budget: "
            f"batch `{train.get('batch_size')}`, accum `{train.get('grad_accum_steps')}`, "
            f"train cap `{train.get('max_train_samples_per_class')}`, "
            f"samples/epoch `{train.get('samples_per_epoch')}`, "
            f"eval caps val/test `{eval_cfg.get('max_val_samples_per_class')}`/`{eval_cfg.get('max_test_samples_per_class')}`"
        )
        lines.append(
            "- Mechanism flags: "
            f"PhysicsTexture `{cfg['model'].get('use_physics_texture')}`, "
            f"LocalPhysicsField `{cfg['model'].get('use_local_physics_field')}`, "
            f"SemanticPhysicsAttention `{cfg['model'].get('use_semantic_physics_attention')}`, "
            f"water-concrete stem `{cfg['model'].get('use_water_concrete_topology_texture_stem_conditioner')}`"
        )
    lines.append("")
    lines.append("## Current Metric Anchors")
    lines.append("")
    lines.append("| Run | Samples | Top-1 | Macro-F1 | Worst class | Worst F1 |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for run in payload["runs"]:
        worst = run.get("worst_class") or {}
        lines.append(
            f"| {run['name']} | {run.get('num_samples') or '-'} | {pct(run.get('top1'))} | "
            f"{pct(run.get('macro_f1'))} | {worst.get('class') or '-'} | {pct(worst.get('f1'))} |"
        )
    lines.append("")
    lines.append("## File Fingerprints")
    lines.append("")
    lines.append("| File | Exists | Size | SHA256 |")
    lines.append("|---|---:|---:|---|")
    for rec in payload["files"]:
        sha = rec.get("sha256") or "-"
        short = sha[:16] + "..." if sha != "-" else "-"
        lines.append(f"| `{rec['path']}` | {rec['exists']} | {rec.get('size_bytes') or '-'} | `{short}` |")
    lines.append("")
    lines.append("## Git State")
    git = payload["git"]
    head = git["rev_parse_head"].get("stdout") or git["rev_parse_head"].get("stderr") or git["rev_parse_head"].get("error")
    lines.append(f"- Git root candidate: `{git.get('git_root_candidate')}`")
    lines.append(f"- `.git` exists: `{git.get('git_dir_exists')}`")
    lines.append(f"- HEAD: `{head}`")
    status = git["status_short"].get("stdout") or git["status_short"].get("stderr") or "(clean)"
    lines.append("")
    lines.append("```text")
    lines.append(status)
    lines.append("```")
    lines.append("")
    lines.append("## Disk")
    lines.append("")
    for drive, info in payload["disk"].items():
        lines.append(f"- `{drive}` free `{info['free_gb']}` GB / total `{info['total_gb']}` GB")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- S135c is a task-adapted early stem mechanism: it conditions the backbone with water-concrete weak-boundary cues instead of appending a late generic head."
    )
    lines.append(
        "- The decisive fair test remains full RSCD training and full RSCD test. Screen results are only promotion gates."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a reproducibility snapshot for an RSCD candidate route.")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--full-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run", action="append", default=[], help="NAME=RUN_DIR metric anchor.")
    parser.add_argument("--extra-file", action="append", default=[], help="Extra file to hash.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_specs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be NAME=RUN_DIR, got: {spec}")
        name, path = spec.split("=", 1)
        run_specs.append((name, Path(path)))

    files: list[Path] = [ROOT / rel for rel in DEFAULT_FILES]
    for config in [args.screen_config, args.full_config]:
        files.extend(_resolve_config_chain(config))
    files.extend(Path(p) for p in args.extra_file)
    unique_files: list[Path] = []
    seen: set[str] = set()
    for path in files:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)

    payload: dict[str, Any] = {
        "candidate_name": args.candidate_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "screen_config": _config_summary(args.screen_config),
        "full_config": _config_summary(args.full_config),
        "files": [_file_record(path) for path in unique_files],
        "runs": [_run_summary(name, run_dir) for name, run_dir in run_specs],
        "environment": _environment_summary(),
        "git": _git_summary(),
        "disk": _disk_summary([args.output_dir, args.screen_config, args.full_config] + [p for _, p in run_specs]),
    }

    json_path = args.output_dir / "rscd_candidate_reproducibility_snapshot.json"
    md_path = args.output_dir / "rscd_candidate_reproducibility_snapshot.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(payload, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

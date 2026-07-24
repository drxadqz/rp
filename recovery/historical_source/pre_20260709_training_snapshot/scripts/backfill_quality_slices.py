from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_QUALITY_CSV = Path("data/quality_flags/image_quality_flags_roadsaw_roadsc_test.csv")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill per-sample quality slice reports for completed protocol runs."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--roadsaw-only-for-mixed",
        action="store_true",
        default=True,
        help="For mixed-dataset runs, evaluate RoadSaW test only to keep quality diagnostics fast and targeted.",
    )
    args = parser.parse_args()

    runs = discover_runs(args.root, only=set(args.only or []))
    rows: list[dict[str, Any]] = []
    for run_dir in runs:
        row = backfill_run(
            run_dir,
            quality_csv=args.quality_csv,
            force=bool(args.force),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            roadsaw_only_for_mixed=bool(args.roadsaw_only_for_mixed),
        )
        rows.append(row)

    report = {"root": str(args.root), "quality_csv": str(args.quality_csv), "runs": rows}
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    (args.summary_dir / "quality_slice_backfill.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.summary_dir / "quality_slice_backfill.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def discover_runs(root: Path, *, only: set[str]) -> list[Path]:
    runs = []
    for run_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        if only and run_dir.name not in only:
            continue
        if (run_dir / "best.pt").exists() and (run_dir / "config.json").exists():
            runs.append(run_dir)
    return runs


def backfill_run(
    run_dir: Path,
    *,
    quality_csv: Path,
    force: bool,
    batch_size: int,
    num_workers: int,
    roadsaw_only_for_mixed: bool,
) -> dict[str, Any]:
    config = run_dir / "config.json"
    checkpoint = run_dir / "best.pt"
    out_dir = run_dir / "quality_slices"
    out_json = out_dir / "quality_slices_test.json"
    if out_json.exists() and not force:
        return {"run": run_dir.name, "status": "exists", "out_json": str(out_json)}

    cfg = _load_json(config)
    if not cfg:
        return {"run": run_dir.name, "status": "missing_or_invalid_config"}
    test_manifests = [str(item) for item in ((cfg.get("data") or {}).get("test_manifests") or [])]
    if not test_manifests:
        return {"run": run_dir.name, "status": "missing_test_manifests"}

    eval_manifests = select_eval_manifests(test_manifests, roadsaw_only_for_mixed=roadsaw_only_for_mixed)
    if not eval_manifests:
        return {"run": run_dir.name, "status": "skipped_no_supported_quality_manifest", "test_manifests": test_manifests}

    cmd = [
        sys.executable,
        "scripts/evaluate_quality_slices.py",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "test",
        "--quality-csv",
        str(quality_csv),
        "--out-dir",
        str(out_dir),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]
    for manifest in eval_manifests:
        cmd.extend(["--eval-manifest", manifest])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {
            "run": run_dir.name,
            "status": "failed",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    return {
        "run": run_dir.name,
        "status": "created",
        "out_json": str(out_json),
        "eval_manifests": eval_manifests,
    }


def select_eval_manifests(test_manifests: list[str], *, roadsaw_only_for_mixed: bool) -> list[str]:
    supported = [path for path in test_manifests if _is_supported_quality_manifest(path)]
    if roadsaw_only_for_mixed and any("roadsaw" in Path(path).name.lower() for path in supported):
        return [path for path in supported if "roadsaw" in Path(path).name.lower()]
    return supported


def _is_supported_quality_manifest(path: str) -> bool:
    name = Path(path).name.lower()
    return "roadsaw" in name or "roadsc" in name


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Quality Slice Backfill",
        "",
        f"- Root: `{report['root']}`",
        f"- Quality CSV: `{report['quality_csv']}`",
        "",
        "| run | status | output |",
        "|---|---|---|",
    ]
    for row in report["runs"]:
        lines.append(f"| {row.get('run')} | {row.get('status')} | {row.get('out_json', '')} |")
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    main()

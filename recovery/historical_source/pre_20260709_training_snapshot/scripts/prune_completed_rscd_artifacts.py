from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_ROOT = Path(r"E:\perception_outputs\rscd_surface_classification")
REQUIRED_COMPLETE = ("test_metrics.json", "per_class_metrics.csv", "predictions_test.csv")
BEST_FILES = ("best_checkpoint.pth", "best.pt")
REMOVABLE_LAST_FILES = ("last_checkpoint.pth", "last.pt")


@dataclass
class Candidate:
    run_dir: str
    path: str
    size_mb: float
    reason: str


@dataclass
class SkippedRun:
    run_dir: str
    reason: str


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


def completed_run(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in REQUIRED_COMPLETE)


def has_best(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in BEST_FILES)


def collect_candidates(root: Path) -> tuple[list[Candidate], list[SkippedRun]]:
    candidates: list[Candidate] = []
    skipped: list[SkippedRun] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not completed_run(run_dir):
            skipped.append(SkippedRun(str(run_dir), "not_complete_or_missing_metrics"))
            continue
        if not has_best(run_dir):
            skipped.append(SkippedRun(str(run_dir), "missing_best_checkpoint_aliases"))
            continue
        best_paths = [run_dir / name for name in BEST_FILES]
        for name in REMOVABLE_LAST_FILES:
            path = run_dir / name
            if not path.exists():
                continue
            if any(same_file(path, best) for best in best_paths):
                skipped.append(SkippedRun(str(path), "last_is_same_filesystem_object_as_best"))
                continue
            candidates.append(
                Candidate(
                    run_dir=str(run_dir),
                    path=str(path),
                    size_mb=round(path.stat().st_size / 1024**2, 3),
                    reason="completed_run_has_best_and_test_metrics; last checkpoint is resume-only",
                )
            )
    return candidates, skipped


def write_reports(output_dir: Path, root: Path, candidates: list[Candidate], skipped: list[SkippedRun], *, applied: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_mb = round(sum(item.size_mb for item in candidates), 3)
    payload = {
        "root": str(root),
        "applied": applied,
        "candidate_count": len(candidates),
        "potential_free_mb": total_mb,
        "potential_free_gb": round(total_mb / 1024.0, 3),
        "candidates": [asdict(item) for item in candidates],
        "skipped_count": len(skipped),
        "skipped_preview": [asdict(item) for item in skipped[:200]],
        "policy": {
            "requires_complete_files": list(REQUIRED_COMPLETE),
            "requires_best_files": list(BEST_FILES),
            "removes_only": list(REMOVABLE_LAST_FILES),
            "never_removes_best": True,
        },
    }
    (output_dir / "prune_completed_rscd_artifacts.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Completed RSCD Artifact Prune Report",
        "",
        f"- Root: `{root}`",
        f"- Applied: `{applied}`",
        f"- Candidate files: `{len(candidates)}`",
        f"- Potential free space: `{payload['potential_free_gb']}` GB",
        "",
        "## Policy",
        "",
        "- Only completed runs with `test_metrics.json`, `per_class_metrics.csv`, and `predictions_test.csv` are eligible.",
        "- Both `best_checkpoint.pth` and `best.pt` must exist.",
        "- Only `last_checkpoint.pth` and `last.pt` are candidates.",
        "- `best*` artifacts are never removed.",
        "- Incomplete or currently-running runs are skipped by construction because they do not have complete test artifacts.",
        "",
        "## Candidates",
        "",
        "| File | Size MB | Reason |",
        "|---|---:|---|",
    ]
    for item in candidates[:300]:
        lines.append(f"| `{item.path}` | {item.size_mb:.3f} | {item.reason} |")
    if len(candidates) > 300:
        lines.append(f"| ... | ... | {len(candidates) - 300} more candidates omitted from markdown; see JSON. |")
    (output_dir / "prune_completed_rscd_artifacts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservatively prune completed RSCD run resume-only checkpoints.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    candidates, skipped = collect_candidates(args.root)
    if args.apply:
        for item in candidates:
            path = Path(item.path)
            if path.exists():
                path.unlink()
    write_reports(args.output_dir, args.root, candidates, skipped, applied=bool(args.apply))
    total_mb = sum(item.size_mb for item in candidates)
    print(
        json.dumps(
            {
                "applied": bool(args.apply),
                "candidate_count": len(candidates),
                "potential_free_gb": round(total_mb / 1024.0, 3),
                "report": str(args.output_dir / "prune_completed_rscd_artifacts.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ANCHOR_DIR = Path(
    "E:/perception_outputs/rscd_surface_classification/"
    "c3_farnet_official_anchor_source_reliable_router_s7_fulltest_20260708/fast_test"
)
DEFAULT_CANDIDATE_CSV = Path("reports/paper_protocol_summary/srbr_route_candidates_20260709/srbr_route_candidates.csv")


def read_confusion(path: Path) -> tuple[list[str], list[list[int]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        row_labels: list[str] = []
        matrix: list[list[int]] = []
        for row in reader:
            row_labels.append(row[0])
            matrix.append([int(float(v)) for v in row[1:]])
    if labels != row_labels:
        raise ValueError(f"row/column labels do not match in {path}")
    return labels, matrix


def read_candidates(path: Path, top_k: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:top_k]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare(anchor_dir: Path, run_dir: Path, candidate_csv: Path, out_dir: Path, top_k: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cm_path = run_dir / "confusion_matrix.csv"
    if not run_cm_path.exists():
        pending = {
            "status": "pending",
            "run_dir": str(run_dir),
            "confusion_matrix_exists": False,
        }
        (out_dir / "boundary_confusion_delta_status.json").write_text(
            json.dumps(pending, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(pending, ensure_ascii=False))
        return 2

    labels, anchor_cm = read_confusion(anchor_dir / "confusion_matrix.csv")
    run_labels, run_cm = read_confusion(run_cm_path)
    if labels != run_labels:
        raise ValueError("anchor and run confusion labels differ")
    idx = {name: i for i, name in enumerate(labels)}
    candidates = read_candidates(candidate_csv, top_k)

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        source = cand["source"]
        target = cand["target"]
        if source not in idx or target not in idx:
            continue
        fix_anchor = anchor_cm[idx[target]][idx[source]]
        fix_run = run_cm[idx[target]][idx[source]]
        reverse_anchor = anchor_cm[idx[source]][idx[target]]
        reverse_run = run_cm[idx[source]][idx[target]]
        fix_delta = fix_run - fix_anchor
        reverse_delta = reverse_run - reverse_anchor
        fixed = -fix_delta
        reverse_hurt_limit = max(3, int(round(0.10 * max(reverse_anchor, 1))))
        safety_pass = fixed > 0 and reverse_delta <= reverse_hurt_limit
        rows.append(
            {
                "source": source,
                "target": target,
                "factor": cand.get("factor", ""),
                "kind": cand.get("kind", ""),
                "anchor_fix_errors": fix_anchor,
                "run_fix_errors": fix_run,
                "fixed_errors_positive_good": fixed,
                "anchor_reverse_errors": reverse_anchor,
                "run_reverse_errors": reverse_run,
                "reverse_delta_positive_bad": reverse_delta,
                "reverse_hurt_limit": reverse_hurt_limit,
                "safety_pass": safety_pass,
            }
        )

    write_csv(
        out_dir / "boundary_confusion_deltas.csv",
        rows,
        [
            "source",
            "target",
            "factor",
            "kind",
            "anchor_fix_errors",
            "run_fix_errors",
            "fixed_errors_positive_good",
            "anchor_reverse_errors",
            "run_reverse_errors",
            "reverse_delta_positive_bad",
            "reverse_hurt_limit",
            "safety_pass",
        ],
    )
    pass_count = sum(1 for row in rows if row["safety_pass"])
    md = [
        "# Boundary Confusion Delta Audit",
        "",
        f"- Anchor dir: `{anchor_dir}`",
        f"- Run dir: `{run_dir}`",
        f"- Candidate csv: `{candidate_csv}`",
        f"- Safety-pass routes: {pass_count}/{len(rows)}",
        "",
        "## Top Watched Routes",
        "",
    ]
    for row in rows[:12]:
        md.append(
            "- `{source} -> {target}`: fixed {fixed} errors; reverse delta {reverse_delta} "
            "(limit {limit}); pass={passed}".format(
                source=row["source"],
                target=row["target"],
                fixed=row["fixed_errors_positive_good"],
                reverse_delta=row["reverse_delta_positive_bad"],
                limit=row["reverse_hurt_limit"],
                passed=row["safety_pass"],
            )
        )
    (out_dir / "boundary_confusion_delta_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    status = {"status": "complete", "out_dir": str(out_dir), "safety_pass": pass_count, "num_routes": len(rows)}
    (out_dir / "boundary_confusion_delta_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare watched boundary confusion counts against the S7 anchor.")
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    raise SystemExit(compare(args.anchor_dir, args.run_dir, args.candidate_csv, args.out_dir, args.top_k))


if __name__ == "__main__":
    main()

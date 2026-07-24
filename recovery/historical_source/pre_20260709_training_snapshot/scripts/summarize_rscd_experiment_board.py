from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SOTA_TOP1 = 0.9286
SOTA_MACRO_F1 = 0.8949
FULL_TEST_SAMPLES = 49_500
SCREEN_TEST_SAMPLES = 6_750
KEY_CLASSES = (
    "water_concrete_slight",
    "wet_concrete_slight",
    "water_concrete_severe",
    "wet_concrete_severe",
    "dry_concrete_slight",
    "dry_concrete_severe",
    "water_asphalt_slight",
)


@dataclass
class RunBoardRow:
    run: str
    path: str
    protocol: str
    num_samples: int | None
    top1: float | None
    macro_f1: float | None
    mean_precision: float | None
    mean_recall: float | None
    weighted_f1: float | None
    param_count: int | None
    worst_class: str | None
    worst_f1: float | None
    water_concrete_slight_f1: float | None
    wet_concrete_slight_f1: float | None
    water_concrete_severe_f1: float | None
    wet_concrete_severe_f1: float | None
    dry_concrete_slight_f1: float | None
    dry_concrete_severe_f1: float | None
    water_asphalt_slight_f1: float | None
    top1_gap_to_sota: float | None
    macro_f1_gap_to_sota: float | None
    extra_correct_to_top1_sota: int | None
    full_sota_pass: bool | None
    modified_time: str


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    payload = read_json(run_dir / "test_metrics.json")
    if payload is None:
        return None
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        return None
    return summary


def read_per_class(run_dir: Path) -> dict[str, dict[str, float]]:
    path = run_dir / "per_class_metrics.csv"
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("class") or row.get("\ufeffclass") or "")
            if not name:
                continue
            rows[name] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return rows


def infer_protocol(num_samples: int | None, run_name: str) -> str:
    if num_samples == FULL_TEST_SAMPLES:
        return "full"
    if num_samples == SCREEN_TEST_SAMPLES:
        return "screen"
    if num_samples is not None and num_samples <= 500:
        return "smoke"
    if "smoke" in run_name.lower():
        return "smoke"
    if num_samples is None:
        return "missing"
    return "partial"


def as_float(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None or value == "":
        return None
    return float(value)


def as_int(summary: dict[str, Any], key: str) -> int | None:
    value = summary.get(key)
    if value is None or value == "":
        return None
    return int(float(value))


def run_row(run_dir: Path) -> RunBoardRow | None:
    summary = read_summary(run_dir)
    if summary is None:
        return None
    per_class = read_per_class(run_dir)
    num_samples = as_int(summary, "num_samples")
    protocol = infer_protocol(num_samples, run_dir.name)
    top1 = as_float(summary, "top1")
    macro_f1 = as_float(summary, "macro_f1")
    worst_class = None
    worst_f1 = None
    if per_class:
        worst_class, worst_payload = min(per_class.items(), key=lambda item: item[1].get("f1", 0.0))
        worst_f1 = float(worst_payload.get("f1", 0.0))
    extra_correct = None
    if protocol == "full" and top1 is not None and num_samples is not None:
        target_correct = int(SOTA_TOP1 * num_samples + 0.999999)
        actual_correct = int(round(top1 * num_samples))
        extra_correct = max(target_correct - actual_correct, 0)
    full_sota_pass = None
    if protocol == "full" and top1 is not None and macro_f1 is not None:
        full_sota_pass = top1 >= SOTA_TOP1 and macro_f1 >= SOTA_MACRO_F1
    mtime = datetime.fromtimestamp((run_dir / "test_metrics.json").stat().st_mtime).isoformat(timespec="seconds")
    key_f1 = {key: per_class.get(key, {}).get("f1") for key in KEY_CLASSES}
    return RunBoardRow(
        run=run_dir.name,
        path=str(run_dir),
        protocol=protocol,
        num_samples=num_samples,
        top1=top1,
        macro_f1=macro_f1,
        mean_precision=as_float(summary, "mean_precision"),
        mean_recall=as_float(summary, "mean_recall"),
        weighted_f1=as_float(summary, "weighted_f1"),
        param_count=as_int(summary, "param_count"),
        worst_class=worst_class,
        worst_f1=worst_f1,
        water_concrete_slight_f1=key_f1["water_concrete_slight"],
        wet_concrete_slight_f1=key_f1["wet_concrete_slight"],
        water_concrete_severe_f1=key_f1["water_concrete_severe"],
        wet_concrete_severe_f1=key_f1["wet_concrete_severe"],
        dry_concrete_slight_f1=key_f1["dry_concrete_slight"],
        dry_concrete_severe_f1=key_f1["dry_concrete_severe"],
        water_asphalt_slight_f1=key_f1["water_asphalt_slight"],
        top1_gap_to_sota=None if top1 is None else top1 - SOTA_TOP1,
        macro_f1_gap_to_sota=None if macro_f1 is None else macro_f1 - SOTA_MACRO_F1,
        extra_correct_to_top1_sota=extra_correct,
        full_sota_pass=full_sota_pass,
        modified_time=mtime,
    )


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.3f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:+.3f} pp"


def collect_rows(root: Path, include_patterns: list[str], exclude_names: set[str]) -> list[RunBoardRow]:
    rows: list[RunBoardRow] = []
    seen: set[Path] = set()
    for pattern in include_patterns:
        for run_dir in root.glob(pattern):
            if not run_dir.is_dir():
                continue
            resolved = run_dir.resolve()
            if resolved in seen or run_dir.name in exclude_names:
                continue
            seen.add(resolved)
            row = run_row(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: ((row.top1 or -1.0), (row.macro_f1 or -1.0)), reverse=True)
    return rows


def write_csv(path: Path, rows: list[RunBoardRow]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, rows: list[RunBoardRow]) -> None:
    full_rows = [row for row in rows if row.protocol == "full"]
    screen_rows = [row for row in rows if row.protocol == "screen"]
    smoke_rows = [row for row in rows if row.protocol == "smoke"]
    sota_pass = [row for row in full_rows if row.full_sota_pass]
    lines = [
        "# RSCD Experiment Evidence Board",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Public SOTA Top-1 threshold: `{pct(SOTA_TOP1)}`",
        f"- Public SOTA Macro-F1 threshold: `{pct(SOTA_MACRO_F1)}`",
        f"- Full-test requirement: `{FULL_TEST_SAMPLES}` samples",
        f"- Runs scanned: `{len(rows)}`",
        f"- Full / screen / smoke / partial: `{len(full_rows)}` / `{len(screen_rows)}` / `{len(smoke_rows)}` / `{len([r for r in rows if r.protocol == 'partial'])}`",
        f"- Full SOTA pass found: `{bool(sota_pass)}`",
        "",
        "## Best Full Runs",
        "",
        "| Run | N | Top-1 | Macro-F1 | Top-1 Gap | Macro-F1 Gap | Extra Correct | Worst Class | Worst F1 | Pass SOTA |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in sorted(full_rows, key=lambda item: ((item.top1 or -1.0), (item.macro_f1 or -1.0)), reverse=True)[:12]:
        lines.append(
            f"| {row.run} | {row.num_samples} | {pct(row.top1)} | {pct(row.macro_f1)} | "
            f"{pp(row.top1_gap_to_sota)} | {pp(row.macro_f1_gap_to_sota)} | "
            f"{row.extra_correct_to_top1_sota if row.extra_correct_to_top1_sota is not None else '-'} | "
            f"{row.worst_class or '-'} | {pct(row.worst_f1)} | {row.full_sota_pass} |"
        )
    lines.extend(
        [
            "",
            "## Best Screen Runs",
            "",
            "| Run | N | Top-1 | Macro-F1 | Worst Class | Worst F1 | water_concrete_slight | wet_concrete_slight |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(screen_rows, key=lambda item: ((item.macro_f1 or -1.0), (item.top1 or -1.0)), reverse=True)[:20]:
        lines.append(
            f"| {row.run} | {row.num_samples} | {pct(row.top1)} | {pct(row.macro_f1)} | "
            f"{row.worst_class or '-'} | {pct(row.worst_f1)} | "
            f"{pct(row.water_concrete_slight_f1)} | {pct(row.wet_concrete_slight_f1)} |"
        )
    lines.extend(
        [
            "",
            "## Current Custom-Backbone Smoke Rows",
            "",
            "| Run | N | Top-1 | Macro-F1 | Params | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in [item for item in smoke_rows if item.run.startswith(("s136", "s137", "s138"))]:
        lines.append(
            f"| {row.run} | {row.num_samples} | {pct(row.top1)} | {pct(row.macro_f1)} | "
            f"{row.param_count if row.param_count is not None else '-'} | code-path only, not performance evidence |"
        )
    lines.extend(
        [
            "",
            "## Key Interpretation",
            "",
            "- Only `full` rows with 49,500 samples can support a final SOTA claim.",
            "- `screen` rows are route-selection evidence; they can justify launching a full run only after promotion/no-spill gates.",
            "- `smoke` rows only prove that code and metrics execute; they must not be used as method performance.",
            "- Top-1 SOTA requires both a high overall correct count and no broad spill into easy classes; Macro-F1 alone is insufficient.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a single evidence board for RSCD experiment outputs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\experiment_board_20260716"),
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=None,
        help="Glob pattern relative to root. Can be passed multiple times.",
    )
    args = parser.parse_args()
    patterns = args.include_pattern or ["*", "c3_farnet_*", "s13*"]
    rows = collect_rows(args.root, patterns, exclude_names={"comparison_live_20260715"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "thresholds": {
            "sota_top1": SOTA_TOP1,
            "sota_macro_f1": SOTA_MACRO_F1,
            "full_test_samples": FULL_TEST_SAMPLES,
        },
        "runs": [asdict(row) for row in rows],
    }
    (args.output_dir / "rscd_experiment_board.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "rscd_experiment_board.csv", rows)
    write_markdown(args.output_dir / "rscd_experiment_board.md", rows)
    print(json.dumps({"runs": len(rows), "report": str(args.output_dir / "rscd_experiment_board.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

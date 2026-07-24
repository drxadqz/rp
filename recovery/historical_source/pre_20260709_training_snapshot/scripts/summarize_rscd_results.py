from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FULL_SOTA = {
    # Literature/reference numbers used in the current project notes.
    # Keep these as reference lines only; formal claims should cite the paper/table.
    "RSPNet-L": {"top1": 0.9201, "macro_f1": 0.8949},
    "RoadFormer-L": {"top1": 0.9286, "macro_f1": 0.8499},
    "RoadMamba-B": {"top1": 0.9281, "macro_f1": 0.8479},
}

DEFAULT_RUN_SPECS = [
    "S7_full=E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709",
    "S96_cap250=E:/perception_outputs/rscd_surface_classification/c3_farnet_screen_s96_wc_pair_relative_boundary_20260712",
    "S133c_full=E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715",
    "S135c_screen=E:/perception_outputs/rscd_surface_classification/c3_farnet_screen_s135c_s96_wc_moderate_film_rough_focus_stem_20260715",
    "S135c_full=E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_s135c_s96_wc_moderate_film_rough_focus_stem_20260715",
]

KEY_CLASSES = [
    "water_concrete_slight",
    "water_concrete_severe",
    "water_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "dry_concrete_slight",
]


@dataclass
class RunRecord:
    name: str
    path: Path
    exists: bool
    summary: dict[str, Any]
    per_class: dict[str, dict[str, float]]
    error: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_per_class(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(name).lstrip("\ufeff") for name in (reader.fieldnames or [])]
        class_field = "class" if "class" in fieldnames else (fieldnames[0] if fieldnames else "class")
        for row in reader:
            if class_field not in row and f"\ufeff{class_field}" in row:
                class_field = f"\ufeff{class_field}"
            name = str(row.get(class_field, ""))
            if not name:
                continue
            out[name] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return out


def _load_run(spec: str) -> RunRecord:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
    else:
        path = Path(spec)
        name, raw_path = path.name, spec
    path = Path(raw_path)
    metrics_path = path / "test_metrics.json"
    if not metrics_path.exists():
        return RunRecord(name=name, path=path, exists=False, summary={}, per_class={}, error="missing test_metrics.json")
    try:
        metrics = _read_json(metrics_path)
        summary = dict(metrics.get("summary", metrics))
        per_class = _read_per_class(path / "per_class_metrics.csv")
        return RunRecord(name=name, path=path, exists=True, summary=summary, per_class=per_class)
    except Exception as exc:  # keep summary script non-destructive and diagnostic
        return RunRecord(name=name, path=path, exists=False, summary={}, per_class={}, error=repr(exc))


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.3f}%"


def _worst_class(per_class: dict[str, dict[str, float]]) -> tuple[str, float] | tuple[None, None]:
    if not per_class:
        return None, None
    name, item = min(per_class.items(), key=lambda kv: float(kv[1].get("f1", 0.0)))
    return name, float(item.get("f1", 0.0))


def _delta(value: float | None, ref: float | None) -> str:
    if value is None or ref is None:
        return "-"
    return f"{100.0 * (value - ref):+.3f} pp"


def _protocol(num_samples: float | None) -> str:
    if num_samples is None:
        return "missing"
    if num_samples >= 40000:
        return "full"
    return "screen/capped"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize RSCD experiment outputs into comparable tables.")
    parser.add_argument("--run", action="append", default=[], help="NAME=DIR or DIR. DIR must contain test_metrics.json.")
    parser.add_argument("--output-dir", type=Path, default=Path("E:/perception_outputs/rscd_surface_classification/final_comparisons"))
    parser.add_argument("--include-default-sota", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_specs = args.run or DEFAULT_RUN_SPECS
    runs = [_load_run(spec) for spec in run_specs]

    rows: list[dict[str, Any]] = []
    for run in runs:
        top1 = _num(run.summary.get("top1"))
        macro = _num(run.summary.get("macro_f1"))
        mean_p = _num(run.summary.get("mean_precision"))
        mean_r = _num(run.summary.get("mean_recall"))
        weighted = _num(run.summary.get("weighted_f1"))
        errors = _num(run.summary.get("num_errors"))
        samples = _num(run.summary.get("num_samples"))
        worst_name, worst_f1 = _worst_class(run.per_class)
        row = {
            "name": run.name,
            "path": str(run.path),
            "available": run.exists,
            "error": run.error or "",
            "num_samples": samples,
            "protocol": _protocol(samples),
            "top1": top1,
            "macro_f1": macro,
            "mean_precision": mean_p,
            "mean_recall": mean_r,
            "weighted_f1": weighted,
            "num_errors": errors,
            "worst_class": worst_name,
            "worst_class_f1": worst_f1,
        }
        for cls in KEY_CLASSES:
            row[f"{cls}_f1"] = float(run.per_class.get(cls, {}).get("f1", 0.0)) if run.per_class else None
        rows.append(row)

    json_path = args.output_dir / "rscd_result_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.output_dir / "rscd_result_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "name",
        "path",
        "available",
        "error",
        "num_samples",
        "top1",
        "macro_f1",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md = ["# RSCD Result Summary", ""]
    md.append("| Run | Available | Protocol | Samples | Top-1 | Macro-F1 | Worst class | Worst F1 | Errors |")
    md.append("|---|---:|---|---:|---:|---:|---|---:|---:|")
    for row in rows:
        md.append(
            "| {name} | {available} | {protocol} | {samples} | {top1} | {macro} | {worst} | {worst_f1} | {errors} |".format(
                name=row["name"],
                available=row["available"],
                protocol=row["protocol"],
                samples="-" if row["num_samples"] is None else int(row["num_samples"]),
                top1=_fmt_pct(row["top1"]),
                macro=_fmt_pct(row["macro_f1"]),
                worst=row["worst_class"] or "-",
                worst_f1=_fmt_pct(row["worst_class_f1"]),
                errors="-" if row["num_errors"] is None else int(row["num_errors"]),
            )
        )

    if args.include_default_sota:
        md.extend(["", "## Literature Reference Lines", ""])
        md.append(
            "Only full-protocol runs (`num_samples >= 40000`) are compared against literature reference lines; "
            "screen/capped runs are excluded from this delta."
        )
        md.append("")
        md.append("| Method | Top-1 | Macro-F1/F1 | Best full run Macro-F1 delta | Best full run Top-1 delta |")
        md.append("|---|---:|---:|---:|---:|")
        available = [row for row in rows if row["available"] and row["protocol"] == "full"]
        best_macro = max((_num(row["macro_f1"]) for row in available), default=None)
        best_top1 = max((_num(row["top1"]) for row in available), default=None)
        for name, vals in DEFAULT_FULL_SOTA.items():
            md.append(
                f"| {name} | {_fmt_pct(vals.get('top1'))} | {_fmt_pct(vals.get('macro_f1'))} | "
                f"{_delta(best_macro, vals.get('macro_f1'))} | {_delta(best_top1, vals.get('top1'))} |"
            )

    md.extend(["", "## Key Class F1", ""])
    md.append("| Run | " + " | ".join(KEY_CLASSES) + " |")
    md.append("|---" + "|---:" * len(KEY_CLASSES) + "|")
    for row in rows:
        vals = [_fmt_pct(row.get(f"{cls}_f1")) for cls in KEY_CLASSES]
        md.append(f"| {row['name']} | " + " | ".join(vals) + " |")

    md_path = args.output_dir / "rscd_result_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"markdown": str(md_path), "csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

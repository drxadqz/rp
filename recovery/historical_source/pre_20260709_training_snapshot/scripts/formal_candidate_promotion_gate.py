from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ANCHOR_DIR = Path(
    "E:/perception_outputs/rscd_surface_classification/"
    "c3_farnet_official_anchor_source_reliable_router_s7_fulltest_20260708/fast_test"
)
DEFAULT_SOTA_CSV = Path("reports/paper_protocol_summary/rscd_literature_sota_protocol_audit_20260703.csv")
DEFAULT_BOUNDARY_CSV = Path("reports/paper_protocol_summary/srbr_route_candidates_20260709/srbr_route_candidates.csv")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_per_class(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("class") or row.get("class_label")
            if not name:
                continue
            f1_key = "f1-score" if "f1-score" in row else "f1"
            rows[name] = {
                "precision": float(row.get("precision", 0.0) or 0.0),
                "recall": float(row.get("recall", 0.0) or 0.0),
                "f1": float(row.get(f1_key, 0.0) or 0.0),
                "support": float(row.get("support", 0.0) or 0.0),
            }
    return rows


def read_confusion(path: Path) -> tuple[list[str], list[list[int]]] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        rows: list[list[int]] = []
        row_labels: list[str] = []
        for row in reader:
            row_labels.append(row[0])
            rows.append([int(float(v)) for v in row[1:]])
    if labels != row_labels:
        raise ValueError(f"confusion row/column labels do not match: {path}")
    return labels, rows


def read_sota_thresholds(path: Path) -> dict[str, dict[str, Any]]:
    best = {
        "top1": {"value": None, "method": ""},
        "macro_f1": {"value": None, "method": ""},
    }
    if not path.exists():
        return best
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            method = str(row.get("method", ""))
            if "Current" in method or "Ours" in method or "FAF" in method:
                continue
            fair_note = str(row.get("use_for_fair_claim", "")).lower()
            if "not comparable" in fair_note or "extra data" in fair_note or "expanded" in fair_note:
                continue
            top1 = parse_float(row.get("top1_%"))
            macro = parse_float(row.get("mean_f1_%") or row.get("macro_f1_%"))
            if top1 is not None and (best["top1"]["value"] is None or top1 > float(best["top1"]["value"])):
                best["top1"] = {"value": top1, "method": method}
            if macro is not None and (
                best["macro_f1"]["value"] is None or macro > float(best["macro_f1"]["value"])
            ):
                best["macro_f1"] = {"value": macro, "method": method}
    return best


def read_config_audit(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config_resolved.yaml"
    if not path.exists():
        return {"config_exists": False, "full_test_protocol": False, "reason": "config_resolved.yaml missing"}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    val_cap = eval_cfg.get("max_val_samples_per_class")
    test_cap = eval_cfg.get("max_test_samples_per_class")
    full_eval = val_cap is None and test_cap is None
    samples_per_epoch = train_cfg.get("samples_per_epoch")
    full_train = samples_per_epoch in (0, None)
    return {
        "config_exists": True,
        "full_test_protocol": bool(full_eval),
        "full_train_protocol": bool(full_train),
        "max_val_samples_per_class": val_cap,
        "max_test_samples_per_class": test_cap,
        "samples_per_epoch": samples_per_epoch,
    }


def read_boundary_watch(path: Path, top_k: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))[:top_k]


def boundary_audit(run_dir: Path, anchor_dir: Path, watch_csv: Path, top_k: int) -> dict[str, Any]:
    run_cm = read_confusion(run_dir / "confusion_matrix.csv")
    anchor_cm = read_confusion(anchor_dir / "confusion_matrix.csv")
    if run_cm is None or anchor_cm is None:
        return {"available": False, "passes": 0, "total": 0, "rows": []}
    labels, run_matrix = run_cm
    anchor_labels, anchor_matrix = anchor_cm
    if labels != anchor_labels:
        raise ValueError("run and anchor confusion labels differ")
    idx = {name: i for i, name in enumerate(labels)}
    rows: list[dict[str, Any]] = []
    for item in read_boundary_watch(watch_csv, top_k):
        source = item["source"]
        target = item["target"]
        if source not in idx or target not in idx:
            continue
        fix_anchor = anchor_matrix[idx[target]][idx[source]]
        fix_run = run_matrix[idx[target]][idx[source]]
        reverse_anchor = anchor_matrix[idx[source]][idx[target]]
        reverse_run = run_matrix[idx[source]][idx[target]]
        fixed = fix_anchor - fix_run
        reverse_delta = reverse_run - reverse_anchor
        reverse_limit = max(3, int(round(0.10 * max(reverse_anchor, 1))))
        rows.append(
            {
                "source": source,
                "target": target,
                "fixed_errors_positive_good": fixed,
                "reverse_delta_positive_bad": reverse_delta,
                "reverse_hurt_limit": reverse_limit,
                "pass": fixed > 0 and reverse_delta <= reverse_limit,
            }
        )
    passes = sum(1 for row in rows if row["pass"])
    return {"available": True, "passes": passes, "total": len(rows), "rows": rows}


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}%"


def pp(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.2f} pp"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def gate(
    run_dir: Path,
    run_name: str,
    out_dir: Path,
    anchor_dir: Path,
    sota_csv: Path,
    boundary_csv: Path,
    expected_test_n: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(run_dir / "metrics.json")
    if metrics is None or not (run_dir / "per_class_metrics.csv").exists():
        pending = {
            "status": "pending",
            "run_dir": str(run_dir),
            "metrics_exists": metrics is not None,
            "per_class_exists": (run_dir / "per_class_metrics.csv").exists(),
            "confusion_exists": (run_dir / "confusion_matrix.csv").exists(),
        }
        (out_dir / "promotion_gate_status.json").write_text(json.dumps(pending, indent=2), encoding="utf-8")
        print(json.dumps(pending, ensure_ascii=False))
        return 2

    anchor_metrics = read_json(anchor_dir / "metrics.json")
    if anchor_metrics is None:
        raise FileNotFoundError(f"anchor metrics missing: {anchor_dir / 'metrics.json'}")
    run_summary = metrics["summary"]
    anchor_summary = anchor_metrics["summary"]
    run_pc = read_per_class(run_dir / "per_class_metrics.csv")
    anchor_pc = read_per_class(anchor_dir / "per_class_metrics.csv")
    sota = read_sota_thresholds(sota_csv)
    config = read_config_audit(run_dir)
    boundary = boundary_audit(run_dir, anchor_dir, boundary_csv, top_k=12)

    top1 = 100.0 * float(run_summary["top1"])
    macro = 100.0 * float(run_summary["macro_f1"])
    anchor_top1 = 100.0 * float(anchor_summary["top1"])
    anchor_macro = 100.0 * float(anchor_summary["macro_f1"])
    top1_delta = top1 - anchor_top1
    macro_delta = macro - anchor_macro
    top1_gate = sota["top1"]["value"]
    macro_gate = sota["macro_f1"]["value"]
    top1_sota_gap = None if top1_gate is None else top1 - float(top1_gate)
    macro_sota_gap = None if macro_gate is None else macro - float(macro_gate)
    test_n = int(run_summary.get("num_samples", 0))
    weakest = min(run_pc.items(), key=lambda item: item[1]["f1"])
    anchor_weakest_f1 = anchor_pc.get(weakest[0], {}).get("f1")
    weakest_delta = None if anchor_weakest_f1 is None else 100.0 * (weakest[1]["f1"] - anchor_weakest_f1)
    wcs_delta = None
    if "water_concrete_slight" in run_pc and "water_concrete_slight" in anchor_pc:
        wcs_delta = 100.0 * (run_pc["water_concrete_slight"]["f1"] - anchor_pc["water_concrete_slight"]["f1"])

    full_data_pass = test_n >= expected_test_n and config.get("full_test_protocol", False)
    external_sota_pass = (top1_sota_gap is not None and top1_sota_gap >= 0.0) and (
        macro_sota_gap is not None and macro_sota_gap >= 0.0
    )
    anchor_improvement_pass = macro_delta >= 0.05 and top1_delta >= -0.05
    weak_class_pass = weakest_delta is None or weakest_delta >= -0.50
    boundary_pass = not boundary["available"] or boundary["passes"] >= max(1, min(3, boundary["total"]))

    if not full_data_pass:
        decision = "invalid_for_fair_claim"
    elif external_sota_pass:
        decision = "sota_candidate_run_exact_pass_audit"
    elif anchor_improvement_pass and weak_class_pass and boundary_pass:
        decision = "promote_over_anchor_then_continue_sota_search"
    elif macro_delta >= -0.05 and wcs_delta is not None and wcs_delta > 0.50:
        decision = "weak_class_gain_candidate_needs_next_screen"
    else:
        decision = "do_not_promote_continue_queue"

    rows = [
        {
            "run_name": run_name,
            "decision": decision,
            "top1_%": top1,
            "macro_f1_%": macro,
            "top1_delta_vs_anchor_pp": top1_delta,
            "macro_f1_delta_vs_anchor_pp": macro_delta,
            "top1_gap_vs_external_pp": top1_sota_gap,
            "macro_f1_gap_vs_external_pp": macro_sota_gap,
            "test_n": test_n,
            "expected_test_n": expected_test_n,
            "weakest_class": weakest[0],
            "weakest_f1_%": 100.0 * weakest[1]["f1"],
            "weakest_delta_vs_anchor_pp": weakest_delta,
            "wcs_delta_vs_anchor_pp": wcs_delta,
            "full_data_pass": full_data_pass,
            "anchor_improvement_pass": anchor_improvement_pass,
            "external_sota_pass": external_sota_pass,
            "boundary_pass": boundary_pass,
        }
    ]
    write_csv(
        out_dir / "promotion_gate_summary.csv",
        rows,
        [
            "run_name",
            "decision",
            "top1_%",
            "macro_f1_%",
            "top1_delta_vs_anchor_pp",
            "macro_f1_delta_vs_anchor_pp",
            "top1_gap_vs_external_pp",
            "macro_f1_gap_vs_external_pp",
            "test_n",
            "expected_test_n",
            "weakest_class",
            "weakest_f1_%",
            "weakest_delta_vs_anchor_pp",
            "wcs_delta_vs_anchor_pp",
            "full_data_pass",
            "anchor_improvement_pass",
            "external_sota_pass",
            "boundary_pass",
        ],
    )
    if boundary["rows"]:
        write_csv(
            out_dir / "promotion_gate_boundary_watch.csv",
            boundary["rows"],
            [
                "source",
                "target",
                "fixed_errors_positive_good",
                "reverse_delta_positive_bad",
                "reverse_hurt_limit",
                "pass",
            ],
        )

    md = [
        "# Formal Candidate Promotion Gate",
        "",
        f"- Run: `{run_name}`",
        f"- Run dir: `{run_dir}`",
        f"- Decision: **{decision}**",
        "",
        "## Aggregate Metrics",
        "",
        f"- Top-1: {pct(top1)} ({pp(top1_delta)} vs S7 anchor; {pp(top1_sota_gap)} vs external gate)",
        f"- Macro-F1: {pct(macro)} ({pp(macro_delta)} vs S7 anchor; {pp(macro_sota_gap)} vs external gate)",
        f"- Test N: {test_n} / expected {expected_test_n}",
        "",
        "## Weak Class",
        "",
        f"- Weakest class: `{weakest[0]}` F1={pct(100.0 * weakest[1]['f1'])}, delta vs anchor={pp(weakest_delta)}",
        f"- `water_concrete_slight` delta vs anchor: {pp(wcs_delta)}",
        "",
        "## Gate Checks",
        "",
        f"- Full data protocol: {full_data_pass} (`max_test_samples_per_class={config.get('max_test_samples_per_class')}`)",
        f"- Anchor improvement: {anchor_improvement_pass}",
        f"- External SOTA pass: {external_sota_pass}",
        f"- Boundary no-harm pass: {boundary_pass} ({boundary['passes']}/{boundary['total']} watched routes)",
        "",
        "## Next Action",
        "",
    ]
    if decision == "sota_candidate_run_exact_pass_audit":
        md.append("- Run exact-pass/no-replacement verification and freeze the result only if it stays above external gates.")
    elif decision == "promote_over_anchor_then_continue_sota_search":
        md.append("- Promote this as the current local best, then continue with the next candidate because external SOTA is not yet beaten.")
    elif decision == "weak_class_gain_candidate_needs_next_screen":
        md.append("- Keep the weak-class gain as evidence, but do not promote globally; run the next queued candidate.")
    elif decision == "invalid_for_fair_claim":
        md.append("- Do not use this result for a fair claim; the full-data protocol check failed.")
    else:
        md.append("- Do not promote; continue the queued candidate search.")
    (out_dir / "promotion_gate_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    status = {
        "status": "complete",
        "decision": decision,
        "summary_csv": str(out_dir / "promotion_gate_summary.csv"),
        "report_md": str(out_dir / "promotion_gate_report.md"),
    }
    (out_dir / "promotion_gate_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a formal RSCD candidate for promotion/fair SOTA claims.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--sota-csv", type=Path, default=DEFAULT_SOTA_CSV)
    parser.add_argument("--boundary-csv", type=Path, default=DEFAULT_BOUNDARY_CSV)
    parser.add_argument("--expected-test-n", type=int, default=49500)
    args = parser.parse_args()
    raise SystemExit(
        gate(
            args.run_dir,
            args.run_name,
            args.out_dir,
            args.anchor_dir,
            args.sota_csv,
            args.boundary_csv,
            args.expected_test_n,
        )
    )


if __name__ == "__main__":
    main()

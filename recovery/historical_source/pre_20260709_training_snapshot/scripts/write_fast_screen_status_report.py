from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path("configs/experiments/fast_screen")
DEFAULT_LOG_DIR = Path("outputs/fast_screen_queue")
DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/fast_screen_status_report.md")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/fast_screen_status_report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--include-all-configs",
        action="store_true",
        help="Scan every screen_*.yaml in config-dir instead of the current fast_screen_manifest.json run list.",
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.config_dir, args.log_dir, include_all_configs=args.include_all_configs)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(config_dir: Path, log_dir: Path, *, include_all_configs: bool = False) -> dict[str, Any]:
    rows = []
    for config_path in _config_paths(config_dir, include_all_configs=include_all_configs):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        row = _row_from_config(config_path, cfg)
        rows.append(row)
    completed = [row for row in rows if row["status"] == "complete"]
    ranked = sorted(completed, key=_rank_score, reverse=True)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_dir": str(config_dir),
        "log_dir": str(log_dir),
        "selection": "all_config_files" if include_all_configs else "manifest_runs",
        "verdict": _verdict(rows),
        "claim_boundary": (
            "Fast-screen rows are proxy experiments for rapid candidate ranking. "
            "They cannot replace full paper-protocol, LODO, matched ConvNeXt, or bootstrap-CI evidence."
        ),
        "counts": {
            "total": len(rows),
            "complete": sum(1 for row in rows if row["status"] == "complete"),
            "running_or_partial": sum(1 for row in rows if row["status"] == "running_or_partial"),
            "missing": sum(1 for row in rows if row["status"] == "missing"),
        },
        "rows": rows,
        "ranked_complete": ranked,
        "decision_rules": _decision_rules(),
        "next_actions": _next_actions(rows, ranked),
    }


def _config_paths(config_dir: Path, *, include_all_configs: bool) -> list[Path]:
    if include_all_configs:
        return sorted(config_dir.glob("screen_*.yaml"))
    manifest = _load_json(config_dir / "fast_screen_manifest.json") or {}
    paths: list[Path] = []
    for row in manifest.get("runs", []) if isinstance(manifest.get("runs"), list) else []:
        if not isinstance(row, dict):
            continue
        raw = row.get("config")
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = config_dir / path
        if path.exists():
            paths.append(path)
    return sorted(paths) if paths else sorted(config_dir.glob("screen_*.yaml"))


def _row_from_config(config_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(cfg.get("output_dir", ""))
    source = str(cfg.get("screen_parent_run") or config_path.stem.removeprefix("screen_"))
    state = _load_json(output_dir / "training_state.json") or {}
    bootstrap = _load_json(output_dir / "bootstrap_metrics.json") or {}
    detailed = _load_json(output_dir / "detailed_test.json") or {}
    diag = _load_json(output_dir / "dataset_id_diagnostic.json") or {}
    audit = _load_json(output_dir / "topvenue_result_audit.json") or {}
    status = _status(output_dir, cfg)
    low_info = _low_friction_info(detailed, bootstrap)
    row = {
        "run": config_path.stem,
        "source_run": source,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "status": status,
        "epoch": state.get("epoch"),
        "epochs": state.get("epochs") or (cfg.get("optim") or {}).get("epochs"),
        "best_val_loss": state.get("best_metric"),
        "safety_proxy": state.get("best_safety_metric"),
        "friction_f1": _metric(bootstrap, "classification", "friction", "macro_f1"),
        "risk_f1": _metric(bootstrap, "classification", "risk", "macro_f1"),
        "low_friction_recall": low_info["recall"],
        "low_friction_recall_applicable": low_info["applicable"],
        "low_friction_positive_count": low_info["num_positive"],
        "raw_coverage": _metric(bootstrap, "mu_interval", "raw_coverage"),
        "calibrated_coverage": _metric(bootstrap, "mu_interval", "calibrated_coverage"),
        "calibrated_width": _metric(bootstrap, "mu_interval", "calibrated_width"),
        "worst_dataset_f1": _metric(bootstrap, "classification", "friction", "worst_dataset_macro_f1"),
        "dataset_id_bal_acc": diag.get("overall_dataset_id_balanced_accuracy"),
        "audit_verdict": audit.get("verdict"),
        "screen_score": None,
    }
    row["screen_score"] = _rank_score(row) if status == "complete" else None
    return row


def _status(output_dir: Path, cfg: dict[str, Any]) -> str:
    if not output_dir.exists():
        return "missing"
    required = [
        "best.pt",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "topvenue_result_audit.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if cfg.get("model", {}).get("use_evidence_field"):
        for name in ["evidence_field_audit.json", "evidence_maps"]:
            if not (output_dir / name).exists():
                missing.append(name)
    if not _is_single_or_baseline_name(output_dir.name) and not (output_dir / "dataset_id_diagnostic.json").exists():
        missing.append("dataset_id_diagnostic.json")
    return "complete" if not missing else "running_or_partial"


def _is_single_or_baseline_name(name: str) -> bool:
    base = name.removeprefix("screen_")
    return base.startswith("single_") or base.startswith("baseline_single_") or base.startswith("final_single_")


def _metric(payload: dict[str, Any], *path: str) -> float | None:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, dict) and "point" in cur:
        return _as_float(cur.get("point"))
    return _as_float(cur)


def _low_friction_info(detailed: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    boot = bootstrap.get("low_friction_detection", {}) if isinstance(bootstrap, dict) else {}
    detail = detailed.get("low_friction_detection", {}) if isinstance(detailed, dict) else {}
    recall = _metric(bootstrap, "low_friction_detection", "recall")
    applicable = boot.get("applicable") if isinstance(boot, dict) else None
    num_positive = boot.get("num_positive") if isinstance(boot, dict) else None
    if applicable is None and isinstance(detail, dict):
        applicable = detail.get("applicable")
    if num_positive is None and isinstance(detail, dict):
        num_positive = detail.get("num_positive")
    if applicable is False or _as_float(num_positive) == 0:
        recall = None
        applicable = False
    return {
        "recall": recall,
        "applicable": applicable,
        "num_positive": num_positive,
    }


def _rank_score(row: dict[str, Any]) -> float:
    risk = _num(row.get("risk_f1"))
    friction = _num(row.get("friction_f1"))
    low = _num(row.get("low_friction_recall"), default=risk)
    raw_cov = _num(row.get("raw_coverage"))
    cov = _num(row.get("calibrated_coverage"))
    width = _num(row.get("calibrated_width"))
    worst = _num(row.get("worst_dataset_f1"))
    shortcut = _num(row.get("dataset_id_bal_acc"), default=0.85)
    raw_coverage_penalty = max(0.45 - raw_cov, 0.0)
    coverage_penalty = abs(cov - 0.90)
    width_penalty = max(width - 0.62, 0.0)
    shortcut_penalty = max(shortcut - 0.85, 0.0)
    return (
        0.22 * risk
        + 0.18 * friction
        + 0.22 * low
        + 0.16 * worst
        + 0.12 * (1.0 - coverage_penalty)
        - 0.08 * raw_coverage_penalty
        - 0.05 * width_penalty
        - 0.05 * shortcut_penalty
    )


def _verdict(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "not_generated"
    if any(row["status"] == "complete" for row in rows):
        return "screening_started"
    if any(row["status"] == "running_or_partial" for row in rows):
        return "running_or_partial"
    return "configured_not_run"


def _decision_rules() -> list[str]:
    return [
        "Use fast-screen only to rank candidates and catch obvious failures; never use it as the final paper table.",
        "Promote a candidate to full formal runs only if it improves risk F1, low-friction recall, or worst-dataset F1 without inflating interval width.",
        "Treat a raw coverage collapse as a safety warning even if calibrated coverage later recovers.",
        "Treat dataset-ID balanced accuracy above 85% as shortcut risk unless LODO metrics improve enough to justify the representation.",
        "RoadSaW stress rows must improve damp/wet/very-wet behavior before making any RoadSaW robustness claim.",
        "All final claims still require full protocol, matched ConvNeXt baselines, LODO, calibration, and bootstrap confidence intervals.",
    ]


def _next_actions(rows: list[dict[str, Any]], ranked: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["Generate fast-screen configs with scripts/make_fast_screen_configs.py."]
    if not ranked:
        return [
            "Run fast-screen only after the current GPU training process is idle.",
            "Start with --scope candidates; run --scope roadsaw after the candidate proxy table exists.",
        ]
    best = ranked[0]
    return [
        f"Inspect `{best['run']}` first; it has the highest current fast-screen score.",
        "Promote only the top one or two candidate routes to full formal reruns unless the proxy rows are statistically indistinguishable.",
        "Refresh the official paper-protocol reports after any promoted full run completes.",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fast-Screen Status Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        f"Counts: `{json.dumps(report['counts'], ensure_ascii=False)}`",
        "",
        "## Runs",
        "",
        "| Run | Source | Status | Epoch | Score | friction F1 | risk F1 | low recall | raw cov | cal cov | width | worst F1 | dataset-ID |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {run} | {source} | `{status}` | {epoch}/{epochs} | {score} | {friction} | {risk} | {low} | {raw_cov} | {cov} | {width} | {worst} | {shortcut} |".format(
                run=row["run"],
                source=row["source_run"],
                status=row["status"],
                epoch=row.get("epoch") or "-",
                epochs=row.get("epochs") or "-",
                score=_fmt_float(row.get("screen_score")),
                friction=_fmt_pct(row.get("friction_f1")),
                risk=_fmt_pct(row.get("risk_f1")),
                low=_fmt_pct(row.get("low_friction_recall")),
                raw_cov=_fmt_pct(row.get("raw_coverage")),
                cov=_fmt_pct(row.get("calibrated_coverage")),
                width=_fmt_float(row.get("calibrated_width")),
                worst=_fmt_pct(row.get("worst_dataset_f1")),
                shortcut=_fmt_pct(row.get("dataset_id_bal_acc")),
            )
        )
    lines.extend(["", "## Decision Rules", ""])
    for item in report["decision_rules"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except json.JSONDecodeError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any, default: float = 0.0) -> float:
    out = _as_float(value)
    return default if out is None else out


def _fmt_pct(value: Any) -> str:
    out = _as_float(value)
    return "-" if out is None else f"{100.0 * out:.2f}%"


def _fmt_float(value: Any) -> str:
    out = _as_float(value)
    return "-" if out is None else f"{out:.4f}"


if __name__ == "__main__":
    main()

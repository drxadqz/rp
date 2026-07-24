from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect when loss-selected checkpoints diverge from interval-safety checkpoints."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "checkpoint_divergence_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "checkpoint_divergence_report.json")
    args = parser.parse_args()

    report = build_report(args.root, args.run)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path, runs: list[str]) -> dict[str, Any]:
    run_dirs = [root / name for name in runs] if runs else sorted(root.glob("*"))
    rows = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        history = _load_json(run_dir / "metrics_history.json")
        if isinstance(history, list) and history:
            rows.append(_row(run_dir, history))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "verdict": _verdict(rows),
        "claim_boundary": (
            "This report diagnoses checkpoint-selection risk. It does not replace final test metrics, "
            "bootstrap CIs, or the predeclared main-table selection protocol."
        ),
        "rows": rows,
    }


def _row(run_dir: Path, history: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [_epoch_row(item) for item in history]
    parsed = [row for row in parsed if row.get("epoch") is not None]
    best_loss = min(parsed, key=lambda row: _num(row.get("val_loss"), default=float("inf")))
    best_coverage = max(parsed, key=lambda row: _num(row.get("val_raw_coverage"), default=-1.0))
    best_safety = max(parsed, key=_safety_score)
    latest = max(parsed, key=lambda row: int(row.get("epoch") or 0))
    loss_vs_safety_cov_drop = _delta(best_loss.get("val_raw_coverage"), best_safety.get("val_raw_coverage"))
    loss_vs_best_cov_drop = _delta(best_loss.get("val_raw_coverage"), best_coverage.get("val_raw_coverage"))
    status, reason = _status(best_loss, best_safety, best_coverage)
    return {
        "run": run_dir.name,
        "status": status,
        "reason": reason,
        "num_epochs_recorded": len(parsed),
        "latest": latest,
        "best_loss": best_loss,
        "best_coverage": best_coverage,
        "best_safety": best_safety,
        "loss_vs_safety_raw_coverage_delta": loss_vs_safety_cov_drop,
        "loss_vs_best_raw_coverage_delta": loss_vs_best_cov_drop,
        "has_best_pt": (run_dir / "best.pt").exists(),
        "has_best_safety_pt": (run_dir / "best_safety.pt").exists(),
    }


def _epoch_row(item: dict[str, Any]) -> dict[str, Any]:
    val = item.get("val_metrics", {}) if isinstance(item.get("val_metrics"), dict) else {}
    return {
        "epoch": item.get("epoch"),
        "val_loss": val.get("loss"),
        "val_friction_acc": val.get("acc_friction"),
        "val_risk_acc": val.get("acc_risk"),
        "val_snow_acc": val.get("acc_snow"),
        "val_raw_coverage": val.get("mu_interval_coverage"),
        "val_width": val.get("mu_interval_width"),
        "best_metric": item.get("best_metric"),
        "best_safety_metric": item.get("best_safety_metric"),
    }


def _safety_score(row: dict[str, Any]) -> float:
    coverage = _num(row.get("val_raw_coverage"), default=0.0)
    friction = _num(row.get("val_friction_acc"), default=0.0)
    risk = _num(row.get("val_risk_acc"), default=0.0)
    width = _num(row.get("val_width"), default=1.2)
    return coverage + 0.5 * risk + 0.2 * friction - 0.1 * width


def _status(
    best_loss: dict[str, Any],
    best_safety: dict[str, Any],
    best_coverage: dict[str, Any],
) -> tuple[str, str]:
    loss_cov = _num(best_loss.get("val_raw_coverage"), default=0.0)
    safety_cov = _num(best_safety.get("val_raw_coverage"), default=0.0)
    best_cov = _num(best_coverage.get("val_raw_coverage"), default=0.0)
    if best_loss.get("epoch") != best_safety.get("epoch") and safety_cov - loss_cov >= 0.10:
        return "diverged_use_safety_for_interval_claims", "loss-selected checkpoint loses at least 10 coverage points versus safety selection"
    if best_cov - loss_cov >= 0.10:
        return "coverage_regression_watch", "loss-selected checkpoint loses at least 10 coverage points versus best-coverage epoch"
    return "aligned_or_minor", "loss and interval-safety checkpoints are aligned enough for this diagnostic"


def _verdict(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    if any(row.get("status") == "diverged_use_safety_for_interval_claims" for row in rows):
        return "checkpoint_divergence_detected"
    if any(row.get("status") == "coverage_regression_watch" for row in rows):
        return "coverage_regression_watch"
    return "no_major_divergence"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint Divergence Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        report["claim_boundary"],
        "",
        "| run | status | best loss epoch/cov | best safety epoch/cov | best cov epoch/cov | latest epoch/cov | decision reason |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    if not report["rows"]:
        lines.append("| - | missing | - | - | - | - | - |")
    for row in report["rows"]:
        lines.append(
            "| {run} | `{status}` | {bl} | {bs} | {bc} | {latest} | {reason} |".format(
                run=row.get("run"),
                status=row.get("status"),
                bl=_epoch_cov(row.get("best_loss")),
                bs=_epoch_cov(row.get("best_safety")),
                bc=_epoch_cov(row.get("best_coverage")),
                latest=_epoch_cov(row.get("latest")),
                reason=row.get("reason"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _epoch_cov(row: Any) -> str:
    if not isinstance(row, dict):
        return "-"
    return f"{row.get('epoch')}/{_fmt_pct(row.get('val_raw_coverage'))}"


def _delta(cur: Any, prev: Any) -> float | None:
    cur_num = _num(cur)
    prev_num = _num(prev)
    if cur_num is None or prev_num is None:
        return None
    return cur_num - prev_num


def _num(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_pct(value: Any) -> str:
    num = _num(value)
    if num is None:
        return "-"
    return f"{100.0 * num:.2f}%"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()

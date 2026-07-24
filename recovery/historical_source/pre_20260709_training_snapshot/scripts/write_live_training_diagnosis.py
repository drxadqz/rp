from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="v5_full_faf")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    out_json = args.out_json or args.summary_dir / f"{args.run}_training_diagnosis.json"
    out_md = args.out_md or args.summary_dir / f"{args.run}_training_diagnosis.md"

    report = build_report(args.run, args.summary_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(run: str, summary_dir: Path) -> dict[str, Any]:
    trend = _load_json(summary_dir / f"{run}_live_training_trend.json") or {}
    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    watch = _load_json(summary_dir / "active_training_watch_report.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    completed = trend.get("completed_epochs", [])
    completed = _merge_watch_epoch(completed, run, watch)
    best_loss = _best_epoch(completed, key_path=("val", "loss"), mode="min")
    best_safety = _best_safety_epoch(completed)
    latest = completed[-1] if completed else (trend.get("latest_completed_epoch") or {})
    active = _prefer_newer_active(_active_from_watch(run, watch), trend.get("active_progress"))

    latest_val = latest.get("val") or {}
    best_val = (best_loss or {}).get("val") or {}
    first_val = ((completed[0] if completed else {}) or {}).get("val") or {}
    prev = completed[-2] if len(completed) >= 2 else {}
    prev_val = prev.get("val") or {}

    final_route = dashboard.get("final_route_sanity") or {}
    module_risks = _module_risks(final_selection)
    signals = _signals(latest_val, prev_val, best_val, first_val)
    recommendation = _recommendation(run, latest, best_loss, best_safety, signals, final_route, module_risks)
    return {
        "run": run,
        "source": str(summary_dir / f"{run}_live_training_trend.json"),
        "active_progress": active,
        "num_completed_epochs": len(completed),
        "latest_epoch": _compact_epoch(latest),
        "best_val_loss_epoch": _compact_epoch(best_loss),
        "best_safety_proxy_epoch": _compact_epoch(best_safety),
        "signals": signals,
        "module_risks": module_risks,
        "final_route_sanity": final_route,
        "recommendation": recommendation,
    }


def _best_epoch(rows: list[dict[str, Any]], *, key_path: tuple[str, str], mode: str) -> dict[str, Any] | None:
    scored = []
    for row in rows:
        value = _dig(row, key_path)
        if value is not None:
            scored.append((float(value), row))
    if not scored:
        return None
    return (min if mode == "min" else max)(scored, key=lambda item: item[0])[1]


def _active_from_watch(run: str, watch: dict[str, Any]) -> dict[str, Any] | None:
    active = watch.get("active") or {}
    if not active or active.get("name") != run:
        return None
    return {
        "phase": active.get("phase"),
        "epoch": active.get("epoch"),
        "epochs": active.get("epochs"),
        "step": active.get("step"),
        "steps": active.get("steps"),
    }


def _prefer_newer_active(watch_active: dict[str, Any] | None, trend_active: Any) -> dict[str, Any] | None:
    if not isinstance(trend_active, dict):
        return watch_active
    if not watch_active:
        return trend_active
    watch_epoch = _as_int(watch_active.get("epoch"))
    trend_epoch = _as_int(trend_active.get("epoch"))
    watch_step = _as_int(watch_active.get("step"))
    trend_step = _as_int(trend_active.get("step"))
    if (
        watch_epoch is None
        or (trend_epoch is not None and trend_epoch > watch_epoch)
        or (trend_epoch == watch_epoch and trend_step is not None and (watch_step is None or trend_step >= watch_step))
    ):
        return {**watch_active, **trend_active}
    return watch_active


def _merge_watch_epoch(rows: list[dict[str, Any]], run: str, watch: dict[str, Any]) -> list[dict[str, Any]]:
    active = watch.get("active") or {}
    latest = watch.get("latest_completed_epoch") or {}
    previous = watch.get("previous_completed_epoch") or {}
    if active.get("name") != run or not latest.get("epoch"):
        return rows
    out = list(rows)
    if previous.get("epoch") and not any(row.get("epoch") == previous.get("epoch") for row in out):
        out.append(_epoch_from_watch(previous))
    watch_row = _epoch_from_watch(latest)
    replaced = False
    for index, row in enumerate(out):
        if row.get("epoch") == watch_row.get("epoch"):
            out[index] = {**row, **watch_row}
            replaced = True
            break
    if not replaced:
        out.append(watch_row)
    return sorted(out, key=lambda row: int(row.get("epoch") or 0))


def _epoch_from_watch(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": row.get("epoch"),
        "epochs": row.get("epochs"),
        "train": {
            "loss": row.get("train_loss"),
            "acc_risk": row.get("train_acc_risk"),
            "mu_interval_coverage": row.get("train_mu_interval_coverage"),
            "mu_interval_width": row.get("train_mu_interval_width"),
        },
        "val": {
            "loss": row.get("val_loss"),
            "acc_risk": row.get("val_acc_risk"),
            "acc_friction": row.get("val_acc_friction"),
            "acc_wetness": row.get("val_acc_wetness"),
            "mu_interval_coverage": row.get("val_mu_interval_coverage"),
            "mu_interval_width": row.get("val_mu_interval_width"),
        },
        "best_checkpoint_saved": row.get("saved_best"),
    }


def _best_safety_epoch(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = []
    for row in rows:
        val = row.get("val") or {}
        risk = _num(val.get("acc_risk"))
        friction = _num(val.get("acc_friction"))
        coverage = _num(val.get("mu_interval_coverage"))
        width = _num(val.get("mu_interval_width"))
        if risk is None or friction is None or coverage is None or width is None:
            continue
        proxy = risk + 0.5 * friction + 0.5 * coverage - 0.1 * width
        scored.append((proxy, row))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _signals(latest: dict[str, Any], prev: dict[str, Any], best: dict[str, Any], first: dict[str, Any]) -> dict[str, Any]:
    latest_loss = _num(latest.get("loss"))
    prev_loss = _num(prev.get("loss"))
    best_loss = _num(best.get("loss"))
    latest_cov = _num(latest.get("mu_interval_coverage"))
    prev_cov = _num(prev.get("mu_interval_coverage"))
    first_cov = _num(first.get("mu_interval_coverage"))
    latest_risk = _num(latest.get("acc_risk"))
    prev_risk = _num(prev.get("acc_risk"))
    best_risk = _num(best.get("acc_risk"))
    return {
        "val_loss_delta_vs_previous": _diff(latest_loss, prev_loss),
        "val_loss_delta_vs_best": _diff(latest_loss, best_loss),
        "risk_acc_delta_vs_previous": _diff(latest_risk, prev_risk),
        "risk_acc_delta_vs_best_loss_epoch": _diff(latest_risk, best_risk),
        "raw_coverage_delta_vs_previous": _diff(latest_cov, prev_cov),
        "raw_coverage_delta_vs_first": _diff(latest_cov, first_cov),
        "validation_degradation_flag": (
            latest_loss is not None and best_loss is not None and latest_loss > best_loss + 0.02
        ),
        "coverage_degradation_flag": (
            latest_cov is not None and first_cov is not None and latest_cov < first_cov - 0.03
        ),
    }


def _module_risks(final_selection: dict[str, Any]) -> list[dict[str, Any]]:
    risks = []
    for item in final_selection.get("risk_register", []):
        name = str(item.get("risk", ""))
        if name.startswith("module_"):
            risks.append(
                {
                    "module": name.replace("module_", ""),
                    "level": item.get("level"),
                    "evidence": item.get("evidence"),
                    "action": item.get("action"),
                }
            )
    return risks


def _recommendation(
    run: str,
    latest: dict[str, Any],
    best_loss: dict[str, Any] | None,
    best_safety: dict[str, Any] | None,
    signals: dict[str, Any],
    final_route: dict[str, Any],
    module_risks: list[dict[str, Any]],
) -> list[str]:
    out = []
    latest_epoch = latest.get("epoch")
    best_epoch = (best_loss or {}).get("epoch")
    best_safety_epoch = (best_safety or {}).get("epoch")
    if best_epoch is not None:
        out.append(f"Use the selected checkpoint, not the latest epoch blindly; current best validation-loss epoch is {best_epoch}.")
    if best_safety_epoch is not None and best_safety_epoch != best_epoch:
        out.append(f"Keep the safety checkpoint as a supplemental analysis; its best safety-proxy epoch is {best_safety_epoch}.")
    if signals.get("validation_degradation_flag"):
        out.append(
            f"{run} shows validation degradation by epoch {latest_epoch}; this supports testing a leaner final route instead of adding more full-stack modules."
        )
    if signals.get("coverage_degradation_flag"):
        out.append(
            "Raw interval coverage is degrading across training; P3 candidates should be judged by coverage-width tradeoff, not by risk F1 alone."
        )
    risky = [item.get("module") for item in module_risks if item.get("level") in {"warn", "block"}]
    if risky:
        out.append("Treat these modules as provisional removals unless LODO/fair-baseline evidence rescues them: " + ", ".join(risky) + ".")
    if final_route.get("verdict") == "pass":
        out.append(
            "The prepared final configs follow the lean PhysicsTexture + EvidenceField + road-ROI safety route with small state-conditioned semantic alignment."
        )
    if not out:
        out.append("No strong degradation signal yet; wait for the completed run and postprocess metrics before changing the queue.")
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Training Diagnosis",
        "",
        f"Run: `{report['run']}`",
        f"Completed epochs: `{report['num_completed_epochs']}`",
        "",
    ]
    active = report.get("active_progress") or {}
    if active:
        lines.append(
            "Active progress: epoch {epoch}/{epochs}, step {step}/{steps}.".format(
                epoch=active.get("epoch", "-"),
                epochs=active.get("epochs", "-"),
                step=active.get("step", "-"),
                steps=active.get("steps", "-"),
            )
        )
        lines.append("")

    lines.extend(["## Epoch Summary", ""])
    lines.append("| Scope | epoch | val loss | risk acc | friction acc | raw cov | raw width |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, row in [
        ("latest completed", report.get("latest_epoch")),
        ("best val loss", report.get("best_val_loss_epoch")),
        ("best safety proxy", report.get("best_safety_proxy_epoch")),
    ]:
        lines.append(_epoch_row(label, row or {}))
    lines.append("")

    signals = report.get("signals") or {}
    lines.extend(["## Signals", ""])
    lines.append(f"- Val loss delta vs previous: `{_fmt_signed_abs(signals.get('val_loss_delta_vs_previous'))}`.")
    lines.append(f"- Val loss delta vs best: `{_fmt_signed_abs(signals.get('val_loss_delta_vs_best'))}`.")
    lines.append(f"- Risk accuracy delta vs previous: `{_fmt_signed_pct(signals.get('risk_acc_delta_vs_previous'))}`.")
    lines.append(f"- Risk accuracy delta vs best-loss epoch: `{_fmt_signed_pct(signals.get('risk_acc_delta_vs_best_loss_epoch'))}`.")
    lines.append(f"- Raw coverage delta vs previous: `{_fmt_signed_pct(signals.get('raw_coverage_delta_vs_previous'))}`.")
    lines.append(f"- Raw coverage delta vs first epoch: `{_fmt_signed_pct(signals.get('raw_coverage_delta_vs_first'))}`.")
    lines.append(f"- Validation degradation flag: `{signals.get('validation_degradation_flag')}`.")
    lines.append(f"- Coverage degradation flag: `{signals.get('coverage_degradation_flag')}`.")
    lines.append("")

    risks = report.get("module_risks") or []
    if risks:
        lines.extend(["## Module Risk Carryover", ""])
        for item in risks:
            lines.append(f"- `{item.get('module')}` `{item.get('level')}`: {item.get('evidence')}")
        lines.append("")

    final_route = report.get("final_route_sanity") or {}
    if final_route:
        lines.extend(["## Final Route Check", ""])
        lines.append(f"- Verdict: `{final_route.get('verdict')}` across `{final_route.get('num_final_runs')}` final configs.")
        lines.append(f"- Policy: {final_route.get('policy')}")
        lines.append("")

    lines.extend(["## Recommendation", ""])
    for item in report.get("recommendation", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _compact_epoch(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    val = row.get("val") or {}
    train = row.get("train") or {}
    return {
        "epoch": row.get("epoch"),
        "train_loss": train.get("loss"),
        "val_loss": val.get("loss"),
        "val_risk_acc": val.get("acc_risk"),
        "val_friction_acc": val.get("acc_friction"),
        "raw_coverage": val.get("mu_interval_coverage"),
        "raw_width": val.get("mu_interval_width"),
        "best_checkpoint_saved": row.get("best_checkpoint_saved"),
        "best_safety_checkpoint_saved": row.get("best_safety_checkpoint_saved"),
    }


def _epoch_row(label: str, row: dict[str, Any]) -> str:
    return (
        f"| {label} | {row.get('epoch', '-')} | {_fmt_abs(row.get('val_loss'))} | "
        f"{_fmt_pct(row.get('val_risk_acc'))} | {_fmt_pct(row.get('val_friction_acc'))} | "
        f"{_fmt_pct(row.get('raw_coverage'))} | {_fmt_abs(row.get('raw_width'))} |"
    )


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dig(row: dict[str, Any], key_path: tuple[str, str]) -> Any:
    cur: Any = row
    for key in key_path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _fmt_signed_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


def _fmt_signed_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):+.2f}%"


if __name__ == "__main__":
    main()

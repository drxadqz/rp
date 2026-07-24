from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEYS = [
    "loss",
    "acc_friction",
    "acc_risk",
    "mu_interval_coverage",
    "mu_interval_width",
    "mu_mean_mae_to_interval_mid",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    report = build_report(
        history,
        args.run_name or args.history.parent.name,
        planned_epochs=load_planned_epochs(args.history.parent),
    )
    md = render_markdown(report)
    print(md)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def build_report(
    history: list[dict[str, Any]],
    run_name: str,
    planned_epochs: int | None = None,
) -> dict[str, Any]:
    rows = [row for row in history if isinstance(row, dict) and isinstance(row.get("val_metrics"), dict)]
    if not rows:
        return {"run": run_name, "status": "empty", "warnings": ["No validation metrics found."]}

    best = min(rows, key=lambda row: float(row["val_metrics"].get("loss", float("inf"))))
    best_safety = max(rows, key=lambda row: safety_proxy(row["val_metrics"]))
    latest = rows[-1]
    planned_epochs = planned_epochs or latest.get("epochs")
    first = rows[0]
    best_metrics = best["val_metrics"]
    best_safety_metrics = best_safety["val_metrics"]
    latest_metrics = latest["val_metrics"]
    first_metrics = first["val_metrics"]
    warnings = []
    latest_cov = latest_metrics.get("mu_interval_coverage")
    best_cov = best_metrics.get("mu_interval_coverage")
    if latest_cov is not None and float(latest_cov) < 0.70:
        warnings.append("Latest raw validation interval coverage is below 70%.")
    if best_cov is not None and float(best_cov) < 0.70:
        warnings.append("Best-loss checkpoint still has low raw interval coverage.")
    if latest["epoch"] != best["epoch"] and latest_metrics.get("loss") is not None:
        delta = float(latest_metrics["loss"]) - float(best_metrics["loss"])
        if delta > 0.01:
            warnings.append("Validation loss has increased after the current best checkpoint.")
    if latest_metrics.get("acc_risk") is not None and first_metrics.get("acc_risk") is not None:
        if float(latest_metrics["acc_risk"]) - float(first_metrics["acc_risk"]) < 0.02:
            warnings.append("Risk accuracy has not improved much from the first epoch.")
    if best_safety.get("epoch") != best.get("epoch"):
        warnings.append("Best validation loss and best safety proxy occur at different epochs.")
        loss_cov = best_metrics.get("mu_interval_coverage")
        safety_cov = best_safety_metrics.get("mu_interval_coverage")
        if loss_cov is not None and safety_cov is not None:
            cov_gap = float(safety_cov) - float(loss_cov)
            if cov_gap > 0.10:
                warnings.append(
                    "Best-loss checkpoint has substantially lower raw coverage than the safety-proxy checkpoint."
                )

    return {
        "run": run_name,
        "status": "ok",
        "num_epochs_observed": len(rows),
        "latest_epoch": latest.get("epoch"),
        "planned_epochs": planned_epochs,
        "best_epoch_by_val_loss": best.get("epoch"),
        "best_epoch_by_safety_proxy": best_safety.get("epoch"),
        "safety_proxy_definition": "acc_risk + 0.5*acc_friction + 0.5*mu_interval_coverage - 0.1*mu_interval_width",
        "first_val": compact_metrics(first_metrics),
        "best_val": compact_metrics(best_metrics),
        "best_safety_val": compact_metrics(best_safety_metrics),
        "latest_val": compact_metrics(latest_metrics),
        "latest_minus_best": metric_delta(latest_metrics, best_metrics),
        "latest_minus_first": metric_delta(latest_metrics, first_metrics),
        "warnings": warnings,
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in KEYS}


def load_planned_epochs(run_dir: Path) -> int | None:
    for name, keys in [
        ("training_state.json", ["epochs"]),
        ("config.json", ["optim", "epochs"]),
    ]:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = dig(payload, keys)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def dig(payload: Any, keys: list[str]) -> Any:
    cur = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def safety_proxy(metrics: dict[str, Any]) -> float:
    risk = float(metrics.get("acc_risk", 0.0) or 0.0)
    friction = float(metrics.get("acc_friction", 0.0) or 0.0)
    coverage = float(metrics.get("mu_interval_coverage", 0.0) or 0.0)
    width = float(metrics.get("mu_interval_width", 0.0) or 0.0)
    return risk + 0.5 * friction + 0.5 * coverage - 0.1 * width


def metric_delta(cur: dict[str, Any], prev: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in KEYS:
        if cur.get(key) is None or prev.get(key) is None:
            out[key] = None
        else:
            out[key] = float(cur[key]) - float(prev[key])
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Training History Summary", "", f"Run: `{report['run']}`", ""]
    if report.get("status") != "ok":
        for warning in report.get("warnings", []):
            lines.append(f"- {warning}")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"- Observed epochs: {report['num_epochs_observed']} / {report.get('planned_epochs')}",
            f"- Best epoch by validation loss: {report['best_epoch_by_val_loss']}",
            f"- Best epoch by safety proxy: {report['best_epoch_by_safety_proxy']}",
            f"- Safety proxy: `{report['safety_proxy_definition']}`",
            f"- Latest epoch: {report['latest_epoch']}",
            "",
            "| Scope | val loss | friction acc | risk acc | raw coverage | raw width | mu MAE-to-mid |",
            "|---|---:|---:|---:|---:|---:|---:|",
            metric_row("First", report["first_val"]),
            metric_row("Best loss", report["best_val"]),
            metric_row("Best safety proxy", report["best_safety_val"]),
            metric_row("Latest", report["latest_val"]),
            "",
            "| Delta | val loss | friction acc | risk acc | raw coverage | raw width | mu MAE-to-mid |",
            "|---|---:|---:|---:|---:|---:|---:|",
            metric_row("Latest - best", report["latest_minus_best"], signed=True),
            metric_row("Latest - first", report["latest_minus_first"], signed=True),
            "",
            "## Warnings",
            "",
        ]
    )
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def metric_row(label: str, metrics: dict[str, Any], signed: bool = False) -> str:
    values = [fmt(metrics.get(key), signed=signed) for key in KEYS]
    return f"| {label} | " + " | ".join(values) + " |"


def fmt(value: Any, signed: bool = False) -> str:
    if value is None:
        return "-"
    number = float(value)
    if signed:
        return f"{number:+.4f}"
    return f"{number:.4f}"


if __name__ == "__main__":
    main()

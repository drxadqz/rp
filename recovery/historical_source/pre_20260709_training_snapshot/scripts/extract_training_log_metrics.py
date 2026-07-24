from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)")
EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s*/\s*(\d+)\s*$")
STEP_RE = re.compile(r"train step\s+(\d+)\s*/\s*(\d+)\s+loss=([-+0-9.eE]+)")
TQDM_RE = re.compile(
    r"(?P<phase>train|eval):\s*(?P<pct>\d+)%.*\|\s*"
    r"(?P<step>\d+)\s*/\s*(?P<steps>\d+)\s*"
    r"\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s*(?P<rate>[^\]]+)\]"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--err-log", type=Path, default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    report = parse_log(args.log, err_log=args.err_log, run=args.run)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def parse_log(path: Path, *, err_log: Path | None = None, run: str | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    epochs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_step: dict[str, Any] | None = None
    last_completed: dict[str, Any] | None = None

    for raw in text.splitlines():
        line = raw.strip()
        epoch_match = EPOCH_RE.match(line)
        if epoch_match:
            current = {
                "epoch": int(epoch_match.group(1)),
                "epochs": int(epoch_match.group(2)),
                "train": None,
                "val": None,
                "best_checkpoint_saved": False,
                "best_safety_checkpoint_saved": False,
                "last_train_step": None,
            }
            epochs.append(current)
            active_step = None
            continue

        if current is None:
            continue

        step_match = STEP_RE.search(line)
        if step_match:
            active_step = {
                "epoch": current["epoch"],
                "epochs": current["epochs"],
                "step": int(step_match.group(1)),
                "steps": int(step_match.group(2)),
                "loss": float(step_match.group(3)),
            }
            current["last_train_step"] = active_step
            continue

        if line.startswith("train:"):
            current["train"] = _parse_metrics(line)
            continue

        if line.startswith("val"):
            current["val"] = _parse_metrics(line)
            last_completed = current
            continue

        if "saved best checkpoint" in line:
            current["best_checkpoint_saved"] = True
            continue

        if "saved best safety checkpoint" in line:
            current["best_safety_checkpoint_saved"] = True

    completed = [row for row in epochs if row.get("train") is not None and row.get("val") is not None]
    active = _active_progress(epochs, completed, active_step)
    active = _merge_tqdm_progress(active, err_log or _default_err_log(path))
    return {
        "run": run or path.stem,
        "log": str(path),
        "parsed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_epochs_seen": len(epochs),
        "num_completed_epochs": len(completed),
        "active_progress": active,
        "latest_completed_epoch": last_completed,
        "completed_epochs": completed,
        "trend": _trend(completed),
    }


def _parse_metrics(line: str) -> dict[str, float]:
    return {key: float(value) for key, value in METRIC_RE.findall(line)}


def _active_progress(
    epochs: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    active_step: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if active_step is not None:
        return active_step
    if not epochs:
        return None
    latest = epochs[-1]
    if completed and completed[-1] is latest:
        return None
    return {"epoch": latest.get("epoch"), "epochs": latest.get("epochs"), "step": None, "steps": None}


def _default_err_log(path: Path) -> Path | None:
    name = path.name
    if name.endswith(".out.log"):
        candidate = path.with_name(name[:-8] + ".err.log")
        return candidate if candidate.exists() else None
    return None


def _merge_tqdm_progress(active: dict[str, Any] | None, err_log: Path | None) -> dict[str, Any] | None:
    if active is None or err_log is None or not err_log.exists():
        return active
    snap = _tqdm_snapshot(err_log)
    if not snap:
        return active
    if snap.get("phase") == "eval":
        active = dict(active)
        active["phase"] = "eval"
        active["eval_step"] = snap.get("step")
        active["eval_steps"] = snap.get("steps")
        active["eval_tqdm_percent"] = snap.get("percent")
        active["eval_tqdm_eta"] = snap.get("eta")
        active["eval_tqdm_rate"] = snap.get("rate")
        active["tqdm_log"] = str(err_log)
        return active
    if snap.get("phase") != "train":
        return active
    step = snap.get("step")
    steps = snap.get("steps")
    if step is None or steps is None:
        return active
    current_step = active.get("step")
    if current_step is None or int(step) >= int(current_step):
        active = dict(active)
        active["step"] = int(step)
        active["steps"] = int(steps)
        active["phase"] = "train"
        active["tqdm_percent"] = snap.get("percent")
        active["tqdm_eta"] = snap.get("eta")
        active["tqdm_rate"] = snap.get("rate")
        active["tqdm_log"] = str(err_log)
    return active


def _tqdm_snapshot(err_log: Path) -> dict[str, Any] | None:
    try:
        lines = err_log.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-500:]):
        match = TQDM_RE.search(line)
        if not match:
            continue
        return {
            "phase": match.group("phase"),
            "percent": int(match.group("pct")),
            "step": int(match.group("step")),
            "steps": int(match.group("steps")),
            "elapsed": match.group("elapsed").strip(),
            "eta": match.group("eta").strip(),
            "rate": match.group("rate").strip(),
        }
    return None


def _trend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {}
    first = rows[0]
    prev = rows[-2]
    last = rows[-1]
    keys = [
        "loss",
        "acc_friction",
        "acc_risk",
        "acc_wetness",
        "mu_interval_coverage",
        "mu_interval_width",
        "attention_bottom_half_mass",
        "attention_center_bottom_mass",
    ]
    return {
        "from_first_to_latest": _delta_block(first, last, keys),
        "from_previous_to_latest": _delta_block(prev, last, keys),
    }


def _delta_block(prev: dict[str, Any], cur: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["train", "val"]:
        prev_metrics = prev.get(split) or {}
        cur_metrics = cur.get(split) or {}
        for key in keys:
            if key in prev_metrics and key in cur_metrics:
                out[f"{split}_{key}"] = cur_metrics[key] - prev_metrics[key]
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Training Log Metrics",
        "",
        f"Run: `{report['run']}`",
        f"Log: `{report['log']}`",
        f"Parsed at: {report['parsed_at']}",
        "",
    ]
    active = report.get("active_progress")
    if active:
        step = active.get("step")
        steps = active.get("steps")
        if step is not None and steps is not None:
            lines.append(f"Active progress: epoch {active.get('epoch')}/{active.get('epochs')}, step {step}/{steps}.")
        else:
            lines.append(f"Active progress: epoch {active.get('epoch')}/{active.get('epochs')}.")
        lines.append("")

    latest = report.get("latest_completed_epoch") or {}
    if latest:
        lines.append("## Latest Completed Epoch")
        lines.append("")
        lines.append("| epoch | train loss | val loss | val friction acc | val risk acc | raw cov | raw width | best | best safety |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
        lines.append(_epoch_row(latest))
        lines.append("")

    lines.append("## Completed Epochs")
    lines.append("")
    lines.append("| epoch | train loss | val loss | val friction acc | val risk acc | raw cov | raw width | best | best safety |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for row in report.get("completed_epochs", []):
        lines.append(_epoch_row(row))
    lines.append("")

    trend = report.get("trend") or {}
    if trend:
        lines.append("## Trend")
        lines.append("")
        lines.append("| Window | val loss | val risk acc | raw cov | raw width |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, block in [
            ("first to latest", trend.get("from_first_to_latest", {})),
            ("previous to latest", trend.get("from_previous_to_latest", {})),
        ]:
            lines.append(
                "| {label} | {loss} | {risk} | {cov} | {width} |".format(
                    label=label,
                    loss=_fmt_signed(block.get("val_loss")),
                    risk=_fmt_signed(block.get("val_acc_risk")),
                    cov=_fmt_signed(block.get("val_mu_interval_coverage")),
                    width=_fmt_signed_abs(block.get("val_mu_interval_width")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _epoch_row(row: dict[str, Any]) -> str:
    train = row.get("train") or {}
    val = row.get("val") or {}
    return (
        f"| {row.get('epoch')} | {_fmt_abs(train.get('loss'))} | {_fmt_abs(val.get('loss'))} | "
        f"{_fmt(val.get('acc_friction'))} | {_fmt(val.get('acc_risk'))} | "
        f"{_fmt(val.get('mu_interval_coverage'))} | {_fmt_abs(val.get('mu_interval_width'))} | "
        f"{'yes' if row.get('best_checkpoint_saved') else '-'} | "
        f"{'yes' if row.get('best_safety_checkpoint_saved') else '-'} |"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def _fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_signed(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):+.2f}"


def _fmt_signed_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


if __name__ == "__main__":
    main()

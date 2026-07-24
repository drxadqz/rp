from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


LOG_DIR = Path("outputs/rscd_surface_formal_queue")
OUT = Path("reports/paper_protocol_summary/rscd_training_trend_report")
RESULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")

RUNS = {
    "formal_convnext_tiny_b12e20_resume": LOG_DIR / "formal_convnext_tiny_b12e20_resume.log",
    "formal_physics_texture_quality_b12e20_parallel": LOG_DIR / "formal_physics_texture_quality_b12e20_parallel.log",
    "formal_physics_wavelet_directional_film_gate_hier": LOG_DIR
    / "formal_texture_film_wavelet_candidate.stdout.log",
}

VAL_RE = re.compile(
    r"val\s+:\s+loss=(?P<loss>[0-9.]+)\s+top1=(?P<top1>[0-9.]+)\s+macro_f1=(?P<macro>[0-9.]+)\s+bal_acc=(?P<bal>[0-9.]+)"
)
EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)")
EPOCH_BLOCK_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)"
    r"(?P<body>.*?)(?=Epoch\s+\d+/\d+|\Z)",
    re.DOTALL,
)


def main() -> None:
    runs = []
    for name, path in RUNS.items():
        runs.append(parse_run(name, path))
    result = {
        "claim_boundary": "Training-trend report only; final claims require evaluate_test.json.",
        "runs": runs,
        "recommendation": recommendation(runs),
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(OUT.with_suffix(".md"))


def parse_run(name: str, path: Path) -> dict[str, Any]:
    epochs: list[dict[str, Any]] = []
    current_epoch = None
    total_epochs = None
    if path.exists():
        text = read_log_text(path).replace("\r", "\n")
        normalized_text = re.sub(r"(?<!\n)\n(?!Epoch\s+\d+/\d+)", " ", text)
        for block_match in EPOCH_BLOCK_RE.finditer(normalized_text):
            current_epoch = int(block_match.group("epoch"))
            total_epochs = int(block_match.group("total"))
            val_match = VAL_RE.search(block_match.group("body"))
            if val_match:
                epochs.append(
                    {
                        "epoch": current_epoch,
                        "loss": float(val_match.group("loss")),
                        "top1": float(val_match.group("top1")),
                        "macro_f1": float(val_match.group("macro")),
                        "balanced_accuracy": float(val_match.group("bal")),
                    }
                )
    best = max(epochs, key=lambda r: (r["macro_f1"], r["top1"]), default=None)
    latest = epochs[-1] if epochs else None
    stale = 0
    if best and latest:
        stale = int(latest["epoch"] - best["epoch"])
    return {
        "name": name,
        "log": str(path),
        "total_epochs": total_epochs,
        "num_val_epochs": len(epochs),
        "latest": latest,
        "best": best,
        "stale_epochs": stale,
        "status": status(name, path, latest, total_epochs),
    }


def status(name: str, path: Path, latest: dict[str, Any] | None, total_epochs: int | None) -> str:
    if (RESULT_ROOT / name / "evaluate_test.json").exists():
        return "complete_test_available"
    if not path.exists():
        return "missing_log"
    text_tail = "\n".join(read_log_text(path).splitlines()[-80:])
    if "Traceback" in text_tail or "out of memory" in text_tail.lower():
        return "error_in_tail"
    if "early stopping" in text_tail.lower():
        return "early_stopped"
    if latest and total_epochs and latest["epoch"] >= total_epochs:
        return "training_epochs_complete_or_testing"
    return "running"


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    if raw.count(b"\x00") > max(16, len(raw) // 20):
        return raw.decode("utf-16-le", errors="ignore")
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            return raw.decode(encoding, errors="ignore")
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def recommendation(runs: list[dict[str, Any]]) -> dict[str, str]:
    if any(r["status"] == "error_in_tail" for r in runs):
        return {"status": "inspect_error", "message": "At least one training log has an error in its tail."}
    if all(r["status"] in {"early_stopped", "training_epochs_complete_or_testing"} for r in runs):
        return {"status": "await_test_or_queue", "message": "Formal training appears complete; wait for test evaluation and queued fast screens."}
    return {"status": "continue", "message": "Formal training is active and healthy; continue without killing jobs."}


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# RSCD Training Trend Report",
        "",
        result["claim_boundary"],
        "",
        "| run | status | latest epoch | latest Top-1 | latest Macro-F1 | best epoch | best Top-1 | best Macro-F1 | stale epochs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        latest = run.get("latest") or {}
        best = run.get("best") or {}
        lines.append(
            "| `{name}` | {status} | {le} | {lt} | {lf} | {be} | {bt} | {bf} | {stale} |".format(
                name=run["name"],
                status=run["status"],
                le=latest.get("epoch", "-"),
                lt=pct(latest.get("top1")),
                lf=pct(latest.get("macro_f1")),
                be=best.get("epoch", "-"),
                bt=pct(best.get("top1")),
                bf=pct(best.get("macro_f1")),
                stale=run.get("stale_epochs", "-"),
            )
        )
    rec = result["recommendation"]
    lines.extend(["", "## Recommendation", "", f"- `{rec['status']}`: {rec['message']}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
SUMMARY = Path("reports/paper_protocol_summary")
OUT_JSON = SUMMARY / "rscd_formal_validation_diagnosis.json"
OUT_MD = SUMMARY / "rscd_formal_validation_diagnosis.md"

RUNS = {
    "baseline": ROOT / "formal_convnext_tiny_b12e20_resume",
    "physics_texture": ROOT / "formal_physics_texture_quality_b12e20_parallel",
}


def main() -> None:
    trend_rows = _rows_from_training_trend()
    if trend_rows:
        rows = trend_rows
    else:
        histories = {name: _load_history(path / "history.json") for name, path in RUNS.items()}
        rows = {name: _summarize_history(history) for name, history in histories.items()}
    diagnosis = {
        "claim_boundary": (
            "Formal validation diagnosis only. It is not a final test result and "
            "cannot support an RSCD SOTA claim without evaluate_test.json."
        ),
        "rows": rows,
        "comparison": _compare(rows),
        "decision": _decision(rows),
    }
    OUT_JSON.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(diagnosis), encoding="utf-8")
    print(OUT_MD)


def _rows_from_training_trend() -> dict[str, dict[str, Any]]:
    trend_path = SUMMARY / "rscd_training_trend_report.json"
    if not trend_path.exists():
        return {}
    trend = json.loads(trend_path.read_text(encoding="utf-8"))
    mapping = {
        "formal_convnext_tiny_b12e20_resume": "baseline",
        "formal_physics_texture_quality_b12e20_parallel": "physics_texture",
    }
    rows: dict[str, dict[str, Any]] = {}
    for item in trend.get("runs", []):
        name = mapping.get(str(item.get("name")))
        if not name:
            continue
        latest_raw = item.get("latest") or {}
        best_raw = item.get("best") or {}
        if not latest_raw or not best_raw:
            rows[name] = {"status": "missing"}
            continue
        latest = {
            "epoch": latest_raw.get("epoch"),
            "val_loss": latest_raw.get("loss"),
            "val_top1": latest_raw.get("top1"),
            "val_macro_f1": latest_raw.get("macro_f1"),
            "val_balanced_accuracy": latest_raw.get("balanced_accuracy"),
        }
        best = {
            "epoch": best_raw.get("epoch"),
            "val_loss": best_raw.get("loss"),
            "val_top1": best_raw.get("top1"),
            "val_macro_f1": best_raw.get("macro_f1"),
            "val_balanced_accuracy": best_raw.get("balanced_accuracy"),
        }
        latest_minus_best = float(latest.get("val_macro_f1") or 0.0) - float(best.get("val_macro_f1") or 0.0)
        rows[name] = {
            "status": "available",
            "num_epochs": item.get("num_val_epochs"),
            "latest": latest,
            "best": best,
            "latest_minus_best_macro_f1": latest_minus_best,
            "last_epoch_delta_macro_f1": None,
            "points": [],
            "stability_flag": _stability_flag(latest_minus_best, None),
            "source": "training_trend_log",
        }
    return rows


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {"status": "missing"}
    points = []
    for row in history:
        val = row.get("val") or {}
        train = row.get("train") or {}
        points.append(
            {
                "epoch": int(row.get("epoch", len(points) + 1)),
                "train_loss": train.get("loss"),
                "train_top1": train.get("top1"),
                "val_loss": val.get("loss"),
                "val_top1": val.get("top1"),
                "val_macro_f1": val.get("macro_f1"),
                "val_balanced_accuracy": val.get("balanced_accuracy"),
            }
        )
    best = max(points, key=lambda r: float(r.get("val_macro_f1") or -1.0))
    latest = points[-1]
    prev = points[-2] if len(points) >= 2 else None
    latest_drop_from_best = float(latest.get("val_macro_f1") or 0.0) - float(best.get("val_macro_f1") or 0.0)
    last_epoch_delta = (
        float(latest.get("val_macro_f1") or 0.0) - float(prev.get("val_macro_f1") or 0.0)
        if prev
        else None
    )
    return {
        "status": "available",
        "num_epochs": len(points),
        "latest": latest,
        "best": best,
        "latest_minus_best_macro_f1": latest_drop_from_best,
        "last_epoch_delta_macro_f1": last_epoch_delta,
        "points": points,
        "stability_flag": _stability_flag(latest_drop_from_best, last_epoch_delta),
    }


def _stability_flag(latest_minus_best: float, last_delta: float | None) -> str:
    if latest_minus_best <= -0.015:
        return "unstable_latest_drop"
    if last_delta is not None and last_delta <= -0.01:
        return "sharp_recent_drop"
    return "stable_or_improving"


def _compare(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = rows.get("baseline", {})
    phys = rows.get("physics_texture", {})
    if base.get("status") != "available" or phys.get("status") != "available":
        return {"status": "missing"}
    base_best = base["best"]
    phys_best = phys["best"]
    base_latest = base["latest"]
    phys_latest = phys["latest"]
    return {
        "status": "available",
        "best_macro_f1_gap_physics_minus_baseline": float(phys_best["val_macro_f1"]) - float(base_best["val_macro_f1"]),
        "latest_macro_f1_gap_physics_minus_baseline": float(phys_latest["val_macro_f1"]) - float(base_latest["val_macro_f1"]),
        "best_top1_gap_physics_minus_baseline": float(phys_best["val_top1"]) - float(base_best["val_top1"]),
        "latest_top1_gap_physics_minus_baseline": float(phys_latest["val_top1"]) - float(base_latest["val_top1"]),
    }


def _decision(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comp = _compare(rows)
    phys = rows.get("physics_texture", {})
    if comp.get("status") != "available":
        return {"status": "waiting", "message": "Need both formal histories."}
    best_gap = float(comp["best_macro_f1_gap_physics_minus_baseline"])
    latest_gap = float(comp["latest_macro_f1_gap_physics_minus_baseline"])
    unstable = phys.get("stability_flag") != "stable_or_improving"
    if best_gap > 0.001 and not unstable:
        return {
            "status": "candidate_general_module",
            "message": "PhysicsTexture validation is ahead and stable enough to await final test.",
        }
    if best_gap > -0.005:
        return {
            "status": "candidate_hard_slice_module",
            "message": (
                "PhysicsTexture is close on best validation but not clearly better. "
                "Keep it only if final test or wet/water/ice class slices improve."
            ),
        }
    return {
        "status": "prune_or_merge_unless_hard_slices_win",
        "message": (
            "PhysicsTexture is behind the baseline on current formal validation. "
            "Do not promote as a general RSCD module unless final test hard-slice evidence rescues it."
        ),
        "latest_gap": latest_gap,
    }


def _to_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# RSCD Formal Validation Diagnosis",
        "",
        data["claim_boundary"],
        "",
        "## Summary",
        "",
        "| run | epochs | latest Top-1 | latest Macro-F1 | best epoch | best Top-1 | best Macro-F1 | stability |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in data["rows"].items():
        if row.get("status") != "available":
            lines.append(f"| `{name}` | - | - | - | - | - | - | missing |")
            continue
        latest = row["latest"]
        best = row["best"]
        lines.append(
            "| `{name}` | {epochs} | {ltop1} | {lf1} | {bepoch} | {btop1} | {bf1} | `{flag}` |".format(
                name=name,
                epochs=row["num_epochs"],
                ltop1=_pct(latest.get("val_top1")),
                lf1=_pct(latest.get("val_macro_f1")),
                bepoch=best.get("epoch"),
                btop1=_pct(best.get("val_top1")),
                bf1=_pct(best.get("val_macro_f1")),
                flag=row.get("stability_flag"),
            )
        )
    comp = data["comparison"]
    lines += [
        "",
        "## Gaps",
        "",
        f"- Best Macro-F1 gap, PhysicsTexture minus baseline: {_pct(comp.get('best_macro_f1_gap_physics_minus_baseline'), signed=True)}",
        f"- Latest Macro-F1 gap, PhysicsTexture minus baseline: {_pct(comp.get('latest_macro_f1_gap_physics_minus_baseline'), signed=True)}",
        f"- Best Top-1 gap, PhysicsTexture minus baseline: {_pct(comp.get('best_top1_gap_physics_minus_baseline'), signed=True)}",
        f"- Latest Top-1 gap, PhysicsTexture minus baseline: {_pct(comp.get('latest_top1_gap_physics_minus_baseline'), signed=True)}",
        "",
        "## Decision",
        "",
        f"- Status: `{data['decision']['status']}`",
        f"- Message: {data['decision']['message']}",
        "",
        "## Rule",
        "",
        "- General-module claim requires final test improvement or stable validation improvement.",
        "- Hard-slice-module claim requires wet/water/ice or low-friction slice gains without severe overall regression.",
        "- If both overall and hard slices lose, prune PhysicsTexture or merge only the useful fixed cues into a gated/directional module.",
        "",
    ]
    return "\n".join(lines)


def _pct(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if signed and val >= 0 else ""
    return f"{sign}{val * 100:.2f}%"


if __name__ == "__main__":
    main()

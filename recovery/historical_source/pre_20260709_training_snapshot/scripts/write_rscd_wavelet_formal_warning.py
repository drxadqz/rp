from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUMMARY = Path("reports/paper_protocol_summary")
TREND_JSON = SUMMARY / "rscd_training_trend_report.json"
OUT = SUMMARY / "rscd_wavelet_formal_warning"
RESULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")

RUN = "formal_physics_wavelet_directional_film_gate_hier"
PHYSICS_RUN = "formal_physics_texture_quality_b12e20_resume"
PHYSICS_TOP1 = 0.8684646464646465
PHYSICS_MACRO_F1 = 0.8477992487999164


def main() -> None:
    trend = _load_json(TREND_JSON)
    run = None
    for item in trend.get("runs", []):
        if item.get("name") == RUN:
            run = item
            break
    if not run:
        raise SystemExit(f"Missing {RUN} in {TREND_JSON}")
    latest = run.get("latest") or {}
    best = run.get("best") or {}
    test_payload = _load_json(RESULT_ROOT / RUN / "evaluate_test.json")
    physics_payload = _load_json(RESULT_ROOT / PHYSICS_RUN / "evaluate_test.json")
    test_summary = test_payload.get("summary", {}) if test_payload else {}
    physics_summary = physics_payload.get("summary", {}) if physics_payload else {}
    test_slices = _slice_metrics(test_payload) if test_payload else {}
    physics_slices = _slice_metrics(physics_payload) if physics_payload else {}
    physics_top1 = _num(physics_summary.get("top1")) or PHYSICS_TOP1
    physics_macro_f1 = _num(physics_summary.get("macro_f1")) or PHYSICS_MACRO_F1
    result = {
        "claim_boundary": "Formal pruning audit for the Wavelet/Directional/FiLM RSCD candidate.",
        "run": RUN,
        "status": run.get("status"),
        "latest_epoch": latest.get("epoch"),
        "latest_top1": latest.get("top1"),
        "latest_macro_f1": latest.get("macro_f1"),
        "best_epoch": best.get("epoch"),
        "best_top1": best.get("top1"),
        "best_macro_f1": best.get("macro_f1"),
        "stale_epochs": run.get("stale_epochs"),
        "test_top1": test_summary.get("top1"),
        "test_macro_f1": test_summary.get("macro_f1"),
        "test_wet_water_f1": test_slices.get("wet_water_f1"),
        "test_water_concrete_f1": test_slices.get("water_concrete_f1"),
        "physics_reference_top1": physics_top1,
        "physics_reference_macro_f1": physics_macro_f1,
        "physics_reference_wet_water_f1": physics_slices.get("wet_water_f1"),
        "physics_reference_water_concrete_f1": physics_slices.get("water_concrete_f1"),
        "best_gap_top1_vs_physics_test": _num(best.get("top1")) - physics_top1,
        "best_gap_macro_f1_vs_physics_test": _num(best.get("macro_f1")) - physics_macro_f1,
        "test_gap_top1_vs_physics": _num(test_summary.get("top1")) - physics_top1,
        "test_gap_macro_f1_vs_physics": _num(test_summary.get("macro_f1")) - physics_macro_f1,
        "test_gap_wet_water_f1_vs_physics": _num(test_slices.get("wet_water_f1")) - _num(physics_slices.get("wet_water_f1")),
        "test_gap_water_concrete_f1_vs_physics": _num(test_slices.get("water_concrete_f1"))
        - _num(physics_slices.get("water_concrete_f1")),
        "decision": _decision(run, best, test_summary, test_slices, physics_slices, physics_top1, physics_macro_f1),
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(_to_markdown(result), encoding="utf-8")
    print(OUT.with_suffix(".md"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _slice_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    if not payload:
        return {}
    rows = []
    report = payload.get("classification_report", {})
    for label, item in report.items():
        if not isinstance(item, dict) or "f1-score" not in item:
            continue
        support = int(item.get("support") or 0)
        if support <= 0:
            continue
        rows.append(
            {
                "label": str(label),
                "friction": _friction_state(str(label)),
                "material": _material_state(str(label)),
                "f1": _num(item.get("f1-score")),
                "support": support,
            }
        )
    return {
        "wet_water_f1": _weighted_slice(rows, lambda r: r["friction"] in {"wet", "water"}),
        "water_concrete_f1": _weighted_slice(
            rows,
            lambda r: r["friction"] == "water" and r["material"] == "concrete",
        ),
    }


def _weighted_slice(rows: list[dict[str, Any]], predicate: Any) -> float:
    selected = [r for r in rows if predicate(r)]
    total = sum(int(r["support"]) for r in selected)
    if total <= 0:
        return 0.0
    return sum(float(r["f1"]) * int(r["support"]) for r in selected) / total


def _friction_state(label: str) -> str:
    label = label.strip().lower().replace("-", "_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return label
    return label.split("_")[0] if label else "unknown"


def _material_state(label: str) -> str | None:
    label = label.strip().lower().replace("-", "_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return None
    parts = label.split("_")
    return parts[1] if len(parts) >= 2 else None


def _decision(
    run: dict[str, Any],
    best: dict[str, Any],
    test_summary: dict[str, Any],
    test_slices: dict[str, float],
    physics_slices: dict[str, float],
    physics_top1: float,
    physics_macro_f1: float,
) -> dict[str, str]:
    if test_summary:
        top1_gap = _num(test_summary.get("top1")) - physics_top1
        macro_gap = _num(test_summary.get("macro_f1")) - physics_macro_f1
        wet_gap = _num(test_slices.get("wet_water_f1")) - _num(physics_slices.get("wet_water_f1"))
        if top1_gap < 0 or macro_gap < 0 or wet_gap < -0.003:
            return {
                "status": "final_prune",
                "action": (
                    "Reject WaveletDirectionalFiLM as the RSCD main method. It failed the formal "
                    "test gate against PhysicsTexture and regressed safety-relevant wet/water slices."
                ),
            }
        return {
            "status": "eligible",
            "action": "Formal test beat PhysicsTexture without wet/water regression; keep for downstream review.",
        }
    stale = int(run.get("stale_epochs") or 0)
    macro_gap = _num(best.get("macro_f1")) - physics_macro_f1
    if stale >= 4 and macro_gap < 0:
        return {
            "status": "likely_prune_unless_test_recovers",
            "action": (
                "Do not promote WaveletDirectionalFiLM as the RSCD main method unless "
                "its final test result unexpectedly beats PhysicsTexture and preserves hard wet slices."
            ),
        }
    return {
        "status": "watch",
        "action": "Wait for final evaluate_test.json before pruning.",
    }


def _pct(value: Any, *, signed: bool = False) -> str:
    number = _num(value)
    sign = "+" if signed and number >= 0 else ""
    return f"{sign}{number * 100:.2f}%"


def _to_markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    lines = [
            "# RSCD Wavelet Formal Warning",
            "",
            result["claim_boundary"],
            "",
            f"- Run: `{result['run']}`",
            f"- Status: `{result['status']}`",
            f"- Latest epoch: `{result['latest_epoch']}`",
            f"- Latest validation Top-1 / Mean-F1: {_pct(result['latest_top1'])} / {_pct(result['latest_macro_f1'])}",
            f"- Best epoch: `{result['best_epoch']}`",
            f"- Best validation Top-1 / Mean-F1: {_pct(result['best_top1'])} / {_pct(result['best_macro_f1'])}",
            f"- Stale epochs since best: `{result['stale_epochs']}`",
            f"- Best validation gap to PhysicsTexture formal test Top-1: {_pct(result['best_gap_top1_vs_physics_test'], signed=True)}",
            f"- Best validation gap to PhysicsTexture formal test Mean-F1: {_pct(result['best_gap_macro_f1_vs_physics_test'], signed=True)}",
    ]
    if result.get("test_top1") is not None:
        lines.extend(
            [
                "",
                "## Final Test Evidence",
                "",
                f"- Candidate test Top-1 / Mean-F1: {_pct(result['test_top1'])} / {_pct(result['test_macro_f1'])}",
                f"- PhysicsTexture test Top-1 / Mean-F1: {_pct(result['physics_reference_top1'])} / {_pct(result['physics_reference_macro_f1'])}",
                f"- Test gap Top-1 / Mean-F1: {_pct(result['test_gap_top1_vs_physics'], signed=True)} / {_pct(result['test_gap_macro_f1_vs_physics'], signed=True)}",
                f"- Candidate wet/water F1: {_pct(result['test_wet_water_f1'])}",
                f"- PhysicsTexture wet/water F1: {_pct(result['physics_reference_wet_water_f1'])}",
                f"- Wet/water F1 gap: {_pct(result['test_gap_wet_water_f1_vs_physics'], signed=True)}",
                f"- Candidate water-concrete F1: {_pct(result['test_water_concrete_f1'])}",
                f"- PhysicsTexture water-concrete F1: {_pct(result['physics_reference_water_concrete_f1'])}",
                f"- Water-concrete F1 gap: {_pct(result['test_gap_water_concrete_f1_vs_physics'], signed=True)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Status: `{decision['status']}`",
            f"- Action: {decision['action']}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

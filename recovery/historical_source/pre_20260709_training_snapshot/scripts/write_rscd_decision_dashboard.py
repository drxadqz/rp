from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUMMARY = Path("reports/paper_protocol_summary")
ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT_JSON = SUMMARY / "rscd_decision_dashboard.json"
OUT_MD = SUMMARY / "rscd_decision_dashboard.md"


def main() -> None:
    trend = _load_json(SUMMARY / "rscd_training_trend_report.json") or {}
    fast = _load_json(SUMMARY / "rscd_surface_candidate_comparison.json") or {}
    formal = _load_json(SUMMARY / "rscd_formal_result_summary.json") or {}
    promotion = _load_json(SUMMARY / "rscd_fast_promotion_decision.json") or {}
    hard_promotion = _load_json(SUMMARY / "rscd_hard_condition_promotion_decision.json") or {}
    residual_promotion = _load_json(SUMMARY / "rscd_residual_adapter_promotion_decision.json") or {}
    film_promotion = _load_json(SUMMARY / "rscd_texture_film_promotion_decision.json") or {}

    formal_files = _formal_eval_files()
    dashboard = {
        "claim_boundary": (
            "RSCD-27 dashboard for local class-label experiments. Final paper claims "
            "require formal evaluate_test.json and matched protocol evidence."
        ),
        "formal_training": _formal_training_summary(trend),
        "fast_reference": _fast_reference_summary(fast),
        "formal_results": {
            "status": "available" if formal_files else "waiting",
            "evaluate_test_files": formal_files,
            "summary_decision": formal.get("decision"),
            "external_sota": formal.get("external_sota", []),
            "local_rows": formal.get("local_rows", []),
        },
        "promotion": {
            "standard_candidates": _promotion_summary(promotion),
            "hard_condition_candidates": _promotion_summary(hard_promotion),
            "residual_adapter_candidates": _promotion_summary(residual_promotion),
            "texture_film_candidates": _promotion_summary(film_promotion),
        },
        "current_decision": _decision(
            trend,
            fast,
            formal_files,
            promotion,
            hard_promotion,
            residual_promotion,
            film_promotion,
        ),
        "next_actions": _next_actions(formal_files, promotion, hard_promotion, residual_promotion, film_promotion),
    }
    OUT_JSON.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(dashboard), encoding="utf-8")
    print(OUT_MD)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _formal_eval_files() -> list[dict[str, Any]]:
    rows = []
    if not ROOT.exists():
        return rows
    for path in sorted(ROOT.glob("formal_*/evaluate_test.json")):
        payload = _load_json(path) or {}
        rows.append(
            {
                "run": path.parent.name,
                "path": str(path),
                "summary": payload.get("summary"),
                "last_write_time": path.stat().st_mtime,
            }
        )
    return rows


def _formal_training_summary(trend: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in trend.get("runs", []):
        latest = row.get("latest") or {}
        best = row.get("best") or {}
        out.append(
            {
                "name": row.get("name"),
                "status": row.get("status"),
                "latest_epoch": latest.get("epoch"),
                "latest_top1": latest.get("top1"),
                "latest_macro_f1": latest.get("macro_f1"),
                "best_epoch": best.get("epoch"),
                "best_macro_f1": best.get("macro_f1"),
            }
        )
    return out


def _fast_reference_summary(fast: dict[str, Any]) -> dict[str, Any]:
    rows = fast.get("rows", [])
    best = rows[0] if rows else None
    convnext = next((r for r in rows if r.get("name") == "fast_convnext_tiny"), None)
    physics = next((r for r in rows if r.get("name") == "fast_physics_texture_quality"), None)
    return {
        "best_fast": _pick_metrics(best),
        "fast_convnext": _pick_metrics(convnext),
        "fast_physics_texture": _pick_metrics(physics),
        "physics_delta_top1_vs_convnext": (
            physics.get("top1") - convnext.get("top1")
            if physics and convnext and physics.get("top1") is not None and convnext.get("top1") is not None
            else None
        ),
        "physics_delta_macro_f1_vs_convnext": (
            physics.get("macro_f1") - convnext.get("macro_f1")
            if physics and convnext and physics.get("macro_f1") is not None and convnext.get("macro_f1") is not None
            else None
        ),
    }


def _pick_metrics(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "name": row.get("name"),
        "top1": row.get("top1"),
        "mean_precision": row.get("mean_precision"),
        "mean_recall": row.get("mean_recall"),
        "macro_f1": row.get("macro_f1"),
        "num_samples": row.get("num_samples"),
    }


def _promotion_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference": data.get("reference"),
        "rule": data.get("promotion_rule"),
        "promoted": data.get("promoted"),
        "missing": [r.get("name") for r in data.get("rows", []) if r.get("status") == "missing"],
        "available": [_pick_metrics(r) for r in data.get("rows", []) if r.get("status") == "available"],
    }


def _decision(
    trend: dict[str, Any],
    fast: dict[str, Any],
    formal_files: list[dict[str, Any]],
    promotion: dict[str, Any],
    hard_promotion: dict[str, Any],
    residual_promotion: dict[str, Any],
    film_promotion: dict[str, Any],
) -> dict[str, Any]:
    pending = _pending_promoted_formals([promotion, hard_promotion, residual_promotion, film_promotion])
    if formal_files:
        if pending:
            return {
                "status": "formal_results_available_promoted_candidate_pending",
                "message": (
                    "Baseline formal test files exist, but at least one promoted candidate still lacks "
                    f"formal evaluate_test.json: {', '.join(pending)}. Do not make the final module "
                    "selection until the promoted formal run finishes."
                ),
            }
        return {
            "status": "formal_results_available",
            "message": (
                "Formal test files exist. Run compare_rscd_surface_candidates.py, "
                "compare_rscd_class_slices.py, and write_rscd_formal_result_summary.py "
                "before any paper claim."
            ),
        }
    trend_runs = trend.get("runs", [])
    baseline = next((r for r in trend_runs if "convnext" in str(r.get("name", "")).lower()), None)
    physics = next((r for r in trend_runs if "physics" in str(r.get("name", "")).lower()), None)
    base_f1 = ((baseline or {}).get("latest") or {}).get("macro_f1")
    phys_f1 = ((physics or {}).get("latest") or {}).get("macro_f1")
    if base_f1 is not None and phys_f1 is not None:
        gap = float(phys_f1) - float(base_f1)
        if gap >= 0:
            message = "Validation trend is close and currently favors PhysicsTexture, but final test is required."
        else:
            message = "Validation trend is close but currently favors baseline; keep PhysicsTexture only if final test or hard slices support it."
        return {
            "status": "waiting_for_formal_test",
            "validation_macro_f1_gap_physics_minus_baseline": gap,
            "message": message,
        }
    return {
        "status": "waiting_for_formal_test",
        "message": "Formal training is active or incomplete; no final test claim is allowed.",
    }


def _next_actions(
    formal_files: list[dict[str, Any]],
    promotion: dict[str, Any],
    hard_promotion: dict[str, Any],
    residual_promotion: dict[str, Any],
    film_promotion: dict[str, Any],
) -> list[str]:
    pending = _pending_promoted_formals([promotion, hard_promotion, residual_promotion, film_promotion])
    if formal_files:
        if pending:
            return [
                f"Wait for promoted formal candidate(s): {', '.join(pending)}.",
                "Regenerate formal result summary and class-slice comparison after the promoted result appears.",
                "Then decide whether to replace PhysicsTexture or keep it as the main module.",
                "Only compare with RoadFormer/RoadMamba as strict SOTA after protocol matching is documented.",
            ]
        return [
            "Regenerate formal result summary and class-slice comparison.",
            "Decide whether PhysicsTexture is a general module, hard-slice module, or prune candidate.",
            "Only compare with RoadFormer/RoadMamba as strict SOTA after protocol matching is documented.",
        ]
    actions = [
        "Continue formal RSCD jobs; do not kill healthy GPU processes.",
        "Wait for DirectionalTexture/hierarchical/gated fast candidates after formal jobs finish.",
        "Promote candidates only through the >0.1 percentage point fast-gain gate.",
    ]
    if hard_promotion.get("promoted"):
        actions.append("Run formal hard-condition candidate after the existing queue clears.")
    else:
        actions.append("Keep hard-condition boost queued as a hypothesis, not a claim.")
    if promotion.get("promoted"):
        actions.append("Run the promoted standard candidate formally before paper claims.")
    if residual_promotion.get("promoted"):
        actions.append("Run the promoted residual-adapter candidate formally before paper claims.")
    else:
        actions.append("Keep residual-adapter queued as the conservative replacement for unstable direct texture concatenation.")
    if film_promotion.get("promoted"):
        actions.append("Run the promoted texture-FiLM candidate formally before paper claims.")
    else:
        actions.append("Keep texture-FiLM queued as the zero-initialized modulation alternative to direct texture concatenation.")
    return actions


def _pending_promoted_formals(promotions: list[dict[str, Any]]) -> list[str]:
    pending = []
    for data in promotions:
        promoted = data.get("promoted") if isinstance(data, dict) else None
        if not promoted:
            continue
        out_dir = promoted.get("formal_output_dir")
        name = promoted.get("name") or out_dir
        if out_dir and not (Path(out_dir) / "evaluate_test.json").exists():
            pending.append(str(name))
    return pending


def _to_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# RSCD Decision Dashboard",
        "",
        data["claim_boundary"],
        "",
        "## Current Decision",
        "",
        f"- Status: `{data['current_decision']['status']}`",
        f"- Message: {data['current_decision']['message']}",
    ]
    gap = data["current_decision"].get("validation_macro_f1_gap_physics_minus_baseline")
    if gap is not None:
        lines.append(f"- Validation Macro-F1 gap, PhysicsTexture minus baseline: {_pct(gap, signed=True)}")

    lines += [
        "",
        "## Formal Training Trend",
        "",
        "| run | status | latest epoch | latest Top-1 | latest Macro-F1 | best Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in data["formal_training"]:
        lines.append(
            "| `{name}` | {status} | {epoch} | {top1} | {mf1} | {best} |".format(
                name=row.get("name"),
                status=row.get("status"),
                epoch=row.get("latest_epoch") if row.get("latest_epoch") is not None else "-",
                top1=_pct(row.get("latest_top1")),
                mf1=_pct(row.get("latest_macro_f1")),
                best=_pct(row.get("best_macro_f1")),
            )
        )

    fast_ref = data["fast_reference"]
    lines += [
        "",
        "## Fast Evidence",
        "",
        f"- Fast PhysicsTexture dTop-1 vs ConvNeXt: {_pct(fast_ref.get('physics_delta_top1_vs_convnext'), signed=True)}",
        f"- Fast PhysicsTexture dMacro-F1 vs ConvNeXt: {_pct(fast_ref.get('physics_delta_macro_f1_vs_convnext'), signed=True)}",
    ]

    lines += [
        "",
        "## Formal Test Files",
        "",
    ]
    if data["formal_results"]["evaluate_test_files"]:
        for row in data["formal_results"]["evaluate_test_files"]:
            summary = row.get("summary") or {}
            lines.append(f"- `{row['run']}`: Top-1 {_pct(summary.get('top1'))}, Macro-F1 {_pct(summary.get('macro_f1'))}")
    else:
        lines.append("- None yet.")

    lines += [
        "",
        "## Promotion State",
        "",
        f"- Standard promoted: `{_promoted_name(data['promotion']['standard_candidates'])}`",
        f"- Standard missing: {', '.join(data['promotion']['standard_candidates'].get('missing') or []) or 'none'}",
        f"- Hard-condition promoted: `{_promoted_name(data['promotion']['hard_condition_candidates'])}`",
        f"- Hard-condition missing: {', '.join(data['promotion']['hard_condition_candidates'].get('missing') or []) or 'none'}",
        f"- Residual-adapter promoted: `{_promoted_name(data['promotion']['residual_adapter_candidates'])}`",
        f"- Residual-adapter missing: {', '.join(data['promotion']['residual_adapter_candidates'].get('missing') or []) or 'none'}",
        f"- Texture-FiLM promoted: `{_promoted_name(data['promotion']['texture_film_candidates'])}`",
        f"- Texture-FiLM missing: {', '.join(data['promotion']['texture_film_candidates'].get('missing') or []) or 'none'}",
        "",
        "## Next Actions",
        "",
    ]
    lines.extend(f"- {item}" for item in data["next_actions"])
    lines.append("")
    return "\n".join(lines)


def _promoted_name(data: dict[str, Any]) -> str:
    promoted = data.get("promoted")
    if not promoted:
        return "none"
    return str(promoted.get("name") or promoted)


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

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT = Path("reports/paper_protocol_summary/rscd_texture_film_promotion_decision")

CANDIDATES = {
    "fast_physics_texture_film": [
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim",
        "96",
        "--use-texture-film",
        "--texture-film-scale",
        "0.20",
    ],
    "fast_physics_directional_film_gate_hier": [
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim",
        "96",
        "--use-directional-texture-branch",
        "--directional-texture-dim",
        "64",
        "--use-texture-gate",
        "--use-texture-film",
        "--texture-film-scale",
        "0.20",
        "--hierarchical-smoothing",
        "0.08",
    ],
    "fast_physics_wavelet_film": [
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim",
        "96",
        "--use-wavelet-texture-branch",
        "--wavelet-texture-dim",
        "48",
        "--use-texture-film",
        "--texture-film-scale",
        "0.20",
    ],
    "fast_physics_wavelet_directional_film_gate_hier": [
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim",
        "96",
        "--use-wavelet-texture-branch",
        "--wavelet-texture-dim",
        "48",
        "--use-directional-texture-branch",
        "--directional-texture-dim",
        "48",
        "--use-texture-gate",
        "--use-texture-film",
        "--texture-film-scale",
        "0.20",
        "--hierarchical-smoothing",
        "0.08",
    ],
    "fast_physics_attention_film": [
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim",
        "96",
        "--use-physics-attention-branch",
        "--physics-attention-dim",
        "32",
        "--use-texture-film",
        "--texture-film-scale",
        "0.20",
    ],
    "fast_physics_attention_wavelet_film_gate_hier": [
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim",
        "96",
        "--use-physics-attention-branch",
        "--physics-attention-dim",
        "32",
        "--use-wavelet-texture-branch",
        "--wavelet-texture-dim",
        "48",
        "--use-texture-gate",
        "--use-texture-film",
        "--texture-film-scale",
        "0.20",
        "--hierarchical-smoothing",
        "0.08",
    ],
}


def main() -> None:
    baseline = load_summary("fast_convnext_tiny")
    reference = load_summary("fast_physics_texture_quality")
    rows = []
    for name, args in CANDIDATES.items():
        summary = load_summary(name)
        if summary is None:
            rows.append({"name": name, "status": "missing"})
            continue
        rows.append(
            {
                "name": name,
                "status": "available",
                "top1": summary["top1"],
                "macro_f1": summary["macro_f1"],
                "balanced_accuracy": summary.get("balanced_accuracy"),
                "delta_top1_vs_baseline": summary["top1"] - baseline["top1"] if baseline else None,
                "delta_macro_f1_vs_baseline": summary["macro_f1"] - baseline["macro_f1"] if baseline else None,
                "delta_top1_vs_physics": summary["top1"] - reference["top1"] if reference else None,
                "delta_macro_f1_vs_physics": summary["macro_f1"] - reference["macro_f1"] if reference else None,
                "formal_args": args,
            }
        )

    promoted = None
    available = [row for row in rows if row.get("status") == "available"]
    if baseline and reference and available:
        best = max(available, key=lambda row: (float(row["macro_f1"]), float(row["top1"])))
        macro_gain_vs_physics = float(best["macro_f1"]) - float(reference["macro_f1"])
        top1_gain_vs_physics = float(best["top1"]) - float(reference["top1"])
        macro_gain_vs_baseline = float(best["macro_f1"]) - float(baseline["macro_f1"])
        top1_gain_vs_baseline = float(best["top1"]) - float(baseline["top1"])
        beats_physics = macro_gain_vs_physics >= 0.005 or (
            top1_gain_vs_physics >= 0.006 and macro_gain_vs_physics >= 0.003
        )
        beats_baseline = macro_gain_vs_baseline >= 0.010 or (
            top1_gain_vs_baseline >= 0.010 and macro_gain_vs_baseline >= 0.006
        )
        if beats_physics and beats_baseline:
            promoted = {
                "name": best["name"],
                "formal_output_dir": str(ROOT / best["name"].replace("fast_", "formal_") / ""),
                "formal_args": best["formal_args"],
                "macro_gain_vs_physics": macro_gain_vs_physics,
                "top1_gain_vs_physics": top1_gain_vs_physics,
                "macro_gain_vs_baseline": macro_gain_vs_baseline,
                "top1_gain_vs_baseline": top1_gain_vs_baseline,
            }

    result = {
        "references": {
            "baseline": "fast_convnext_tiny",
            "physics": "fast_physics_texture_quality",
        },
        "promotion_rule": (
            "Promote texture-FiLM/Wavelet only if the fast screen shows a nontrivial "
            "gain over direct PhysicsTexture: at least +0.5pp Macro-F1, or at least "
            "+0.6pp Top-1 with at least +0.3pp Macro-F1. It must also beat the clean "
            "ConvNeXt fast baseline by at least +1.0pp Macro-F1, or +1.0pp Top-1 with "
            "at least +0.6pp Macro-F1. "
            "Wavelet candidates are included as an RSPNet-inspired frequency/detail-preserving "
            "route; PhysicsAttention candidates test a weak segmentation + attention + physics "
            "route without pixel labels. RSCD is treated as a patch dataset, so future "
            "formal arguments disable bottom-vs-top PhysicsTexture region cues. Final retention "
            "is stricter and also checks wet/water hard-slice regressions."
        ),
        "rows": rows,
        "promoted": promoted,
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def load_summary(name: str) -> dict | None:
    path = ROOT / name / "evaluate_test.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("summary")


def pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def to_markdown(result: dict) -> str:
    lines = [
        "# RSCD Texture-FiLM Promotion Decision",
        "",
        f"Baseline reference: `{result['references']['baseline']}`",
        f"Physics reference: `{result['references']['physics']}`",
        "",
        result["promotion_rule"],
        "",
        "| candidate | status | Top-1 | Macro-F1 | dTop1 vs base | dMacro-F1 vs base | dTop1 vs physics | dMacro-F1 vs physics |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            "| `{name}` | {status} | {top1} | {mf1} | {dtb} | {dfb} | {dtp} | {dfp} |".format(
                name=row["name"],
                status=row["status"],
                top1=pct(row.get("top1")),
                mf1=pct(row.get("macro_f1")),
                dtb=pct(row.get("delta_top1_vs_baseline"), signed=True),
                dfb=pct(row.get("delta_macro_f1_vs_baseline"), signed=True),
                dtp=pct(row.get("delta_top1_vs_physics"), signed=True),
                dfp=pct(row.get("delta_macro_f1_vs_physics"), signed=True),
            )
        )
    lines.append("")
    if result["promoted"]:
        promoted = result["promoted"]
        lines.append(f"Promoted: `{promoted['name']}`")
        lines.append("")
        lines.append(f"Formal output dir: `{promoted['formal_output_dir']}`")
    else:
        lines.append("Promoted: none yet.")
        legacy_formal = ROOT / "formal_physics_wavelet_directional_film_gate_hier"
        if legacy_formal.exists():
            lines.append("")
            lines.append(
                "Note: `formal_physics_wavelet_directional_film_gate_hier` was already launched under "
                "the earlier lenient fast-promotion rule. It is now treated as a legacy promoted "
                "candidate and must pass the stricter final retention gate before it can replace "
                "`PhysicsTexture`."
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

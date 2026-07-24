from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_ROOT = Path("D:/NMI_SPWFM_datasets/friction_affordance_outputs/paper_protocol")

P0_METHOD_TO_RUN = {
    "Global-only": "v0_global_only",
    "+ PhysicsTexture": "v1_physics_texture",
    "+ FrictionSet": "v2_friction_set",
    "+ DG losses": "v3_dg_losses",
    "+ EvidenceField aux": "v4_evidence_aux",
    "Full model": "v5_full_faf",
}

COMPACT_METRIC_ORDER = [
    "v0_global_only",
    "v1_physics_texture",
    "v2_friction_set",
    "v3_dg_losses",
    "v4_evidence_aux",
    "v5_full_faf",
    "lodo_roadsaw_full_faf",
    "lodo_rscd_full_faf",
    "lodo_roadsc_full_faf",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "quality_domain_diagnostic_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "quality_domain_diagnostic_report.json")
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path, summary_dir: Path) -> dict[str, Any]:
    quality = _load_json(summary_dir / "image_quality_flags_roadsaw_roadsc_test.json") or {}
    lodo_roadsaw = _load_json(root / "lodo_roadsaw_full_faf" / "quality_slices" / "quality_slices_test.json") or {}
    v1_roadsaw = _load_json(root / "v1_physics_texture" / "quality_slices_roadsaw_test" / "quality_slices_test.json") or {}
    bootstrap_metrics = {
        "v0_global_only": _load_json(root / "v0_global_only" / "bootstrap_metrics.json"),
        "v1_physics_texture": _load_json(root / "v1_physics_texture" / "bootstrap_metrics.json"),
        "v5_full_faf": _load_json(root / "v5_full_faf" / "bootstrap_metrics.json"),
        "lodo_rscd_full_faf": _load_json(root / "lodo_rscd_full_faf" / "bootstrap_metrics.json"),
        "lodo_roadsaw_full_faf": _load_json(root / "lodo_roadsaw_full_faf" / "bootstrap_metrics.json"),
    }
    p0_metrics = {key: _compact_bootstrap(value) for key, value in bootstrap_metrics.items() if value}
    p0_metrics.update(_paper_p0_metrics(summary_dir))
    return {
        "claim_boundary": (
            "This report diagnoses available local artifacts. It supports route "
            "selection and ablation decisions, but it is not a replacement for "
            "the remaining single-dataset and baseline experiments."
        ),
        "quality_summary": quality,
        "quality_slices": {
            "v1_physics_texture_on_roadsaw_test": v1_roadsaw,
            "lodo_roadsaw_full_faf_on_roadsaw_test": lodo_roadsaw,
        },
        "p0_metrics": p0_metrics,
        "diagnosis": _diagnosis(quality, v1_roadsaw, lodo_roadsaw, p0_metrics),
        "next_actions": _next_actions(),
    }


def _diagnosis(
    quality: dict[str, Any],
    v1_roadsaw: dict[str, Any],
    lodo_roadsaw: dict[str, Any],
    p0: dict[str, Any],
) -> list[dict[str, str]]:
    roadsaw_quality = ((quality.get("by_dataset") or {}).get("roadsaw") or {})
    v1_all = _slice(v1_roadsaw, "all")
    v1_near = _slice(v1_roadsaw, "near_white")
    v1_normal = _slice(v1_roadsaw, "normal_quality")
    lodo_all = _slice(lodo_roadsaw, "all")
    lodo_normal = _slice(lodo_roadsaw, "normal_quality")
    return [
        {
            "finding": "RoadSaW contains systematic near-white/low-contrast samples.",
            "evidence": (
                f"RoadSaW test near-white rate is {_pct(roadsaw_quality.get('near_white_rate'))}; "
                f"suspicious quality rate is {_pct(roadsaw_quality.get('suspicious_quality_rate'))}. "
                "The highest near-white concentration is in wet concrete and very-wet asphalt classes."
            ),
            "implication": (
                "RoadSaW should keep a quality-aware slice in every table. "
                "Near-white samples are valid stress cases for wet-road reflection/overexposure, "
                "not rows to silently discard."
            ),
        },
        {
            "finding": "Near-white images hurt performance, but they are not the only reason LODO fails.",
            "evidence": (
                f"v1 on RoadSaW test has friction F1 {_pct(_metric(v1_all, 'classification', 'friction', 'macro_f1'))}; "
                f"normal-quality RoadSaW is {_pct(_metric(v1_normal, 'classification', 'friction', 'macro_f1'))}; "
                f"near-white RoadSaW drops to {_pct(_metric(v1_near, 'classification', 'friction', 'macro_f1'))}. "
                f"LODO-RoadSaW normal-quality friction F1 is only "
                f"{_pct(_metric(lodo_normal, 'classification', 'friction', 'macro_f1'))}."
            ),
            "implication": (
                "The main weakness is domain generalization from RSCD/RoadSC to RoadSaW. "
                "Quality handling is necessary, but not sufficient."
            ),
        },
        {
            "finding": "The current strongest retained module is PhysicsTexture, not the full stack.",
            "evidence": (
                f"v1 PhysicsTexture friction F1 is {_pct(_metric(p0, 'v1_physics_texture', 'friction_f1'))}; "
                f"v5 Full FAF is {_pct(_metric(p0, 'v5_full_faf', 'friction_f1'))}; "
                f"LODO-RoadSaW friction F1 is {_pct(_metric(p0, 'lodo_roadsaw_full_faf', 'friction_f1'))}."
            ),
            "implication": (
                "The paper should not claim that every proposed module is useful. "
                "FrictionSet, generic DG losses, and Full fusion need removal, merging, or redesign."
            ),
        },
    ]


def _next_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "P0",
            "action": "Let the active single-dataset/fair-baseline queue finish, then run paired FAF-vs-ConvNeXt comparisons.",
            "reason": "LODO is complete and failed; same-split fair baselines are now the minimum evidence package for claiming method advantage.",
        },
        {
            "priority": "P0",
            "action": "Run quality-slice evaluation for every completed main run and baseline.",
            "reason": "RoadSaW near-white and low-contrast slices are material failure modes, especially for wet-road friction.",
        },
        {
            "priority": "P1",
            "action": "Run v17_lean_quality_physics_safety after the core single/baseline rows.",
            "reason": "It promotes PhysicsTexture into a quality-aware wet-road branch for RoadSaW near-white wet patches and RoadSC low-texture snow patches.",
        },
        {
            "priority": "P1",
            "action": "Replace generic domain alignment with conditional alignment or dataset-specific adapters.",
            "reason": "LODO-RoadSaW collapse shows that a shared representation has not learned dataset-invariant road friction evidence.",
        },
        {
            "priority": "P2",
            "action": "Rework EvidenceField around road/contact ROI and augmentation consistency.",
            "reason": "Evidence maps are publishable only if attention aligns with physically meaningful road regions and improves metrics or diagnostics.",
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Quality And Domain Diagnostic Report",
        "",
        f"Boundary: {report['claim_boundary']}",
        "",
        "## Key Diagnosis",
        "",
    ]
    for item in report["diagnosis"]:
        lines += [
            f"### {item['finding']}",
            "",
            f"Evidence: {item['evidence']}",
            "",
            f"Implication: {item['implication']}",
            "",
        ]
    lines += [
        "## Compact Metrics",
        "",
        "| run | friction F1 | risk F1 | low recall | calibrated coverage | worst dataset F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in _ordered_metrics(report["p0_metrics"]):
        lines.append(
            f"| {name} | {_pct(row.get('friction_f1'))} | {_pct(row.get('risk_f1'))} | "
            f"{_pct(row.get('low_recall'))} | {_pct(row.get('calibrated_coverage'))} | "
            f"{_pct(row.get('worst_dataset_f1'))} |"
        )
    lines += ["", "## Next Actions", ""]
    for item in report["next_actions"]:
        lines.append(f"- {item['priority']}: {item['action']} Reason: {item['reason']}")
    return "\n".join(lines) + "\n"


def _compact_bootstrap(data: dict[str, Any] | None) -> dict[str, float | None]:
    if not data:
        return {}
    return {
        "friction_f1": _metric(data, "classification", "friction", "macro_f1", "point"),
        "risk_f1": _metric(data, "classification", "risk", "macro_f1", "point"),
        "low_recall": _metric(data, "low_friction_detection", "recall", "point"),
        "calibrated_coverage": _metric(data, "mu_interval", "calibrated_coverage", "point"),
        "worst_dataset_f1": _metric(data, "classification", "friction", "worst_dataset_macro_f1", "point"),
    }


def _paper_p0_metrics(summary_dir: Path) -> dict[str, dict[str, float | None]]:
    data = _load_json(summary_dir / "paper_p0_ablation_table.json") or {}
    out: dict[str, dict[str, float | None]] = {}
    for row in data.get("rows") or []:
        run = P0_METHOD_TO_RUN.get(str(row.get("method") or ""))
        if not run:
            continue
        out[run] = {
            "friction_f1": row.get("friction_macro_f1"),
            "risk_f1": row.get("risk_macro_f1"),
            "low_recall": row.get("low_friction_recall"),
            "calibrated_coverage": row.get("calibrated_coverage"),
            "worst_dataset_f1": row.get("worst_dataset_f1"),
        }
    return out


def _slice(data: dict[str, Any], name: str) -> dict[str, Any]:
    return ((data.get("slices") or {}).get(name) or {})


def _metric(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _ordered_metrics(rows: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    emitted = set()
    out = []
    for name in COMPACT_METRIC_ORDER:
        if name in rows:
            out.append((name, rows[name]))
            emitted.add(name)
    out.extend((name, row) for name, row in rows.items() if name not in emitted)
    return out


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    main()

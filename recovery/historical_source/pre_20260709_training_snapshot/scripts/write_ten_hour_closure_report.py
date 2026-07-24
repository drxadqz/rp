from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SUMMARY_DIR = Path("reports/paper_protocol_summary")
ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")


P0_ROWS = [
    ("Global-only", "v0_global_only"),
    ("+ PhysicsTexture", "v1_physics_texture"),
    ("+ FrictionSet", "v2_friction_set"),
    ("+ DG losses", "v3_dg_losses"),
    ("+ EvidenceField aux", "v4_evidence_aux"),
    ("Full model", "v5_full_faf"),
]

SINGLE_ROWS = [
    ("RoadSaW", "single_roadsaw_full_faf", "baseline_single_roadsaw_global_convnext"),
    ("RSCD", "single_rscd_full_faf", "baseline_single_rscd_global_convnext"),
    ("RoadSC", "single_roadsc_full_faf", "baseline_single_roadsc_global_convnext"),
]

LODO_ROWS = [
    ("held-out RoadSaW", "lodo_roadsaw_full_faf"),
    ("held-out RSCD", "lodo_rscd_full_faf"),
    ("held-out RoadSC", "lodo_roadsc_full_faf"),
]


def main() -> None:
    report = build_report()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    md_path = SUMMARY_DIR / "ten_hour_closure_report.md"
    json_path = SUMMARY_DIR / "ten_hour_closure_report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md_path)
    print(json_path)


def build_report() -> dict[str, Any]:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claim_boundary": (
            "Current results support public-label visual friction-affordance interval estimation. "
            "They do not support a measured tire-road friction-coefficient claim."
        ),
        "dataset_audit": dataset_audit(),
        "p0_ablation": p0_ablation(),
        "lodo": lodo_results(),
        "single_dataset_fairness": single_dataset_fairness(),
        "module_decisions": module_decisions(),
        "ten_hour_execution": ten_hour_execution(),
        "paper_story": paper_story(),
        "source_anchors": source_anchors(),
    }


def dataset_audit() -> dict[str, Any]:
    path_report = _load_json(SUMMARY_DIR / "dataset_path_completeness_report.json") or {}
    view_report = _load_json(SUMMARY_DIR / "dataset_integrity_view_audit.json") or {}
    style_report = _load_json(SUMMARY_DIR / "dataset_image_style_audit.json") or {}
    return {
        "path_completeness": {
            "rows": _dig(path_report, "overall", "rows"),
            "unique_paths": _dig(path_report, "overall", "unique_paths"),
            "missing_unique_paths": _dig(path_report, "overall", "missing_unique_paths"),
            "datasets": _dig(path_report, "datasets") or {},
        },
        "view_and_quality": {
            "datasets": _dig(view_report, "datasets") or {},
            "recommendation": _dig(view_report, "recommendation") or {},
        },
        "style_gap": {
            "cross_dataset_signals": _dig(style_report, "cross_dataset_signals") or {},
            "recommendations": _dig(style_report, "recommendations") or [],
        },
        "interpretation": [
            "All manifest-referenced image paths are present; label/interval audits report no invalid labels.",
            "RoadSaW near-white images are concentrated in wet/very-wet concrete/cobble/asphalt states and should be treated as a hard wetness/optical-quality slice, not automatically discarded.",
            "RSCD should be described conservatively as local/narrow road-surface patches from vehicle imagery; local evidence does not prove left-wheel or right-wheel camera placement.",
            "RoadSaW and RoadSC are square prepared vehicle-scene/road crops; RSCD has a different 240x360 narrow-patch geometry. Use hierarchical evaluation, not naive pooling.",
        ],
    }


def p0_ablation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, run in P0_ROWS:
        metrics = _run_metrics(run)
        rows.append(
            {
                "method": label,
                "run": run,
                "status": metrics["status"],
                "friction_f1": metrics["friction_f1"],
                "risk_f1": metrics["risk_f1"],
                "low_friction_recall": metrics["low_friction_recall"],
                "calibrated_coverage": metrics["calibrated_coverage"],
                "calibrated_width": metrics["calibrated_width"],
                "raw_coverage": metrics["raw_coverage"],
                "worst_dataset_f1": metrics["worst_dataset_f1"],
                "dataset_id_balanced_accuracy": _dataset_id(run),
            }
        )
    return rows


def lodo_results() -> list[dict[str, Any]]:
    out = []
    for label, run in LODO_ROWS:
        metrics = _run_metrics(run)
        out.append(
            {
                "held_out": label,
                "run": run,
                "status": metrics["status"],
                "friction_f1": metrics["friction_f1"],
                "risk_f1": metrics["risk_f1"],
                "low_friction_recall": metrics["low_friction_recall"],
                "raw_coverage": metrics["raw_coverage"],
                "calibrated_coverage": metrics["calibrated_coverage"],
                "calibrated_width": metrics["calibrated_width"],
                "interpretation": "OOD failure" if (metrics["risk_f1"] or 0.0) < 0.55 else "promising",
            }
        )
    return out


def single_dataset_fairness() -> list[dict[str, Any]]:
    rows = []
    for dataset, faf_run, baseline_run in SINGLE_ROWS:
        faf = _run_metrics(faf_run)
        baseline = _run_metrics(baseline_run)
        rows.append(
            {
                "dataset": dataset,
                "faf_run": faf_run,
                "baseline_run": baseline_run,
                "faf_status": faf["status"],
                "baseline_status": baseline["status"],
                "faf_friction_f1": faf["friction_f1"],
                "baseline_friction_f1": baseline["friction_f1"],
                "friction_f1_delta": _delta(faf["friction_f1"], baseline["friction_f1"]),
                "faf_risk_f1": faf["risk_f1"],
                "baseline_risk_f1": baseline["risk_f1"],
                "risk_f1_delta": _delta(faf["risk_f1"], baseline["risk_f1"]),
                "faf_coverage": faf["calibrated_coverage"],
                "baseline_coverage": baseline["calibrated_coverage"],
                "coverage_delta": _delta(faf["calibrated_coverage"], baseline["calibrated_coverage"]),
            }
        )
    return rows


def module_decisions() -> list[dict[str, str]]:
    return [
        {
            "module": "PhysicsTexture",
            "decision": "keep_core",
            "evidence": "Best P0 friction F1, low-friction recall, and worst-dataset F1; improves raw coverage strongly.",
            "next": "Use as the protected base for all final lean variants.",
        },
        {
            "module": "FrictionSet",
            "decision": "merge_or_rework",
            "evidence": "Improves risk F1 and calibrated coverage but hurts worst-dataset F1.",
            "next": "Keep only as an uncertainty/calibration branch if it does not reduce RoadSaW/LODO robustness.",
        },
        {
            "module": "Current generic DG losses",
            "decision": "remove_current_form",
            "evidence": "P0 DG losses reduce friction F1, risk F1, and low-friction recall.",
            "next": "Replace broad alignment with condition-aware CORAL, Fourier style jitter, MixStyle, and dataset-specific adapters tested by shortcut probes.",
        },
        {
            "module": "EvidenceField aux",
            "decision": "keep_as_reworked_local_evidence_route",
            "evidence": "Current aux branch does not beat PhysicsTexture, but produces interpretable evidence maps and enables v23/v24/v25 local-region candidates.",
            "next": "Use segmentation-style region mixture, multi-query evidence, masked consistency, and ROI constraints; prune if hard-slice metrics do not improve.",
        },
        {
            "module": "Full model v5",
            "decision": "not_final",
            "evidence": "Full fusion sharply hurts risk F1, low-friction recall, and worst-dataset F1.",
            "next": "Do not present v5 as final method; use it as a negative ablation motivating lean module selection.",
        },
    ]


def ten_hour_execution() -> list[dict[str, str]]:
    return [
        {
            "priority": "0",
            "action": "Let the current official GPU queue finish matched ConvNeXt baselines.",
            "why": "Single-dataset FAF-vs-ConvNeXt is the cleanest fair numeric comparison within the current public datasets.",
            "gate": "baseline_single_rscd_global_convnext and baseline_single_roadsc_global_convnext complete; refresh postprocess and fair deltas.",
        },
        {
            "priority": "1",
            "action": "Use P0 ablation + LODO as the immediate reportable core.",
            "why": "These are already complete and answer the reviewer questions: which module helps, and whether naive cross-dataset generalization works.",
            "gate": "Use PhysicsTexture as current best completed method; state LODO RoadSaW is a failure signal.",
        },
        {
            "priority": "2",
            "action": "Run fail-fast CV-transfer candidates after fair baselines, not all full candidates first.",
            "why": "The 10-hour target needs rapid evidence. Prioritize v17/v21/v23/v24/v25 over broad exhaustive training.",
            "gate": "Keep only modules that improve RoadSaW wet/near-white slices, low-friction recall, calibrated coverage-width, or dataset shortcut without task regression.",
        },
        {
            "priority": "3",
            "action": "Do RSCD-27 fast external comparison after current queue is idle.",
            "why": "RSCD is the only dataset here with a plausible external classification benchmark story; friction interval claims remain a separate weak-label protocol.",
            "gate": "Report Top-1, macro-F1, weighted-F1, balanced accuracy; no SOTA claim unless split/label/metric comparability is proven.",
        },
        {
            "priority": "4",
            "action": "Freeze final method only after pruning.",
            "why": "The current full model is worse than lean PhysicsTexture; final architecture should be lean and evidence-driven.",
            "gate": "Final method = PhysicsTexture + only the CV-transfer/ROI/interval modules that pass fast-screen and fair-baseline gates.",
        },
    ]


def paper_story() -> list[str]:
    return [
        "Problem: public visual road datasets rarely provide measured tire-road friction, so the academically defensible target is a weak visual friction-affordance interval.",
        "Data insight: RSCD, RoadSaW, and RoadSC are valid but not homogeneous. RSCD gives scale, RoadSaW gives wetness stress, RoadSC gives snow stress.",
        "Method insight: a global image classifier is insufficient; road friction cues are local material/optical evidence with uncertainty under wet glare, snow, and low texture.",
        "Current result: PhysicsTexture is the strongest completed module; full unfiltered fusion is a negative ablation, proving module pruning is necessary.",
        "Generalization result: LODO fails badly, which becomes the motivation for style canonicalization, condition-aware alignment, ROI attention, and masked local evidence.",
        "Next innovation: Segmentation-Transferred Friction Affordance Field, especially v25 masked multi-query evidence consistency, is the best near-term top-venue route.",
    ]


def source_anchors() -> list[dict[str, str]]:
    return [
        {"name": "RSCD official page", "url": "https://thu-rsxd.com/rscd/"},
        {"name": "RSCD dataset paper", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9343931/"},
        {"name": "RoadSaW", "url": "https://openaccess.thecvf.com/content/CVPR2022W/WAD/html/Cordes_RoadSaW_A_Large-Scale_Dataset_for_Camera-Based_Road_Surface_and_Wetness_CVPRW_2022_paper.html"},
        {"name": "RoadSC", "url": "https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/html/Cordes_Camera-Based_Road_Snow_Coverage_Estimation_ICCVW_2023_paper.html"},
        {"name": "Mask2Former", "url": "https://arxiv.org/abs/2112.01527"},
        {"name": "MIC", "url": "https://arxiv.org/abs/2212.01322"},
        {"name": "DINOv2", "url": "https://arxiv.org/abs/2304.07193"},
    ]


def _run_metrics(run: str) -> dict[str, Any]:
    run_dir = ROOT / run
    bootstrap = _load_json(run_dir / "bootstrap_metrics.json") or {}
    state = _load_json(run_dir / "training_state.json") or {}
    status = "complete" if (run_dir / "bootstrap_metrics.json").exists() else ("running_or_partial" if run_dir.exists() else "missing")
    return {
        "status": status,
        "epoch": state.get("epoch"),
        "friction_f1": _dig(bootstrap, "classification", "friction", "macro_f1", "point"),
        "risk_f1": _dig(bootstrap, "classification", "risk", "macro_f1", "point"),
        "worst_dataset_f1": min(
            [
                v
                for v in [
                    _dig(bootstrap, "classification", "friction", "worst_dataset_macro_f1", "point"),
                    _dig(bootstrap, "classification", "risk", "worst_dataset_macro_f1", "point"),
                ]
                if v is not None
            ],
            default=None,
        ),
        "low_friction_recall": _point_or_value(_dig(bootstrap, "low_friction_detection", "recall")),
        "raw_coverage": _dig(bootstrap, "mu_interval", "raw_coverage", "point"),
        "calibrated_coverage": _dig(bootstrap, "mu_interval", "calibrated_coverage", "point"),
        "calibrated_width": _dig(bootstrap, "mu_interval", "calibrated_width", "point"),
    }


def _dataset_id(run: str) -> float | None:
    data = _load_json(ROOT / run / "dataset_id_diagnostic.json") or {}
    for path in [
        ("balanced_accuracy",),
        ("dataset_id_balanced_accuracy",),
        ("linear_probe", "balanced_accuracy"),
    ]:
        value = _dig(data, *path)
        if value is not None:
            return value
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _dig(data: Any, *keys: str) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _point_or_value(value: Any) -> Any:
    if isinstance(value, dict) and "point" in value:
        return value.get("point")
    return value


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ten-Hour Closure Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Dataset Verdict",
        "",
    ]
    for item in report["dataset_audit"]["interpretation"]:
        lines.append(f"- {item}")
    rec = report["dataset_audit"]["view_and_quality"].get("recommendation") or {}
    if rec:
        lines.append(f"- Recommended route: `{rec.get('route')}`. {rec.get('decision')}")

    lines.extend(
        [
            "",
            "## P0 Ablation Core Table",
            "",
            "| Method | Status | friction F1 | risk F1 | low-friction recall | calibrated coverage | raw coverage | worst dataset F1 | dataset-ID bal acc | Decision |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    decisions = {
        "Global-only": "baseline",
        "+ PhysicsTexture": "current best completed core",
        "+ FrictionSet": "merge/rework for uncertainty only",
        "+ DG losses": "remove current form",
        "+ EvidenceField aux": "rework into local evidence route",
        "Full model": "not final",
    }
    for row in report["p0_ablation"]:
        lines.append(
            "| {method} | {status} | {ff1} | {rf1} | {low} | {cov} | {raw} | {worst} | {did} | {decision} |".format(
                method=row["method"],
                status=row["status"],
                ff1=_pct(row["friction_f1"]),
                rf1=_pct(row["risk_f1"]),
                low=_pct(row["low_friction_recall"]),
                cov=_pct(row["calibrated_coverage"]),
                raw=_pct(row["raw_coverage"]),
                worst=_pct(row["worst_dataset_f1"]),
                did=_pct(row["dataset_id_balanced_accuracy"]),
                decision=decisions.get(row["method"], "-"),
            )
        )

    lines.extend(
        [
            "",
            "## LODO Stress Test",
            "",
            "| Held-out | Status | friction F1 | risk F1 | low-friction recall | raw coverage | calibrated coverage | width | Interpretation |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["lodo"]:
        lines.append(
            "| {held} | {status} | {ff1} | {rf1} | {low} | {raw} | {cov} | {width} | {interp} |".format(
                held=row["held_out"],
                status=row["status"],
                ff1=_pct(row["friction_f1"]),
                rf1=_pct(row["risk_f1"]),
                low=_pct(row["low_friction_recall"]),
                raw=_pct(row["raw_coverage"]),
                cov=_pct(row["calibrated_coverage"]),
                width=_num(row["calibrated_width"]),
                interp=row["interpretation"],
            )
        )

    lines.extend(
        [
            "",
            "## Same-Split Single-Dataset Fairness",
            "",
            "| Dataset | FAF status | ConvNeXt status | FAF friction F1 | ConvNeXt friction F1 | delta | FAF risk F1 | ConvNeXt risk F1 | delta |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["single_dataset_fairness"]:
        lines.append(
            "| {dataset} | {fs} | {bs} | {fff1} | {bff1} | {fd} | {frf1} | {brf1} | {rd} |".format(
                dataset=row["dataset"],
                fs=row["faf_status"],
                bs=row["baseline_status"],
                fff1=_pct(row["faf_friction_f1"]),
                bff1=_pct(row["baseline_friction_f1"]),
                fd=_pct(row["friction_f1_delta"]),
                frf1=_pct(row["faf_risk_f1"]),
                brf1=_pct(row["baseline_risk_f1"]),
                rd=_pct(row["risk_f1_delta"]),
            )
        )

    lines.extend(["", "## Module Retention Decisions", ""])
    for row in report["module_decisions"]:
        lines.append(f"- `{row['module']}` -> `{row['decision']}`. Evidence: {row['evidence']} Next: {row['next']}")

    lines.extend(["", "## Ten-Hour Execution Plan", ""])
    for row in report["ten_hour_execution"]:
        lines.append(f"{int(row['priority']) + 1}. {row['action']} Gate: {row['gate']}")

    lines.extend(["", "## Paper Story", ""])
    lines.extend(f"- {item}" for item in report["paper_story"])

    lines.extend(["", "## Source Anchors", "", "| Source | URL |", "|---|---|"])
    for source in report["source_anchors"]:
        lines.append(f"| {source['name']} | {source['url']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PUBLIC_SOURCES = [
    {
        "name": "RoadSaW CVPRW 2022",
        "url": "https://openaccess.thecvf.com/content/CVPR2022W/WAD/papers/Cordes_RoadSaW_A_Large-Scale_Dataset_for_Camera-Based_Road_Surface_and_Wetness_CVPRW_2022_paper.pdf",
        "role": "wetness/surface public benchmark and held-out wet-road stress test",
    },
    {
        "name": "RSCD Figshare",
        "url": "https://figshare.com/articles/dataset/Road_Surface_Image_Dataset_with_Detailed_Annotations_for_Driving_Assistance/20424582",
        "role": "large road-surface patch benchmark with detailed public annotations",
    },
    {
        "name": "RSCD official site",
        "url": "https://thu-rsxd.com/rscd/",
        "role": "RSCD provenance and label/task description",
    },
    {
        "name": "RoadSC ICCVW 2023",
        "url": "https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/papers/Cordes_Camera-Based_Road_Snow_Coverage_Estimation_ICCVW_2023_paper.pdf",
        "role": "snow/winter road-patch stress benchmark",
    },
    {
        "name": "SIWNet road-friction prediction intervals",
        "url": "https://arxiv.org/html/2310.00923v2",
        "role": "interval-prediction inspiration; not a matched RSCD/RoadSaW/RoadSC baseline",
    },
    {
        "name": "WCamNet / DINOv2-style visual friction regression",
        "url": "https://arxiv.org/html/2404.16578v1",
        "role": "foundation-feature and direct-friction inspiration; not a matched public-data number",
    },
    {
        "name": "WCamNet GitHub",
        "url": "https://github.com/ojalar/wcamnet",
        "role": "open implementation reference for a DINOv2-plus-CNN visual friction model",
    },
    {
        "name": "RoadFormer RSCD local-global feature fusion",
        "url": "https://arxiv.org/html/2506.02358v1",
        "role": "RSCD-focused local/global texture-semantics architecture reference; not a friction-interval baseline",
    },
    {
        "name": "RoadMamba RSCD dual-branch state-space model",
        "url": "https://arxiv.org/html/2508.01210v1",
        "role": "RSCD-focused global/local branch and auxiliary-loss reference for future backbone comparisons",
    },
    {
        "name": "RSCD per-day split",
        "url": "https://github.com/MiviaLab/Road-Surface-Dataset-per-day-split",
        "role": "same-dataset RSCD generalization split that reduces acquisition-day leakage",
    },
]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pct(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value) * 100:.2f}"
    except (TypeError, ValueError):
        return default


def _num(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return default


def _list(items: list[str] | None) -> str:
    if not items:
        return "-"
    return ", ".join(f"`{item}`" for item in items)


def _active_text(dashboard: dict[str, Any]) -> str:
    active = dashboard.get("active_rows") or []
    if not active:
        return "No active training row is visible."
    row = active[0]
    parts = [f"`{row.get('name', '-')}`"]
    epoch = row.get("active_epoch")
    epochs = row.get("active_epochs")
    step = row.get("active_step")
    steps = row.get("active_steps")
    if epoch is not None and epochs is not None:
        parts.append(f"epoch `{epoch}/{epochs}`")
    if step is not None and steps is not None:
        parts.append(f"step `{step}/{steps}`")
    tqdm = (dashboard.get("active_tqdm") or {}).get(row.get("name"), {})
    if tqdm.get("eta"):
        parts.append(f"ETA `{tqdm.get('eta')}`")
    if tqdm.get("rate"):
        parts.append(f"rate `{tqdm.get('rate')}`")
    return ", ".join(parts) + "."


def _overlay_live_active(dashboard: dict[str, Any], summary_dir: Path) -> dict[str, Any]:
    live_report = (
        _load_json(summary_dir / "active_training_watch_report.json")
        or _load_json(summary_dir / "active_live_training_reports.json")
        or {}
    )
    active = live_report.get("active") or {}
    name = active.get("name")
    if not name:
        return dashboard
    out = dict(dashboard)
    rows = list(out.get("active_rows") or [])
    base = next((row for row in rows if row.get("name") == name), rows[0] if rows else {})
    merged = {
        **base,
        "name": name,
        "active_epoch": active.get("epoch") or base.get("active_epoch") or base.get("epoch"),
        "active_epochs": active.get("epochs") or base.get("active_epochs") or base.get("epochs"),
        "active_step": active.get("step") or base.get("active_step"),
        "active_steps": active.get("steps") or base.get("active_steps"),
    }
    out["active_rows"] = [merged] + [row for row in rows if row.get("name") != name]
    active_tqdm = dict(out.get("active_tqdm") or {})
    active_tqdm[name] = {
        **(active_tqdm.get(name) or {}),
        "eta": active.get("eta") or active.get("tqdm_eta") or (active_tqdm.get(name) or {}).get("eta"),
        "rate": active.get("rate") or active.get("tqdm_rate") or (active_tqdm.get(name) or {}).get("rate"),
    }
    out["active_tqdm"] = active_tqdm
    out["report_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


def _gpu_text(dashboard: dict[str, Any]) -> str:
    gpu = ((dashboard.get("system") or {}).get("gpu") or {})
    if not gpu:
        return "unavailable"
    return (
        f"{gpu.get('name', 'GPU')}, util `{gpu.get('utilization_percent', '-') }%`, "
        f"memory `{gpu.get('memory_used_mb', '-')}/{gpu.get('memory_total_mb', '-')} MB`, "
        f"temp `{gpu.get('temperature_c', '-')} C`"
    )


def _p0_rows(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    return list(dashboard.get("core_ablation") or [])


def _p0_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | friction F1 | risk F1 | low-friction recall | calibrated coverage | worst dataset F1 | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    decisions = {
        "Global-only": "strong baseline",
        "+ PhysicsTexture": "keep as current core",
        "+ FrictionSet": "remove or merge unless interval evidence rescues it",
        "+ DG losses": "remove or redesign",
        "+ EvidenceField aux": "keep provisionally for interpretability/ROI route",
        "Full model": "not acceptable as final method",
    }
    for row in rows:
        method = str(row.get("method", "-"))
        lines.append(
            "| {method} | {friction} | {risk} | {low} | {cov} | {worst} | {decision} |".format(
                method=method,
                friction=_pct(row.get("friction_f1")),
                risk=_pct(row.get("risk_f1")),
                low=_pct(row.get("low_friction_recall")),
                cov=_pct(row.get("calibrated_coverage")),
                worst=_pct(row.get("worst_dataset_f1")),
                decision=decisions.get(method, "inspect after paired evidence"),
            )
        )
    return lines


def _lodo_rows(goal: dict[str, Any]) -> list[dict[str, Any]]:
    rows = ((goal.get("tables") or {}).get("lodo") or []) if isinstance(goal, dict) else []
    if rows:
        return rows
    return [
        {"method": "held-out RoadSaW", "friction_f1": 0.0152, "risk_f1": 0.0108, "low_friction_recall": 0.0, "calibrated_coverage": 0.2043},
        {"method": "held-out RSCD", "friction_f1": 0.0372, "risk_f1": 0.0556, "low_friction_recall": 0.7766, "calibrated_coverage": 0.3412},
        {"method": "held-out RoadSC", "friction_f1": 0.0276, "risk_f1": 0.0, "low_friction_recall": 0.0, "calibrated_coverage": 0.0681},
    ]


def _lodo_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Held-out dataset | friction F1 | risk F1 | low-friction recall | calibrated coverage | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        method = str(row.get("method", "-"))
        label = method.replace("held-out ", "")
        lines.append(
            "| {label} | {friction} | {risk} | {low} | {cov} | severe transfer failure |".format(
                label=label,
                friction=_pct(row.get("friction_f1")),
                risk=_pct(row.get("risk_f1")),
                low=_pct(row.get("low_friction_recall")),
                cov=_pct(row.get("calibrated_coverage")),
            )
        )
    return lines


def _missing(dashboard: dict[str, Any], group_names: list[str]) -> list[str]:
    groups = dashboard.get("group_status") or {}
    out: list[str] = []
    for name in group_names:
        row = groups.get(name) or {}
        out.extend(row.get("missing_runs") or [])
    return out


def build_report(summary_dir: Path) -> dict[str, Any]:
    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    dashboard = _overlay_live_active(dashboard, summary_dir)
    goal = _load_json(summary_dir / "goal_evidence_audit.json") or {}
    claim = _load_json(summary_dir / "claim_evidence_ledger.json") or {}
    watcher = _load_json(summary_dir / "followup_watcher_report.json") or {}
    dataset_audit = _load_json(summary_dir / "dataset_integrity_view_audit.json") or {}
    fail_fast = _load_json(summary_dir / "fail_fast_exploration_report.json") or {}
    gpu_guard = _load_json(summary_dir / "gpu_scheduling_guard_report.json") or {}
    runtime_report = _load_json(summary_dir / "runtime_guard_report.json") or {}
    live = dashboard.get("live_training") or {}
    runtime = dashboard.get("runtime_guard") or runtime_report or {}
    artifact = dashboard.get("artifact_contract") or {}
    p0 = _p0_rows(dashboard)
    lodo = _lodo_rows(goal)
    missing_single = _missing(dashboard, ["single_dataset_faf"])
    missing_baseline = _missing(dashboard, ["single_dataset_baselines"])
    missing_candidates = _missing(dashboard, ["p1_candidates"])
    missing_final = _missing(dashboard, ["final_method_lodo", "final_method_single_dataset"])
    claims = claim.get("claims") or claim.get("claim_status") or []

    return {
        "generated_at": dashboard.get("report_generated_at") or dashboard.get("generated_at"),
        "scope": "camera-only weak-supervised visual friction-affordance interval estimation",
        "runtime": {
            "active": _active_text(dashboard),
            "gpu": _gpu_text(dashboard),
            "runtime_guard": runtime.get("verdict"),
            "artifact_contract": artifact.get("verdict"),
            "watchers": len(watcher.get("watchers") or []),
            "gpu_guard": gpu_guard.get("verdict"),
            "manual_gpu_launch_allowed": gpu_guard.get("manual_launch_allowed"),
        },
        "dataset_evidence": _dataset_evidence(dataset_audit),
        "fail_fast": _fail_fast_summary(fail_fast),
        "latest_training": {
            "run": live.get("run"),
            "latest_completed_epoch": live.get("latest_completed_epoch"),
            "val_loss": live.get("latest_val_loss"),
            "friction_acc": live.get("latest_val_friction_acc"),
            "risk_acc": live.get("latest_val_risk_acc"),
            "raw_coverage": live.get("latest_raw_coverage"),
            "raw_width": live.get("latest_raw_width"),
        },
        "p0_rows": p0,
        "lodo_rows": lodo,
        "missing": {
            "single_dataset_faf": missing_single,
            "matched_convnext_baselines": missing_baseline,
            "candidates": missing_candidates,
            "final_method": missing_final,
        },
        "module_decisions": {
            "keep": ["PhysicsTexture"],
            "provisional_keep": ["EvidenceField"],
            "remove_or_merge": ["FrictionSet", "DG losses", "Full fusion"],
            "pending": [
                "Fourier/style augmentation",
                "bottom-road ROI",
                "color constancy",
                "condition-aware alignment",
                "domain adapters",
                "wet-state hard sampling",
                "coverage-aware interval safety",
            ],
        },
        "claim_counts": (claim.get("status_counts") or {}),
        "claims": claims,
        "public_sources": PUBLIC_SOURCES,
    }


def _dataset_evidence(dataset_audit: dict[str, Any]) -> dict[str, Any]:
    path_checks = dataset_audit.get("path_checks") or {}
    rows = {
        item.get("dataset"): item
        for item in path_checks.get("by_dataset", [])
        if isinstance(item, dict) and item.get("dataset")
    }
    cross = dataset_audit.get("cross_dataset") or {}
    white_records = dataset_audit.get("white_records_top") or []
    if not isinstance(white_records, list):
        white_records = []
    return {
        "total_rows": path_checks.get("total_rows"),
        "total_unique_paths": path_checks.get("total_unique_paths"),
        "missing_unique_paths": path_checks.get("missing_unique_paths"),
        "rows_by_dataset": {name: row.get("rows") for name, row in rows.items()},
        "median_width": cross.get("median_width") or {},
        "median_height": cross.get("median_height") or {},
        "median_aspect": cross.get("median_aspect") or {},
        "near_white_rate": cross.get("near_white_rate") or {},
        "roadsaw_near_white": [row for row in white_records if row.get("dataset") == "roadsaw"][:20],
        "recommendation": (dataset_audit.get("recommendation") or {}).get("route"),
    }


def _fail_fast_summary(report: dict[str, Any]) -> dict[str, Any]:
    policy = report.get("formal_policy") or {}
    return {
        "verdict": report.get("verdict"),
        "policy": policy.get("verdict"),
        "promoted_or_fallback": policy.get("promoted_or_fallback") or [],
        "first_wave": policy.get("fast_screen_first_wave") or [],
        "held_until_screen": policy.get("held_until_screen") or [],
        "full_stack_held_until_screen": policy.get("full_stack_held_until_screen") or [],
        "kill": report.get("kill_or_downgrade") or [],
        "keep": report.get("protect_or_conditional_keep") or [],
    }


def render_markdown(report: dict[str, Any]) -> str:
    train = report.get("latest_training") or {}
    runtime = report.get("runtime") or {}
    missing = report.get("missing") or {}
    decisions = report.get("module_decisions") or {}
    dataset = report.get("dataset_evidence") or {}
    fail_fast = report.get("fail_fast") or {}
    lines = [
        "# Live Research Route Update",
        "",
        f"Generated: {report.get('generated_at', '-')}",
        "",
        "Scope: camera-only visual friction-affordance interval estimation from public road-condition labels. The current public labels support weak visual friction/risk intervals, not synchronized measured tire-road friction coefficients.",
        "",
        "## Current Runtime State",
        "",
        f"- Active run: {runtime.get('active', '-')}",
        f"- GPU: {runtime.get('gpu', '-')}.",
        f"- Runtime guard: `{runtime.get('runtime_guard', '-')}`.",
        f"- GPU scheduling guard: `{runtime.get('gpu_guard', '-')}`, manual launch allowed `{runtime.get('manual_gpu_launch_allowed', '-')}`.",
        f"- Artifact contract: `{runtime.get('artifact_contract', '-')}`.",
        f"- Follow-up watchers visible: `{runtime.get('watchers', 0)}`.",
        f"- Latest completed epoch for `{train.get('run', '-')}`: `{train.get('latest_completed_epoch', '-')}`.",
        f"- Latest validation snapshot: friction accuracy `{_pct(train.get('friction_acc'))}%`, risk accuracy `{_pct(train.get('risk_acc'))}%`, val loss `{_num(train.get('val_loss'))}`, raw interval coverage `{_pct(train.get('raw_coverage'))}%`, raw width `{_num(train.get('raw_width'))}`.",
        "",
        "## What Is Proven Now",
        "",
        "Dataset integrity and view evidence:",
        "",
        f"- Local manifest integrity: `{dataset.get('total_unique_paths', '-')}/{dataset.get('total_unique_paths', '-')}` unique paths accounted for, missing `{dataset.get('missing_unique_paths', '-')}`.",
        f"- Dataset rows: `{json.dumps(dataset.get('rows_by_dataset') or {}, ensure_ascii=False, sort_keys=True)}`.",
        f"- Route recommendation: `{dataset.get('recommendation', '-')}`.",
        "- RoadSaW near-white samples are concentrated in wet/very-wet and bright-surface classes; keep them as a wet-road quality stress slice, not as corruption unless a later decoder audit contradicts this.",
        "- RSCD should be described as road-surface image patches/crops. Do not describe every RSCD image as left/right wheel-front imagery because the local files and source context do not prove wheel-specific pose labels.",
        "",
        "The P0 ablation is complete and currently supports a leaner route rather than the full fusion stack.",
        "",
    ]
    lines.extend(_p0_table(report.get("p0_rows") or []))
    lines.extend(
        [
            "",
            "The base LODO suite is also complete, but it is a failure signal rather than a generalization win.",
            "",
        ]
    )
    lines.extend(_lodo_table(report.get("lodo_rows") or []))
    lines.extend(
        [
            "",
            "This means the current model still captures dataset style and label priors too strongly. Cross-dataset success cannot be claimed yet.",
            "",
            "## Dataset Route Decision",
            "",
            "Do not treat RSCD, RoadSaW, and RoadSC as one homogeneous benchmark.",
            "",
            "- RSCD is best used as a strict same-dataset road-surface patch benchmark and, separately, as an RSCD-27/per-day split benchmark.",
            "- RoadSaW is best used as a wetness/surface-state benchmark and as the main held-out wet-road stress test.",
            "- RoadSC is best used as a snow/winter low-friction stress benchmark.",
            "- Multi-dataset training and LODO should be framed as domain-generalization stress testing, not as a single pooled SOTA table.",
            "",
            "The current primary fair comparison should therefore be local and matched:",
            "",
            "1. `single_rscd_full_faf` vs `baseline_single_rscd_global_convnext`.",
            "2. `single_roadsaw_full_faf` vs `baseline_single_roadsaw_global_convnext`.",
            "3. `single_roadsc_full_faf` vs `baseline_single_roadsc_global_convnext`.",
            "",
            "Published direct-friction papers remain motivation unless their public data, labels, splits, and metrics are reproduced locally.",
            "",
            "## External Benchmark Boundary",
            "",
            "Relevant external sources confirm the route:",
            "",
            "- RSCD official/Figshare descriptions support road-surface patch classification with detailed annotations, not a guaranteed left/right wheel-contact camera view.",
            "- RoadSaW uses calibrated camera road patches and MARWIS-derived wetness/surface information, so near-white wet/reflection images should be treated as quality/wetness stress cases unless file-level audits prove corruption.",
            "- RoadSC is a related calibrated road-patch snow-coverage dataset, closer to RoadSaW than to RSCD in image geometry.",
            "- SIWNet and WCamNet are highly relevant for interval/friction-regression design, but their targets and protocols are not the same as current RSCD/RoadSaW/RoadSC weak-label benchmarks.",
            "- RoadFormer and RoadMamba are relevant RSCD classification references because they emphasize local pavement texture plus global context; their reported accuracies should not be used as direct friction-interval baselines unless the exact RSCD label task, split, preprocessing, and metric are reproduced.",
            "",
            "Therefore, fair numeric wins must come from matched local baselines first. External papers can be cited for motivation, interval methodology, and architectural inspiration.",
            "",
            "Sources already logged in the project:",
            "",
        ]
    )
    for source in report.get("public_sources") or []:
        lines.append(f"- {source['name']}: {source['url']}")
    lines.extend(
        [
            "",
            "## Remaining Experiments Before Any Strong Claim",
            "",
            "Must finish:",
            "",
            f"- Single-dataset FAF: {_list(missing.get('single_dataset_faf'))}.",
            f"- Matched ConvNeXt baselines: {_list(missing.get('matched_convnext_baselines'))}.",
            f"- P1/P2/P3 candidates: {_list(missing.get('candidates'))}.",
            f"- Final lean method rows: {_list(missing.get('final_method'))}.",
            "- RSCD-27/per-day benchmark and direct ExtremeRoad route, but only after the active official queue and watcher chain permit it.",
            "",
            "## Current Algorithm Decision Rules",
            "",
            f"- Keep: {_list(decisions.get('keep'))}.",
            f"- Keep provisionally: {_list(decisions.get('provisional_keep'))}.",
            f"- Remove or merge unless rescued: {_list(decisions.get('remove_or_merge'))}.",
            f"- Pending mechanisms to verify: {_list(decisions.get('pending'))}.",
            "",
            "## Fail-Fast Candidate Policy",
            "",
            f"- Verdict: `{fail_fast.get('verdict', '-')}`; policy `{fail_fast.get('policy', '-')}`.",
            f"- Formal fallback/promoted candidates: {_list(fail_fast.get('promoted_or_fallback'))}.",
            f"- First-wave screen only: {_list(fail_fast.get('first_wave'))}.",
            f"- Held until screen evidence: {_list(fail_fast.get('held_until_screen'))}.",
            f"- Full-stack routes held until screen evidence: {_list(fail_fast.get('full_stack_held_until_screen'))}.",
            "",
            "## Next Technical Route",
            "",
            "The highest-probability final route is:",
            "",
            "1. Shared ConvNeXt visual backbone.",
            "2. Lean PhysicsTexture branch for roughness, local contrast, specular/water/snow cues, and low-texture signals.",
            "3. Road-grounded EvidenceField with bottom-road ROI, pseudo-road-mask attention, augmentation consistency, and local/global evidence fusion inspired by recent RSCD local-global models.",
            "4. Safety-weighted interval head with conditional coverage tracking.",
            "5. Carefully limited state-conditioned alignment or adapters, only if they reduce dataset-ID probes without hurting wetness/low-friction metrics.",
            "",
            "This route should be promoted only if it improves at least one of:",
            "",
            "- paired same-split delta over ConvNeXt;",
            "- RoadSaW wet/damp/very-wet slices;",
            "- low-friction recall;",
            "- worst-dataset F1;",
            "- dataset-ID shortcut score;",
            "- conditional coverage-width tradeoff;",
            "- evidence-map road-region grounding.",
            "",
            "## Immediate Operational Instruction",
            "",
            "Do not manually start another GPU experiment while the GPU scheduling guard is busy. Let the active official queue finish its current fair single-dataset/baseline work. The fail-fast gate is the only intentional watcher now: if the old queue tries to enter an excluded full-stack candidate, it should stop that path, run the lean-first-wave fast screen, then run only the fail-fast-promoted formal candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=Path("reports/paper_protocol_summary"))
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    out_md = args.out_md or args.summary_dir / "live_research_route_update.md"
    out_json = args.out_json or args.summary_dir / "live_research_route_update.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

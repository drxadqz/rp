from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a reviewer-facing decision report that maps CV subfield ideas "
            "to concrete, feasible visual friction-affordance experiments."
        )
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "cv_transfer_decision_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "cv_transfer_decision_report.json",
    )
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    artifact = _load_json(summary_dir / "artifact_contract_report.json") or {}
    data_audit = _load_json(summary_dir / "dataset_integrity_view_audit.json") or {}
    region = _load_json(summary_dir / "region_mixture_summary.json") or {}
    pseudo_mask = (
        _load_json(summary_dir / "pseudo_segmentation_masks" / "pseudo_segmentation_mask_audit.json")
        or {}
    )
    external_mask = (
        _load_json(summary_dir / "external_segmentation_masks" / "external_segmentation_mask_audit.json")
        or {}
    )
    external_mask_cache = _latest_json(summary_dir / "external_road_mask_cache") or _latest_json(
        summary_dir / "external_road_mask_cache_smoke"
    ) or {}
    segmentation_transfer_config = _load_json(summary_dir / "segmentation_transfer_config_audit.json") or {}
    foundation_forward = _load_json(summary_dir / "foundation_probe_forward_loss.json") or {}
    foundation_batch = _load_json(summary_dir / "foundation_probe_batch_check.json") or {}
    roadmap = _load_json(summary_dir / "topvenue_innovation_roadmap.json") or {}
    current_gap = _load_json(summary_dir / "current_algorithm_gap_analysis.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}
    lodo = _load_json(summary_dir / "lodo_generalization_report.json") or {}

    statuses = _run_statuses(artifact)
    dataset_notes = _dataset_notes(data_audit)
    rows = _route_rows(
        statuses,
        region,
        pseudo_mask,
        external_mask,
        external_mask_cache,
        segmentation_transfer_config,
        foundation_forward,
        foundation_batch,
    )
    counts = _counts(rows)
    verdict = _verdict(rows, statuses, shortcut, lodo)
    next_actions = _next_actions(statuses, rows, current_gap)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "claim_boundary": (
            "These routes target weak visual friction-affordance intervals from "
            "public road-condition labels. They are not measured tire-road "
            "friction-coefficient claims unless a protocol-equivalent measured "
            "friction dataset is reproduced."
        ),
        "dataset_notes": dataset_notes,
        "counts": counts,
        "source_anchors": _source_anchors(),
        "rows": rows,
        "next_actions": next_actions,
        "reviewer_rules": _reviewer_rules(),
        "input_evidence": {
            "artifact_runs": artifact.get("num_runs"),
            "artifact_complete": artifact.get("num_contract_complete"),
            "artifact_incomplete": artifact.get("num_contract_incomplete"),
            "pseudo_mask_verdict": pseudo_mask.get("verdict"),
            "pseudo_mask_samples": pseudo_mask.get("samples_total"),
            "external_mask_backend": external_mask.get("backend"),
            "external_mask_verdict": external_mask.get("verdict"),
            "external_mask_samples": external_mask.get("samples_total"),
            "external_mask_cache_verdict": external_mask_cache.get("verdict"),
            "external_mask_cache_root": external_mask_cache.get("cache_root"),
            "segmentation_transfer_config_verdict": segmentation_transfer_config.get("verdict"),
            "roadmap_sources": len(roadmap.get("source_rows", []) or []),
            "roadmap_patterns": len(roadmap.get("pattern_rows", []) or []),
            "dataset_shortcut_verdict": shortcut.get("verdict"),
            "conditional_interval_watchlist": len(interval.get("watchlist", []) or []),
            "lodo_verdict": lodo.get("verdict"),
        },
    }


def _route_rows(
    statuses: dict[str, dict[str, Any]],
    region: dict[str, Any],
    pseudo_mask: dict[str, Any],
    external_mask: dict[str, Any],
    external_mask_cache: dict[str, Any],
    segmentation_transfer_config: dict[str, Any],
    foundation_forward: dict[str, Any],
    foundation_batch: dict[str, Any],
) -> list[dict[str, Any]]:
    region_decision = region.get("verdict", "missing")
    pseudo_mask_verdict = pseudo_mask.get("verdict", "missing")
    external_mask_summary = _external_mask_summary(external_mask)
    external_mask_cache_summary = _external_mask_cache_summary(external_mask_cache)
    segmentation_config_summary = _segmentation_transfer_config_summary(segmentation_transfer_config)
    foundation_forward_ok = int(foundation_forward.get("checks", 0) or 0) > 0 and not foundation_forward.get("failures")
    foundation_batch_ok = int(foundation_batch.get("checks", 0) or 0) > 0 and not foundation_batch.get("failures")

    return [
        {
            "priority": 1,
            "route": "Segmentation-style local region evidence",
            "cv_subfield": "semantic segmentation / material region grouping",
            "source_pattern": "Mask2Former, SAM, SegFormer: reason over regions/masks rather than a single global image vector.",
            "local_feasibility": "high",
            "implemented_as": (
                "v23_lean_region_mixture_evidence_safety plus post-hoc region-mixture conformal calibration; "
                "v24_lean_multi_query_region_evidence_safety adds mask-query-style multi-region pooling and query-disagreement intervals; "
                "v25_lean_masked_query_consistency_safety adds MIC-style masked weak-view consistency"
            ),
            "status": _combined_status(
                statuses,
                ["v23_lean_region_mixture_evidence_safety", "v24_lean_multi_query_region_evidence_safety", "v25_lean_masked_query_consistency_safety"],
            ),
            "evidence_now": region_decision,
            "promote_if": "Improves wet/snow/low-texture quality-slice coverage or RoadSaW wet-state robustness beyond v21/v22 at bounded interval width.",
            "drop_if": "Only widens intervals or fails to beat v21/v22 on conditional slices.",
            "claim_status": "implemented_candidate_pending_metrics",
            "next_action": "Run v23/v24/v25 in the candidate queue; keep v25 only if masked consistency improves hard slices, shortcut metrics, or coverage-width beyond v24/v23/v21/v22 without safety regression.",
        },
        {
            "priority": 2,
            "route": "Road/contact pseudo-mask supervision",
            "cv_subfield": "promptable and universal segmentation",
            "source_pattern": "SAM/Mask2Former/CLIPSeg: external masks can constrain where visual evidence is allowed to come from.",
            "local_feasibility": "medium",
            "implemented_as": (
                "heuristic pseudo-road and ROI constraints in v8/v12/v14-v25; "
                "ManifestDataset can now read cached road_mask_path pseudo masks; "
                "cache_external_road_masks.py creates deterministic mask manifests"
            ),
            "status": _combined_status(statuses, ["v8_full_faf_fourier_roadprior", "v12_full_faf_roi_interval_safety", "v14_lean_road_roi_safety"]),
            "evidence_now": (
                f"pseudo={pseudo_mask_verdict}; external={external_mask_summary}; "
                f"cache={external_mask_cache_summary}; config={segmentation_config_summary}"
            ),
            "promote_if": "Evidence maps show attention leakage outside road/contact regions and a 100-image mask audit shows stable road/contact masks.",
            "drop_if": "Images are road patches where SAM segments texture fragments instead of useful contact regions.",
            "claim_status": "training_loader_bridge_implemented_small_cache_smoke_only",
            "next_action": (
                "Run a 100-image external SAM/Mask2Former/CLIPSeg audit after current queue is idle; "
                "then full-cache masks only if it beats lightweight ROI/region mixture. "
                "Any mask-training config must set load_road_masks=true, road_mask_pretransformed=true, "
                "horizontal_flip_p=0, and random_resized_crop=false."
            )
            if pseudo_mask_verdict == "small_external_mask_audit_worthwhile"
            else "Keep lightweight ROI/region mixture and defer external masks.",
        },
        {
            "priority": 3,
            "route": "Foundation dense visual features",
            "cv_subfield": "self-supervised/foundation visual representation",
            "source_pattern": "DINOv2/MAE: broad pretraining can produce robust texture/material descriptors with limited labels.",
            "local_feasibility": "medium_high",
            "implemented_as": "foundation_dinov2_global_probe and foundation_dinov2_quality_faf_probe configs",
            "status": "configured_probe_forward_ok" if foundation_forward_ok and foundation_batch_ok else "configured_probe_needs_smoke_fix",
            "evidence_now": f"forward_ok={foundation_forward_ok}; batch_ok={foundation_batch_ok}",
            "promote_if": "DINOv2 global or DINOv2+FAF beats matched ConvNeXt on same split or materially improves shortcut/quality slices.",
            "drop_if": "It is slower without paired gains over ConvNeXt/FAF or OOMs under the guarded batch envelope.",
            "claim_status": "feasibility_probe_not_final_claim",
            "next_action": "Run only after formal queue and watcher are idle; keep it separate from the current paper contract until metrics exist.",
        },
        {
            "priority": 4,
            "route": "Style-shortcut suppression",
            "cv_subfield": "domain generalization / semantic-segmentation adaptation",
            "source_pattern": "FDA, MixStyle, DANN, DomainBed: break low-level source-domain shortcuts and validate on held-out domains.",
            "local_feasibility": "high",
            "implemented_as": "v6/v7/v11/v15/v16/v18/v19 plus final lean rows",
            "status": _combined_status(
                statuses,
                [
                    "v6_full_faf_fourier",
                    "v7_full_faf_fourier_dann",
                    "v11_full_faf_domain_adapter",
                    "v15_lean_bottom_square_style_safety",
                    "v16_lean_bottom_square_color_constancy_safety",
                    "v18_lean_mixstyle_quality_safety",
                    "v19_lean_state_contrast_quality_safety",
                ],
            ),
            "evidence_now": "dataset_id_probe_high_on_completed_rows",
            "promote_if": "Dataset-ID balanced accuracy drops while risk F1, low-friction recall, and worst-dataset F1 do not regress.",
            "drop_if": "Shortcut probe stays high or safety metrics collapse, especially for DANN/full DG losses.",
            "claim_status": "implemented_candidate_pending_metrics",
            "next_action": "Run lean-first shortcut candidates before heavy full-stack variants if GPU time is constrained.",
        },
        {
            "priority": 5,
            "route": "Wet-road optical ambiguity cues",
            "cv_subfield": "water detection / physical vision / uncertainty",
            "source_pattern": "Reflection, glare, low texture, and water-film ambiguity should affect uncertainty, not just class logits.",
            "local_feasibility": "high",
            "implemented_as": "v17/v21/v22/v23/v24/v25 quality-aware physics and visual-quality coverage terms",
            "status": _combined_status(
                statuses,
                [
                    "v17_lean_quality_physics_safety",
                    "v21_lean_quality_uncertainty_safety",
                    "v22_lean_quality_order_contrast_safety",
                    "v23_lean_region_mixture_evidence_safety",
                    "v24_lean_multi_query_region_evidence_safety",
                    "v25_lean_masked_query_consistency_safety",
                ],
            ),
            "evidence_now": "RoadSaW near-white slice exists and is class-structured, not corrupt",
            "promote_if": "Near-white/wet/very-wet slices improve without erasing normal-quality wetness cues.",
            "drop_if": "The model learns quality artifacts as dataset ID or widens all intervals uniformly.",
            "claim_status": "implemented_candidate_pending_metrics",
            "next_action": "Rank v17/v21/v22/v23/v24/v25 by RoadSaW quality-slice coverage, wetness F1, low-friction recall, shortcut metrics, and width.",
        },
        {
            "priority": 6,
            "route": "Weak interval-order physics",
            "cv_subfield": "ordinal learning / constrained optimization",
            "source_pattern": "Use physically ordered public labels as inequality constraints instead of fake point friction measurements.",
            "local_feasibility": "high",
            "implemented_as": "v20 and v22 interval-order losses",
            "status": _combined_status(statuses, ["v20_lean_interval_order_quality_safety", "v22_lean_quality_order_contrast_safety"]),
            "evidence_now": "configured_pending_metrics",
            "promote_if": "Low-friction recall or worst-domain F1 improves without excessive width inflation.",
            "drop_if": "Noisy weak intervals over-constrain the model or hurt RoadSaW/RoadSC ambiguous states.",
            "claim_status": "implemented_candidate_pending_metrics",
            "next_action": "Evaluate after v17/v21 so order loss is compared against simpler quality-aware uncertainty.",
        },
        {
            "priority": 7,
            "route": "Monocular depth / geometric ROI",
            "cv_subfield": "depth estimation / 3D scene understanding",
            "source_pattern": "Depth Anything-style geometry can suppress background and focus near-contact road regions in full driving scenes.",
            "local_feasibility": "low_for_current_main_datasets",
            "implemented_as": "not implemented beyond bottom/center ROI heuristics",
            "status": "demoted",
            "evidence_now": "RSCD/RoadSaW/RoadSC are mostly road patches or prepared square crops, not full scenes",
            "promote_if": "A future full-scene dataset is added or mask audit shows geometry helps RoadSaW/RoadSC.",
            "drop_if": "Depth maps are flat/noisy on close-up road-surface patches.",
            "claim_status": "future_only",
            "next_action": "Do not spend formal GPU time on depth for the current RSCD/RoadSaW/RoadSC benchmark.",
        },
        {
            "priority": 8,
            "route": "RSCD-27 class-label benchmark",
            "cv_subfield": "standard supervised classification benchmark",
            "source_pattern": "If external RSCD papers use class-label metrics, compare on matching label protocol rather than weak friction intervals.",
            "local_feasibility": "medium",
            "implemented_as": "scripts/run_rscd_surface_classification.py",
            "status": "implemented_protocol_pending_runs",
            "evidence_now": "script_exists",
            "promote_if": "The split/label/metric protocol matches an external RSCD paper closely enough for a fair table.",
            "drop_if": "External splits or label spaces are incompatible; keep as local contextual benchmark only.",
            "claim_status": "separate_from_main_weak_interval_protocol",
            "next_action": "Run after matched ConvNeXt rows if RSCD external comparison is still needed.",
        },
    ]


def _external_mask_summary(report: dict[str, Any]) -> str:
    if not isinstance(report, dict) or not report:
        return "missing"
    backend = report.get("backend", "-")
    verdict = report.get("verdict", "-")
    samples = report.get("samples_total")
    if samples is None:
        return f"{backend}:{verdict}"
    return f"{backend}:{verdict},n={samples}"


def _external_mask_cache_summary(report: dict[str, Any]) -> str:
    if not isinstance(report, dict) or not report:
        return "missing"
    verdict = report.get("verdict", "-")
    cache_root = report.get("cache_root")
    manifest_reports = report.get("manifest_reports") or []
    rows = sum(int(item.get("rows", 0) or 0) for item in manifest_reports if isinstance(item, dict))
    backend = report.get("backend", "-")
    if rows:
        return f"{backend}:{verdict},cached_rows={rows}"
    if cache_root:
        return f"{backend}:{verdict},root={cache_root}"
    return str(verdict)


def _segmentation_transfer_config_summary(report: dict[str, Any]) -> str:
    if not isinstance(report, dict) or not report:
        return "missing"
    batch = report.get("batch_report", {}) if isinstance(report.get("batch_report"), dict) else {}
    pseudo_loss = batch.get("loss_evidence_attention_pseudo_road")
    if pseudo_loss is None:
        return str(report.get("verdict", "-"))
    return f"{report.get('verdict', '-')},pseudo_loss={float(pseudo_loss):.4f}"


def _dataset_notes(data_audit: dict[str, Any]) -> dict[str, Any]:
    rows = data_audit.get("dataset_rows", {}) if isinstance(data_audit.get("dataset_rows"), dict) else {}
    cross = data_audit.get("cross_dataset", {}) if isinstance(data_audit.get("cross_dataset"), dict) else {}
    recommendation = data_audit.get("recommendation", {}) if isinstance(data_audit.get("recommendation"), dict) else {}
    return {
        "datasets": {
            name: {
                "rows": row.get("rows"),
                "dominant_dimensions": row.get("dimension_top"),
                "near_white_rate": (row.get("near_white") or {}).get("rate")
                if isinstance(row.get("near_white"), dict)
                else row.get("near_white_rate"),
                "view_inference": (row.get("view_inference") or {}).get("inference"),
            }
            for name, row in rows.items()
            if isinstance(row, dict)
        },
        "cross_dataset": cross,
        "recommended_route": recommendation.get("route"),
        "decision": recommendation.get("decision"),
    }


def _run_statuses(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = artifact.get("rows", []) if isinstance(artifact.get("rows"), list) else []
    return {str(row.get("name")): row for row in rows if isinstance(row, dict) and row.get("name")}


def _status_for(statuses: dict[str, dict[str, Any]], run: str) -> str:
    row = statuses.get(run)
    if not row:
        return "not_in_contract"
    return str(row.get("contract_status") or row.get("progress_status") or "unknown")


def _combined_status(statuses: dict[str, dict[str, Any]], runs: list[str]) -> str:
    states = [_status_for(statuses, run) for run in runs]
    if any(state == "complete" for state in states):
        return "partly_complete"
    if any(state in {"partial", "incomplete"} for state in states):
        return "partly_in_progress_or_incomplete"
    if all(state in {"missing", "not_in_contract"} for state in states):
        return "configured_missing_results"
    return ",".join(sorted(set(states)))


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("claim_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    counts["routes"] = len(rows)
    counts["high_feasibility"] = sum(1 for row in rows if str(row.get("local_feasibility", "")).startswith("high"))
    counts["future_or_demoted"] = sum(
        1 for row in rows if row.get("claim_status") == "future_only" or row.get("status") == "demoted"
    )
    return counts


def _verdict(
    rows: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    shortcut: dict[str, Any],
    lodo: dict[str, Any],
) -> str:
    implemented = [row for row in rows if "implemented" in str(row.get("claim_status"))]
    v23_status = _status_for(statuses, "v23_lean_region_mixture_evidence_safety")
    v24_status = _status_for(statuses, "v24_lean_multi_query_region_evidence_safety")
    if shortcut.get("verdict") == "warn" or lodo.get("verdict") == "generalization_failure_needs_algorithm_update":
        return "routes_ready_but_metric_evidence_pending"
    if implemented and (v23_status in {"missing", "incomplete", "partial"} or v24_status in {"missing", "incomplete", "partial"}):
        return "implemented_routes_waiting_for_candidate_results"
    return "informational"


def _next_actions(statuses: dict[str, dict[str, Any]], rows: list[dict[str, Any]], current_gap: dict[str, Any]) -> list[str]:
    active = [
        name
        for name, row in statuses.items()
        if row.get("progress_status") in {"partial", "running"}
        or row.get("active_epoch")
        or row.get("active_step")
    ]
    actions = []
    if active:
        actions.append(
            "Let the active formal GPU run finish before starting foundation or mask-audit jobs: "
            + ", ".join(f"`{name}`" for name in active[:3])
            + "."
        )
    actions.extend(
        [
            "Finish matched single-dataset ConvNeXt baselines before claiming an external-performance win.",
            "Run v17/v21/v22/v23/v24/v25 as the first CV-transfer slice because they directly target RoadSaW white/wet ambiguity, multi-region evidence, masked consistency, and interval quality.",
            "Use v6/v15/v16/v18/v19 to attack dataset style shortcuts; keep only candidates that reduce dataset-ID probe without safety regression.",
            "Run DINOv2 probes only after the formal queue is idle; treat them as strong baselines, not automatic method components.",
            "Do not run full SAM/Mask2Former preprocessing until a small pseudo-mask audit proves that masks align with road/contact evidence.",
        ]
    )
    if current_gap.get("key_failures"):
        actions.append("Use current_algorithm_gap_analysis.json as the metric gate for pruning weak modules.")
    return actions


def _reviewer_rules() -> list[str]:
    return [
        "External papers are numeric baselines only when split, labels, and metrics match.",
        "RSCD/RoadSaW/RoadSC pooled accuracy is not a top-venue claim because the views and styles differ.",
        "LODO failures must be reported as failure evidence, not hidden or reframed as success.",
        "A module stays in the final method only if it earns safety, interval, generalization, or interpretability evidence.",
        "Segmentation/foundation/depth methods are claims only after code/config/artifact evidence exists locally.",
    ]


def _source_anchors() -> list[dict[str, str]]:
    return [
        {
            "name": "Mask2Former",
            "role": "segmentation-style mask/region reasoning",
            "url": "https://arxiv.org/abs/2112.01527",
            "code": "https://github.com/facebookresearch/Mask2Former",
        },
        {
            "name": "Segment Anything",
            "role": "promptable pseudo-mask source",
            "url": "https://arxiv.org/abs/2304.02643",
            "code": "https://github.com/facebookresearch/segment-anything",
        },
        {
            "name": "SAM 2",
            "role": "newer promptable segmentation/video mask source for future audit",
            "url": "https://arxiv.org/abs/2408.00714",
            "code": "https://github.com/facebookresearch/sam2",
        },
        {
            "name": "CLIPSeg",
            "role": "text-prompted road/wet/snow pseudo-mask source",
            "url": "https://arxiv.org/abs/2112.10003",
            "code": "https://github.com/timojl/clipseg",
        },
        {
            "name": "SegFormer",
            "role": "efficient semantic-segmentation design",
            "url": "https://arxiv.org/abs/2105.15203",
            "code": "https://github.com/NVlabs/SegFormer",
        },
        {
            "name": "DINOv2",
            "role": "self-supervised foundation features",
            "url": "https://arxiv.org/abs/2304.07193",
            "code": "https://github.com/facebookresearch/dinov2",
        },
        {
            "name": "Depth Anything V2",
            "role": "future geometric ROI cue",
            "url": "https://arxiv.org/abs/2406.09414",
            "code": "https://github.com/DepthAnything/Depth-Anything-V2",
        },
        {
            "name": "FDA",
            "role": "Fourier style/domain adaptation",
            "url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html",
            "code": "https://github.com/YanchaoYang/FDA",
        },
        {
            "name": "MixStyle",
            "role": "feature-statistics domain generalization",
            "url": "https://openreview.net/forum?id=6xHJ37MVxxp",
            "code": "https://github.com/KaiyangZhou/mixstyle-release",
        },
        {
            "name": "DomainBed",
            "role": "strict domain-generalization evaluation discipline",
            "url": "https://openreview.net/forum?id=lQdXeXDoWtI",
            "code": "https://github.com/facebookresearch/DomainBed",
        },
        {
            "name": "RoadSaW",
            "role": "public road surface and wetness dataset",
            "url": "https://openaccess.thecvf.com/content/CVPR2022W/WAD/html/Cordes_RoadSaW_A_Large-Scale_Dataset_for_Camera-Based_Road_Surface_and_Wetness_CVPRW_2022_paper.html",
            "code": "https://viscoda.com/index.php/de/downloads-de/roadsaw-dataset-de",
        },
        {
            "name": "RoadSC",
            "role": "public road snow coverage dataset",
            "url": "https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/html/Cordes_Camera-Based_Road_Snow_Coverage_Estimation_ICCVW_2023_paper.html",
            "code": "https://viscoda.com/index.php/en/downloads-en/roadsc-dataset",
        },
        {
            "name": "RSCD",
            "role": "public road surface condition dataset",
            "url": "https://thu-rsxd.com/rscd/",
            "code": "https://github.com/ztsrxh/RSCD-Road_Surface_Classification_Dataset",
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CV Subfield Transfer Decision Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Dataset Implication",
        "",
    ]
    dataset_notes = report.get("dataset_notes", {})
    lines.append(f"- Recommended route: `{dataset_notes.get('recommended_route', '-')}`.")
    lines.append(f"- Decision: {dataset_notes.get('decision', '-')}")
    lines.append("")
    lines.extend(
        [
            "| Dataset | Rows | Dominant dimensions | Near-white rate | View inference |",
            "|---|---:|---|---:|---|",
        ]
    )
    for name, row in (dataset_notes.get("datasets") or {}).items():
        lines.append(
            "| {name} | {rows} | {dims} | {white} | {view} |".format(
                name=name,
                rows=row.get("rows", "-"),
                dims=_fmt_dims(row.get("dominant_dimensions")),
                white=_pct(row.get("near_white_rate")),
                view=row.get("view_inference", "-"),
            )
        )
    lines.extend(["", "## Transfer Routes", ""])
    lines.extend(
        [
            "| Priority | Route | CV subfield | Feasibility | Status | Evidence now | Claim status | Next action |",
            "|---:|---|---|---|---|---|---|---|",
        ]
    )
    for row in report.get("rows", []):
        lines.append(
            "| {priority} | {route} | {field} | {feas} | `{status}` | {evidence} | `{claim}` | {action} |".format(
                priority=row.get("priority", "-"),
                route=row.get("route", "-"),
                field=row.get("cv_subfield", "-"),
                feas=row.get("local_feasibility", "-"),
                status=row.get("status", "-"),
                evidence=row.get("evidence_now", "-"),
                claim=row.get("claim_status", "-"),
                action=row.get("next_action", "-"),
            )
        )
    lines.extend(["", "## Promotion And Drop Rules", ""])
    lines.extend(["| Route | Promote if | Drop if |", "|---|---|---|"])
    for row in report.get("rows", []):
        lines.append(
            "| {route} | {promote} | {drop} |".format(
                route=row.get("route", "-"),
                promote=row.get("promote_if", "-"),
                drop=row.get("drop_if", "-"),
            )
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(report.get("next_actions", []), start=1))
    lines.extend(["", "## Source Anchors", ""])
    lines.extend(["| Source | Role | Paper/source | Code/data |", "|---|---|---|---|"])
    for source in report.get("source_anchors", []):
        lines.append(
            "| {name} | {role} | {url} | {code} |".format(
                name=source.get("name", "-"),
                role=source.get("role", "-"),
                url=source.get("url", "-"),
                code=source.get("code", "-"),
            )
        )
    lines.extend(["", "## Reviewer Rules", ""])
    lines.extend(f"- {item}" for item in report.get("reviewer_rules", []))
    return "\n".join(lines) + "\n"


def _fmt_dims(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}:{v}" for k, v in value.items()) or "-"
    return str(value or "-")


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_json(path: Path) -> Any:
    if not path.exists():
        return None
    files = sorted(path.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for item in files:
        data = _load_json(item)
        if data:
            return data
    return None


if __name__ == "__main__":
    main()

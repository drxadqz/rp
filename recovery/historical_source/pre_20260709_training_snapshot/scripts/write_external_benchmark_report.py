from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


PUBLIC_SOURCES = [
    {
        "name": "RSCD / Road Surface Classification Dataset",
        "url": "https://thu-rsxd.com/rscd/",
        "role": "Primary public road-surface dataset for material, friction-proxy, and unevenness labels.",
    },
    {
        "name": "RSCD paper",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9343931/",
        "role": "Dataset paper describing annotated road-surface image categories.",
    },
    {
        "name": "RSCD per-day split",
        "url": "https://github.com/MiviaLab/Road-Surface-Dataset-per-day-split",
        "role": "Optional stronger RSCD protocol source for acquisition-day-disjoint evaluation if current RSCD split proves too easy.",
    },
    {
        "name": "RoadFormer on RSCD",
        "url": "https://arxiv.org/html/2506.02358v1",
        "role": "Recent RSCD architecture/reference result using local-global feature fusion and foreground/background separation; use only as contextual RSCD evidence unless split and labels are replicated exactly.",
    },
    {
        "name": "RoadSaW",
        "url": "https://openaccess.thecvf.com/content/CVPR2022W/WAD/papers/Cordes_RoadSaW_A_Large-Scale_Dataset_for_Camera-Based_Road_Surface_and_Wetness_CVPRW_2022_paper.pdf",
        "role": "Public camera-based road surface and wetness dataset used for domain-shift stress testing.",
    },
    {
        "name": "RoadSC",
        "url": "https://viscoda.com/index.php/en/downloads-en/roadsc-dataset",
        "role": "Public road snow coverage dataset used for winter-condition and low-friction stress testing.",
    },
    {
        "name": "RoadSC paper",
        "url": "https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/papers/Cordes_Camera-Based_Road_Snow_Coverage_Estimation_ICCVW_2023_paper.pdf",
        "role": "Dataset paper for camera-based road snow coverage estimation.",
    },
    {
        "name": "ROAD Camera-IMU road-surface dataset",
        "url": "https://arxiv.org/abs/2601.20847",
        "role": "Future multimodal RGB-IMU road-surface dataset candidate; keep outside current visual-only fair comparisons until public files, splits, labels, and license are audited.",
    },
    {
        "name": "Extreme Road Image Dataset",
        "url": "https://github.com/sean-shiyuez/Extreme-Road-Image-Dataset",
        "role": "Separate public extreme-road image route now downloaded/audited locally; useful for a direct visual friction-affordance stress protocol with its own splits and metrics.",
    },
    {
        "name": "WCamNet winter friction estimation",
        "url": "https://arxiv.org/abs/2404.16578",
        "role": "Contextual vision-based winter road-friction reference using DINOv2-style foundation features and local road texture cues.",
    },
    {
        "name": "WCamNet GitHub",
        "url": "https://github.com/ojalar/wcamnet",
        "role": "Open MIT-licensed implementation of WCamNet; code reference only unless the measured-friction data protocol is reproduced locally.",
    },
    {
        "name": "WARD Weather-Aware Road Dataset",
        "url": "https://library.utia.cas.cz/separaty/2026/ZOI/nesnidalova-0644268.pdf",
        "role": "Future road-weather/road-surface-condition dataset candidate; audit downloadability, license, splits, and label mapping before use.",
    },
    {
        "name": "MixStyle",
        "url": "https://openreview.net/forum?id=6xHJ37MVxxp",
        "role": "ICLR domain-generalization method motivating v18 feature-statistics style randomization.",
    },
    {
        "name": "Supervised Contrastive Learning",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html",
        "role": "NeurIPS label-aware representation learning method motivating v19 cross-dataset same-state alignment.",
    },
    {
        "name": "SIWNet prediction intervals",
        "url": "https://arxiv.org/abs/2310.00923",
        "role": "Highly relevant image-to-road-friction regression with prediction-interval estimation; interval-method reference, not a matched RSCD/RoadSaW/RoadSC benchmark.",
    },
    {
        "name": "Finnish Winter Driving Dataset",
        "url": "https://zenodo.org/records/14856338",
        "role": "Public winter driving images with GNSS/INS/lidar and segmentation labels; possible future road-mask/winter-domain source, not current friction-label evidence.",
    },
    {
        "name": "Continual cross-dataset road-surface adaptation",
        "url": "https://arxiv.org/abs/2309.02210",
        "role": "Protocol reference showing road-surface models can degrade across datasets; motivates LODO and possible future continual adaptation.",
    },
    {
        "name": "FDA Fourier domain adaptation",
        "url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html",
        "role": "Style/domain perturbation inspiration for Fourier low-frequency jitter.",
    },
    {
        "name": "DANN",
        "url": "https://jmlr.org/papers/v17/15-239.html",
        "role": "Gradient-reversal domain-adversarial representation learning.",
    },
    {
        "name": "GroupDRO",
        "url": "https://arxiv.org/abs/1911.08731",
        "role": "Worst-group robustness reference.",
    },
    {
        "name": "DomainBed",
        "url": "https://openreview.net/forum?id=lQdXeXDoWtI",
        "role": "Cautionary domain-generalization benchmark practice: fair ERM baselines and held-out domains.",
    },
    {
        "name": "RAPS conformal prediction",
        "url": "https://openreview.net/forum?id=eNdiU_DbM9",
        "role": "Distribution-free uncertainty-set calibration reference.",
    },
    {
        "name": "eCFR 49 CFR 575.104 UTQG",
        "url": "https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-575/subpart-B/section-575.104",
        "role": "Public tire traction coefficient anchor for wet asphalt/concrete grading.",
    },
    {
        "name": "FHWA pavement friction safety primer",
        "url": "https://highways.dot.gov/safety/rwd/keep-vehicles-road/pavement-friction/pavement-friction-road-safety-primer-friction",
        "role": "Official pavement-friction management reference supporting conservative affordance-interval framing.",
    },
    {
        "name": "TRB snow/ice skid resistance report",
        "url": "https://onlinepubs.trb.org/Onlinepubs/sr/sr115/115-010.pdf",
        "role": "Public snow/ice skid-resistance anchors for winter-condition friction interval sanity checks.",
    },
]


DATASET_ALIGNMENT = [
    {
        "dataset": "RSCD",
        "fair_unit": "single_rscd_full_faf vs baseline_single_rscd_global_convnext",
        "task_alignment": "Bonnet-mounted monocular-camera road-area patches cropped to 360x240 around the tire-passed road region; friction/weather/material labels can be evaluated as public road-state classification and weak friction-interval proxy.",
        "primary_metrics": "friction macro-F1, risk macro-F1, low-friction recall, calibrated coverage/width",
        "claim_status": "pending_single_dataset_runs; RoadFormer/RSCD numbers are contextual until protocol-equivalent",
    },
    {
        "dataset": "RSCD-27 original class protocol",
        "fair_unit": "scripts/run_rscd_surface_classification.py",
        "task_alignment": "Original RSCD class_label classification, separated from the weak friction-affordance interval task.",
        "primary_metrics": "Top-1 accuracy, macro-F1, weighted-F1, balanced accuracy",
        "claim_status": "implemented_protocol_pending_runs; use only for RSCD/RoadFormer-style comparisons",
    },
    {
        "dataset": "RoadSaW",
        "fair_unit": "single_roadsaw_full_faf vs baseline_single_roadsaw_global_convnext; lodo_roadsaw_full_faf for held-out generalization",
        "task_alignment": "High-resolution bird's-eye-view road patches derived from calibrated in-vehicle sensors with surface type and MARWIS wetness labels; use as wet-road friction-risk proxy and domain-shift stress test.",
        "primary_metrics": "risk macro-F1, wetness/friction macro-F1, low-friction recall, RoadSaW conditional interval quality",
        "claim_status": "held-out_lodo_complete_and_failed; single_dataset_fair_rows_pending",
    },
    {
        "dataset": "RoadSC",
        "fair_unit": "single_roadsc_full_faf vs baseline_single_roadsc_global_convnext; lodo_roadsc_full_faf for snow-state transfer",
        "task_alignment": "Bird's-eye-view road patches with snow-coverage type labels; use as winter low-friction proxy and RoadSaW-combinable stress test, not measured tire-road mu.",
        "primary_metrics": "risk macro-F1, friction/snow macro-F1, calibrated coverage/width",
        "claim_status": "held-out_lodo_complete_and_failed; single_dataset_fair_rows_pending",
    },
    {
        "dataset": "ExtremeRoad direct route",
        "fair_unit": "extreme_road_quality_physics_fast vs extreme_road_global_convnext_fast plus six-class class-label context",
        "task_alignment": "Six public extreme-condition visual classes are evaluated as a separate direct visual friction-affordance route, not merged into RSCD/RoadSaW/RoadSC.",
        "primary_metrics": "same-task friction/risk/interval metrics, six-class top-1/macro-F1 context, paired bootstrap",
        "claim_status": "local_files_audited_and_queue_pending; separate_from_main_weak_label_protocol",
    },
]


DATASET_ROUTE_DECISION = [
    {
        "route": "Single RSCD as main RSCD benchmark",
        "decision": "keep_as_required_secondary_or_primary_table",
        "reason": "RSCD has the largest public scale and explicit friction/material/unevenness labels, so it is the only current route where RSCD-style SOTA classification comparison is plausible.",
        "risk": "Its official split is approximately i.i.d.; high scores may reflect acquisition/style regularities rather than friction physics.",
        "required_evidence": "Run RSCD-27/class-label protocol and same-split ConvNeXt; consider per-day split if RSCD looks too easy.",
    },
    {
        "route": "RoadSaW/RoadSC as separate public road-condition benchmarks",
        "decision": "keep_separate_not_pooled_as_same_distribution",
        "reason": "They use BEV road patches and sensor-derived wetness/snow labels; their image geometry and label semantics differ from RSCD but are exactly the stress cases for wet/snow friction affordance.",
        "risk": "The near-white RoadSaW slice is a valid wet/reflection/exposure stress case; dropping it would make the method less honest.",
        "required_evidence": "Run same-dataset FAF vs ConvNeXt plus wetness/snow quality slices.",
    },
    {
        "route": "Naive multi-dataset pooling",
        "decision": "do_not_use_as_main_claim",
        "reason": "LODO already shows severe transfer failure and dataset-ID probes are high; pooled accuracy alone would reward dataset/style shortcuts.",
        "risk": "Reviewers can reject the paper if the model learns dataset identity instead of road-surface friction evidence.",
        "required_evidence": "Use pooling only as diagnosis; main claims need matched single-dataset rows and LODO.",
    },
    {
        "route": "Hierarchical protocol",
        "decision": "recommended_main_route",
        "reason": "Separate in-domain benchmark strength from OOD stress: RSCD same-split/class-label, RoadSaW/RoadSC same-split, then LODO and quality slices.",
        "risk": "More tables, but claims are defensible and each result has a clear interpretation.",
        "required_evidence": "Finalize matched ConvNeXt baselines, paired bootstrap, calibration coverage-width, and claim ledger.",
    },
]


DIRECT_FRICTION_BENCHMARK_ROWS = [
    {
        "paper_or_dataset": "SIWNet",
        "target": "image-to-friction regression with prediction intervals",
        "public_comparability": "not_matched",
        "why_not_fair_numeric": "Uses images paired with road-friction sensor readings from SeeingThroughFog; RSCD/RoadSaW/RoadSC provide road-condition labels rather than the same measured scalar target.",
        "usable_idea": "Prediction-interval head, coverage-width reporting, and lightweight uncertainty-aware design.",
        "local_action": "Cite as interval-method reference; do not compare numeric interval coverage unless the same image+sensor target is reproduced.",
    },
    {
        "paper_or_dataset": "WCamNet",
        "target": "winter roadside-camera friction regression from images and optical road-surface friction sensors",
        "public_comparability": "not_matched",
        "why_not_fair_numeric": "Its data are Finnish roadside-camera images paired with optical friction sensors; the current local protocol is public weak road-condition labels.",
        "usable_idea": "Hybrid foundation visual features plus CNN/local texture branch for winter/wet road friction.",
        "local_action": "Use as a future DINOv2/foundation-feature baseline only if a protocol-equivalent measured-friction dataset becomes locally reproducible.",
    },
    {
        "paper_or_dataset": "RoadSaW/RoadSC",
        "target": "surface/wetness/snow state classification with uncertainty",
        "public_comparability": "partially_matched",
        "why_not_fair_numeric": "They are public and locally available, but labels are road condition proxies, not direct friction coefficients.",
        "usable_idea": "Wetness and snow state labels support friction-risk intervals and OOD/quality slices.",
        "local_action": "Use same-split local FAF-vs-ConvNeXt and avoid comparing to unmatched published friction-regression numbers.",
    },
    {
        "paper_or_dataset": "RSCD/RoadFormer-style RSCD classification",
        "target": "RSCD road-surface/class-label recognition",
        "public_comparability": "possible_if_protocol_matched",
        "why_not_fair_numeric": "Published RSCD numbers are only fair if split, label canonicalization, preprocessing, and metric definition match.",
        "usable_idea": "Local-global and foreground/background road evidence design.",
        "local_action": "Run local RSCD-27 protocol and compare only under identical labels/splits or report as contextual.",
    },
]


FAIR_CLAIM_LADDER = [
    {
        "level": "L0 data sanity",
        "allowed_claim": "Datasets are downloaded, local paths exist, and view/style/quality differences are audited.",
        "required_artifact": "dataset_integrity_view_audit.md/json",
    },
    {
        "level": "L1 in-domain fair baseline",
        "allowed_claim": "FAF improves or trades off against a modern ConvNeXt baseline on the same dataset split.",
        "required_artifact": "single_*_full_faf, baseline_single_*_global_convnext, paired bootstrap.",
    },
    {
        "level": "L2 RSCD external context",
        "allowed_claim": "The method is competitive on RSCD-style labels under a documented local protocol.",
        "required_artifact": "RSCD-27/class-label protocol with exact label mapping and split.",
    },
    {
        "level": "L3 cross-dataset robustness",
        "allowed_claim": "The method generalizes across dataset/view/style shifts.",
        "required_artifact": "final LODO rows with acceptable RoadSaW/RoadSC risk F1, low-friction recall, and calibrated coverage-width.",
    },
    {
        "level": "L4 direct friction regression",
        "allowed_claim": "The method exceeds direct image-to-friction regression papers.",
        "required_artifact": "Same measured-friction dataset, same target, same split, same metric; current project does not yet have this.",
    },
]


COMPARABILITY_ROWS = [
    {
        "source": "Matched local ConvNeXt",
        "task_or_dataset": "RSCD / RoadSaW / RoadSC same-split visual classification",
        "reported_metric": "risk macro-F1, friction macro-F1, low-friction recall, coverage-width",
        "split_match": "yes",
        "label_match": "yes",
        "metric_match": "yes",
        "comparison_level": "primary_numeric_baseline",
        "allowed_use": "Main paper comparison after local runs finish.",
        "required_action": "Run baseline_single_*_global_convnext and paired bootstrap deltas.",
    },
    {
        "source": "RoadFormer",
        "task_or_dataset": "RSCD-style fine-grained road-surface classification",
        "reported_metric": "Top-1 accuracy on RSCD/simple-RSCD",
        "split_match": "not_verified",
        "label_match": "partial",
        "metric_match": "no",
        "comparison_level": "context_or_reimplementation_target",
        "allowed_use": "Cite as local-global/foreground-background inspiration and contextual RSCD reference.",
        "required_action": "Only compare numerically after reproducing the same split/label metric or converting our run to that exact protocol.",
    },
    {
        "source": "Local RSCD-27 ConvNeXt protocol",
        "task_or_dataset": "Original RSCD class_label classification",
        "reported_metric": "Top-1 accuracy, macro-F1, weighted-F1, balanced accuracy",
        "split_match": "yes_for_local_manifests",
        "label_match": "yes_after_hyphen_underscore_canonicalization",
        "metric_match": "yes_for_Rscd_classification",
        "comparison_level": "secondary_numeric_baseline_for_Rscd_classification",
        "allowed_use": "Use to determine whether an RSCD SOTA-style claim is plausible under a clearly stated local protocol.",
        "required_action": "Run fast and formal RSCD-27 ConvNeXt rows after the current FAF/v17 GPU queue.",
    },
    {
        "source": "RSCD dataset papers/pages",
        "task_or_dataset": "RSCD public road-state annotations",
        "reported_metric": "dataset size and label taxonomy",
        "split_match": "dataset_source",
        "label_match": "dataset_source",
        "metric_match": "not_algorithm",
        "comparison_level": "data_provenance",
        "allowed_use": "Justify public weak-label source and label mapping.",
        "required_action": "Keep train/val/test manifests versioned and report the exact local split.",
    },
    {
        "source": "RoadSaW / RoadSC dataset papers",
        "task_or_dataset": "wetness and snow-condition public road datasets",
        "reported_metric": "dataset/task definitions",
        "split_match": "dataset_source",
        "label_match": "dataset_source",
        "metric_match": "not_algorithm",
        "comparison_level": "data_provenance",
        "allowed_use": "Use for held-out RoadSaW/RoadSC stress-test motivation.",
        "required_action": "Report local same-split FAF/ConvNeXt and LODO metrics, not unmatched paper numbers.",
    },
    {
        "source": "WCamNet",
        "task_or_dataset": "winter roadside-camera friction regression with optical friction sensor labels",
        "reported_metric": "direct friction-estimation error/accuracy on a different measured-friction dataset",
        "split_match": "no",
        "label_match": "no",
        "metric_match": "no",
        "comparison_level": "method_context_future_baseline",
        "allowed_use": "Cite as evidence that DINOv2/global visual features plus local texture are promising for direct friction.",
        "required_action": "Do not compare numeric results unless the measured-friction dataset and metric are reproduced locally.",
    },
    {
        "source": "WCamNet GitHub",
        "task_or_dataset": "implementation of WCamNet visual friction model",
        "reported_metric": "code implementation; no local matched data/split evidence in current protocol",
        "split_match": "no",
        "label_match": "no",
        "metric_match": "no",
        "comparison_level": "code_reference_only",
        "allowed_use": "Use architectural patterns and reproduction notes; do not report a numeric baseline from this repo alone.",
        "required_action": "Add a separate WCamNet-style local reproduction only if a measured-friction dataset with license/splits is available.",
    },
    {
        "source": "SIWNet",
        "task_or_dataset": "camera-image friction regression with prediction intervals",
        "reported_metric": "scalar friction regression and prediction-interval quality on sensor-ground-truth data",
        "split_match": "no",
        "label_match": "no",
        "metric_match": "partial",
        "comparison_level": "interval_method_context",
        "allowed_use": "Use as motivation for coverage-width reporting, prediction intervals, and uncertainty-aware friction estimation.",
        "required_action": "Do not compare numeric interval coverage/width unless the same image+sensor dataset and target friction definition are reproduced.",
    },
    {
        "source": "Finnish Winter Driving Dataset",
        "task_or_dataset": "winter driving perception with images and road segmentation annotations",
        "reported_metric": "public data source, not friction labels",
        "split_match": "future_only",
        "label_match": "no_for_friction",
        "metric_match": "no",
        "comparison_level": "future_pseudo_mask_or_domain_source",
        "allowed_use": "Potentially useful for road-mask or winter-domain pretraining after license/split audit.",
        "required_action": "Keep outside current friction-affordance results until a separate protocol is created.",
    },
    {
        "source": "ExtremeRoad local direct route",
        "task_or_dataset": "public extreme-road image classes mapped to a separate direct visual friction-affordance task",
        "reported_metric": "local same-task friction/risk/interval metrics and six-class classification context",
        "split_match": "yes_for_local_extreme_manifests",
        "label_match": "yes_for_local_extreme_protocol",
        "metric_match": "yes_for_local_same_task",
        "comparison_level": "separate_numeric_baseline_not_main_RSCD_RoadSaW_RoadSC",
        "allowed_use": "Use as a separate public-data stress route after queued global ConvNeXt and FAF rows finish.",
        "required_action": "Keep tables separate from RSCD/RoadSaW/RoadSC and report the dataset-label semantics explicitly.",
    },
    {
        "source": "Two-stage camera RSC/RFE studies",
        "task_or_dataset": "road-condition classification followed by rule/ROI friction-level estimation",
        "reported_metric": "RSC/RFE accuracy under private or non-matching protocols",
        "split_match": "no",
        "label_match": "partial",
        "metric_match": "no",
        "comparison_level": "method_context",
        "allowed_use": "Use as support for ROI patching, weak friction-level mapping, and rule baselines.",
        "required_action": "Keep our non-visual rule baseline clearly separated from visual-model comparisons.",
    },
    {
        "source": "DomainBed / DANN / GroupDRO / FDA",
        "task_or_dataset": "domain-generalization and style-shift methodology",
        "reported_metric": "method/protocol references, not road-friction benchmark scores",
        "split_match": "not_applicable",
        "label_match": "not_applicable",
        "metric_match": "not_applicable",
        "comparison_level": "protocol_method_reference",
        "allowed_use": "Use to justify LODO, dataset-ID probes, style augmentation, and strong ERM/ConvNeXt baselines.",
        "required_action": "Judge all such components by local ablation, LODO, and shortcut diagnostics.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "external_benchmark_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "external_benchmark_report.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    summary = load_json(summary_dir / "paper_protocol_summary.json") or {}
    completeness = load_json(summary_dir / "protocol_completeness.json") or {}
    fair = summary.get("fair_single_dataset_deltas", [])
    final_fair = summary.get("final_fair_single_dataset_deltas", [])
    lodo = summary.get("lodo", [])
    final_lodo = summary.get("final_lodo", [])
    rule = summary.get("rule_baselines", [])
    requirements = {item.get("name"): item for item in completeness.get("requirements", [])}
    return {
        "summary_dir": str(summary_dir),
        "public_sources": PUBLIC_SOURCES,
        "dataset_alignment": DATASET_ALIGNMENT,
        "dataset_route_decision": DATASET_ROUTE_DECISION,
        "direct_friction_benchmark_rows": DIRECT_FRICTION_BENCHMARK_ROWS,
        "fair_claim_ladder": FAIR_CLAIM_LADDER,
        "comparability_matrix": COMPARABILITY_ROWS,
        "fair_single_dataset_deltas": fair,
        "final_fair_single_dataset_deltas": final_fair,
        "lodo": lodo,
        "final_lodo": final_lodo,
        "rule_baselines": rule,
        "completion_relevant_requirements": {
            name: requirements.get(name, {})
            for name in [
                "lodo_complete",
                "fair_single_dataset_complete",
                "final_method_complete",
            ]
        },
        "strict_comparison_rule": (
            "Use external published numbers only when label space, train/val/test split, and metric definition match. "
            "Use the matched ConvNeXt single-dataset baseline as the main fair visual baseline when published protocols do not match."
        ),
        "route_verdict": (
            "Recommended route: hierarchical public-data protocol. Keep RSCD, RoadSaW, and RoadSC, "
            "but do not pool them as one i.i.d. benchmark. Use RSCD for RSCD-style public benchmark claims, "
            "RoadSaW/RoadSC for wet/snow friction-risk stress tests, and LODO only as OOD evidence."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External Benchmark And Fair-Comparison Report",
        "",
        f"Summary dir: `{report['summary_dir']}`",
        "",
        "## Strict Comparison Rule",
        "",
        report["strict_comparison_rule"],
        "",
        f"Route verdict: {report['route_verdict']}",
        "",
        "Combined multi-dataset results are useful for method diagnosis, but they are not a clean replacement for published single-dataset benchmark comparisons. The paper claim should rely on matched single-dataset FAF vs ConvNeXt rows and LODO rows.",
        "",
        "## Public Sources",
        "",
        "| Source | Role | Link |",
        "|---|---|---|",
    ]
    for item in report["public_sources"]:
        lines.append(f"| {item['name']} | {item['role']} | {item['url']} |")

    lines.extend(["", "## Dataset Alignment", ""])
    lines.append("| Dataset | Fair comparison unit | Task alignment | Primary metrics | Status |")
    lines.append("|---|---|---|---|---|")
    for item in report["dataset_alignment"]:
        lines.append(
            "| {dataset} | `{fair_unit}` | {task_alignment} | {metrics} | `{status}` |".format(
                dataset=item["dataset"],
                fair_unit=item["fair_unit"],
                task_alignment=item["task_alignment"],
                metrics=item["primary_metrics"],
                status=item["claim_status"],
            )
        )

    lines.extend(["", "## Dataset Route Decision", ""])
    lines.append("| Route | Decision | Reason | Risk | Required evidence |")
    lines.append("|---|---|---|---|---|")
    for item in report.get("dataset_route_decision", []):
        lines.append(
            "| {route} | `{decision}` | {reason} | {risk} | {evidence} |".format(
                route=item["route"],
                decision=item["decision"],
                reason=item["reason"],
                risk=item["risk"],
                evidence=item["required_evidence"],
            )
        )

    lines.extend(["", "## Direct Friction Benchmark Boundary", ""])
    lines.append("| Paper/Dataset | Target | Public comparability | Why numeric comparison is not yet fair | Usable idea | Local action |")
    lines.append("|---|---|---|---|---|---|")
    for item in report.get("direct_friction_benchmark_rows", []):
        lines.append(
            "| {paper} | {target} | `{level}` | {why} | {idea} | {action} |".format(
                paper=item["paper_or_dataset"],
                target=item["target"],
                level=item["public_comparability"],
                why=item["why_not_fair_numeric"],
                idea=item["usable_idea"],
                action=item["local_action"],
            )
        )

    lines.extend(["", "## Fair Claim Ladder", ""])
    lines.append("| Level | Allowed claim | Required artifact |")
    lines.append("|---|---|---|")
    for item in report.get("fair_claim_ladder", []):
        lines.append(
            "| `{level}` | {claim} | {artifact} |".format(
                level=item["level"],
                claim=item["allowed_claim"],
                artifact=item["required_artifact"],
            )
        )

    lines.extend(["", "## External Comparability Matrix", ""])
    lines.append("| Source | Task/Dataset | Reported metric | Split | Label | Metric | Level | Allowed use | Required action |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for item in report.get("comparability_matrix", []):
        lines.append(
            "| {source} | {task} | {metric} | `{split}` | `{label}` | `{metric_match}` | `{level}` | {allowed} | {action} |".format(
                source=item.get("source"),
                task=item.get("task_or_dataset"),
                metric=item.get("reported_metric"),
                split=item.get("split_match"),
                label=item.get("label_match"),
                metric_match=item.get("metric_match"),
                level=item.get("comparison_level"),
                allowed=item.get("allowed_use"),
                action=item.get("required_action"),
            )
        )

    lines.extend(["", "## Matched Single-Dataset Evidence", ""])
    lines.append("| Dataset | Status | delta friction F1 | delta risk F1 | delta low recall | delta calibrated coverage |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in report.get("fair_single_dataset_deltas", []):
        lines.append(
            "| {dataset} | {status} | {df} | {dr} | {dl} | {dc} |".format(
                dataset=row.get("dataset"),
                status=row.get("status"),
                df=fmt_delta(row.get("delta_friction_macro_f1")),
                dr=fmt_delta(row.get("delta_risk_macro_f1")),
                dl=fmt_delta(row.get("delta_low_friction_recall")),
                dc=fmt_delta(row.get("delta_calibrated_coverage")),
            )
        )

    lines.extend(["", "## LODO Generalization Evidence", ""])
    lines.append("| Held-out dataset | Status | friction F1 | risk F1 | low recall | calibrated coverage | worst F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in report.get("lodo", []):
        lines.append(
            "| {method} | {status} | {friction} | {risk} | {low} | {cov} | {worst} |".format(
                method=row.get("method"),
                status=row.get("status"),
                friction=fmt_pct(row.get("friction_macro_f1")),
                risk=fmt_pct(row.get("risk_macro_f1")),
                low=fmt_pct(row.get("low_friction_recall")),
                cov=fmt_pct(row.get("calibrated_coverage")),
                worst=fmt_pct(row.get("worst_dataset_f1")),
            )
        )

    lines.extend(["", "## Final Method Evidence", ""])
    lines.append("| Dataset | Status | final delta friction F1 | final delta risk F1 | final delta low recall | final delta calibrated coverage |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in report.get("final_fair_single_dataset_deltas", []):
        lines.append(
            "| {dataset} | {status} | {df} | {dr} | {dl} | {dc} |".format(
                dataset=row.get("dataset"),
                status=row.get("status"),
                df=fmt_delta(row.get("delta_friction_macro_f1")),
                dr=fmt_delta(row.get("delta_risk_macro_f1")),
                dl=fmt_delta(row.get("delta_low_friction_recall")),
                dc=fmt_delta(row.get("delta_calibrated_coverage")),
            )
        )

    lines.extend(["", "## Non-Visual Rule Baseline", ""])
    lines.append("| Dataset | Status | coverage | avg width | mid MAE | Note |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in report.get("rule_baselines", []):
        lines.append(
            "| {dataset} | {status} | {coverage} | {width} | {mae} | {note} |".format(
                dataset=row.get("dataset"),
                status=row.get("status"),
                coverage=fmt_pct(row.get("coverage")),
                width=fmt_abs(row.get("avg_width")),
                mae=fmt_abs(row.get("mid_mae")),
                note=row.get("note", "-"),
            )
        )

    lines.extend(["", "## Completion Status", ""])
    for name, item in report.get("completion_relevant_requirements", {}).items():
        status = item.get("status", "missing")
        missing = ", ".join(item.get("missing", []) or [])
        lines.append(f"- `{name}`: `{status}`" + (f"; missing: {missing}" if missing else ""))
    lines.append("")
    return "\n".join(lines)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def fmt_delta(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):+.2f}"


def fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()

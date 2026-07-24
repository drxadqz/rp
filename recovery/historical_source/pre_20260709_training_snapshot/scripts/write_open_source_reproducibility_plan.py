from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


OPEN_SOURCE_ROWS = [
    {
        "name": "DomainBed",
        "venue_or_role": "Domain generalization benchmark practice",
        "official_url": "https://github.com/facebookresearch/DomainBed",
        "paper_url": "https://openreview.net/forum?id=lQdXeXDoWtI",
        "project_use": "Protocol inspiration only: held-out-domain LODO, fair ERM/ConvNeXt baselines, fixed splits, and no OOD claim before held-out RoadSaW evidence.",
        "integration_status": "protocol_aligned",
        "current_artifacts": [
            "lodo_*_full_faf configs",
            "baseline_single_*_global_convnext configs",
            "dataset_id_diagnostic.json",
            "topvenue_readiness_gate.json",
        ],
        "do_not_claim": "Do not claim a DomainBed benchmark result; the datasets and label space are project-specific.",
    },
    {
        "name": "TorchVision ConvNeXt",
        "venue_or_role": "Strong CNN public baseline",
        "official_url": "https://pytorch.org/vision/stable/models/convnext.html",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html",
        "project_use": "Main matched visual baseline: global ConvNeXt trained on the same public dataset split and label mapping as FAF.",
        "integration_status": "implemented_in_configs_pending_runs",
        "current_artifacts": [
            "baseline_single_rscd_global_convnext.yaml",
            "baseline_single_roadsaw_global_convnext.yaml",
            "baseline_single_roadsc_global_convnext.yaml",
            "fair_pairwise paired bootstrap scripts",
        ],
        "do_not_claim": "Do not compare against published numbers unless split, label mapping, and metric definitions match.",
    },
    {
        "name": "FDA Fourier Domain Adaptation",
        "venue_or_role": "Style-shift mitigation",
        "official_url": "https://github.com/YanchaoYang/FDA",
        "paper_url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html",
        "project_use": "Implemented as Fourier low-frequency amplitude jitter, preserving phase/geometry while perturbing camera and dataset style.",
        "integration_status": "implemented_candidate_pending_results",
        "current_artifacts": [
            "src/friction_affordance/transforms.py",
            "v6-v12/v14/v15/v16/final configs",
            "algorithm_module_audit.json",
        ],
        "do_not_claim": "Do not claim full FDA semantic-segmentation adaptation; the project uses a lightweight augmentation inspired by FDA.",
    },
    {
        "name": "DANN",
        "venue_or_role": "Domain-adversarial representation learning",
        "official_url": "https://jmlr.org/papers/v17/15-239.html",
        "paper_url": "https://jmlr.org/papers/v17/15-239.html",
        "project_use": "Gradient-reversal domain head is a candidate for reducing dataset-ID shortcut, judged by dataset-ID probe plus task metrics.",
        "integration_status": "implemented_candidate_pending_results",
        "current_artifacts": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/losses.py",
            "v7_full_faf_fourier_dann.yaml",
        ],
        "do_not_claim": "Do not keep DANN if it reduces safety metrics or worst-dataset performance.",
    },
    {
        "name": "GroupDRO",
        "venue_or_role": "Worst-group robustness",
        "official_url": "https://github.com/kohpangwei/group_DRO",
        "paper_url": "https://arxiv.org/abs/1911.08731",
        "project_use": "Group-aware losses and reports motivate worst-dataset, RoadSaW wet-state, and conditional interval watchlists.",
        "integration_status": "partly_implemented_needs_selection",
        "current_artifacts": [
            "loss_group_dro",
            "loss_group_vrex",
            "dataset_shortcut_report.json",
            "wetness_state_report.json",
        ],
        "do_not_claim": "Current P0 DG losses are provisional because they hurt primary metrics; final route must be evidence-selected.",
    },
    {
        "name": "DINOv2",
        "venue_or_role": "Foundation visual representation candidate",
        "official_url": "https://github.com/facebookresearch/dinov2",
        "paper_url": "https://arxiv.org/abs/2304.07193",
        "project_use": "Future optional frozen-feature or backbone baseline if ConvNeXt/FAF evidence remains weak after queued runs.",
        "integration_status": "future_only_not_current_claim",
        "current_artifacts": [],
        "do_not_claim": "Do not mention DINOv2 as part of the method unless a matched run is implemented and audited.",
    },
    {
        "name": "Extreme Road Image Dataset",
        "venue_or_role": "Future public extreme-road image data",
        "official_url": "https://github.com/sean-shiyuez/Extreme-Road-Image-Dataset",
        "paper_url": "https://doi.org/10.1016/j.ymssp.2024.112039",
        "project_use": "Future validation or multimodal extension route for direct friction-estimation research after split, label, license, and metric audits.",
        "integration_status": "future_dataset_candidate_not_current_claim",
        "current_artifacts": [],
        "do_not_claim": "Do not mix it into the current RSCD/RoadSaW/RoadSC benchmark or claim direct measured friction without a separate protocol.",
    },
    {
        "name": "Continual road-surface classification",
        "venue_or_role": "Cross-dataset adaptation reference",
        "official_url": "https://github.com/PCudrano/continual_road_surface_classification",
        "paper_url": "https://arxiv.org/abs/2309.02210",
        "project_use": "Future continual or test-time adaptation route if LODO shows severe RoadSaW/RoadSC degradation.",
        "integration_status": "future_protocol_candidate",
        "current_artifacts": [],
        "do_not_claim": "Do not add continual adaptation to the final method without controlled configs, matched baselines, and leakage checks.",
    },
    {
        "name": "Segment Anything",
        "venue_or_role": "Road-region pseudo-mask candidate",
        "official_url": "https://github.com/facebookresearch/segment-anything",
        "paper_url": "https://arxiv.org/abs/2304.02643",
        "project_use": "Possible offline pseudo-road-mask generator; current queue uses simple bottom-road/road-prior constraints instead of a deployed SAM dependency.",
        "integration_status": "future_optional_pseudo_label_source",
        "current_artifacts": [
            "evidence_attention_prior",
            "bottom-road ROI constraints",
            "v12/v14/v15/v16/final configs",
        ],
        "do_not_claim": "Do not claim SAM supervision unless masks are actually generated, versioned, and audited.",
    },
    {
        "name": "Mask2Former",
        "venue_or_role": "Segmentation foundation for road ROI",
        "official_url": "https://github.com/facebookresearch/Mask2Former",
        "paper_url": "https://arxiv.org/abs/2112.01527",
        "project_use": "Alternative pseudo-road-mask source if SAM is not stable for road scenes; use only offline to supervise EvidenceField attention.",
        "integration_status": "future_optional_pseudo_label_source",
        "current_artifacts": [
            "evidence_field_audit.json",
            "evidence_maps",
            "road ROI candidate configs",
        ],
        "do_not_claim": "Do not add a segmentation dependency to the method without a controlled ablation.",
    },
    {
        "name": "RAPS / conformal prediction",
        "venue_or_role": "Calibration and uncertainty reporting",
        "official_url": "https://openreview.net/forum?id=eNdiU_DbM9",
        "paper_url": "https://openreview.net/forum?id=eNdiU_DbM9",
        "project_use": "Motivates reporting coverage together with interval width, plus pooled and conditional calibration.",
        "integration_status": "implemented_reporting_and_calibration",
        "current_artifacts": [
            "interval_calibration_90.json",
            "interval_quality_report.json",
            "paper_tables.tex",
        ],
        "do_not_claim": "Do not claim distribution-free coverage beyond the exact split/calibration protocol used in the project.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "open_source_reproducibility_plan.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "open_source_reproducibility_plan.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    rows = [dict(row) for row in OPEN_SOURCE_ROWS]
    implemented = [
        row
        for row in rows
        if row["integration_status"]
        in {
            "implemented_in_configs_pending_runs",
            "implemented_candidate_pending_results",
            "implemented_reporting_and_calibration",
            "partly_implemented_needs_selection",
        }
    ]
    future_only = [row for row in rows if row["integration_status"].startswith("future")]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "claim_policy": (
            "Use open-source/top-venue projects as protocol, baseline, augmentation, "
            "or future-candidate references only when the corresponding code path, "
            "config, and result artifacts exist. The main fair comparison remains "
            "same-split FAF versus global ConvNeXt plus LODO generalization."
        ),
        "rows": rows,
        "num_sources": len(rows),
        "num_implemented_or_configured": len(implemented),
        "num_future_only": len(future_only),
        "strict_claim_rules": [
            "A cited repository is not a result unless a matching run exists in the paper protocol root.",
            "Published numbers are used only when label space, split, and metric definition match.",
            "If they do not match, use matched ConvNeXt and rule/conformal baselines instead.",
            "Foundation or segmentation models may create future pseudo-labels, but they must be versioned and ablated before inclusion.",
            "Weak visual friction intervals are not measured tire-road friction coefficients.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Open-Source Reproducibility Plan",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Claim Policy",
        "",
        report["claim_policy"],
        "",
        "## Source Map",
        "",
        "| Source | Role | Status | Official URL | Project use |",
        "|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {name} | {role} | `{status}` | {url} | {use} |".format(
                name=row["name"],
                role=row["venue_or_role"],
                status=row["integration_status"],
                url=row["official_url"],
                use=row["project_use"],
            )
        )
    lines.extend(["", "## Strict Claim Rules", ""])
    for item in report["strict_claim_rules"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Current Decision", ""])
    lines.append(
        "The project should not chase a larger stack of external models until the queued P0, LODO, single-dataset, and baseline runs finish. "
        "If held-out RoadSaW or matched ConvNeXt comparisons are weak, the next controlled extension should be one of: "
        "DINOv2 frozen-feature baseline, offline SAM/Mask2Former road masks, or a stricter state-conditioned alignment variant."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

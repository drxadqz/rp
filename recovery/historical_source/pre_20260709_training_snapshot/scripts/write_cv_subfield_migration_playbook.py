from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/cv_subfield_migration_playbook.md")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/cv_subfield_migration_playbook.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a project-specific playbook for transferring ideas from "
            "semantic segmentation and related CV subfields into visual "
            "friction-affordance estimation."
        )
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report()
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report() -> dict[str, Any]:
    rows = [
        {
            "rank": 1,
            "subfield": "Semantic segmentation / mask classification",
            "papers": ["Mask2Former", "Mask DINO", "SegFormer", "OneFormer"],
            "transfer": (
                "Replace a single global image vector with local road-material regions, "
                "attention maps, and mask-like pooling. The important migration is not "
                "pixel labels; it is region-wise reasoning under weak image labels."
            ),
            "implementation": (
                "Already mapped to EvidenceField, ROI attention, road likelihood, and "
                "v23 region-mixture evidence. v24 adds segmentation-style multi-query "
                "local evidence masks with query-disagreement interval expansion. v25 "
                "adds masked-query consistency so the same local road evidence must remain "
                "stable after region masking. Add external pseudo-mask supervision only "
                "after the mask audit is stable."
            ),
            "data_fit": "High: image-level road-state labels are enough for weak MIL pooling.",
            "reviewer_risk": (
                "If it only widens intervals without improving wet/snow/low-friction "
                "slices, reviewers will call it heuristic complexity."
            ),
            "decision": "Primary route; run v23/v24/v25 and compare to v21/v22/no-mask.",
        },
        {
            "rank": 2,
            "subfield": "Semi-supervised semantic segmentation consistency",
            "papers": ["UniMatch", "MIC", "ClassMix", "AugSeg"],
            "transfer": (
                "Use weak-to-strong consistency, but define consistency on friction-relevant "
                "outputs: risk logits, friction interval, and road-limited evidence attention. "
                "The segmentation insight is that perturbations should hide regions and change "
                "style while preserving the object/material decision."
            ),
            "implementation": (
                "v10 provides the earlier weak-view consistency route. v25 is the sharper "
                "version: it applies MIC-style random block masks to the weak view and uses "
                "mask-aware attention consistency through the road-likelihood support."
            ),
            "data_fit": "High: no extra labels are required; only paired augmented views are needed.",
            "reviewer_risk": (
                "Can over-smooth rare slippery states. Keep only if low-friction recall, "
                "RoadSaW wetness, and RoadSC snow slices do not collapse."
            ),
            "decision": "Run v25 as the main consistency probe; prune if it beats attention metrics but hurts safety metrics.",
        },
        {
            "rank": 3,
            "subfield": "Promptable/open-vocabulary segmentation",
            "papers": ["SAM", "SAM 2", "CLIPSeg", "ODISE"],
            "transfer": (
                "Use external foundation models as offline teachers for road/contact/wet/"
                "snow pseudo masks, not as ground truth and not as a hidden test-time "
                "dependency."
            ),
            "implementation": (
                "Mask cache, manifest road_mask_path, dataloader bridge, and pseudo-road "
                "attention loss are implemented. CLIPSeg/SAM audits are queued after the "
                "formal GPU run to avoid interrupting training."
            ),
            "data_fit": "Medium: works only when pseudo masks align with actual road/contact evidence.",
            "reviewer_risk": (
                "Prompt masks can segment texture fragments or background; must show mask "
                "quality and attention-on-road diagnostics before claiming a method gain."
            ),
            "decision": "Conditional route; promote only after mask audit and bounded candidate metrics.",
        },
        {
            "rank": 4,
            "subfield": "Domain-adaptive semantic segmentation",
            "papers": ["DAFormer", "HRDA", "MIC", "FDA", "MixStyle", "DomainBed"],
            "transfer": (
                "Use rare-condition sampling, low-frequency style perturbation, feature-stat "
                "mixing, and strict held-domain evaluation to attack dataset/camera shortcuts."
            ),
            "implementation": (
                "Fourier jitter, MixStyle, conditional CORAL, contrastive alignment, domain "
                "adapter, and LODO audits are implemented or queued."
            ),
            "data_fit": (
                "High for stress testing, but the LODO evidence so far says naive "
                "cross-dataset generalization is failing."
            ),
            "reviewer_risk": (
                "DANN/full DG already hurt P0 safety metrics; keep only modules that reduce "
                "dataset-ID shortcut without harming risk F1 and low-friction recall."
            ),
            "decision": "Use as pruning discipline, not as automatic final method.",
        },
        {
            "rank": 5,
            "subfield": "Weakly supervised segmentation / multiple-instance learning",
            "papers": ["image-level weak segmentation", "MIL pooling", "CAM/attention supervision"],
            "transfer": (
                "Treat each image as a bag of road-material patches. The label supervises the "
                "bag, while latent region weights decide which patches explain dry/wet/snow/ice "
                "and the friction interval."
            ),
            "implementation": (
                "EvidenceField already implements patch-grid risk and interval maps with "
                "attention pooling. v23/v24/v25 are the stronger forms because they add "
                "region mixture, multiple latent queries, and query disagreement uncertainty."
            ),
            "data_fit": "High: exactly matches public image-level labels without pixel labels.",
            "reviewer_risk": (
                "Attention maps are not proof by themselves; they need task-metric, hard-slice, "
                "and calibrated-interval evidence."
            ),
            "decision": "Use as the paper's methodological language if v23/v24/v25 improve hard slices.",
        },
        {
            "rank": 6,
            "subfield": "Foundation dense representation",
            "papers": ["DINOv2", "MAE-style self-supervision"],
            "transfer": (
                "Use frozen dense tokens as robust texture/material descriptors, or distill "
                "their local token clusters into the small EvidenceField branch."
            ),
            "implementation": (
                "DINOv2 probe path exists. Full integration should wait until the current "
                "formal queue is idle because it can be slower and memory-sensitive."
            ),
            "data_fit": "Medium-high: useful as a strong baseline and potential teacher.",
            "reviewer_risk": (
                "If it is just a larger backbone, the innovation claim is weak. The claim "
                "must be dense-token distillation or material-region uncertainty."
            ),
            "decision": "Run as strong baseline/teacher after fair ConvNeXt rows.",
        },
        {
            "rank": 7,
            "subfield": "Material/texture recognition and physical vision",
            "papers": ["physics-inspired texture cues", "wet/glare/low-texture ambiguity"],
            "transfer": (
                "Treat wetness, snow, glare, low texture, and material mixture as uncertainty "
                "signals for a friction interval, not just as classes."
            ),
            "implementation": (
                "PhysicsTexture, visual-quality coverage weights, near-white/low-texture/"
                "specular weighting, and v17/v21/v22/v23 are already configured."
            ),
            "data_fit": "High: public labels and raw RGB support this directly.",
            "reviewer_risk": "Must prove the model is not merely learning dataset brightness artifacts.",
            "decision": "Core route; protect PhysicsTexture and quality-aware intervals.",
        },
        {
            "rank": 8,
            "subfield": "Uncertainty, calibration, and conformal prediction",
            "papers": ["risk-controlled prediction", "conformal calibration"],
            "transfer": (
                "Because datasets provide weak friction intervals rather than measured mu, "
                "optimize coverage-width tradeoff and condition-specific safety coverage."
            ),
            "implementation": (
                "Interval heads, coverage losses, bootstrap metrics, and calibration reports "
                "are already in the paper protocol."
            ),
            "data_fit": "High: exactly matches weak-label friction-affordance setting.",
            "reviewer_risk": "Wide intervals alone are not a win; report calibrated coverage and width.",
            "decision": "Keep as part of the scientific claim boundary.",
        },
        {
            "rank": 9,
            "subfield": "Monocular depth / geometry",
            "papers": ["Depth Anything V2"],
            "transfer": (
                "Use depth to separate near-road contact region from background in full driving "
                "scenes."
            ),
            "implementation": "Not worth formal GPU time for current patch/square-crop datasets.",
            "data_fit": "Low for RSCD/RoadSaW/RoadSC; maybe useful only with full-scene datasets.",
            "reviewer_risk": "Depth on close road patches can be flat or noisy.",
            "decision": "Demote for the current paper route.",
        },
        {
            "rank": 10,
            "subfield": "Image restoration / adverse-weather vision",
            "papers": ["deraining/dehazing/enhancement for recognition"],
            "transfer": (
                "Adverse-weather models can normalize haze/rain/illumination, but they may "
                "erase precisely the wetness and glare cues that carry friction evidence."
            ),
            "implementation": "No formal candidate unless a failure slice proves weather artifacts dominate labels.",
            "data_fit": "Low-medium: useful for diagnostics, risky as a training transform.",
            "reviewer_risk": "A reviewer can object that restoration changes the physical evidence.",
            "decision": "Do not prioritize now; keep only as a later diagnostic ablation.",
        },
    ]
    algorithm = {
        "name": "Segmentation-Transferred Friction Affordance Field",
        "short_name": "ST-FAF",
        "core": [
            "Local evidence tokens over the road surface instead of global-only classification.",
            "Optional multi-query mask-style evidence pooling for heterogeneous wet/dry/snow/glare regions.",
            "MIC-style masked consistency so local material evidence survives region occlusion and camera-style perturbations.",
            "Mask/ROI constrained evidence attention using heuristic road likelihood now and external pseudo masks only after audit.",
            "Region-mixture uncertainty expansion for spatially mixed wet/snow/specular/rough-texture regions.",
            "Weak-to-strong consistency on logits, friction interval, and evidence attention.",
            "Safety-weighted interval training and condition-specific calibration.",
            "Strict single-dataset FAF-vs-ConvNeXt primary comparison plus LODO stress testing.",
        ],
        "first_experiments": [
            "Finish current formal queue through matched single-dataset ConvNeXt baselines.",
            "Run v17/v21/v22/v23/v24/v25 and prune by low-friction recall, RoadSaW wet/white slices, calibrated coverage, and interval width.",
            "Promote v25 only if masked query consistency improves hard slices or coverage-width beyond v24 without suppressing low-friction recall.",
            "After GPU queue is idle, run CLIPSeg/SAM mask audit and only then a bounded mask-supervised candidate.",
            "Run DINOv2 as a strong teacher/baseline, not as an unearned method component.",
        ],
    }
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": "segmentation_style_local_evidence_is_the_best_transfer_route",
        "claim_boundary": (
            "The project should claim visual friction-affordance intervals from public road-condition "
            "labels, not measured tire-road friction coefficients, unless measured friction data are added."
        ),
        "rows": rows,
        "algorithm": algorithm,
        "sources": sources(),
    }


def sources() -> list[dict[str, str]]:
    return [
        {"name": "Mask2Former", "url": "https://arxiv.org/abs/2112.01527"},
        {"name": "OneFormer", "url": "https://arxiv.org/abs/2211.06220"},
        {"name": "Segment Anything", "url": "https://arxiv.org/abs/2304.02643"},
        {"name": "SAM 2", "url": "https://arxiv.org/abs/2408.00714"},
        {"name": "SegFormer", "url": "https://arxiv.org/abs/2105.15203"},
        {"name": "CLIPSeg", "url": "https://arxiv.org/abs/2112.10003"},
        {"name": "DINOv2", "url": "https://arxiv.org/abs/2304.07193"},
        {"name": "Depth Anything V2", "url": "https://arxiv.org/abs/2406.09414"},
        {"name": "DAFormer", "url": "https://arxiv.org/abs/2111.14887"},
        {"name": "HRDA", "url": "https://arxiv.org/abs/2204.13132"},
        {"name": "UniMatch", "url": "https://arxiv.org/abs/2208.09910"},
        {"name": "MIC", "url": "https://arxiv.org/abs/2212.01322"},
        {"name": "ClassMix", "url": "https://arxiv.org/abs/2007.07936"},
        {"name": "Mask DINO", "url": "https://arxiv.org/abs/2206.02777"},
        {"name": "ODISE", "url": "https://arxiv.org/abs/2303.04803"},
        {
            "name": "FDA",
            "url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html",
        },
        {"name": "MixStyle", "url": "https://openreview.net/forum?id=6xHJ37MVxxp"},
        {"name": "DomainBed", "url": "https://openreview.net/forum?id=lQdXeXDoWtI"},
        {"name": "Tent", "url": "https://arxiv.org/abs/2006.10726"},
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CV Subfield Migration Playbook",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Transfer Ranking",
        "",
        "| Rank | CV subfield | Transfer mechanism | Project implementation | Data fit | Decision |",
        "|---:|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {rank} | {subfield} | {transfer} | {implementation} | {data_fit} | {decision} |".format(
                rank=row["rank"],
                subfield=row["subfield"],
                transfer=row["transfer"],
                implementation=row["implementation"],
                data_fit=row["data_fit"],
                decision=row["decision"],
            )
        )
    lines.extend(["", "## Reviewer Risks", ""])
    for row in report["rows"]:
        lines.append(f"- `{row['subfield']}`: {row['reviewer_risk']}")
    algorithm = report["algorithm"]
    lines.extend(
        [
            "",
            "## Recommended Algorithm Route",
            "",
            f"Name: `{algorithm['name']}` (`{algorithm['short_name']}`)",
            "",
            "Core components:",
        ]
    )
    lines.extend(f"- {item}" for item in algorithm["core"])
    lines.extend(["", "First experiments:"])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(algorithm["first_experiments"], start=1))
    lines.extend(["", "## Source Anchors", ""])
    lines.extend("| Source | URL |".splitlines())
    lines.append("|---|---|")
    for source in report["sources"]:
        lines.append(f"| {source['name']} | {source['url']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


SOURCE_ROWS = [
    {
        "group": "dataset",
        "name": "RoadSaW",
        "url": "https://openaccess.thecvf.com/content/CVPR2022W/WAD/html/Cordes_RoadSaW_A_Large-Scale_Dataset_for_Camera-Based_Road_Surface_and_Wetness_CVPRW_2022_paper.html",
        "role": "public road-surface and wetness dataset",
        "use_in_project": "held-out RoadSaW LODO, RoadSaW single-dataset FAF, and RoadSaW ConvNeXt baseline",
        "claim_boundary": "Supports visual wetness/surface-state evaluation; not measured tire-road friction.",
    },
    {
        "group": "dataset_download",
        "name": "RoadSaW dataset page",
        "url": "https://viscoda.com/index.php/de/downloads-de/roadsaw-dataset-de",
        "role": "dataset availability and citation page",
        "use_in_project": "documents that RoadSaW is a public/reusable held-out stress dataset",
        "claim_boundary": "The dataset provides surface/wetness labels from road-weather sensing, not image-level friction coefficients.",
    },
    {
        "group": "dataset",
        "name": "RoadSC",
        "url": "https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/html/Cordes_Camera-Based_Road_Snow_Coverage_Estimation_ICCVW_2023_paper.html",
        "role": "public road snow-coverage dataset",
        "use_in_project": "held-out RoadSC LODO and RoadSC single-dataset winter-state comparison",
        "claim_boundary": "Supports snow/ice proxy stress testing; not direct adhesion measurement.",
    },
    {
        "group": "dataset_download",
        "name": "RoadSC dataset page",
        "url": "https://viscoda.com/index.php/en/downloads-en/roadsc-dataset",
        "role": "dataset availability and citation page",
        "use_in_project": "documents that RoadSC is a public/reusable winter-road stress dataset",
        "claim_boundary": "Snow coverage and wetness states are proxy labels for visual friction affordance.",
    },
    {
        "group": "dataset",
        "name": "RSCD",
        "url": "https://thu-rsxd.com/rscd/",
        "role": "public road-surface condition dataset",
        "use_in_project": "RSCD single-dataset FAF vs ConvNeXt and held-out RSCD LODO",
        "claim_boundary": "Use only same-split, same-label comparisons as numeric evidence.",
    },
    {
        "group": "dataset_download",
        "name": "RSCD GitHub",
        "url": "https://github.com/ztsrxh/RSCD-Road_Surface_Classification_Dataset",
        "role": "public dataset repository and dataset news",
        "use_in_project": "documents open-source dataset availability and label taxonomy for reproducibility",
        "claim_boundary": "Repository metadata is not a baseline result; numeric claims require matched protocol runs.",
    },
    {
        "group": "future_multimodal_dataset",
        "name": "ROAD Camera-IMU road-surface dataset",
        "url": "https://arxiv.org/abs/2601.20847",
        "role": "2026 road-surface classification dataset with synchronized RGB-IMU and vision-only subsets",
        "use_in_project": "future multimodal extension only if public files, splits, labels, and license are confirmed and a separate protocol is added",
        "claim_boundary": "Do not mix into current RSCD/RoadSaW/RoadSC main evidence; it is a future route for camera+vehicle-dynamics fusion, not current visual-only friction-affordance evidence.",
    },
    {
        "group": "future_multimodal_dataset",
        "name": "Extreme Road Image Dataset",
        "url": "https://github.com/sean-shiyuez/Extreme-Road-Image-Dataset",
        "role": "public image dataset associated with image+dynamics tire-road friction estimation research",
        "use_in_project": "separate direct-visual-friction route; local raw files and manifests are audited, and global ConvNeXt/FAF same-task rows are queued",
        "claim_boundary": "Keep it separate from the current RSCD/RoadSaW/RoadSC weak-label benchmark because the label semantics and target task differ.",
    },
    {
        "group": "external_benchmark",
        "name": "RoadFormer",
        "url": "https://arxiv.org/abs/2506.02358",
        "role": "recent local-global road-surface classification reference on RSCD-scale data",
        "use_in_project": "external RSCD reference and architectural inspiration for local-global/evidence grounding",
        "claim_boundary": "Use as context only unless its split, labels, preprocessing, and metric definition exactly match the local RSCD protocol.",
    },
    {
        "group": "external_benchmark",
        "name": "WCamNet",
        "url": "https://arxiv.org/abs/2404.16578",
        "role": "DINOv2+CNN visual friction-regression reference for winter roadside camera images",
        "use_in_project": "contextual evidence that foundation visual features plus local texture modeling are a plausible future baseline",
        "claim_boundary": "Use as method inspiration only until the dataset, labels, split, and friction target are reproduced locally.",
    },
    {
        "group": "external_benchmark_code",
        "name": "WCamNet GitHub",
        "url": "https://github.com/ojalar/wcamnet",
        "role": "MIT-licensed implementation of a DINOv2+CNN visual road-friction model",
        "use_in_project": "future code-level reference for a WCamNet-style local baseline if a protocol-equivalent measured-friction dataset is available",
        "claim_boundary": "Code availability alone is not a numeric benchmark; do not compare scores without matching data, labels, splits, and metric.",
    },
    {
        "group": "future_dataset",
        "name": "WARD Weather-Aware Road Dataset",
        "url": "https://library.utia.cas.cz/separaty/2026/ZOI/nesnidalova-0644268.pdf",
        "role": "recent road-weather/road-surface-condition dataset candidate with driving-scene images and weather-aware labels",
        "use_in_project": "future public-data expansion only after download availability, license, splits, and label compatibility are audited",
        "claim_boundary": "Do not mix into current RSCD/RoadSaW/RoadSC evidence until files and labels are reproducible locally.",
    },
    {
        "group": "external_benchmark",
        "name": "SIWNet",
        "url": "https://arxiv.org/abs/2310.00923",
        "role": "image-based road-friction regression with prediction-interval estimation",
        "use_in_project": "method inspiration for interval calibration, coverage-width reporting, and uncertainty-aware friction-affordance intervals",
        "claim_boundary": "Sensor-ground-truth friction regression is not the same target as public RSCD/RoadSaW/RoadSC weak labels.",
    },
    {
        "group": "future_dataset",
        "name": "Finnish Winter Driving Dataset",
        "url": "https://zenodo.org/records/14856338",
        "role": "public winter driving images with segmentation/GNSS/INS/lidar annotations",
        "use_in_project": "future winter-road mask/domain source only after license, splits, and label compatibility are audited",
        "claim_boundary": "No direct friction labels for current task; keep outside current main protocol.",
    },
    {
        "group": "fair_baseline",
        "name": "ConvNeXt",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html",
        "role": "strong modern CNN baseline",
        "use_in_project": "matched single-dataset global ConvNeXt baseline",
        "claim_boundary": "Primary fair baseline only after identical splits, label mapping, metrics, and calibration are complete.",
    },
    {
        "group": "protocol",
        "name": "DomainBed",
        "url": "https://openreview.net/forum?id=lQdXeXDoWtI",
        "role": "domain-generalization protocol sanity check",
        "use_in_project": "LODO protocol, strong ERM/ConvNeXt baseline discipline, and no-OOD-claim gate",
        "claim_boundary": "Protocol reference, not a numeric baseline for road friction.",
    },
    {
        "group": "protocol",
        "name": "Continual Cross-Dataset Adaptation",
        "url": "https://arxiv.org/abs/2309.02210",
        "role": "road-surface classification study showing poor cross-dataset generalization and continual adaptation options",
        "use_in_project": "supports LODO-first evaluation and future adaptation experiments if held-out RoadSaW or RoadSC fails",
        "claim_boundary": "Not a direct baseline for current friction-affordance metrics; use only for protocol motivation unless reimplemented.",
    },
    {
        "group": "shortcut_mitigation",
        "name": "MixStyle",
        "url": "https://openreview.net/forum?id=6xHJ37MVxxp",
        "role": "feature-statistics style randomization for domain generalization",
        "use_in_project": "v18 training-only grouped Feature MixStyle shortcut probe on shared normalized features",
        "claim_boundary": "Keep only if dataset-ID shortcut or worst-domain behavior improves without hurting risk F1, low-friction recall, or interval coverage-width.",
    },
    {
        "group": "shortcut_mitigation",
        "name": "Supervised Contrastive Learning",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html",
        "role": "label-aware representation learning that pulls same-label samples together and separates different labels",
        "use_in_project": "v19 cross-dataset same-state contrastive alignment for weak friction/risk/wetness labels",
        "claim_boundary": "Use only as a shortcut/generalization candidate; do not keep it for pooled accuracy alone.",
    },
    {
        "group": "shortcut_mitigation",
        "name": "FDA",
        "url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html",
        "role": "Fourier low-frequency style adaptation",
        "use_in_project": "v6/v7/v8/v10/v11/v12/v14/v15/v16/v17/v18/v19/final Fourier style jitter; v15 tests bottom-square road input canonicalization and v16/v17/v18/v19 add soft Gray-World color constancy",
        "claim_boundary": "Keep only if dataset-ID shortcut drops without safety/generalization collapse.",
    },
    {
        "group": "shortcut_mitigation",
        "name": "Gray-World color constancy",
        "url": "https://alumni.media.mit.edu/~wad/color/exp1/newgray/",
        "role": "classic color-constancy prior for reducing global camera/illumination color cast",
        "use_in_project": "v16 soft Gray-World input canonicalization before Fourier style jitter",
        "claim_boundary": "Use only as a deterministic preprocessing candidate; keep it only if it reduces dataset-ID shortcut without weakening wet-road cues.",
    },
    {
        "group": "shortcut_mitigation",
        "name": "DANN",
        "url": "https://jmlr.org/papers/v17/15-239.html",
        "role": "domain-adversarial feature learning",
        "use_in_project": "v7 candidate after Fourier style jitter",
        "claim_boundary": "Keep only if it reduces dataset predictability and preserves held-out RoadSaW/worst-domain metrics.",
    },
    {
        "group": "robust_optimization",
        "name": "GroupDRO",
        "url": "https://arxiv.org/abs/1911.08731",
        "role": "worst-group robustness",
        "use_in_project": "motivates worst-dataset, RoadSaW wetness, and conditional interval reports",
        "claim_boundary": "Current DG-loss bundle is provisional because completed v3 hurt primary metrics.",
    },
    {
        "group": "evidence_grounding",
        "name": "Segment Anything",
        "url": "https://arxiv.org/abs/2304.02643",
        "role": "optional road-mask pseudo-label source",
        "use_in_project": "future optional offline pseudo-road-mask generator if heuristic ROI is insufficient",
        "claim_boundary": "No claim until masks are generated, versioned, audited, and ablated.",
    },
    {
        "group": "calibration",
        "name": "RAPS conformal prediction",
        "url": "https://openreview.net/forum?id=eNdiU_DbM9",
        "role": "coverage-size uncertainty-set discipline",
        "use_in_project": "coverage-width reporting and conditional calibration watchlists",
        "claim_boundary": "Report coverage together with interval width and subgroup failures.",
    },
    {
        "group": "friction_interval",
        "name": "FHWA pavement-friction primer",
        "url": "https://highways.dot.gov/safety/rwd/keep-vehicles-road/pavement-friction/pavement-friction-road-safety-primer-friction",
        "role": "official pavement-friction framing",
        "use_in_project": "justifies conservative visual friction-affordance intervals",
        "claim_boundary": "Friction depends on tire, pavement, texture, speed, water, and measurement method; images provide weak evidence.",
    },
    {
        "group": "friction_interval",
        "name": "49 CFR 575.104 UTQG",
        "url": "https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-575/subpart-B/section-575.104",
        "role": "official wet traction coefficient sanity anchor",
        "use_in_project": "independent wet-asphalt anchor for ontology sanity checking",
        "claim_boundary": "Regulatory wet traction thresholds are anchors, not public image labels.",
    },
]


DECISION_RULES = [
    {
        "gate": "P0 ablation closure",
        "evidence": "Full model must have test metrics, calibration, bootstrap, and audit artifacts.",
        "action": "Do not finalize any module until the Full model row is complete.",
    },
    {
        "gate": "Module retention",
        "evidence": "Adjacent or candidate row improves at least one safety/generalization/interpretability metric without a major regression.",
        "action": "Keep PhysicsTexture/EvidenceField only if gains survive LODO or fair baselines; remove or merge FrictionSet/DG unless later evidence rescues them.",
    },
    {
        "gate": "Held-out RoadSaW",
        "evidence": "LODO RoadSaW risk F1, friction F1, low-friction recall, wetness macro-F1, conditional coverage, and width.",
        "action": "Use as the first OOD claim gate; weak results become the main failure analysis and motivate P1/P2/P3.",
    },
    {
        "gate": "Fair public comparison",
        "evidence": "Single-dataset FAF vs ConvNeXt rows use identical train/val/test manifests, labels, metrics, calibration, and bootstrap.",
        "action": "Only these matched rows can support numeric claims against a strong baseline when published protocols differ.",
    },
    {
        "gate": "Shortcut reduction",
        "evidence": "Dataset-ID balanced accuracy must drop while risk F1, low-friction recall, and worst-domain F1 remain competitive.",
        "action": "Rank Fourier, bottom-square input canonicalization, DANN, domain adapter, and condition-aware alignment candidates by joint shortcut+safety score.",
    },
    {
        "gate": "Interval quality",
        "evidence": "Raw/calibrated coverage must be reported with width, including dataset/core-state/risk conditional cells.",
        "action": "Prefer coverage-width tradeoff improvements over wider intervals that only inflate coverage.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "online_source_refresh_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "online_source_refresh_report.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    gate = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    watch = _load_json(summary_dir / "active_training_watch_report.json") or {}
    active_live = _load_json(summary_dir / "active_live_training_reports.json") or {}
    p0 = _load_json(summary_dir / "p0_claim_report.json") or {}
    friction_sources = _load_json(summary_dir / "friction_interval_source_audit.json") or {}
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_rows": SOURCE_ROWS,
        "decision_rules": DECISION_RULES,
        "current_status": {
            "readiness": gate.get("verdict"),
            "blocks": gate.get("num_blocks"),
            "warnings": gate.get("num_warnings"),
            "queue": _queue_summary(queue, watch, active_live),
            "p0_status": p0.get("core_status") or p0.get("status"),
            "friction_interval_source_verdict": friction_sources.get("verdict"),
            "friction_interval_source_anchors": len(friction_sources.get("rows", []))
            if isinstance(friction_sources.get("rows"), list)
            else None,
        },
        "paper_claim_policy": [
            "The target is visual friction-affordance interval estimation from public road-condition labels.",
            "Do not describe RSCD/RoadSaW/RoadSC labels as synchronized measured tire-road friction coefficients.",
            "Use external papers and GitHub repositories as protocol/method references unless split, label space, preprocessing, and metric definitions match.",
            "The main numeric comparison is same-split FAF versus same-split ConvNeXt, plus LODO generalization.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = report["current_status"]
    lines = [
        "# Online Source Refresh And Reviewer Rules",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "This report records the current external-source basis for the experiment route. "
        "It distinguishes reusable public datasets, fair baselines, method inspirations, and friction-interval anchors.",
        "",
        "## Current Local Status",
        "",
        f"- Readiness: `{status.get('readiness')}` with `{status.get('blocks')}` blocks and `{status.get('warnings')}` warnings.",
        f"- Queue: {_fmt_queue(status.get('queue') or {})}.",
        f"- P0 status: `{status.get('p0_status')}`.",
        f"- Friction interval source audit: `{status.get('friction_interval_source_verdict')}` with `{status.get('friction_interval_source_anchors')}` anchors.",
        "",
        "## Source Map",
        "",
        "| Group | Source | Role | Project use | Claim boundary |",
        "|---|---|---|---|---|",
    ]
    for row in report["source_rows"]:
        lines.append(
            "| {group} | [{name}]({url}) | {role} | {use} | {boundary} |".format(
                group=row["group"],
                name=row["name"],
                url=row["url"],
                role=row["role"],
                use=row["use_in_project"],
                boundary=row["claim_boundary"],
            )
        )
    lines.extend(["", "## Reviewer Decision Rules", ""])
    lines.append("| Gate | Evidence | Action |")
    lines.append("|---|---|---|")
    for row in report["decision_rules"]:
        lines.append(f"| {row['gate']} | {row['evidence']} | {row['action']} |")
    lines.extend(["", "## Paper Claim Policy", ""])
    for item in report["paper_claim_policy"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _queue_summary(
    queue: dict[str, Any],
    watch: dict[str, Any] | None = None,
    active_live: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not queue:
        return {}
    active = queue.get("active_rows") or []
    active_row = active[0] if active else queue.get("next_incomplete") or {}
    watch_active = (watch or {}).get("active") or {}
    queue_time = _parse_generated_at(queue.get("generated_at"))
    watch_time = _parse_generated_at((watch or {}).get("generated_at"))
    use_watch = (
        bool(watch_active.get("name"))
        and watch_time is not None
        and (queue_time is None or watch_time >= queue_time)
    )
    if use_watch:
        active_row = {
            **active_row,
            "name": watch_active.get("name"),
            "active_epoch": watch_active.get("epoch"),
            "active_epochs": watch_active.get("epochs"),
            "active_step": watch_active.get("step"),
            "active_steps": watch_active.get("steps"),
        }
    live_active = (active_live or {}).get("active") or {}
    if live_active.get("name") and not use_watch:
        active_row = {
            **active_row,
            "name": live_active.get("name"),
            "active_epoch": live_active.get("epoch") or active_row.get("active_epoch") or active_row.get("epoch"),
            "active_epochs": live_active.get("epochs") or active_row.get("active_epochs") or active_row.get("epochs"),
            "active_step": live_active.get("step") or active_row.get("active_step"),
            "active_steps": live_active.get("steps") or active_row.get("active_steps"),
        }
    return {
        "total": queue.get("num_total"),
        "complete": queue.get("num_complete"),
        "running_or_partial": queue.get("num_partial"),
        "missing": queue.get("num_missing"),
        "active": active_row.get("name"),
        "active_epoch": active_row.get("active_epoch") or active_row.get("epoch"),
        "active_epochs": active_row.get("active_epochs") or active_row.get("epochs"),
        "active_step": active_row.get("active_step"),
        "active_steps": active_row.get("active_steps"),
    }


def _parse_generated_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _fmt_queue(queue: dict[str, Any]) -> str:
    if not queue:
        return "`-`"
    active = queue.get("active") or "-"
    epoch = queue.get("active_epoch")
    epochs = queue.get("active_epochs")
    step = queue.get("active_step")
    steps = queue.get("active_steps")
    progress = f"epoch {epoch}/{epochs}" if epoch is not None and epochs is not None else (f"epoch {epoch}" if epoch is not None else "-")
    if step is not None and steps is not None:
        progress = f"{progress}, step {step}/{steps}"
    return (
        f"`{queue.get('complete')}/{queue.get('total')}` complete, "
        f"`{queue.get('running_or_partial')}` running/partial, "
        f"`{queue.get('missing')}` missing; active `{active}` ({progress})"
    )


if __name__ == "__main__":
    main()

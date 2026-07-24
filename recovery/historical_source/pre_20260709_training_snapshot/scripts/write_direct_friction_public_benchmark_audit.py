from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


SOURCE_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "SIWNet",
        "url": "https://arxiv.org/abs/2310.00923",
        "role": "direct_image_to_friction_interval_reference",
        "data_public_for_current_protocol": "not_protocol_equivalent",
        "visual_input": "forward camera images",
        "target": "measured road-friction factor with prediction intervals",
        "public_split_target_match": "no",
        "measured_friction": "yes_in_paper_target",
        "local_dataset_match": "no",
        "numeric_use": "context_only",
        "action": "Borrow interval/calibration discipline; do not compare scores against RSCD/RoadSaW/RoadSC weak labels.",
    },
    {
        "name": "WCamNet",
        "url": "https://arxiv.org/abs/2404.16578",
        "code_url": "https://github.com/ojalar/wcamnet",
        "role": "direct_winter_road_friction_reference",
        "data_public_for_current_protocol": "not_protocol_equivalent",
        "visual_input": "roadside winter-road images",
        "target": "optical-sensor road-friction measurements",
        "public_split_target_match": "no",
        "measured_friction": "yes_in_paper_target",
        "local_dataset_match": "no",
        "numeric_use": "context_only",
        "action": "Borrow the foundation-feature plus local-texture idea; run only as a future measured-friction route if data/splits are reproduced.",
    },
    {
        "name": "RSCD",
        "url": "https://thu-rsxd.com/rscd/",
        "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9343931/",
        "role": "public_proxy_dataset",
        "data_public_for_current_protocol": "yes",
        "visual_input": "road-surface image patches",
        "target": "road-surface class labels and weak friction/material/condition proxies",
        "public_split_target_match": "local_splits_defined",
        "measured_friction": "no",
        "local_dataset_match": "yes",
        "numeric_use": "fair_if_same_split_label_metric",
        "action": "Use same-split FAF vs ConvNeXt and an RSCD-27 class-label protocol for contextual RSCD-style comparison.",
    },
    {
        "name": "RoadSaW",
        "url": "https://openaccess.thecvf.com/content/CVPR2022W/WAD/html/Cordes_RoadSaW_A_Large-Scale_Dataset_for_Camera-Based_Road_Surface_and_Wetness_CVPRW_2022_paper.html",
        "dataset_url": "https://viscoda.com/index.php/de/downloads-de/roadsaw-dataset-de",
        "role": "public_proxy_dataset",
        "data_public_for_current_protocol": "yes",
        "visual_input": "bird's-eye-view road patches",
        "target": "road surface and wetness labels",
        "public_split_target_match": "local_splits_defined",
        "measured_friction": "no",
        "local_dataset_match": "yes",
        "numeric_use": "fair_if_same_split_label_metric",
        "action": "Use as wet-road friction-affordance stress data; keep near-white wet/reflection cases instead of deleting them.",
    },
    {
        "name": "RoadSC",
        "url": "https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/html/Cordes_Camera-Based_Road_Snow_Coverage_Estimation_ICCVW_2023_paper.html",
        "dataset_url": "https://viscoda.com/index.php/en/downloads-en/roadsc-dataset",
        "role": "public_proxy_dataset",
        "data_public_for_current_protocol": "yes",
        "visual_input": "bird's-eye-view road patches",
        "target": "snow-coverage condition labels",
        "public_split_target_match": "local_splits_defined",
        "measured_friction": "no",
        "local_dataset_match": "yes",
        "numeric_use": "fair_if_same_split_label_metric",
        "action": "Use as winter low-friction proxy and separate same-dataset/LODO stress test.",
    },
    {
        "name": "Extreme Road Image Dataset",
        "url": "https://github.com/sean-shiyuez/Extreme-Road-Image-Dataset",
        "role": "separate_public_extreme_condition_route",
        "data_public_for_current_protocol": "yes_separate_route",
        "visual_input": "extreme-condition road images",
        "target": "extreme road-condition classes",
        "public_split_target_match": "local_splits_defined_for_separate_protocol",
        "measured_friction": "no",
        "local_dataset_match": "yes_separate_route",
        "numeric_use": "separate_same_task_only",
        "action": "Keep outside RSCD/RoadSaW/RoadSC tables; use only as a separate visual-affordance stress benchmark.",
    },
    {
        "name": "ROAD Camera-IMU",
        "url": "https://arxiv.org/abs/2601.20847",
        "role": "future_multimodal_dataset_candidate",
        "data_public_for_current_protocol": "not_audited",
        "visual_input": "RGB plus IMU road-surface data",
        "target": "road-surface classification",
        "public_split_target_match": "unknown",
        "measured_friction": "no",
        "local_dataset_match": "no",
        "numeric_use": "future_only",
        "action": "Do not merge now; audit files/license/splits later if a camera-plus-dynamics branch is opened.",
    },
]


DECISION_RULES: list[dict[str, str]] = [
    {
        "rule": "Same target",
        "keep": "The external method predicts the same label/target as the local experiment.",
        "discard": "Measured friction scores cannot be compared to weak proxy-label friction intervals.",
    },
    {
        "rule": "Same split",
        "keep": "Train/validation/test files or acquisition-day split are identical or reproduced exactly.",
        "discard": "Published numbers with unknown or different splits are context only.",
    },
    {
        "rule": "Same metric",
        "keep": "Metrics share class definitions, averaging, calibration target, and interval-width reporting.",
        "discard": "Regression MAE, condition classification accuracy, and weak-interval F1 are not interchangeable.",
    },
    {
        "rule": "Public reproducibility",
        "keep": "Images, labels, preprocessing, split files, and license can be audited locally.",
        "discard": "Private sensor-friction datasets cannot be the main numeric baseline.",
    },
]


ROUTE_DECISIONS: list[dict[str, str]] = [
    {
        "route": "Direct image-to-measured-friction benchmark",
        "decision": "discard_as_current_main_numeric_claim",
        "reason": "The most relevant direct-friction papers use measured sensor targets that are not present in RSCD/RoadSaW/RoadSC.",
        "next_action": "Keep SIWNet/WCamNet as method inspirations and future measured-friction routes only.",
    },
    {
        "route": "Naive pooled multi-dataset weak-label benchmark",
        "decision": "discard_as_main_claim",
        "reason": "Completed LODO and shortcut diagnostics show the model can exploit dataset style and fails severe held-out transfer.",
        "next_action": "Use pooled runs only for diagnosis and candidate screening.",
    },
    {
        "route": "Hierarchical public proxy benchmark",
        "decision": "keep_as_current_main_route",
        "reason": "It separates in-domain fairness from cross-dataset stress: RSCD, RoadSaW, and RoadSC each get matched FAF-vs-ConvNeXt rows.",
        "next_action": "Finish single-dataset FAF/baselines, paired bootstrap, and conditional coverage-width tables.",
    },
    {
        "route": "RSCD-27 class-label protocol",
        "decision": "keep_as_secondary_external_context",
        "reason": "RSCD is the only current source where an RSCD-style road-surface classification comparison may be made if labels/splits/metrics match.",
        "next_action": "Run the implemented RSCD-27 protocol and compare only under the documented local setup.",
    },
    {
        "route": "ExtremeRoad separate route",
        "decision": "keep_as_optional_stress_route",
        "reason": "It has reusable public images but different label semantics; it can strengthen robustness evidence only as a separate table.",
        "next_action": "Run same-task FAF and ConvNeXt rows after the current queue if GPU budget allows.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "direct_friction_public_benchmark_audit.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "direct_friction_public_benchmark_audit.json",
    )
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    external = _load_json(summary_dir / "external_benchmark_report.json") or {}
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    readiness = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}

    direct_rows = [row for row in SOURCE_REGISTRY if row["numeric_use"] == "context_only"]
    fair_proxy_rows = [
        row for row in SOURCE_REGISTRY if row["numeric_use"] == "fair_if_same_split_label_metric"
    ]
    future_rows = [
        row
        for row in SOURCE_REGISTRY
        if row["numeric_use"] in {"future_only", "separate_same_task_only"}
        or str(row.get("data_public_for_current_protocol", "")).startswith("yes_separate")
    ]

    final_verdict = "strict_proxy_route_required"
    if not fair_proxy_rows:
        final_verdict = "blocked_no_public_proxy_route"
    elif not direct_rows:
        final_verdict = "proxy_route_required_external_direct_sources_missing"

    missing_single_dataset: list[str] = []
    completion_status = completeness.get("status_by_run", {})
    if isinstance(completion_status, dict):
        required = [
            "single_rscd_full_faf",
            "single_roadsaw_full_faf",
            "single_roadsc_full_faf",
            "baseline_single_rscd_global_convnext",
            "baseline_single_roadsaw_global_convnext",
            "baseline_single_roadsc_global_convnext",
        ]
        missing_single_dataset = [
            name
            for name in required
            if str(completion_status.get(name, "")).lower() not in {"complete", "completed", "done"}
        ]

    return {
        "summary_dir": str(summary_dir),
        "verdict": final_verdict,
        "public_sources": SOURCE_REGISTRY,
        "decision_rules": DECISION_RULES,
        "route_decisions": ROUTE_DECISIONS,
        "counts": {
            "sources": len(SOURCE_REGISTRY),
            "direct_context_sources": len(direct_rows),
            "fair_proxy_sources": len(fair_proxy_rows),
            "future_or_separate_sources": len(future_rows),
            "external_benchmark_sources": len(external.get("public_sources", []) or []),
            "readiness_blocks": readiness.get("num_blocks"),
            "missing_single_dataset_rows": len(missing_single_dataset),
        },
        "required_missing_single_dataset_rows": missing_single_dataset,
        "reviewer_safe_claim": (
            "Current public data support visual friction-affordance interval estimation from "
            "road-condition proxy labels, not direct measured tire-road friction regression."
        ),
        "current_main_comparison": (
            "Matched same-split FAF versus ConvNeXt on RSCD, RoadSaW, and RoadSC, with LODO as "
            "stress evidence and SIWNet/WCamNet as context-only method references."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Direct Friction Public Benchmark Audit",
        "",
        f"Summary dir: `{report['summary_dir']}`",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        "Reviewer-safe claim: " + report["reviewer_safe_claim"],
        "",
        "Current main comparison: " + report["current_main_comparison"],
        "",
        "## Decision Rules",
        "",
        "| Rule | Keep | Discard |",
        "|---|---|---|",
    ]
    for row in report["decision_rules"]:
        lines.append(f"| {row['rule']} | {row['keep']} | {row['discard']} |")

    lines.extend(
        [
            "",
            "## Source Audit",
            "",
            "| Source | Role | Target | Measured friction | Local match | Numeric use | Action |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in report["public_sources"]:
        source = _source_link(row)
        lines.append(
            "| {source} | {role} | {target} | {measured} | {local} | {numeric} | {action} |".format(
                source=source,
                role=row["role"],
                target=row["target"],
                measured=row["measured_friction"],
                local=row["local_dataset_match"],
                numeric=row["numeric_use"],
                action=row["action"],
            )
        )

    lines.extend(
        [
            "",
            "## Route Decisions",
            "",
            "| Route | Decision | Reason | Next action |",
            "|---|---|---|---|",
        ]
    )
    for row in report["route_decisions"]:
        lines.append(
            f"| {row['route']} | `{row['decision']}` | {row['reason']} | {row['next_action']} |"
        )

    lines.extend(["", "## Missing Evidence For Main Route", ""])
    missing = report.get("required_missing_single_dataset_rows") or []
    if missing:
        lines.append(
            "The following matched single-dataset rows are still required before any strong numeric claim:"
        )
        lines.append("")
        for item in missing:
            lines.append(f"- `{item}`")
    else:
        lines.append("All required matched single-dataset rows are present.")

    lines.extend(["", "## Counts", "", "```json", json.dumps(report["counts"], indent=2), "```", ""])
    return "\n".join(lines)


def _source_link(row: dict[str, Any]) -> str:
    url = row.get("url")
    text = row["name"]
    if not url:
        return text
    return f"[{text}]({url})"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.ontology import (  # noqa: E402
    FRICTION_INTERVAL_BENCHMARKS,
    FRICTION_INTERVAL_REFERENCE_SOURCES,
    FRICTION_STATES,
    weak_mu_interval_from_state,
)


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "friction_interval_claim_matrix.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "friction_interval_claim_matrix.json"


MANIFESTS = [
    Path("data/manifests_full/rscd_prepared_train.csv"),
    Path("data/manifests_full/rscd_prepared_val.csv"),
    Path("data/manifests_full/rscd_prepared_test.csv"),
    Path("data/manifests_full/roadsaw_train.csv"),
    Path("data/manifests_full/roadsaw_val.csv"),
    Path("data/manifests_full/roadsaw_test.csv"),
    Path("data/manifests_full/roadsc_train.csv"),
    Path("data/manifests_full/roadsc_val.csv"),
    Path("data/manifests_full/roadsc_test.csv"),
]


DATASET_CLAIMS = {
    "rscd": {
        "view": "road-surface crop / narrow road patch",
        "measured_friction": "no",
        "fair_numeric_use": "same-split FAF-vs-ConvNeXt and secondary RSCD-27 class-label protocol",
        "unsafe_claim": "measured tire-road friction regression or left/right wheel-front camera claim",
    },
    "roadsaw": {
        "view": "prepared square road-surface and wetness patch",
        "measured_friction": "no",
        "fair_numeric_use": "same-split wetness/surface proxy comparison and held-out wet-road stress test",
        "unsafe_claim": "deleting near-white wet/reflection cases as corrupt without evidence",
    },
    "roadsc": {
        "view": "prepared square snow-coverage road patch",
        "measured_friction": "no",
        "fair_numeric_use": "same-split snow/winter proxy comparison and held-out winter stress test",
        "unsafe_claim": "direct measured ice/snow friction regression",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report(summary_dir: Path) -> dict[str, Any]:
    manifest_rows = _manifest_summary()
    interval_rows = _interval_rows()
    source_audit = _load_json(summary_dir / "friction_interval_source_audit.json") or {}
    direct_audit = _load_json(summary_dir / "direct_friction_public_benchmark_audit.json") or {}
    rscd_external = _load_json(summary_dir / "rscd_external_comparison_readiness.json") or {}
    fair_protocol = _load_json(summary_dir / "fair_comparison_protocol_audit.json") or {}

    blockers = []
    warnings = []
    if source_audit.get("verdict") != "pass":
        blockers.append("friction_interval_source_audit_not_pass")
    if not manifest_rows:
        blockers.append("manifest_interval_rows_missing")
    for row in manifest_rows:
        if row["rows"] <= 0:
            blockers.append(f"{row['dataset']}_has_no_manifest_rows")
        if row["missing_mu_rows"] > 0:
            blockers.append(f"{row['dataset']}_has_missing_mu_rows")
        if row["unique_intervals"] < 1:
            blockers.append(f"{row['dataset']}_has_no_interval_variety")
    if direct_audit.get("verdict") != "strict_proxy_route_required":
        warnings.append("direct_friction_benchmark_boundary_not_strict")
    if rscd_external.get("verdict") not in {"protocol_ready_results_pending", "results_available"}:
        warnings.append("rscd_27_external_context_not_ready")
    if fair_protocol.get("num_blocks"):
        blockers.append("fair_comparison_protocol_has_blocks")

    verdict = "pass" if not blockers else "block"
    if verdict == "pass" and warnings:
        verdict = "pass_with_warnings"

    return {
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": (
            "The supervised target is a conservative visual friction-affordance interval "
            "derived from public road-condition labels and public friction anchors. It is "
            "not a synchronized measured tire-road friction coefficient."
        ),
        "safe_main_claim": (
            "A method can be claimed better only under matched public proxy labels: "
            "same dataset, same split, same label ontology, same metrics, paired "
            "bootstrap where possible, and coverage-width reporting for intervals."
        ),
        "unsafe_claims": [
            "Do not compare RSCD/RoadSaW/RoadSC weak-interval F1 directly with measured-friction MAE/RMSE papers.",
            "Do not claim RSCD images are left-wheel or right-wheel front-view frames.",
            "Do not claim cross-dataset robustness until LODO and dataset-ID probes improve.",
            "Do not call wider intervals better unless coverage and width are reported together.",
        ],
        "dataset_claim_rows": _dataset_claim_rows(manifest_rows),
        "manifest_interval_rows": manifest_rows,
        "state_interval_rows": interval_rows,
        "source_rows": _source_rows(),
        "comparison_ladder": _comparison_ladder(),
        "required_evidence": _required_evidence(),
        "linked_reports": {
            "friction_interval_source_audit": source_audit.get("verdict"),
            "direct_friction_public_benchmark_audit": direct_audit.get("verdict"),
            "rscd_external_comparison_readiness": rscd_external.get("verdict"),
            "fair_comparison_protocol_audit": fair_protocol.get("verdict"),
        },
    }


def _manifest_summary() -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    interval_counts: dict[str, Counter[str]] = defaultdict(Counter)
    friction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    risk_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for path in MANIFESTS:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = str(row.get("dataset") or "unknown").lower()
                item = by_dataset.setdefault(
                    dataset,
                    {
                        "dataset": dataset,
                        "rows": 0,
                        "missing_mu_rows": 0,
                        "min_mu_low": None,
                        "max_mu_high": None,
                    },
                )
                item["rows"] += 1
                split_counts[dataset][str(row.get("split") or "unknown")] += 1
                label = str(row.get("class_label") or "unknown")
                class_counts[dataset][label] += 1
                friction_counts[dataset][str(row.get("friction_label") or "unknown")] += 1
                risk_counts[dataset][str(row.get("risk_label") or "unknown")] += 1
                low = _num(row.get("mu_low"))
                high = _num(row.get("mu_high"))
                if low is None or high is None:
                    item["missing_mu_rows"] += 1
                    continue
                interval_counts[dataset][f"{low:.2f}-{high:.2f}"] += 1
                item["min_mu_low"] = low if item["min_mu_low"] is None else min(item["min_mu_low"], low)
                item["max_mu_high"] = high if item["max_mu_high"] is None else max(item["max_mu_high"], high)

    rows = []
    for dataset, item in sorted(by_dataset.items()):
        rows.append(
            {
                **item,
                "splits": dict(split_counts[dataset]),
                "classes": len(class_counts[dataset]),
                "top_classes": _top(class_counts[dataset]),
                "friction_states": dict(sorted(friction_counts[dataset].items())),
                "risk_states": dict(sorted(risk_counts[dataset].items())),
                "unique_intervals": len(interval_counts[dataset]),
                "top_intervals": _top(interval_counts[dataset]),
            }
        )
    return rows


def _interval_rows() -> list[dict[str, Any]]:
    rows = []
    anchors_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in FRICTION_INTERVAL_BENCHMARKS:
        anchors_by_state[str(anchor["mapped_state"])].append(anchor)
    for state in FRICTION_STATES:
        low, high = weak_mu_interval_from_state(friction=state, wetness=state, snow=state)
        anchors = anchors_by_state.get(state, [])
        rows.append(
            {
                "state": state,
                "weak_interval": [low, high],
                "width": None if low is None or high is None else high - low,
                "source_anchor_count": len(anchors),
                "source_anchors": [
                    {
                        "anchor": item["anchor"],
                        "reference": [item["reference_low"], item["reference_high"]],
                        "source": item["source"],
                    }
                    for item in anchors
                ],
                "claim_strength": "anchored" if anchors else "ontology_only_from_related_state",
            }
        )
    return rows


def _dataset_claim_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_dataset = {row["dataset"]: row for row in manifest_rows}
    for dataset, claim in DATASET_CLAIMS.items():
        summary = by_dataset.get(dataset, {})
        rows.append(
            {
                "dataset": dataset,
                **claim,
                "rows": summary.get("rows", 0),
                "classes": summary.get("classes", 0),
                "unique_intervals": summary.get("unique_intervals", 0),
                "safe_claim": claim["fair_numeric_use"],
            }
        )
    return rows


def _source_rows() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "url": item.get("doi"),
            "role": item.get("role"),
            "use": "interval_anchor_or_claim_boundary",
        }
        for key, item in FRICTION_INTERVAL_REFERENCE_SOURCES.items()
    ]


def _comparison_ladder() -> list[dict[str, str]]:
    return [
        {
            "level": "L1 same-split proxy baseline",
            "allowed": "FAF vs ConvNeXt on the same RSCD/RoadSaW/RoadSC split and labels.",
            "not_allowed": "Using external measured-friction MAE as if it were the same target.",
            "priority": "primary",
        },
        {
            "level": "L2 RSCD-27 class-label context",
            "allowed": "RSCD original class-label protocol if labels, split policy, and metrics are documented.",
            "not_allowed": "Calling it direct friction estimation.",
            "priority": "secondary",
        },
        {
            "level": "L3 LODO stress",
            "allowed": "Held-out dataset reporting as a stress test and failure diagnosis.",
            "not_allowed": "Broad OOD superiority when held-out RoadSaW remains near zero.",
            "priority": "stress_evidence",
        },
        {
            "level": "L4 direct measured-friction papers",
            "allowed": "Context/method inspiration only unless public measured-friction data and splits are reproduced.",
            "not_allowed": "Direct numeric comparison with weak proxy-label intervals.",
            "priority": "context_only",
        },
    ]


def _required_evidence() -> list[dict[str, str]]:
    return [
        {
            "claim": "same-dataset method improvement",
            "required": "single_*_full_faf plus baseline_single_*_global_convnext, paired bootstrap, matching labels and split.",
        },
        {
            "claim": "interval quality",
            "required": "coverage, width, conditional cells by dataset/core-state/risk, and calibrated vs raw distinction.",
        },
        {
            "claim": "domain-generalization improvement",
            "required": "LODO metrics, dataset-ID probe reduction, worst-dataset F1, and low-friction recall stability.",
        },
        {
            "claim": "interpretable local visual evidence",
            "required": "Evidence maps, attention road-mass metrics, failure cases, and no metric regression.",
        },
        {
            "claim": "RSCD SOTA-style context",
            "required": "RSCD-27 class-label protocol result with documented split and metrics.",
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Friction Interval Claim Matrix",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        f"Safe main claim: {report['safe_main_claim']}",
        "",
        "## Dataset Claim Cards",
        "",
        "| Dataset | Rows | Classes | Intervals | View | Measured friction | Safe numeric use | Unsafe claim |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in report["dataset_claim_rows"]:
        lines.append(
            "| {dataset} | {rows} | {classes} | {intervals} | {view} | {measured} | {safe} | {unsafe} |".format(
                dataset=row["dataset"],
                rows=row["rows"],
                classes=row["classes"],
                intervals=row["unique_intervals"],
                view=row["view"],
                measured=row["measured_friction"],
                safe=row["safe_claim"],
                unsafe=row["unsafe_claim"],
            )
        )
    lines.extend(
        [
            "",
            "## Weak Friction Intervals",
            "",
            "| State | Weak interval | Width | Source anchors | Claim strength |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in report["state_interval_rows"]:
        low, high = row["weak_interval"]
        lines.append(
            f"| {row['state']} | {_fmt_interval(low, high)} | {_fmt_num(row['width'])} | {row['source_anchor_count']} | {row['claim_strength']} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Manifest Interval Audit",
            "",
            "| Dataset | Splits | Missing mu rows | Mu range | Top intervals | Top classes |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in report["manifest_interval_rows"]:
        lines.append(
            "| {dataset} | {splits} | {missing} | {mu_range} | {intervals} | {classes} |".format(
                dataset=row["dataset"],
                splits=_compact(row["splits"]),
                missing=row["missing_mu_rows"],
                mu_range=_fmt_interval(row["min_mu_low"], row["max_mu_high"]),
                intervals=_compact(row["top_intervals"]),
                classes=_compact(row["top_classes"]),
            )
        )

    lines.extend(
        [
            "",
            "## Comparison Ladder",
            "",
            "| Level | Allowed | Not allowed | Priority |",
            "|---|---|---|---|",
        ]
    )
    for row in report["comparison_ladder"]:
        lines.append(f"| {row['level']} | {row['allowed']} | {row['not_allowed']} | {row['priority']} |")

    lines.extend(["", "## Unsafe Claims", ""])
    for item in report["unsafe_claims"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Required Evidence", ""])
    for row in report["required_evidence"]:
        lines.append(f"- `{row['claim']}`: {row['required']}")

    lines.extend(["", "## Linked Reports", "", "```json", json.dumps(report["linked_reports"], indent=2), "```", ""])
    if report["blockers"] or report["warnings"]:
        lines.extend(["## Gate Notes", ""])
        for item in report["blockers"]:
            lines.append(f"- BLOCK: `{item}`")
        for item in report["warnings"]:
            lines.append(f"- WARN: `{item}`")
        lines.append("")
    return "\n".join(lines)


def _top(counter: Counter[str], n: int = 5) -> dict[str, int]:
    return dict(counter.most_common(n))


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_num(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _fmt_interval(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "-"
    return f"[{float(low):.2f}, {float(high):.2f}]"


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

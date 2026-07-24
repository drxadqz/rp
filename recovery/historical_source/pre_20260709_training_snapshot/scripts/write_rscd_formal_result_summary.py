from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT = Path("reports/paper_protocol_summary/rscd_formal_result_summary")

EXTERNAL = [
    {
        "method": "RoadFormer-B",
        "top1": 0.9252,
        "mean_precision": 0.8568,
        "mean_recall": 0.8334,
        "macro_f1": 0.8442,
        "strict_target": True,
        "note": "contextual original-RSCD 40-epoch result; exact split/preprocessing still needs audit",
    },
    {
        "method": "RoadMamba-B",
        "top1": 0.9281,
        "mean_precision": 0.8592,
        "mean_recall": 0.8373,
        "macro_f1": 0.8479,
        "strict_target": True,
        "note": "contextual original-RSCD 40-epoch result; reported with larger GPU/batch budget",
    },
    {
        "method": "RoadFormer-L",
        "top1": 0.9286,
        "mean_precision": 0.8617,
        "mean_recall": 0.8395,
        "macro_f1": 0.8499,
        "strict_target": True,
        "note": "verified original-RSCD 40-epoch row in RoadFormer Table II; strongest Top-1/Mean-F1 context target found so far",
    },
    {
        "method": "RSPNet-L",
        "top1": 0.9201,
        "mean_precision": None,
        "mean_recall": 0.8905,
        "macro_f1": 0.8949,
        "strict_target": False,
        "note": "GitHub README result; metric naming/protocol and RSCD-Expand relation need audit before strict ranking",
    },
]

STRICT_TOP1_TARGET = 0.9286
STRICT_F1_TARGET = 0.8499
STRICT_TARGET_NAME = "RoadFormer-L"


def main() -> None:
    local_rows = []
    for path in sorted(ROOT.glob("formal_*/evaluate_test.json")):
        payload = load_json(path)
        if not payload:
            continue
        summary = payload.get("summary", {})
        mean_precision, mean_recall = mean_pr_from_payload(payload)
        local_rows.append(
            {
                "method": path.parent.name,
                "top1": number(summary.get("top1")),
                "mean_precision": number(summary.get("mean_precision", mean_precision)),
                "mean_recall": number(summary.get("mean_recall", mean_recall)),
                "macro_f1": number(summary.get("macro_f1")),
                "balanced_accuracy": number(summary.get("balanced_accuracy")),
                "num_samples": int(summary.get("num_samples") or 0),
                "path": str(path),
            }
        )

    baseline = find(local_rows, "formal_convnext_tiny_b12e20_resume")
    physics = find(local_rows, "formal_physics_texture_quality_b12e20_parallel")
    for row in local_rows:
        row["delta_top1_vs_baseline"] = delta(row, baseline, "top1")
        row["delta_f1_vs_baseline"] = delta(row, baseline, "macro_f1")
        row["delta_top1_vs_physics"] = delta(row, physics, "top1")
        row["delta_f1_vs_physics"] = delta(row, physics, "macro_f1")
        row["gap_top1_to_strict_context"] = row["top1"] - STRICT_TOP1_TARGET
        row["gap_f1_to_strict_context"] = row["macro_f1"] - STRICT_F1_TARGET

    result = {
        "claim_boundary": (
            "Local rows are RSCD-27 class-label results from this machine. External rows are "
            "published or public RSCD-related results. Only rows marked strict_target=True are used as "
            "original-RSCD context targets here, and even those require matched splits/preprocessing/training "
            "before a strict SOTA claim."
        ),
        "strict_context_target": {
            "method": STRICT_TARGET_NAME,
            "top1": STRICT_TOP1_TARGET,
            "macro_f1": STRICT_F1_TARGET,
        },
        "external_sota": EXTERNAL,
        "local_rows": sorted(local_rows, key=lambda r: (r["macro_f1"], r["top1"]), reverse=True),
        "decision": decision(local_rows),
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(OUT.with_suffix(".md"))


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean_pr_from_payload(payload: dict[str, Any]) -> tuple[float, float]:
    report = payload.get("classification_report", {})
    precisions = []
    recalls = []
    for name, item in report.items():
        if not isinstance(item, dict) or name in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if "precision" in item and "recall" in item:
            precisions.append(number(item["precision"]))
            recalls.append(number(item["recall"]))
    if not precisions:
        return 0.0, 0.0
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def find(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row["method"] == name:
            return row
    return None


def delta(row: dict[str, Any], base: dict[str, Any] | None, key: str) -> float | None:
    if base is None:
        return None
    return row[key] - base[key]


def decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "waiting",
            "message": "No formal evaluate_test.json files yet.",
        }
    best = max(rows, key=lambda r: (r["macro_f1"], r["top1"]))
    if best["top1"] > STRICT_TOP1_TARGET and best["macro_f1"] > STRICT_F1_TARGET:
        status = "possible_sota_claim_requires_protocol_audit"
        message = f"Best local result exceeds {STRICT_TARGET_NAME} Top-1 and Mean-F1 context targets; verify exact protocol before claiming SOTA."
    elif best["top1"] > STRICT_TOP1_TARGET:
        status = "possible_top1_sota_requires_protocol_audit"
        message = f"Best local Top-1 exceeds {STRICT_TARGET_NAME} context target but Mean-F1 may not; verify protocol."
    else:
        status = "module_gain_claim_only"
        message = "Do not claim RSCD SOTA; report matched local module gains and hard-slice improvements."
    return {"status": status, "message": message, "best_method": best["method"]}


def pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# RSCD Formal Result Summary",
        "",
        result["claim_boundary"],
        "",
        "## External SOTA",
        "",
        "| method | Top-1 | Mean-P | Mean-R | Mean-F1 | strict target | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["external_sota"]:
        lines.append(
            f"| {row['method']} | {pct(row['top1'])} | {pct(row['mean_precision'])} | "
            f"{pct(row['mean_recall'])} | {pct(row['macro_f1'])} | "
            f"{'yes' if row.get('strict_target') else 'no'} | {row['note']} |"
        )
    target = result["strict_context_target"]
    lines.extend(
        [
            "",
            "## Local Formal Results",
            "",
            f"Strict contextual target used for gaps: {target['method']} Top-1 {pct(target['top1'])}, Mean-F1 {pct(target['macro_f1'])}.",
            "",
            "| method | Top-1 | Mean-P | Mean-R | Mean-F1 | samples | dTop1 vs baseline | dF1 vs baseline | gap Top-1 to strict context | gap F1 to strict context |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["local_rows"]:
        lines.append(
            "| `{method}` | {top1} | {mp} | {mr} | {mf1} | {samples} | {dt} | {df} | {gt} | {gf} |".format(
                method=row["method"],
                top1=pct(row["top1"]),
                mp=pct(row["mean_precision"]),
                mr=pct(row["mean_recall"]),
                mf1=pct(row["macro_f1"]),
                samples=row["num_samples"],
                dt=pct(row.get("delta_top1_vs_baseline"), signed=True),
                df=pct(row.get("delta_f1_vs_baseline"), signed=True),
                gt=pct(row.get("gap_top1_to_strict_context"), signed=True),
                gf=pct(row.get("gap_f1_to_strict_context"), signed=True),
            )
        )
    if not result["local_rows"]:
        lines.append("| waiting | - | - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Status: `{result['decision']['status']}`",
            f"- Message: {result['decision']['message']}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

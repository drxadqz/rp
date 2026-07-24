from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUMMARY = Path("reports/paper_protocol_summary")
OUT_JSON = SUMMARY / "rscd_external_sota_gap.json"
OUT_MD = SUMMARY / "rscd_external_sota_gap.md"

EXTERNAL = [
    {
        "method": "RoadFormer-L",
        "top1": 0.9286,
        "mean_precision": 0.8617,
        "mean_recall": 0.8395,
        "macro_f1": 0.8499,
        "protocol": "RSCD, published 40-epoch local-global transformer/CNN result",
    },
    {
        "method": "RoadMamba-B",
        "top1": 0.9281,
        "mean_precision": 0.8592,
        "mean_recall": 0.8373,
        "macro_f1": 0.8479,
        "protocol": "RSCD, 40 epochs, single RTX 4090, dual local/global visual state-space model",
    },
    {
        "method": "RSPNet-L",
        "top1": 0.9201,
        "mean_precision": None,
        "mean_recall": 0.8905,
        "macro_f1": 0.8949,
        "protocol": (
            "RSCD, 2026 lightweight road-surface perception repo; reports Top-1/Top-5/Recall/F1 "
            "with 3.69M params and wavelet/frequency-decoupled texture preservation. Treat as an "
            "efficiency/F1 reference until split and F1 definition are fully protocol-matched."
        ),
    },
]


def main() -> None:
    trend = _load_json(SUMMARY / "rscd_training_trend_report.json") or {}
    formal = _load_json(SUMMARY / "rscd_formal_result_summary.json") or {}
    local_rows = _local_rows(trend, formal)
    report = {
        "claim_boundary": (
            "External RSCD SOTA gap analysis. Published numbers are contextual "
            "unless label mapping, split, preprocessing, training budget, and metrics match."
        ),
        "external_sota": EXTERNAL,
        "local_rows": local_rows,
        "gaps": _gaps(local_rows),
        "protocol_gap": {
            "external": (
                "RoadFormer/RoadMamba report RSCD results under 40-epoch settings; RoadMamba states single "
                "RTX 4090, batch size 32. RSPNet-L reports a 2026 lightweight RSCD table with Top-1 92.01 "
                "and F1 89.49, but its split/F1 definition must be checked before strict numeric ranking."
            ),
            "local": "Current formal local runs use RTX 3050 Laptop GPU, 20 epochs, image size 192, batch size 12 with gradient accumulation, and sampled balanced epochs.",
            "decision": "Use external values as SOTA targets, not strict numeric claims, until a matched protocol run or faithful reimplementation exists.",
        },
        "sources": [
            "https://figshare.com/articles/dataset/Road_Surface_Image_Dataset_with_Detailed_Annotations_for_Driving_Assistance/20424582",
            "https://arxiv.org/html/2506.02358v1",
            "https://arxiv.org/html/2508.01210v1",
            "https://github.com/YLong-maker/RSPNet",
        ],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    print(OUT_MD)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _local_rows(trend: dict[str, Any], formal: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in formal.get("local_rows", []) or []:
        rows.append(
            {
                "name": row.get("method") or row.get("name"),
                "source": "formal_test",
                "top1": row.get("top1"),
                "mean_precision": row.get("mean_precision"),
                "mean_recall": row.get("mean_recall"),
                "macro_f1": row.get("macro_f1"),
            }
        )
    if rows:
        return rows
    for row in trend.get("runs", []) or []:
        best = row.get("best") or {}
        rows.append(
            {
                "name": row.get("name"),
                "source": "formal_validation_best",
                "top1": best.get("top1"),
                "mean_precision": None,
                "mean_recall": None,
                "macro_f1": best.get("macro_f1"),
            }
        )
    return rows


def _gaps(local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for local in local_rows:
        for ext in EXTERNAL:
            out.append(
                {
                    "local": local.get("name"),
                    "local_source": local.get("source"),
                    "external": ext["method"],
                    "top1_gap_local_minus_external": _sub(local.get("top1"), ext.get("top1")),
                    "macro_f1_gap_local_minus_external": _sub(local.get("macro_f1"), ext.get("macro_f1")),
                }
            )
    return out


def _sub(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RSCD External SOTA Gap",
        "",
        report["claim_boundary"],
        "",
        "## External Targets",
        "",
        "| method | Top-1 | Mean-P | Mean-R | Mean-F1 | protocol |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["external_sota"]:
        lines.append(
            "| {method} | {top1} | {mp} | {mr} | {mf1} | {protocol} |".format(
                method=row["method"],
                top1=_pct(row.get("top1")),
                mp=_pct(row.get("mean_precision")),
                mr=_pct(row.get("mean_recall")),
                mf1=_pct(row.get("macro_f1")),
                protocol=row["protocol"],
            )
        )
    lines += [
        "",
        "## Local Rows",
        "",
        "| run | source | Top-1 | Macro-F1 |",
        "|---|---|---:|---:|",
    ]
    for row in report["local_rows"]:
        lines.append(
            f"| `{row.get('name')}` | {row.get('source')} | {_pct(row.get('top1'))} | {_pct(row.get('macro_f1'))} |"
        )
    lines += [
        "",
        "## Gaps",
        "",
        "| local | source | external | dTop-1 | dMacro-F1 |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["gaps"]:
        lines.append(
            "| `{local}` | {source} | {external} | {dtop} | {df1} |".format(
                local=row.get("local"),
                source=row.get("local_source"),
                external=row.get("external"),
                dtop=_pct(row.get("top1_gap_local_minus_external"), signed=True),
                df1=_pct(row.get("macro_f1_gap_local_minus_external"), signed=True),
            )
        )
    gap = report["protocol_gap"]
    lines += [
        "",
        "## Protocol Gap",
        "",
        f"- External: {gap['external']}",
        f"- Local: {gap['local']}",
        f"- Decision: {gap['decision']}",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- {src}" for src in report["sources"])
    lines.append("")
    return "\n".join(lines)


def _pct(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    val = float(value)
    sign = "+" if signed and val >= 0 else ""
    return f"{sign}{val * 100:.2f}%"


if __name__ == "__main__":
    main()

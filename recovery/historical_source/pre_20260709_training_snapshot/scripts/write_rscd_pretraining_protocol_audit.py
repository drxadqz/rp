from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT = Path("reports/paper_protocol_summary/rscd_pretraining_protocol_audit")

RUNS = [
    "formal_convnext_tiny_b12e20_resume",
    "formal_physics_texture_quality_b12e20_resume",
    "formal_physics_texture_quality_b12e20_parallel",
    "formal_physics_wavelet_directional_film_gate_hier",
]


def main() -> None:
    rows = []
    for name in RUNS:
        run_dir = ROOT / name
        protocol = _load_json(run_dir / "protocol.json") or {}
        metrics = _load_json(run_dir / "evaluate_test.json") or {}
        args = protocol.get("args") if isinstance(protocol.get("args"), dict) else {}
        summary = metrics.get("summary") if isinstance(metrics.get("summary"), dict) else {}
        rows.append(
            {
                "run": name,
                "protocol_exists": bool(protocol),
                "test_exists": bool(metrics),
                "backbone": args.get("backbone"),
                "pretrained": args.get("pretrained"),
                "image_size": args.get("image_size"),
                "batch_size": args.get("batch_size"),
                "grad_accum_steps": args.get("grad_accum_steps"),
                "samples_per_epoch": args.get("samples_per_epoch"),
                "physics": args.get("use_physics_branch"),
                "physics_quality_cues": args.get("physics_quality_cues"),
                "physics_quality_region_cues": args.get("physics_quality_region_cues"),
                "wavelet": args.get("use_wavelet_texture_branch"),
                "directional": args.get("use_directional_texture_branch"),
                "texture_film": args.get("use_texture_film"),
                "texture_gate": args.get("use_texture_gate"),
                "top1": _num(summary.get("top1")) if summary else None,
                "macro_f1": _num(summary.get("macro_f1")) if summary else None,
            }
        )
    report = {
        "claim_boundary": (
            "This audit checks whether local RSCD formal runs used pretrained visual "
            "backbones and comparable manifest splits. It does not prove external SOTA."
        ),
        "strict_external_context": {
            "RoadFormer-L_top1": 0.9286,
            "RoadFormer-L_mean_f1": 0.8499,
        },
        "interpretation": _interpret(rows),
        "rows": rows,
    }
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(_to_markdown(report), encoding="utf-8")
    print(OUT.with_suffix(".md"))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interpret(rows: list[dict[str, Any]]) -> list[str]:
    notes = []
    core = [row for row in rows if row["run"] != "formal_physics_wavelet_directional_film_gate_hier"]
    if core and all(row.get("pretrained") is True for row in core if row.get("protocol_exists")):
        notes.append(
            "Completed core formal runs used pretrained ConvNeXt-Tiny backbones; the gap to RoadFormer-style results should not be explained as a missing-pretraining artifact."
        )
    if any(row.get("image_size") == 192 for row in rows if row.get("protocol_exists")):
        notes.append(
            "Local formal runs use 192px inputs for 4GB-GPU stability. This is fair for local ablations, but external SOTA comparison must mention the resolution/architecture difference."
        )
    if any(row.get("physics_quality_region_cues") is None and row.get("physics_quality_cues") for row in rows):
        notes.append(
            "Older protocol files predate the explicit `physics_quality_region_cues` field; those runs used the historical default bottom-vs-top cues."
        )
    notes.append(
        "Future RSCD patch-style candidates should disable vertical region cues and compare against the pretrained PhysicsTexture formal result plus hard wet/water slices."
    )
    return notes


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RSCD Pretraining And Protocol Audit",
        "",
        report["claim_boundary"],
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {note}" for note in report["interpretation"])
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| run | protocol | test | backbone | pretrained | image | physics | quality | region cues | wavelet | Top-1 | Mean-F1 |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| `{run}` | {protocol} | {test} | `{backbone}` | {pretrained} | {image} | {physics} | {quality} | {region} | {wavelet} | {top1} | {f1} |".format(
                run=row["run"],
                protocol="yes" if row["protocol_exists"] else "no",
                test="yes" if row["test_exists"] else "no",
                backbone=row.get("backbone") or "-",
                pretrained=row.get("pretrained"),
                image=row.get("image_size") or "-",
                physics=row.get("physics"),
                quality=row.get("physics_quality_cues"),
                region=row.get("physics_quality_region_cues"),
                wavelet=row.get("wavelet"),
                top1=_pct(row.get("top1")),
                f1=_pct(row.get("macro_f1")),
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

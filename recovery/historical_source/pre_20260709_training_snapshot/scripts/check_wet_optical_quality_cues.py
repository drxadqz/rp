from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from friction_affordance.models.texture import PhysicsTextureBranch


DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/wet_optical_quality_cues_smoke.md")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/wet_optical_quality_cues_smoke.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test PhysicsTexture wet optical quality cues on CPU."
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
    torch.manual_seed(79)
    image = torch.randn(3, 3, 96, 96)
    base = PhysicsTextureBranch(out_dim=24, quality_cues=False).cpu().eval()
    quality = PhysicsTextureBranch(out_dim=24, quality_cues=True).cpu().eval()
    with torch.no_grad():
        base_out = base(image)
        quality_out = quality(image)
    checks = [
        {
            "name": "base_shape",
            "pass": list(base_out.shape) == [3, 24],
            "shape": list(base_out.shape),
            "num_stats": base.num_stats,
        },
        {
            "name": "quality_shape",
            "pass": list(quality_out.shape) == [3, 24],
            "shape": list(quality_out.shape),
            "num_stats": quality.num_stats,
        },
        {
            "name": "quality_stats_expanded",
            "pass": quality.num_stats > base.num_stats,
            "base_num_stats": base.num_stats,
            "quality_num_stats": quality.num_stats,
        },
        {
            "name": "finite_outputs",
            "pass": bool(torch.isfinite(base_out).all() and torch.isfinite(quality_out).all()),
            "base_abs_mean": float(base_out.abs().mean()),
            "quality_abs_mean": float(quality_out.abs().mean()),
        },
    ]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ok" if all(row["pass"] for row in checks) else "fail",
        "claim_boundary": (
            "This smoke test proves the wet optical quality cues are wired and finite. "
            "It does not prove metric improvement."
        ),
        "checks": checks,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Wet Optical Quality Cues Smoke",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Status: `{report['status']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "| Check | Pass | Details |",
        "|---|---:|---|",
    ]
    for row in report["checks"]:
        details = {key: val for key, val in row.items() if key not in {"name", "pass"}}
        lines.append(f"| {row['name']} | {row['pass']} | {details} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

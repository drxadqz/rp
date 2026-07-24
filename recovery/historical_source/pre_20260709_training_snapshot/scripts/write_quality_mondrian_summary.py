from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize post-hoc quality/Mondrian conformal interval probes."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "quality_mondrian_summary.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "quality_mondrian_summary.json")
    args = parser.parse_args()

    report = build_report(args.root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.glob("*/quality_mondrian_*/*quality_mondrian_conformal.json")):
        payload = _load_json(path)
        if isinstance(payload, dict):
            rows.append(_row(path, payload))
    rows = sorted(rows, key=lambda row: (row.get("run", ""), row.get("probe", "")))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "verdict": _verdict(rows),
        "claim_boundary": (
            "Quality-Mondrian conformal probes are post-hoc interval calibration evidence. "
            "They do not prove measured tire-road friction coefficients or better visual recognition."
        ),
        "rows": rows,
        "decisions": _decisions(rows),
    }


def _row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    run = path.parent.parent.name
    probe = path.parent.name
    policies = payload.get("policies", {}) if isinstance(payload.get("policies"), dict) else {}
    global_row = _policy_values(policies.get("global"))
    hier = _policy_values(policies.get("hierarchical_safety"))
    quality = _policy_values(policies.get("hierarchical_quality_safety"))
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    checkpoint = str(payload.get("checkpoint", ""))
    checkpoint_name = Path(checkpoint).name if checkpoint else ""
    row = {
        "run": run,
        "probe": probe,
        "path": str(path),
        "checkpoint": checkpoint,
        "checkpoint_name": checkpoint_name,
        "target_coverage": payload.get("target_coverage"),
        "quality_join_rate": (payload.get("test") or {}).get("quality_join_rate")
        if isinstance(payload.get("test"), dict)
        else None,
        "calibration_mu_samples": (payload.get("calibration") or {}).get("num_mu_samples")
        if isinstance(payload.get("calibration"), dict)
        else None,
        "test_mu_samples": (payload.get("test") or {}).get("num_mu_samples")
        if isinstance(payload.get("test"), dict)
        else None,
        "global": global_row,
        "hierarchical_safety": hier,
        "hierarchical_quality_safety": quality,
        "delta_quality_vs_global_coverage": _delta(
            quality.get("calibrated_coverage"), global_row.get("calibrated_coverage")
        ),
        "delta_quality_vs_hierarchical_coverage": _delta(
            _worst_cov(quality), _worst_cov(hier)
        ),
        "delta_quality_vs_hierarchical_width": _delta(
            quality.get("calibrated_width"), hier.get("calibrated_width")
        ),
        "script_decision": decision.get("status"),
        "summary_decision": None,
        "reason": None,
    }
    row["summary_decision"], row["reason"] = _summary_decision(row)
    return row


def _policy_values(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    pooled = policy.get("pooled", {}) if isinstance(policy.get("pooled"), dict) else {}
    worst = policy.get("worst_slice", {}) if isinstance(policy.get("worst_slice"), dict) else {}
    return {
        "raw_coverage": pooled.get("raw_coverage"),
        "calibrated_coverage": pooled.get("calibrated_coverage"),
        "calibrated_width": pooled.get("calibrated_width"),
        "mean_radius": pooled.get("mean_radius") or (policy.get("radius") or {}).get("mean_radius")
        if isinstance(policy.get("radius"), dict)
        else pooled.get("mean_radius"),
        "worst_scope": worst.get("scope"),
        "worst_name": worst.get("name"),
        "worst_coverage": worst.get("calibrated_coverage"),
        "worst_width": worst.get("calibrated_width"),
    }


def _summary_decision(row: dict[str, Any]) -> tuple[str, str]:
    run = str(row.get("run", ""))
    probe = str(row.get("probe", ""))
    checkpoint = str(row.get("checkpoint_name", ""))
    quality = row.get("hierarchical_quality_safety", {})
    hier = row.get("hierarchical_safety", {})
    global_row = row.get("global", {})
    quality_cov = _num(quality.get("calibrated_coverage"))
    global_cov = _num(global_row.get("calibrated_coverage"))
    quality_worst = _num(quality.get("worst_coverage"))
    hier_worst = _num(hier.get("worst_coverage"))
    width_delta = _num(row.get("delta_quality_vs_hierarchical_width"))

    if "roadsc" in run and checkpoint == "best_safety.pt":
        return "prefer_safety_checkpoint", "safety checkpoint already has high raw/test coverage; no post-hoc gain needed"
    if "roadsc" in run and checkpoint == "best.pt":
        return "discard_best_checkpoint_rescue", "best.pt remains weak on worst winter slice after quality-Mondrian calibration"
    if quality_cov is not None and global_cov is not None and quality_cov - global_cov >= 0.03:
        if width_delta is None or width_delta <= 0.08:
            return "keep_for_interval_calibration", "coverage gain over global with bounded width cost"
    if quality_worst is not None and hier_worst is not None and quality_worst > hier_worst and (width_delta or 0.0) <= 0.08:
        return "keep_as_small_hierarchical_addon", "small worst-slice gain over hierarchical calibration"
    if "fast" in probe or "smoke" in probe:
        return "screening_only", "diagnostic probe, not final evidence"
    return "discard_or_hold", "no material coverage or worst-slice gain"


def _verdict(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    if any(row.get("summary_decision") == "keep_for_interval_calibration" for row in rows):
        return "posthoc_interval_route_has_support"
    if any("keep" in str(row.get("summary_decision")) for row in rows):
        return "posthoc_interval_route_partial_support"
    return "posthoc_route_not_supported"


def _decisions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(str(row.get("summary_decision")), []).append(f"{row.get('run')}::{row.get('probe')}")
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Quality-Mondrian Summary",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        report["claim_boundary"],
        "",
        "| run | probe | checkpoint | q join | raw | global cov/w | hier cov/w | hq cov/w | worst hq | decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    if not report["rows"]:
        lines.append("| - | - | - | - | - | - | - | - | - | missing |")
    for row in report["rows"]:
        global_row = row["global"]
        hier = row["hierarchical_safety"]
        quality = row["hierarchical_quality_safety"]
        lines.append(
            "| {run} | {probe} | {ckpt} | {join} | {raw} | {gcov}/{gw} | {hcov}/{hw} | {qcov}/{qw} | {worst} | `{decision}` |".format(
                run=row.get("run"),
                probe=row.get("probe"),
                ckpt=row.get("checkpoint_name") or "-",
                join=_fmt_pct(row.get("quality_join_rate")),
                raw=_fmt_pct(global_row.get("raw_coverage")),
                gcov=_fmt_pct(global_row.get("calibrated_coverage")),
                gw=_fmt_abs(global_row.get("calibrated_width")),
                hcov=_fmt_pct(hier.get("calibrated_coverage")),
                hw=_fmt_abs(hier.get("calibrated_width")),
                qcov=_fmt_pct(quality.get("calibrated_coverage")),
                qw=_fmt_abs(quality.get("calibrated_width")),
                worst=_worst_text(quality),
                decision=row.get("summary_decision"),
            )
        )
    lines.extend(["", "## Decision Notes", ""])
    for row in report["rows"]:
        lines.append(f"- `{row.get('run')}::{row.get('probe')}`: {row.get('reason')}")
    lines.append("")
    return "\n".join(lines)


def _worst_text(row: dict[str, Any]) -> str:
    if not row:
        return "-"
    name = row.get("worst_name") or "-"
    return f"{name} {_fmt_pct(row.get('worst_coverage'))}"


def _worst_cov(row: dict[str, Any]) -> Any:
    return row.get("worst_coverage") if isinstance(row, dict) else None


def _delta(cur: Any, prev: Any) -> float | None:
    cur_num = _num(cur)
    prev_num = _num(prev)
    if cur_num is None or prev_num is None:
        return None
    return cur_num - prev_num


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: Any) -> str:
    num = _num(value)
    if num is None:
        return "-"
    return f"{100.0 * num:.2f}%"


def _fmt_abs(value: Any) -> str:
    num = _num(value)
    if num is None:
        return "-"
    return f"{num:.4f}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

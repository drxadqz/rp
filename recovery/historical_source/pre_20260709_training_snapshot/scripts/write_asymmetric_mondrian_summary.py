from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize asymmetric Mondrian conformal probes.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "asymmetric_mondrian_summary.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "asymmetric_mondrian_summary.json")
    args = parser.parse_args()

    report = build_report(args.root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.glob("*/asymmetric_mondrian_*/*asymmetric_mondrian_conformal.json")):
        payload = _load_json(path)
        if isinstance(payload, dict):
            rows.append(_row(path, payload))
    rows.sort(key=lambda row: (row.get("run", ""), row.get("probe", "")))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "verdict": _verdict(rows),
        "claim_boundary": (
            "Asymmetric Mondrian conformal probes are post-hoc interval calibration evidence. "
            "They do not prove measured tire-road friction coefficients or better visual recognition."
        ),
        "rows": rows,
        "decisions": _decisions(rows),
    }


def _row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    policies = payload.get("policies", {}) if isinstance(payload.get("policies"), dict) else {}
    baseline = _policy_values(policies.get("symmetric_hierarchical_quality"))
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    best = decision.get("best", {}) if isinstance(decision.get("best"), dict) else {}
    best_name = str(best.get("name") or "")
    best_policy = _policy_values(policies.get(best_name)) if best_name else {}
    row = {
        "run": path.parent.parent.name,
        "probe": path.parent.name,
        "path": str(path),
        "checkpoint": str(payload.get("checkpoint", "")),
        "checkpoint_name": Path(str(payload.get("checkpoint", ""))).name if payload.get("checkpoint") else "",
        "target_coverage": payload.get("target_coverage"),
        "baseline": baseline,
        "best_policy_name": best_name,
        "best_policy": best_policy,
        "script_status": decision.get("status"),
        "script_reason": decision.get("reason"),
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
        "calibrated_coverage": pooled.get("calibrated_coverage"),
        "calibrated_width": pooled.get("calibrated_width"),
        "raw_coverage": pooled.get("raw_coverage"),
        "worst_scope": worst.get("scope"),
        "worst_name": worst.get("name"),
        "worst_coverage": worst.get("calibrated_coverage"),
        "worst_width": worst.get("calibrated_width"),
    }


def _summary_decision(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("script_status") != "keep_for_more_eval":
        return "discard_or_hold", "script gate did not find a clear coverage-width improvement"
    baseline = row.get("baseline", {})
    best = row.get("best_policy", {})
    width_delta = _delta(best.get("calibrated_width"), baseline.get("calibrated_width"))
    cov_delta = _delta(best.get("calibrated_coverage"), baseline.get("calibrated_coverage"))
    worst_delta = _delta(best.get("worst_coverage"), baseline.get("worst_coverage"))
    if (
        _num(best.get("calibrated_coverage")) >= 0.90
        and width_delta is not None
        and width_delta <= -0.01
        and (worst_delta is None or worst_delta >= -0.01)
    ):
        return "keep_for_interval_width_reduction", (
            f"coverage delta {cov_delta:+.4f}, width delta {width_delta:+.4f}, "
            f"worst-slice delta {(worst_delta or 0.0):+.4f}"
        )
    return "discard_or_hold", "improvement is too small or worsens worst-slice coverage"


def _verdict(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    if any(row.get("summary_decision") == "keep_for_interval_width_reduction" for row in rows):
        return "asymmetric_posthoc_route_has_support"
    return "no_clear_asymmetric_gain"


def _decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run": row.get("run"),
            "probe": row.get("probe"),
            "decision": row.get("summary_decision"),
            "reason": row.get("reason"),
        }
        for row in rows
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Asymmetric Mondrian Summary",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        report["claim_boundary"],
        "",
        "| run | probe | checkpoint | baseline cov/w | best policy | best cov/w | worst best | decision |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    if not report["rows"]:
        lines.append("| - | - | - | - | - | - | - | `missing` |")
    for row in report["rows"]:
        base = row.get("baseline", {})
        best = row.get("best_policy", {})
        lines.append(
            "| {run} | {probe} | {ckpt} | {base} | {policy} | {best} | {worst} | `{decision}` |".format(
                run=row.get("run"),
                probe=row.get("probe"),
                ckpt=row.get("checkpoint_name") or "-",
                base=_cov_width(base),
                policy=row.get("best_policy_name") or "-",
                best=_cov_width(best),
                worst=_worst(best),
                decision=row.get("summary_decision"),
            )
        )
    if report.get("decisions"):
        lines.extend(["", "## Decision Notes", ""])
        for row in report["decisions"]:
            lines.append(
                f"- `{row.get('run')}::{row.get('probe')}`: `{row.get('decision')}`; {row.get('reason')}"
            )
    return "\n".join(lines) + "\n"


def _cov_width(row: dict[str, Any]) -> str:
    if not row:
        return "-"
    return f"{_fmt_pct(row.get('calibrated_coverage'))}/{_fmt_abs(row.get('calibrated_width'))}"


def _worst(row: dict[str, Any]) -> str:
    if not row:
        return "-"
    return f"{row.get('worst_scope')}::{row.get('worst_name')} {_fmt_pct(row.get('worst_coverage'))}"


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
    number = _num(value)
    if number is None:
        return "-"
    return f"{100.0 * number:.2f}%"


def _fmt_abs(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "-"
    return f"{number:.4f}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()

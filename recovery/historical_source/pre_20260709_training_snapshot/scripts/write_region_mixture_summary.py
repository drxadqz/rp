from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize segmentation-style region-mixture conformal probes.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "region_mixture_summary.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "region_mixture_summary.json")
    args = parser.parse_args()

    report = build_report(args.root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.glob("*/region_mixture_*/*region_mixture_conformal.json")):
        payload = _load_json(path)
        if isinstance(payload, dict):
            rows.append(_row(path, payload))
    rows.sort(key=lambda row: (row.get("run", ""), row.get("probe", "")))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "verdict": _verdict(rows),
        "claim_boundary": (
            "Region-mixture probes are segmentation-style post-hoc interval calibration evidence. "
            "They use unsupervised visual regions from existing images and do not add measured friction labels."
        ),
        "rows": rows,
        "decisions": _decisions(rows),
    }


def _row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    policies = payload.get("policies", {}) if isinstance(payload.get("policies"), dict) else {}
    global_row = _policy_values(policies.get("global"))
    best_name, best_policy = _best_policy(policies)
    best = _policy_values(best_policy)
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    row = {
        "run": path.parent.parent.name,
        "probe": path.parent.name,
        "path": str(path),
        "checkpoint": str(payload.get("checkpoint", "")),
        "checkpoint_name": Path(str(payload.get("checkpoint", ""))).name if payload.get("checkpoint") else "",
        "target_coverage": payload.get("target_coverage"),
        "clusters": payload.get("clusters"),
        "global": global_row,
        "best_policy_name": best_name,
        "best_policy": best,
        "script_status": (decision or {}).get("status"),
        "script_reason": (decision or {}).get("reason"),
        "summary_decision": None,
        "reason": None,
    }
    row["summary_decision"], row["reason"] = _summary_decision(row)
    return row


def _best_policy(policies: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = []
    for name, policy in policies.items():
        if name == "global" or not isinstance(policy, dict):
            continue
        pooled = policy.get("pooled", {}) if isinstance(policy.get("pooled"), dict) else {}
        worst = policy.get("worst_slice", {}) if isinstance(policy.get("worst_slice"), dict) else {}
        candidates.append(
            (
                name,
                policy,
                _num(worst.get("calibrated_coverage"), -1.0),
                _num(pooled.get("calibrated_coverage"), -1.0),
                -_num(pooled.get("calibrated_width"), 9.0),
            )
        )
    if not candidates:
        return "", {}
    candidates.sort(key=lambda item: (item[2], item[3], item[4]), reverse=True)
    return candidates[0][0], candidates[0][1]


def _policy_values(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    pooled = policy.get("pooled", {}) if isinstance(policy.get("pooled"), dict) else {}
    worst = policy.get("worst_slice", {}) if isinstance(policy.get("worst_slice"), dict) else {}
    return {
        "raw_coverage": pooled.get("raw_coverage"),
        "calibrated_coverage": pooled.get("calibrated_coverage"),
        "calibrated_width": pooled.get("calibrated_width"),
        "mean_radius": pooled.get("mean_radius"),
        "worst_scope": worst.get("scope"),
        "worst_name": worst.get("name"),
        "worst_coverage": worst.get("calibrated_coverage"),
        "worst_width": worst.get("calibrated_width"),
    }


def _summary_decision(row: dict[str, Any]) -> tuple[str, str]:
    global_row = row.get("global", {})
    best = row.get("best_policy", {})
    cov_delta = _delta(best.get("calibrated_coverage"), global_row.get("calibrated_coverage"))
    width_delta = _delta(best.get("calibrated_width"), global_row.get("calibrated_width"))
    worst_delta = _delta(best.get("worst_coverage"), global_row.get("worst_coverage"))
    if (
        _num(best.get("calibrated_coverage"), 0.0) >= 0.90
        and cov_delta is not None
        and cov_delta >= 0.02
        and worst_delta is not None
        and worst_delta >= 0.02
        and width_delta is not None
        and width_delta <= 0.04
    ):
        return "keep_for_segmentation_style_interval_calibration", (
            f"coverage delta {cov_delta:+.4f}, width delta {width_delta:+.4f}, "
            f"worst-slice delta {worst_delta:+.4f}"
        )
    if row.get("script_status") == "keep_for_more_eval":
        return "screen_positive_needs_full_or_width_check", (
            f"coverage delta {(cov_delta or 0.0):+.4f}, width delta {(width_delta or 0.0):+.4f}, "
            f"worst-slice delta {(worst_delta or 0.0):+.4f}"
        )
    return "discard_or_hold", "no clear coverage-width-worst-slice improvement"


def _verdict(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    if any(row.get("summary_decision") == "keep_for_segmentation_style_interval_calibration" for row in rows):
        return "region_mixture_route_has_support"
    if any(row.get("summary_decision") == "screen_positive_needs_full_or_width_check" for row in rows):
        return "screen_positive_needs_more_evidence"
    return "no_clear_region_mixture_gain"


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
        "# Region Mixture Summary",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        report["claim_boundary"],
        "",
        "| run | probe | checkpoint | global cov/w | best policy | best cov/w | worst best | decision |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    if not report["rows"]:
        lines.append("| - | - | - | - | - | - | - | `missing` |")
    for row in report["rows"]:
        global_row = row.get("global", {})
        best = row.get("best_policy", {})
        lines.append(
            "| {run} | {probe} | {ckpt} | {global_vals} | {policy} | {best_vals} | {worst} | `{decision}` |".format(
                run=row.get("run"),
                probe=row.get("probe"),
                ckpt=row.get("checkpoint_name") or "-",
                global_vals=_cov_width(global_row),
                policy=row.get("best_policy_name") or "-",
                best_vals=_cov_width(best),
                worst=_worst(best),
                decision=row.get("summary_decision"),
            )
        )
    if report.get("decisions"):
        lines.extend(["", "## Decision Notes", ""])
        for row in report["decisions"]:
            lines.append(f"- `{row.get('run')}::{row.get('probe')}`: `{row.get('decision')}`; {row.get('reason')}")
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


def _num(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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

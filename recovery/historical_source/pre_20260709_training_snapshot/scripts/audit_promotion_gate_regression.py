from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_DIR = Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715")
DEFAULT_OUTPUT_DIR = DEFAULT_COMPARISON_DIR / "promotion_gate_regression_20260716"
DEFAULT_S96_DIR = Path(r"E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s96_wc_pair_relative_boundary_20260712")
DEFAULT_S7_DIR = Path(r"E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709")


@dataclass
class GateCase:
    name: str
    returncode: int
    expected_returncode: int
    passed: bool | None
    expected_passed: bool
    ok: bool
    checks: dict[str, Any]
    report: str


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_case(
    *,
    name: str,
    candidate_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    expected_returncode: int,
    expected_passed: bool,
    require_sota: bool = False,
) -> GateCase:
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        "scripts/audit_rscd_candidate_promotion.py",
        "--candidate-dir",
        str(candidate_dir),
        "--baseline-dir",
        str(baseline_dir),
        "--candidate-name",
        name,
        "--baseline-name",
        "baseline",
        "--output-dir",
        str(case_dir),
    ]
    if require_sota:
        args.append("--require-sota")
    completed = subprocess.run(
        args,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    payload = read_json(case_dir / "promotion_audit.json") or {}
    decision = payload.get("decision") or {}
    passed = decision.get("passed")
    checks = decision.get("checks") or {}
    ok = completed.returncode == expected_returncode and passed == expected_passed
    return GateCase(
        name=name,
        returncode=completed.returncode,
        expected_returncode=expected_returncode,
        passed=passed if isinstance(passed, bool) else None,
        expected_passed=expected_passed,
        ok=ok,
        checks=checks,
        report=str(case_dir / "promotion_audit.md"),
    )


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Promotion Gate Regression",
        "",
        f"- Overall: `{payload['overall']}`",
        "",
        "## Cases",
        "",
        "| Case | OK | Return | Expected Return | Passed | Expected Passed | Failed Checks |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        failed = [key for key, value in case["checks"].items() if not value]
        lines.append(
            f"| {case['name']} | {case['ok']} | {case['returncode']} | {case['expected_returncode']} | "
            f"{case['passed']} | {case['expected_passed']} | {', '.join(failed) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `s96_self` proves a same-protocol screen comparison can pass when all metrics and predictions align.",
            "- `screen_vs_full_mismatch` proves screen/full sample mismatch is now blocked by protocol and prediction-row gates.",
            "- `s7_self_require_sota` proves a full run with 49,500 aligned samples still fails final promotion if it does not beat public SOTA and has no paired net improvement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression-test RSCD promotion audit fairness gates.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--s96-dir", type=Path, default=DEFAULT_S96_DIR)
    parser.add_argument("--s7-dir", type=Path, default=DEFAULT_S7_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        run_case(
            name="s96_self",
            candidate_dir=args.s96_dir,
            baseline_dir=args.s96_dir,
            output_dir=args.output_dir,
            expected_returncode=0,
            expected_passed=True,
        ),
        run_case(
            name="screen_vs_full_mismatch",
            candidate_dir=args.s96_dir,
            baseline_dir=args.s7_dir,
            output_dir=args.output_dir,
            expected_returncode=2,
            expected_passed=False,
        ),
        run_case(
            name="s7_self_require_sota",
            candidate_dir=args.s7_dir,
            baseline_dir=args.s7_dir,
            output_dir=args.output_dir,
            expected_returncode=2,
            expected_passed=False,
            require_sota=True,
        ),
    ]
    payload = {
        "overall": "pass" if all(case.ok for case in cases) else "fail",
        "cases": [asdict(case) for case in cases],
    }
    (args.output_dir / "promotion_gate_regression.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "promotion_gate_regression.md")
    print(json.dumps({"overall": payload["overall"], "report": str(args.output_dir / "promotion_gate_regression.md")}, ensure_ascii=False))
    return 0 if payload["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

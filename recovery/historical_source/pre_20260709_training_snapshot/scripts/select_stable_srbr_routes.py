from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_CSV = Path("reports/paper_protocol_summary/srbr_route_candidates_20260709/srbr_route_candidates.csv")
DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary/stable_srbr_route_selection_20260709")


def pair_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))  # type: ignore[return-value]


def family(label: str) -> str:
    parts = label.split("_")
    if len(parts) == 3:
        return f"{parts[0]}_{parts[1]}"
    return parts[0] if parts else label


def read_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in [
                "score",
                "errors_to_fix",
                "reverse_errors_watch",
                "source_f1_%",
                "target_f1_%",
                "target_recall_%",
                "target_support",
                "target_miss_rate_%",
                "suggested_min_source_f1",
                "suggested_route_scale",
                "suggested_margin",
            ]:
                parsed[key] = float(parsed[key])
            parsed["supported_kind"] = str(parsed.get("supported_kind", "")).lower() == "true"
            rows.append(parsed)
    return rows


def stable_scale(row: dict[str, Any]) -> float:
    fix = max(float(row["errors_to_fix"]), 1.0)
    reverse = max(float(row["reverse_errors_watch"]), 0.0)
    source_f1 = float(row["source_f1_%"]) / 100.0
    target_f1 = float(row["target_f1_%"]) / 100.0
    base = float(row["suggested_route_scale"])
    reverse_ratio = reverse / (fix + reverse)
    reliability = max(min((source_f1 - 0.78) / 0.18, 1.0), 0.10)
    weak_target_boost = 1.10 if target_f1 < 0.82 else 1.0
    damp = (1.0 - 0.55 * reverse_ratio) * reliability * weak_target_boost
    return round(min(max(base * damp, 0.018), 0.090), 3)


def stable_score(row: dict[str, Any]) -> float:
    fix = max(float(row["errors_to_fix"]), 1.0)
    reverse = max(float(row["reverse_errors_watch"]), 0.0)
    source_f1 = float(row["source_f1_%"]) / 100.0
    target_f1 = float(row["target_f1_%"]) / 100.0
    reverse_ratio = reverse / (fix + reverse)
    balance = min(fix, reverse) / max(fix, reverse, 1.0)
    factor = str(row["factor"])
    kind = str(row["kind"])
    factor_prior = {"roughness": 1.24, "friction": 0.86, "material": 0.62}.get(factor, 0.50)
    weak_target_bonus = 1.35 if target_f1 < 0.80 else 1.18 if target_f1 < 0.84 else 1.0
    kind_bonus = 1.0 if kind != "generic" else 0.72
    source_reliability = max(min((source_f1 - 0.78) / 0.16, 1.0), 0.0)
    stability = 1.0 / (1.0 + 1.8 * reverse_ratio + 0.8 * balance)
    return fix * factor_prior * weak_target_bonus * kind_bonus * (0.35 + 0.65 * source_reliability) * stability


def select_routes(
    candidates: list[dict[str, Any]],
    *,
    max_routes: int,
    min_source_f1: float,
    allow_generic: bool,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in candidates:
        if float(row["source_f1_%"]) < 100.0 * min_source_f1:
            continue
        if not bool(row["supported_kind"]):
            continue
        if not allow_generic and str(row["kind"]) == "generic":
            continue
        row = dict(row)
        row["stable_score"] = round(stable_score(row), 4)
        row["stable_route_scale"] = stable_scale(row)
        row["source_family"] = family(str(row["source"]))
        row["target_family"] = family(str(row["target"]))
        filtered.append(row)
    filtered.sort(key=lambda row: float(row["stable_score"]), reverse=True)

    selected: list[dict[str, Any]] = []
    selected_pairs: set[tuple[str, str]] = set()
    factor_count: dict[str, int] = defaultdict(int)
    family_count: dict[str, int] = defaultdict(int)
    for row in filtered:
        if len(selected) >= max_routes:
            break
        factor_name = str(row["factor"])
        key = pair_key(str(row["source"]), str(row["target"]))
        fam = str(row["source_family"]) if row["source_family"] == row["target_family"] else "cross_family"
        if key in selected_pairs:
            continue
        if factor_name == "roughness" and factor_count[factor_name] >= max(2, max_routes - 1):
            continue
        if factor_name != "roughness" and factor_count[factor_name] >= 1:
            continue
        if family_count[fam] >= 2:
            continue
        selected.append(row)
        selected_pairs.add(key)
        factor_count[factor_name] += 1
        family_count[fam] += 1
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def select(candidate_csv: Path, out_dir: Path, max_routes: int, min_source_f1: float, allow_generic: bool) -> int:
    candidates = read_candidates(candidate_csv)
    selected = select_routes(
        candidates,
        max_routes=max_routes,
        min_source_f1=min_source_f1,
        allow_generic=allow_generic,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "stable_score",
        "source",
        "target",
        "kind",
        "factor",
        "errors_to_fix",
        "reverse_errors_watch",
        "source_f1_%",
        "target_f1_%",
        "suggested_route_scale",
        "stable_route_scale",
        "suggested_margin",
        "no_harm_watch",
    ]
    write_csv(out_dir / "stable_srbr_selected_routes.csv", selected, fields)

    yaml_lines: list[str] = []
    for row in selected:
        yaml_lines.extend(
            [
                "    - source: " + str(row["source"]),
                "      target: " + str(row["target"]),
                "      topk: 3",
                f"      margin: {float(row['suggested_margin']):.2f}",
                f"      source_f1: {float(row['source_f1_%']) / 100.0:.12f}",
                f"      min_source_f1: {max(float(row['source_f1_%']) / 100.0 - 0.02, 0.0):.6f}",
                f"      route_scale: {float(row['stable_route_scale']):.3f}",
                "      kind: " + str(row["kind"]),
            ]
        )

    lines = [
        "# Stable SRBR Route Selection",
        "",
        f"- Candidate csv: `{candidate_csv}`",
        f"- Max selected routes: {max_routes}",
        f"- Minimum source F1: {100.0 * min_source_f1:.2f}%",
        f"- Generic kinds allowed: {allow_generic}",
        "",
        "## Selected Routes",
        "",
    ]
    for row in selected:
        lines.append(
            "- `{source} -> {target}` ({factor}, {kind}): stable score {score:.2f}, "
            "fix {fix} errors, reverse watch {rev}, scale {scale}".format(
                source=row["source"],
                target=row["target"],
                factor=row["factor"],
                kind=row["kind"],
                score=float(row["stable_score"]),
                fix=int(float(row["errors_to_fix"])),
                rev=int(float(row["reverse_errors_watch"])),
                scale=float(row["stable_route_scale"]),
            )
        )
    lines.extend(
        [
            "",
            "## Control Interpretation",
            "",
            "- Each route is treated as a bounded feedback correction on a directed confusion edge.",
            "- Reverse confusions damp the scale, preventing a high-gain two-node oscillation where fixing one boundary damages the opposite boundary.",
            "- The selector prefers roughness/coupling edges with supported PhysicsTexture gates and rejects generic edges by default.",
            "- The route remains inside the existing SRBR mechanism: feature update direction is `w_target - w_source`, gated by pair ambiguity and physics evidence.",
            "",
            "## YAML Snippet",
            "",
            "```yaml",
            *yaml_lines,
            "```",
        ]
    )
    (out_dir / "stable_srbr_route_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out_dir": str(out_dir), "selected": len(selected)}, ensure_ascii=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Select stable SRBR route sets using graph/control-inspired constraints.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-routes", type=int, default=3)
    parser.add_argument("--min-source-f1", type=float, default=0.80)
    parser.add_argument("--allow-generic", action="store_true")
    args = parser.parse_args()
    raise SystemExit(select(args.candidate_csv, args.out_dir, args.max_routes, args.min_source_f1, args.allow_generic))


if __name__ == "__main__":
    main()

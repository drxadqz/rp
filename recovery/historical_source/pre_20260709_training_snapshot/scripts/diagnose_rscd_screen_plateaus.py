from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


KEY_CLASSES = (
    "water_concrete_slight",
    "wet_concrete_slight",
    "water_concrete_severe",
    "wet_concrete_severe",
    "dry_concrete_slight",
    "dry_concrete_severe",
    "water_asphalt_slight",
)


ROUTE_KEYWORDS = {
    "late_gate": ("gate_", "selector", "router"),
    "ordinal_boundary": ("ordinal", "boundary", "lowmargin"),
    "factor_graph": ("factor_graph", "factorized", "cotune"),
    "contact_kernel": ("contact_kernel", "kernel"),
    "film_texture": ("film", "water_concrete", "wetconcrete", "wc_"),
    "custom_backbone": ("s136", "s137", "s138"),
    "smoke_debug": ("debug", "smoke"),
}


@dataclass
class Plateau:
    signature: str
    count: int
    top1: float
    macro_f1: float
    worst_class: str
    worst_f1: float
    water_concrete_slight_f1: float | None
    representative_runs: list[str]


def read_board(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
    return rows


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or value == "":
        return default
    return int(float(value))


def quant(value: float | None, places: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{value:.{places}f}"


def signature(row: dict[str, Any]) -> str:
    parts = [
        quant(f(row, "top1")),
        quant(f(row, "macro_f1")),
        str(row.get("worst_class") or ""),
        quant(f(row, "worst_f1")),
        quant(f(row, "water_concrete_slight_f1")),
        quant(f(row, "wet_concrete_slight_f1")),
        quant(f(row, "water_concrete_severe_f1")),
        quant(f(row, "wet_concrete_severe_f1")),
        quant(f(row, "dry_concrete_slight_f1")),
        quant(f(row, "dry_concrete_severe_f1")),
    ]
    return "|".join(parts)


def route_family(run_name: str) -> str:
    lower = run_name.lower()
    for family, needles in ROUTE_KEYWORDS.items():
        if any(needle in lower for needle in needles):
            return family
    return "other"


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.3f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:+.3f} pp"


def plateau_rows(screen_rows: list[dict[str, Any]]) -> list[Plateau]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in screen_rows:
        buckets[signature(row)].append(row)
    plateaus: list[Plateau] = []
    for sig, members in buckets.items():
        rep = sorted(members, key=lambda item: str(item.get("run") or ""))[:8]
        first = members[0]
        plateaus.append(
            Plateau(
                signature=sig,
                count=len(members),
                top1=f(first, "top1"),
                macro_f1=f(first, "macro_f1"),
                worst_class=str(first.get("worst_class") or ""),
                worst_f1=f(first, "worst_f1"),
                water_concrete_slight_f1=None if first.get("water_concrete_slight_f1") is None else f(first, "water_concrete_slight_f1"),
                representative_runs=[str(item.get("run") or "") for item in rep],
            )
        )
    plateaus.sort(key=lambda item: (item.count, item.top1, item.macro_f1), reverse=True)
    return plateaus


def deltas(row: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, float | None]:
    if baseline is None:
        return {"top1": None, "macro_f1": None, **{key: None for key in KEY_CLASSES}}
    out: dict[str, float | None] = {
        "top1": f(row, "top1") - f(baseline, "top1"),
        "macro_f1": f(row, "macro_f1") - f(baseline, "macro_f1"),
    }
    for key in KEY_CLASSES:
        col = f"{key}_f1"
        if row.get(col) is None or baseline.get(col) is None:
            out[key] = None
        else:
            out[key] = f(row, col) - f(baseline, col)
    return out


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# RSCD Screen Plateau Diagnosis",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Board: `{payload['board_path']}`",
        f"- Screen rows: `{payload['screen_count']}`",
        f"- Unique metric/key-class signatures: `{payload['unique_signatures']}`",
        f"- Largest plateau count: `{payload['largest_plateau_count']}`",
        "",
        "## Baseline And Best Screen",
        "",
        "| Item | Run | Top-1 | Macro-F1 | Worst class | Worst F1 | water_concrete_slight |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for label in ["baseline_screen", "best_screen"]:
        row = payload.get(label)
        if not row:
            continue
        lines.append(
            f"| {label} | {row['run']} | {pct(f(row, 'top1'))} | {pct(f(row, 'macro_f1'))} | "
            f"{row.get('worst_class') or '-'} | {pct(f(row, 'worst_f1'))} | "
            f"{pct(None if row.get('water_concrete_slight_f1') is None else f(row, 'water_concrete_slight_f1'))} |"
        )
    best_delta = payload.get("best_screen_delta_vs_baseline") or {}
    lines.extend(
        [
            "",
            "Best screen delta versus S96 baseline:",
            "",
            f"- Top-1: `{pp(best_delta.get('top1'))}`",
            f"- Macro-F1: `{pp(best_delta.get('macro_f1'))}`",
            f"- water_concrete_slight F1: `{pp(best_delta.get('water_concrete_slight'))}`",
            f"- wet_concrete_slight F1: `{pp(best_delta.get('wet_concrete_slight'))}`",
            "",
            "## Largest Repeated Plateaus",
            "",
            "| Count | Top-1 | Macro-F1 | Worst class | Worst F1 | water_concrete_slight | Representative runs |",
            "|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for item in payload["plateaus"][:10]:
        lines.append(
            f"| {item['count']} | {pct(item['top1'])} | {pct(item['macro_f1'])} | "
            f"{item['worst_class'] or '-'} | {pct(item['worst_f1'])} | "
            f"{pct(item.get('water_concrete_slight_f1'))} | "
            f"{'; '.join(item['representative_runs'][:4])} |"
        )
    lines.extend(
        [
            "",
            "## Route-Family Counts",
            "",
            "| Family | Count | Best Top-1 | Best Macro-F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for family, item in payload["family_summary"].items():
        lines.append(f"| {family} | {item['count']} | {pct(item['best_top1'])} | {pct(item['best_macro_f1'])} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Many late/gated/factor-boundary screen variants collapse onto the same metric and key-class signature.",
            "- The best screen improves S96 only marginally and does not repair `water_concrete_slight`; that class remains the repeated worst class.",
            "- This is negative evidence against adding more late heads, lightweight gates, or post-hoc boundary tweaks around the same S96 anchor.",
            "- The useful next route remains early mechanism-conditioned custom backbones with same-budget controls: S136/S136d/S137, then S138 if S137 fails.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose repeated RSCD screen plateaus from the experiment board.")
    parser.add_argument(
        "--board-csv",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\experiment_board_20260716\rscd_experiment_board.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\screen_plateau_diagnosis_20260716"),
    )
    parser.add_argument("--baseline-run", default="c3_farnet_screen_s96_wc_pair_relative_boundary_20260712")
    args = parser.parse_args()

    rows = read_board(args.board_csv)
    screen_rows = [row for row in rows if row.get("protocol") == "screen"]
    baseline = next((row for row in screen_rows if row.get("run") == args.baseline_run), None)
    best_screen = max(screen_rows, key=lambda row: (f(row, "macro_f1"), f(row, "top1")), default=None)
    plateaus = plateau_rows(screen_rows)
    family_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in screen_rows:
        family_members[route_family(str(row.get("run") or ""))].append(row)
    family_summary = {}
    for family, members in sorted(family_members.items()):
        family_summary[family] = {
            "count": len(members),
            "best_top1": max(f(row, "top1") for row in members),
            "best_macro_f1": max(f(row, "macro_f1") for row in members),
        }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "board_path": str(args.board_csv),
        "screen_count": len(screen_rows),
        "unique_signatures": len(plateaus),
        "largest_plateau_count": plateaus[0].count if plateaus else 0,
        "baseline_screen": baseline,
        "best_screen": best_screen,
        "best_screen_delta_vs_baseline": deltas(best_screen, baseline) if best_screen else {},
        "plateaus": [asdict(item) for item in plateaus],
        "family_summary": family_summary,
        "route_family_counts": dict(Counter(route_family(str(row.get("run") or "")) for row in screen_rows)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "screen_plateau_diagnosis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "screen_plateau_diagnosis.md")
    print(json.dumps({"screen_rows": len(screen_rows), "unique_signatures": len(plateaus), "report": str(args.output_dir / "screen_plateau_diagnosis.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

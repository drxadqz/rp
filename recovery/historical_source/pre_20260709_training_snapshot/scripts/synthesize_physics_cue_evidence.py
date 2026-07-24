from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FOCUS_PAIRS = [
    "water_concrete_slight::water_concrete_severe",
    "wet_concrete_slight::wet_concrete_severe",
    "dry_concrete_slight::dry_concrete_severe",
    "water_concrete_smooth::wet_concrete_smooth",
    "water_asphalt_slight::water_asphalt_severe",
]

FEATURE_FAMILIES = {
    "gray_std": "contrast_visibility",
    "sat_std": "chromatic_micro_variation",
    "gray_q10": "dark_film_quantile",
    "gray_mean": "illumination_level",
    "lower_gray_mean": "contact_region_illumination",
    "upper_gray_mean": "global_illumination",
    "dark_water_ratio": "water_film_darkness",
    "specular_ratio": "water_film_reflectance",
    "wet_proxy_mean": "water_film_proxy",
    "lower_wet_proxy": "contact_region_water_film",
    "rough_proxy_mean": "texture_roughness",
    "lower_rough_proxy": "contact_region_texture_roughness",
    "grad_mean": "edge_texture",
    "grad_q90": "strong_edge_texture",
    "lap_abs_mean": "micro_texture_laplacian",
    "lap_abs_q90": "strong_micro_texture_laplacian",
    "texture_to_wet_ratio": "texture_water_balance",
    "wet_to_texture_ratio": "water_texture_balance",
}


def _parse_analysis_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, raw = spec.split("=", 1)
    else:
        path = Path(spec)
        name, raw = path.name, spec
    return name, Path(raw)


def _pair_key(a: str, b: str) -> str:
    aa, bb = sorted((a, b))
    return f"{aa}::{bb}"


def _normalise_pair_spec(spec: str) -> str:
    if "::" not in spec:
        return spec
    a, b = spec.split("::", 1)
    return _pair_key(a, b)


def _read_pair_rows(path: Path, run_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_a = str(row.get("class_a", ""))
            class_b = str(row.get("class_b", ""))
            feature = str(row.get("feature", ""))
            if not class_a or not class_b or not feature:
                continue
            rows.append(
                {
                    "run_name": run_name,
                    "pair_key": _pair_key(class_a, class_b),
                    "class_a": class_a,
                    "class_b": class_b,
                    "feature": feature,
                    "family": FEATURE_FAMILIES.get(feature, "other"),
                    "mean_a": float(row.get("mean_a") or 0.0),
                    "mean_b": float(row.get("mean_b") or 0.0),
                    "delta_a_minus_b": float(row.get("delta_a_minus_b") or 0.0),
                    "cohen_d_a_minus_b": float(row.get("cohen_d_a_minus_b") or 0.0),
                    "abs_cohen_d": float(row.get("abs_cohen_d") or 0.0),
                    "auc_a_greater_b": float(row.get("auc_a_greater_b") or 0.5),
                    "confusion_count_a_to_b": int(float(row.get("confusion_count_a_to_b") or 0)),
                    "confusion_count_b_to_a": int(float(row.get("confusion_count_b_to_a") or 0)),
                }
            )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mechanism_hint(pair_key: str, stable_rows: list[dict[str, Any]]) -> list[str]:
    if not stable_rows:
        return ["No stable cue found for this pair."]
    top = stable_rows[0]
    lines = [
        f"Top stable cue: `{top['feature']}` ({top['family']}); mean |d| = {_fmt(float(top['mean_abs_cohen_d']))}.",
    ]
    if pair_key == _pair_key("water_concrete_slight", "water_concrete_severe"):
        lines.extend(
            [
                "Candidate route: build a Contrast-Visibility Coupled stem that estimates local gray contrast, low-brightness water-film quantiles, and saturation variance before the first backbone downsampling.",
                "Task target: separate `water + concrete + slight` from `water + concrete + severe` by conditioning early texture extraction on whether concrete texture remains visible through the water film.",
                "Implementation rule: use this as an early conditioner or task-specific stem; do not add it as a late residual/head.",
            ]
        )
    elif "concrete_slight" in pair_key or "concrete_severe" in pair_key:
        lines.append("Candidate route: use class-conditional roughness gates only inside concrete-like material evidence, because the same roughness threshold is not stable across asphalt/gravel/mud.")
    elif "smooth" in pair_key and "wet" in pair_key:
        lines.append("Candidate route: use film-vs-wetness contrast cues to stop water/wet smooth concrete swaps, not generic roughness features.")
    else:
        lines.append("Candidate route: keep this cue as a diagnostic unless it overlaps the current water/concrete bottleneck.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize stable physics-cue evidence across RSCD runs.")
    parser.add_argument("--analysis", action="append", required=True, help="NAME=DIR containing pair_physics_separability.csv.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--focus-pair", action="append", default=[])
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    focus_pairs = {_normalise_pair_spec(spec) for spec in DEFAULT_FOCUS_PAIRS}
    focus_pairs.update(_normalise_pair_spec(spec) for spec in args.focus_pair)

    all_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    for spec in args.analysis:
        name, path = _parse_analysis_spec(spec)
        pair_path = path / "pair_physics_separability.csv"
        inputs.append({"name": name, "path": str(path), "pair_csv": str(pair_path), "exists": str(pair_path.exists())})
        if pair_path.exists():
            all_rows.extend(_read_pair_rows(pair_path, name))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["pair_key"] in focus_pairs:
            grouped[(str(row["pair_key"]), str(row["feature"]))].append(row)

    stable_rows: list[dict[str, Any]] = []
    for (pair_key, feature), rows in grouped.items():
        run_names = sorted({str(row["run_name"]) for row in rows})
        if len(run_names) < args.min_runs:
            continue
        signs = [_sign(float(row["cohen_d_a_minus_b"])) for row in rows]
        nonzero = [sign for sign in signs if sign != 0]
        sign_consistency = 0.0
        if nonzero:
            majority = max(set(nonzero), key=nonzero.count)
            sign_consistency = nonzero.count(majority) / len(nonzero)
        stable_rows.append(
            {
                "pair_key": pair_key,
                "feature": feature,
                "family": FEATURE_FAMILIES.get(feature, "other"),
                "runs": len(run_names),
                "run_names": ";".join(run_names),
                "mean_abs_cohen_d": _mean([float(row["abs_cohen_d"]) for row in rows]),
                "mean_signed_cohen_d": _mean([float(row["cohen_d_a_minus_b"]) for row in rows]),
                "mean_auc_a_greater_b": _mean([float(row["auc_a_greater_b"]) for row in rows]),
                "sign_consistency": sign_consistency,
                "total_confusion_count": sum(int(row["confusion_count_a_to_b"]) + int(row["confusion_count_b_to_a"]) for row in rows),
            }
        )
    stable_rows.sort(
        key=lambda row: (
            str(row["pair_key"]),
            -float(row["sign_consistency"]),
            -float(row["mean_abs_cohen_d"]),
            -int(row["total_confusion_count"]),
        )
    )
    _write_csv(args.output_dir / "stable_physics_cue_evidence.csv", stable_rows)

    best_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stable_rows:
        best_by_pair[str(row["pair_key"])].append(row)

    payload = {
        "ok": True,
        "inputs": inputs,
        "focus_pairs": sorted(focus_pairs),
        "min_runs": args.min_runs,
        "num_raw_rows": len(all_rows),
        "num_stable_rows": len(stable_rows),
        "top_by_pair": {pair: rows[: args.top_k] for pair, rows in sorted(best_by_pair.items())},
    }
    (args.output_dir / "stable_physics_cue_evidence.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Stable RSCD Physics-Cue Evidence",
        "",
        "This synthesis merges physics-cue separability from multiple runs. A cue is considered useful here only if it appears for the same hard class pair in at least the requested number of runs.",
        "",
        "## Inputs",
        "",
        "| Name | Pair CSV | Exists |",
        "|---|---|---:|",
    ]
    for item in inputs:
        md.append(f"| {item['name']} | `{item['pair_csv']}` | {item['exists']} |")

    md.extend(["", "## Stable Cue Rankings", ""])
    for pair in sorted(best_by_pair):
        md.extend([f"### {pair.replace('::', ' vs ')}", "", "| Feature | Family | Runs | Mean |d| | Signed d | Sign consistency | AUC | Confusions |", "|---|---|---:|---:|---:|---:|---:|---:|"])
        for row in best_by_pair[pair][: args.top_k]:
            md.append(
                f"| {row['feature']} | {row['family']} | {row['runs']} | {_fmt(float(row['mean_abs_cohen_d']))} | "
                f"{_fmt(float(row['mean_signed_cohen_d']))} | {_fmt(float(row['sign_consistency']))} | "
                f"{_fmt(float(row['mean_auc_a_greater_b']))} | {row['total_confusion_count']} |"
            )
        md.extend(["", "Mechanism implication:"])
        md.extend([f"- {line}" for line in _mechanism_hint(pair, best_by_pair[pair])])
        md.append("")

    wc_pair = _pair_key("water_concrete_slight", "water_concrete_severe")
    wc_rows = best_by_pair.get(wc_pair, [])
    md.extend(["## Next Single Route Constraint", ""])
    if wc_rows:
        top_features = ", ".join(f"`{row['feature']}`" for row in wc_rows[:4])
        md.append(
            f"If the queued S135c route fails promotion, the next single route should be built around {top_features}. "
            "The intended mechanism is early contrast-visibility conditioning under water film, not another late classifier correction."
        )
    else:
        md.append("No stable water-concrete slight/severe cue was found; rerun analyses before designing a new route.")

    (args.output_dir / "stable_physics_cue_evidence.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "stable_physics_cue_evidence.md"), "ok": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

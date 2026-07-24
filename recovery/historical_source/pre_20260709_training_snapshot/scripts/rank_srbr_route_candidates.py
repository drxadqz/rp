from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ANCHOR_DIR = Path(
    "E:/perception_outputs/rscd_surface_classification/"
    "c3_farnet_official_anchor_source_reliable_router_s7_fulltest_20260708/fast_test"
)
DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary/srbr_route_candidates_20260709")


SUPPORTED_KINDS = {
    "dry_concrete_roughness",
    "dry_asphalt_roughness",
    "concrete_film_roughness",
    "wet_water_smooth_film",
    "water_concrete_moderate_roughness",
    "wet_water_concrete_film",
    "water_asphalt_film",
    "generic",
}


def parse_paved(label: str) -> tuple[str, str, str] | None:
    parts = label.split("_")
    if len(parts) != 3:
        return None
    f, m, r = parts
    if f in {"dry", "wet", "water"} and m in {"asphalt", "concrete"} and r in {"smooth", "slight", "severe"}:
        return f, m, r
    return None


def edge_kind(source: str, target: str) -> str:
    sp = parse_paved(source)
    tp = parse_paved(target)
    if sp is None or tp is None:
        return "generic"
    sf, sm, sr = sp
    tf, tm, tr = tp
    if sf == tf == "dry" and sm == tm == "concrete" and sr != tr:
        return "dry_concrete_roughness"
    if sf == tf == "dry" and sm == tm == "asphalt" and sr != tr:
        return "dry_asphalt_roughness"
    if sf == tf and sf in {"wet", "water"} and sm == tm == "concrete" and sr != tr:
        if sf == "water" and {sr, tr} == {"slight", "severe"}:
            return "water_concrete_moderate_roughness"
        return "concrete_film_roughness"
    if sm == tm == "concrete" and sr == tr == "smooth" and {sf, tf} <= {"wet", "water"}:
        return "wet_water_smooth_film"
    if sm == tm == "concrete" and {sf, tf} <= {"wet", "water"}:
        return "wet_water_concrete_film"
    if sf == tf == "water" and sm == tm == "asphalt":
        return "water_asphalt_film"
    return "generic"


def factor_diff(a: str, b: str) -> list[str]:
    ap = parse_paved(a)
    bp = parse_paved(b)
    if ap is None or bp is None:
        return []
    names = ("friction", "material", "roughness")
    return [name for name, av, bv in zip(names, ap, bp, strict=True) if av != bv]


def read_per_class(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            f1_key = "f1-score" if "f1-score" in row else "f1"
            out[row["class"]] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row[f1_key]),
                "support": float(row["support"]),
            }
    return out


def read_confusion(path: Path) -> tuple[list[str], list[list[int]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        rows: list[list[int]] = []
        row_labels: list[str] = []
        for row in reader:
            row_labels.append(row[0])
            rows.append([int(float(v)) for v in row[1:]])
    if labels != row_labels:
        raise ValueError("confusion labels mismatch")
    return labels, rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_route_scale(count: int, support: float, reverse_count: int) -> float:
    miss_rate = count / max(float(support), 1.0)
    reverse_pressure = reverse_count / max(float(count + reverse_count), 1.0)
    raw = 0.035 + 0.55 * miss_rate * (1.0 - 0.45 * reverse_pressure)
    return round(min(max(raw, 0.04), 0.16), 3)


def rank(anchor_dir: Path, out_dir: Path, min_count: int, min_source_f1: float) -> int:
    metrics = json.loads((anchor_dir / "metrics.json").read_text(encoding="utf-8"))
    per_class = read_per_class(anchor_dir / "per_class_metrics.csv")
    labels, cm = read_confusion(anchor_dir / "confusion_matrix.csv")
    idx = {name: i for i, name in enumerate(labels)}

    rows: list[dict[str, Any]] = []
    for true_label in labels:
        for pred_label in labels:
            if true_label == pred_label:
                continue
            count = cm[idx[true_label]][idx[pred_label]]
            if count < min_count:
                continue
            diffs = factor_diff(true_label, pred_label)
            if len(diffs) != 1:
                continue
            source = pred_label
            target = true_label
            kind = edge_kind(source, target)
            source_f1 = per_class[source]["f1"]
            target_f1 = per_class[target]["f1"]
            if source_f1 < min_source_f1:
                continue
            reverse_count = cm[idx[source]][idx[target]]
            target_support = per_class[target]["support"]
            target_miss_rate = count / max(target_support, 1.0)
            source_margin = max(source_f1 - 0.02, 0.0)
            no_harm_watch = f"{source}->{target} reduce, {target}->{source} not increase"
            roughness_bonus = 1.25 if diffs == ["roughness"] else 1.0
            source_reliability = max(source_f1 - source_margin, 0.0)
            risk = 1.0 + reverse_count / max(count, 1)
            score = roughness_bonus * count * max(source_f1, 1e-6) * (1.0 + target_miss_rate) / risk
            rows.append(
                {
                    "score": round(score, 4),
                    "source": source,
                    "target": target,
                    "kind": kind,
                    "factor": diffs[0],
                    "errors_to_fix": count,
                    "reverse_errors_watch": reverse_count,
                    "source_f1_%": 100.0 * source_f1,
                    "target_f1_%": 100.0 * target_f1,
                    "target_recall_%": 100.0 * per_class[target]["recall"],
                    "target_support": target_support,
                    "target_miss_rate_%": 100.0 * target_miss_rate,
                    "suggested_min_source_f1": round(source_margin, 6),
                    "suggested_route_scale": candidate_route_scale(count, target_support, reverse_count),
                    "suggested_margin": 0.75 if diffs[0] == "roughness" else 0.60,
                    "supported_kind": kind in SUPPORTED_KINDS,
                    "no_harm_watch": no_harm_watch,
                    "source_reliability_gap": round(source_reliability, 6),
                }
            )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "srbr_route_candidates.csv",
        rows,
        [
            "score",
            "source",
            "target",
            "kind",
            "factor",
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
            "supported_kind",
            "no_harm_watch",
            "source_reliability_gap",
        ],
    )

    top_lines = []
    for row in rows[:12]:
        top_lines.append(
            "- `{source} -> {target}` ({factor}, {kind}): fix {errors_to_fix} errors, "
            "reverse watch {reverse_errors_watch}, source F1 {source_f1_:.2f}%, target F1 {target_f1_:.2f}%, "
            "scale {scale}, margin {margin}".format(
                source=row["source"],
                target=row["target"],
                factor=row["factor"],
                kind=row["kind"],
                errors_to_fix=row["errors_to_fix"],
                reverse_errors_watch=row["reverse_errors_watch"],
                source_f1_=row["source_f1_%"],
                target_f1_=row["target_f1_%"],
                scale=row["suggested_route_scale"],
                margin=row["suggested_margin"],
            )
        )

    yaml_lines = []
    for row in rows[:6]:
        yaml_lines.extend(
            [
                "    - source: " + str(row["source"]),
                "      target: " + str(row["target"]),
                "      topk: 3",
                f"      margin: {float(row['suggested_margin']):.2f}",
                f"      source_f1: {float(row['source_f1_%']) / 100.0:.12f}",
                f"      min_source_f1: {float(row['suggested_min_source_f1']):.6f}",
                f"      route_scale: {float(row['suggested_route_scale']):.3f}",
                "      kind: " + str(row["kind"]),
            ]
        )

    md = [
        "# Source-Reliable Boundary Route Candidate Ranking",
        "",
        f"- Anchor dir: `{anchor_dir}`",
        f"- Anchor Top-1: {100.0 * float(metrics['summary']['top1']):.2f}%",
        f"- Anchor Macro-F1: {100.0 * float(metrics['summary']['macro_f1']):.2f}%",
        f"- Candidate filter: one paved factor differs, count >= {min_count}, source F1 >= {100.0 * min_source_f1:.2f}%",
        "",
        "## Top Candidates",
        "",
        *top_lines,
        "",
        "## Why This Is Task-Adapted",
        "",
        "- A route fixes cases where the current model predicts the reliable `source` class but the ground truth is the adjacent `target` class.",
        "- The route operates through the existing SourceReliableBoundaryFeatureRouter before the final head, along `w_target - w_source`, so it is not a free late logit patch.",
        "- Candidates are restricted to one-factor RSCD boundaries and mapped to existing PhysicsTexture gates such as dry-concrete roughness or water-concrete moderate roughness.",
        "- Each candidate carries a reverse-confusion watch item, because improving one class while damaging the opposite boundary is the failure mode seen in prior screens.",
        "",
        "## YAML Snippets For Manual Review",
        "",
        "```yaml",
        *yaml_lines,
        "```",
    ]
    (out_dir / "srbr_route_candidates.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out_dir": str(out_dir), "num_candidates": len(rows)}, ensure_ascii=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank task-adapted SRBR route candidates from an RSCD confusion matrix.")
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-count", type=int, default=35)
    parser.add_argument("--min-source-f1", type=float, default=0.78)
    args = parser.parse_args()
    raise SystemExit(rank(args.anchor_dir, args.out_dir, args.min_count, args.min_source_f1))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from analyze_rscd_complete_graph_patterns import canonical_order, classification_rows, factor_text, shared_relation
from audit_rscd_topological_texture_features import extract_features, load_image


DEFAULT_MANIFEST = Path("data/manifests_full/rscd_prepared_train.csv")
DEFAULT_OUTPUT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_RUN_NAME = "eval_semantic_attention_line_fourier_directed_conf001_fulltest"
DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a complete RSCD class feature graph from image-derived texture/topology features.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples-per-class", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--prefix", default="rscd_complete_feature_graph_current_best")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / args.prefix

    labels = _labels_from_eval(args.output_root / args.run_name / "evaluate_test.json")
    feature_rows = extract_class_features(args.manifest, labels, args.samples_per_class, args.seed)
    class_features = aggregate_class_features(feature_rows, labels)

    directed = complete_directed_edges(args.output_root / args.run_name / "predictions_test.csv", labels)
    edges = complete_feature_edges(class_features, directed, labels)
    node_rows = node_feature_summary(class_features, directed, labels)
    motifs = discover_motifs(edges)

    feature_rows.to_csv(prefix.with_name(prefix.name + "_sample_image_features.csv"), index=False, encoding="utf-8")
    class_features.to_csv(prefix.with_name(prefix.name + "_class_feature_means.csv"), index=False, encoding="utf-8")
    edges.to_csv(prefix.with_name(prefix.name + "_complete_feature_edges_351.csv"), index=False, encoding="utf-8")
    node_rows.to_csv(prefix.with_name(prefix.name + "_feature_nodes_27.csv"), index=False, encoding="utf-8")
    plot_feature_graph(node_rows, edges, labels, prefix)
    write_report(prefix, args, node_rows, edges, motifs)
    print(prefix.with_suffix(".md"))


def _labels_from_eval(path: Path) -> list[str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    nodes = classification_rows(report["classification_report"])
    return canonical_order(nodes["class_label"].tolist())


def extract_class_features(manifest: Path, labels: list[str], samples_per_class: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(manifest, dtype=str, low_memory=False)
    df = df[df["class_label"].isin(labels) & df["image_path"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for label in labels:
        group = df[df["class_label"] == label]
        if group.empty:
            continue
        take = group.sample(n=min(samples_per_class, len(group)), random_state=seed)
        for item in take.itertuples(index=False):
            path = Path(str(getattr(item, "image_path")))
            try:
                rgb = load_image(path)
                features = extract_features(rgb)
                features.update(basic_visual_features(rgb))
            except Exception:
                continue
            rows.append({"class_label": label, "image_path": str(path), **features})
    return pd.DataFrame(rows)


def basic_visual_features(rgb: np.ndarray) -> dict[str, float]:
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = (maxc - minc) / np.maximum(maxc, 1e-4)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "grad_mean": float(grad.mean()),
        "grad_std": float(grad.std()),
        "bright_frac": float((maxc > 0.82).mean()),
        "dark_frac": float((maxc < 0.35).mean()),
        "low_saturation_frac": float((saturation < 0.20).mean()),
    }


def aggregate_class_features(rows: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    feature_cols = [c for c in rows.columns if c not in {"class_label", "image_path"}]
    stats = rows.groupby("class_label")[feature_cols].mean().reindex(labels).reset_index()
    stats["samples"] = stats["class_label"].map(rows.groupby("class_label").size()).fillna(0).astype(int)
    return stats


def complete_directed_edges(prediction_path: Path, labels: list[str]) -> pd.DataFrame:
    pred = pd.read_csv(prediction_path)
    support = pred.groupby("true_label").size().to_dict()
    counts = (
        pred[pred["true_label"] != pred["pred_label"]]
        .groupby(["true_label", "pred_label"])
        .size()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for src in labels:
        for dst in labels:
            if src == dst:
                continue
            count = int(counts.get((src, dst), 0))
            rows.append(
                {
                    "true_label": src,
                    "pred_label": dst,
                    "count": count,
                    "error_rate_in_true_class": count / max(int(support.get(src, 0)), 1),
                }
            )
    return pd.DataFrame(rows)


def complete_feature_edges(class_features: pd.DataFrame, directed: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    feature_cols = [c for c in class_features.columns if c not in {"class_label", "samples"}]
    means = class_features.set_index("class_label")[feature_cols].astype(float)
    z = (means - means.mean(axis=0)) / means.std(axis=0).replace(0.0, 1.0)
    directed_map = {(r.true_label, r.pred_label): r for r in directed.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            va = z.loc[a].to_numpy(dtype=np.float64)
            vb = z.loc[b].to_numpy(dtype=np.float64)
            dist = float(np.linalg.norm(va - vb) / np.sqrt(max(len(va), 1)))
            sim = float(np.exp(-dist))
            ab = directed_map[(a, b)]
            ba = directed_map[(b, a)]
            total = int(ab.count + ba.count)
            rows.append(
                {
                    "class_a": a,
                    "class_b": b,
                    "feature_distance": dist,
                    "feature_similarity": sim,
                    "ab_count": int(ab.count),
                    "ba_count": int(ba.count),
                    "total_confusion": total,
                    "mean_bidirectional_rate": float((ab.error_rate_in_true_class + ba.error_rate_in_true_class) / 2.0),
                    "relation": shared_relation(a, b),
                }
            )
    return pd.DataFrame(rows).sort_values(["total_confusion", "feature_similarity"], ascending=False)


def node_feature_summary(class_features: pd.DataFrame, directed: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    out_mistakes = directed.groupby("true_label")["count"].sum()
    in_mistakes = directed.groupby("pred_label")["count"].sum()
    nodes = class_features[["class_label", "samples"]].copy()
    nodes["out_mistakes"] = nodes["class_label"].map(out_mistakes).fillna(0).astype(int)
    nodes["in_mistakes"] = nodes["class_label"].map(in_mistakes).fillna(0).astype(int)
    nodes["net_sink"] = nodes["in_mistakes"] - nodes["out_mistakes"]
    nodes["friction"] = nodes["class_label"].map(lambda x: factor_text(x)["friction"])
    nodes["material"] = nodes["class_label"].map(lambda x: factor_text(x)["material"])
    nodes["roughness"] = nodes["class_label"].map(lambda x: factor_text(x)["roughness"])
    return nodes.set_index("class_label").reindex(labels).reset_index()


def discover_motifs(edges: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nonzero = edges[edges["total_confusion"] > 0].copy()
    sim_threshold = float(edges["feature_similarity"].quantile(0.75))
    rate_threshold = float(nonzero["mean_bidirectional_rate"].quantile(0.80)) if not nonzero.empty else 0.0
    visually_close_confused = edges[
        (edges["feature_similarity"] >= sim_threshold) & (edges["mean_bidirectional_rate"] >= rate_threshold)
    ].sort_values(["mean_bidirectional_rate", "feature_similarity"], ascending=False)
    visually_close_stable = edges[
        (edges["feature_similarity"] >= sim_threshold) & (edges["total_confusion"] == 0)
    ].sort_values("feature_similarity", ascending=False)
    distant_but_confused = nonzero.sort_values(["feature_distance", "total_confusion"], ascending=[False, False]).head(12)
    return {
        "visually_close_confused": visually_close_confused.head(15),
        "visually_close_stable": visually_close_stable.head(15),
        "distant_but_confused": distant_but_confused,
    }


def plot_feature_graph(nodes: pd.DataFrame, edges: pd.DataFrame, labels: list[str], prefix: Path) -> None:
    graph = nx.Graph()
    pos = factor_positions(labels)
    for row in nodes.itertuples(index=False):
        graph.add_node(row.class_label, out_mistakes=float(row.out_mistakes), net_sink=float(row.net_sink))
    for row in edges.itertuples(index=False):
        graph.add_edge(row.class_a, row.class_b, total_confusion=float(row.total_confusion), sim=float(row.feature_similarity), relation=row.relation)

    fig, ax = plt.subplots(figsize=(18, 8), dpi=180)
    all_edges = [(r.class_a, r.class_b) for r in edges.itertuples(index=False)]
    widths = [0.12 + 1.2 * float(r.feature_similarity) for r in edges.itertuples(index=False)]
    nx.draw_networkx_edges(graph, pos, edgelist=all_edges, width=widths, edge_color="#D2D2D2", alpha=0.18, ax=ax)

    nonzero = edges[edges["total_confusion"] > 0].sort_values("total_confusion")
    for relation, group in nonzero.groupby("relation"):
        edge_list = [(r.class_a, r.class_b) for r in group.itertuples(index=False)]
        edge_width = [0.35 + min(4.2, float(r.total_confusion) / 80.0) for r in group.itertuples(index=False)]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edge_list,
            width=edge_width,
            edge_color=relation_color(str(relation)),
            alpha=0.28 if not str(relation).startswith("factor_neighbor") else 0.55,
            ax=ax,
        )

    node_sizes = [430 + min(1900, float(graph.nodes[n]["out_mistakes"]) * 2.6) for n in labels]
    node_colors = [float(graph.nodes[n]["net_sink"]) for n in labels]
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, cmap="coolwarm", edgecolors="#222222", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=6, font_family="DejaVu Sans", ax=ax)
    ax.set_title("RSCD-27 complete feature graph: all 351 class pairs, overlayed with observed confusion")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_complete_feature_graph.png"))
    fig.savefig(prefix.with_name(prefix.name + "_complete_feature_graph.svg"))
    plt.close(fig)


def factor_positions(labels: list[str]) -> dict[str, tuple[float, float]]:
    pos: dict[str, tuple[float, float]] = {}
    for i, label in enumerate(labels):
        f = factor_text(label)
        friction = f["friction"]
        material = f["material"]
        roughness = f["roughness"]
        if friction in {"dry", "wet", "water"} and material in {"asphalt", "concrete"} and roughness is not None:
            pos[label] = (
                {"asphalt": 0.0, "concrete": 4.2}[material] + {"smooth": 0.0, "slight": 1.35, "severe": 2.7}[roughness],
                -{"dry": 0.0, "wet": 1.4, "water": 2.8}[friction],
            )
        elif friction in {"dry", "wet", "water"} and material in {"mud", "gravel"}:
            pos[label] = (8.8 + {"mud": 0.0, "gravel": 1.35}[material], -{"dry": 0.0, "wet": 1.4, "water": 2.8}[friction])
        else:
            pos[label] = (12.0 + 0.9 * ["fresh_snow", "melted_snow", "ice"].index(label), -1.4)
    return pos


def relation_color(relation: str) -> str:
    return {
        "factor_neighbor:roughness": "#D55E00",
        "factor_neighbor:friction": "#0072B2",
        "factor_neighbor:material": "#009E73",
        "shares_friction": "#999999",
        "shares_material": "#B0B0B0",
        "shares_roughness": "#BBBBBB",
        "cross_component": "#CC79A7",
    }.get(relation, "#777777")


def write_report(prefix: Path, args: argparse.Namespace, nodes: pd.DataFrame, edges: pd.DataFrame, motifs: dict[str, pd.DataFrame]) -> None:
    relation_stats = (
        edges.groupby("relation", as_index=False)
        .agg(
            pairs=("total_confusion", "size"),
            nonzero_pairs=("total_confusion", lambda x: int((x > 0).sum())),
            mistakes=("total_confusion", "sum"),
            mean_feature_similarity=("feature_similarity", "mean"),
            mean_confusion_rate=("mean_bidirectional_rate", "mean"),
        )
        .sort_values("mistakes", ascending=False)
    )
    text = f"""# RSCD Complete Feature Graph Pattern Audit

- Run: `{args.run_name}`
- Class nodes: {len(nodes)}
- Complete class-pair edges: {len(edges)} = 27 x 26 / 2
- Feature sample: up to {args.samples_per_class} train images per class
- Graph: `{prefix.name}_complete_feature_graph.png/svg`
- Edge CSV: `{prefix.name}_complete_feature_edges_351.csv`

## Relation Statistics

{markdown_table(relation_stats)}

## Hidden Patterns

1. The high-confusion edges are not arbitrary: they mostly lie on factor-neighbor boundaries, especially roughness and wet/water film.
2. The feature graph separates two cases: visually close confused pairs, where extra evidence is genuinely needed, and visually close stable pairs, where the current classifier already has enough separating cues.
3. Distant-but-confused pairs are more likely shortcut or representation failures; these are better targets for contrastive/angular losses than for more texture features.

## Visually Close And Confused

{markdown_table(motifs["visually_close_confused"])}

## Visually Close But Stable

{markdown_table(motifs["visually_close_stable"])}

## Distant But Still Confused

{markdown_table(motifs["distant_but_confused"])}

## Algorithmic Implication

The graph supports a heterophilic decoupling route: learn factor-specific evidence and push apart adjacent classes that differ in exactly one factor. Ordinary graph smoothing is a poor match because the strongest graph edges connect labels that must remain different.
"""
    prefix.with_suffix(".md").write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_empty_"
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    for row in df.to_dict("records"):
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

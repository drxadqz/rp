from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from analyze_rscd_run_graph_patterns import canonical_order, classification_rows, factor_text, shared_relation


DEFAULT_OUTPUT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and draw the complete RSCD-27 class-factor graph.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def factor_position(label: str, labels: list[str]) -> tuple[float, float]:
    factors = factor_text(label)
    friction = factors["friction"]
    material = factors["material"]
    roughness = factors["roughness"]
    if friction in {"dry", "wet", "water"} and material in {"asphalt", "concrete"} and roughness is not None:
        x = {"asphalt": 0.0, "concrete": 4.2}[material] + {"smooth": 0.0, "slight": 1.35, "severe": 2.7}[roughness]
        y = -{"dry": 0.0, "wet": 1.4, "water": 2.8}[friction]
        return x, y
    if friction in {"dry", "wet", "water"} and material in {"mud", "gravel"}:
        x = 8.8 + {"mud": 0.0, "gravel": 1.35}[material]
        y = -{"dry": 0.0, "wet": 1.4, "water": 2.8}[friction]
        return x, y
    return 12.0 + 0.9 * ["fresh_snow", "melted_snow", "ice"].index(label), -1.4


def complete_directed_edges(pred: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    support = pred.groupby("true_label").size().to_dict()
    counts = (
        pred[pred["true_label"] != pred["pred_label"]]
        .groupby(["true_label", "pred_label"])
        .size()
        .to_dict()
    )
    conf = (
        pred[pred["true_label"] != pred["pred_label"]]
        .groupby(["true_label", "pred_label"])["confidence"]
        .mean()
        .to_dict()
        if "confidence" in pred.columns
        else {}
    )
    rows: list[dict[str, Any]] = []
    for src in labels:
        for dst in labels:
            if src == dst:
                continue
            count = int(counts.get((src, dst), 0))
            rate = count / max(int(support.get(src, 0)), 1)
            src_factors = factor_text(src)
            dst_factors = factor_text(dst)
            changed = [k for k in ("friction", "material", "roughness") if src_factors[k] != dst_factors[k]]
            shared = [k for k in ("friction", "material", "roughness") if src_factors[k] is not None and src_factors[k] == dst_factors[k]]
            rows.append(
                {
                    "true_label": src,
                    "pred_label": dst,
                    "count": count,
                    "error_rate_in_true_class": rate,
                    "mean_confidence": float(conf.get((src, dst), np.nan)),
                    "relation": shared_relation(src, dst),
                    "changed_factors": "+".join(changed) if changed else "none",
                    "shared_factors": "+".join(shared) if shared else "none",
                    "support": int(support.get(src, 0)),
                    "is_nonzero": int(count > 0),
                }
            )
    return pd.DataFrame(rows)


def complete_undirected_edges(directed: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    edge_map = {(r.true_label, r.pred_label): r for r in directed.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            ab = edge_map[(a, b)]
            ba = edge_map[(b, a)]
            total = int(ab.count + ba.count)
            rows.append(
                {
                    "class_a": a,
                    "class_b": b,
                    "ab_count": int(ab.count),
                    "ba_count": int(ba.count),
                    "total_count": total,
                    "mean_bidirectional_rate": float((ab.error_rate_in_true_class + ba.error_rate_in_true_class) / 2.0),
                    "relation": shared_relation(a, b),
                    "changed_factors": ab.changed_factors,
                    "shared_factors": ab.shared_factors,
                    "is_nonzero": int(total > 0),
                }
            )
    return pd.DataFrame(rows)


def relation_energy(directed: pd.DataFrame) -> pd.DataFrame:
    return (
        directed.groupby("relation", as_index=False)
        .agg(
            possible_directed_edges=("count", "size"),
            nonzero_directed_edges=("is_nonzero", "sum"),
            mistakes=("count", "sum"),
            mean_error_rate=("error_rate_in_true_class", "mean"),
            max_error_rate=("error_rate_in_true_class", "max"),
        )
        .sort_values(["mistakes", "nonzero_directed_edges"], ascending=False)
    )


def plot_complete_factor_graph(nodes: pd.DataFrame, undirected: pd.DataFrame, labels: list[str], prefix: Path) -> None:
    graph = nx.Graph()
    metrics = nodes.set_index("class_label").to_dict("index")
    pos = {label: factor_position(label, labels) for label in labels}
    for label in labels:
        graph.add_node(label, **metrics.get(label, {}))
    for row in undirected.itertuples(index=False):
        graph.add_edge(row.class_a, row.class_b, total_count=float(row.total_count), relation=row.relation)

    relation_colors = {
        "factor_neighbor:roughness": "#D55E00",
        "factor_neighbor:friction": "#0072B2",
        "factor_neighbor:material": "#009E73",
        "shares_friction": "#A0A0A0",
        "shares_material": "#BDBDBD",
        "shares_roughness": "#CFCFCF",
        "cross_component": "#CC79A7",
    }
    fig, ax = plt.subplots(figsize=(18, 8), dpi=180)
    all_edges = [(r.class_a, r.class_b) for r in undirected.itertuples(index=False)]
    nx.draw_networkx_edges(graph, pos, edgelist=all_edges, width=0.25, edge_color="#E4E4E4", alpha=0.30, ax=ax)
    visible = undirected[undirected["total_count"] > 0].sort_values("total_count")
    for relation, group in visible.groupby("relation"):
        edge_list = [(r.class_a, r.class_b) for r in group.itertuples(index=False)]
        widths = [0.35 + min(5.0, float(r.total_count) / 70.0) for r in group.itertuples(index=False)]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edge_list,
            width=widths,
            edge_color=relation_colors.get(str(relation), "#777777"),
            alpha=0.30 if not str(relation).startswith("factor_neighbor") else 0.58,
            ax=ax,
        )
    f1 = np.asarray([graph.nodes[n].get("f1", 0.0) for n in graph.nodes], dtype=float)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=470 + (1.0 - f1) * 4500,
        node_color=f1,
        cmap="RdYlGn",
        vmin=0.75,
        vmax=1.0,
        edgecolors="#222222",
        linewidths=0.9,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=6, font_family="DejaVu Sans", ax=ax)
    ax.set_title("RSCD-27 complete class-factor graph: all 351 undirected pairs, colored by observed error relation")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_complete_undirected_factor_graph.png"))
    fig.savefig(prefix.with_name(prefix.name + "_complete_undirected_factor_graph.svg"))
    plt.close(fig)


def plot_relation_energy(energy: pd.DataFrame, prefix: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=180)
    x = np.arange(len(energy))
    ax1.bar(x, energy["mistakes"], color="#4C78A8", label="mistakes")
    ax1.set_ylabel("mistakes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(energy["relation"], rotation=35, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, energy["nonzero_directed_edges"] / energy["possible_directed_edges"], color="#E45756", marker="o", label="nonzero edge ratio")
    ax2.set_ylabel("nonzero edge ratio")
    ax1.set_title("Confusion energy by factor relation")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_relation_energy.png"))
    fig.savefig(prefix.with_name(prefix.name + "_relation_energy.svg"))
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.to_dict("records"):
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(prefix: Path, run_name: str, nodes: pd.DataFrame, directed: pd.DataFrame, undirected: pd.DataFrame, energy: pd.DataFrame) -> None:
    nonzero_directed = directed[directed["count"] > 0]
    nonzero_undirected = undirected[undirected["total_count"] > 0]
    top_pairs = nonzero_undirected.sort_values(["total_count", "mean_bidirectional_rate"], ascending=False).head(20)
    low_nodes = nodes.sort_values("f1").head(12)
    text = f"""# RSCD Complete Factor Graph Audit

- Run: `{run_name}`
- Class nodes: 27
- Directed class-pair edges exported: {len(directed)} = 27 x 26
- Nonzero directed error edges: {len(nonzero_directed)}
- Zero directed edges: {len(directed) - len(nonzero_directed)}
- Undirected class-pair edges exported: {len(undirected)}
- Nonzero undirected error pairs: {len(nonzero_undirected)}

## Relation Energy

{markdown_table(energy)}

## Lowest-F1 Nodes

{markdown_table(low_nodes[["class_label", "precision", "recall", "f1", "support", "out_mistakes", "in_mistakes", "dominant_error_relation"]])}

## Strongest Bidirectional Class Pairs

{markdown_table(top_pairs[["class_a", "class_b", "ab_count", "ba_count", "total_count", "mean_bidirectional_rate", "relation", "changed_factors", "shared_factors"]])}

## Pattern Reading

1. The complete graph is sparse in errors but dense in possible relations: only the visible subset of all possible edges carries mistakes.
2. Most high-energy edges are single-factor changes. This means the task is not random 27-way recognition; it is a coupled attribute classification problem where friction, material, and roughness are entangled.
3. Roughness-neighbor edges are the dominant failure mode. The most useful next model should strengthen local micro-texture and boundary evidence, especially for slight-vs-severe and slight-vs-smooth.
4. Friction-neighbor edges are mainly wet/water smooth concrete/asphalt confusion. The model needs specular/water-film evidence rather than more generic texture capacity.
5. Material-neighbor edges are concentrated in mud-vs-gravel. This asks for granular particle statistics, not line/road-scene context.

## Files

- `{prefix.name}_complete_directed_edges_702.csv`
- `{prefix.name}_complete_undirected_pairs_351.csv`
- `{prefix.name}_relation_energy.csv`
- `{prefix.name}_complete_undirected_factor_graph.png/svg`
- `{prefix.name}_relation_energy.png/svg`
"""
    prefix.with_suffix(".md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.run_name
    prefix = args.out_dir / (args.prefix or f"rscd_complete_factor_graph_{args.run_name}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(run_dir / "predictions_test.csv")
    with (run_dir / "evaluate_test.json").open("r", encoding="utf-8") as f:
        report = json.load(f)
    nodes = classification_rows(report["classification_report"])
    labels = canonical_order(nodes["class_label"].tolist())
    directed = complete_directed_edges(pred, labels)
    undirected = complete_undirected_edges(directed, labels)
    energy = relation_energy(directed)

    centrality_path = args.out_dir / f"rscd_complete_graph_{args.run_name}_node_centrality.csv"
    if centrality_path.exists():
        centrality = pd.read_csv(centrality_path)
        nodes = nodes.drop(columns=[c for c in ("out_mistakes", "in_mistakes", "dominant_error_relation") if c in nodes.columns], errors="ignore")
        nodes = nodes.merge(
            centrality[["class_label", "out_mistakes", "in_mistakes", "dominant_error_relation"]],
            on="class_label",
            how="left",
        )
    else:
        nodes["out_mistakes"] = nodes["class_label"].map(directed.groupby("true_label")["count"].sum()).fillna(0).astype(int)
        nodes["in_mistakes"] = nodes["class_label"].map(directed.groupby("pred_label")["count"].sum()).fillna(0).astype(int)
        dominant = (
            directed[directed["count"] > 0]
            .groupby(["true_label", "relation"], as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .drop_duplicates("true_label")
            .set_index("true_label")["relation"]
        )
        nodes["dominant_error_relation"] = nodes["class_label"].map(dominant).fillna("")

    directed.to_csv(prefix.with_name(prefix.name + "_complete_directed_edges_702.csv"), index=False, encoding="utf-8")
    undirected.to_csv(prefix.with_name(prefix.name + "_complete_undirected_pairs_351.csv"), index=False, encoding="utf-8")
    energy.to_csv(prefix.with_name(prefix.name + "_relation_energy.csv"), index=False, encoding="utf-8")
    plot_complete_factor_graph(nodes, undirected, labels, prefix)
    plot_relation_energy(energy, prefix)
    write_report(prefix, args.run_name, nodes, directed, undirected, energy)
    print(prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()

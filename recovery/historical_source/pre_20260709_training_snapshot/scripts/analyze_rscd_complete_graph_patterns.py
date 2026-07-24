from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


DEFAULT_RUN_NAME = "eval_semantic_attention_line_fourier_directed_conf001_fulltest"
DEFAULT_OUTPUT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RSCD nonzero directed confusion graph patterns.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def canonical_order(labels: list[str]) -> list[str]:
    ordered: list[str] = []
    for friction in ("dry", "wet", "water"):
        for material in ("asphalt", "concrete"):
            for roughness in ("smooth", "slight", "severe"):
                name = f"{friction}_{material}_{roughness}"
                if name in labels:
                    ordered.append(name)
        for material in ("mud", "gravel"):
            name = f"{friction}_{material}"
            if name in labels:
                ordered.append(name)
    for name in ("fresh_snow", "melted_snow", "ice"):
        if name in labels:
            ordered.append(name)
    for name in sorted(set(labels).difference(ordered)):
        ordered.append(name)
    return ordered


def factor_text(label: str) -> dict[str, str | None]:
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return {"friction": label, "material": None, "roughness": None}
    parts = label.split("_")
    if len(parts) == 2:
        return {"friction": parts[0], "material": parts[1], "roughness": None}
    return {"friction": parts[0], "material": parts[1], "roughness": parts[2]}


def shared_relation(src: str, dst: str) -> str:
    a = factor_text(src)
    b = factor_text(dst)
    same = [name for name in ("friction", "material", "roughness") if a[name] is not None and a[name] == b[name]]
    changed = [name for name in ("friction", "material", "roughness") if a[name] != b[name]]
    if len(changed) == 1:
        return f"factor_neighbor:{changed[0]}"
    if same:
        return "shares_" + "+".join(same)
    return "cross_component"


def classification_rows(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"} or not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "class_label": label,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1-score", 0.0)),
                "support": int(metrics.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def build_edges(pred: pd.DataFrame) -> pd.DataFrame:
    support = pred.groupby("true_label").size().rename("support")
    edges = (
        pred[pred["true_label"] != pred["pred_label"]]
        .groupby(["true_label", "pred_label"], as_index=False)
        .agg(count=("pred_label", "size"), mean_confidence=("confidence", "mean"))
    )
    edges = edges.merge(support, left_on="true_label", right_index=True, how="left")
    edges["error_rate_in_true_class"] = edges["count"] / edges["support"].clip(lower=1)
    edges["relation"] = [shared_relation(a, b) for a, b in zip(edges["true_label"], edges["pred_label"], strict=True)]
    return edges.sort_values(["count", "error_rate_in_true_class"], ascending=False)


def plot_confusion_matrix(pred: pd.DataFrame, labels: list[str], out_prefix: Path) -> None:
    idx = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for row in pred.itertuples(index=False):
        matrix[idx[row.true_label], idx[row.pred_label]] += 1.0
    row_sum = matrix.sum(axis=1, keepdims=True)
    rate = np.divide(matrix, np.maximum(row_sum, 1.0))
    np.fill_diagonal(rate, 0.0)

    pd.DataFrame(matrix.astype(int), index=labels, columns=labels).to_csv(
        out_prefix.with_name(out_prefix.name + "_confusion_counts_27x27.csv"),
        encoding="utf-8",
    )
    pd.DataFrame(rate, index=labels, columns=labels).to_csv(
        out_prefix.with_name(out_prefix.name + "_confusion_rates_27x27.csv"),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(13, 11), dpi=180)
    im = ax.imshow(rate, cmap="magma", vmin=0.0, vmax=max(0.13, float(rate.max())))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("RSCD-27 complete directed confusion matrix, diagonal hidden")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="error rate in true class")
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_confusion_matrix.png"))
    fig.savefig(out_prefix.with_name(out_prefix.name + "_confusion_matrix.svg"))
    plt.close(fig)


def plot_all_edge_graph(nodes: pd.DataFrame, edges: pd.DataFrame, labels: list[str], out_prefix: Path) -> None:
    graph = nx.DiGraph()
    node_metrics = nodes.set_index("class_label").to_dict("index")
    for label in labels:
        graph.add_node(label, **node_metrics.get(label, {}))
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.true_label,
            row.pred_label,
            count=float(row.count),
            relation=row.relation,
            rate=float(row.error_rate_in_true_class),
        )

    pos: dict[str, tuple[float, float]] = {}
    for i, label in enumerate(labels):
        factors = factor_text(label)
        friction = factors["friction"]
        material = factors["material"]
        roughness = factors["roughness"]
        if friction in {"dry", "wet", "water"} and material in {"asphalt", "concrete"}:
            fx = {"dry": 0, "wet": 1, "water": 2}[friction]
            mx = {"asphalt": 0, "concrete": 1}[material]
            rx = {"smooth": 0, "slight": 1, "severe": 2}[roughness]
            pos[label] = (mx * 4 + rx, -fx)
        elif friction in {"dry", "wet", "water"} and material in {"mud", "gravel"}:
            fx = {"dry": 0, "wet": 1, "water": 2}[friction]
            mx = {"mud": 0, "gravel": 1}[material]
            pos[label] = (9 + mx, -fx)
        else:
            pos[label] = (12 + i * 0.35, -1.0)

    relation_colors = {
        "factor_neighbor:roughness": "#D55E00",
        "factor_neighbor:friction": "#0072B2",
        "factor_neighbor:material": "#009E73",
        "shares_friction": "#999999",
        "shares_material": "#B0B0B0",
        "shares_roughness": "#BBBBBB",
        "shares_friction+material": "#777777",
        "shares_friction+roughness": "#777777",
        "shares_material+roughness": "#777777",
        "cross_component": "#CC79A7",
    }
    fig, ax = plt.subplots(figsize=(18, 8), dpi=180)
    f1 = np.asarray([graph.nodes[n].get("f1", 0.0) for n in graph.nodes], dtype=float)
    sizes = 450 + (1.0 - f1) * 4200
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=sizes,
        node_color=f1,
        cmap="RdYlGn",
        vmin=0.75,
        vmax=1.0,
        edgecolors="#222222",
        linewidths=0.8,
        ax=ax,
    )
    for relation, rel_edges in edges.groupby("relation"):
        edge_list = [(r.true_label, r.pred_label) for r in rel_edges.itertuples(index=False)]
        widths = [0.25 + min(3.5, float(r.count) / 80.0) for r in rel_edges.itertuples(index=False)]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edge_list,
            edge_color=relation_colors.get(str(relation), "#666666"),
            width=widths,
            alpha=0.22 if not str(relation).startswith("factor_neighbor") else 0.38,
            arrows=True,
            arrowsize=6,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
            ax=ax,
        )
    nx.draw_networkx_labels(graph, pos, font_size=6, font_family="DejaVu Sans", ax=ax)
    ax.set_title(f"RSCD-27 nonzero directed confusion graph: 27 nodes, {len(edges)} edges")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_all_edges_graph.png"))
    fig.savefig(out_prefix.with_name(out_prefix.name + "_all_edges_graph.svg"))
    plt.close(fig)


def reciprocal_pairs(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    edge_map = {(r.true_label, r.pred_label): r for r in edges.itertuples(index=False)}
    seen: set[tuple[str, str]] = set()
    for (a, b), row_ab in edge_map.items():
        if (a, b) in seen or (b, a) in seen or (b, a) not in edge_map:
            continue
        row_ba = edge_map[(b, a)]
        rows.append(
            {
                "class_a": a,
                "class_b": b,
                "ab_count": int(row_ab.count),
                "ba_count": int(row_ba.count),
                "total_count": int(row_ab.count + row_ba.count),
                "mean_rate": float((row_ab.error_rate_in_true_class + row_ba.error_rate_in_true_class) / 2.0),
                "relation": shared_relation(a, b),
            }
        )
        seen.add((a, b))
        seen.add((b, a))
    return pd.DataFrame(rows).sort_values(["total_count", "mean_rate"], ascending=False)


def node_centrality(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out_error = edges.groupby("true_label")["count"].sum().rename("out_mistakes")
    in_error = edges.groupby("pred_label")["count"].sum().rename("in_mistakes")
    relation_out = (
        edges.groupby(["true_label", "relation"])["count"].sum().reset_index().sort_values("count", ascending=False)
    )
    dominant_relation = relation_out.drop_duplicates("true_label").set_index("true_label")["relation"].rename("dominant_error_relation")
    table = nodes.set_index("class_label").join(out_error).join(in_error).join(dominant_relation)
    table[["out_mistakes", "in_mistakes"]] = table[["out_mistakes", "in_mistakes"]].fillna(0).astype(int)
    table["net_confusion_sink"] = table["in_mistakes"] - table["out_mistakes"]
    return table.reset_index().sort_values(["out_mistakes", "in_mistakes"], ascending=False)


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4f", max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_empty_"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.to_dict("records"):
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.run_name
    out_prefix = args.out_dir / (args.prefix or f"rscd_complete_graph_{args.run_name}")
    pred = pd.read_csv(run_dir / "predictions_test.csv")
    metrics = json.loads((run_dir / "evaluate_test.json").read_text(encoding="utf-8"))
    nodes = classification_rows(metrics["classification_report"])
    labels = canonical_order(sorted(nodes["class_label"].tolist()))
    edges = build_edges(pred)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(pred, labels, out_prefix)
    plot_all_edge_graph(nodes, edges, labels, out_prefix)

    centrality = node_centrality(nodes, edges)
    reciprocal = reciprocal_pairs(edges)
    relation_summary = (
        edges.groupby("relation", as_index=False)
        .agg(edges=("count", "size"), mistakes=("count", "sum"), mean_error_rate=("error_rate_in_true_class", "mean"))
        .sort_values("mistakes", ascending=False)
    )

    centrality.to_csv(out_prefix.with_name(out_prefix.name + "_node_centrality.csv"), index=False, encoding="utf-8")
    reciprocal.to_csv(out_prefix.with_name(out_prefix.name + "_reciprocal_pairs.csv"), index=False, encoding="utf-8")
    relation_summary.to_csv(out_prefix.with_name(out_prefix.name + "_relation_summary.csv"), index=False, encoding="utf-8")
    edges.to_csv(out_prefix.with_name(out_prefix.name + "_all_edges.csv"), index=False, encoding="utf-8")

    lines = [
        "# RSCD Complete Node-Edge Pattern Audit",
        "",
        f"- Run: `{args.run_name}`",
        f"- Test images: {len(pred)}",
        f"- Complete nodes: {len(labels)}",
        f"- Complete nonzero directed confusion edges: {len(edges)}",
        f"- Complete all-edge graph: `{out_prefix.with_name(out_prefix.name + '_all_edges_graph.png')}`",
        f"- Complete 27x27 confusion matrix: `{out_prefix.with_name(out_prefix.name + '_confusion_matrix.png')}`",
        f"- Complete edge CSV: `{out_prefix.with_name(out_prefix.name + '_all_edges.csv')}`",
        "",
        "## Relation-Level Pattern",
        "",
        markdown_table(relation_summary, floatfmt=".4f"),
        "",
        "## Most Central Error Sources",
        "",
        markdown_table(
            centrality[
                [
                    "class_label",
                    "f1",
                    "support",
                    "out_mistakes",
                    "in_mistakes",
                    "net_confusion_sink",
                    "dominant_error_relation",
                ]
            ],
            floatfmt=".4f",
            max_rows=12,
        ),
        "",
        "## Strongest Bidirectional Confusion Pairs",
        "",
        markdown_table(reciprocal, floatfmt=".4f", max_rows=15),
        "",
        "## Pattern Interpretation",
        "",
        "The graph is a heterophilic hard-negative graph. Edges mostly connect labels that share two physical factors and differ in one factor, so message passing or smoothing makes adjacent-but-different classes collapse. The strongest repeated motif is a reciprocal two-node boundary: slight versus severe roughness, wet versus water film, and mud versus gravel texture. This points to angular/prototype separation, pair-aware sampling, or boundary-aware texture evidence rather than ordinary graph smoothing.",
    ]
    out_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(out_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()

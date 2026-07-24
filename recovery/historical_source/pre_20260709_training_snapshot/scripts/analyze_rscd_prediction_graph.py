from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

RUN_NAME = "screen_physics_texture_hardboost025_lr1e5_s36k_e1_seed101_from_best"
OUTPUT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
RUN_DIR = OUTPUT_ROOT / RUN_NAME
PREDICTIONS = RUN_DIR / "predictions_test.csv"
EVALUATE = RUN_DIR / "evaluate_test.json"
GRAPH_AUDIT = Path("reports/paper_protocol_summary/rscd_label_factor_graph.json")
OUT_PREFIX = Path("reports/paper_protocol_summary/rscd_prediction_confusion_graph")


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
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(metrics, dict):
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
    return pd.DataFrame(rows).sort_values("f1")


def build_confusion_edges(pred: pd.DataFrame) -> pd.DataFrame:
    total_by_true = pred.groupby("true_label").size().rename("support")
    edges = (
        pred[pred["true_label"] != pred["pred_label"]]
        .groupby(["true_label", "pred_label"], as_index=False)
        .agg(count=("pred_label", "size"), mean_confidence=("confidence", "mean"))
    )
    edges = edges.merge(total_by_true, left_on="true_label", right_index=True, how="left")
    edges["error_rate_in_true_class"] = edges["count"] / edges["support"].clip(lower=1)
    edges["relation"] = [shared_relation(src, dst) for src, dst in zip(edges["true_label"], edges["pred_label"], strict=True)]
    return edges.sort_values(["count", "error_rate_in_true_class"], ascending=False)


def draw_confusion_graph(nodes: pd.DataFrame, edges: pd.DataFrame, label_edges: list[dict[str, str]]) -> None:
    graph = nx.DiGraph()
    for row in nodes.to_dict("records"):
        graph.add_node(row["class_label"], f1=row["f1"], support=row["support"])
    for edge in label_edges:
        graph.add_edge(edge["source"], edge["target"], kind=edge["relation"], weight=0.15)
        graph.add_edge(edge["target"], edge["source"], kind=edge["relation"], weight=0.15)

    top_edges = edges.head(70)
    for row in top_edges.to_dict("records"):
        graph.add_edge(row["true_label"], row["pred_label"], kind="confusion", weight=max(0.5, row["count"] / 28.0))

    pos = nx.spring_layout(graph, seed=7, k=1.6, weight="weight", iterations=300)
    plt.figure(figsize=(18, 13), dpi=180)
    f1 = np.asarray([graph.nodes[n].get("f1", 0.0) for n in graph.nodes], dtype=float)
    sizes = 450 + (1.0 - f1) * 5200
    colors = f1
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=sizes,
        node_color=colors,
        cmap="RdYlGn",
        vmin=0.72,
        vmax=1.0,
        linewidths=0.9,
        edgecolors="#303030",
    )

    label_edge_pairs = [(e["source"], e["target"]) for e in label_edges] + [(e["target"], e["source"]) for e in label_edges]
    confusion_pairs = [(r["true_label"], r["pred_label"]) for r in top_edges.to_dict("records")]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=label_edge_pairs,
        edge_color="#B0B0B0",
        width=0.8,
        alpha=0.35,
        arrows=False,
    )
    widths = [1.0 + min(5.0, float(r["count"]) / 55.0) for r in top_edges.to_dict("records")]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=confusion_pairs,
        edge_color="#C43C39",
        width=widths,
        alpha=0.72,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=11,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_labels(graph, pos, font_size=7, font_family="DejaVu Sans")
    plt.title("RSCD-27 label graph plus top directed model-confusion edges", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PREFIX.with_suffix(".png"))
    plt.savefig(OUT_PREFIX.with_suffix(".svg"))
    plt.close()


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
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
    pred = pd.read_csv(PREDICTIONS)
    metrics = json.loads(EVALUATE.read_text(encoding="utf-8"))
    graph_audit = json.loads(GRAPH_AUDIT.read_text(encoding="utf-8"))
    nodes = classification_rows(metrics["classification_report"])
    edges = build_confusion_edges(pred)

    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_nodes.csv"), index=False, encoding="utf-8")
    edges.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_all_confusion_edges.csv"), index=False, encoding="utf-8")
    draw_confusion_graph(nodes, edges, graph_audit["edges"])

    relation_summary = (
        edges.groupby("relation", as_index=False)
        .agg(edges=("count", "size"), mistakes=("count", "sum"), mean_error_rate=("error_rate_in_true_class", "mean"))
        .sort_values("mistakes", ascending=False)
    )
    relation_summary.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_relation_summary.csv"), index=False, encoding="utf-8")

    hardest = nodes.head(8).copy()
    top_errors = edges.head(18).copy()
    lines = [
        "# RSCD Prediction Confusion Graph",
        "",
        f"- Run: `{RUN_NAME}`",
        f"- Samples: {len(pred)}",
        f"- Nodes: {len(nodes)} classes",
        f"- Nonzero directed confusion edges: {len(edges)}",
        f"- PNG: `{OUT_PREFIX.with_suffix('.png')}`",
        f"- SVG: `{OUT_PREFIX.with_suffix('.svg')}`",
        f"- Complete edge CSV: `{OUT_PREFIX.with_name(OUT_PREFIX.name + '_all_confusion_edges.csv')}`",
        "",
        "## Main Pattern",
        "",
        "- The dominant weak region is not random: errors concentrate around water/wet, asphalt/concrete, and slight/severe boundaries.",
        "- Most useful graph edges are heterophilic discriminative edges: adjacent labels are visually close but semantically different, so naive smoothing is harmful.",
        "- The graph should be used as a hard-negative/margin/calibration structure, not as a homophilic message-passing prior.",
        "",
        "## Relation Summary",
        "",
        markdown_table(relation_summary, floatfmt=".4f"),
        "",
        "## Hardest Nodes",
        "",
        markdown_table(hardest, floatfmt=".4f"),
        "",
        "## Largest Directed Confusion Edges",
        "",
        markdown_table(top_errors, floatfmt=".4f"),
        "",
        "## Method Implication",
        "",
        "Promote PhysicsTexture as the main strict model. Keep graph reasoning as a protected pairwise calibration or hard-negative supervision idea, because trainable graph residuals improved water_concrete_slight but reduced global wet/water F1.",
    ]
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "run": RUN_NAME,
        "num_samples": int(len(pred)),
        "num_nodes": int(len(nodes)),
        "num_nonzero_confusion_edges": int(len(edges)),
        "top_confusion_edges": top_errors.to_dict("records"),
        "relation_summary": relation_summary.to_dict("records"),
    }
    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT_PREFIX.with_suffix(".md"))


if __name__ == "__main__":
    main()

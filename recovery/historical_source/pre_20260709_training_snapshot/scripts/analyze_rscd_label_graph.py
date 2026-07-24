from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_RUN = "screen_physics_texture_hardboost025_lr1e5_s36k_e1_seed101_from_best"
DEFAULT_ALT_RUNS = [
    "screen_physics_texture_graph_diffusion_s002_t035_core_s8k_from_best",
    "screen_physics_texture_local_factor_graph_w001_m010_s8k_from_best",
    "screen_physics_texture_local_factor_graph_w0002_m005_s8k_from_best",
    "tta_ensemble_physics_texture_cd_owr_hardboost025_hflip",
]
DEFAULT_OUT = Path("reports/paper_protocol_summary/rscd_label_factor_graph")


FRICTION_ORDER = {
    "dry": 0,
    "wet": 1,
    "water": 2,
    "fresh_snow": 3,
    "melted_snow": 4,
    "ice": 5,
}
MATERIAL_ORDER = {"asphalt": 0, "concrete": 1, "gravel": 2, "mud": 3, None: 4}
ROUGHNESS_ORDER = {"smooth": 0, "slight": 1, "severe": 2, None: 1}
FRICTION_COLORS = {
    "dry": "#d9b35f",
    "wet": "#4197d8",
    "water": "#1f4e99",
    "fresh_snow": "#dfefff",
    "melted_snow": "#98cce6",
    "ice": "#b9d6ff",
}
EDGE_COLORS = {"friction": "#1f77b4", "material": "#2ca02c", "roughness": "#d62728"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and draw the complete RSCD-27 label factor graph.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--alt-run", action="append", default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    main_payload = load_eval(args.root / args.run / "evaluate_test.json")
    class_names = sorted(classification_rows(main_payload))
    alt_names = list(dict.fromkeys(list(DEFAULT_ALT_RUNS) + list(args.alt_run)))
    alt_payloads = {
        name: load_eval(args.root / name / "evaluate_test.json")
        for name in alt_names
        if (args.root / name / "evaluate_test.json").exists()
    }
    graph = build_graph(class_names)
    node_rows = build_node_rows(graph, main_payload, alt_payloads)
    edge_rows = build_edge_rows(graph)
    communities = factor_communities(graph)
    out_payload = {
        "run": args.run,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "edge_type_counts": edge_type_counts(graph),
        "communities": communities,
        "nodes": node_rows,
        "edges": edge_rows,
        "observations": summarize(node_rows, edge_rows, communities),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_graph(graph, node_rows, args.out.with_suffix(".png"))
    draw_graph(graph, node_rows, args.out.with_suffix(".svg"))
    args.out.with_suffix(".md").write_text(to_markdown(out_payload, args.out), encoding="utf-8")
    print(args.out.with_suffix(".md"))


def load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classification_rows(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    report = payload.get("classification_report", {})
    rows = {}
    for name, item in report.items():
        if str(name) in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if isinstance(item, dict) and "f1-score" in item:
            rows[str(name)] = {
                "precision": float(item.get("precision") or 0.0),
                "recall": float(item.get("recall") or 0.0),
                "f1": float(item.get("f1-score") or 0.0),
                "support": float(item.get("support") or 0.0),
            }
    return rows


def factors(label: str) -> dict[str, str | None]:
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return {"friction": label, "material": None, "roughness": None}
    parts = label.split("_")
    return {
        "friction": parts[0] if len(parts) > 0 else None,
        "material": parts[1] if len(parts) > 1 else None,
        "roughness": parts[2] if len(parts) > 2 else None,
    }


def friction_neighbors(a: str | None, b: str | None) -> bool:
    return frozenset((a, b)) in {
        frozenset(("dry", "wet")),
        frozenset(("wet", "water")),
        frozenset(("fresh_snow", "melted_snow")),
        frozenset(("melted_snow", "ice")),
    }


def material_neighbors(a: str | None, b: str | None) -> bool:
    return frozenset((a, b)) in {
        frozenset(("asphalt", "concrete")),
        frozenset(("mud", "gravel")),
    }


def roughness_neighbors(a: str | None, b: str | None) -> bool:
    return frozenset((a, b)) in {
        frozenset(("smooth", "slight")),
        frozenset(("slight", "severe")),
    }


def relation_type(a_label: str, b_label: str) -> str | None:
    a = factors(a_label)
    b = factors(b_label)
    same_f = a["friction"] is not None and a["friction"] == b["friction"]
    same_m = a["material"] is not None and a["material"] == b["material"]
    same_r = a["roughness"] is not None and a["roughness"] == b["roughness"]
    none_m = a["material"] is None and b["material"] is None
    none_r = a["roughness"] is None and b["roughness"] is None
    if (same_m or none_m) and (same_r or none_r) and friction_neighbors(a["friction"], b["friction"]):
        return "friction"
    if same_f and (same_r or none_r) and material_neighbors(a["material"], b["material"]):
        return "material"
    if same_f and same_m and roughness_neighbors(a["roughness"], b["roughness"]):
        return "roughness"
    return None


def build_graph(class_names: list[str]) -> nx.Graph:
    graph = nx.Graph()
    for name in class_names:
        graph.add_node(name, **factors(name))
    for i, a in enumerate(class_names):
        for b in class_names[i + 1 :]:
            rel = relation_type(a, b)
            if rel:
                graph.add_edge(a, b, relation=rel)
    return graph


def build_node_rows(
    graph: nx.Graph,
    payload: dict[str, Any],
    alt_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = classification_rows(payload)
    alt_rows = {name: classification_rows(p) for name, p in alt_payloads.items()}
    out = []
    for node in sorted(graph.nodes()):
        row = {"class_label": node, **factors(node), **rows.get(node, {})}
        for alt_name, alt in alt_rows.items():
            if node in alt:
                row[f"delta_f1_vs_{alt_name}"] = alt[node]["f1"] - row.get("f1", 0.0)
        out.append(row)
    return out


def build_edge_rows(graph: nx.Graph) -> list[dict[str, Any]]:
    return [
        {"source": a, "target": b, "relation": data["relation"]}
        for a, b, data in sorted(graph.edges(data=True), key=lambda x: (x[2]["relation"], x[0], x[1]))
    ]


def edge_type_counts(graph: nx.Graph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        counts[data["relation"]] = counts.get(data["relation"], 0) + 1
    return dict(sorted(counts.items()))


def factor_communities(graph: nx.Graph) -> dict[str, list[str]]:
    return {
        "connected_components": [sorted(c) for c in nx.connected_components(graph)],
        "articulation_points": sorted(nx.articulation_points(graph)),
    }


def node_position(name: str) -> tuple[float, float]:
    f = factors(name)
    friction = f["friction"]
    material = f["material"]
    roughness = f["roughness"]
    x = float(FRICTION_ORDER.get(friction, 6))
    y = -float(MATERIAL_ORDER.get(material, 4)) * 1.35
    y += (float(ROUGHNESS_ORDER.get(roughness, 1)) - 1.0) * 0.28
    if material is None:
        y = 0.55 - float(FRICTION_ORDER.get(friction, 3)) * 0.42
        x = 3.35 + 0.55 * float(FRICTION_ORDER.get(friction, 3) - 3)
    return (x, y)


def draw_graph(graph: nx.Graph, node_rows: list[dict[str, Any]], out_path: Path) -> None:
    row_by_name = {row["class_label"]: row for row in node_rows}
    pos = {name: node_position(name) for name in graph.nodes()}
    fig, ax = plt.subplots(figsize=(15, 9), dpi=180)
    for rel, color in EDGE_COLORS.items():
        edges = [(a, b) for a, b, data in graph.edges(data=True) if data["relation"] == rel]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=color, width=1.4, alpha=0.62, ax=ax)
    node_colors = [FRICTION_COLORS.get(factors(n)["friction"], "#cccccc") for n in graph.nodes()]
    node_sizes = []
    for n in graph.nodes():
        f1 = float(row_by_name.get(n, {}).get("f1", 0.0))
        node_sizes.append(260 + 900 * max(0.0, min(1.0, f1)))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#222222",
        linewidths=0.8,
        ax=ax,
    )
    labels = {}
    for n in graph.nodes():
        short = n.replace("_asphalt", "_asp").replace("_concrete", "_conc")
        labels[n] = f"{short}\nF1={row_by_name[n]['f1']*100:.1f}"
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6.7, font_family="DejaVu Sans", ax=ax)
    ax.set_title("RSCD-27 Label Factor Graph: Nodes are compositional classes, edges differ by one factor", fontsize=12)
    ax.text(
        0.01,
        -0.045,
        "Blue edge: neighboring friction state; Green edge: neighboring material; Red edge: neighboring roughness. "
        "Node color encodes friction state; node size follows current best per-class F1.",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize(
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    communities: dict[str, Any],
) -> list[str]:
    worst = sorted(node_rows, key=lambda r: r.get("f1", 0.0))[:6]
    water_concrete = [r for r in node_rows if r["friction"] == "water" and r["material"] == "concrete"]
    rough_slight = [r for r in node_rows if r["roughness"] == "slight"]
    relation_counts: dict[str, int] = {}
    for edge in edge_rows:
        relation_counts[edge["relation"]] = relation_counts.get(edge["relation"], 0) + 1
    return [
        "The graph separates into one large dry/wet/water material-roughness component and one winter chain.",
        "The weakest nodes cluster on the water/wet concrete and slight/severe boundary, not randomly across all 27 labels.",
        f"Edge counts are {relation_counts}; roughness and friction edges dominate the actionable hard-neighbor structure.",
        "A broad graph loss is risky because water-related nodes have many adjacent friction/material/roughness edges; local improvements can lower global wet/water recall.",
        "A better next module should gate graph reasoning at the node/pair level and protect wet/water margins rather than smoothing every neighbor.",
        "Worst nodes: " + ", ".join(f"{r['class_label']}={r.get('f1', 0.0)*100:.2f}%" for r in worst),
        "Water-concrete mean F1: " + f"{np.mean([r.get('f1', 0.0) for r in water_concrete])*100:.2f}%",
        "Slight-roughness mean F1: " + f"{np.mean([r.get('f1', 0.0) for r in rough_slight])*100:.2f}%",
        f"Connected components: {len(communities['connected_components'])}; articulation points: {communities['articulation_points']}",
    ]


def to_markdown(payload: dict[str, Any], out_base: Path) -> str:
    lines = [
        "# RSCD-27 Label Factor Graph Audit",
        "",
        f"- Run: `{payload['run']}`",
        f"- Nodes: {payload['num_nodes']}",
        f"- Edges: {payload['num_edges']}",
        f"- Edge type counts: `{payload['edge_type_counts']}`",
        f"- Figure PNG: `{out_base.with_suffix('.png')}`",
        f"- Figure SVG: `{out_base.with_suffix('.svg')}`",
        "",
        "## Observations",
        "",
    ]
    for item in payload["observations"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Nodes",
        "",
        "| class | friction | material | roughness | F1 | precision | recall |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(payload["nodes"], key=lambda r: r.get("f1", 0.0)):
        lines.append(
            "| {class_label} | {friction} | {material} | {roughness} | {f1:.2%} | {precision:.2%} | {recall:.2%} |".format(
                class_label=row["class_label"],
                friction=row.get("friction"),
                material=row.get("material"),
                roughness=row.get("roughness"),
                f1=float(row.get("f1", 0.0)),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
            )
        )
    lines += [
        "",
        "## Edges",
        "",
        "| source | target | relation |",
        "|---|---|---|",
    ]
    for edge in payload["edges"]:
        lines.append(f"| {edge['source']} | {edge['target']} | {edge['relation']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

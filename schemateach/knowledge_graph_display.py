from __future__ import annotations

import argparse
import pathlib
import matplotlib.pyplot as plt
import networkx as nx

from .graph import KnowledgeGraph


def _build_nx_graph(kg: KnowledgeGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    for node in kg.nodes.values():
        g.add_node(
            node.node_id,
            label=f"{node.section_id} {node.title}",
            pdf=node.pdf_id,
        )
    for edge in kg.edges:
        g.add_edge(edge.source, edge.target, edge_type=edge.edge_type)
    return g


def display_knowledge_graph(graph_path: str, save_path: str | None = None, node_size: int = 1400) -> None:
    kg = KnowledgeGraph.load(pathlib.Path(graph_path))
    if not kg.nodes:
        raise ValueError(f"No nodes found in graph file: {graph_path}")

    g = _build_nx_graph(kg)
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(g, seed=42, k=0.5, iterations=50)
    node_labels = nx.get_node_attributes(g, "label")
    edge_labels = nx.get_edge_attributes(g, "edge_type")

    nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color="#8fbbe8", edgecolors="#1f4b7f", linewidths=1.0)
    nx.draw_networkx_edges(g, pos, arrowstyle="->", arrowsize=12, edge_color="#555")
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7, label_pos=0.4)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display SchemaTeach knowledge graph")
    parser.add_argument("--graph", type=str, default="graph/graph.jsonl", help="Path to graph.jsonl")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save the figure (png)")
    parser.add_argument("--node-size", type=int, default=1400, help="Node size for visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    display_knowledge_graph(args.graph, save_path=args.out, node_size=args.node_size)


if __name__ == "__main__":
    main()

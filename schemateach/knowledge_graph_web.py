from __future__ import annotations

import argparse
import pathlib
from typing import Optional

from pyvis.network import Network

from .graph import KnowledgeGraph


def build_web_graph(
    graph_path: str,
    out_path: str = "graph.html",
    max_depth: Optional[int] = None,
    physics: bool = True,
) -> None:
    kg = KnowledgeGraph.load(pathlib.Path(graph_path))
    if not kg.nodes:
        raise ValueError(f"No nodes found in graph file: {graph_path}")

    net = Network(height="100%", width="100%", directed=True, notebook=False, cdn_resources="remote")
    net.toggle_physics(physics)
    net.set_options(
        """
{
  "interaction": { "hover": true, "navigationButtons": true, "multiselect": true },
  "physics": { "solver": "forceAtlas2Based", "stabilization": { "iterations": 150 } },
  "edges": { "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 }}, "color": "#666" },
  "nodes": { "shape": "dot", "borderWidth": 1, "color": { "background": "#8fbbe8", "border": "#1f4b7f" } }
}
"""
    )

    for node in kg.nodes.values():
        depth = node.section_id.count(".") + 1
        if max_depth is not None and depth > max_depth:
            continue
        label = f"{node.section_id} {node.title}".strip()
        summary = node.summary.replace("\n", " ").strip()[:400]
        prereq = ", ".join(node.prerequisites) if node.prerequisites else "None"
        tooltip = f"<b>{label}</b><br/>PDF: {node.pdf_id}<br/>Prereqs: {prereq}<br/>{summary}"
        net.add_node(
            node.node_id,
            label=label,
            title=tooltip,
            group=str(depth),
            value=depth,
        )

    for edge in kg.edges:
        if max_depth is not None:
            src_depth = kg.nodes.get(edge.source, None)
            tgt_depth = kg.nodes.get(edge.target, None)
            if not src_depth or not tgt_depth:
                continue
            if src_depth.section_id.count(".") + 1 > max_depth or tgt_depth.section_id.count(".") + 1 > max_depth:
                continue
        net.add_edge(edge.source, edge.target, title=edge.edge_type, label=edge.edge_type)

    net.show(out_path)
    print(f"Saved web graph to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive web visualization of SchemaTeach knowledge graph")
    parser.add_argument("--graph", type=str, default="graph/graph.jsonl", help="Path to graph.jsonl")
    parser.add_argument("--out", type=str, default="graph.html", help="Output HTML path")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional max depth of sections to include")
    parser.add_argument("--no-physics", action="store_true", help="Disable physics for a static layout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_web_graph(
        graph_path=args.graph,
        out_path=args.out,
        max_depth=args.max_depth,
        physics=not args.no_physics,
    )


if __name__ == "__main__":
    main()

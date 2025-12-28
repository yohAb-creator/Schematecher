from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class GraphNode:
    node_id: str
    pdf_id: str
    section_id: str
    title: str
    summary: str
    topics: List[str]
    centroid: List[float]
    prerequisites: List[str]
    path: str

    @classmethod
    def from_dict(cls, data: Dict) -> "GraphNode":
        return cls(
            node_id=data["node_id"],
            pdf_id=data["pdf_id"],
            section_id=data["section_id"],
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            topics=data.get("topics", []),
            centroid=data.get("centroid", []),
            prerequisites=data.get("prerequisites", []),
            path=data["path"],
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    def centroid_array(self) -> np.ndarray:
        return np.array(self.centroid, dtype=np.float32)


@dataclass
class GraphEdge:
    edge_type: str
    source: str
    target: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KnowledgeGraph:
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]

    @classmethod
    def empty(cls) -> "KnowledgeGraph":
        return cls(nodes={}, edges=[])

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for node in self.nodes.values():
                f.write(json.dumps({"type": "node", **node.to_dict()}) + "\n")
            for edge in self.edges:
                f.write(json.dumps({"type": "edge", **edge.to_dict()}) + "\n")

    @classmethod
    def load(cls, path: pathlib.Path) -> "KnowledgeGraph":
        kg = cls.empty()
        if not path.exists():
            return kg
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "node":
                    kg.add_node(GraphNode.from_dict(data))
                elif data.get("type") == "edge":
                    kg.add_edge(GraphEdge(edge_type=data["edge_type"], source=data["source"], target=data["target"]))
        return kg

    def prerequisites_of(self, node_id: str) -> List[str]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return node.prerequisites

    def neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        out = []
        for edge in self.edges:
            if edge.source == node_id and (edge_type is None or edge.edge_type == edge_type):
                out.append(edge.target)
        return out


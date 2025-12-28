from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .embeddings import Embedder
from .graph import KnowledgeGraph


def route_nodes(
    query: str,
    kg: KnowledgeGraph,
    embedder: Embedder,
    top_k: int = 5,
    w_semantic: float = 1.0,
    w_prereq: float = 0.2,
) -> List[Tuple[str, float]]:
    if not kg.nodes:
        return []
    query_vec = embedder.embed([query])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    scores: Dict[str, float] = {}
    for node_id, node in kg.nodes.items():
        centroid = node.centroid_array()
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sem = float(centroid @ query_vec)
        prereq_bonus = w_prereq * len(node.prerequisites)
        scores[node_id] = w_semantic * sem + prereq_bonus
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]


from __future__ import annotations

from typing import Dict, List, Tuple

import re

import numpy as np

from .embeddings import Embedder
from .graph import KnowledgeGraph
from .index import GlobalChunkIndex


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_\\-]+", text.lower())


def route_nodes(
    query: str,
    kg: KnowledgeGraph,
    embedder: Embedder,
    top_k: int = 5,
    w_semantic: float = 1.0,
    w_prereq: float = 0.2,
    global_index: GlobalChunkIndex | None = None,
    global_chunk_top_k: int = 30,
    w_chunk: float = 0.7,
    w_lex: float = 0.2,
    concept_map: Dict[str, List[str]] | None = None,
    w_concept: float = 0.6,
) -> List[Tuple[str, float]]:
    if not kg.nodes:
        return []
    query_vec = embedder.embed([query])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    scores: Dict[str, float] = {}
    query_tokens = _tokenize(query)
    for node_id, node in kg.nodes.items():
        centroid = node.centroid_array()
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sem = float(centroid @ query_vec)
        prereq_bonus = w_prereq * len(node.prerequisites)
        # Lexical overlap on node summary/title as a sparse backstop
        node_text = f"{node.title} {node.summary}".lower()
        lex_hits = sum(1 for tok in query_tokens if tok in node_text) if query_tokens else 0
        lex_score = (lex_hits / max(1, len(query_tokens))) if query_tokens else 0.0
        scores[node_id] = w_semantic * sem + prereq_bonus + (w_lex * lex_score)

    if global_index is not None:
        hits = global_index.search(query_vec, top_k=global_chunk_top_k)
        for score, meta in hits:
            scores[meta.node_id] = scores.get(meta.node_id, 0.0) + (w_chunk * score)

    if concept_map:
        query_lower = query.lower()
        for term, nodes in concept_map.items():
            if term and term in query_lower:
                for node_id in nodes:
                    scores[node_id] = scores.get(node_id, 0.0) + w_concept
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]

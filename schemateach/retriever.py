from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pathlib

from .embeddings import Embedder
from .graph import KnowledgeGraph
from .storage import VectorStore


def retrieve_chunks(
    query: str,
    kg: KnowledgeGraph,
    node_ids: List[str],
    embedder: Embedder,
    top_k: int = 6,
) -> List[Tuple[float, str, str, str]]:
    query_vec = embedder.embed([query])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    results: List[Tuple[float, str, str, str, str]] = []  # score, pdf, section, chunk_id, text
    for node_id in node_ids:
        node = kg.nodes[node_id]
        vs = VectorStore(pathlib.Path(node.path))
        vs.load()
        for score, chunk in vs.search(query_vec, top_k=top_k):
            results.append((score, node.pdf_id, node.section_id, chunk.chunk_id, chunk.text))

    # Deduplicate identical chunk_ids and texts, keep highest score
    best_by_chunk: Dict[str, Tuple[float, str, str, str]] = {}
    for score, pdf, section, chunk_id, text in results:
        key = f"{pdf}:{section}:{chunk_id}"
        if key not in best_by_chunk or score > best_by_chunk[key][0]:
            best_by_chunk[key] = (score, pdf, section, text)

    deduped = list(best_by_chunk.values())
    deduped.sort(key=lambda x: -x[0])
    return deduped[: top_k * max(1, len(node_ids))]


def expand_nodes(kg: KnowledgeGraph, seeds: List[str], hops: int = 1) -> List[str]:
    seen = set(seeds)
    frontier = list(seeds)
    for _ in range(hops):
        new_frontier = []
        for node_id in frontier:
            for neigh in kg.neighbors(node_id, edge_type="prereq"):
                if neigh not in seen and neigh in kg.nodes:
                    seen.add(neigh)
                    new_frontier.append(neigh)
        frontier = new_frontier
        if not frontier:
            break
    return list(seeds) + [n for n in seen if n not in seeds]


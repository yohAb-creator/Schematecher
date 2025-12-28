from __future__ import annotations

import argparse

from .config import Config
from .embeddings import Embedder
from .graph import KnowledgeGraph
from .retriever import expand_nodes, retrieve_chunks
from .router import route_nodes
from .synthesizer import synthesize_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SchemaTeach query runner")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--node-top-k", type=int, default=None)
    parser.add_argument("--chunk-top-k", type=int, default=None)
    parser.add_argument("--llm-model", type=str, default=None, help="Override LLM model (e.g., google/flan-t5-large)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_file(args.config)
    embedder = Embedder(cfg.embedding_provider, cfg.embedding_model, cfg.embedding_dims)
    kg_path = cfg.graph_dir / "graph.jsonl"
    kg = KnowledgeGraph.load(kg_path)
    node_top_k = args.node_top_k or cfg.node_top_k
    chunk_top_k = args.chunk_top_k or cfg.chunk_top_k

    routed = route_nodes(
        args.query,
        kg,
        embedder=embedder,
        top_k=node_top_k,
        w_semantic=cfg.w_semantic,
        w_prereq=cfg.w_prereq,
    )
    node_ids = [nid for nid, _ in routed]
    expanded = expand_nodes(kg, seeds=node_ids, hops=cfg.expansion_hops)
    chunks = retrieve_chunks(args.query, kg, node_ids=expanded, embedder=embedder, top_k=chunk_top_k)
    answer = synthesize_answer(
        args.query,
        chunks,
        provider=cfg.llm_provider,
        model=args.llm_model or cfg.llm_model,
        max_tokens=cfg.llm_max_tokens,
    )
    print("=== Routed Nodes ===")
    for node_id, score in routed:
        print(f"{node_id}: {score:.3f}")
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()

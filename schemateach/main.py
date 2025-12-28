from __future__ import annotations

import argparse

from .config import Config
from .embeddings import Embedder
from .graph import KnowledgeGraph
from .index import GlobalChunkIndex, load_concept_map
from .retriever import expand_nodes, prereq_paths, retrieve_chunks
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
    global_index = GlobalChunkIndex.load(cfg.graph_dir)
    concept_map = load_concept_map(cfg.graph_dir)
    node_top_k = args.node_top_k or cfg.node_top_k
    chunk_top_k = args.chunk_top_k or cfg.chunk_top_k
    model = (args.llm_model or cfg.llm_model).strip()
    provider = cfg.llm_provider
    lower_model = model.lower()
    if lower_model.startswith("openai:"):
        provider = "openai"
        model = model.split(":", 1)[1]
    elif lower_model.startswith("gpt-"):
        provider = "openai"

    routed = route_nodes(
        args.query,
        kg,
        embedder=embedder,
        top_k=node_top_k,
        w_semantic=cfg.w_semantic,
        w_prereq=cfg.w_prereq,
        global_index=global_index,
        global_chunk_top_k=cfg.global_chunk_top_k,
        w_chunk=cfg.w_chunk,
        w_lex=cfg.w_lex,
        concept_map=concept_map,
        w_concept=cfg.w_concept,
    )
    node_ids = [nid for nid, _ in routed]
    expanded = expand_nodes(kg, seeds=node_ids, hops=cfg.expansion_hops)
    paths = prereq_paths(kg, seeds=node_ids, max_hops=cfg.graph_max_hops, max_paths=cfg.graph_path_top_k)
    chunks = retrieve_chunks(args.query, kg, node_ids=expanded, embedder=embedder, top_k=chunk_top_k)
    answer = synthesize_answer(
        args.query,
        chunks,
        provider=provider,
        model=model,
        max_tokens=cfg.llm_max_tokens,
        reasoning_paths=paths,
    )
    print("=== Routed Nodes ===")
    for node_id, score in routed:
        print(f"{node_id}: {score:.3f}")
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import re
import json
import pathlib
import os

from .config import Config
from .embeddings import Embedder
from .graph import KnowledgeGraph
from .index import GlobalChunkIndex, load_concept_map
from .retriever import expand_nodes, prereq_paths, retrieve_chunks
from .router import route_nodes
from .synthesizer import synthesize_answer


class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    node_top_k: Optional[int] = None
    chunk_top_k: Optional[int] = None


class EvidenceItem(BaseModel):
    id: str
    title: str
    content: str
    source: str


class GraphNodeOut(BaseModel):
    id: str
    label: str
    color: Optional[str] = None


class GraphEdgeOut(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = ConfigDict(populate_by_name=True)


class GraphDataOut(BaseModel):
    nodes: List[GraphNodeOut]
    edges: List[GraphEdgeOut]


class QueryResponse(BaseModel):
    response: str
    evidence: List[EvidenceItem]
    graphData: GraphDataOut
    reasoningPaths: List[List[str]] = []


app = FastAPI(title="SchemaTeach API")


@lru_cache(maxsize=1)
def _load_state() -> tuple[Config, Embedder, KnowledgeGraph, GlobalChunkIndex, dict]:
    cfg = Config.from_file("config.yaml")
    embedder = Embedder(cfg.embedding_provider, cfg.embedding_model, cfg.embedding_dims)
    kg = KnowledgeGraph.load(cfg.graph_dir / "graph.jsonl")
    global_index = GlobalChunkIndex.load(cfg.graph_dir)
    concept_map = load_concept_map(cfg.graph_dir)
    return cfg, embedder, kg, global_index, concept_map


def _color_for_depth(section_id: str) -> str:
    depth = section_id.count(".")
    palette = ["#3b82f6", "#8b5cf6", "#ec4899", "#22c55e", "#f59e0b"]
    return palette[min(depth, len(palette) - 1)]


def _extract_query_term(query: str) -> str | None:
    q = query.strip().lower()
    q = re.sub(r"^(what is|what's|define|definition of)\\s+", "", q)
    q = re.sub(r"[\\?\\.\\!]+$", "", q).strip()
    return q or None


def _match_concept_nodes(term: str, concept_map: dict) -> List[str]:
    matches: List[str] = []
    if not term:
        return matches
    for concept, nodes in concept_map.items():
        if len(concept) < 3:
            continue
        if term in concept or concept in term:
            matches.extend(nodes)
    return list(dict.fromkeys(matches))


def _definition_from_manifest(node: GraphNode) -> str | None:
    manifest_path = pathlib.Path(node.path) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    defs = manifest.get("definitions", [])
    if defs:
        return defs[0]
    return None


def _build_graph_data(kg: KnowledgeGraph, node_ids: List[str]) -> GraphDataOut:
    nodes_out = []
    for node_id in node_ids:
        node = kg.nodes.get(node_id)
        if not node:
            continue
        nodes_out.append(
            GraphNodeOut(
                id=node.node_id,
                label=f"{node.section_id} {node.title}".strip(),
                color=_color_for_depth(node.section_id),
            )
        )
    edges_out = []
    for edge in kg.edges:
        if edge.source in node_ids and edge.target in node_ids:
            edges_out.append(GraphEdgeOut(from_=edge.source, to=edge.target))
    return GraphDataOut(nodes=nodes_out, edges=edges_out)


def _resolve_provider(model: str | None, default_provider: str) -> tuple[str, str | None]:
    if not model:
        return default_provider, model
    clean = model.strip()
    lower = clean.lower()
    if lower.startswith("openai:"):
        return "openai", clean.split(":", 1)[1]
    if lower.startswith("gpt-"):
        return "openai", clean
    return default_provider, clean


@app.post("/query", response_model=QueryResponse, response_model_by_alias=True)
def query(req: QueryRequest) -> QueryResponse:
    cfg, embedder, kg, global_index, concept_map = _load_state()
    node_top_k = req.node_top_k or cfg.node_top_k
    chunk_top_k = req.chunk_top_k or cfg.chunk_top_k
    provider, model = _resolve_provider(req.model, cfg.llm_provider)
    lower_q = req.query.lower()
    wants_def = any(phrase in lower_q for phrase in ("what is", "define", "definition", "what's"))

    if wants_def:
        term = _extract_query_term(req.query)
        matched_nodes = _match_concept_nodes(term or "", concept_map)
        if matched_nodes:
            expanded = expand_nodes(kg, seeds=matched_nodes, hops=cfg.expansion_hops)
            paths = prereq_paths(kg, seeds=matched_nodes, max_hops=cfg.graph_max_hops, max_paths=cfg.graph_path_top_k)
            evidence: List[EvidenceItem] = []
            response_text = None
            for idx, node_id in enumerate(matched_nodes[:3], start=1):
                node = kg.nodes.get(node_id)
                if not node:
                    continue
                definition = _definition_from_manifest(node)
                if definition:
                    if response_text is None:
                        response_text = f"{definition} [{node.pdf_id}:{node.section_id}]"
                    evidence.append(
                        EvidenceItem(
                            id=f"{node.pdf_id}:{node.section_id}:{idx}",
                            title=f"{node.section_id} ({node.pdf_id})",
                            content=definition,
                            source=node.pdf_id,
                        )
                    )
            if response_text is None:
                response_text = f"No definition text found for '{term}'."
            graph_out = _build_graph_data(kg, expanded)
            return QueryResponse(response=response_text, evidence=evidence, graphData=graph_out, reasoningPaths=paths)

    routed = route_nodes(
        req.query,
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
    chunks = retrieve_chunks(req.query, kg, node_ids=expanded, embedder=embedder, top_k=chunk_top_k)

    answer = synthesize_answer(
        req.query,
        chunks,
        provider=provider,
        model=model or cfg.llm_model,
        max_tokens=cfg.llm_max_tokens,
        reasoning_paths=paths,
    )

    evidence: List[EvidenceItem] = []
    def _trim_definition(text: str) -> str:
        match = re.search(r"definition\\s+\\d+(?:\\.\\d+)*", text, re.I)
        if not match:
            return text
        snippet = text[match.start() :]
        stop = re.search(
            r"\\b(For example|Another example|Example\\s+\\d|Lemma\\s+\\d|Theorem\\s+\\d|Proposition\\s+\\d|Corollary\\s+\\d|Remark\\s+\\d|Exercise\\s+\\d|Proof)\\b",
            snippet,
            re.I,
        )
        if stop:
            snippet = snippet[: stop.start()]
        return " ".join(snippet.split())

    for idx, (score, pdf_id, section_id, text) in enumerate(chunks[: min(6, len(chunks))], start=1):
        content = text
        if wants_def:
            node = kg.nodes.get(f"{pdf_id}:{section_id}")
            if node:
                manifest_path = pathlib.Path(node.path) / "manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        defs = manifest.get("definitions", [])
                        if defs:
                            content = defs[0]
                        else:
                            content = _trim_definition(text)
                    except Exception:
                        content = _trim_definition(text)
            else:
                content = _trim_definition(text)
        evidence.append(
            EvidenceItem(
                id=f"{pdf_id}:{section_id}:{idx}",
                title=f"{section_id} ({pdf_id})",
                content=content[:700],
                source=pdf_id,
            )
        )

    graph_out = _build_graph_data(kg, expanded)
    return QueryResponse(response=answer, evidence=evidence, graphData=graph_out, reasoningPaths=paths)


@app.get("/health")
def health() -> dict:
    cfg, _, _, _, _ = _load_state()
    key = os.environ.get("OPENAI_API_KEY")
    return {
        "openai_key_set": bool(key),
        "openai_key_prefix": key[:6] + "..." if key else None,
        "default_provider": cfg.llm_provider,
        "default_model": cfg.llm_model,
    }

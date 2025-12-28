from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
    embedding_provider: str
    embedding_model: str
    embedding_dims: int
    node_top_k: int
    chunk_top_k: int
    expansion_hops: int
    w_semantic: float
    w_prereq: float
    global_chunk_top_k: int
    w_chunk: float
    w_lex: float
    w_concept: float
    graph_max_hops: int
    graph_path_top_k: int
    graph_dir: pathlib.Path
    nodes_dir: pathlib.Path
    llm_provider: str
    llm_model: str
    llm_max_tokens: int

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "Config":
        data = yaml.safe_load(pathlib.Path(path).read_text())
        return cls(
            embedding_provider=data["embedding"]["provider"],
            embedding_model=data["embedding"]["model"],
            embedding_dims=int(data["embedding"]["dimensions"]),
            node_top_k=int(data["retrieval"]["node_top_k"]),
            chunk_top_k=int(data["retrieval"]["chunk_top_k"]),
            expansion_hops=int(data["retrieval"]["expansion_hops"]),
            w_semantic=float(data["retrieval"].get("w_semantic", 1.0)),
            w_prereq=float(data["retrieval"].get("w_prereq", 0.2)),
            global_chunk_top_k=int(data["retrieval"].get("global_chunk_top_k", 30)),
            w_chunk=float(data["retrieval"].get("w_chunk", 0.7)),
            w_lex=float(data["retrieval"].get("w_lex", 0.2)),
            w_concept=float(data["retrieval"].get("w_concept", 0.6)),
            graph_max_hops=int(data["retrieval"].get("graph_max_hops", 2)),
            graph_path_top_k=int(data["retrieval"].get("graph_path_top_k", 3)),
            graph_dir=pathlib.Path(data["paths"]["graph_dir"]),
            nodes_dir=pathlib.Path(data["paths"]["nodes_dir"]),
            llm_provider=data["llm"]["provider"],
            llm_model=data["llm"]["model"],
            llm_max_tokens=int(data["llm"]["max_tokens"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "dimensions": self.embedding_dims,
            },
            "retrieval": {
                "node_top_k": self.node_top_k,
                "chunk_top_k": self.chunk_top_k,
                "expansion_hops": self.expansion_hops,
                "w_semantic": self.w_semantic,
                "w_prereq": self.w_prereq,
                "global_chunk_top_k": self.global_chunk_top_k,
                "w_chunk": self.w_chunk,
                "w_lex": self.w_lex,
                "w_concept": self.w_concept,
                "graph_max_hops": self.graph_max_hops,
                "graph_path_top_k": self.graph_path_top_k,
            },
            "paths": {"graph_dir": str(self.graph_dir), "nodes_dir": str(self.nodes_dir)},
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "max_tokens": self.llm_max_tokens,
            },
        }

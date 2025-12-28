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
            w_semantic=float(data["retrieval"]["w_semantic"]),
            w_prereq=float(data["retrieval"]["w_prereq"]),
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
            },
            "paths": {"graph_dir": str(self.graph_dir), "nodes_dir": str(self.nodes_dir)},
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "max_tokens": self.llm_max_tokens,
            },
        }


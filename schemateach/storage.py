from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:  # Optional FAISS acceleration.
    import faiss  # type: ignore
except Exception:  # pragma: no cover - environment-specific import
    faiss = None


@dataclass
class Chunk:
    chunk_id: str
    text: str
    rank: int
    metadata: dict


class VectorStore:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.embeddings: np.ndarray | None = None
        self.chunks: List[Chunk] = []
        self._faiss_index = None

    def load(self) -> None:
        idx_path = self.path / "index.npz"
        meta_path = self.path / "chunks.jsonl"
        if idx_path.exists():
            data = np.load(idx_path)
            self.embeddings = data["embeddings"]
        if meta_path.exists():
            self.chunks = []
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.chunks.append(
                        Chunk(chunk_id=row["chunk_id"], text=row["text"], rank=row.get("rank", 0), metadata=row.get("metadata", {}))
                    )

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.savez(self.path / "index.npz", embeddings=self.embeddings)
        meta_path = self.path / "chunks.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(
                    json.dumps({"chunk_id": chunk.chunk_id, "text": chunk.text, "rank": chunk.rank, "metadata": chunk.metadata}) + "\n"
                )

    def set_embeddings(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.chunks = chunks
        self._faiss_index = None

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[float, Chunk]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        index = self._get_faiss_index()
        if index is not None:
            qn = query_vec.astype(np.float32)
            qn = qn / (np.linalg.norm(qn) + 1e-8)
            scores, idxs = index.search(qn.reshape(1, -1), top_k)
            hits: List[Tuple[float, Chunk]] = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                hits.append((float(score), self.chunks[int(idx)]))
            return hits
        norms = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        sims = (self.embeddings @ qn) / norms
        idxs = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.chunks[int(i)]) for i in idxs]

    def _get_faiss_index(self):
        if faiss is None or self.embeddings is None or len(self.chunks) == 0:
            return None
        if self._faiss_index is not None:
            return self._faiss_index
        vectors = self.embeddings.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vectors = vectors / norms
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._faiss_index = index
        return index

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:  # Optional FAISS acceleration.
    import faiss  # type: ignore
except Exception:  # pragma: no cover - environment-specific import
    faiss = None


@dataclass
class ChunkMeta:
    node_id: str
    pdf_id: str
    section_id: str
    chunk_id: str
    text: str


class GlobalChunkIndex:
    def __init__(self, embeddings: np.ndarray | None = None, meta: List[ChunkMeta] | None = None):
        self.embeddings = embeddings
        self.meta = meta or []
        self._faiss_index = None

    @classmethod
    def load(cls, base_dir: pathlib.Path) -> "GlobalChunkIndex":
        idx_path = base_dir / "chunks_index.npz"
        meta_path = base_dir / "chunks_meta.jsonl"
        if not idx_path.exists() or not meta_path.exists():
            return cls()
        data = np.load(idx_path)
        embeddings = data["embeddings"]
        meta: List[ChunkMeta] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                meta.append(
                    ChunkMeta(
                        node_id=row["node_id"],
                        pdf_id=row["pdf_id"],
                        section_id=row["section_id"],
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                    )
                )
        return cls(embeddings=embeddings, meta=meta)

    def save(self, base_dir: pathlib.Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.savez(base_dir / "chunks_index.npz", embeddings=self.embeddings.astype(np.float32))
        with (base_dir / "chunks_meta.jsonl").open("w", encoding="utf-8") as f:
            for row in self.meta:
                f.write(
                    json.dumps(
                        {
                            "node_id": row.node_id,
                            "pdf_id": row.pdf_id,
                            "section_id": row.section_id,
                            "chunk_id": row.chunk_id,
                            "text": row.text,
                        }
                    )
                    + "\n"
                )

    def append(self, embeddings: np.ndarray, meta: List[ChunkMeta]) -> None:
        if embeddings.size == 0:
            return
        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        self.meta.extend(meta)
        self._faiss_index = None

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> List[Tuple[float, ChunkMeta]]:
        if self.embeddings is None or len(self.meta) == 0:
            return []
        index = self._get_faiss_index()
        if index is not None:
            qn = query_vec.astype(np.float32)
            qn = qn / (np.linalg.norm(qn) + 1e-8)
            scores, idxs = index.search(qn.reshape(1, -1), top_k)
            hits: List[Tuple[float, ChunkMeta]] = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                hits.append((float(score), self.meta[int(idx)]))
            return hits
        norms = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        sims = (self.embeddings @ qn) / norms
        idxs = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.meta[int(i)]) for i in idxs]

    def _get_faiss_index(self):
        if faiss is None or self.embeddings is None or len(self.meta) == 0:
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


def load_concept_map(base_dir: pathlib.Path) -> Dict[str, List[str]]:
    path = base_dir / "concepts.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_concept_map(base_dir: pathlib.Path, concept_map: Dict[str, List[str]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "concepts.json").write_text(json.dumps(concept_map, indent=2), encoding="utf-8")

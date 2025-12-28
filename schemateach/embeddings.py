from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


def _hash_vector(text: str, dims: int) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    repeated = (digest * ((dims // len(digest)) + 1))[:dims]
    arr = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
    arr = arr - np.mean(arr)
    norm = np.linalg.norm(arr) + 1e-8
    return arr / norm


@dataclass
class Embedder:
    provider: str
    model: str
    dims: int

    def embed(self, texts: Iterable[str]) -> List[np.ndarray]:
        if self.provider == "openai":
            return self._embed_openai(list(texts))
        if self.provider == "hf":
            return self._embed_hf(list(texts))
        return [self._embed_fallback(t) for t in texts]

    def _embed_openai(self, texts: List[str]) -> List[np.ndarray]:
        if openai is None:
            return [self._embed_fallback(t) for t in texts]
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return [self._embed_fallback(t) for t in texts]
        client = openai.OpenAI(api_key=api_key)
        resp = client.embeddings.create(input=texts, model=self.model)
        return [np.array(item.embedding, dtype=np.float32) for item in resp.data]

    def _embed_hf(self, texts: List[str]) -> List[np.ndarray]:
        if SentenceTransformer is None:
            return [self._embed_fallback(t) for t in texts]
        model = SentenceTransformer(self.model)
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [np.array(v, dtype=np.float32) for v in emb]

    def _embed_fallback(self, text: str) -> np.ndarray:
        return _hash_vector(text, self.dims)


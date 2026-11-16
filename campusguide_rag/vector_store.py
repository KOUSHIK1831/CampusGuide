"""Normalized vector indexing with FAISS by default and a NumPy test backend.

Persistence is JSON + FAISS/NumPy, never untrusted pickle deserialization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    def __init__(
        self,
        directory: str | Path = "data/processed",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: Any = None,
        backend: str = "faiss",
    ) -> None:
        if backend not in {"faiss", "numpy"}:
            raise ValueError("backend must be 'faiss' or 'numpy'")
        self.directory = Path(directory)
        self.model_name = model_name
        self.encoder = encoder
        self.backend = backend
        self.index: Any = None
        self.chunks: list[dict] = []

    def _encoder(self) -> Any:
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name, device="cpu")
        return self.encoder

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(self._encoder().encode(texts), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise ValueError("Encoder must return one 2D embedding per input text")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(arr).all():
            raise ValueError("Embeddings must be finite and nonzero")
        return np.ascontiguousarray(arr / norms, dtype=np.float32)

    @property
    def ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index without text chunks")
        vectors = self.embed([chunk["text"] for chunk in chunks])
        if self.backend == "faiss":
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self.index = index
        else:
            self.index = vectors
        self.chunks = [dict(chunk) for chunk in chunks]

    def search(self, query: str, top_k: int = 4, min_score: float = 0.30) -> list[dict]:
        if not self.ready or not query.strip():
            return []
        if top_k < 1 or not -1 <= min_score <= 1:
            raise ValueError("top_k must be positive and min_score within [-1,1]")
        vector = self.embed([query])
        count = min(top_k, len(self.chunks))
        if self.backend == "faiss":
            scores, positions = self.index.search(vector, count)
            pairs = zip(scores[0], positions[0])
        else:
            scores = self.index @ vector[0]
            positions = np.argsort(-scores, kind="stable")[:count]
            pairs = ((scores[i], i) for i in positions)
        return [
            {**self.chunks[int(i)], "score": round(float(score), 4)}
            for score, i in pairs if int(i) >= 0 and float(score) >= min_score
        ]

    def save(self) -> None:
        if not self.ready:
            raise ValueError("No index to save")
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_name": self.model_name,
            "backend": self.backend,
            "chunks": self.chunks,
        }
        if self.backend == "faiss":
            import faiss
            faiss.write_index(self.index, str(self.directory / "index.faiss"))
        else:
            np.save(self.directory / "index.npy", self.index, allow_pickle=False)
        (self.directory / "chunks.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

    def load(self) -> bool:
        meta_path = self.directory / "chunks.json"
        index_path = self.directory / ("index.faiss" if self.backend == "faiss" else "index.npy")
        if not meta_path.is_file() or not index_path.is_file():
            return False
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("model_name") != self.model_name or metadata.get("backend") != self.backend:
            raise ValueError("Stored index config mismatch; rebuild the index")
        if self.backend == "faiss":
            import faiss
            index = faiss.read_index(str(index_path))
            count = index.ntotal
        else:
            index = np.load(index_path, allow_pickle=False)
            count = len(index)
        chunks = metadata["chunks"]
        if count != len(chunks):
            raise ValueError("Index and metadata count mismatch; rebuild the index")
        self.index, self.chunks = index, chunks
        return self.ready

    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "documents": len({c["source"] for c in self.chunks}), "backend": self.backend}

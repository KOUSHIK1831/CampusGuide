"""Normalized vector indexing with FAISS by default and a NumPy test backend.

Published index/metadata pairs are immutable snapshots; concurrent rebuilds cannot mismatch results.
Persistence is JSON + FAISS/NumPy, never untrusted pickle deserialization.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
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
        self._lock = RLock()
        # Publish a single index + metadata snapshot; never mutate one after publication.
        self._snapshot: tuple[Any, tuple[dict, ...]] | None = None

    def _encoder(self) -> Any:
        if self.encoder is None:
            with self._lock:
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
        with self._lock:
            return self._snapshot is not None

    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index without text chunks")
        # Build separately; publishing never changes a snapshot held by an active search.
        vectors = self.embed([chunk["text"] for chunk in chunks])
        if self.backend == "faiss":
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        else:
            index = vectors
        metadata = tuple(dict(chunk) for chunk in chunks)
        with self._lock:
            self._snapshot = (index, metadata)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.30) -> list[dict]:
        if not query.strip():
            return []
        if top_k < 1 or not -1 <= min_score <= 1:
            raise ValueError("top_k must be positive and min_score within [-1,1]")
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return []
        # Retain the matching index and metadata throughout this retrieval.
        index, chunks = snapshot
        vector = self.embed([query])
        count = min(top_k, len(chunks))
        if self.backend == "faiss":
            scores, positions = index.search(vector, count)
            pairs = zip(scores[0], positions[0])
        else:
            scores = index @ vector[0]
            positions = np.argsort(-scores, kind="stable")[:count]
            pairs = ((scores[i], i) for i in positions)
        return [
            {**chunks[int(i)], "score": round(float(score), 4)}
            for score, i in pairs if 0 <= int(i) < len(chunks) and float(score) >= min_score
        ]

    def save(self) -> None:
        # Write index and metadata from the SAME snapshot, under the instance I/O lock.
        with self._lock:
            snapshot = self._snapshot
            if snapshot is None:
                raise ValueError("No index to save")
            index, chunks = snapshot
            self.directory.mkdir(parents=True, exist_ok=True)
            metadata = {
                "model_name": self.model_name,
                "backend": self.backend,
                "chunks": list(chunks),
            }
            if self.backend == "faiss":
                import faiss
                faiss.write_index(index, str(self.directory / "index.faiss"))
            else:
                np.save(self.directory / "index.npy", index, allow_pickle=False)
            (self.directory / "chunks.json").write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )

    def load(self) -> bool:
        with self._lock:
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
            if not chunks:
                return False
            self._snapshot = (index, tuple(dict(chunk) for chunk in chunks))
            return True

    def stats(self) -> dict:
        with self._lock:
            snapshot = self._snapshot
        chunks = snapshot[1] if snapshot is not None else ()
        return {
            "chunks": len(chunks),
            "documents": len({chunk["source"] for chunk in chunks}),
            "backend": self.backend,
        }

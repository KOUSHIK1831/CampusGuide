"""Normalized vector indexing with FAISS by default and a NumPy test backend.

Dense, lexical, and metadata indexes are published together as immutable snapshots.
Persistence is JSON + FAISS/NumPy, never untrusted pickle deserialization.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from .retrieval import BM25Index, MAX_RERANK_CANDIDATES, Reranker, fuse_ranks, rerank_hits

# Hybrid search may fill the evidence pool with passages scoring at least this
# fraction of the dense cutoff, but only after a passage clears the cutoff.
EXPANSION_FLOOR_RATIO = 0.5


@dataclass(frozen=True)
class _IndexSnapshot:
    index: Any
    chunks: tuple[dict, ...]
    lexical: BM25Index


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
        self._snapshot: _IndexSnapshot | None = None

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
        snapshot = _IndexSnapshot(index, metadata, BM25Index(metadata))
        with self._lock:
            self._snapshot = snapshot

    def search(
        self, query: str, top_k: int = 4, min_score: float = 0.30,
        *, mode: str = "dense", reranker: Reranker | None = None,
    ) -> list[dict]:
        """Search using cosine alone or fuse it with BM25, then optionally rerank."""
        if not query.strip():
            return []
        if top_k < 1 or not -1 <= min_score <= 1:
            raise ValueError("top_k must be positive and min_score within [-1,1]")
        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode must be 'dense' or 'hybrid'")
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return []
        vector = self.embed([query])
        limit = top_k
        if mode == "hybrid" or reranker is not None:
            limit = max(top_k, min(4 * top_k, MAX_RERANK_CANDIDATES))
        # Read past repeats so they cannot crowd out the bounded pool of unique hits.
        duplicates = snapshot.lexical.duplicate_count if mode == "hybrid" else 0
        count = min(limit + duplicates, len(snapshot.chunks))
        pairs = [
            (score, position) for score, position in self._dense_pairs(snapshot.index, vector, count)
            if 0 <= position < len(snapshot.chunks)
        ]
        dense = [(score, position) for score, position in pairs if score >= min_score]

        if mode == "hybrid":
            # Once a passage clears the dense cutoff, expand the fusion pool with
            # the next-best passages. Multi-part questions often retrieve only a
            # header/profile chunk at the cutoff while the education, skills, or
            # experience chunks score lower on their own. Lexical matching is
            # relaxed for this anchored pool; queries without any confident hit
            # keep the stricter gate and can still return no evidence.
            anchored = bool(dense)
            strong = {position for _, position in dense}
            if anchored:
                expansion_floor = min_score * EXPANSION_FLOOR_RATIO
                dense = [(score, position) for score, position in pairs if score >= expansion_floor]
            hits = self._hybrid_hits(
                snapshot, query, vector[0], dense, limit,
                relaxed_lexical=anchored, strong_positions=strong,
            )
        else:
            hits = [
                {**snapshot.chunks[position], "score": round(score, 4)}
                for score, position in dense
            ]
        if reranker is not None:
            hits = rerank_hits(query, hits, reranker)
        return hits[:top_k]

    def _dense_pairs(self, index: Any, vector: np.ndarray, count: int) -> list[tuple[float, int]]:
        if self.backend == "faiss":
            scores, positions = index.search(vector, count)
            return [(float(score), int(position)) for score, position in zip(scores[0], positions[0])]
        scores = index @ vector[0]
        positions = np.argsort(-scores, kind="stable")[:count]
        return [(float(scores[position]), int(position)) for position in positions]

    def _hybrid_hits(
        self, snapshot: _IndexSnapshot, query: str, vector: np.ndarray,
        dense: list[tuple[float, int]], limit: int, *,
        relaxed_lexical: bool = False, strong_positions: set[int] | None = None,
    ) -> list[dict]:
        dense_rows = {}
        for score, position in sorted(dense, key=lambda result: (-result[0], result[1])):
            canonical = snapshot.lexical.canonical_positions[position]
            dense_rows.setdefault(canonical, (position, score))
            if len(dense_rows) == limit:
                break
        lexical = snapshot.lexical.search(query, dense_rows, limit, relaxed=relaxed_lexical)
        fused = fuse_ranks(list(dense_rows), list(lexical))
        ranked = list(fused.items())
        if strong_positions:
            # Keep passages that cleared the cutoff ahead of the expansion pool.
            strong = {snapshot.lexical.canonical_positions[position] for position in strong_positions}
            ranked.sort(key=lambda item: (item[0] not in strong, -item[1]))
        hits = []
        for canonical, fusion_score in ranked[:limit]:
            if canonical in dense_rows:
                position, score = dense_rows[canonical]
            else:
                position = canonical
                if self.backend == "faiss":
                    passage_vector = snapshot.index.reconstruct(position)
                else:
                    passage_vector = snapshot.index[position]
                score = float(passage_vector @ vector)
            hits.append({
                **snapshot.chunks[position], "score": round(score, 4),
                "lexical_score": lexical.get(canonical), "fusion_score": fusion_score,
            })
        return hits

    def save(self) -> None:
        # Write index and metadata from the SAME snapshot, under the instance I/O lock.
        with self._lock:
            snapshot = self._snapshot
            if snapshot is None:
                raise ValueError("No index to save")
            index, chunks = snapshot.index, snapshot.chunks
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
            metadata = tuple(dict(chunk) for chunk in chunks)
            self._snapshot = _IndexSnapshot(index, metadata, BM25Index(metadata))
            return True

    def stats(self) -> dict:
        with self._lock:
            snapshot = self._snapshot
        chunks = snapshot.chunks if snapshot is not None else ()
        return {
            "chunks": len(chunks),
            "documents": len({chunk["source"] for chunk in chunks}),
            "backend": self.backend,
        }

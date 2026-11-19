"""Lexical indexing, rank fusion, and optional local cross-encoder scoring."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Container, Sequence
from heapq import nlargest
import math
import re
from threading import Lock
from typing import Callable

import numpy as np

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L2-v2"
MAX_RERANK_CANDIDATES = 32
Reranker = Callable[[str, list[str]], Sequence[float] | np.ndarray]

_TERM_PATTERN = re.compile(r"[^\W_]+(?:[-_/][^\W_]+)*")
_IDENTIFIER_PATTERN = re.compile(r"(?:[a-z]{2,}[-_/]?|[a-z][-_])\d+[a-z0-9]*(?:[-_/][a-z0-9]+)*")
_STOP_WORDS = frozenset("""
    a about an and are as at be been by can could did do does for from had has have
    how i if in is it its may me more much many my of on or our please should than
    that the their them then there these they this those to us was we were what
    when where which who why will with would you your
""".split())
_PROVENANCE_FIELDS = ("source", "page", "document_id", "policy_id", "version", "effective_date")
_BM25_K1 = 1.2
_BM25_B = 0.75
_RRF_K = 60


def _terms(text: str) -> list[str]:
    return [
        token for token in _TERM_PATTERN.findall(text.casefold())
        if len(token) > 1 and token not in _STOP_WORDS
    ]


class BM25Index:
    """Index unique passages while retaining original source/page/version identity."""

    def __init__(self, chunks: tuple[dict, ...]) -> None:
        postings = defaultdict(list)
        lengths = {}
        seen = {}
        canonical_positions = []
        for position, chunk in enumerate(chunks):
            identity = (
                " ".join(chunk["text"].split()),
                *(str(chunk.get(field, "")) for field in _PROVENANCE_FIELDS),
            )
            canonical = seen.setdefault(identity, position)
            canonical_positions.append(canonical)
            if canonical != position:
                continue
            terms = Counter(_terms(chunk["text"]))
            lengths[position] = sum(terms.values())
            for term, count in terms.items():
                postings[term].append((position, count))
        document_count = len(lengths)
        average_length = sum(lengths.values()) / max(document_count, 1) or 1.0
        self._postings = {term: tuple(entries) for term, entries in postings.items()}
        self._idf = {
            term: math.log1p((document_count - len(entries) + 0.5) / (len(entries) + 0.5))
            for term, entries in self._postings.items()
        }
        self._norm = {
            position: _BM25_K1 * (1 - _BM25_B + _BM25_B * length / average_length)
            for position, length in lengths.items()
        }
        self.canonical_positions = tuple(canonical_positions)
        self.duplicate_count = len(chunks) - document_count

    def search(self, query: str, dense_positions: Container[int], limit: int) -> dict[int, float]:
        query_terms = sorted(set(_terms(query)))
        scores = defaultdict(float)
        overlap = Counter()
        identifier_matches = set()
        for term in query_terms:
            if term not in self._postings:
                continue
            exact_identifier = _IDENTIFIER_PATTERN.fullmatch(term) is not None
            weight = self._idf[term] * (_BM25_K1 + 1)
            for position, count in self._postings[term]:
                scores[position] += weight * count / (count + self._norm[position])
                overlap[position] += 1
                if exact_identifier:
                    identifier_matches.add(position)

        # Stronger overlap is required to introduce evidence below the dense gate.
        minimum_overlap = max(2, (len(query_terms) + 1) // 2)
        eligible = (
            (position, score) for position, score in scores.items()
            if position in dense_positions or position in identifier_matches or overlap[position] >= minimum_overlap
        )
        return dict(nlargest(limit, eligible, key=lambda item: (item[1], -item[0])))


def fuse_ranks(dense_positions: Sequence[int], lexical_positions: Sequence[int]) -> dict[int, float]:
    """Combine independent ranked lists without mixing cosine and BM25 scores."""
    scores = defaultdict(float)
    for ranking in (dense_positions, lexical_positions):
        for rank, position in enumerate(ranking, 1):
            scores[position] += 1.0 / (_RRF_K + rank)
    # Stable sort favors the dense ordering in ties across channels.
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


class CrossEncoderReranker:
    """Lazy CPU scoring. A shared instance serializes model loading and inference."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = Lock()

    def __call__(self, query: str, passages: list[str]) -> np.ndarray:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)
            return self._model.predict(
                [(query, passage) for passage in passages],
                batch_size=8,
                show_progress_bar=False,
            )


def rerank_hits(query: str, hits: list[dict], reranker: Reranker) -> list[dict]:
    candidates = hits[:MAX_RERANK_CANDIDATES]
    if len(candidates) < 2:
        return [{**hit, "rerank_status": "skipped"} for hit in hits]
    try:
        scores = np.asarray(reranker(query, [hit["text"] for hit in candidates]), dtype=float)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("Reranker must return a finite scalar for each candidate")
    except Exception:
        return [{**hit, "rerank_status": "fallback"} for hit in hits]

    reranked = [
        {**hit, "rerank_status": "applied", "rerank_score": float(score)}
        for hit, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda hit: hit["rerank_score"], reverse=True)
    return reranked + [{**hit, "rerank_status": "skipped"} for hit in hits[MAX_RERANK_CANDIDATES:]]

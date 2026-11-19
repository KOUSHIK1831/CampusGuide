from dataclasses import replace

import numpy as np
import pytest

from campusguide_rag import VectorStore
from campusguide_rag.retrieval import MAX_RERANK_CANDIDATES


class FakeEncoder:
    """Deterministic stand-in: no model downloads or API traffic in tests."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            text = text.lower()
            if any(word in text for word in ("library", "books", "borrow")):
                vectors.append([4.0, 0.0, 0.0])
            elif any(word in text for word in ("advising", "advisor")):
                vectors.append([0.0, 7.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 2.0])
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    store = VectorStore(tmp_path, encoder=FakeEncoder(), model_name="fake-test-model", backend="numpy")
    store.build([
        {"id": "1", "source": "library.md", "page": None, "text": "Library students borrow four books with permit LB-204."},
        {"id": "2", "source": "advising.txt", "page": None, "text": "Advising sessions happen on Wednesday in AD-501."},
    ])
    return store


def test_embedding_normalization(store):
    vectors = store.embed(["library", "advising"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), [1, 1])


def test_retrieval_ranks_semantically_similar_chunk(store):
    results = store.search("How many books may I borrow?", top_k=2, min_score=0.4)
    assert len(results) == 1
    assert results[0]["source"] == "library.md"
    assert results[0]["score"] == pytest.approx(1.0)


def test_unanswerable_is_refused_at_retrieval_gate(store):
    assert store.search("Weather on Jupiter?", top_k=2, min_score=0.4) == []
    assert store.search("   ") == []


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_hybrid_finds_exact_identifier_with_separate_cosine_score(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
    store = VectorStore(tmp_path, encoder=FakeEncoder(), backend=backend)
    store.build([{"id": "1", "source": "library.md", "page": 2, "text": "Library students use permit LB-204."}])
    question = "What does LB-204 cover?"
    assert store.search(question) == []
    (hit,) = store.search(question, mode="hybrid")
    assert hit["source"] == "library.md"
    assert hit["page"] == 2
    assert hit["score"] == pytest.approx(0.0)
    assert hit["lexical_score"] > 0
    assert hit["fusion_score"] > 0


def test_hybrid_admits_strong_overlap_and_rejects_a_single_shared_term(store):
    assert [hit["id"] for hit in store.search("How much is that Wednesday session?", mode="hybrid")] == []
    assert [hit["id"] for hit in store.search("What about those Wednesday sessions?", mode="hybrid")] == ["2"]
    assert store.search("How do I get a permit to access lunar rocks?", mode="hybrid") == []


def test_hybrid_fuses_overlap_without_requiring_it_for_semantic_hits(tmp_path):
    store = VectorStore(tmp_path, encoder=FakeEncoder(), backend="numpy")
    store.build([
        {"id": "irrelevant", "source": "library.md", "text": "Library desks are by the window."},
        {"id": "relevant", "source": "circulation.md", "text": "Students may borrow four library books."},
        {"id": "semantic", "source": "loans.md", "text": "Take library materials home with your campus ID."},
    ])
    question = "How many books can I borrow?"
    assert [hit["id"] for hit in store.search(question)] == ["irrelevant", "relevant", "semantic"]
    assert [hit["id"] for hit in store.search(question, mode="hybrid")] == ["relevant", "irrelevant", "semantic"]


def test_hybrid_dedup_preserves_distinct_sources_pages_and_versions(tmp_path):
    store = VectorStore(tmp_path, encoder=FakeEncoder(), backend="numpy")
    original = {"id": "original", "source": "policy.pdf", "page": 1, "text": "Library books", "version": "1"}
    store.build([
        original,
        {**original, "id": "repeat", "text": "Library  books"},
        {**original, "id": "new-version", "version": "2"},
        {**original, "id": "other-page", "page": 2},
        {**original, "id": "other-source", "source": "annex.pdf"},
    ])
    assert len(store.search("books", top_k=5)) == 5
    assert [hit["id"] for hit in store.search("books", top_k=5, mode="hybrid")] == [
        "original", "new-version", "other-page", "other-source",
    ]


def test_duplicates_cannot_crowd_out_unique_dense_hits(tmp_path):
    store = VectorStore(tmp_path, encoder=FakeEncoder(), backend="numpy")
    store.build([
        {"id": str(i), "source": "library.md", "text": "Students borrow four books."}
        for i in range(40)
    ] + [{"id": "unique", "source": "reserves.md", "text": "Instructors reserve library materials."}])
    assert [hit["id"] for hit in store.search("Library borrowers", top_k=2, mode="hybrid")] == ["unique", "0"]


def test_reranking_reorders_a_bounded_pool_and_preserves_similarity(tmp_path):
    store = VectorStore(tmp_path, encoder=FakeEncoder(), backend="numpy")
    store.build([
        {"id": str(i), "source": "library.md", "page": i, "text": f"Library books passage {i}"}
        for i in range(50)
    ] + [{"id": "unrelated", "source": "advising.txt", "text": "Advising news"}])

    def reverse_rank(query, passages):
        assert query == "Library books"
        assert len(passages) == MAX_RERANK_CANDIDATES
        assert all(passage.startswith("Library books") for passage in passages)
        return range(len(passages))

    hits = store.search("Library books", top_k=40, mode="hybrid", reranker=reverse_rank)
    assert len(hits) == 40
    assert all(hit["score"] == 1.0 for hit in hits)
    assert hits[0]["id"] == "31"
    assert hits[0]["rerank_score"] == 31
    assert hits[0]["rerank_status"] == "applied"
    assert hits[-1]["id"] == "39"
    assert hits[-1]["rerank_status"] == "skipped"


@pytest.mark.parametrize("scores", [[1], [float("nan"), 1]])
def test_invalid_reranker_scores_fall_back_with_original_order(store, scores):
    baseline = store.search("Library advising", min_score=0.0, mode="hybrid")
    hits = store.search(
        "Library advising", min_score=0.0, mode="hybrid", reranker=lambda query, passages: scores,
    )
    assert len(baseline) == 2
    assert hits == [{**hit, "rerank_status": "fallback"} for hit in baseline]


def test_reranker_errors_fall_back_and_skips_single_hit(store):
    def unavailable(query, passages):
        raise RuntimeError("Model unavailable")

    hits = store.search("Library advising", min_score=0.0, mode="hybrid", reranker=unavailable)
    assert [hit["rerank_status"] for hit in hits] == ["fallback", "fallback"]
    (hit,) = store.search("LB-204", mode="hybrid", reranker=unavailable)
    assert hit["rerank_status"] == "skipped"
    assert store.search("Weather on Jupiter?", mode="hybrid", reranker=unavailable) == []


def test_index_persistence_without_pickle(store, tmp_path):
    store.save()
    loaded = VectorStore(tmp_path, encoder=FakeEncoder(), model_name="fake-test-model", backend="numpy")
    assert loaded.load()
    assert loaded.search("library books", top_k=1)[0]["source"] == "library.md"
    assert loaded.search("LB-204", mode="hybrid")[0]["source"] == "library.md"
    assert (tmp_path / "chunks.json").is_file()
    assert (tmp_path / "index.npy").is_file()


def test_fails_on_embedding_model_mismatch(store, tmp_path):
    store.save()
    loaded = VectorStore(tmp_path, encoder=FakeEncoder(), model_name="different-model", backend="numpy")
    with pytest.raises(ValueError, match="config mismatch"):
        loaded.load()


def test_faiss_backend_when_available(tmp_path):
    pytest.importorskip("faiss")
    store = VectorStore(tmp_path, encoder=FakeEncoder(), model_name="fake-test-model", backend="faiss")
    store.build([{"id": "1", "source": "library.md", "text": "Library books", "page": 1}])
    store.save()
    loaded = VectorStore(tmp_path, encoder=FakeEncoder(), model_name="fake-test-model", backend="faiss")
    assert loaded.load()
    assert loaded.search("books")[0]["page"] == 1


def test_build_copies_chunk_metadata(tmp_path):
    store = VectorStore(tmp_path, encoder=FakeEncoder(), model_name="fake", backend="numpy")
    chunks = [{"id": "1", "source": "original.txt", "page": None, "text": "Library books"}]
    store.build(chunks)
    chunks[0]["source"] = "changed.txt"
    chunks.append({"id": "2", "source": "extra.txt", "page": None, "text": "Weather"})
    assert store.stats()["documents"] == 1
    assert store.search("books")[0]["source"] == "original.txt"


@pytest.mark.parametrize(("mode", "question"), [("dense", "advising"), ("hybrid", "AD-501")])
def test_search_keeps_old_index_with_old_metadata_during_rebuild(store, mode, question):
    """Reproduce the race reported by Copilot with two coordinated threads."""
    import threading

    snapshot = store._snapshot
    entered = threading.Event()
    release = threading.Event()

    class BlockingMatrix:
        def __matmul__(self, vector):
            entered.set()
            if not release.wait(timeout=8):
                raise TimeoutError("Search was not released")
            return snapshot.index @ vector

        def __getitem__(self, position):
            return snapshot.index[position]

    with store._lock:
        store._snapshot = replace(snapshot, index=BlockingMatrix())

    results = []
    errors = []

    def run_search():
        try:
            results.extend(store.search(question, top_k=2, min_score=0.4, mode=mode))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_search, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=8), "Search never reached the old index"
        store.build([{
            "id": "replacement", "source": "replacement.txt", "page": None,
            "text": "Weather on Jupiter",
        }])
    finally:
        release.set()
        worker.join(timeout=8)

    assert not worker.is_alive()
    assert not errors
    assert [hit["source"] for hit in results] == ["advising.txt"]
    assert store.stats()["documents"] == 1
    assert store.search("Weather", min_score=0.4)[0]["source"] == "replacement.txt"

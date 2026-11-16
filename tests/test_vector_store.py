import numpy as np
import pytest

from campusguide_rag import VectorStore


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
        {"id": "1", "source": "library.md", "page": None, "text": "Library students borrow four books."},
        {"id": "2", "source": "advising.txt", "page": None, "text": "Advising sessions happen on Wednesday."},
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


def test_index_persistence_without_pickle(store, tmp_path):
    store.save()
    loaded = VectorStore(tmp_path, encoder=FakeEncoder(), model_name="fake-test-model", backend="numpy")
    assert loaded.load()
    assert loaded.search("library books", top_k=1)[0]["source"] == "library.md"
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


def test_search_keeps_old_index_with_old_metadata_during_rebuild(store):
    """Reproduce the race reported by Copilot with two coordinated threads."""
    import threading

    old_index, old_chunks = store._snapshot
    entered = threading.Event()
    release = threading.Event()

    class BlockingMatrix:
        def __matmul__(self, vector):
            entered.set()
            if not release.wait(timeout=8):
                raise TimeoutError("Search was not released")
            return old_index @ vector

    with store._lock:
        store._snapshot = (BlockingMatrix(), old_chunks)

    results = []
    errors = []

    def run_search():
        try:
            results.extend(store.search("advising", top_k=2, min_score=0.4))
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

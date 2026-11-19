import json
from pathlib import Path

import pytest

from scripts.evaluate import evaluate, load_corpus, load_manifest


class FakeStore:
    def __init__(self):
        self.requests = []

    def search(self, query, top_k, min_score, *, mode, reranker):
        self.requests.append((query, top_k, min_score, mode, reranker))
        if query in {"library rules", "library loan period"}:
            return [
                {"source": "campus_faq.md", "text": "Advising takes place on Wednesdays."},
                {"source": "campus_faq.md", "text": "Loan periods are   fourteen days.", "page": 2},
            ]
        if query == "What is the time limit?":
            return [{"source": "handbook.md", "text": "Applications open 14 days before."}]
        return []


def test_evidence_rank_refusal_and_supplied_follow_up_are_scored_separately():
    proof = {"source": "campus_faq.md", "contains": "Fourteen days", "page": 2}
    cases = [
        {"id": "library", "kind": "answerable", "question": "library rules", "evidence": [proof]},
        {"id": "follow-up", "kind": "follow_up", "question": "how long?", "evidence": [proof],
         "standalone_query": "library loan period"},
        {"id": "outside", "kind": "negative", "question": "interplanetary travel", "evidence": []},
        {"id": "conflict", "kind": "conflict", "question": "What is the time limit?", "evidence": [
            {"source": "handbook.md", "contains": "14 days"},
            {"source": "student_union.md", "contains": "30 days"},
        ]},
    ]
    store = FakeStore()
    result = evaluate(store, cases, k=4, threshold=0.3, mode="hybrid", details=True)
    assert result["original_questions"]["evidence_hit_at_4"] == 0.6667
    assert result["original_questions"]["full_evidence_at_4"] == 0.3333
    assert result["original_questions"]["mrr_at_4"] == 0.5
    assert result["original_questions"]["negative_empty_retrieval_rate"] == 1.0
    assert result["conflict_questions"]["full_evidence_at_4"] == 0
    assert result["follow_ups"]["original"]["mrr_at_4"] == 0
    assert result["follow_ups"]["supplied_standalone"]["mrr_at_4"] == 0.5
    assert result["cases"][0]["evidence_ranks"] == [2]
    assert len(store.requests) == 5
    assert store.requests[-1] == ("library loan period", 4, 0.3, "hybrid", None)


def test_benchmark_manifest_has_matching_local_evidence_and_provenance():
    manifest_path = Path(__file__).resolve().parents[1] / "evaluation/demo_cases.json"
    manifest = load_manifest(manifest_path.read_bytes())
    chunks, files = load_corpus(manifest, manifest_path.parent)
    assert {case["kind"] for case in manifest["cases"]} == {"answerable", "negative", "follow_up", "conflict"}
    assert len({chunk["source"] for chunk in chunks}) == len(files)
    assert any("Repository text snapshot" in entry["provenance"] for entry in files)
    assert any("Synthetic fictional" in entry["provenance"] for entry in files)
    assert all(len(entry["sha256"]) == 64 for entry in files)


def test_positive_case_cannot_silently_become_a_negative():
    manifest = {
        "schema_version": 1,
        "description": "Synthetic test",
        "corpus": [],
        "cases": [{"id": "missing", "kind": "answerable", "question": "What time?", "evidence": []}],
    }
    with pytest.raises(ValueError, match="only negative cases"):
        load_manifest(json.dumps(manifest).encode())

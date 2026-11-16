from scripts.evaluate import evaluate


class FakeStore:
    def search(self, query, top_k=4, min_score=0.3):
        return ([{"source": "campus_faq.md"}] if "library" in query else [])


def test_retrieval_metrics_are_reported(capsys):
    result = evaluate(FakeStore(), [
        {"question": "library rules", "expected_source": "campus_faq.md"},
        {"question": "advising rules", "expected_source": "campus_faq.md"},
        {"question": "interplanetary travel", "expected_source": None},
    ], k=4, threshold=0.3)
    assert result["source_hit_at_4"] == 0.5
    assert result["retrieval_refusal_rate"] == 1.0

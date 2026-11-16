from types import SimpleNamespace

from campusguide_rag import RAGChatbot, REFUSAL_TEXT


class FakeStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=4, min_score=0.30):
        return self.hits


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))])


HITS = [{"source": "policy.pdf", "page": 2, "text": "Students borrow four books.", "score": 0.85}]


def test_no_sources_refuses_without_llm_call():
    client = FakeClient("Not needed")
    result = RAGChatbot(FakeStore([]), client).ask("Which books?")
    assert result["answer"] == REFUSAL_TEXT
    assert client.calls == []


def test_grounded_answer_includes_source():
    client = FakeClient("Students may borrow four books [1].")
    result = RAGChatbot(FakeStore(HITS), client).ask("How many books?")
    assert result["answer"].endswith("[1].")
    assert result["sources"][0]["page"] == 2
    assert "untrusted data" in client.calls[0]["messages"][0]["content"]


def test_missing_or_invalid_citations_fail_closed():
    for text in ("Students borrow four books.", "Students borrow four books [99].", ""):
        result = RAGChatbot(FakeStore(HITS), FakeClient(text)).ask("How many books?")
        assert result["answer"] == REFUSAL_TEXT


def test_question_length_guard():
    client = FakeClient("Something [1]")
    result = RAGChatbot(FakeStore(HITS), client).ask("q" * 1001)
    assert "1000" in result["answer"]
    assert client.calls == []

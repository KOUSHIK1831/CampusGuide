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


class RaisingClient(FakeClient):
    def __init__(self, error):
        super().__init__("")
        self.error = error

    def create(self, **kwargs):
        raise self.error


class ProviderStatusError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__("SECRET_PROVIDER_RESPONSE_DO_NOT_ECHO")


def test_default_llm_model_is_supported():
    client = FakeClient("Grounded answer [1].")
    RAGChatbot(FakeStore(HITS), client).ask("Question?")
    assert client.calls[0]["model"] == "openai/gpt-oss-20b"


def test_legacy_model_is_mapped_to_supported_model():
    client = FakeClient("Grounded answer [1].")
    RAGChatbot(FakeStore(HITS), client, model="llama-3.1-8b-instant").ask("Question?")
    assert client.calls[0]["model"] == "openai/gpt-oss-20b"


def test_provider_errors_are_actionable_and_do_not_echo_raw_responses():
    for code, expected in {401: "API key", 403: "denied", 429: "rate limit",
                           400: "model", 404: "model", 422: "model",
                           503: "unavailable"}.items():
        try:
            RAGChatbot(FakeStore(HITS), RaisingClient(ProviderStatusError(code))).ask("Question?")
            assert False, "Expected a sanitized RuntimeError"
        except RuntimeError as exc:
            assert expected.lower() in str(exc).lower()
            assert "SECRET_PROVIDER_RESPONSE_DO_NOT_ECHO" not in str(exc)


def test_unexpected_provider_exception_does_not_leak_secret():
    try:
        RAGChatbot(FakeStore(HITS), RaisingClient(ValueError("SECRET_TOKEN_DO_NOT_ECHO"))).ask("Question?")
        assert False, "Expected a sanitized RuntimeError"
    except RuntimeError as exc:
        assert "SECRET_TOKEN_DO_NOT_ECHO" not in str(exc)
        assert "Groq" in str(exc)

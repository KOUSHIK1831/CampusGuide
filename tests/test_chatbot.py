import json
from types import SimpleNamespace

import httpx
import pytest
from groq import Groq

from campusguide_rag import RAGChatbot, REFUSAL_TEXT


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, top_k=4, min_score=0.30):
        self.queries.append(query)
        if isinstance(self.hits, dict):
            return self.hits.get(query, [])
        return self.hits


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answer.pop(0) if isinstance(self.answer, list) else self.answer
        if isinstance(answer, Exception):
            raise answer
        return {"choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}


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


def test_missing_or_invalid_citations_are_explained_not_refused():
    for text, expected in (("Students borrow four books.", "Students borrow four books."),
                           ("Students borrow four books [99].", "Students borrow four books.")):
        result = RAGChatbot(FakeStore(HITS), FakeClient(text)).ask("How many books?")
        assert result["answer"] == expected
        assert result["sources"] == [{
            "ref": 1, "source": "policy.pdf", "page": 2, "score": 0.85,
            "excerpt": "Students borrow four books.",
        }]
        assert result["citations_verified"] is False
        assert any("citation checks" in item for item in result["notices"])

    result = RAGChatbot(FakeStore(HITS), FakeClient("")).ask("How many books?")
    assert result["answer"] == REFUSAL_TEXT
    assert any("empty" in item.lower() for item in result["notices"])


def test_uncited_answer_is_repaired_with_citations():
    draft = "Koushik has a B.Tech from ICFAI University."
    repaired = "Koushik has a B.Tech from ICFAI University [1]."
    client = FakeClient([draft, repaired])

    result = RAGChatbot(FakeStore(HITS), client).ask("education of koushik")

    assert result["answer"] == repaired
    assert result["citations_verified"] is True
    assert result["sources"][0]["excerpt"] == HITS[0]["text"]
    assert len(client.calls) == 2
    assert draft in client.calls[1]["messages"][1]["content"]


def test_model_refusal_explains_insufficient_evidence():
    result = RAGChatbot(FakeStore(HITS), FakeClient(REFUSAL_TEXT)).ask("How many books?")
    assert result["answer"] == REFUSAL_TEXT
    assert any("did not find" in item.lower() for item in result["notices"])


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


def test_citations_include_all_passage_text_sent_to_llm():
    """Supporting facts after character 400 must still be visible in the UI."""
    passage = ("This is background context. " * 24) + "Deadline is May 24."
    assert 400 < len(passage) < 1250
    hit = {"source": "calendar.pdf", "page": 7, "text": passage, "score": 0.97}
    client = FakeClient("The deadline is May 24 [1].")
    answer = RAGChatbot(FakeStore([hit]), client).ask("What is the deadline?")
    assert answer["sources"][0]["excerpt"] == passage
    assert passage in client.calls[0]["messages"][1]["content"]
    assert "May 24" in answer["sources"][0]["excerpt"]


def test_rewrite_uses_bounded_completed_history_and_keeps_original_question():
    original = "How many can I borrow?"
    standalone = "How many library books can I borrow?"
    history = [
        {"role": role, "content": f"{role}-{i} " + ("x" * 1200), "sources": ["never forwarded"]}
        for i in range(7) for role in ("user", "assistant")
    ]
    history += [{"role": "user", "content": "unanswered turn"}]
    client = FakeClient([standalone, "Students may borrow four books [1]."])
    store = FakeStore({standalone: HITS})
    result = RAGChatbot(store, client).ask(original, history=history)

    rewrite_prompt = client.calls[0]["messages"][1]["content"]
    prior = json.loads(rewrite_prompt.split("\n\nCurrent question:")[0].split("\n", 1)[1])
    assert 0 < len(prior) <= 8
    assert all(set(message) == {"role", "content"} for message in prior)
    assert all(len(message["content"]) <= 1000 for message in prior)
    assert "assistant-6" in rewrite_prompt
    assert "user-0" not in rewrite_prompt
    assert "unanswered turn" not in rewrite_prompt
    assert store.queries == [standalone, original]
    assert f"Original question: {original}" in client.calls[1]["messages"][1]["content"]
    assert "assistant-6" not in client.calls[1]["messages"][1]["content"]
    assert result["search_query"] == standalone
    assert result["rewrite_status"] == "rewritten"


def test_rewritten_follow_up_fuses_original_question_evidence():
    original = "skills of koushik"
    standalone = "What are Koushik's skills?"
    rewrite_hit = {"source": "resume.pdf", "page": 1, "text": "Profile summary.", "score": 0.60}
    original_hit = {"source": "resume.pdf", "page": 1, "text": "Skills: Python, SQL.", "score": 0.40}
    client = FakeClient([standalone, "Koushik knows Python [2]."])
    store = FakeStore({standalone: [rewrite_hit], original: [original_hit]})
    history = [
        {"role": "user", "content": "Tell me about this resume"},
        {"role": "assistant", "content": "Koushik is a data analyst [1]."},
    ]

    result = RAGChatbot(store, client).ask(original, history=history)

    assert store.queries == [standalone, original]
    assert "Skills: Python, SQL." in client.calls[1]["messages"][1]["content"]
    assert result["sources"] == [{
        "ref": 2, "source": "resume.pdf", "page": 1, "score": 0.40, "excerpt": "Skills: Python, SQL.",
    }]
    assert result["rewrite_status"] == "rewritten"


@pytest.mark.parametrize(
    ("rewrite", "queries"),
    [
        ("Wrong topic?", ["Wrong topic?", "What about this one?"]),
        ("line 1\nline 2", ["What about this one?"]),
        (ProviderStatusError(503), ["What about this one?"]),
    ],
)
def test_unusable_rewrite_recovers_to_original_question(rewrite, queries):
    question = "What about this one?"
    store = FakeStore({question: HITS})
    client = FakeClient([rewrite, "Students may borrow four books [1]."])
    history = [{"role": "user", "content": "Library?"}, {"role": "assistant", "content": "Books [1]."}]

    result = RAGChatbot(store, client).ask(question, history=history)

    assert store.queries == queries
    assert result["search_query"] == question
    assert result["rewrite_status"] == "fallback"
    assert result["sources"]


def test_follow_up_with_no_direct_match_retries_with_previous_question():
    class ContextStore:
        def __init__(self):
            self.queries = []

        def search(self, query, top_k=4, min_score=0.30):
            self.queries.append(query)
            return HITS if query == "Hi education Koushik skills" else []

    store = ContextStore()
    client = FakeClient(["Hi education", "Koushik has Python skills [1]."])
    history = [
        {"role": "user", "content": "Koushik skills"},
        {"role": "assistant", "content": "Koushik has Python skills [1]."},
    ]

    result = RAGChatbot(store, client).ask("Hi education", history=history)

    assert store.queries == ["Hi education", "Hi education Koushik skills"]
    assert result["search_query"] == "Hi education Koushik skills"
    assert result["rewrite_status"] == "context_fallback"
    assert result["sources"]
    assert any("previous question" in notice for notice in result["notices"])


def test_short_weak_follow_up_searches_with_previous_question():
    header = {"source": "resume.pdf", "page": 1, "text": "Profile summary.", "score": 0.277}
    skills = {"source": "resume.pdf", "page": 1, "text": "Skills: Python, SQL.", "score": 0.223}

    class ContextStore:
        def __init__(self):
            self.queries = []

        def search(self, query, top_k=4, min_score=0.30):
            self.queries.append(query)
            if query == "skills":
                return [header]
            if query == "skills education of koushik":
                return [{**header, "score": 0.405}, skills]
            return []

    store = ContextStore()
    client = FakeClient(["skills", "Koushik knows Python [2]."])
    history = [
        {"role": "user", "content": "education of koushik"},
        {"role": "assistant", "content": "ICFAI University [1]."},
    ]

    result = RAGChatbot(store, client).ask("skills", history=history)

    assert store.queries == ["skills", "skills education of koushik"]
    assert result["rewrite_status"] == "context_fallback"
    assert "Skills: Python, SQL." in client.calls[1]["messages"][1]["content"]
    assert [source["excerpt"] for source in result["sources"]] == ["Skills: Python, SQL."]


@pytest.mark.parametrize(
    ("ending", "finish", "expected", "has_sources", "verified"),
    [(" [1].", "stop", "Four books [1].", True, True),
     (" [99].", "stop", "Four books.", True, False),
     (" [1].", "length", REFUSAL_TEXT, False, False),
     (" [1].", None, REFUSAL_TEXT, False, False)],
)
def test_groq_sdk_stream_via_langchain_validates_complete_response(
    ending, finish, expected, has_sources, verified,
):
    deltas = []

    def post_completion(request):
        params = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer test-session-key"
        assert params["stream"] is True
        assert params["include_reasoning"] is False
        assert params["reasoning_effort"] == "low"
        frames = [
            ({"role": "assistant", "reasoning": "hidden text"}, None),
            ({"content": "Four books"}, None),
            ({"content": ending}, None),
            ({}, finish),
        ]
        body = [
            "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": reason}]})
            for delta, reason in frames
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="\n\n".join([*body, "data: [DONE]", ""]))

    transport = httpx.MockTransport(post_completion)
    with Groq(api_key="test-session-key", http_client=httpx.Client(transport=transport)) as client:
        result = RAGChatbot(FakeStore(HITS), client).ask("How many?", on_delta=deltas.append)

    assert deltas == ["Four books", ending]
    assert result["answer"] == expected
    assert bool(result["sources"]) == has_sources
    assert result["citations_verified"] is verified


def test_provider_error_after_partial_stream_is_sanitized():
    class InterruptedClient(FakeClient):
        def create(self, **kwargs):
            yield {"choices": [{"delta": {"role": "assistant", "content": "Partial [1]"}}]}
            raise ProviderStatusError(503)

    deltas = []
    with pytest.raises(RuntimeError, match="temporarily unavailable") as error:
        RAGChatbot(FakeStore(HITS), InterruptedClient("")).ask("How many?", on_delta=deltas.append)
    assert deltas == ["Partial [1]"]
    assert "SECRET_PROVIDER_RESPONSE" not in str(error.value)


def test_policy_metadata_and_conflict_guidance_reach_generation_and_citations():
    old = {**HITS[0], "policy_id": "LIB-001", "version": "1", "effective_date": "2024-09-01"}
    new = {**old, "source": "library-v2.pdf", "version": "2", "effective_date": "2025-01-01",
           "text": "Students borrow six books."}
    client = FakeClient("The 2025 version states six books [2]. The 2024 version states four [1].")
    result = RAGChatbot(FakeStore([old, new]), client).ask("How did the rule change?")
    system, user = client.calls[0]["messages"]
    assert "conflicting guidance" in system["content"]
    assert '"effective_date": "2025-01-01"' in user["content"]
    assert [source["version"] for source in result["sources"]] == ["1", "2"]
    assert all(source["policy_id"] == "LIB-001" for source in result["sources"])

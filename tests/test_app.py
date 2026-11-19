"""Test the Streamlit chat lifecycle with a simulated provider and index."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

import campusguide_rag
from campusguide_rag import REFUSAL_TEXT


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PARTIAL_ANSWER = "Unvalidated draft content"
PROVIDER_ERROR = "Groq could not complete the request."


class ReadyStore:
    ready = True

    def stats(self):
        return {"documents": 1, "chunks": 1}


def test_chat_commits_final_responses_and_reuses_only_successful_turns(monkeypatch):
    calls = []
    sources = [{
        "ref": 1, "source": "exam.md", "page": 2, "score": 0.72,
        "excerpt": "A makeup application is due Friday.", "version": "2",
        "effective_date": "2026-02-01",
    }]

    class FakeChatbot:
        def __init__(self, store, *, api_key, retrieval_mode, **kwargs):
            assert isinstance(store, ReadyStore)
            assert api_key == "mock-key"
            assert retrieval_mode == "hybrid"

        def ask(self, question, *, history, on_delta):
            calls.append((question, history))
            on_delta(PARTIAL_ANSWER)
            if question == "Provider error":
                raise RuntimeError(PROVIDER_ERROR)
            if question == "Unanswerable question":
                return {"answer": REFUSAL_TEXT, "sources": []}
            return {"answer": "The application is due Friday [1].", "sources": sources}

    monkeypatch.setattr(campusguide_rag, "RAGChatbot", FakeChatbot)
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("RERANKER_MODEL", "")
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["document_store"] = ReadyStore()
    app.session_state["byok_api_key"] = "mock-key"
    app.run()

    questions = ["When is my exam application due?", "Unanswerable question",
                 "Provider error", "What about that application?"]
    for question in questions:
        app.chat_input[0].set_value(question).run()
        assert not app.exception
        assert all(node.value != PARTIAL_ANSWER for node in app.text)
        assert all("Draft answer" not in node.value for node in app.caption)

    assert calls[0] == (questions[0], [])
    assert calls[-1] == (questions[-1], [
        {"role": "user", "content": questions[0]},
        {"role": "assistant", "content": "The application is due Friday [1]."},
    ])
    responses = [message["content"] for message in app.session_state["chat_history"]
                 if message["role"] == "assistant"]
    assert responses == ["The application is due Friday [1].", REFUSAL_TEXT,
                         PROVIDER_ERROR, "The application is due Friday [1]."]
    assert [node.value for node in app.markdown] == responses
    assert [source.label for source in app.expander] == ["[1] exam.md · page 2"] * 2

    app.session_state["upload_signature"] = "stale-files-fingerprint"
    app.run()
    assert "document_store" not in app.session_state
    assert app.session_state["chat_history"] == []
    assert app.chat_input[0].disabled
    assert not app.chat_message

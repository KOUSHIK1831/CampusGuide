"""CampusGuide: per-session document Q&A with a bring-your-own Groq API key."""
from __future__ import annotations

import os
from itertools import pairwise

import streamlit as st
from dotenv import load_dotenv

from campusguide_rag import RAGChatbot, VectorStore
from campusguide_rag.chatbot import MAX_HISTORY_TURNS
from campusguide_rag.retrieval import CrossEncoderReranker
from campusguide_rag.upload_workflow import (
    MAX_UPLOAD_FILES, prepare_uploaded_documents, upload_fingerprint,
)

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
MIN_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.30"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "").strip()


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedding_model(model_name: str):
    """Share only model weights: never cache an index, uploaded text, or API key."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource
def get_reranker(model_name: str) -> CrossEncoderReranker:
    """Cache the scorer with model weights only, never its inputs or results."""
    return CrossEncoderReranker(model_name)


def recent_conversation(messages: list[dict]) -> list[dict]:
    """Return bounded, complete turns with finalized responses and sources."""
    recent = []
    for assistant, user in pairwise(reversed(messages)):
        if (assistant["role"] != "assistant" or user["role"] != "user"
                or not assistant.get("sources")):
            continue
        recent.extend([
            {"role": "assistant", "content": assistant["content"]},
            {"role": "user", "content": user["content"]},
        ])
        if len(recent) == 2 * MAX_HISTORY_TURNS:
            break
    return recent[::-1]


def render_message(message: dict) -> None:
    if message["role"] == "user":
        st.text(message["content"])
        return
    st.markdown(message["content"])
    if message.get("sources") and message.get("citations_verified", True):
        st.caption("Citation references checked.")
    for notice in message.get("notices", []):
        st.caption(notice)
    for source in message.get("sources", []):
        label = f"[{source['ref']}] {source['source']}"
        if source.get("page"):
            label += f" · page {source['page']}"
        with st.expander(label):
            details = []
            if source.get("score") is not None:
                details.append(f"Cosine similarity {source['score']:.3f}")
            for field, name in (("policy_id", "Policy"), ("version", "Version"),
                                ("effective_date", "Effective")):
                if source.get(field):
                    details.append(f"{name} {source[field]}")
            if details:
                st.text(" · ".join(details))
            st.code(source["excerpt"], language=None)


def clear_my_session() -> None:
    """Forget one visitor's API key, uploaded widget, indexed corpus, and chat."""
    for key in ("byok_api_key", "uploaded_documents", "document_store",
                "upload_signature", "chat_history"):
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title="CampusGuide | Chat with Documents", page_icon="📚", layout="wide")
    st.title("📚 CampusGuide — Chat with your documents")
    st.caption("Upload files, bring your own Groq API key, and ask questions with source citations.")

    with st.sidebar:
        st.subheader("1. Your Groq API key")
        api_key = st.text_input(
            "Groq API key", type="password", key="byok_api_key",
            placeholder="gsk_...", help="Your key stays in this Streamlit browser session and is never saved by this app.",
        )
        st.caption("Get a key at https://console.groq.com/keys. Uses GPT-OSS 20B by default; Groq account limits may apply.")
        st.subheader("2. Upload documents")
        uploads = st.file_uploader(
            "Choose PDF, TXT, DOCX, Markdown or JSON files",
            type=["pdf", "txt", "docx", "md", "markdown", "json"],
            accept_multiple_files=True, key="uploaded_documents",
            help=f"Up to {MAX_UPLOAD_FILES} files; 15 MB per file, 40 MB combined. PDFs need extractable text.",
        )
        selected = [(item.name, item.getvalue()) for item in (uploads or [])]
        current_signature = upload_fingerprint(selected)
        if current_signature != st.session_state.get("upload_signature"):
            # Never show answers or sources from documents that are no longer selected.
            st.session_state.pop("document_store", None)
            st.session_state.pop("upload_signature", None)
            st.session_state["chat_history"] = []
        if st.button("Process uploaded documents", type="primary", disabled=not selected, use_container_width=True):
            try:
                with st.spinner("Extracting and indexing your documents..."):
                    chunks, doc_count = prepare_uploaded_documents(selected)
                    store = VectorStore(
                        model_name=EMBEDDING_MODEL,
                        encoder=get_embedding_model(EMBEDDING_MODEL),
                        backend=BACKEND,
                    )
                    store.build(chunks)
                st.session_state["document_store"] = store
                st.session_state["upload_signature"] = current_signature
                st.session_state["chat_history"] = []
                st.success(f"Ready: {doc_count} document(s), {len(chunks)} chunks.")
            except (ValueError, OSError, RuntimeError, ImportError) as exc:
                st.error(f"Could not process the documents: {exc}")
        store = st.session_state.get("document_store")
        if store and store.ready:
            stats = store.stats()
            st.metric("Indexed documents", stats["documents"])
            st.metric("Indexed chunks", stats["chunks"])
        st.button("Clear my session", on_click=clear_my_session, use_container_width=True)
        st.caption(
            "Your uploads and index stay in this app session’s memory. CampusGuide sends "
            "your question, relevant excerpts, and recent conversation context to Groq "
            "when needed to answer a question or interpret a follow-up. "
            "Avoid uploading sensitive documents."
        )

    messages = st.session_state.setdefault("chat_history", [])
    for message in messages:
        with st.chat_message(message["role"]):
            render_message(message)

    store = st.session_state.get("document_store")
    if not store or not store.ready:
        st.info("Start by uploading documents and clicking **Process uploaded documents** in the sidebar.")
    if not api_key.strip():
        st.info("Enter your own Groq API key in the sidebar to enable answering questions.")
    question = st.chat_input(
        "Ask a question about your uploaded files...",
        disabled=not store or not store.ready or not api_key.strip(),
    )
    if not question:
        return

    user_message = {"role": "user", "content": question}
    with st.chat_message("user"):
        render_message(user_message)
    with st.chat_message("assistant"):
        draft = st.empty()
        with draft.container():
            status = st.empty()
            status.caption("Searching your documents…")
            preview = st.empty()
        deltas: list[str] = []

        def show_delta(delta: str) -> None:
            if not deltas:
                status.caption("Draft answer · citation checks pending")
            deltas.append(delta)
            preview.text("".join(deltas))

        try:
            reranker = None
            if RERANKER_MODEL and RETRIEVAL_MODE == "hybrid":
                reranker = get_reranker(RERANKER_MODEL)
            bot = RAGChatbot(
                store, api_key=api_key.strip(), model=LLM_MODEL, min_score=MIN_SCORE,
                retrieval_mode=RETRIEVAL_MODE, reranker=reranker,
            )
            result = bot.ask(question, history=recent_conversation(messages), on_delta=show_delta)
            assistant_message = {
                "role": "assistant", "content": result["answer"], "sources": result["sources"],
                "notices": result.get("notices", []),
                "citations_verified": result.get("citations_verified", False),
            }
        except RuntimeError as exc:
            assistant_message = {"role": "assistant", "content": str(exc), "sources": []}
        except ValueError:
            assistant_message = {
                "role": "assistant", "content": "CampusGuide could not process this request.",
                "sources": [],
            }
        finally:
            draft.empty()
        render_message(assistant_message)
    messages.extend([user_message, assistant_message])


if __name__ == "__main__":
    main()

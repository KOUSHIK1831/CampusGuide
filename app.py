"""CampusGuide: per-session document Q&A with a bring-your-own Groq API key."""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from campusguide_rag import RAGChatbot, VectorStore
from campusguide_rag.upload_workflow import (
    MAX_UPLOAD_FILES, prepare_uploaded_documents, upload_fingerprint,
)

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
MIN_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.30"))


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedding_model(model_name: str):
    """Share only model weights: never cache an index, uploaded text, or API key."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device="cpu")


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
        st.caption("Uploaded content and its index are kept in this session's memory, not written by the app to disk. Avoid sensitive documents. Relevant excerpts are sent to Groq when you ask a question.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                label = f"[{source['ref']}] {source['source']}"
                if source.get("page"):
                    label += f" · page {source['page']}"
                with st.expander(label):
                    st.caption(f"Similarity: {source['score']:.3f}")
                    st.code(source["excerpt"], language=None)

    store = st.session_state.get("document_store")
    if not store or not store.ready:
        st.info("Start by uploading documents and clicking **Process uploaded documents** in the sidebar.")
    if not api_key.strip():
        st.info("Enter your own Groq API key in the sidebar to enable answering questions.")
    question = st.chat_input(
        "Ask a question about your uploaded files...",
        disabled=not store or not store.ready or not api_key.strip(),
    )
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        try:
            from groq import Groq
            # Never place the user's key in os.environ, cache_resource, disk, or logs.
            # Create a client for the current user's request only.
            with st.spinner("Searching the uploaded documents and generating an answer..."):
                bot = RAGChatbot(
                    store, llm_client=Groq(api_key=api_key.strip()),
                    model=LLM_MODEL, min_score=MIN_SCORE,
                )
                result = bot.ask(question)
            st.session_state["chat_history"].append({
                "role": "assistant", "content": result["answer"], "sources": result["sources"],
            })
        except (RuntimeError, ValueError) as exc:
            st.session_state["chat_history"].append({
                "role": "assistant", "content": str(exc), "sources": [],
            })
        st.rerun()


if __name__ == "__main__":
    main()

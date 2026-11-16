"""Streamlit entry point for the CampusGuide RAG application."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from campusguide_rag import (
    RAGChatbot, SUPPORTED_EXTENSIONS, VectorStore,
    chunk_documents, load_directory,
)
from campusguide_rag.ingestion import MAX_FILE_BYTES

load_dotenv()
RAW_DIR = Path(os.getenv("DATA_RAW_PATH", "data/raw"))
INDEX_DIR = Path(os.getenv("VECTOR_STORE_PATH", "data/processed"))
BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading the local vector index...")
def get_store() -> VectorStore:
    store = VectorStore(INDEX_DIR, model_name=EMBEDDING_MODEL, backend=BACKEND)
    try:
        loaded = store.load()
    except (OSError, ValueError, ImportError) as exc:
        st.warning(f"Stored index cannot be loaded ({exc}). Rebuild it from source documents.")
        loaded = False
    # Hosted demo uses only bundled fictional content (data/examples); uploads are disabled.
    if not loaded and os.getenv("AUTO_INGEST_ON_START", "false").lower() == "true":
        try:
            rebuild(store)
        except (ValueError, OSError, ImportError) as exc:
            st.warning(f"Automatic demo indexing failed: {exc}")
    return store


def rebuild(store: VectorStore) -> tuple[int, int, list[str]]:
    units, errors = load_directory(RAW_DIR)
    chunks = chunk_documents(units)
    if not chunks:
        raise ValueError("No extractable documents. Add TXT, MD, JSON, DOCX or text-based PDF files.")
    store.build(chunks)
    store.save()
    return len({c["source"] for c in chunks}), len(chunks), errors


def save_uploads(files: list) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for item in files:
        # Both Linux and Windows separators are stripped; only a filename is stored.
        name = item.name.replace("\\", "/").rsplit("/", 1)[-1]
        if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            st.warning(f"Skipped unsupported file: {name}")
            continue
        data = item.getvalue()
        if len(data) > MAX_FILE_BYTES:
            st.warning(f"Skipped {name}: exceeds 15 MB limit")
            continue
        (RAW_DIR / name).write_bytes(data)
        saved += 1
    return saved


def main() -> None:
    st.set_page_config(page_title="CampusGuide | Document RAG", page_icon="🎓", layout="wide")
    st.title("🎓 CampusGuide")
    st.caption("Ask questions about locally indexed documents. Answers include retrieval citations, not guarantees of correctness.")
    store = get_store()
    with st.sidebar:
        st.header("Knowledge base")
        uploads_enabled = os.getenv("ALLOW_UPLOADS", "true").lower() == "true"
        uploads = None
        if uploads_enabled:
            uploads = st.file_uploader(
                "Add documents", type=["pdf", "txt", "docx", "md", "markdown", "json"],
                accept_multiple_files=True,
                help="Text-based PDFs only; maximum 15 MB per file. Duplicate filenames are replaced.",
            )
        else:
            st.info("Hosted demo: uploads are disabled; only fictional sample documents are indexed.")
        if uploads_enabled and st.button("Save uploads and rebuild", type="primary", disabled=not uploads):
            try:
                saved = save_uploads(uploads)
                if saved:
                    with st.spinner("Embedding all documents..."):
                        docs, chunks, errors = rebuild(store)
                    st.success(f"Indexed {docs} document(s), {chunks} chunks")
                    for error in errors:
                        st.warning(error)
                    st.session_state["chat_history"] = []
                else:
                    st.warning("No supported documents were uploaded")
            except (ValueError, OSError, ImportError) as exc:
                st.error(f"Could not rebuild: {exc}")
        if uploads_enabled and st.button("Rebuild index from data/raw"):
            try:
                with st.spinner("Embedding corpus..."):
                    docs, chunks, errors = rebuild(store)
                st.success(f"Indexed {docs} document(s), {chunks} chunks")
                for error in errors:
                    st.warning(error)
                st.session_state["chat_history"] = []
            except (ValueError, OSError, ImportError) as exc:
                st.error(f"Could not rebuild: {exc}")
        stats = store.stats()
        st.metric("Indexed documents", stats["documents"])
        st.metric("Indexed chunks", stats["chunks"])
        st.caption(f"Vector backend: {stats['backend']}")
        st.caption("This demo has no authentication or document-level authorization. Use non-sensitive documents only.")

    if not store.ready:
        st.info("Index a document using the sidebar or run `python scripts/ingest.py` first.")
        return

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                with st.expander(f"[{source['ref']}] {source['source']}" + (f" · page {source['page']}" if source['page'] else "")):
                    st.caption(f"Similarity: {source['score']:.3f}")
                    st.code(source["excerpt"], language=None)

    # Session quota reduces accidental usage; this is NOT a global security/rate-limit control.
    max_demo_queries = int(os.getenv("MAX_DEMO_QUERIES", "0"))
    used_queries = sum(m["role"] == "user" for m in st.session_state["chat_history"])
    quota_reached = max_demo_queries > 0 and used_queries >= max_demo_queries
    if quota_reached:
        st.info("Demo session query limit reached. Start a new browser session to try again.")
    prompt = st.chat_input("What do the documents say about ...?", disabled=quota_reached)
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                bot = RAGChatbot(store, model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
                                 min_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.30")))
                response = bot.ask(prompt)
            st.session_state["chat_history"].append({
                "role": "assistant", "content": response["answer"], "sources": response["sources"]
            })
        except (RuntimeError, ValueError) as exc:
            st.session_state["chat_history"].append({"role": "assistant", "content": str(exc), "sources": []})
        st.rerun()


if __name__ == "__main__":
    main()

# CampusGuide delivery roadmap and audit

## Baseline audit (main branch inspected 2026-11-16)

Existing project: Python 3.12 / Streamlit / Groq / Sentence Transformers / FAISS, pre-existing text corpus and modular folders. The original document loader only handled `.txt`; sidebar advertised PDF although uploads passed through a mismatched loader. The old generator referenced `st.secrets` without importing `streamlit`, while `.env.example` was configured for `GROQ_API_KEY`. The original UI and ingestor held different vector-store instances; new indexes could be stale for queries. There were no `tests/` unit tests. The old README described OCR, guaranteed hallucination prevention, role-based access and deployed behaviors without matching implementation.

This update introduces `campusguide_rag/` as the maintained pipeline and rewrites `app.py`, preserving old code files for audit/history. No production deployment or real access-control claims are made.

## Phase 1 — MVP, implemented

- Extract text from TXT/Markdown/JSON/DOCX and text-based PDFs; reject unreadable documents and scanned PDFs.
- Chunk and index with a local sentence encoder and FAISS `IndexFlatIP`; normalize vectors to use cosine similarity.
- Search top k with a configurable threshold, send numbered excerpts to Groq, and display cited passages.
- Return no-evidence refusal when retrieval is empty or response citations are invalid.
- Full index rebuild on upload; initial indexing via CLI or sidebar.

## Phase 2 — quality, partially implemented

**Included** in the [RAG upgrade](RAG_UPGRADE.md) are follow-up rewriting, hybrid retrieval, optional CPU reranking, streaming, and excerpt-level evaluation of hits, ranks, evidence coverage, and empty retrieval. Tests cover both vector backends and exercise mocked provider streams and Streamlit state in CI.

**Next** are independently reviewed evaluation questions, citation-support annotation, live rewrite and answer evaluation, model latency and cost measurements, and accessibility and chat-history controls.

## Phase 3 — shipping, partial

**Included:** reproducible pip setup, Dockerfile, Render Blueprint (`render.yaml`) for a public fictional-only demo, demo corpus and prompts, example dialogue, README and deployment guide.

**Next:** authenticated hosted launch, live smoke tests, persistent volume, authentication/authorization, request quotas, upload security scanning, observability, secrets management, privacy impact review, API timeout/retries, concurrency tests, dependency lock renewal.

## Engineering choices

- **Reuse vs rewrite:** preserve existing legacy modules but isolate a small supported pipeline; avoid advertising legacy experimental paths as active features.
- **FAISS vs hosted vectors:** FAISS is free, portable, and fast for a small static corpus; it cannot provide multi-tenant security, managed incremental updates or shared persistence on its own.
- **MiniLM vs hosted embeddings:** local model avoids per-document API costs and works offline after initial download; its accuracy and multilingual quality require domain evaluation.
- **Guardrails:** threshold filtering, context-only prompts, explicit instruction/data separation and numeric citation checks. These are best-effort checks, **not** formal proof that every answer is supported.
- **Session isolation** keeps the upload index and conversation in the visitor's session. A public demo has no account-level access control or malware scanning. Groq receives selected excerpts and bounded rewrite dialogue.
- **Evaluation** checks for labeled excerpts at the retrieved ranks. Claim correctness requires separate evaluation.

## Demo checklist

1. Copy `data/examples/campus_faq.md` into `data/raw/` and run the ingestion CLI.
2. Launch Streamlit locally; ask an answerable library question and open the citation expander.
3. Ask an unrelated question, and confirm a refusal **if no relevant chunks pass the threshold**.
4. Run pytest and offline evaluation, recording actual counts, runtime and environment.
5. Explain the flow, the threshold tradeoff, where retrieved text is sent, and how you would reduce hallucination risks further.

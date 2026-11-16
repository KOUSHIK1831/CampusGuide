# CampusGuide — source-cited RAG chatbot

**A reproducible academic RAG application:** ingest campus-related documents, create local embeddings and a FAISS index, retrieve relevant passages, and ask Groq's Llama model to answer with numbered source citations. Built with **Python 3.12, Streamlit, Sentence Transformers, FAISS, and Groq**.

The repository also contains earlier experimental modules (`ingestion/`, `embeddings/`, `retriever/`, `llm/`, `validation/`, `ui/`). The **maintained entry point** is `app.py`, backed by the new `campusguide_rag/` package. The older modules are retained as history, not wired into the new UI.

## Implemented capabilities

- PDF (extractable text), DOCX, TXT, Markdown, and JSON ingestion; per-document source metadata and PDF page tracking.
- Overlapping chunks, offline/local `all-MiniLM-L6-v2` embeddings, normalized FAISS inner-product vector search (cosine similarity), and JSON metadata persistence.
- Retrieval score gating, context-only generation with Groq Llama, citation checks, safe fallback for missing evidence, and example prompts.
- Streamlit uploads and knowledge-base rebuilding; pure-Python CLI ingestion, offline retrieval evaluation, unit tests, GitHub Actions CI, and Docker.

**Important limitations:** Citations and guardrails lower hallucination risk but **do not guarantee factual accuracy**. Citation checks validate references, not entailment. This demo has no login, actual role-based authorization, document versioning, OCR for scanned PDFs, or production hardening. Do not upload confidential data; retrieved excerpts are sent to Groq when you ask a question.

## Quick start

Requires **Python 3.12+** and an internet connection on first run to download the embedding model; Groq generation requires a `GROQ_API_KEY`.

```bash
git clone https://github.com/KOUSHIK1831/CampusGuide.git
cd CampusGuide
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # on Windows: copy .env.example .env
# Edit .env and enter GROQ_API_KEY
```

The repository includes sample raw data in `data/raw/`. For a clearly fictional, reproducible demo, first copy `data/examples/campus_faq.md` to `data/raw/campus_faq.md`:

```bash
cp data/examples/campus_faq.md data/raw/campus_faq.md
python scripts/ingest.py
streamlit run app.py
```

Open `http://localhost:8501`. Alternatively, upload documents through the sidebar and click **Save uploads and rebuild**. The index is stored in the ignored `data/processed/` directory. New uploads trigger a full rebuild; this is intentional for a small MVP corpus.

Supported inputs: `.pdf` (text-based only), `.txt`, `.docx`, `.md`, `.markdown`, `.json`. A JSON top-level list or `{ "documents": [...] }` is ingested record by record. Files larger than 15 MB are rejected. Current UI uploads overwrite a prior file of the same name.

## Run tests and retrieval evaluation

```bash
pip install 'pytest>=8,<10'
python -m pytest -q
python scripts/evaluate.py --cases evaluation/demo_cases.json --k 4
```

**Evaluation meaning:** `source_hit_at_4` measures whether the expected document appears in the top four retrieved chunks; `retrieval_refusal_rate` measures how often a negative query returns no retrieved chunks. It does **not** measure answer accuracy, grounding, hallucination rate, or user satisfaction. `evaluation/demo_cases.json` assumes the fictional `campus_faq.md` file was indexed. Record actual results after running; do not publish unmeasured percentages as results. Tune `MIN_RETRIEVAL_SCORE` using a balanced set of positive and negative questions.

## Example conversation (illustrative, not a recorded live response)

**User:** How many library books can I borrow?

**CampusGuide:** Students may borrow up to four library books at once, for fourteen days. [1]

**Source [1]:** `campus_faq.md`, Library section.

**User:** What is the university's shuttle service schedule?

**CampusGuide:** The requested information is not available in the provided documents.

The example knowledge base is **fictional**, not an official university document. Actual responses depend on retrieved chunks and the selected LLM.

## Architecture

```text
TXT / MD / JSON / PDF / DOCX
       │
       ▼
Per-file extraction + page metadata
       │
       ▼
Character chunks with overlap
       │
       ▼
Sentence Transformers (normalized embeddings)
       │
       ▼
FAISS IndexFlatIP + JSON source metadata
       │
Question ──► embedding ──► top-k similarity gate
                                      │ no hits: refuse
                                      ▼
                           Untrusted context excerpts
                                      ▼
                           Groq Llama + grounded prompt
                                      ▼
                           Citation number verification
                                      ▼
                           Answer + source excerpts
```

Persistence uses JSON and FAISS's native index format, **not pickle**. Rebuild indexes when changing the embedding model. For a non-FAISS local experiment or unit tests, set `VECTOR_BACKEND=numpy` before ingesting; this stores vectors as `.npy` and uses exact NumPy cosine search.

## Configuration

Edit `.env` using the keys listed in `.env.example`: `GROQ_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, `VECTOR_BACKEND`, `VECTOR_STORE_PATH`, `DATA_RAW_PATH`, `MIN_RETRIEVAL_SCORE`. Never commit secrets. The outdated `uv.lock` was removed because Streamlit Cloud prioritizes it over the active root `requirements.txt`. Only regenerate it if intentionally maintaining a synchronized lockfile.

## Docker (local demonstration)

```bash
docker build -t campusguide-rag .
docker run --rm -p 8501:8501 --env-file .env -v "$(pwd)/data:/app/data" campusguide-rag
```

Create `data/processed` locally first if your platform needs it, and ensure the mount is writable by the container user (UID 10001). After starting, upload documents or use the sidebar's **Rebuild index** button. The first embedding-model download may take time. Docker runs **locally**; no hosted deployment is claimed.

## Deploy on Streamlit Community Cloud (primary)

CampusGuide is prepared for **Streamlit Community Cloud**. Deploy from `KOUSHIK1831/CampusGuide`, branch `main`, entrypoint `app.py`, with **Python 3.12**. Add a fresh `GROQ_API_KEY` under Streamlit Cloud's **Advanced settings → Secrets**. See [docs/STREAMLIT_CLOUD.md](docs/STREAMLIT_CLOUD.md) for setup, security guidance, troubleshooting, and live-app verification.

**Pushing to GitHub does not provision a Streamlit Cloud app.** The repository owner must create it at https://share.streamlit.io/ and obtain the actual assigned `*.streamlit.app` URL. Do not publish an unverified address.

Without local overrides, the hosted demo automatically indexes **only fictional** content from `data/examples`, disables uploads, and limits each browser session to 10 questions. For local document uploads, copy `.env.example` to `.env`. Never commit `.streamlit/secrets.toml`.

The optional [Render deployment guide](docs/DEPLOY.md) and [Render Blueprint](render.yaml) remain as alternate hosting options.

## Three-phase delivery

| Phase | Deliverable | State |
| --- | --- | --- |
| 1 — MVP | Loaders, chunking, FAISS, retrieval, grounded generation, local Streamlit | Implemented in this update |
| 2 — Quality | Citation display, modular tests, retrieval metrics, CI | Implemented baseline; expand gold dataset and human answer reviews next |
| 3 — Packaging | Docker, Streamlit Cloud settings, README, demonstration | Source prepared; Streamlit deployment and live-generation tests remain |

Read [docs/ROADMAP.md](docs/ROADMAP.md) for tradeoffs, safeguards and next steps.

## Project overview and demonstration

**Suggested title:** CampusGuide — Document-Grounded RAG Academic Assistant

**Technical summary:**

> Built a document-grounded RAG chatbot using Python, Streamlit, Sentence Transformers, FAISS and Groq Llama, supporting five document families, semantic retrieval, source citations, and automated ingestion/retrieval tests.

To quantify **measured** impact, run the evaluation, report corpus size and chunk count, and expand the question set. For example: “Evaluated retrieval over **N** labeled queries, achieving **X%** source hit@4 at a **Y** similarity threshold” — replace placeholders only with logged measurements. Demonstrate answerable questions, no-evidence refusals and source traceability side by side; disclose error cases and limits.

## Data and license

The bundled raw document files may reflect third-party institutional material. Verify permission and license before redistributing or deploying. The `campus_faq.md` fixture is fictional. Existing project license: [GNU AGPLv3](LICENSE).

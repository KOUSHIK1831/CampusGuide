# CampusGuide — Chat with Your Documents

**An open-source, source-cited RAG chatbot built with Streamlit, Sentence Transformers, FAISS, and Groq.**

Upload your own PDFs, DOCX, TXT, Markdown, or JSON files, provide **your own Groq API key** (BYOK), and ask questions based on the uploaded content. Answers show source filenames and PDF page numbers when available. No shared API key or preloaded university data is required for the interactive app.

## How it works

1. **Enter your Groq API key** in the password-style sidebar field. [Get a key from Groq](https://console.groq.com/keys).
2. **Upload one or more documents** (up to ten files; 15 MB each and 40 MB combined).
3. Click **Process uploaded documents** to extract text, chunk it, and build an in-memory FAISS index.
4. **Ask questions in chat** about those documents. Answers cite numbered passages; expand source references to inspect excerpts.
5. Click **Clear my session** to clear the uploaded files, API key, index, and chat. Changing the uploaded files also clears the previous index and conversation.

### Privacy and limitations

- **Bring your own key:** each visitor supplies a Groq key in their own Streamlit session; the interactive app does not use a developer-provided secret or commit keys to GitHub. The key is passed directly to Groq's client for that visitor's questions and is not cached or written to disk by this application.
- **Upload isolation:** document bytes and vector indexes are processed in memory and held in a **per-session** store; the app does not write uploaded content or its index to a shared filesystem. Only the pretrained embedding model is cached across sessions.
- **Third-party processing:** Streamlit's hosting infrastructure receives uploads, and relevant retrieved document excerpts are sent to Groq for answer generation. **Do not upload highly sensitive or confidential files.** Session storage is not equivalent to end-to-end encryption or a regulated data-processing service.
- **RAG is fallible:** responses may be incomplete or wrong; citation checks verify citation numbers, not entailment. Scanned/image-only PDFs need OCR, which is not implemented. Files and chat disappear on restart/clear. The public demo has no user authentication or global anti-abuse controls.
- **BYOK protects the operator's API allowance, not the user from Groq usage charges or limits.** Users should protect their keys and rotate keys they accidentally expose.

## Tech stack

| Area | Technology |
| --- | --- |
| UI | Python 3.12+, Streamlit |
| Extraction | PyMuPDF, python-docx, built-in TXT/Markdown/JSON parsers |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Search | FAISS `IndexFlatIP` with normalized vectors; optional NumPy backend |
| Generation | Groq Llama model; user-supplied Groq credentials |
| Tests & CI | pytest, GitHub Actions |

## Run locally

```bash
git clone https://github.com/KOUSHIK1831/CampusGuide.git
cd CampusGuide
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Enter a Groq key **inside the application**; no `.env` file is necessary for the interactive experience. The initial model download requires network access.

The original CLI scripts remain available for local corpus experiments:

```bash
python scripts/ingest.py
python scripts/evaluate.py --cases evaluation/demo_cases.json --k 4
```

These CLI scripts use the optional local `data/raw` corpus and `data/processed` path. They are **separate from the interactive BYOK upload flow**. The repository includes older experimental modules in `ingestion/`, `embeddings/`, `retriever/`, `llm/`, `validation/`, and `ui/`; the maintained web UI is `app.py` with the `campusguide_rag/` package.

## Deploy to Streamlit Community Cloud

See **[Streamlit Cloud deployment instructions](docs/STREAMLIT_CLOUD.md)**. Deploy `KOUSHIK1831/CampusGuide` from `main` with entrypoint `app.py` and Python 3.12. **No Groq secret needs to be configured in the hosting account** because each user enters their own key inside the application. The repository being pushed to GitHub does **not** automatically create a Streamlit app or assign a `streamlit.app` URL.

The alternative [Render deployment guide](docs/DEPLOY.md) and [Dockerfile](Dockerfile) remain available for self-hosting. If moving away from the old shared-key Render demo, revoke its previously shared Groq key and remove that environment variable from Render.

## Run tests

```bash
pip install 'pytest>=8,<10'
python -m pytest -q
```

The suite includes in-memory document parsing, upload validation, document isolation, citation checks, index retrieval, and persistence for the **optional CLI** mode. Tests use fake embeddings and model responses when possible; they do not prove real-world answer accuracy.

## Project structure

```text
app.py                            Streamlit BYOK upload-and-chat interface
campusguide_rag/                 Current extraction, retrieval, and RAG pipeline
  ingestion.py                   PDF/DOCX/TXT/MD/JSON loading, including bytes
  upload_workflow.py             Upload validation, chunking, change detection
  vector_store.py                FAISS/NumPy search and optional disk persistence
  chatbot.py                     Grounded LLM prompt and citation checks
tests/                           Offline tests and optional FAISS integration
scripts/                         CLI ingestion and evaluation
data/examples/                   Fictional sample; not loaded by default in UI
docs/STREAMLIT_CLOUD.md          Deployment instructions
requirements.txt                 Python dependencies
packages.txt                     System dependencies for Streamlit Cloud
```

## Example question

After uploading a document that discusses a library policy, ask: **"How many books can I borrow?"** CampusGuide will retrieve relevant passages, request a cited answer from Groq, and display the supporting source. If the documents lack an answer, the chatbot is designed to refuse rather than make one up. Actual behavior depends on the supplied documents and model.

## License and contributions

See [LICENSE](LICENSE) for the existing AGPLv3 license. Contributions and bug reports are welcome through GitHub issues and pull requests. Verify permission before redistributing any third-party documents in the repository's legacy `data/raw` directory.

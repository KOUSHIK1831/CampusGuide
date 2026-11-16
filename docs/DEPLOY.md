# Alternative: Deploy CampusGuide on Render

The primary hosting guide is [Streamlit Community Cloud](STREAMLIT_CLOUD.md). Render is an optional Docker-based alternative, using the included [render.yaml](../render.yaml).

## Bring your own Groq API key (BYOK)

CampusGuide no longer uses a shared Groq API key for the interactive UI. Each visitor enters their own key in a password-style sidebar field, uploads documents, processes them into a **per-session in-memory index**, and asks questions grounded in those uploads. The app does not automatically index the bundled sample documents or save uploaded documents to disk.

If you previously configured `GROQ_API_KEY` in an existing Render service, **remove that environment variable and revoke the previously exposed key**. The interactive app does not need it. Do not add a key to the Blueprint, repository, or Render shared environment.

## Deploy

1. Connect your GitHub account to [Render](https://dashboard.render.com/).
2. Create a Blueprint from `KOUSHIK1831/CampusGuide`, branch `main`, reading the root `render.yaml`.
3. Confirm it uses the `free` plan, Docker runtime, and the `/_stcore/health` route. Do not select a paid plan without evaluating the cost.
4. Watch the Docker build, runtime logs, and health status. Streamlit binds to the `PORT` provided by Render.
5. Open the **actual Render-assigned URL**. Supply your own Groq key in the sidebar, upload a text-based document, click **Process uploaded documents**, ask a grounded question, and inspect its source citations.

Limitations: Render's free plan may lack enough RAM for PyTorch and Sentence Transformers; filesystem state is ephemeral; uploads are limited to ten files (15 MB per file / 40 MB combined). The app has no global anti-abuse controls or authentication. Avoid uploading confidential data; document excerpts are sent to Groq during answer generation.

For local Docker testing, use:

```bash
docker build -t campusguide-rag .
docker run --rm -p 8501:8501 campusguide-rag
```

Then visit http://localhost:8501 and enter your own key in the UI. No `.env` file is needed for the web app.

## Model compatibility

The default Groq model is `openai/gpt-oss-20b`. An existing Render service with `LLM_MODEL=llama-3.1-8b-instant` will be mapped to the supported model in application code; update the Render environment setting as well to avoid confusion.

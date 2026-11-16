# Deploy CampusGuide on Render (Streamlit + Docker)

This repository includes a `render.yaml` blueprint. It is **deployment configuration only**, not proof of a live app. No credentials or hosting account are bundled.

## Preflight

1. Push your reviewed update to `KOUSHIK1831/CampusGuide` (feature branch + PR, then merge into `main`). Check GitHub Actions, especially the FAISS integration test.
2. Keep `.env` and any actual `GROQ_API_KEY` **out of Git**. Create a Groq key for your own account and add it to your Render service as a secret environment variable; never put it in `render.yaml`.
3. Verify your rights to publish each document. The Render blueprint indexes **only `data/examples/campus_faq.md`**, a fictional example, and disables file uploads by default. The repository may contain other institutional material, which is **not indexed** in this hosted configuration.
4. Be aware of cost and quotas: `plan: free` is configured to avoid specifying a paid plan. Free memory (~512 MB) may be **insufficient** for PyTorch/Sentence Transformers. If the service runs out of memory, choose a suitable provider/plan explicitly after checking costs, or deploy on a platform with enough free memory. This guide does not authorize a paid upgrade.
5. There is no strong authentication or server-side global rate limiting. The optional `MAX_DEMO_QUERIES=10` is a per-session courtesy limit, **not** abuse prevention. Use a limited/quota-controlled API key; do not expose a paid key without stronger protection.

## Deploy through the Render dashboard

1. Log in to [Render](https://dashboard.render.com/). Connect your GitHub account and grant it access to the CampusGuide repository.
2. Select **New > Blueprint** and choose `KOUSHIK1831/CampusGuide`. Render reads the root `render.yaml`.
3. Confirm the Docker service configuration and `plan: free`, and set `GROQ_API_KEY` in the secret-input field. Do not paste the key into GitHub or chat.
4. Deploy. Watch build and service logs. The image installs heavy machine-learning dependencies; the initial `all-MiniLM-L6-v2` download requires outbound internet access and sufficient storage/RAM.
5. Visit the Render-provided `onrender.com` URL and confirm the Streamlit chat, a positive answer with a citation, and a refusal for an unrelated question. Your actual URL is assigned by Render: **do not guess it**.

The Blueprint uses `/_stcore/health` as the health-check route, binds Streamlit to `${PORT:-8501}`, sets `AUTO_INGEST_ON_START=true`, points `DATA_RAW_PATH` at the fictional `data/examples` corpus, and disables public uploads. Files under `data/processed` are ephemeral on a free service and will be recreated on restart. If the model download or index rebuild fails, check the logs and instance memory. Never claim a successful deployment without checking the URL and logs.

## Local Docker smoke test

```bash
docker build -t campusguide-rag .
docker run --rm --env-file .env -p 8501:8501 \
  -e DATA_RAW_PATH=data/examples -e AUTO_INGEST_ON_START=true \
  -e ALLOW_UPLOADS=false -e MAX_DEMO_QUERIES=10 campusguide-rag
```

Then open `http://localhost:8501` and use the example prompt from the README.

## Deployment limitations

- A free service can restart or sleep; the FAISS index is rebuilt from the bundled fictional corpus.
- Session query limits can be bypassed by starting another session. Production requires authentication, true per-client quotas, logging and monitoring, and secret rotation.
- Avoid presenting this demo as production-secure or as a tool authorized to interpret official university policies.
- If you later enable uploading, add access control, data retention policies, quotas, malware scanning, and storage isolation first.

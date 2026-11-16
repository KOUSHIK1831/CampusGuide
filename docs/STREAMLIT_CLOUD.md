# Deploy CampusGuide on Streamlit Community Cloud

Streamlit Community Cloud is the **primary public demo hosting target** for CampusGuide. Deploy the GitHub `main` branch; Docker and Render remain optional hosting paths.

## Deploy from GitHub

1. Visit https://share.streamlit.io/ and sign in with GitHub as `KOUSHIK1831`. Authorize Streamlit to access the public `CampusGuide` repository.
2. Select **Create app**, then **Yup, I have an app**.
3. Configure the app:

   | Setting | Value |
   | --- | --- |
   | Repository | `KOUSHIK1831/CampusGuide` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | Python version (Advanced settings) | `3.12` |
   | App URL | Choose an available subdomain or accept the generated one |

4. In **Advanced settings → Secrets**, paste [`.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example) and replace the placeholder `GROQ_API_KEY` with a **newly rotated** Groq API key. Never commit the real `.streamlit/secrets.toml`, post the key to a public location, or include it in the README.
5. Select **Deploy**. Streamlit Community Cloud will provision your app and show its unique, real `*.streamlit.app` URL.
6. Verify the sidebar shows indexed documents, ask **“How many library books can I borrow?”**, inspect its citation, and try an unrelated question to confirm the refusal path. Verify live Groq generation as well as UI startup.

## Public demo defaults

- With no `.env` or secret overrides, `app.py` selects `data/examples` (fictional corpus), auto-indexes on first launch, disables uploads, and limits each session to ten questions.
- Set `GROQ_API_KEY` as a **root-level** secret; it is required to generate Groq answers. Local development can use `.env` instead.
- The vector index is rebuilt from the included sample document after the instance restarts; Community Cloud filesystem state is ephemeral.
- Keep private and unlicensed third-party files out of the public knowledge base. The bundled demo corpus is **fictional**.
- The app has no authentication. A session query count is **not** a global rate limit. Retrieval and citations reduce but cannot eliminate hallucination risk.
- PyTorch, Sentence Transformers and FAISS can exceed Community Cloud memory limits. Review build/runtime logs before claiming a functional deployment.
- `packages.txt` supplies `libgomp1` for native dependencies. The outdated `uv.lock` is removed so Community Cloud installs from the root `requirements.txt`.

## Troubleshooting

- **Missing Groq key**: Set the root-level `GROQ_API_KEY` under Cloud app settings → Secrets, then reboot.
- **No indexed documents**: Confirm `DATA_RAW_PATH=data/examples` and `AUTO_INGEST_ON_START=true`.
- **Dependency build errors**: Select Python 3.12 and check `requirements.txt` plus `packages.txt`.
- **Exceeds resource limits**: Reduce the model/corpus or consider another host; inspect runtime logs.
- **Changes do not appear**: Confirm the Cloud app tracks the `main` branch and reboot it.

Official guidance: [Deploy on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [Manage secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

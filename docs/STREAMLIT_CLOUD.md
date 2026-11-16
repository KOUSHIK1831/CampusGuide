# Deploy CampusGuide to Streamlit Community Cloud

CampusGuide is a **bring-your-own-Groq-key, upload-and-chat RAG app**. The public UI has no shared Groq key. Each visitor enters their own key using a password-style field and uploads their own documents to a session-local index.

## 1. Create the Streamlit app

1. Open **https://share.streamlit.io/** and sign in using your GitHub account.
2. Choose **Create app** and select **Yup, I have an app** (or **Deploy a public app**).
3. Configure:

   | Setting | Value |
   | --- | --- |
   | Repository | `KOUSHIK1831/CampusGuide` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | Python version | `3.12` |

4. Click **Deploy**. Streamlit will display the actual app's unique `*.streamlit.app` URL when provisioned.
5. No Streamlit Cloud **Secrets** are required for Groq. **Do not paste your Groq key into the hosting app's shared Secrets** for the BYOK workflow.

## 2. Test the live app

1. Open the deployed URL and enter **your own Groq API key** in the sidebar. Obtain a fresh key at https://console.groq.com/keys. Do not reuse a publicly exposed key.
2. Upload a **PDF, DOCX, TXT, Markdown, or JSON** file. Scanned PDFs without selectable text are unsupported.
3. Click **Process uploaded documents**. Verify that the sidebar reports indexed documents and chunks.
4. Ask something directly answerable from your uploaded file, such as **"Summarize the key points in this document"**. Inspect source citation expanders and PDF page numbers when applicable.
5. Ask a question unrelated to the file and check the fallback answer.
6. Click **Clear my session**, and verify that the chat, API key, and files reset.
7. Try a second, separate browser session. Documents and question histories should not be shared between sessions.

## Operational behavior

- Uploads are parsed **in memory** (not written by the application to disk) and indexes are held in `st.session_state` per user. The only shared cached resource is the pretrained embedding model. Do not change this to a shared `@st.cache_resource` vector store.
- Each session can upload up to **10 files**, at most **15 MB per file** and **40 MB combined**. Uploaded files are re-indexed only after pressing the process button. Changing selected files invalidates the old index and conversation.
- Your Groq key is used to construct a client **per request** and is never stored in environment variables, project files, or a shared cache by this application. Groq receives selected document excerpts for each question.
- Session data is not permanent. Reconnecting, clearing state, and app restarts can remove documents and conversation. Avoid sensitive documents; the host and model provider still process submitted data.
- This is an unauthenticated public application with no durable user storage, system-wide rate limiting, or OCR. Community Cloud memory limits may affect embedding-model startup and large uploads.
- If an old shared Groq API key was placed on Render, revoke it and remove it from Render's service configuration when no longer needed; it is not needed here.

## Troubleshooting

- **No upload widget**: Verify the app follows the latest `main` commit and reboot from the Cloud dashboard.
- **Question entry disabled**: Enter your Groq key, upload a file, and click **Process uploaded documents** first.
- **Missing source text**: Ensure the PDF is text-based; scanned PDFs require external OCR.
- **Import or binary dependency errors**: Confirm Python 3.12 and the root `requirements.txt` / `packages.txt`. `libgomp1` is specified for native dependencies.
- **Memory limit**: Reduce file sizes or use a hosting plan suitable for local embedding models.
- **Groq request failed**: Check the key's validity, quota, and provider availability. Do not print or share the full key in issues.

More information: [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Model compatibility and request failures

CampusGuide defaults to `openai/gpt-oss-20b`. Groq retired `llama-3.1-8b-instant` for Free and Developer accounts on August 16, 2026. The chatbot automatically maps the old model setting to the new default.

The UI differentiates rejected API keys (401), account restrictions (403), rate limits (429), unavailable models or invalid requests (400/404/422), and server or network failures without exposing the original provider response or your credentials.

Check [Groq Keys](https://console.groq.com/keys) and [Active Models](https://console.groq.com/docs/models). Rotate any key you have posted in a chat, issue, or screenshot.

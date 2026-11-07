# SmartDocs RAG

SmartDocs indexes PDF, TXT, and Markdown files, retrieves relevant passages, and
answers questions using those passages with source references. It uses FastAPI
for the backend, ChromaDB for persistent vectors, Groq for answer generation,
and Streamlit for the user interface.

## Requirements

- Python 3.12 or newer with compatible package wheels. Python 3.13 is supported
  by current dependencies.
- A Groq API key for chat answers.
- A Clerk development application configured as an OpenID Connect provider.

## Install on Windows PowerShell

Run these commands from the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Open `.env` and set `GROQ_API_KEY` to the key from your Groq console. Generate a
long random `SMARTDOCS_INTERNAL_AUTH_TOKEN` with
`python -c "import secrets; print(secrets.token_urlsafe(32))"` and add it to
`.env`. Keep both values private; `.env` is ignored by Git. The default model
is `openai/gpt-oss-20b`.

Configure Clerk's OAuth/OIDC application for the local callback
`http://localhost:8501/oauth2callback`. Create the Streamlit secrets file from
the template and fill in the OAuth client ID, client secret, and Clerk
development instance's OpenID Connect metadata URL:

```powershell
New-Item -ItemType Directory -Force .streamlit | Out-Null
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
```

The metadata URL has this form:
`https://<your-clerk-instance>.clerk.accounts.dev/.well-known/oauth-authorization-server`.
The client secret belongs only in `.streamlit/secrets.toml`; that file is
ignored by Git. Streamlit completes Clerk sign-in and passes the verified Clerk
user ID to FastAPI over the private server-to-server connection. FastAPI scopes
conversations, document metadata, and vector searches to that user.
The provider configuration sets `client_kwargs = { "prompt" = "login" }` to
override Streamlit's default `select_account` prompt, which Clerk rejects.

If the Python launcher does not have Python 3.13 registered, create the
environment with the Python executable you installed, for example:

```powershell
python -m venv .venv
```

## Run the application

Start the API in the first terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api.main:app --reload
```

The API listens at `http://127.0.0.1:8000`. Its interactive endpoint
documentation is at `http://127.0.0.1:8000/docs`; the health route is
`http://127.0.0.1:8000/health`.

Start Streamlit in a second terminal from the same project directory after
creating `.streamlit/secrets.toml`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/ui/streamlit_app.py
```

Streamlit opens in your browser and connects to the API at
`http://127.0.0.1:8000`. To use a different API address, set
`SMARTDOCS_API_URL` before starting Streamlit:

```powershell
$env:SMARTDOCS_API_URL = "http://127.0.0.1:8000"
python -m streamlit run app/ui/streamlit_app.py
```

## Run with Docker Compose

Docker Desktop with Docker Compose v2 is required. Create `.env` and Clerk
secrets from their examples, then fill in the Groq key, internal random token,
and Clerk OAuth values:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
New-Item -ItemType Directory -Force .streamlit | Out-Null
if (-not (Test-Path .streamlit/secrets.toml)) {
  Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
}
```

Build the application image and start the API and Streamlit services:

```powershell
docker compose up --build -d
docker compose ps
```

Open Streamlit at `http://localhost:8501`. The API is available at
`http://localhost:8000`, with interactive API docs at
`http://localhost:8000/docs`. Compose waits for the API health check before it
starts the frontend. View logs with `docker compose logs -f` and stop the
services with `docker compose down`.

ChromaDB persists in the `chroma-data` named volume, and downloaded embedding
model files persist in `huggingface-cache`. `docker compose down` keeps both
volumes, so indexed documents remain available after a restart. Avoid
`docker compose down -v` unless you intend to permanently delete the index and
model cache. Compose first runs a small root-owned initializer that fixes the
ownership of both volumes for the non-root application user. A marker prevents
repeating the recursive permission change on later starts, so existing Chroma
data is retained.

Run the unit and API tests in the API container after starting Compose:

```powershell
docker compose exec api python -m unittest discover -s tests -v
```

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/documents/upload` | Upload and index a PDF, TXT, or Markdown file. |
| `GET` | `/api/conversations` | List conversations. |
| `POST` | `/api/conversations` | Create a conversation with `{"title": "..."}`. |
| `GET` | `/api/conversations/{id}/messages` | Load saved messages and source references. |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation, its messages, and indexed chunks. |
| `GET` | `/api/documents?conversation_id=...` | List files in one conversation. |
| `DELETE` | `/api/documents/{document_id}?conversation_id=...` | Delete one file from a conversation. |
| `POST` | `/api/chat` | Ask a question; accepts `{"question": "...", "top_k": 4, "conversation_id": "..."}`. |
| `GET` | `/health` | Confirm the API process is responding. |

The default conversation keeps documents indexed before conversation support.
On the first authenticated request, pre-authentication conversations and their
chunks are assigned to that first Clerk user. New conversations belong to the
user who creates them and isolate their files and retrieval results. SQLite persists
conversation titles and message history under `data/chroma/`, alongside the
ChromaDB data; the Docker `chroma-data` volume preserves both.

The chat response includes `answer` and `sources`. Each source includes its
source ID, filename, page number when available, chunk index, and similarity
returned by retrieval. TXT and Markdown files do not have page numbers.

## Evaluation

Run the 30-question retrieval benchmark and local reliability checks using the
instructions in [evaluation/README.md](evaluation/README.md). The benchmark
reports Hit Rate@K, MRR@K, and retrieval latency; it does not score generated
answer faithfulness or answer relevance yet.

## Architecture

See [the architecture diagram](docs/architecture.md) for the upload, retrieval,
LLM, persistence, and response flow. Screenshots are not included yet; the
headless browser available during setup returned blank captures, so they need
to be captured from a normal browser session to accurately represent the app.

## Local data

ChromaDB persists its index under `data/chroma/`. The uploaded source file is
held in a temporary directory while it is processed; SmartDocs stores the
extracted chunks and metadata in ChromaDB. Keep original documents separately
if you need to upload them again.

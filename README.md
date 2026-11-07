# SmartDocs RAG

SmartDocs indexes PDF, TXT, and Markdown files, retrieves relevant passages, and
answers questions using those passages with source references. It uses FastAPI
for the backend, ChromaDB for persistent vectors, Groq for answer generation,
and Streamlit for the user interface.

## Requirements

- Python 3.12 or newer with compatible package wheels. Python 3.13 is supported
  by current dependencies.
- A Groq API key for chat answers.

## Install on Windows PowerShell

Run these commands from the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set `GROQ_API_KEY` to the key from your Groq console. Keep that
key private; `.env` is ignored by Git. The default model is
`openai/gpt-oss-20b`.

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

Start Streamlit in a second terminal from the same project directory:

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

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/documents/upload` | Upload and index a PDF, TXT, or Markdown file. |
| `GET` | `/api/documents` | List indexed documents and their chunk counts. |
| `DELETE` | `/api/documents/{document_id}` | Delete one indexed document and its chunks. |
| `POST` | `/api/chat` | Ask a question; accepts `{"question": "...", "top_k": 4}`. |
| `GET` | `/health` | Confirm the API process is responding. |

The chat response includes `answer` and `sources`. Each source includes its
source ID, filename, page number when available, chunk index, and similarity
returned by retrieval. TXT and Markdown files do not have page numbers.

## Evaluation

Run the 30-question retrieval benchmark and local reliability checks using the
instructions in [evaluation/README.md](evaluation/README.md). The benchmark
reports Hit Rate@K, MRR@K, and retrieval latency; it does not score generated
answer faithfulness or answer relevance yet.

## Local data

ChromaDB persists its index under `data/chroma/`. The uploaded source file is
held in a temporary directory while it is processed; SmartDocs stores the
extracted chunks and metadata in ChromaDB. Keep original documents separately
if you need to upload them again.

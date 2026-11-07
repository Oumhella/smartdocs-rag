# SmartDocs architecture

```mermaid
flowchart LR
    User[User] -->|Upload and questions| UI[Streamlit frontend]
    UI -->|HTTP| API[FastAPI API]
    API --> DOC[Document service]
    DOC --> ING[PDF / TXT / Markdown extraction and chunking]
    ING --> EMB[all-MiniLM-L6-v2 embeddings]
    EMB --> CHROMA[(ChromaDB persistent volume)]
    API --> RAG[RAG service]
    RAG -->|Question embedding and top-k search| CHROMA
    RAG -->|Context plus question| GROQ[Groq API: gpt-oss-20b]
    GROQ -->|Answer and source IDs| RAG
    RAG -->|Validated answer and references| API
    API --> UI
```

Docker Compose builds the same Python image for both application services. The
API owns the ChromaDB and Hugging Face cache volumes. Streamlit talks to the
API through Compose's internal `api` service name; users open Streamlit on port
8501.

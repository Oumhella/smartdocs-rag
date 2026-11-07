# SmartDocs architecture

```mermaid
flowchart LR
    User[User] -->|Create or select a conversation| UI[Streamlit frontend]
    UI -->|Upload and questions with conversation ID| API[FastAPI API]
    API --> SESSIONS[(SQLite conversations and message history)]
    API --> DOC[Document service]
    DOC --> ING[PDF / TXT / Markdown extraction and chunking]
    ING --> EMB[all-MiniLM-L6-v2 embeddings]
    EMB --> CHROMA[(ChromaDB persistent volume, tagged by conversation)]
    API --> RAG[RAG service]
    RAG -->|Conversation-filtered search| CHROMA
    RAG -->|Context plus question| GROQ[Groq API: gpt-oss-20b]
    GROQ -->|Answer and source IDs| RAG
    RAG -->|Validated answer and references| API
    API --> UI
```

Docker Compose builds the same Python image for both application services. The
API owns the ChromaDB and Hugging Face cache volumes. Conversation records and
message history live in SQLite under the ChromaDB data directory. Streamlit
talks to the API through Compose's internal `api` service name; users open
Streamlit on port 8501. All document listing, deletion, and retrieval requests
are scoped to the selected conversation ID.

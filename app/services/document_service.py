"""Orchestrate document ingestion and vector-store operations."""

from pathlib import Path
from uuid import uuid4

from app.ingestion.pdf import process_document
from app.vectorstore.chroma_store import ChromaVectorStore


class DocumentService:
    """Index supported files and expose document-level search and deletion."""

    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        self.vector_store = vector_store or ChromaVectorStore()

    def index_file(self, file_path: str | Path) -> dict[str, str | int]:
        """Extract, chunk, embed, and persist a PDF, TXT, or Markdown file."""
        path = Path(file_path)
        chunks = process_document(path)
        if not chunks:
            raise ValueError(f"No extractable text found in {path.name}")

        document_id = uuid4().hex
        chunk_count = self.vector_store.add_document(document_id, chunks)
        return {
            "document_id": document_id,
            "filename": path.name,
            "chunks_stored": chunk_count,
        }

    def delete_document(self, document_id: str) -> int:
        """Delete a document's stored chunks by its returned document ID."""
        return self.vector_store.delete_document(document_id)

    def list_documents(self) -> list[dict[str, str | int]]:
        """List indexed documents with their stored chunk counts."""
        return self.vector_store.list_documents()

    def search(self, query: str, top_k: int = 4) -> list[dict[str, object]]:
        """Search indexed document chunks without generating an LLM answer."""
        return self.vector_store.search(query=query, top_k=top_k)

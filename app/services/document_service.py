"""Orchestrate document ingestion and vector-store operations."""

from pathlib import Path
import hashlib
from uuid import uuid4

import pymupdf

from app.ingestion.pdf import process_document
from app.services.conversation_service import DEFAULT_CONVERSATION_ID
from app.vectorstore.chroma_store import ChromaVectorStore


class DocumentService:
    """Index supported files and expose document-level search and deletion."""

    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        self.vector_store = vector_store or ChromaVectorStore()

    def index_file(
        self,
        file_path: str | Path,
        conversation_id: str = DEFAULT_CONVERSATION_ID,
    ) -> dict[str, str | int | bool]:
        """Extract, chunk, embed, and persist a PDF, TXT, or Markdown file."""
        path = Path(file_path)
        invalid_document = False
        try:
            chunks = process_document(path)
        except (pymupdf.FileDataError, pymupdf.EmptyFileError):
            # Leave the parser's exception scope before raising our clean error.
            # This also releases its Windows file handle before upload cleanup.
            invalid_document = True
            chunks = []
        if invalid_document:
            raise ValueError(f"Could not read {path.name} as a valid document")
        if not chunks:
            raise ValueError(f"No extractable text found in {path.name}")

        digest = _file_sha256(path)
        existing = self.vector_store.find_document_by_hash(
            path.name, digest, conversation_id
        )
        if existing is not None:
            return {**existing, "duplicate": True}

        document_id = uuid4().hex
        chunk_count = self.vector_store.add_document(
            document_id,
            chunks,
            conversation_id=conversation_id,
            content_sha256=digest,
        )
        return {
            "document_id": document_id,
            "filename": path.name,
            "chunks_stored": chunk_count,
            "duplicate": False,
        }

    def delete_document(
        self, document_id: str, conversation_id: str = DEFAULT_CONVERSATION_ID
    ) -> int:
        """Delete a document's stored chunks by its returned document ID."""
        return self.vector_store.delete_document(document_id, conversation_id)

    def list_documents(
        self, conversation_id: str = DEFAULT_CONVERSATION_ID
    ) -> list[dict[str, str | int]]:
        """List indexed documents with their stored chunk counts."""
        return self.vector_store.list_documents(conversation_id)

    def search(
        self,
        query: str,
        top_k: int = 4,
        conversation_id: str = DEFAULT_CONVERSATION_ID,
    ) -> list[dict[str, object]]:
        """Search indexed document chunks without generating an LLM answer."""
        return self.vector_store.search(
            query=query, top_k=top_k, conversation_id=conversation_id
        )

    def migrate_legacy_chunks(self, conversation_id: str) -> int:
        """Put chunks created before conversations into the default workspace."""
        return self.vector_store.assign_legacy_chunks(conversation_id)

    def delete_conversation(self, conversation_id: str) -> int:
        """Remove every indexed chunk for a conversation."""
        return self.vector_store.delete_conversation(conversation_id)


def _file_sha256(path: Path) -> str:
    """Hash a file incrementally so large uploads do not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

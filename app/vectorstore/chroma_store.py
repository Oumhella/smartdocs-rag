"""Persistent ChromaDB storage for document chunks."""

from pathlib import Path
from typing import Any

import chromadb

from app.embeddings.model import TextEmbedder
from app.ingestion.pdf import TextChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "chroma"


class ChromaVectorStore:
    """Store and search chunks in a local, persistent ChromaDB collection."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        *,
        collection_name: str = "smartdocs_chunks",
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or TextEmbedder()

        client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        document_id: str,
        chunks: list[TextChunk],
        *,
        content_sha256: str | None = None,
    ) -> int:
        """Embed and store one document's chunks; return the number stored."""
        if not document_id:
            raise ValueError("document_id must not be empty")
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids: list[str] = []
        metadatas: list[dict[str, str | int]] = []

        for position, chunk in enumerate(chunks, start=1):
            # Position is document-wide, so IDs stay unique even though the
            # chunk index restarts for each PDF page.
            ids.append(f"{document_id}:{position}")
            metadata: dict[str, str | int] = {
                "document_id": document_id,
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
            }
            if content_sha256 is not None:
                metadata["content_sha256"] = content_sha256
            # Chroma metadata does not accept None; text formats have no page.
            if chunk["page_number"] is not None:
                metadata["page_number"] = chunk["page_number"]
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk belonging to a document; return chunks deleted."""
        if not document_id:
            raise ValueError("document_id must not be empty")

        existing = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        ids = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def find_document_by_hash(
        self, filename: str, content_sha256: str
    ) -> dict[str, str | int] | None:
        """Find a previously indexed document with the same name and bytes."""
        matches = self.collection.get(
            where={"content_sha256": content_sha256},
            include=["metadatas"],
        )
        duplicate_metadata = [
            metadata
            for metadata in matches["metadatas"] or []
            if metadata and metadata.get("filename") == filename
        ]
        for metadata in duplicate_metadata:
            return {
                "document_id": str(metadata["document_id"]),
                "filename": filename,
                "chunks_stored": len(duplicate_metadata),
            }
        return None

    def list_documents(self) -> list[dict[str, str | int]]:
        """Summarize stored chunks as one record per indexed document."""
        stored = self.collection.get(include=["metadatas"])
        documents: dict[str, dict[str, str | int]] = {}

        for metadata in stored["metadatas"] or []:
            if not metadata:
                continue
            document_id = str(metadata.get("document_id", ""))
            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": str(metadata.get("filename", "Unknown")),
                    "chunks_stored": 0,
                }
            documents[document_id]["chunks_stored"] += 1

        return sorted(
            documents.values(),
            key=lambda document: str(document["filename"]).casefold(),
        )

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Return nearest chunks, their metadata, cosine distance and similarity."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)
        matches = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict[str, Any]] = []
        documents = matches["documents"][0]
        metadatas = matches["metadatas"][0]
        distances = matches["distances"][0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            cosine_distance = float(distance)
            result_metadata = dict(metadata or {})
            # Return a consistent shape, while only storing page_number for PDFs.
            result_metadata.setdefault("page_number", None)
            results.append(
                {
                    "text": text,
                    "metadata": result_metadata,
                    "distance": cosine_distance,
                    "similarity": 1.0 - cosine_distance,
                }
            )

        return results

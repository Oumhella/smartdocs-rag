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

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Close the persistent client (useful for short-lived tools/tests)."""
        self.client.close()

    def add_document(
        self,
        document_id: str,
        chunks: list[TextChunk],
        *,
        conversation_id: str,
        owner_user_id: str | None = None,
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
                "conversation_id": conversation_id,
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
            }
            if owner_user_id is not None:
                metadata["owner_user_id"] = owner_user_id
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

    def delete_document(
        self,
        document_id: str,
        conversation_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> int:
        """Delete every chunk belonging to a document; return chunks deleted."""
        if not document_id:
            raise ValueError("document_id must not be empty")

        filters: list[dict[str, Any]] = [{"document_id": document_id}]
        if conversation_id is not None:
            filters.append({"conversation_id": conversation_id})
        if owner_user_id is not None:
            filters.append({"owner_user_id": owner_user_id})
        where = {"$and": filters} if len(filters) > 1 else filters[0]
        existing = self.collection.get(
            where=where,
            include=["metadatas"],
        )
        ids = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def find_document_by_hash(
        self,
        filename: str,
        content_sha256: str,
        conversation_id: str,
        owner_user_id: str | None = None,
    ) -> dict[str, str | int] | None:
        """Find a previously indexed document with the same name and bytes."""
        filters: list[dict[str, Any]] = [
            {"content_sha256": content_sha256},
            {"conversation_id": conversation_id},
        ]
        if owner_user_id is not None:
            filters.append({"owner_user_id": owner_user_id})
        matches = self.collection.get(
            where={"$and": filters},
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

    def list_documents(
        self,
        conversation_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[dict[str, str | int]]:
        """Summarize stored chunks as one record per indexed document."""
        filters: list[dict[str, Any]] = []
        if conversation_id:
            filters.append({"conversation_id": conversation_id})
        if owner_user_id:
            filters.append({"owner_user_id": owner_user_id})
        where = {"$and": filters} if len(filters) > 1 else filters[0] if filters else None
        stored = self.collection.get(
            where=where,
            include=["metadatas"],
        )
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

    def search(
        self,
        query: str,
        top_k: int = 4,
        conversation_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks, their metadata, cosine distance and similarity."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)
        filters: list[dict[str, Any]] = []
        if conversation_id:
            filters.append({"conversation_id": conversation_id})
        if owner_user_id:
            filters.append({"owner_user_id": owner_user_id})
        where = {"$and": filters} if len(filters) > 1 else filters[0] if filters else None
        matches = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
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

    def assign_legacy_chunks(self, conversation_id: str) -> int:
        """Assign pre-conversation chunks to the initial/default conversation."""
        legacy = self.collection.get(include=["metadatas"])
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk_id, metadata in zip(legacy["ids"], legacy["metadatas"] or []):
            updated = dict(metadata or {})
            if "conversation_id" not in updated:
                updated["conversation_id"] = conversation_id
                ids.append(chunk_id)
                metadatas.append(updated)
        if ids:
            self.collection.update(ids=ids, metadatas=metadatas)
        return len(ids)

    def delete_conversation(
        self, conversation_id: str, owner_user_id: str | None = None
    ) -> int:
        """Delete all indexed chunks for one conversation."""
        where: dict[str, Any] = {"conversation_id": conversation_id}
        if owner_user_id is not None:
            where = {"$and": [where, {"owner_user_id": owner_user_id}]}
        existing = self.collection.get(
            where=where, include=["metadatas"]
        )
        ids = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def assign_conversation_owner(
        self, conversation_ids: list[str], owner_user_id: str
    ) -> int:
        """Tag previously unowned chunks when legacy workspaces are claimed."""
        if not conversation_ids:
            return 0
        owned_ids: list[str] = []
        owned_metadatas: list[dict[str, Any]] = []
        stored = self.collection.get(include=["metadatas"])
        conversation_set = set(conversation_ids)
        for chunk_id, metadata in zip(stored["ids"], stored["metadatas"] or []):
            updated = dict(metadata or {})
            if (
                updated.get("conversation_id") in conversation_set
                and "owner_user_id" not in updated
            ):
                updated["owner_user_id"] = owner_user_id
                owned_ids.append(chunk_id)
                owned_metadatas.append(updated)
        if owned_ids:
            self.collection.update(ids=owned_ids, metadatas=owned_metadatas)
        return len(owned_ids)

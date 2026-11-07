"""FastAPI routes that expose document management and RAG chat."""

from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.service import RAGService
from app.services.conversation_service import (
    DEFAULT_CONVERSATION_ID,
    ConversationService,
)
from app.services.document_service import DocumentService


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

app = FastAPI(title="SmartDocs API", version="0.1.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)
    conversation_id: str = DEFAULT_CONVERSATION_ID


class ConversationRequest(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=120)


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    """Create the document service on first use, not during health checks."""
    service = DocumentService()
    service.migrate_legacy_chunks(DEFAULT_CONVERSATION_ID)
    return service


@lru_cache(maxsize=1)
def get_conversation_service() -> ConversationService:
    """Create the persistent conversation store on first use."""
    return ConversationService()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Create the RAG service only when a chat request arrives."""
    service = RAGService()
    # Chat can be the first API request after an upgrade; ensure old chunks
    # without conversation metadata remain searchable in the default workspace.
    service.vector_store.assign_legacy_chunks(DEFAULT_CONVERSATION_ID)
    return service


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP process is responding."""
    return {"status": "ok"}


def require_conversation(conversation_id: str) -> dict[str, str]:
    conversation = get_conversation_service().get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.get("/api/conversations")
def list_conversations() -> list[dict[str, str]]:
    return get_conversation_service().list_conversations()


@app.post("/api/conversations", status_code=201)
def create_conversation(request: ConversationRequest) -> dict[str, str]:
    return get_conversation_service().create(request.title)


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
    require_conversation(conversation_id)
    return get_conversation_service().messages(conversation_id)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, str | int]:
    require_conversation(conversation_id)
    if conversation_id == DEFAULT_CONVERSATION_ID:
        raise HTTPException(
            status_code=400, detail="The default conversation cannot be deleted."
        )
    deleted_chunks = get_document_service().delete_conversation(conversation_id)
    get_conversation_service().delete(conversation_id)
    return {"conversation_id": conversation_id, "chunks_deleted": deleted_chunks}


@app.post("/api/documents/upload", status_code=201)
def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(DEFAULT_CONVERSATION_ID),
) -> dict[str, Any]:
    """Save an upload temporarily, index it, then remove the temporary copy."""
    original_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(original_name).suffix.lower()
    if not original_name or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Supported file types are PDF, TXT, and Markdown (.md).",
        )
    require_conversation(conversation_id)

    try:
        with tempfile.TemporaryDirectory(prefix="smartdocs-upload-") as directory:
            temporary_path = Path(directory) / original_name
            with temporary_path.open("wb") as destination:
                shutil.copyfileobj(file.file, destination)
            if temporary_path.stat().st_size == 0:
                raise HTTPException(status_code=422, detail="The uploaded file is empty.")
            return get_document_service().index_file(temporary_path, conversation_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        file.file.close()


@app.get("/api/documents")
def list_documents(
    conversation_id: str = DEFAULT_CONVERSATION_ID,
) -> list[dict[str, str | int]]:
    """List indexed documents, grouped from their stored chunk metadata."""
    require_conversation(conversation_id)
    return get_document_service().list_documents(conversation_id)


@app.delete("/api/documents/{document_id}")
def delete_document(
    document_id: str, conversation_id: str = DEFAULT_CONVERSATION_ID
) -> dict[str, str | int]:
    """Delete every indexed chunk for one document."""
    require_conversation(conversation_id)
    deleted_count = get_document_service().delete_document(
        document_id, conversation_id
    )
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "chunks_deleted": deleted_count}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Answer from retrieved document context and return validated references."""
    try:
        require_conversation(request.conversation_id)
        conversations = get_conversation_service()
        conversations.add_message(
            request.conversation_id, "user", request.question.strip()
        )
        result = get_rag_service().answer(
            request.question,
            top_k=request.top_k,
            conversation_id=request.conversation_id,
        )
        conversations.add_message(
            request.conversation_id,
            "assistant",
            result["answer"],
            result.get("sources", []),
        )
        return {**result, "conversation_id": request.conversation_id}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

"""FastAPI routes that expose document management and RAG chat."""

from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.service import RAGService
from app.services.document_service import DocumentService


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

app = FastAPI(title="SmartDocs API", version="0.1.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    """Create the document service on first use, not during health checks."""
    return DocumentService()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Create the RAG service only when a chat request arrives."""
    return RAGService()


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP process is responding."""
    return {"status": "ok"}


@app.post("/api/documents/upload", status_code=201)
def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Save an upload temporarily, index it, then remove the temporary copy."""
    original_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(original_name).suffix.lower()
    if not original_name or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Supported file types are PDF, TXT, and Markdown (.md).",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="smartdocs-upload-") as directory:
            temporary_path = Path(directory) / original_name
            with temporary_path.open("wb") as destination:
                shutil.copyfileobj(file.file, destination)
            if temporary_path.stat().st_size == 0:
                raise HTTPException(status_code=422, detail="The uploaded file is empty.")
            return get_document_service().index_file(temporary_path)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        file.file.close()


@app.get("/api/documents")
def list_documents() -> list[dict[str, str | int]]:
    """List indexed documents, grouped from their stored chunk metadata."""
    return get_document_service().list_documents()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str | int]:
    """Delete every indexed chunk for one document."""
    deleted_count = get_document_service().delete_document(document_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "chunks_deleted": deleted_count}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Answer from retrieved document context and return validated references."""
    try:
        return get_rag_service().answer(request.question, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

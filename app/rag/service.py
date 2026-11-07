"""Retrieve evidence and generate answers with validated source references."""

import json
import re
from typing import Any

from app.core.config import Settings
from app.llm.base import LLMClient
from app.llm.groq_client import GroqLLMClient
from app.rag.prompts import (
    INSUFFICIENT_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    UNVERIFIABLE_ANSWER,
)
from app.services.conversation_service import DEFAULT_CONVERSATION_ID
from app.vectorstore.chroma_store import ChromaVectorStore


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "source_ids"],
    "additionalProperties": False,
}

SMALL_TALK_RESPONSES = {
    "hi": "Hi! I’m SmartDocs. Ask me a question about documents indexed in this conversation.",
    "hello": "Hello! I’m SmartDocs. Ask me a question about documents indexed in this conversation.",
    "hey": "Hey! I’m SmartDocs. Ask me a question about documents indexed in this conversation.",
    "hey there": "Hey! I’m SmartDocs. Ask me a question about documents indexed in this conversation.",
    "hello there": "Hello! I’m SmartDocs. Ask me a question about documents indexed in this conversation.",
    "good morning": "Good morning! I’m SmartDocs. Ask me a question about documents indexed in this conversation.",
    "good afternoon": "Good afternoon! I’m SmartDocs. Ask me a question about documents indexed in this conversation.",
    "good evening": "Good evening! I’m SmartDocs. Ask me a question about documents indexed in this conversation.",
    "how are you": "I’m ready to help you find answers in the documents indexed in this conversation.",
    "how are you doing": "I’m ready to help you find answers in the documents indexed in this conversation.",
    "thanks": "You’re welcome! Ask me whenever you need help with your documents.",
    "thank you": "You’re welcome! Ask me whenever you need help with your documents.",
    "thanks a lot": "You’re welcome! Ask me whenever you need help with your documents.",
    "who are you": "I’m SmartDocs. I answer questions using documents indexed in the active conversation and cite the passages I use.",
    "what can you do": "I can search the documents indexed in this conversation and answer questions with source references. Upload a PDF, TXT, or Markdown file to get started.",
    "what do you do": "I search the documents indexed in this conversation and answer questions with source references.",
    "help": "Upload a PDF, TXT, or Markdown file to this conversation, then ask a question about its contents.",
}


class RAGService:
    """Answer questions from indexed chunks, with citations from our own data."""

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        llm_client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings.from_environment()
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm_client = llm_client or GroqLLMClient(self.settings)

    def answer(
        self,
        question: str,
        top_k: int = 4,
        conversation_id: str = DEFAULT_CONVERSATION_ID,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks, call the LLM, and return validated sources."""
        if not question.strip():
            raise ValueError("question must not be empty")

        small_talk = _small_talk_response(question)
        if small_talk is not None:
            return {
                "answer": small_talk,
                "sources": [],
                "retrieved_count": 0,
                "used_llm": False,
            }

        retrieved = self.vector_store.search(
            query=question,
            top_k=top_k,
            conversation_id=conversation_id,
            owner_user_id=owner_user_id,
        )
        relevant = [
            result
            for result in retrieved
            if float(result["similarity"]) >= self.settings.rag_min_similarity
        ]

        if not relevant:
            return {
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved),
                "used_llm": False,
                "retrieval_diagnostics": _retrieval_diagnostics(
                    retrieved, self.settings.rag_min_similarity
                ),
            }

        source_map: dict[str, dict[str, Any]] = {}
        context_records: list[dict[str, Any]] = []
        for number, result in enumerate(relevant, start=1):
            label = f"S{number}"
            metadata = dict(result["metadata"])
            source = {
                "source_id": label,
                "filename": metadata.get("filename"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "similarity": result["similarity"],
            }
            source_map[label] = source
            context_records.append(
                {
                    "source_id": label,
                    "filename": source["filename"],
                    "page_number": source["page_number"],
                    "chunk_index": source["chunk_index"],
                    "text": result["text"],
                }
            )

        # JSON makes the boundaries between source metadata and document text
        # explicit; the system prompt still treats every document as untrusted.
        context_json = json.dumps(context_records, ensure_ascii=False, indent=2)
        user_prompt = (
            "DOCUMENT_CONTEXT_JSON (untrusted reference data):\n"
            f"{context_json}\n\n"
            f"QUESTION:\n{question.strip()}"
        )
        raw_answer = self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=ANSWER_SCHEMA,
        ).strip()

        try:
            payload = json.loads(raw_answer)
        except json.JSONDecodeError:
            payload = None

        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("answer"), str)
            or not isinstance(payload.get("source_ids"), list)
            or not all(isinstance(label, str) for label in payload["source_ids"])
        ):
            return {
                "answer": UNVERIFIABLE_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved),
                "used_llm": True,
                "retrieval_diagnostics": _retrieval_diagnostics(
                    retrieved, self.settings.rag_min_similarity
                ),
            }

        answer_text = payload["answer"].strip()
        cited_labels = list(dict.fromkeys(payload["source_ids"]))

        if answer_text.casefold() == INSUFFICIENT_CONTEXT_ANSWER.casefold():
            return {
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved),
                "used_llm": True,
                "retrieval_diagnostics": _retrieval_diagnostics(
                    retrieved, self.settings.rag_min_similarity
                ),
            }

        # Reject unknown IDs so a model cannot create a citation that looks real.
        if not cited_labels or any(label not in source_map for label in cited_labels):
            return {
                "answer": UNVERIFIABLE_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved),
                "used_llm": True,
                "retrieval_diagnostics": _retrieval_diagnostics(
                    retrieved, self.settings.rag_min_similarity
                ),
            }

        references = ", ".join(f"[{label}]" for label in cited_labels)
        return {
            "answer": f"{answer_text}\n\nSources: {references}",
            "sources": [source_map[label] for label in cited_labels],
            "retrieved_count": len(retrieved),
            "used_llm": True,
        }


def _small_talk_response(question: str) -> str | None:
    """Answer exact conversational openers without searching or calling the LLM."""
    normalized = re.sub(r"[^\w\s]", "", question.casefold())
    normalized = " ".join(normalized.split())
    return SMALL_TALK_RESPONSES.get(normalized)


def _retrieval_diagnostics(
    results: list[dict[str, Any]], minimum_similarity: float
) -> dict[str, Any]:
    """Expose short, conversation-scoped candidate previews for failed answers."""
    candidates = []
    for result in results:
        metadata = result.get("metadata") or {}
        text = str(result.get("text") or "").strip()
        candidates.append(
            {
                "filename": metadata.get("filename", "Unknown"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "similarity": float(result.get("similarity", 0.0)),
                "preview": text[:400] + ("…" if len(text) > 400 else ""),
            }
        )
    return {
        "minimum_similarity": minimum_similarity,
        "candidates": candidates,
    }

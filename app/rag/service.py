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
from app.vectorstore.chroma_store import ChromaVectorStore


_SOURCE_LABEL = re.compile(r"\[(S\d+)\]")


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

    def answer(self, question: str, top_k: int = 4) -> dict[str, Any]:
        """Retrieve relevant chunks, call the LLM, and return validated sources."""
        if not question.strip():
            raise ValueError("question must not be empty")

        retrieved = self.vector_store.search(query=question, top_k=top_k)
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
        ).strip()

        if raw_answer.casefold() == INSUFFICIENT_CONTEXT_ANSWER.casefold():
            return {
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved),
                "used_llm": True,
            }

        cited_labels = list(
            dict.fromkeys(
                match
                for match in _SOURCE_LABEL.findall(raw_answer)
                if match in source_map
            )
        )
        # Do not return fabricated citations as if they were verified sources.
        answer_text = _SOURCE_LABEL.sub(
            lambda match: match.group(0) if match.group(1) in source_map else "",
            raw_answer,
        ).strip()

        if not cited_labels:
            return {
                "answer": UNVERIFIABLE_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved),
                "used_llm": True,
            }

        return {
            "answer": answer_text,
            "sources": [source_map[label] for label in cited_labels],
            "retrieved_count": len(retrieved),
            "used_llm": True,
        }

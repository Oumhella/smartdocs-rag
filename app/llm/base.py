"""The small interface the RAG pipeline expects from any LLM provider."""

from typing import Any, Protocol


class LLMClient(Protocol):
    """Generate one answer from system instructions and a user message."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return model output, optionally constrained by a JSON Schema."""
        ...

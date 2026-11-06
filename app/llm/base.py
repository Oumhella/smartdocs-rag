"""The small interface the RAG pipeline expects from any LLM provider."""

from typing import Protocol


class LLMClient(Protocol):
    """Generate one answer from system instructions and a user message."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's plain-text response."""
        ...

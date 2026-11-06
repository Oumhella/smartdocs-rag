"""Environment-backed settings for local development and deployment."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    rag_min_similarity: float

    @classmethod
    def from_environment(cls) -> "Settings":
        """Read settings from the process environment and project .env file."""
        load_dotenv(PROJECT_ROOT / ".env")

        min_similarity = float(os.getenv("RAG_MIN_SIMILARITY", "0.15"))
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("RAG_MIN_SIMILARITY must be between -1 and 1")

        api_key = os.getenv("GROQ_API_KEY")
        return cls(
            groq_api_key=api_key.strip() if api_key else None,
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
            rag_min_similarity=min_similarity,
        )

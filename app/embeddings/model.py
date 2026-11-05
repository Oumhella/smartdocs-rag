"""Sentence Transformers wrapper for turning text into vectors."""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder:
    """Load the embedding model on first use and encode text in batches."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document chunks, returning plain Python lists for ChromaDB."""
        if not texts:
            return []

        vectors = self.model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=len(texts) > 32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode one search query with the same model used for document chunks."""
        if not query.strip():
            raise ValueError("query must not be empty")

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32).tolist()

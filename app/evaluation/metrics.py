"""Ranking metrics for the SmartDocs retrieval benchmark."""

from collections.abc import Sequence
import re
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for matching a gold passage in a chunk."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def is_relevant(result: dict[str, Any], case: dict[str, Any]) -> bool:
    """Match a retrieval result to a gold document, page, and passage."""
    metadata = result.get("metadata", {})
    if str(metadata.get("filename", "")).casefold() != str(
        case["source_document"]
    ).casefold():
        return False

    expected_page = case.get("expected_page")
    if expected_page is not None and metadata.get("page_number") != expected_page:
        return False

    expected_passage = normalize_text(case["expected_passage"])
    return expected_passage in normalize_text(str(result.get("text", "")))


def reciprocal_rank(
    results: Sequence[dict[str, Any]], case: dict[str, Any]
) -> tuple[int | None, float]:
    """Return the one-based rank of the first gold passage and its reciprocal."""
    for rank, result in enumerate(results, start=1):
        if is_relevant(result, case):
            return rank, 1.0 / rank
    return None, 0.0


def summarize(
    evaluated: Sequence[dict[str, Any]], top_k: int
) -> dict[str, float | int]:
    """Calculate Hit Rate@K, MRR@K, and retrieval latency statistics."""
    if not evaluated:
        raise ValueError("Cannot calculate metrics for an empty evaluation set")

    hits = sum(row["rank"] is not None and row["rank"] <= top_k for row in evaluated)
    reciprocal_ranks = [float(row["reciprocal_rank"]) for row in evaluated]
    latencies = [float(row["latency_ms"]) for row in evaluated]
    return {
        "questions": len(evaluated),
        "top_k": top_k,
        "hit_rate_at_k": hits / len(evaluated),
        "mrr_at_k": sum(reciprocal_ranks) / len(evaluated),
        "mean_retrieval_latency_ms": sum(latencies) / len(latencies),
    }

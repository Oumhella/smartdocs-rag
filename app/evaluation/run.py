"""Run the labeled retrieval benchmark against the local Chroma collection."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from app.evaluation.metrics import reciprocal_rank, summarize
from app.services.document_service import DocumentService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "sickness_policy.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "latest.json"


def run_evaluation(
    dataset_path: Path,
    *,
    top_k: int = 4,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Search every gold question and write a JSON report."""
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a non-empty JSON list")

    service = DocumentService()
    indexed_names = {str(document["filename"]).casefold() for document in service.list_documents()}
    source_inventory: dict[str, dict[str, bool]] = {}
    for source_name in {str(case["source_document"]) for case in cases}:
        source_path = PROJECT_ROOT / "data" / source_name
        if source_name.casefold() not in indexed_names and source_path.is_file():
            print(f"Indexing missing benchmark source: {source_name}")
            service.index_file(source_path)
            indexed_names.add(source_name.casefold())
        elif source_name.casefold() not in indexed_names:
            raise FileNotFoundError(
                f"'{source_name}' is not indexed and was not found in data/"
            )
        source_inventory[source_name] = {
            "indexed": source_name.casefold() in indexed_names,
            "source_file_available": source_path.is_file(),
        }

    # Load the embedding model before measuring per-question retrieval latency.
    service.search(str(cases[0]["question"]), top_k=1)

    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        results = service.search(case["question"], top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        rank, score = reciprocal_rank(results, case)
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_source": case["source_document"],
                "expected_page": case.get("expected_page"),
                "rank": rank,
                "reciprocal_rank": score,
                "latency_ms": round(latency_ms, 2),
                "top_results": [
                    {
                        "filename": result["metadata"].get("filename"),
                        "page_number": result["metadata"].get("page_number"),
                        "chunk_index": result["metadata"].get("chunk_index"),
                        "similarity": result["similarity"],
                        "text_preview": result["text"][:320],
                    }
                    for result in results
                ],
            }
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "sources": source_inventory,
        "latency_method": "Per-question retrieval after one unscored query warms the embedder.",
        "metrics": summarize(rows, top_k),
        "results": rows,
        "limitations": [
            "Gold relevance requires the expected passage to appear in the retrieved chunk.",
            "A locally indexed source can be evaluated even if its original file is absent; that makes the run dependent on this machine's Chroma data.",
            "This report evaluates retrieval only; it does not score generated answer faithfulness or relevance.",
            "Latency measures local retrieval and embedding, not Groq generation or full API latency.",
        ],
    }
    destination = output_path or DEFAULT_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")

    report = run_evaluation(args.dataset, top_k=args.top_k, output_path=args.output)
    print(json.dumps(report["metrics"], indent=2))
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()

"""Extract text from PDFs and split it into source-aware chunks."""

from pathlib import Path
from typing import TypedDict

import fitz


class PageText(TypedDict):
    text: str
    filename: str
    page_number: int


class TextChunk(TypedDict):
    text: str
    filename: str
    page_number: int
    chunk_index: int


def extract_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Return the non-empty text from each PDF page, with its source details.

    Page numbers are one-based so they match the numbers people see in a PDF
    viewer. Pages without extractable text are skipped.
    """
    path = Path(pdf_path)
    pages: list[PageText] = []

    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "text": text,
                        "filename": path.name,
                        "page_number": page_index + 1,
                    }
                )

    return pages


def chunk_pages(
    pages: list[PageText],
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split page text into character-sized chunks while keeping page metadata.

    The overlap repeats a small amount of text between neighboring chunks. This
    helps later search retain context when a useful passage crosses a boundary.
    Chunk indexes start at 1 and restart on each page.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]
        chunk_index = 1
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "filename": page["filename"],
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1
            start += step

    return chunks


def process_pdf(
    pdf_path: str | Path,
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Extract a PDF and return chunks ready for later indexing."""
    pages = extract_pdf_pages(pdf_path)
    return chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)

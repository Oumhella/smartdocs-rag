"""Extract text from supported documents and split it into source-aware chunks."""

from pathlib import Path
import re
from typing import TypedDict

import pymupdf


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")
_EXCESSIVE_BLANK_LINES = re.compile(r"\n(?:[ \t]*\n){2,}")


class TextContent(TypedDict):
    text: str
    filename: str
    page_number: int | None


class TextChunk(TypedDict):
    text: str
    filename: str
    page_number: int | None
    chunk_index: int


def clean_text(text: str) -> str:
    """Normalize extracted text without changing its words or formatting.

    Line endings become ``\n``, non-printing control characters are removed,
    runs of horizontal whitespace become one space, and three or more
    consecutive line breaks become a single blank line.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARACTERS.sub("", text)
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)
    text = _EXCESSIVE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str | Path) -> list[TextContent]:
    """Return the non-empty text from each PDF page, with its source details.

    Page numbers are one-based so they match the numbers people see in a PDF
    viewer. Pages without extractable text are skipped.
    """
    path = Path(pdf_path)
    pages: list[TextContent] = []

    with pymupdf.open(path) as document:
        for page_index, page in enumerate(document):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(
                    {
                        "text": text,
                        "filename": path.name,
                        "page_number": page_index + 1,
                    }
                )

    return pages


def extract_text_file(text_path: str | Path) -> list[TextContent]:
    """Read a UTF-8 TXT or Markdown file as one source text record."""
    path = Path(text_path)
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Text files must use the .txt or .md extension")

    text = clean_text(path.read_text(encoding="utf-8-sig"))
    if not text:
        return []

    return [{"text": text, "filename": path.name, "page_number": None}]


def chunk_text(
    contents: list[TextContent],
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split extracted text into character-sized chunks, preserving metadata.

    The overlap repeats a small amount of text between neighboring chunks. This
    helps later search retain context when a useful passage crosses a boundary.
    Chunk indexes start at 1 and restart for each extracted page or text file.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    step = chunk_size - overlap

    for content in contents:
        text = clean_text(content["text"])
        chunk_index = 1
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "filename": content["filename"],
                        "page_number": content["page_number"],
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
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("process_pdf only accepts .pdf files; use process_document for .txt and .md")

    contents = extract_pdf_pages(path)
    return chunk_text(contents, chunk_size=chunk_size, overlap=overlap)


def process_document(
    document_path: str | Path,
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Process a PDF, TXT, or Markdown file into source-aware text chunks."""
    path = Path(document_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        contents = extract_pdf_pages(path)
    elif suffix in {".txt", ".md"}:
        contents = extract_text_file(path)
    else:
        raise ValueError("Supported document types are .pdf, .txt, and .md")

    return chunk_text(contents, chunk_size=chunk_size, overlap=overlap)

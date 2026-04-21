"""
docs.py
-------
Load Markdown and PDF reference documentation.
Uses paragraph chunking — splits on double newlines to preserve section
structure. Each chunk carries source_type, title, and section metadata
for filtered search.

AI-103 concept: Per-source ingestion strategy — different document types
need different chunking. Docs have clear section structure, so paragraph
chunking preserves semantic units better than fixed or sentence splitting.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger()


@dataclass
class DocChunk:
    id: str
    content: str
    source: str
    source_type: str = "doc"
    title: str = ""
    section: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def load_docs(folder: str, chunk_size: int = 512) -> list[DocChunk]:
    """
    Load all .md and .pdf files from folder and split into DocChunks.

    Args:
        folder: Path to docs directory.
        chunk_size: Max characters per chunk (paragraph strategy uses chars not tokens).

    Returns:
        List of DocChunk objects ready for embedding.
    """
    chunks: list[DocChunk] = []
    folder_path = Path(folder)

    for file_path in sorted(folder_path.iterdir()):
        if file_path.suffix.lower() == ".md":
            text = file_path.read_text(encoding="utf-8")
            title = _extract_title(text, file_path.stem)
        elif file_path.suffix.lower() == ".pdf":
            text = _read_pdf(file_path)
            title = file_path.stem
        else:
            continue

        file_chunks = _chunk_by_paragraph(text, file_path.name, title, chunk_size)
        chunks.extend(file_chunks)
        log.info("docs_loaded", file=file_path.name, chunks=len(file_chunks))

    return chunks


def _extract_title(text: str, fallback: str) -> str:
    """Extract the first # heading as document title."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return fallback


def _extract_section(text: str) -> str:
    """Extract the nearest ## heading above this paragraph."""
    for line in text.splitlines():
        if line.startswith("## "):
            return line.lstrip("# ").strip()
    return ""


def _read_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        log.error("pdf_read_failed", file=str(file_path), error=str(exc))
        return ""


def _chunk_by_paragraph(
    text: str,
    source: str,
    title: str,
    max_chars: int,
) -> list[DocChunk]:
    """
    Split text on double newlines. Accumulate paragraphs until max_chars,
    then start a new chunk. Track the most recent ## heading as section.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[DocChunk] = []
    current_parts: list[str] = []
    current_len = 0
    current_section = ""
    chunk_idx = 0

    for para in paragraphs:
        # Track section headings
        if para.startswith("## "):
            current_section = para.lstrip("# ").strip()

        if current_len + len(para) > max_chars and current_parts:
            chunks.append(DocChunk(
                id=f"{source}_{chunk_idx}",
                content="\n\n".join(current_parts),
                source=source,
                source_type="doc",
                title=title,
                section=current_section,
                chunk_index=chunk_idx,
                metadata={"format": Path(source).suffix.lstrip(".")},
            ))
            chunk_idx += 1
            current_parts = []
            current_len = 0

        current_parts.append(para)
        current_len += len(para)

    if current_parts:
        chunks.append(DocChunk(
            id=f"{source}_{chunk_idx}",
            content="\n\n".join(current_parts),
            source=source,
            source_type="doc",
            title=title,
            section=current_section,
            chunk_index=chunk_idx,
            metadata={"format": Path(source).suffix.lstrip(".")},
        ))

    return chunks
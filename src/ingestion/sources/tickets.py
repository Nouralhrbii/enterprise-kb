"""
tickets.py
----------
Load support tickets from a CSV export (Zendesk, Freshdesk, etc.)
and split into TicketChunks using sentence chunking at a small size.

Why small chunks (256 tokens) for tickets?
Tickets are short and focused. A ticket about "password reset" should
produce a chunk that is ONLY about password reset — not mixed with
billing context. Small chunks = precise retrieval.

Expected CSV columns (at minimum):
  ticket_id, subject, body, resolution, status, created_at

AI-103 concept: Source-aware chunking — the optimal strategy and size
depend on the document type, not a single global setting.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger()


@dataclass
class TicketChunk:
    id: str
    content: str
    source: str
    source_type: str = "ticket"
    ticket_id: str = ""
    subject: str = ""
    status: str = ""
    created_at: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def load_tickets(folder: str, chunk_size: int = 256) -> list[TicketChunk]:
    """
    Load all .csv files from folder and convert to TicketChunks.

    Args:
        folder: Path to tickets directory.
        chunk_size: Max tokens per chunk (sentence strategy).

    Returns:
        List of TicketChunk objects ready for embedding.
    """
    chunks: list[TicketChunk] = []
    folder_path = Path(folder)

    for file_path in sorted(folder_path.iterdir()):
        if file_path.suffix.lower() != ".csv":
            continue

        file_chunks = _process_csv(file_path, chunk_size)
        chunks.extend(file_chunks)
        log.info("tickets_loaded", file=file_path.name, chunks=len(file_chunks))

    return chunks


def _process_csv(file_path: Path, chunk_size: int) -> list[TicketChunk]:
    chunks: list[TicketChunk] = []

    with open(file_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticket_id = row.get("ticket_id", "").strip()
            subject = row.get("subject", "").strip()
            body = row.get("body", "").strip()
            resolution = row.get("resolution", "").strip()
            status = row.get("status", "open").strip()
            created_at = row.get("created_at", "").strip()

            # Combine subject + body + resolution into one text block
            full_text = f"Subject: {subject}\n\n{body}"
            if resolution:
                full_text += f"\n\nResolution: {resolution}"

            ticket_chunks = _sentence_chunk(
                text=full_text,
                source=file_path.name,
                ticket_id=ticket_id,
                subject=subject,
                status=status,
                created_at=created_at,
                max_tokens=chunk_size,
            )
            chunks.extend(ticket_chunks)

    return chunks


def _sentence_chunk(
    text: str,
    source: str,
    ticket_id: str,
    subject: str,
    status: str,
    created_at: str,
    max_tokens: int,
) -> list[TicketChunk]:
    """
    Split text on sentence boundaries, accumulating until max_tokens.
    Uses a simple word-count approximation (1 token ≈ 0.75 words) to
    avoid importing tiktoken — keeping this module lightweight.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[TicketChunk] = []
    current: list[str] = []
    current_words = 0
    chunk_idx = 0
    # 1 token ≈ 0.75 words → max_words = max_tokens * 0.75
    max_words = int(max_tokens * 0.75)

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_words + word_count > max_words and current:
            chunk_text = " ".join(current)
            chunks.append(TicketChunk(
                id=f"ticket_{ticket_id}_{chunk_idx}",
                content=chunk_text,
                source=source,
                source_type="ticket",
                ticket_id=ticket_id,
                subject=subject,
                status=status,
                created_at=created_at,
                chunk_index=chunk_idx,
                metadata={"word_count": current_words},
            ))
            chunk_idx += 1
            current = []
            current_words = 0

        current.append(sentence)
        current_words += word_count

    if current:
        chunks.append(TicketChunk(
            id=f"ticket_{ticket_id}_{chunk_idx}",
            content=" ".join(current),
            source=source,
            source_type="ticket",
            ticket_id=ticket_id,
            subject=subject,
            status=status,
            created_at=created_at,
            chunk_index=chunk_idx,
            metadata={"word_count": current_words},
        ))

    return chunks
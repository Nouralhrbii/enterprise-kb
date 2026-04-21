"""
ingest_tickets.py
-----------------
Ingest support tickets from CSV exports.

Run with:
  python scripts/ingest_tickets.py --folder data/raw/tickets

Do NOT use --overwrite unless you want to wipe docs and changelog too.
Tickets are added alongside existing index content.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from config.settings import settings
from src.ingestion.deduplicator import deduplicate
from src.ingestion.embedder import embed_chunks
from src.ingestion.indexer import upload_documents
from src.ingestion.sources.tickets import load_tickets

log = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(description="Ingest support tickets into the knowledge base.")
    parser.add_argument("--folder", default="data/raw/tickets")
    parser.add_argument("--chunk-size", type=int, default=settings.tickets_chunk_size)
    args = parser.parse_args()

    log.info("ingest_tickets_started", folder=args.folder)

    chunks = load_tickets(args.folder, chunk_size=args.chunk_size)
    log.info("tickets_chunked", total=len(chunks))

    documents = embed_chunks(chunks)
    documents, removed = deduplicate(documents, threshold=settings.dedup_similarity_threshold)
    log.info("deduplication_done", removed=removed)

    upload_documents(documents)
    log.info("ingest_tickets_complete", uploaded=len(documents))


if __name__ == "__main__":
    main()
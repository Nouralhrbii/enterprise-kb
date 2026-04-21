"""
ingest_docs.py
--------------
Ingest Markdown and PDF reference documentation.

Run with:
  python scripts/ingest_docs.py --folder data/raw/docs --overwrite
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from config.settings import settings
from src.ingestion.deduplicator import deduplicate
from src.ingestion.embedder import embed_chunks
from src.ingestion.indexer import create_index, upload_documents
from src.ingestion.sources.docs import load_docs

log = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(description="Ingest documentation into the knowledge base.")
    parser.add_argument("--folder", default="data/raw/docs")
    parser.add_argument("--chunk-size", type=int, default=settings.docs_chunk_size)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log.info("ingest_docs_started", folder=args.folder)

    chunks = load_docs(args.folder, chunk_size=args.chunk_size)
    log.info("docs_chunked", total=len(chunks))

    if args.overwrite:
        create_index(overwrite=True)

    documents = embed_chunks(chunks)
    documents, removed = deduplicate(documents, threshold=settings.dedup_similarity_threshold)
    log.info("deduplication_done", removed=removed)

    upload_documents(documents)
    log.info("ingest_docs_complete", uploaded=len(documents))


if __name__ == "__main__":
    main()
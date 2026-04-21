"""
ingest_changelog.py
-------------------
Ingest product changelog Markdown files.

Run with:
  python scripts/ingest_changelog.py --folder data/raw/changelog
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.ingestion.deduplicator import deduplicate
from src.ingestion.embedder import embed_chunks
from src.ingestion.indexer import upload_documents
from src.ingestion.sources.changelog import load_changelog
from config.settings import settings

log = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(description="Ingest changelog into the knowledge base.")
    parser.add_argument("--folder", default="data/raw/changelog")
    args = parser.parse_args()

    log.info("ingest_changelog_started", folder=args.folder)

    chunks = load_changelog(args.folder)
    log.info("changelog_chunked", total=len(chunks))

    documents = embed_chunks(chunks)
    documents, removed = deduplicate(documents, threshold=settings.dedup_similarity_threshold)
    log.info("deduplication_done", removed=removed)

    upload_documents(documents)
    log.info("ingest_changelog_complete", uploaded=len(documents))


if __name__ == "__main__":
    main()
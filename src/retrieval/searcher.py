"""
searcher.py
-----------
Hybrid search with OData metadata filters.

New:
  - source_type filter  — search only "doc", "ticket", or "changelog"
  - date filter         — search tickets/changelogs after a given date
  - version filter      — search a specific changelog version
  - rerank_top_n        — fetch more candidates for the re-ranker

The filter logic builds OData expressions from the optional parameters.
Multiple filters are combined with 'and'.

AI-103 concept: OData $filter in Azure AI Search — filterable fields
can be combined with 'and', 'or', 'not', comparison operators (eq, gt, lt),
and string functions (search.in). This is a commonly tested topic.
"""

import structlog
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from config.settings import settings
from src.ingestion.embedder import embed_text, get_openai_client
from src.ingestion.indexer import get_search_client

log = structlog.get_logger()


class RetrievedChunk:
    def __init__(
        self,
        id: str,
        content: str,
        source: str,
        source_type: str,
        score: float,
        metadata: dict | None = None,
    ):
        self.id = id
        self.content = content
        self.source = source
        self.source_type = source_type
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"RetrievedChunk(source={self.source!r}, type={self.source_type!r}, score={self.score:.3f})"


def _build_filter(
    source_type: str | None = None,
    after_date: str | None = None,
    version: str | None = None,
) -> str | None:
    """
    Build an OData filter string from optional parameters.

    Args:
        source_type: "doc" | "ticket" | "changelog"
        after_date:  ISO date string "YYYY-MM-DD" — filters created_at or release_date
        version:     Exact version string e.g. "2.1.0"

    Returns:
        OData filter string or None if no filters applied.

    Examples:
        source_type="ticket"              → "source_type eq 'ticket'"
        source_type="ticket", after="2025-01-01"
          → "source_type eq 'ticket' and created_at ge '2025-01-01'"
        source_type="changelog", version="2.1.0"
          → "source_type eq 'changelog' and version eq '2.1.0'"
    """
    parts: list[str] = []

    if source_type:
        parts.append(f"source_type eq '{source_type}'")

    if after_date:
        if source_type == "changelog":
            parts.append(f"release_date ge '{after_date}'")
        else:
            parts.append(f"created_at ge '{after_date}'")

    if version:
        parts.append(f"version eq '{version}'")

    return " and ".join(parts) if parts else None


def search(
    query: str,
    top_k: int | None = None,
    source_type: str | None = None,
    after_date: str | None = None,
    version: str | None = None,
) -> list[RetrievedChunk]:
    """
    Hybrid search (BM25 + HNSW vector + RRF) with optional metadata filters.

    Args:
        query:       User's natural language question.
        top_k:       Number of results to return (defaults to settings.top_k_results).
        source_type: Filter to a specific source type.
        after_date:  Filter to tickets/changelogs after this ISO date.
        version:     Filter to a specific changelog version.

    Returns:
        List of RetrievedChunk objects ordered by relevance score.
    """
    k = top_k or settings.top_k_results
    fetch_n = settings.rerank_top_n  # fetch more for re-ranker to work with

    oai_client = get_openai_client()
    search_client = get_search_client()

    query_vector = embed_text(oai_client, query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=fetch_n,
        fields="embedding",
    )

    odata_filter = _build_filter(source_type, after_date, version)

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content", "source", "source_type", "chunk_index",
                "title", "section", "ticket_id", "subject", "ticket_status",
                "created_at", "version", "release_date"],
        filter=odata_filter,
        top=fetch_n,   # fetch rerank_top_n candidates — reranker will trim to top_k
    )

    chunks = []
    for r in results:
        chunks.append(RetrievedChunk(
            id=r["id"],
            content=r["content"],
            source=r["source"],
            source_type=r.get("source_type", ""),
            score=r.get("@search.score", 0.0),
            metadata={
                "title": r.get("title", ""),
                "section": r.get("section", ""),
                "ticket_id": r.get("ticket_id", ""),
                "subject": r.get("subject", ""),
                "ticket_status": r.get("ticket_status", ""),
                "created_at": r.get("created_at", ""),
                "version": r.get("version", ""),
                "release_date": r.get("release_date", ""),
            },
        ))

    log.info("search_complete", query_preview=query[:60], results=len(chunks), filter=odata_filter)
    return chunks
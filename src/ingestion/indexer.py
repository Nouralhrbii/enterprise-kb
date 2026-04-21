"""
indexer.py
----------
Azure AI Search index with an extended schema to support all three
source types (docs, tickets, changelog) and metadata-filtered search.

New fields:
  source_type  — filterable: "doc" | "ticket" | "changelog"
  title        — doc section title
  section      — doc subsection heading
  ticket_id    — original ticket identifier
  subject      — ticket subject line (filterable, searchable)
  ticket_status — "open" | "resolved" | "pending"
  created_at   — ticket creation date (filterable for date-range queries)
  version      — changelog version string (filterable)
  release_date — changelog release date (filterable)

AI-103 concept: Filterable vs searchable fields — filterable fields use
OData $filter expressions, searchable fields participate in BM25 ranking.
A field can be both (e.g. subject). Setting the wrong combination wastes
compute or breaks filter queries.
"""

import structlog
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from config.settings import settings

log = structlog.get_logger()

INDEX_SCHEMA = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),

    # Content — participates in BM25 keyword search
    SearchableField(name="content", type=SearchFieldDataType.String),

    # Source metadata
    SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, retrievable=True),
    SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, retrievable=True),
    SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, retrievable=True),

    # Doc fields
    SearchableField(name="title", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="section", type=SearchFieldDataType.String, filterable=True),

    # Ticket fields
    SimpleField(name="ticket_id", type=SearchFieldDataType.String, filterable=True, retrievable=True),
    SearchableField(name="subject", type=SearchFieldDataType.String, filterable=True),
    SimpleField(name="ticket_status", type=SearchFieldDataType.String, filterable=True, retrievable=True),
    SimpleField(name="created_at", type=SearchFieldDataType.String, filterable=True, retrievable=True),

    # Changelog fields
    SimpleField(name="version", type=SearchFieldDataType.String, filterable=True, retrievable=True),
    SimpleField(name="release_date", type=SearchFieldDataType.String, filterable=True, retrievable=True),

    # Vector field — HNSW ANN index
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_profile_name="hnsw-profile",
    ),
]

VECTOR_SEARCH_CONFIG = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
    profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-algo")],
)


def get_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=DefaultAzureCredential(),
    )


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=DefaultAzureCredential(),
    )


def create_index(overwrite: bool = False) -> None:
    """Create or update the Azure AI Search index."""
    client = get_index_client()
    if overwrite:
        try:
            client.delete_index(settings.azure_search_index_name)
            log.info("index_deleted", name=settings.azure_search_index_name)
        except Exception:
            pass

    index = SearchIndex(
        name=settings.azure_search_index_name,
        fields=INDEX_SCHEMA,
        vector_search=VECTOR_SEARCH_CONFIG,
    )
    client.create_or_update_index(index)
    log.info("index_created", name=settings.azure_search_index_name)


def upload_documents(documents: list[dict], batch_size: int = 100) -> None:
    """Upload embedded documents in batches."""
    client = get_search_client()
    for i in range(0, len(documents), batch_size):
        batch = documents[i: i + batch_size]
        result = client.upload_documents(documents=batch)
        succeeded = sum(1 for r in result if r.succeeded)
        log.info("batch_uploaded", batch=i // batch_size + 1, succeeded=succeeded)
    log.info("upload_complete", total=len(documents))
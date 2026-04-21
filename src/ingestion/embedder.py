"""
embedder.py
-----------
Embed any chunk type (DocChunk, TicketChunk, ChangelogChunk) into a
flat dict ready for Azure AI Search upload.

Uses text-embedding-3-large via managed identity
The key difference: each source type contributes different metadata fields
to the output dict, all mapped to the extended index schema in indexer.py.

AI-103 concept: Unified embedding layer — regardless of source type,
all chunks go through the same embedding model. The source_type field
in the index lets you filter later without separate indexes.
"""

import structlog
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.ingestion.sources.changelog import ChangelogChunk
from src.ingestion.sources.docs import DocChunk
from src.ingestion.sources.tickets import TicketChunk

log = structlog.get_logger()

AnyChunk = DocChunk | TicketChunk | ChangelogChunk


def get_openai_client() -> AzureOpenAI:
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        azure_ad_token=token.token,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_text(client: AzureOpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=text,
    )
    return response.data[0].embedding


def embed_chunks(chunks: list[AnyChunk]) -> list[dict]:
    """
    Embed all chunks and return a list of dicts matching the index schema.

    Each chunk type contributes its own metadata fields — fields not
    present for a given type are set to empty string so the index
    schema is always satisfied.

    Args:
        chunks: Mixed list of DocChunk, TicketChunk, ChangelogChunk.

    Returns:
        List of dicts ready for upload_documents() in indexer.py.
    """
    client = get_openai_client()
    documents: list[dict] = []

    for i, chunk in enumerate(chunks):
        log.info("embedding_chunk", index=i + 1, total=len(chunks), id=chunk.id)
        vector = embed_text(client, chunk.content)

        doc = {
            "id": chunk.id,
            "content": chunk.content,
            "source": chunk.source,
            "source_type": chunk.source_type,
            "chunk_index": chunk.chunk_index,
            "embedding": vector,
            # Doc-specific
            "title": getattr(chunk, "title", ""),
            "section": getattr(chunk, "section", ""),
            # Ticket-specific
            "ticket_id": getattr(chunk, "ticket_id", ""),
            "subject": getattr(chunk, "subject", ""),
            "ticket_status": getattr(chunk, "status", ""),
            "created_at": getattr(chunk, "created_at", ""),
            # Changelog-specific
            "version": getattr(chunk, "version", ""),
            "release_date": getattr(chunk, "release_date", ""),
        }
        documents.append(doc)

    log.info("embedding_complete", total=len(documents))
    return documents
"""
reranker.py
-----------
Semantic re-ranking pass after hybrid retrieval.

Why re-rank?
RRF (Reciprocal Rank Fusion) merges two ranked lists mathematically.
It's fast and effective but has no understanding of the question's
meaning. Re-ranking uses a cross-encoder model that scores each
(query, chunk) pair together — catching cases where a chunk ranked
10th by RRF is actually the most relevant answer.

How it works here:
We use GPT-4o as the re-ranker (LLM-as-reranker pattern). For each
candidate chunk, we ask GPT-4o to score its relevance to the query
on a 1-10 scale. Then we sort by score and return the top_k.

This is more expensive than a dedicated cross-encoder (like Cohere
Rerank) but requires no extra Azure service — only the OpenAI
deployment you already have.

Production alternative: Use Azure AI Search's built-in semantic ranker
(L2 re-ranking) — faster, cheaper, purpose-built. Enabled by adding
SemanticConfiguration to the index and passing semantic_query to search().

AI-103 concept: Multi-stage retrieval — fetch wide (rerank_top_n),
then re-rank to get precise top_k. This is a standard production pattern.
"""

import json

import structlog
from openai import AzureOpenAI

from config.settings import settings
from src.ingestion.embedder import get_openai_client
from src.retrieval.searcher import RetrievedChunk

log = structlog.get_logger()

RERANK_SYSTEM_PROMPT = """\
You are a relevance scoring assistant. Given a user query and a document passage,
score how relevant the passage is to the query on a scale of 1 to 10.

Rules:
- 10 = the passage directly answers the query with specific details
- 7-9 = the passage is clearly relevant and contains useful information
- 4-6 = the passage is tangentially related but does not directly answer
- 1-3 = the passage is not relevant to the query

Respond with ONLY a JSON object: {"score": <integer 1-10>}
No explanation, no other text.
"""


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
    client: AzureOpenAI | None = None,
) -> list[RetrievedChunk]:
    """
    Re-rank retrieved chunks by relevance to the query using GPT-4o.

    Args:
        query:  The user's original question.
        chunks: Candidate chunks from searcher.search() (rerank_top_n of them).
        top_k:  Number of chunks to return after re-ranking.
        client: Optional pre-built AzureOpenAI client (useful in tests).

    Returns:
        Top top_k chunks sorted by re-rank score descending.

    Note: Each chunk requires one GPT-4o call. With rerank_top_n=20 this
    means 20 small calls per request. Use a fast, cheap deployment for this
    — gpt-4o-mini is ideal. For this project we reuse the same deployment
    for simplicity.
    """
    k = top_k or settings.top_k_results
    if not chunks:
        return []

    oai_client = client or get_openai_client()
    scored: list[tuple[RetrievedChunk, int]] = []

    for chunk in chunks:
        score = _score_chunk(oai_client, query, chunk.content)
        scored.append((chunk, score))
        log.debug("chunk_scored", id=chunk.id, score=score)

    scored.sort(key=lambda x: x[1], reverse=True)
    result = [chunk for chunk, _ in scored[:k]]

    log.info("rerank_complete", candidates=len(chunks), returned=len(result))
    return result


def _score_chunk(client: AzureOpenAI, query: str, passage: str) -> int:
    """Ask GPT-4o to score a single (query, passage) pair. Returns 1-10."""
    try:
        response = client.chat.completions.create(
            model=settings.azure_openai_chat_deployment,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nPassage: {passage[:1000]}"},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return int(data.get("score", 5))
    except Exception as exc:
        log.warning("rerank_score_failed", error=str(exc))
        return 5  # neutral fallback — don't discard the chunk
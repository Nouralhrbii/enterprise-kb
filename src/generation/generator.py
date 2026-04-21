"""
generator.py
------------
Generate a grounded answer using GPT-4o.
with one addition: the GenerationResult now includes which source
types contributed to the answer.
"""

from dataclasses import dataclass, field

import structlog
from openai import AzureOpenAI

from config.settings import settings
from src.generation.prompt_templates import build_messages
from src.ingestion.embedder import get_openai_client
from src.retrieval.searcher import RetrievedChunk

log = structlog.get_logger()


@dataclass
class GenerationResult:
    answer: str
    sources: list[str]
    source_types: list[str]          # which types contributed: ["doc", "ticket"]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * 5 + self.completion_tokens * 15) / 1_000_000


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    client: AzureOpenAI | None = None,
) -> GenerationResult:
    """
    Generate a grounded answer from the retrieved chunks.

    Args:
        question: The user's original question.
        chunks:   Re-ranked chunks from reranker.rerank().
        client:   Optional pre-built AzureOpenAI client (for tests).

    Returns:
        GenerationResult with answer, sources, types, and token usage.
    """
    if not chunks:
        return GenerationResult(
            answer="I could not find relevant information in the knowledge base.",
            sources=[],
            source_types=[],
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    oai_client = client or get_openai_client()
    messages = build_messages(question, chunks)

    response = oai_client.chat.completions.create( # old ??
        model=settings.azure_openai_chat_deployment,
        messages=messages,
        max_tokens=settings.max_tokens_response,
        temperature=0.0,
    )

    answer = response.choices[0].message.content or ""
    usage = response.usage

    sources = list(dict.fromkeys(c.source for c in chunks))
    source_types = list(dict.fromkeys(c.source_type for c in chunks))

    result = GenerationResult(
        answer=answer,
        sources=sources,
        source_types=source_types,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )

    log.info(
        "answer_generated",
        total_tokens=result.total_tokens,
        cost_usd=f"${result.cost_usd:.5f}",
        source_types=result.source_types,
    )
    return result
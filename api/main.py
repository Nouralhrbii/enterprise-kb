"""
main.py
-------
FastAPI application for the Enterprise Knowledge Base.

New vs Document Q&A App:
  - /ask accepts source_type, after_date, version filter params
  - Re-ranking step between retrieval and generation
  - source_types field in response shows which source types contributed

Pipeline per request:
  1. Safety check — input
  2. Hybrid search with optional metadata filters
  3. Semantic re-ranking (trim rerank_top_n → top_k)
  4. Grounded generation
  5. Safety check — output
  6. Return response
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.models import AskRequest, AskResponse, HealthResponse
from config.logging_config import configure_logging
from src.generation.generator import generate_answer
from src.retrieval.reranker import rerank
from src.retrieval.searcher import search
from src.safety.content_safety import BLOCKED_RESPONSE, SafetyDecision, check_text

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("enterprise_kb_starting", version="1.0.0")
    yield
    log.info("enterprise_kb_stopping")


app = FastAPI(
    title="Enterprise Knowledge Base API",
    description="Multi-source grounded Q&A over docs, tickets, and changelogs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse, tags=["qa"])
async def ask(request: AskRequest, http_request: Request):
    start = time.perf_counter()

    # Step 1: safety check on input
    input_safety = check_text(request.question)
    if not input_safety.is_safe:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "content_safety_violation",
                "message": BLOCKED_RESPONSE,
                "flagged_categories": input_safety.flagged_categories,
            },
        )

    # Step 2: hybrid search with optional metadata filters
    candidates = search(
        query=request.question,
        top_k=request.top_k,
        source_type=request.source_type,
        after_date=request.after_date,
        version=request.version,
    )

    if not candidates:
        return AskResponse(
            answer="I couldn't find any relevant information in the knowledge base.",
            sources=[],
            source_types=[],
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    # Step 3: semantic re-ranking — trim candidates to top_k
    chunks = rerank(query=request.question, chunks=candidates, top_k=request.top_k)

    # Step 4: grounded generation
    result = generate_answer(question=request.question, chunks=chunks)

    # Step 5: safety check on output
    output_safety = check_text(result.answer)
    if not output_safety.is_safe:
        raise HTTPException(
            status_code=500,
            detail={"error": "output_safety_violation",
                    "message": "The generated response was blocked by content safety filters."},
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info("request_complete", latency_ms=latency_ms, source_types=result.source_types)

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        source_types=result.source_types,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        latency_ms=latency_ms,
    )
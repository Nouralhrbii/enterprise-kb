# Architecture — Enterprise Knowledge Base

## Overview

A production-ready multi-source RAG API that answers questions grounded in reference documentation, support tickets, and product changelogs. Each source type uses a different ingestion strategy optimised for its structure. Built as Phase 2 of the Microsoft Certified: Azure AI App and Agent Developer Associate (AI-103) portfolio.

---

## What's new vs Phase 1

| | Phase 1 (Doc Q&A) | Phase 2 (Enterprise KB) |
|---|---|---|
| Sources | Single folder | Docs + tickets + changelog |
| Chunking | One strategy | Per-source: paragraph / sentence / version-header |
| Retrieval | Hybrid BM25 + vector | Hybrid + OData metadata filters + semantic re-ranking |
| Index schema | 6 fields | 15 fields including source_type, version, created_at |
| Deduplication | None | Cosine similarity dedup before indexing |
| API filters | None | source_type, after_date, version |
| Re-ranking | None | LLM-as-reranker (GPT-4o scoring per candidate) |

---

## System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ingestion pipelines (offline)                 │
│                                                                  │
│  data/raw/docs/        ──► docs.py        (paragraph chunking)  │
│  data/raw/tickets/     ──► tickets.py     (sentence chunking)   │
│  data/raw/changelog/   ──► changelog.py   (version-header split)│
│                                    │                             │
│                              embedder.py                        │
│                          (text-embedding-3-large)               │
│                                    │                             │
│                            deduplicator.py                      │
│                          (cosine similarity ≥ 0.97 → skip)      │
│                                    │                             │
│                              indexer.py                         │
│                    (Azure AI Search — 15-field schema)          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    Azure AI Search Index (HNSW + BM25)
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                     Query pipeline (per request)                 │
│                                                                  │
│  POST /ask  {question, source_type?, after_date?, version?}     │
│      │                                                           │
│      ▼                                                           │
│  content_safety.py  ──── blocked → 400                          │
│      │ safe                                                      │
│      ▼                                                           │
│  searcher.py                                                     │
│  Hybrid search + OData filter → rerank_top_n candidates         │
│      │                                                           │
│      ▼                                                           │
│  reranker.py                                                     │
│  GPT-4o scores each (query, chunk) pair → top_k chunks          │
│      │                                                           │
│      ▼                                                           │
│  generator.py  ──── source-aware prompt → GPT-4o (temp=0)       │
│      │                                                           │
│      ▼                                                           │
│  content_safety.py  ──── blocked → 500                          │
│      │ safe                                                      │
│      ▼                                                           │
│  AskResponse {answer, sources, source_types, tokens, latency}   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Per-source chunking strategy

### Docs — paragraph chunking
Reference documentation has clear section structure (headers, lists, tables). Paragraph chunking (`\n\n` split) preserves each section as a coherent unit. Section headings are tracked in the `section` metadata field for filtering.

### Tickets — sentence chunking (256 tokens)
Support tickets are short and focused. Small chunks mean high retrieval precision — a "password reset" question retrieves only chunks about password reset, not surrounding billing context. Subject, status, and creation date are stored as filterable metadata.

### Changelog — version-header splitting
Each version block (`## v2.1.0 — 2025-03-01`) becomes exactly one chunk. Splitting further risks separating "Added" and "Fixed" entries from the same release. Version and release date are stored as filterable fields.

---

## Deduplication

Support tickets often contain repeated boilerplate — greetings, closings, legal disclaimers. Before uploading, every chunk embedding is compared against all already-accepted embeddings using cosine similarity. Chunks with similarity ≥ 0.97 are skipped.

This uses an O(n²) algorithm — acceptable for thousands of chunks. For corpora above ~100k chunks, replace with MinHash LSH.

---

## Re-ranking

Hybrid search with RRF merges BM25 and vector rankings efficiently but has no semantic understanding of the question. The re-ranker uses GPT-4o as a cross-encoder: for each of the `rerank_top_n` candidates it asks the model to score relevance 1–10, then returns the top `top_k` by score.

Production alternative: Azure AI Search's built-in semantic ranker (L2 re-ranking) is faster and cheaper. Enable with `SemanticConfiguration` in the index and `semantic_query` in the search call.

---

## Azure services

| Service | Purpose |
|---|---|
| Microsoft Foundry | Project hub, model deployment, tracing |
| Azure OpenAI (GPT-4o) | Answer generation + re-ranking scoring |
| Azure OpenAI (text-embedding-3-large) | Document and query embedding |
| Azure AI Search | 15-field hybrid index with HNSW |
| Azure AI Content Safety | Dual input/output screening |
| Azure AI Evaluation | Groundedness, relevance, coherence scoring |
| Azure App Service | API hosting |
| Azure Key Vault | Secret management |
| Azure Monitor | Latency, token, error rate dashboards |

---

## API reference

### POST /ask

```json
{
  "question": "What changed in v2.0?",
  "top_k": 5,
  "source_type": "changelog",
  "after_date": "2024-01-01",
  "version": "2.0.0"
}
```

All filter fields are optional. `source_type` must be one of `"doc"`, `"ticket"`, `"changelog"`.

```json
{
  "answer": "In v2.0.0 (released 2024-09-01)...",
  "sources": ["changelog.md"],
  "source_types": ["changelog"],
  "prompt_tokens": 1240,
  "completion_tokens": 95,
  "total_tokens": 1335,
  "latency_ms": 1840
}
```

---

## Evaluation results

Run `make eval` to regenerate from `data/eval_set.json` (20 questions, 7 docs, 7 tickets, 6 changelog).

| Metric | Score |
|---|---|
| Avg groundedness | — / 5 |
| Avg relevance | — / 5 |
| Avg coherence | — / 5 |
| Avg latency | — ms |
| Cost per query | — |

> Run `make eval` after ingesting to populate these numbers.

---

## Ingestion order

Always ingest in this order — docs first to create the index, then add tickets and changelog:

```bash
make ingest-all
# Equivalent to:
python scripts/ingest_docs.py --folder data/raw/docs --overwrite
python scripts/ingest_tickets.py --folder data/raw/tickets
python scripts/ingest_changelog.py --folder data/raw/changelog
```

---

## AI-103 exam coverage

| Domain | How it's demonstrated |
|---|---|
| Microsoft Foundry | Hub + project, model registry, tracing |
| Azure OpenAI | Embeddings, generation, re-ranking — all via managed identity |
| RAG | Multi-source pipeline with per-type chunking strategy |
| Azure AI Search | 15-field schema, hybrid search, OData filters, HNSW |
| Responsible AI | Deduplication, dual safety screening, groundedness evaluation |
| Chunking strategies | Three distinct strategies with documented trade-offs |
| Deployment | App Service, Dockerfile, managed identity, CI/CD |
| Observability | Structured JSON logging, latency tracking, eval report |
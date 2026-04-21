# Enterprise Knowledge Base

A production-ready multi-source RAG API that answers questions grounded in reference documentation, support tickets, and product changelogs. Each source type uses a different chunking strategy, retrieval includes semantic re-ranking, and the index supports metadata-filtered search by source type, date, and version.

---

## What's new vs Document Q&A App

- Three source pipelines with per-type chunking strategies
- Cosine similarity deduplication before indexing
- OData metadata filters on the `/ask` endpoint
- Semantic re-ranking (LLM-as-reranker) after hybrid retrieval
- 15-field index schema vs 6 in Phase 1

---

## Quickstart

```bash
git clone https://github.com/Nouralhrbii/enterprise-kb.git
cd enterprise-kb
make install
# Fill in .env with your Azure endpoints

# Ingest all three source types
make ingest-all

# Run the API
make serve   # http://localhost:8001/docs

# Ask a question
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What changed in v2.0?", "source_type": "changelog"}'

# Run unit tests (no Azure needed)
make test

# Run evaluation (requires Azure)
make eval
```

---

## Project structure

```
enterprise-kb/
├── api/
│   ├── main.py              ← POST /ask with source_type, after_date, version filters
│   └── models.py
├── config/
│   ├── settings.py          ← Per-source chunk sizes + rerank_top_n
│   └── logging_config.py
├── src/
│   ├── ingestion/
│   │   ├── sources/
│   │   │   ├── docs.py      ← Paragraph chunking + section tracking
│   │   │   ├── tickets.py   ← CSV loader + sentence chunking
│   │   │   └── changelog.py ← Version-header splitter
│   │   ├── embedder.py      ← Unified embedder for all source types
│   │   ├── deduplicator.py  ← Cosine similarity dedup
│   │   └── indexer.py       ← 15-field schema with HNSW
│   ├── retrieval/
│   │   ├── searcher.py      ← Hybrid + OData filter builder
│   │   └── reranker.py      ← LLM-as-reranker (GPT-4o scoring)
│   ├── generation/
│   │   ├── generator.py
│   │   └── prompt_templates.py ← Source-aware context labels
│   ├── safety/
│   │   └── content_safety.py
│   └── evaluation/
│       └── evaluator.py
├── scripts/
│   ├── ingest_docs.py
│   ├── ingest_tickets.py
│   ├── ingest_changelog.py
│   └── run_evaluation.py
├── tests/                   ← 8 test files, all mocked
├── data/
│   ├── raw/
│   │   ├── docs/            ← Drop .md and .pdf files here
│   │   ├── tickets/         ← Drop .csv exports here
│   │   └── changelog/       ← Drop changelog .md files here
│   └── eval_set.json        ← 20 Q&A pairs across all 3 source types
├── docs/architecture.md
├── Dockerfile
├── .github/workflows/ci.yml
└── Makefile
```

---

## Evaluation results

| Metric | Score |
|---|---|
| Avg groundedness | — / 5 |
| Avg relevance | — / 5 |
| Avg coherence | — / 5 |

> Run `make eval` after ingesting to populate these.

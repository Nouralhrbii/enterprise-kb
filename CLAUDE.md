# CLAUDE.md — enterprise-kb

This file gives Claude (and any AI coding assistant) the context needed to work
effectively in this codebase without asking repetitive questions.

---

## What this project is

A production-ready multi-source RAG API that answers questions grounded in
reference documentation, support tickets, and product changelogs. Each source
type uses a different chunking strategy. Retrieval includes OData metadata
filtering and semantic re-ranking. Built as Phase 2 of an AI-103 portfolio.

Stack: Python 3.11+, FastAPI, Azure OpenAI (GPT-4o + text-embedding-3-large),
Azure AI Search (15-field hybrid index), Azure AI Content Safety,
Azure AI Evaluation, numpy (cosine similarity), pandas (CSV loading),
pydantic-settings, structlog.

---

## Project structure

```
enterprise-kb/
├── api/
│   ├── main.py              # FastAPI — POST /ask with filter params, GET /health
│   └── models.py            # AskRequest (source_type, after_date, version), AskResponse
├── config/
│   ├── settings.py          # Per-source chunk sizes, rerank_top_n, dedup threshold
│   └── logging_config.py
├── src/
│   ├── ingestion/
│   │   ├── sources/
│   │   │   ├── docs.py      # Markdown/PDF → DocChunk (paragraph chunking)
│   │   │   ├── tickets.py   # CSV → TicketChunk (sentence chunking, 256 tokens)
│   │   │   └── changelog.py # Markdown → ChangelogChunk (version-header split)
│   │   ├── embedder.py      # Unified embedder for all 3 chunk types → list[dict]
│   │   ├── deduplicator.py  # Cosine similarity dedup (threshold 0.97) before upload
│   │   └── indexer.py       # 15-field Azure AI Search schema + upload
│   ├── retrieval/
│   │   ├── searcher.py      # Hybrid search + _build_filter (OData expressions)
│   │   └── reranker.py      # LLM-as-reranker: GPT-4o scores each (query,chunk) 1-10
│   ├── generation/
│   │   ├── generator.py     # GPT-4o temp=0, returns source_types in result
│   │   └── prompt_templates.py  # Source-aware labels: [doc], [ticket], [changelog]
│   ├── safety/
│   │   └── content_safety.py
│   └── evaluation/
│       ├── evaluator.py     # Supports source_type per EvalSample for filtered eval
│       └── metrics.py       # Local checks: has_citation, has_version_citation, no_refusal
├── scripts/
│   ├── ingest_docs.py       # Docs pipeline — ALWAYS run first with --overwrite
│   ├── ingest_tickets.py    # Tickets pipeline — run after index exists
│   ├── ingest_changelog.py  # Changelog pipeline — run after index exists
│   └── run_evaluation.py
├── tests/
│   ├── test_docs_loader.py
│   ├── test_tickets_loader.py
│   ├── test_changelog_loader.py
│   ├── test_deduplicator.py
│   ├── test_searcher.py
│   ├── test_reranker.py
│   ├── test_generation.py
│   ├── test_prompt_templates.py
│   ├── test_safety.py
│   ├── test_metrics.py
│   └── test_api.py
├── data/
│   ├── raw/
│   │   ├── docs/            # .md and .pdf reference documentation
│   │   ├── tickets/         # .csv exports (ticket_id,subject,body,resolution,status,created_at)
│   │   └── changelog/       # .md changelog (## vX.Y.Z — YYYY-MM-DD format)
│   └── eval_set.json        # 20 Q&A pairs with source_type per question
├── conftest.py              # ROOT — stubs azure.* + sets env vars
├── docs/architecture.md
├── Dockerfile
├── docker-compose.yml
├── startup.sh
├── Makefile
├── pytest.ini
├── requirements.txt
└── .env.example
```

---

## Source types and their rules

### docs — paragraph chunking
- **Loader**: `src/ingestion/sources/docs.py::load_docs()`
- **Strategy**: Split on `\n\n`, accumulate until `docs_chunk_size` chars
- **Key metadata**: `source_type="doc"`, `title`, `section` (nearest `##` heading)
- **Chunk id**: `"{filename}_{index}"`
- **When to re-ingest**: When docs change or chunking strategy changes

### tickets — sentence chunking
- **Loader**: `src/ingestion/sources/tickets.py::load_tickets()`
- **CSV columns required**: `ticket_id, subject, body, resolution, status, created_at`
- **Content**: Combined as `"Subject: {subject}\n\n{body}\n\nResolution: {resolution}"`
- **Strategy**: Sentence split at `tickets_chunk_size=256` tokens (word-count approximation)
- **Key metadata**: `source_type="ticket"`, `ticket_id`, `subject`, `ticket_status`, `created_at`
- **Chunk id**: `"ticket_{ticket_id}_{index}"`

### changelog — version-header split
- **Loader**: `src/ingestion/sources/changelog.py::load_changelog()`
- **Expected format**: `## v2.1.0 — 2025-03-01` headers, keepachangelog-style
- **Strategy**: One chunk per version block (regex: `^## v?(\d+\.\d+[\.\d]*)`)
- **Key metadata**: `source_type="changelog"`, `version`, `release_date`
- **Chunk id**: `"{filename}_v{version_underscored}"`
- **Never split further**: Version blocks are the correct semantic unit

---

## Key conventions

### Ingestion order (critical)
Always run in this sequence:
```bash
python scripts/ingest_docs.py --folder data/raw/docs --overwrite   # creates index
python scripts/ingest_tickets.py --folder data/raw/tickets          # adds to index
python scripts/ingest_changelog.py --folder data/raw/changelog      # adds to index
```
`ingest_docs.py --overwrite` creates the 15-field index schema. The other two
scripts call `upload_documents()` without creating the index — they will fail
if run before `ingest_docs.py`.

### Deduplication
`deduplicator.deduplicate()` runs before every `upload_documents()` call.
Default threshold: 0.97 (cosine similarity). Chunks with similarity ≥ threshold
are skipped. This is O(n²) — fine for thousands of chunks, use MinHash LSH
for 100k+ chunks.

### OData filter building
`searcher._build_filter()` builds OData expressions from optional params:
- `source_type` → `source_type eq 'ticket'`
- `after_date` + `source_type="changelog"` → `release_date ge 'YYYY-MM-DD'`
- `after_date` + other → `created_at ge 'YYYY-MM-DD'`
- `version` → `version eq '2.1.0'`
Multiple conditions joined with ` and `.

### Re-ranking
`reranker.rerank()` calls GPT-4o once per candidate chunk to score (query, chunk)
relevance 1-10. With `rerank_top_n=20` this means 20 GPT-4o calls + 1 generation
call = 21 total per request. Keep `rerank_top_n` low. Fallback score on exception
is 5 (neutral — never 0 or 10).

### Embedder — unified for all chunk types
`embedder.embed_chunks()` uses `getattr(chunk, 'field', '')` for every metadata
field. This avoids AttributeError when DocChunk lacks `version`, TicketChunk
lacks `section`, etc. All three chunk types produce the same dict schema.

### Source-aware prompt labels
`prompt_templates.build_context_block()` labels each chunk:
- `[doc] guide.md — Installation`
- `[ticket] tickets.csv — Login issue (resolved)`
- `[changelog] v2.1.0 (2025-03-01)`

These labels help GPT-4o cite appropriately. Prompt rules: "For changelog
questions, always include the version number."

### Pipeline flow (request)
```
POST /ask {question, source_type?, after_date?, version?, top_k?}
  → content_safety.check_text(question)            # 400 if blocked
  → searcher.search(query, rerank_top_n candidates) # BM25+HNSW+RRF + OData filter
  → reranker.rerank(candidates → top_k)            # GPT-4o scoring per chunk
  → generator.generate_answer(question, chunks)    # GPT-4o temp=0
  → content_safety.check_text(answer)              # 500 if blocked
  → AskResponse {answer, sources, source_types, tokens, latency_ms}
```

---

## Testing

### Run all unit tests (no Azure needed)
```bash
pytest -m "not integration" -v
```

### How mocking works
Root `conftest.py` stubs `azure.*`, `openai`, `structlog`, `tenacity`, `pandas`
in `sys.modules` before any test file imports `src.*`. It also sets all required
env vars so `Settings()` never raises `ValidationError`.

`numpy` is NOT stubbed — it is a real math library needed by `deduplicator.py`.
Never add numpy to the stubs.

### Test patterns
```python
# Searcher tests — mock both search client and embed function
@patch("src.retrieval.searcher.get_search_client")
@patch("src.retrieval.searcher.embed_text")
def test_filter_passed(mock_embed, mock_search):
    mock_embed.return_value = [0.1] * 3072
    mock_search.return_value.search.return_value = iter([])
    search("query", source_type="ticket")
    assert "source_type eq 'ticket'" in mock_search.return_value.search.call_args.kwargs["filter"]

# Reranker tests — pass client explicitly
def test_highest_scored_first():
    client = MagicMock()
    client.chat.completions.create.side_effect = [mock_score(2), mock_score(9)]
    result = rerank("query", chunks, top_k=2, client=client)
    assert result[0].id == "high_scored_chunk_id"
```

### Deduplicator tests — use real vectors
```python
# numpy is real — use actual float vectors, not mocks
def test_identical_vectors_removed():
    v = [1.0, 0.0, 0.0]
    docs = [_make_doc("a", v), _make_doc("b", v)]
    result, removed = deduplicate(docs, threshold=0.97)
    assert removed == 1
```

---

## Common tasks

### Add a new source type (e.g. Confluence pages)
1. Create `src/ingestion/sources/confluence.py` with a `ConfluenceChunk` dataclass
   and a `load_confluence(folder)` function
2. Add metadata fields to `indexer.py::INDEX_SCHEMA` (e.g. `page_id`, `space_key`)
3. Add `getattr(chunk, 'page_id', '')` lines to `embedder.py::embed_chunks()`
4. Add a `confluence` option to `AskRequest.source_type` Literal in `models.py`
5. Add `scripts/ingest_confluence.py`
6. Add `tests/test_confluence_loader.py`

### Add a new filter parameter (e.g. filter by ticket_id)
1. Add `ticket_id: str | None` to `AskRequest` in `models.py`
2. Add `if ticket_id: parts.append(f"ticket_id eq '{ticket_id}'")` to `_build_filter()`
3. Pass `ticket_id=request.ticket_id` through `api/main.py` to `search()`
4. Add parameter to `searcher.search()` signature
5. Add test in `tests/test_searcher.py` and `tests/test_api.py`

### Change dedup threshold
In `.env`:
```
DEDUP_SIMILARITY_THRESHOLD=0.95   # more aggressive
DEDUP_SIMILARITY_THRESHOLD=0.99   # more conservative
```
Then re-run the relevant ingest script.

### Tune re-ranking
In `.env`:
```
RERANK_TOP_N=30   # fetch more candidates (more GPT-4o calls)
TOP_K_RESULTS=3   # return fewer final chunks (cheaper generation)
```

---

## Deployment

### Local dev
```bash
make install
make ingest-all    # requires real .env
make serve         # http://localhost:8001/docs
make test          # 142 tests, no Azure needed
```

### Docker
```bash
docker build -t enterprise-kb .
docker-compose up
```

### Azure App Service
```bash
make deploy
make assign-identity
# Grant managed identity:
# - Cognitive Services OpenAI User (Azure OpenAI)
# - Search Index Data Contributor (Azure AI Search)
# - Cognitive Services User (Content Safety)
```

---

## Azure service dependencies

| Service | Used in | Notes |
|---|---|---|
| Azure OpenAI (GPT-4o) | generator.py, reranker.py | Same deployment for both |
| Azure OpenAI (text-embedding-3-large) | embedder.py | 3072 dimensions |
| Azure AI Search | indexer.py, searcher.py | 15-field schema |
| Azure AI Content Safety | content_safety.py | Dual screening |
| Azure AI Evaluation | evaluator.py | Groundedness, relevance, coherence |

---

## What NOT to do

- Do not run `ingest_tickets.py` or `ingest_changelog.py` before `ingest_docs.py --overwrite`
- Do not stub `numpy` in conftest.py — it is a real dependency for deduplicator.py
- Do not sentence-split changelog version blocks — they are atomic semantic units
- Do not set `rerank_top_n` above 50 — each candidate costs one GPT-4o call
- Do not change `temperature=0.0` in generator.py without a documented reason
- Do not add `data/raw/` to git — documents may be proprietary or contain PII
- Do not hardcode API keys — use DefaultAzureCredential throughout
- Do not run evaluation before all three ingest scripts — changelog/ticket scores will be ~1.0
- Do not import `azure.*` at module top-level in test files — breaks test collection
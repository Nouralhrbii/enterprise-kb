"""
test_searcher.py
----------------
Unit tests for hybrid search and OData filter building.
All Azure calls are mocked.
"""

from unittest.mock import MagicMock, patch
import pytest
from src.retrieval.searcher import RetrievedChunk, _build_filter, search


class TestBuildFilter:
    def test_no_filters_returns_none(self):
        assert _build_filter() is None

    def test_source_type_only(self):
        f = _build_filter(source_type="ticket")
        assert f == "source_type eq 'ticket'"

    def test_source_type_and_after_date_for_ticket(self):
        f = _build_filter(source_type="ticket", after_date="2025-01-01")
        assert "source_type eq 'ticket'" in f
        assert "created_at ge '2025-01-01'" in f

    def test_source_type_and_after_date_for_changelog(self):
        f = _build_filter(source_type="changelog", after_date="2025-01-01")
        assert "release_date ge '2025-01-01'" in f

    def test_version_filter(self):
        f = _build_filter(source_type="changelog", version="2.1.0")
        assert "version eq '2.1.0'" in f

    def test_multiple_filters_joined_with_and(self):
        f = _build_filter(source_type="ticket", after_date="2025-01-01")
        assert " and " in f

    def test_version_without_source_type(self):
        f = _build_filter(version="2.0.0")
        assert f == "version eq '2.0.0'"


class TestSearch:
    def _make_result(self, id, content, source, source_type, score):
        r = {
            "id": id, "content": content, "source": source,
            "source_type": source_type, "chunk_index": 0,
            "title": "", "section": "", "ticket_id": "", "subject": "",
            "ticket_status": "", "created_at": "", "version": "", "release_date": "",
            "@search.score": score,
        }
        return r

    @patch("src.retrieval.searcher.get_search_client")
    @patch("src.retrieval.searcher.get_openai_client")
    @patch("src.retrieval.searcher.embed_text")
    def test_returns_retrieved_chunks(self, mock_embed, mock_oai, mock_search):
        mock_embed.return_value = [0.1] * 3072
        mock_search.return_value.search.return_value = iter([
            self._make_result("doc_0", "Content.", "guide.md", "doc", 0.95),
        ])
        results = search("test query")
        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].source_type == "doc"

    @patch("src.retrieval.searcher.get_search_client")
    @patch("src.retrieval.searcher.get_openai_client")
    @patch("src.retrieval.searcher.embed_text")
    def test_passes_source_type_filter(self, mock_embed, mock_oai, mock_search):
        mock_embed.return_value = [0.1] * 3072
        mock_search.return_value.search.return_value = iter([])
        search("query", source_type="ticket")
        call_kwargs = mock_search.return_value.search.call_args.kwargs
        assert "source_type eq 'ticket'" in (call_kwargs.get("filter") or "")

    @patch("src.retrieval.searcher.get_search_client")
    @patch("src.retrieval.searcher.get_openai_client")
    @patch("src.retrieval.searcher.embed_text")
    def test_no_filter_when_no_params(self, mock_embed, mock_oai, mock_search):
        mock_embed.return_value = [0.1] * 3072
        mock_search.return_value.search.return_value = iter([])
        search("query")
        call_kwargs = mock_search.return_value.search.call_args.kwargs
        assert call_kwargs.get("filter") is None

    @patch("src.retrieval.searcher.get_search_client")
    @patch("src.retrieval.searcher.get_openai_client")
    @patch("src.retrieval.searcher.embed_text")
    def test_empty_results_returns_empty_list(self, mock_embed, mock_oai, mock_search):
        mock_embed.return_value = [0.1] * 3072
        mock_search.return_value.search.return_value = iter([])
        assert search("query") == []
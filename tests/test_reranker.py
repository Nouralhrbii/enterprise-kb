"""
test_reranker.py
----------------
Unit tests for the LLM-as-reranker.
Mocks the AzureOpenAI client.
"""

from unittest.mock import MagicMock, patch
import pytest
from src.retrieval.reranker import rerank, _score_chunk
from src.retrieval.searcher import RetrievedChunk


def _chunk(id, content, score=0.8, source_type="doc"):
    return RetrievedChunk(id=id, content=content, source="doc.md",
                          source_type=source_type, score=score)


def _mock_score_response(score: int):
    mock = MagicMock()
    mock.choices[0].message.content = f'{{"score": {score}}}'
    return mock


class TestScoreChunk:
    def test_returns_integer_score(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_score_response(8)
        score = _score_chunk(client, "How do I reset password?", "Go to Settings.")
        assert score == 8

    def test_returns_fallback_on_bad_json(self):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = "invalid json"
        score = _score_chunk(client, "query", "passage")
        assert score == 5  # neutral fallback

    def test_returns_fallback_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        score = _score_chunk(client, "query", "passage")
        assert score == 5


class TestRerank:
    def test_empty_chunks_returns_empty(self):
        result = rerank("query", [], top_k=5)
        assert result == []

    def test_returns_top_k_chunks(self):
        chunks = [_chunk(f"c{i}", f"Content {i}") for i in range(10)]
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_score_response(i) for i in range(10)
        ]
        with patch("src.retrieval.reranker.get_openai_client", return_value=client):
            result = rerank("query", chunks, top_k=3, client=client)
        assert len(result) == 3

    def test_highest_scored_chunk_is_first(self):
        chunks = [
            _chunk("low", "Low relevance content", score=0.9),
            _chunk("high", "High relevance content", score=0.5),
        ]
        client = MagicMock()
        # low gets score 2, high gets score 9
        client.chat.completions.create.side_effect = [
            _mock_score_response(2),
            _mock_score_response(9),
        ]
        result = rerank("query", chunks, top_k=2, client=client)
        assert result[0].id == "high"

    def test_uses_temperature_zero(self):
        chunks = [_chunk("c1", "Content")]
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_score_response(7)
        rerank("query", chunks, top_k=1, client=client)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.0
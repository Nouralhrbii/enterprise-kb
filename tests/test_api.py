"""
test_api.py
-----------
Integration tests for the Enterprise KB FastAPI endpoints.
Covers filter params, re-ranking step, and safety gates.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.retrieval.searcher import RetrievedChunk
from src.safety.content_safety import SafetyDecision, SafetyResult

client = TestClient(app)


def _safe():
    return SafetyResult(decision=SafetyDecision.SAFE, flagged_categories=[],
                        max_severity=0, raw_scores={})


def _blocked():
    return SafetyResult(decision=SafetyDecision.BLOCKED, flagged_categories=["Hate"],
                        max_severity=6, raw_scores={"Hate": 6})


def _chunk(source_type="doc"):
    return RetrievedChunk(id="c1", content="Reset in Settings.", source="guide.md",
                          source_type=source_type, score=0.9)


def _gen_result(source_type="doc"):
    from src.generation.generator import GenerationResult
    return GenerationResult(
        answer="Go to Settings. (Source: guide.md)",
        sources=["guide.md"],
        source_types=[source_type],
        prompt_tokens=200,
        completion_tokens=30,
        total_tokens=230,
    )


class TestHealthEndpoint:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_returns_ok(self):
        assert client.get("/health").json() == {"status": "ok"}


class TestAskEndpoint:
    @patch("api.main.check_text", return_value=_safe())
    @patch("api.main.search", return_value=[_chunk()])
    @patch("api.main.rerank", return_value=[_chunk()])
    @patch("api.main.generate_answer", return_value=_gen_result())
    def test_happy_path_returns_200(self, *mocks):
        r = client.post("/ask", json={"question": "How do I reset my password?"})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "source_types" in body
        assert "latency_ms" in body

    @patch("api.main.check_text", return_value=_safe())
    @patch("api.main.search", return_value=[_chunk("ticket")])
    @patch("api.main.rerank", return_value=[_chunk("ticket")])
    @patch("api.main.generate_answer", return_value=_gen_result("ticket"))
    def test_source_types_in_response(self, *mocks):
        r = client.post("/ask", json={"question": "Any open ticket?", "source_type": "ticket"})
        assert "ticket" in r.json()["source_types"]

    @patch("api.main.check_text", return_value=_blocked())
    def test_blocked_input_returns_400(self, _):
        r = client.post("/ask", json={"question": "harmful content here"})
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "content_safety_violation"

    @patch("api.main.check_text", return_value=_safe())
    @patch("api.main.search", return_value=[])
    def test_empty_search_returns_200_fallback(self, *mocks):
        r = client.post("/ask", json={"question": "no match question here"})
        assert r.status_code == 200
        assert r.json()["sources"] == []

    def test_missing_question_returns_422(self):
        assert client.post("/ask", json={}).status_code == 422

    def test_short_question_returns_422(self):
        assert client.post("/ask", json={"question": "Hi"}).status_code == 422

    def test_invalid_source_type_returns_422(self):
        r = client.post("/ask", json={"question": "Valid question?", "source_type": "invalid"})
        assert r.status_code == 422

    def test_invalid_after_date_returns_422(self):
        r = client.post("/ask", json={"question": "Valid question?", "after_date": "not-a-date"})
        assert r.status_code == 422

    def test_valid_after_date_accepted(self):
        with patch("api.main.check_text", return_value=_safe()), \
             patch("api.main.search", return_value=[]), \
             patch("api.main.rerank", return_value=[]):
            r = client.post("/ask", json={"question": "Valid question?", "after_date": "2025-01-01"})
            assert r.status_code == 200

    @patch("api.main.check_text", return_value=_safe())
    @patch("api.main.search", return_value=[_chunk()])
    @patch("api.main.rerank", return_value=[_chunk()])
    @patch("api.main.generate_answer", return_value=_gen_result())
    def test_search_called_with_filter_params(self, mock_gen, mock_rerank, mock_search, mock_safety):
        client.post("/ask", json={
            "question": "What changed?",
            "source_type": "changelog",
            "version": "2.1.0",
        })
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("source_type") == "changelog"
        assert call_kwargs.get("version") == "2.1.0"
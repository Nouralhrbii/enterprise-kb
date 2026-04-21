"""
test_generation.py
------------------
Unit tests for generator.py and prompt_templates.py.
All Azure calls are mocked.
"""

from unittest.mock import MagicMock
import pytest
from src.generation.generator import GenerationResult, generate_answer
from src.generation.prompt_templates import build_context_block, build_messages
from src.retrieval.searcher import RetrievedChunk


def _chunk(content, source, source_type="doc", version="", ticket_id="", subject="", ticket_status="", section=""):
    c = RetrievedChunk(id=f"{source}_0", content=content, source=source,
                       source_type=source_type, score=0.9)
    c.metadata = {
        "version": version, "release_date": "", "ticket_id": ticket_id,
        "subject": subject, "ticket_status": ticket_status, "section": section,
    }
    return c


def _oai_response(content="Answer.", prompt=200, completion=50):
    mock = MagicMock()
    mock.choices[0].message.content = content
    mock.usage.prompt_tokens = prompt
    mock.usage.completion_tokens = completion
    mock.usage.total_tokens = prompt + completion
    return mock


class TestBuildContextBlock:
    def test_doc_chunk_labelled_correctly(self):
        chunk = _chunk("Some doc content.", "guide.md", source_type="doc", section="Installation")
        ctx = build_context_block([chunk])
        assert "[doc]" in ctx
        assert "guide.md" in ctx
        assert "Installation" in ctx

    def test_ticket_chunk_labelled_correctly(self):
        chunk = _chunk("Ticket body.", "tickets.csv", source_type="ticket",
                       subject="Login issue", ticket_status="resolved")
        ctx = build_context_block([chunk])
        assert "[ticket]" in ctx
        assert "Login issue" in ctx
        assert "resolved" in ctx

    def test_changelog_chunk_labelled_correctly(self):
        chunk = _chunk("Added feature X.", "changelog.md", source_type="changelog", version="2.1.0")
        ctx = build_context_block([chunk])
        assert "[changelog]" in ctx
        assert "v2.1.0" in ctx

    def test_multiple_chunks_separated_by_divider(self):
        chunks = [
            _chunk("Content A.", "a.md", source_type="doc"),
            _chunk("Content B.", "b.csv", source_type="ticket"),
        ]
        ctx = build_context_block(chunks)
        assert "---" in ctx
        assert "Content A." in ctx
        assert "Content B." in ctx

    def test_empty_chunks_returns_empty_string(self):
        assert build_context_block([]) == ""

    def test_numbered_labels(self):
        chunks = [_chunk("A", "a.md"), _chunk("B", "b.md")]
        ctx = build_context_block(chunks)
        assert "[1]" in ctx
        assert "[2]" in ctx


class TestBuildMessages:
    def test_returns_system_and_user_messages(self):
        chunks = [_chunk("Context.", "doc.md")]
        messages = build_messages("What is X?", chunks)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_contains_context(self):
        chunks = [_chunk("The answer is 42.", "doc.md")]
        messages = build_messages("What is the answer?", chunks)
        assert "The answer is 42." in messages[0]["content"]

    def test_user_message_contains_question(self):
        chunks = [_chunk("ctx", "doc.md")]
        messages = build_messages("How do I reset my password?", chunks)
        assert "How do I reset my password?" in messages[1]["content"]


class TestGenerateAnswer:
    def test_returns_generation_result(self):
        chunks = [_chunk("Reset in Settings.", "guide.md", source_type="doc")]
        client = MagicMock()
        client.chat.completions.create.return_value = _oai_response("Go to Settings. (Source: guide.md)")
        result = generate_answer("How do I reset?", chunks, client=client)
        assert isinstance(result, GenerationResult)
        assert "Settings" in result.answer
        assert "guide.md" in result.sources
        assert "doc" in result.source_types

    def test_returns_fallback_when_no_chunks(self):
        result = generate_answer("question", [])
        assert result.total_tokens == 0
        assert result.sources == []
        assert result.source_types == []

    def test_deduplicates_sources_and_types(self):
        chunks = [
            _chunk("A", "guide.md", source_type="doc"),
            _chunk("B", "guide.md", source_type="doc"),
            _chunk("C", "tickets.csv", source_type="ticket"),
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = _oai_response("Answer.")
        result = generate_answer("question", chunks, client=client)
        assert result.sources.count("guide.md") == 1
        assert result.source_types.count("doc") == 1
        assert "ticket" in result.source_types

    def test_uses_temperature_zero(self):
        chunks = [_chunk("ctx", "doc.md")]
        client = MagicMock()
        client.chat.completions.create.return_value = _oai_response()
        generate_answer("q", chunks, client=client)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.0

    def test_cost_usd_is_positive(self):
        chunks = [_chunk("ctx", "doc.md")]
        client = MagicMock()
        client.chat.completions.create.return_value = _oai_response(prompt=500, completion=100)
        result = generate_answer("q", chunks, client=client)
        assert result.cost_usd > 0
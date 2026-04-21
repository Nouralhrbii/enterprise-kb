"""
test_prompt_templates.py
------------------------
Unit tests for source-aware prompt template building.
Pure Python — no Azure calls.
"""

from src.generation.prompt_templates import (
    SYSTEM_PROMPT_BASE,
    build_context_block,
    build_messages,
)
from src.retrieval.searcher import RetrievedChunk


def _chunk(content, source, source_type, **meta):
    c = RetrievedChunk(id=f"{source}_0", content=content, source=source,
                       source_type=source_type, score=0.9)
    c.metadata = {
        "version": meta.get("version", ""),
        "release_date": meta.get("release_date", ""),
        "ticket_id": meta.get("ticket_id", ""),
        "subject": meta.get("subject", ""),
        "ticket_status": meta.get("ticket_status", ""),
        "section": meta.get("section", ""),
    }
    return c


class TestBuildContextBlock:
    def test_empty_list_returns_empty_string(self):
        assert build_context_block([]) == ""

    def test_single_doc_chunk(self):
        chunk = _chunk("Content here.", "guide.md", "doc", section="Setup")
        ctx = build_context_block([chunk])
        assert "guide.md" in ctx
        assert "Setup" in ctx
        assert "[doc]" in ctx
        assert "Content here." in ctx

    def test_ticket_includes_subject_and_status(self):
        chunk = _chunk("Body text.", "t.csv", "ticket",
                       subject="Login bug", ticket_status="resolved")
        ctx = build_context_block([chunk])
        assert "Login bug" in ctx
        assert "resolved" in ctx

    def test_changelog_includes_version_and_date(self):
        chunk = _chunk("## v2.1.0\n- Added X", "changelog.md", "changelog",
                       version="2.1.0", release_date="2025-03-01")
        ctx = build_context_block([chunk])
        assert "v2.1.0" in ctx
        assert "2025-03-01" in ctx

    def test_doc_section_omitted_when_empty(self):
        chunk = _chunk("Content.", "guide.md", "doc", section="")
        ctx = build_context_block([chunk])
        assert "[doc] guide.md" in ctx
        assert " — " not in ctx   # no dash when no section

    def test_multiple_chunks_numbered_sequentially(self):
        chunks = [
            _chunk("A", "a.md", "doc"),
            _chunk("B", "b.csv", "ticket", subject="Issue", ticket_status="open"),
            _chunk("C", "c.md", "changelog", version="1.0.0"),
        ]
        ctx = build_context_block(chunks)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "[3]" in ctx

    def test_chunks_separated_by_horizontal_rule(self):
        chunks = [_chunk("A", "a.md", "doc"), _chunk("B", "b.md", "doc")]
        ctx = build_context_block(chunks)
        assert "---" in ctx


class TestBuildMessages:
    def test_returns_two_messages(self):
        chunks = [_chunk("ctx", "doc.md", "doc")]
        messages = build_messages("What is X?", chunks)
        assert len(messages) == 2

    def test_system_role_first(self):
        messages = build_messages("q", [_chunk("c", "d.md", "doc")])
        assert messages[0]["role"] == "system"

    def test_user_role_second(self):
        messages = build_messages("q", [_chunk("c", "d.md", "doc")])
        assert messages[1]["role"] == "user"

    def test_question_in_user_message(self):
        messages = build_messages("How do I export data?", [_chunk("c", "d.md", "doc")])
        assert "How do I export data?" in messages[1]["content"]

    def test_context_in_system_message(self):
        chunks = [_chunk("Reset in Settings.", "guide.md", "doc")]
        messages = build_messages("q", chunks)
        assert "Reset in Settings." in messages[0]["content"]

    def test_system_prompt_contains_grounding_rules(self):
        messages = build_messages("q", [_chunk("c", "d.md", "doc")])
        system = messages[0]["content"]
        assert "ONLY" in system or "only" in system
        assert "cite" in system.lower()
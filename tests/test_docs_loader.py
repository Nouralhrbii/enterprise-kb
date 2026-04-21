"""
test_docs_loader.py
-------------------
Unit tests for the docs source loader and paragraph chunker.
Uses real temporary files — no Azure calls needed.
"""

import pytest
from src.ingestion.sources.docs import DocChunk, load_docs, _chunk_by_paragraph, _extract_title


class TestExtractTitle:
    def test_extracts_h1_heading(self):
        text = "# My Document\n\nSome content."
        assert _extract_title(text, "fallback") == "My Document"

    def test_falls_back_to_filename_stem(self):
        text = "No heading here, just content."
        assert _extract_title(text, "my-doc") == "my-doc"

    def test_uses_first_h1_only(self):
        text = "# First\n\n# Second\n\nContent."
        assert _extract_title(text, "fb") == "First"


class TestChunkByParagraph:
    def test_produces_chunks(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = _chunk_by_paragraph(text, "doc.md", "My Doc", max_chars=9999)
        assert len(chunks) == 1  # all fit in one chunk at large max_chars
        assert "Para one." in chunks[0].content

    def test_splits_on_max_chars(self):
        paras = ["A" * 100 for _ in range(10)]
        text = "\n\n".join(paras)
        chunks = _chunk_by_paragraph(text, "doc.md", "Title", max_chars=150)
        assert len(chunks) > 1

    def test_source_type_is_doc(self):
        chunks = _chunk_by_paragraph("Content here.", "doc.md", "Title", max_chars=9999)
        assert chunks[0].source_type == "doc"

    def test_source_preserved(self):
        chunks = _chunk_by_paragraph("Content.", "user-guide.md", "Title", max_chars=9999)
        assert chunks[0].source == "user-guide.md"

    def test_title_preserved(self):
        chunks = _chunk_by_paragraph("Content.", "doc.md", "My Guide", max_chars=9999)
        assert chunks[0].title == "My Guide"

    def test_section_tracked(self):
        text = "## Installation\n\nInstall the package.\n\n## Configuration\n\nSet env vars."
        chunks = _chunk_by_paragraph(text, "doc.md", "Guide", max_chars=50)
        section_values = [c.section for c in chunks]
        assert "Installation" in section_values or "Configuration" in section_values

    def test_chunk_ids_unique(self):
        paras = ["Word " * 50 for _ in range(10)]
        text = "\n\n".join(paras)
        chunks = _chunk_by_paragraph(text, "doc.md", "Title", max_chars=100)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_text_returns_no_chunks(self):
        chunks = _chunk_by_paragraph("", "doc.md", "Title", max_chars=512)
        assert chunks == []


class TestLoadDocs:
    def test_loads_markdown_file(self, tmp_path):
        (tmp_path / "guide.md").write_text("# Guide\n\nSome content here.")
        chunks = load_docs(str(tmp_path))
        assert len(chunks) >= 1
        assert any("Some content" in c.content for c in chunks)

    def test_skips_non_md_files(self, tmp_path):
        (tmp_path / "data.csv").write_text("col1,col2\n1,2")
        chunks = load_docs(str(tmp_path))
        assert chunks == []

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n\nContent A.")
        (tmp_path / "b.md").write_text("# B\n\nContent B.")
        chunks = load_docs(str(tmp_path))
        sources = {c.source for c in chunks}
        assert "a.md" in sources
        assert "b.md" in sources

    def test_empty_folder(self, tmp_path):
        assert load_docs(str(tmp_path)) == []
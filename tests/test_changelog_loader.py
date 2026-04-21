"""
test_changelog_loader.py
------------------------
Unit tests for the version-header changelog splitter.
"""

import pytest
from src.ingestion.sources.changelog import (
    ChangelogChunk, load_changelog, _split_by_version, _parse_version_and_date
)

SAMPLE_CHANGELOG = """# Changelog

## v2.1.0 — 2025-03-01

### Added
- Feature A
- Feature B

### Fixed
- Bug C

## v2.0.0 — 2025-01-15

### Added
- Feature X

### Breaking
- Removed old API
"""


class TestParseVersionAndDate:
    def test_parses_version_and_date(self):
        v, d = _parse_version_and_date("## v2.1.0 — 2025-03-01")
        assert v == "2.1.0"
        assert d == "2025-03-01"

    def test_parses_version_without_date(self):
        v, d = _parse_version_and_date("## v1.0.0")
        assert v == "1.0.0"
        assert d == ""

    def test_parses_version_without_v_prefix(self):
        v, d = _parse_version_and_date("## 3.0.0 — 2025-06-01")
        assert v == "3.0.0"

    def test_empty_header_returns_empty_strings(self):
        v, d = _parse_version_and_date("## Unreleased")
        assert v == ""
        assert d == ""


class TestSplitByVersion:
    def test_produces_correct_number_of_chunks(self):
        chunks = _split_by_version(SAMPLE_CHANGELOG, "changelog.md")
        assert len(chunks) == 2

    def test_version_extracted_correctly(self):
        chunks = _split_by_version(SAMPLE_CHANGELOG, "changelog.md")
        versions = [c.version for c in chunks]
        assert "2.1.0" in versions
        assert "2.0.0" in versions

    def test_release_date_extracted(self):
        chunks = _split_by_version(SAMPLE_CHANGELOG, "changelog.md")
        dates = [c.release_date for c in chunks]
        assert "2025-03-01" in dates
        assert "2025-01-15" in dates

    def test_source_type_is_changelog(self):
        chunks = _split_by_version(SAMPLE_CHANGELOG, "changelog.md")
        assert all(c.source_type == "changelog" for c in chunks)

    def test_content_contains_version_block(self):
        chunks = _split_by_version(SAMPLE_CHANGELOG, "changelog.md")
        v210 = next(c for c in chunks if c.version == "2.1.0")
        assert "Feature A" in v210.content
        assert "Bug C" in v210.content

    def test_chunks_do_not_bleed_between_versions(self):
        chunks = _split_by_version(SAMPLE_CHANGELOG, "changelog.md")
        v200 = next(c for c in chunks if c.version == "2.0.0")
        assert "Feature A" not in v200.content

    def test_no_version_headers_returns_single_chunk(self):
        text = "Some text without version headers."
        chunks = _split_by_version(text, "changelog.md")
        assert len(chunks) == 1
        assert chunks[0].content == text.strip()

    def test_chunk_ids_are_unique(self):
        chunks = _split_by_version(SAMPLE_CHANGELOG, "changelog.md")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestLoadChangelog:
    def test_loads_md_file(self, tmp_path):
        (tmp_path / "changelog.md").write_text(SAMPLE_CHANGELOG)
        chunks = load_changelog(str(tmp_path))
        assert len(chunks) == 2

    def test_skips_non_md_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not a changelog")
        chunks = load_changelog(str(tmp_path))
        assert chunks == []

    def test_empty_folder(self, tmp_path):
        assert load_changelog(str(tmp_path)) == []
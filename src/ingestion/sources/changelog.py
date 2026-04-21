"""
changelog.py
------------
Load a Markdown changelog and split it into one chunk per version block.

Why one chunk per version?
A changelog entry for v2.1.0 is a complete, coherent unit. Splitting it
further risks separating "Added X" from "Fixed Y" in the same release,
making retrieval fragile for questions like "what changed in v2.1?".

Expected format:
  ## v2.1.0 — 2025-03-15
  ### Added
  - Feature A
  ### Fixed
  - Bug B

  ## v2.0.0 — 2025-01-10
  ...

AI-103 concept: Custom chunking strategy — when paragraph or sentence
splitting doesn't match the document's logical units, write a domain-
specific splitter. This is a common scenario-based exam question.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger()

VERSION_HEADER_RE = re.compile(r"^##\s+v?(\d+\.\d+[\.\d]*)", re.MULTILINE)


@dataclass
class ChangelogChunk:
    id: str
    content: str
    source: str
    source_type: str = "changelog"
    version: str = ""
    release_date: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def load_changelog(folder: str) -> list[ChangelogChunk]:
    """
    Load all .md files from the changelog folder.
    Each file is expected to be a standard keepachangelog-style Markdown file.

    Args:
        folder: Path to changelog directory.

    Returns:
        List of ChangelogChunk objects, one per version block.
    """
    chunks: list[ChangelogChunk] = []
    folder_path = Path(folder)

    for file_path in sorted(folder_path.iterdir()):
        if file_path.suffix.lower() != ".md":
            continue

        text = file_path.read_text(encoding="utf-8")
        file_chunks = _split_by_version(text, file_path.name)
        chunks.extend(file_chunks)
        log.info("changelog_loaded", file=file_path.name, versions=len(file_chunks))

    return chunks


def _parse_version_and_date(header_line: str) -> tuple[str, str]:
    """
    Extract version and optional date from a header line.

    Examples:
      '## v2.1.0 — 2025-03-15'  →  ('2.1.0', '2025-03-15')
      '## 1.0.0'                 →  ('1.0.0', '')
    """
    version = ""
    date = ""

    version_match = re.search(r"v?(\d+\.\d+[\.\d]*)", header_line)
    if version_match:
        version = version_match.group(1)

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", header_line)
    if date_match:
        date = date_match.group(1)

    return version, date


def _split_by_version(text: str, source: str) -> list[ChangelogChunk]:
    """
    Split the changelog text into blocks, one per version header (## vX.Y.Z).

    Strategy:
      1. Find all version header positions.
      2. Slice the text between consecutive headers.
      3. Each slice becomes one ChangelogChunk.
    """
    matches = list(VERSION_HEADER_RE.finditer(text))

    if not matches:
        # No version headers found — treat the whole file as one chunk
        log.warning("no_version_headers_found", source=source)
        return [ChangelogChunk(
            id=f"{source}_0",
            content=text.strip(),
            source=source,
            source_type="changelog",
            chunk_index=0,
        )]

    chunks: list[ChangelogChunk] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        if not block:
            continue

        header_line = block.splitlines()[0]
        version, release_date = _parse_version_and_date(header_line)

        chunks.append(ChangelogChunk(
            id=f"{source}_v{version.replace('.', '_')}",
            content=block,
            source=source,
            source_type="changelog",
            version=version,
            release_date=release_date,
            chunk_index=i,
            metadata={"lines": len(block.splitlines())},
        ))

    return chunks
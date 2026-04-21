"""
metrics.py
----------
Lightweight local metric helpers — no Azure calls, no token spend.
Run these as a fast sanity check before the full Azure AI Evaluation run.

Three checks:
  has_citation      — does the answer mention a source file or version?
  answer_length_ok  — is the answer a sensible length?
  no_refusal        — did the model actually attempt to answer?

New vs Document Q&A App:
  has_version_citation — specifically checks changelog answers cite a version
  source_type_present  — checks the answer mentions the expected source type
"""

import re


def has_citation(answer: str, sources: list[str]) -> bool:
    """Answer mentions at least one source filename."""
    answer_lower = answer.lower()
    return any(src.lower() in answer_lower for src in sources)


def has_version_citation(answer: str) -> bool:
    """
    Answer contains a version string (e.g. v2.1.0 or 2.1.0).
    Use this for changelog questions — the version number must appear.
    """
    return bool(re.search(r"v?\d+\.\d+[\.\d]*", answer))


def answer_length_ok(answer: str, min_chars: int = 20, max_chars: int = 2000) -> bool:
    """Answer is within a sensible character range."""
    length = len(answer.strip())
    return min_chars <= length <= max_chars


def no_refusal(answer: str) -> bool:
    """Answer is not a refusal / fallback message."""
    refusal_patterns = [
        r"i (don't|do not|cannot|can't) (find|have|know)",
        r"not (enough|sufficient) information",
        r"the (documents?|context|knowledge base) (do(?:es)? not|don't) (contain|mention|include)",
        r"i('m| am) (unable|not able)",
        r"no information (available|found)",
        r"outside (the scope|my knowledge)",
    ]
    answer_lower = answer.lower()
    return not any(re.search(p, answer_lower) for p in refusal_patterns)


def source_type_present(answer: str, expected_source_type: str) -> bool:
    """
    Check whether the answer references the expected source type.
    The source-aware prompt labels context with [doc], [ticket], [changelog].
    GPT-4o often echoes these labels in citations.

    Args:
        answer: Generated answer text.
        expected_source_type: "doc" | "ticket" | "changelog"
    """
    return expected_source_type.lower() in answer.lower()


def compute_local_metrics(
    answer: str,
    sources: list[str],
    source_type: str | None = None,
) -> dict:
    """
    Run all local checks and return a summary dict.

    Args:
        answer:      Generated answer text.
        sources:     Source filenames used during retrieval.
        source_type: Expected source type for source_type_present check.

    Returns:
        Dict with boolean flags and char_count.
    """
    result = {
        "has_citation": has_citation(answer, sources),
        "has_version_citation": has_version_citation(answer),
        "answer_length_ok": answer_length_ok(answer),
        "no_refusal": no_refusal(answer),
        "char_count": len(answer.strip()),
    }
    if source_type:
        result["source_type_present"] = source_type_present(answer, source_type)
    return result
"""
test_deduplicator.py
--------------------
Unit tests for cosine similarity deduplication.
Uses real numpy vectors — no Azure calls needed.
"""

import math
import pytest
from src.ingestion.deduplicator import cosine_similarity, deduplicate


def _make_doc(id: str, vector: list[float]) -> dict:
    return {"id": id, "content": "text", "embedding": vector}


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors_score_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_similar_vectors_high_score(self):
        a = [1.0, 0.1]
        b = [1.0, 0.2]
        assert cosine_similarity(a, b) > 0.99


class TestDeduplicate:
    def test_empty_input_returns_empty(self):
        result, removed = deduplicate([])
        assert result == []
        assert removed == 0

    def test_unique_docs_all_kept(self):
        docs = [
            _make_doc("a", [1.0, 0.0, 0.0]),
            _make_doc("b", [0.0, 1.0, 0.0]),
            _make_doc("c", [0.0, 0.0, 1.0]),
        ]
        result, removed = deduplicate(docs, threshold=0.97)
        assert len(result) == 3
        assert removed == 0

    def test_duplicate_docs_removed(self):
        v = [1.0, 0.0, 0.0]
        docs = [
            _make_doc("a", v),
            _make_doc("b", v),   # identical — should be removed
        ]
        result, removed = deduplicate(docs, threshold=0.97)
        assert len(result) == 1
        assert removed == 1
        assert result[0]["id"] == "a"

    def test_near_duplicate_removed_at_high_threshold(self):
        docs = [
            _make_doc("a", [1.0, 0.0]),
            _make_doc("b", [1.0, 0.01]),  # very similar
        ]
        result, removed = deduplicate(docs, threshold=0.97)
        assert removed == 1

    def test_near_duplicate_kept_at_low_threshold(self):
        docs = [
            _make_doc("a", [1.0, 0.0]),
            _make_doc("b", [0.9, 0.1]),   # similar but not identical
        ]
        result, removed = deduplicate(docs, threshold=0.999)
        assert len(result) == 2
        assert removed == 0

    def test_doc_without_embedding_accepted(self):
        docs = [{"id": "x", "content": "text", "embedding": []}]
        result, removed = deduplicate(docs, threshold=0.97)
        assert len(result) == 1
        assert removed == 0

    def test_first_doc_always_kept(self):
        v = [1.0, 0.0]
        docs = [_make_doc("first", v), _make_doc("second", v)]
        result, _ = deduplicate(docs, threshold=0.97)
        assert result[0]["id"] == "first"
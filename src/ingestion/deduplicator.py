"""
deduplicator.py
---------------
Remove near-duplicate chunks before indexing using cosine similarity.

Why deduplicate?
Support tickets often contain boilerplate: "Thank you for contacting us",
"Please let me know if you need anything else", legal footers, etc.
Indexing hundreds of near-identical chunks:
  1. Wastes index storage
  2. Inflates retrieval results with redundant content
  3. Pushes unique, useful chunks lower in rankings

How it works:
  1. For each new chunk, compare its embedding to all already-accepted
     embeddings using cosine similarity.
  2. If similarity >= threshold (default 0.97), skip the chunk.
  3. Otherwise, accept it and add its embedding to the reference set.

This is an O(n²) algorithm — fine for thousands of chunks, slow for
millions. For very large corpora, use MinHash LSH instead.

AI-103 concept: Production RAG quality — deduplication is a real-world
concern that tutorial projects skip but production systems must handle.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns a value between -1 and 1.
    Values >= 0.97 indicate near-duplicate text.

    Args:
        a: First embedding vector.
        b: Second embedding vector.
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    if np.all(va == 0) or np.all(vb == 0):
        return 0.0
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def deduplicate(
    documents: list[dict],
    threshold: float = 0.97,
) -> tuple[list[dict], int]:
    """
    Filter out near-duplicate documents based on embedding cosine similarity.

    Args:
        documents: List of dicts with at least an 'embedding' key (list[float])
                   and an 'id' key. These are the output of embedder.embed_chunks().
        threshold: Cosine similarity threshold above which a chunk is considered
                   a duplicate (default 0.97 — very conservative).

    Returns:
        Tuple of (deduplicated_documents, num_removed).

    Example:
        docs, removed = deduplicate(embedded_docs, threshold=0.97)
        print(f"Removed {removed} duplicates, kept {len(docs)}")
    """
    if not documents:
        return [], 0

    accepted: list[dict] = []
    accepted_embeddings: list[list[float]] = []
    num_removed = 0

    for doc in documents:
        embedding = doc.get("embedding", [])
        if not embedding:
            # No embedding — accept without checking
            accepted.append(doc)
            continue

        is_duplicate = False
        for ref_embedding in accepted_embeddings:
            sim = cosine_similarity(embedding, ref_embedding)
            if sim >= threshold:
                is_duplicate = True
                log.debug(
                    "duplicate_skipped",
                    id=doc.get("id"),
                    similarity=round(sim, 4),
                )
                break

        if is_duplicate:
            num_removed += 1
        else:
            accepted.append(doc)
            accepted_embeddings.append(embedding)

    log.info(
        "deduplication_complete",
        total_input=len(documents),
        kept=len(accepted),
        removed=num_removed,
        threshold=threshold,
    )
    return accepted, num_removed
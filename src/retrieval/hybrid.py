"""
Hybrid Search: Combine semantic search (vectors) with keyword search (BM25).

Why hybrid?
- Semantic search (Day 3): understands meaning. "connect database" finds "init_db_connection"
  BUT it might miss exact matches like a specific variable name "db_config"
- BM25 keyword search: great at exact matching. Searching "db_config" finds all code with "db_config"
  BUT it doesn't understand that "connect database" and "init_db_connection" mean the same thing

Hybrid = best of both worlds. We run both searches, then combine the results.

How do we combine them?
- We use Reciprocal Rank Fusion (RRF)
- Each search produces a ranked list of results
- RRF gives higher scores to chunks that appear in BOTH lists
- A chunk ranked #1 in both searches gets a very high combined score
- A chunk ranked #1 in one but absent from the other still gets a decent score
"""

import re
from rank_bm25 import BM25Okapi

from src.embedding.store import search as vector_search
from src.config import TOP_K


# ---- BM25 Index ----

# We store the BM25 index and chunk data in memory
# In production you'd persist this, but for our project this is fine
_bm25_index = None
_bm25_chunks = None


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase, split by non-alphanumeric characters.

    "def connect_database():" → ["def", "connect", "database"]

    For code, this works well because:
    - snake_case gets split: "connect_database" → ["connect", "database"]
    - camelCase stays as one token (could improve later)
    - Punctuation is removed
    """
    return re.findall(r'\w+', text.lower())


def build_bm25_index(chunks: list) -> None:
    """
    Build a BM25 index from code chunks.

    This should be called after chunking (Day 2), before searching.
    The index lets us do fast keyword search over all chunks.

    Args:
        chunks: List of CodeChunk objects or dicts with "content" and metadata
    """
    global _bm25_index, _bm25_chunks

    # Store chunks for later retrieval
    _bm25_chunks = []
    tokenized_docs = []

    for chunk in chunks:
        # Support both CodeChunk objects and dicts
        if hasattr(chunk, 'content'):
            content = chunk.content
            metadata = chunk.to_dict()
        else:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", chunk)

        # Combine code + metadata for better keyword matching
        search_text = f"{metadata.get('name', '')} {metadata.get('file_path', '')} {content}"

        tokenized_docs.append(_tokenize(search_text))
        _bm25_chunks.append({
            "content": content,
            "metadata": metadata,
        })

    _bm25_index = BM25Okapi(tokenized_docs)
    print(f"Built BM25 index with {len(tokenized_docs)} documents")


def bm25_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Search using BM25 keyword matching.

    Args:
        query: User's question
        top_k: Number of results to return

    Returns:
        List of dicts with content, metadata, and BM25 score
    """
    if _bm25_index is None:
        raise RuntimeError("BM25 index not built. Call build_bm25_index() first.")

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-k indices sorted by score (highest first)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": _bm25_chunks[idx]["content"],
            "metadata": _bm25_chunks[idx]["metadata"],
            "bm25_score": float(scores[idx]),
        })

    return results


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion (RRF).

    The idea: a result at rank r gets a score of 1/(k+r).
    - Rank 1 → score = 1/61 = 0.0164
    - Rank 2 → score = 1/62 = 0.0161
    - Rank 5 → score = 1/65 = 0.0154

    If a chunk appears in BOTH lists, its scores are added together,
    so it gets a higher combined score.

    Args:
        vector_results: Results from semantic search
        bm25_results: Results from BM25 keyword search
        k: Constant to prevent high scores for top-ranked items (default 60)

    Returns:
        Combined results sorted by RRF score (highest first)
    """
    scores = {}  # id → {"score": float, "data": dict}

    # Score vector search results
    for rank, result in enumerate(vector_results, start=1):
        doc_id = result.get("id") or f"{result['metadata']['file_path']}::{result['metadata']['name']}"
        rrf_score = 1.0 / (k + rank)

        if doc_id not in scores:
            scores[doc_id] = {"score": 0, "data": result, "sources": []}
        scores[doc_id]["score"] += rrf_score
        scores[doc_id]["sources"].append("semantic")

    # Score BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        meta = result["metadata"]
        doc_id = f"{meta.get('file_path', '')}::{meta.get('name', '')}"
        rrf_score = 1.0 / (k + rank)

        if doc_id not in scores:
            scores[doc_id] = {"score": 0, "data": result, "sources": []}
        scores[doc_id]["score"] += rrf_score
        scores[doc_id]["sources"].append("bm25")

    # Sort by combined score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    # Format results
    results = []
    for item in ranked:
        data = item["data"]
        results.append({
            "id": f"{data['metadata'].get('file_path', '')}::{data['metadata'].get('name', '')}",
            "content": data.get("content", ""),
            "metadata": data["metadata"],
            "rrf_score": item["score"],
            "found_by": item["sources"],  # ["semantic"], ["bm25"], or ["semantic", "bm25"]
            "distance": data.get("distance"),
            "bm25_score": data.get("bm25_score"),
        })

    return results


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Main hybrid search function: runs both semantic and BM25, then fuses results.

    Args:
        query: User's question
        top_k: Number of final results to return

    Returns:
        Combined and re-ranked results
    """
    # Get more results from each search to have a better pool for fusion
    fetch_k = top_k * 2

    # Run both searches
    vector_results = vector_search(query, top_k=fetch_k)
    bm25_results = bm25_search(query, top_k=fetch_k)

    # Combine with RRF
    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    # Return top_k results
    return fused[:top_k]

"""
Evaluation Script: Measure and compare search quality.

Why evaluate?
- Without measurement, we're just guessing if our changes actually improve things
- This script lets us compare different search methods side by side:
  1. Vector search only (Day 3)
  2. Hybrid search (Day 6)
  3. Hybrid + Reranking (Day 7)

How it works:
- We define test queries with "expected" results (what we know should be found)
- Run each query through each search method
- Check if the expected results appear, and at what rank
- Calculate metrics like "hit rate" and "Mean Reciprocal Rank (MRR)"

Metrics explained:
- Hit Rate: Did the expected result appear at all? (0 or 1)
- MRR (Mean Reciprocal Rank): Where did it appear?
  - If rank 1: MRR = 1/1 = 1.0 (perfect!)
  - If rank 2: MRR = 1/2 = 0.5
  - If rank 3: MRR = 1/3 = 0.33
  - If not found: MRR = 0

Usage:
    cd rag-repo-copilot
    python -m scripts.evaluate
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.store import search as vector_search
from src.retrieval.hybrid import hybrid_search, build_bm25_index, bm25_search
from src.retrieval.reranker import rerank
from src.ingestion.loader import clone_repo, collect_python_files
from src.ingestion.chunker import chunk_repo
from src.embedding.embedder import embed_chunks
from src.embedding.store import store_embeddings, get_collection


# ---- Test Cases ----
# Each test case has:
# - query: what the user would ask
# - expected_names: function/class names that SHOULD appear in results
# - description: what we're testing

TEST_CASES = [
    {
        "query": "How does the app handle incoming web requests?",
        "expected_names": ["core_module"],
        "description": "Natural language → semantic understanding needed",
    },
    {
        "query": "test_digest_auth",
        "expected_names": ["test_digest_auth"],
        "description": "Exact function name → keyword matching needed",
    },
    {
        "query": "How are cookies set and managed?",
        "expected_names": ["set_cookie", "view_cookies"],
        "description": "Mixed query → both semantic and keyword help",
    },
    {
        "query": "What HTTP methods does the API support?",
        "expected_names": ["view_get", "view_post"],
        "description": "Conceptual question about HTTP verbs",
    },
    {
        "query": "redirect",
        "expected_names": ["redirect_to", "relative_redirect_n_times"],
        "description": "Single keyword → BM25 should excel",
    },
]


def find_rank(results: list[dict], expected_name: str) -> int:
    """
    Find the rank of an expected result in the search results.

    Returns:
        rank (1-indexed) if found, 0 if not found
    """
    for i, result in enumerate(results):
        name = result.get("metadata", {}).get("name", "")
        if expected_name.lower() in name.lower():
            return i + 1
    return 0


def evaluate_search_method(method_name: str, search_fn, test_cases: list, top_k: int = 5) -> dict:
    """
    Evaluate a search method on all test cases.

    Args:
        method_name: Display name for the method
        search_fn: Function that takes (query, top_k) and returns results
        test_cases: List of test case dicts
        top_k: Number of results to retrieve

    Returns:
        Dict with metrics: hit_rate, mrr, details
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*60}")

    total_hits = 0
    total_rr = 0  # Sum of reciprocal ranks
    total_expected = 0
    details = []

    for tc in test_cases:
        query = tc["query"]
        expected_names = tc["expected_names"]

        # Run search
        start = time.time()
        results = search_fn(query, top_k)
        elapsed = time.time() - start

        # Check each expected result
        tc_hits = 0
        tc_rr = 0

        for name in expected_names:
            rank = find_rank(results, name)
            total_expected += 1

            if rank > 0:
                tc_hits += 1
                total_hits += 1
                rr = 1.0 / rank
                tc_rr += rr
                total_rr += rr
                status = f"✓ rank {rank}"
            else:
                status = "✗ not found"

            print(f"  Query: \"{query[:40]}...\" → {name}: {status}")

        details.append({
            "query": query,
            "description": tc["description"],
            "hits": tc_hits,
            "expected": len(expected_names),
            "time_ms": round(elapsed * 1000),
        })

    hit_rate = total_hits / total_expected if total_expected > 0 else 0
    mrr = total_rr / total_expected if total_expected > 0 else 0

    print(f"\n  Hit Rate: {hit_rate:.1%} ({total_hits}/{total_expected})")
    print(f"  MRR:      {mrr:.3f}")

    return {
        "method": method_name,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "total_hits": total_hits,
        "total_expected": total_expected,
        "details": details,
    }


def main():
    """Run the full evaluation."""
    print("=" * 60)
    print("RAG Repo Copilot — Search Quality Evaluation")
    print("=" * 60)

    # Check if data is already indexed
    collection = get_collection()
    count = collection.count()

    if count == 0:
        print("\nNo data in vector store. Indexing httpbin repo first...")
        repo_url = "https://github.com/postmanlabs/httpbin"
        repo_dir = clone_repo(repo_url)
        python_files = collect_python_files(repo_dir)
        chunks = chunk_repo(python_files)
        embedded = embed_chunks(chunks)
        store_embeddings(embedded)
        print(f"Indexed {len(chunks)} chunks")
    else:
        print(f"\nUsing existing index with {count} chunks")
        # We still need chunks for BM25
        chunks = None

    # Build BM25 index (needs the raw chunks)
    if chunks is None:
        # Rebuild chunks from stored data (without re-embedding)
        print("Rebuilding chunks for BM25 index...")
        repo_url = "https://github.com/postmanlabs/httpbin"
        repo_dir = clone_repo(repo_url)
        python_files = collect_python_files(repo_dir)
        chunks = chunk_repo(python_files)

    build_bm25_index(chunks)

    top_k = 5

    # ---- Method 1: Vector search only ----
    vector_results = evaluate_search_method(
        "Vector Search (semantic only)",
        lambda q, k: vector_search(q, top_k=k),
        TEST_CASES,
        top_k=top_k,
    )

    # ---- Method 2: BM25 only ----
    bm25_results = evaluate_search_method(
        "BM25 (keyword only)",
        lambda q, k: bm25_search(q, top_k=k),
        TEST_CASES,
        top_k=top_k,
    )

    # ---- Method 3: Hybrid search ----
    hybrid_results = evaluate_search_method(
        "Hybrid Search (vector + BM25 + RRF)",
        lambda q, k: hybrid_search(q, top_k=k),
        TEST_CASES,
        top_k=top_k,
    )

    # ---- Method 4: Hybrid + Reranking ----
    def hybrid_rerank(query, k):
        # Fetch more, then rerank down to k
        candidates = hybrid_search(query, top_k=k * 2)
        return rerank(query, candidates, top_k=k)

    hybrid_rerank_results = evaluate_search_method(
        "Hybrid + Reranking (full pipeline)",
        hybrid_rerank,
        TEST_CASES,
        top_k=top_k,
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<40} {'Hit Rate':>10} {'MRR':>8}")
    print("-" * 60)

    for result in [vector_results, bm25_results, hybrid_results, hybrid_rerank_results]:
        print(f"{result['method']:<40} {result['hit_rate']:>9.1%} {result['mrr']:>8.3f}")

    print("-" * 60)
    print("\nHigher is better for both metrics.")
    print("Hit Rate = did we find the expected result at all?")
    print("MRR = how high was it ranked? (1.0 = always rank 1)")


if __name__ == "__main__":
    main()

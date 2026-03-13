"""
Test Hybrid Search: compare semantic-only vs hybrid (semantic + BM25).

Prerequisites:
- Day 3 test must have run (Chroma has data)
- Valid OPENAI_API_KEY in .env

This test shows WHY hybrid is better:
- Some queries work better with semantic search (meaning-based)
- Some queries work better with BM25 (exact keyword matching)
- Hybrid combines both and usually beats either one alone
"""

from src.ingestion.loader import clone_repo, collect_python_files
from src.ingestion.chunker import chunk_repo
from src.embedding.embedder import embed_chunks
from src.embedding.store import search as vector_search, store_embeddings, clear_collection
from src.retrieval.hybrid import build_bm25_index, bm25_search, hybrid_search


def test_hybrid_vs_semantic():
    """
    Compare semantic search vs BM25 vs hybrid on the same queries.
    """
    # First, make sure we have data indexed
    print("\n" + "=" * 60)
    print("SETTING UP: Indexing httpbin repo")
    print("=" * 60)

    repo_url = "https://github.com/postmanlabs/httpbin"
    repo_dir = clone_repo(repo_url)
    python_files = collect_python_files(repo_dir)
    chunks = chunk_repo(python_files)

    # Embed and store in Chroma (for semantic search)
    embedded = embed_chunks(chunks)
    clear_collection()
    store_embeddings(embedded)

    # Build BM25 index (for keyword search)
    build_bm25_index(chunks)

    # Test queries — designed to show different strengths
    test_queries = [
        # This query uses natural language → semantic search should do well
        "How does the app handle incoming web requests?",

        # This query has exact keywords → BM25 should do well
        "test_digest_auth",

        # This query is a mix → hybrid should shine
        "How are cookies set and managed in the HTTP responses?",
    ]

    print("\n" + "=" * 60)
    print("COMPARISON: Semantic vs BM25 vs Hybrid")
    print("=" * 60)

    for query in test_queries:
        print(f"\n{'━' * 60}")
        print(f"Query: \"{query}\"")
        print(f"{'━' * 60}")

        # Semantic search
        semantic_results = vector_search(query, top_k=5)
        print(f"\n  📊 Semantic Search (vector similarity):")
        for i, r in enumerate(semantic_results[:3], 1):
            print(f"    {i}. [{r['metadata']['chunk_type']}] {r['metadata']['name']} (dist={r['distance']:.4f})")

        # BM25 search
        bm25_results = bm25_search(query, top_k=5)
        print(f"\n  🔤 BM25 Search (keyword matching):")
        for i, r in enumerate(bm25_results[:3], 1):
            print(f"    {i}. [{r['metadata']['chunk_type']}] {r['metadata']['name']} (score={r['bm25_score']:.4f})")

        # Hybrid search
        hybrid_results = hybrid_search(query, top_k=5)
        print(f"\n  🔀 Hybrid Search (combined):")
        for i, r in enumerate(hybrid_results[:3], 1):
            found_by = " + ".join(r["found_by"])
            print(f"    {i}. [{r['metadata']['chunk_type']}] {r['metadata']['name']} (rrf={r['rrf_score']:.4f}, found by: {found_by})")

    print("\n" + "=" * 60)
    print("HYBRID SEARCH TEST COMPLETE!")
    print("=" * 60)
    print("\nKey observations:")
    print("  - Semantic search excels at understanding meaning")
    print("  - BM25 excels at exact keyword/function name matching")
    print("  - Hybrid combines both → more robust results")
    print("  - Results found by BOTH methods get the highest RRF scores")


if __name__ == "__main__":
    test_hybrid_vs_semantic()

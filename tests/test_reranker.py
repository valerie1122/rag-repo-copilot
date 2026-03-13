"""
Test Day 7: Reranker + Full Pipeline Evaluation

This test verifies:
1. Reranker correctly scores and re-orders results
2. Updated qa_chain works with hybrid + rerank
3. Compares before/after reranking quality
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import clone_repo, collect_python_files
from src.ingestion.chunker import chunk_repo
from src.embedding.embedder import embed_chunks
from src.embedding.store import store_embeddings, get_collection
from src.retrieval.hybrid import build_bm25_index, hybrid_search
from src.retrieval.reranker import rerank, rerank_with_details
from src.retrieval.qa_chain import ask


def setup_data():
    """Ensure we have indexed data to search."""
    collection = get_collection()
    count = collection.count()

    repo_url = "https://github.com/postmanlabs/httpbin"

    if count == 0:
        print("Setting up: indexing httpbin repo...")
        repo_dir = clone_repo(repo_url)
        python_files = collect_python_files(repo_dir)
        chunks = chunk_repo(python_files)
        embedded = embed_chunks(chunks)
        store_embeddings(embedded)
        print(f"Indexed {len(chunks)} chunks")
    else:
        print(f"Using existing index with {count} chunks")
        repo_dir = clone_repo(repo_url)
        python_files = collect_python_files(repo_dir)
        chunks = chunk_repo(python_files)

    # Build BM25 index
    build_bm25_index(chunks)
    return chunks


def test_reranker():
    """Test that reranker scores and reorders results."""
    print("\n" + "=" * 50)
    print("TEST 1: Reranker scoring")
    print("=" * 50)

    query = "How does the app handle cookies?"

    # Get hybrid search results
    candidates = hybrid_search(query, top_k=8)
    print(f"\nBefore reranking (hybrid order):")
    for i, r in enumerate(candidates):
        name = r["metadata"].get("name", "unknown")
        rrf = r.get("rrf_score", 0)
        print(f"  {i+1}. {name} (rrf={rrf:.4f})")

    # Rerank with details
    details = rerank_with_details(query, candidates, top_k=5)

    print(f"\nAfter reranking (GPT relevance order):")
    for i, r in enumerate(details["results"]):
        name = r["metadata"].get("name", "unknown")
        score = r.get("relevance_score", 0)
        print(f"  {i+1}. {name} (relevance={score}/10)")

    # Verify reranker returned results
    assert len(details["results"]) > 0, "Reranker should return results"
    assert len(details["results"]) <= 5, "Should return at most top_k results"

    # Verify scores are present
    for r in details["results"]:
        assert "relevance_score" in r, "Each result should have a relevance_score"
        assert 0 <= r["relevance_score"] <= 10, "Score should be 0-10"

    print("\n✓ Reranker test passed!")


def test_full_pipeline():
    """Test the complete RAG pipeline with hybrid + rerank."""
    print("\n" + "=" * 50)
    print("TEST 2: Full pipeline (hybrid + rerank + GPT)")
    print("=" * 50)

    question = "How does the httpbin app handle redirect requests?"

    # Test with full pipeline
    result = ask(question, top_k=5, use_hybrid=True, use_rerank=True)

    print(f"\nQuestion: {question}")
    print(f"Search method: {result['search_method']}")
    print(f"\nAnswer (first 200 chars):")
    print(f"  {result['answer'][:200]}...")
    print(f"\nSources ({len(result['sources'])} chunks):")
    for s in result["sources"]:
        score_info = ""
        if "relevance_score" in s:
            score_info = f", relevance={s['relevance_score']}"
        if "rrf_score" in s:
            score_info += f", rrf={s['rrf_score']:.4f}"
        print(f"  - {s['name']} ({s['file_path']}{score_info})")

    # Verify
    assert result["answer"], "Should get an answer"
    assert len(result["sources"]) > 0, "Should have sources"
    assert result["search_method"] == "hybrid+rerank", "Should use hybrid+rerank"

    print("\n✓ Full pipeline test passed!")


def test_compare_methods():
    """Compare different search methods on the same query."""
    print("\n" + "=" * 50)
    print("TEST 3: Method comparison")
    print("=" * 50)

    query = "test_digest_auth"

    # Method 1: Vector only
    result_v = ask(query, top_k=3, use_hybrid=False, use_rerank=False)
    print(f"\nVector only → sources: {[s['name'] for s in result_v['sources']]}")

    # Method 2: Hybrid only
    result_h = ask(query, top_k=3, use_hybrid=True, use_rerank=False)
    print(f"Hybrid only → sources: {[s['name'] for s in result_h['sources']]}")

    # Method 3: Hybrid + Rerank
    result_hr = ask(query, top_k=3, use_hybrid=True, use_rerank=True)
    print(f"Hybrid+rerank → sources: {[s['name'] for s in result_hr['sources']]}")

    # Verify all methods return results
    assert len(result_v["sources"]) > 0
    assert len(result_h["sources"]) > 0
    assert len(result_hr["sources"]) > 0

    print("\n✓ Method comparison test passed!")


if __name__ == "__main__":
    print("Day 7 Tests: Reranker + Full Pipeline")
    print("=" * 50)

    # Setup
    chunks = setup_data()

    # Run tests
    test_reranker()
    test_full_pipeline()
    test_compare_methods()

    print("\n" + "=" * 50)
    print("ALL DAY 7 TESTS PASSED! ✓")
    print("=" * 50)

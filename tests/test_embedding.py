"""
Test the embedding pipeline: chunk code → embed → store → search.

This test uses a small repo (httpbin) and runs real OpenAI API calls,
so it requires a valid OPENAI_API_KEY in .env and costs a tiny amount (~$0.001).
"""

from src.ingestion.loader import clone_repo, collect_python_files
from src.ingestion.chunker import chunk_repo
from src.embedding.embedder import embed_chunks
from src.embedding.store import store_embeddings, search, clear_collection


def test_full_pipeline():
    """
    End-to-end test: clone → chunk → embed → store → search.
    """
    print("\n" + "=" * 60)
    print("FULL EMBEDDING PIPELINE TEST")
    print("=" * 60)

    # Step 1: Clone a small repo
    repo_url = "https://github.com/postmanlabs/httpbin"
    print(f"\n[Step 1] Cloning {repo_url}...")
    repo_dir = clone_repo(repo_url)

    # Step 2: Collect Python files
    print("\n[Step 2] Collecting Python files...")
    python_files = collect_python_files(repo_dir)

    # Step 3: Chunk the code
    print("\n[Step 3] Chunking code with AST...")
    chunks = chunk_repo(python_files)

    # Only use first 30 chunks to save API cost during testing
    test_chunks = chunks[:30]
    print(f"  Using {len(test_chunks)} chunks for testing (out of {len(chunks)} total)")

    # Step 4: Embed the chunks (this calls OpenAI API)
    print("\n[Step 4] Embedding chunks with OpenAI...")
    embedded = embed_chunks(test_chunks)
    print(f"  Got {len(embedded)} embeddings, each with {len(embedded[0]['embedding'])} dimensions")

    # Step 5: Store in Chroma
    print("\n[Step 5] Storing in ChromaDB...")
    clear_collection()  # Start fresh
    store_embeddings(embedded)

    # Step 6: Search!
    print("\n[Step 6] Testing search...")
    test_queries = [
        "How does the app handle HTTP requests?",
        "How are cookies managed?",
        "What test cases exist?",
    ]

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")
        results = search(query, top_k=3)
        for i, r in enumerate(results):
            print(f"    Result {i + 1}: [{r['metadata']['chunk_type']}] {r['metadata']['name']}")
            print(f"             File: {r['metadata']['file_path']}:{r['metadata']['start_line']}-{r['metadata']['end_line']}")
            print(f"             Distance: {r['distance']:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()

"""
Vector Store: Store and query embeddings using ChromaDB.

What is a vector store?
- A database optimized for storing vectors (lists of numbers)
- Its superpower: "given this vector, find the most similar ones"
- This is how we find relevant code when a user asks a question:
    1. Convert user's question to a vector
    2. Ask Chroma: "find the 5 closest vectors to this one"
    3. Those 5 vectors correspond to 5 code chunks → those are our search results

Why Chroma?
- Simple to use, runs locally (no server needed)
- Good for development and small/medium projects
- For production, you'd switch to Pinecone or Weaviate (cloud-hosted, scalable)
"""

import chromadb

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K
from src.embedding.embedder import embed_text


def get_collection() -> chromadb.Collection:
    """
    Get or create the Chroma collection.

    A "collection" in Chroma is like a "table" in a regular database.
    All our code chunk vectors go into one collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity for comparison
    )
    return collection


def store_embeddings(embedded_chunks: list[dict]) -> None:
    """
    Store embedded code chunks into Chroma.

    Args:
        embedded_chunks: Output from embedder.embed_chunks()
            Each dict has: id, embedding, content, metadata
    """
    collection = get_collection()

    # Chroma has a batch limit, so we add in batches
    batch_size = 100

    for i in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[i:i + batch_size]

        collection.add(
            ids=[item["id"] for item in batch],
            embeddings=[item["embedding"] for item in batch],
            documents=[item["content"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )

        print(f"  Stored batch {i // batch_size + 1}/{(len(embedded_chunks) - 1) // batch_size + 1}")

    print(f"Stored {len(embedded_chunks)} chunks in Chroma (collection: {CHROMA_COLLECTION_NAME})")


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Search for code chunks most relevant to a query.

    This is the core of RAG retrieval:
    1. Convert the question into a vector
    2. Find the closest vectors in the database
    3. Return the corresponding code chunks

    Args:
        query: User's question, e.g. "How does authentication work?"
        top_k: Number of results to return (default: 5)

    Returns:
        List of dicts, each containing:
        - content: the code chunk text
        - metadata: file_path, name, etc.
        - distance: how far the result is from the query (lower = more relevant)
    """
    collection = get_collection()

    # Convert the question to a vector using the same embedding model
    query_embedding = embed_text(query)

    # Ask Chroma to find the closest vectors
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # Format results into a cleaner structure
    search_results = []
    for i in range(len(results["ids"][0])):
        search_results.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results["distances"] else None,
        })

    return search_results


def clear_collection() -> None:
    """Delete all data in the collection. Useful for re-indexing."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"Cleared collection: {CHROMA_COLLECTION_NAME}")
    except ValueError:
        print(f"Collection {CHROMA_COLLECTION_NAME} doesn't exist, nothing to clear")

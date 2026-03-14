"""
Embedder: Convert code chunks into vectors using OpenAI's Embedding API.

What is embedding?
- Take a piece of text (code, in our case) and turn it into a list of numbers (a "vector")
- Similar code → similar vectors (close together in space)
- Different code → different vectors (far apart in space)
- This lets us do "semantic search": find code by meaning, not just keywords

Example:
  "def connect_database():" → [0.12, -0.45, 0.78, ...] (1536 numbers)
  "def init_db_connection():" → [0.11, -0.44, 0.79, ...] (very similar numbers!)
  "def send_email():" → [0.89, 0.23, -0.56, ...] (very different numbers)
"""

from openai import OpenAI

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
from src.ingestion.chunker import CodeChunk


# Create OpenAI client once, reuse for all requests
client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str) -> list[float]:
    """
    Convert a single piece of text into a vector.

    Args:
        text: Any text string (code, question, etc.)

    Returns:
        A list of 1536 floats — the embedding vector
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_chunks(chunks: list[CodeChunk]) -> list[dict]:
    """
    Convert a list of code chunks into vectors.

    For each chunk, we create a "document" that combines:
    - The code itself
    - Metadata (file path, function name, docstring)
    This gives the embedding model more context about what the code does.

    Args:
        chunks: List of CodeChunk objects from the chunker

    Returns:
        List of dicts, each containing:
        - id: unique identifier
        - embedding: the vector (list of floats)
        - content: the text that was embedded
        - metadata: file_path, name, chunk_type, etc.
    """
    results = []

    # Process in batches of 50 to avoid API rate limits
    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Build the text to embed for each chunk
        # We prepend metadata so the embedding captures context
        texts = []
        for chunk in batch:
            # Combine metadata + code for richer embeddings
            embed_text_str = (
                f"File: {chunk.file_path}\n"
                f"Type: {chunk.chunk_type}\n"
                f"Name: {chunk.name}\n"
            )
            if chunk.docstring:
                embed_text_str += f"Description: {chunk.docstring}\n"
            embed_text_str += f"\n{chunk.content}"

            texts.append(embed_text_str)

        # Call OpenAI API for the whole batch at once (faster than one by one)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        # Pair each embedding with its chunk
        for j, embedding_data in enumerate(response.data):
            chunk = batch[j]
            results.append({
                "id": f"{chunk.file_path}::{chunk.name}",
                "embedding": embedding_data.embedding,
                "content": texts[j],
                "metadata": {
                    "file_path": chunk.file_path,
                    "name": chunk.name,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "docstring": chunk.docstring,
                    "raw_code": chunk.content,
                },
            })

        print(f"  Embedded batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({len(batch)} chunks)")

    print(f"Embedded {len(results)} chunks total")
    return results

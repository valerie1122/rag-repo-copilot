"""
Test the ingestion pipeline: clone a repo → collect files → chunk by AST.

We use a small, well-known repo (FastAPI) as our test target.
"""

from src.ingestion.loader import clone_repo, collect_python_files
from src.ingestion.chunker import chunk_file, chunk_repo


def test_chunker_with_sample_code():
    """
    Test AST chunker with a sample Python file (no cloning needed).
    This is a fast test you can run anytime.
    """
    sample_code = '''
import os

DB_URL = "postgresql://localhost/mydb"

def connect_db():
    """Connect to the database."""
    return create_connection(DB_URL)

def get_user(user_id: int):
    """Fetch a user by ID."""
    conn = connect_db()
    return conn.query("SELECT * FROM users WHERE id = ?", user_id)

class UserService:
    """Service for user operations."""

    def __init__(self, db):
        self.db = db

    def create_user(self, name: str, email: str):
        """Create a new user."""
        return self.db.insert("users", {"name": name, "email": email})

    def delete_user(self, user_id: int):
        """Delete a user."""
        return self.db.delete("users", user_id)
'''

    chunks = chunk_file("src/services/user.py", sample_code)

    print("\n" + "=" * 60)
    print("CHUNKING RESULTS")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Name: {chunk.name}")
        print(f"  File: {chunk.file_path}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Docstring: {chunk.docstring[:50]}..." if len(chunk.docstring) > 50 else f"  Docstring: {chunk.docstring}")
        print(f"  Code preview: {chunk.content[:80]}...")
        print()

    # Verify we got the expected chunks
    chunk_names = [c.name for c in chunks]
    print(f"All chunk names: {chunk_names}")

    assert "connect_db" in chunk_names, "Should find connect_db function"
    assert "get_user" in chunk_names, "Should find get_user function"
    assert "UserService" in chunk_names, "Should find UserService class"
    assert "UserService.create_user" in chunk_names, "Should find create_user method"
    assert "UserService.delete_user" in chunk_names, "Should find delete_user method"

    print("\nAll assertions passed! Chunking works correctly.")


def test_with_real_repo():
    """
    Test the full pipeline with a real GitHub repo.
    This clones a small repo, so it takes a few seconds.
    """
    # Use a small repo for testing — httpbin is simple and small
    repo_url = "https://github.com/postmanlabs/httpbin"

    print("\n" + "=" * 60)
    print(f"FULL PIPELINE TEST: {repo_url}")
    print("=" * 60)

    # Step 1: Clone
    repo_dir = clone_repo(repo_url)

    # Step 2: Collect Python files
    python_files = collect_python_files(repo_dir)
    print(f"\nPython files found:")
    for f in python_files[:10]:  # Show first 10
        print(f"  {f['file_path']} ({len(f['content'])} chars)")

    # Step 3: Chunk all files
    chunks = chunk_repo(python_files)
    print(f"\nTotal chunks: {len(chunks)}")

    # Show some example chunks
    print("\nSample chunks:")
    for chunk in chunks[:8]:
        print(f"  [{chunk.chunk_type}] {chunk.name} ({chunk.file_path}:{chunk.start_line}-{chunk.end_line})")

    print("\nFull pipeline test passed!")


if __name__ == "__main__":
    # Run the fast test first
    test_chunker_with_sample_code()

    # Then run the full pipeline test
    test_with_real_repo()

"""
Repo Loader: Clone a GitHub repo and collect all Python files.

This is Step 1 of the RAG pipeline:
  User gives a repo URL → we clone it → find all .py files → return their paths and content
"""

import os
import shutil
from git import Repo

from src.config import REPOS_DIR


def clone_repo(repo_url: str, target_dir: str | None = None) -> str:
    """
    Clone a GitHub repo to local disk.

    Args:
        repo_url: GitHub repo URL, e.g. "https://github.com/tiangolo/fastapi"
        target_dir: Where to clone to. If None, auto-generates from repo name.

    Returns:
        Path to the cloned repo on disk.
    """
    # Extract repo name from URL: "https://github.com/user/repo-name" → "repo-name"
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    if target_dir is None:
        target_dir = os.path.join(REPOS_DIR, repo_name)

    # If already cloned, remove and re-clone (ensures fresh copy)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(REPOS_DIR, exist_ok=True)

    print(f"Cloning {repo_url} → {target_dir} ...")
    Repo.clone_from(repo_url, target_dir, depth=1)  # depth=1 = only latest commit, faster
    print(f"Done! Cloned to {target_dir}")

    return target_dir


def collect_python_files(repo_dir: str) -> list[dict]:
    """
    Walk through the cloned repo and collect all .py files.

    Args:
        repo_dir: Path to the cloned repo.

    Returns:
        List of dicts, each containing:
        - file_path: relative path within the repo (e.g. "src/api/main.py")
        - content: the full file content as a string
    """
    python_files = []

    for root, dirs, files in os.walk(repo_dir):
        # Skip hidden dirs, venv, __pycache__, .git, etc.
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in ("venv", ".venv", "env", "__pycache__", "node_modules", ".git")
        ]

        for filename in files:
            if not filename.endswith(".py"):
                continue

            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, repo_dir)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read
                continue

            # Skip empty files and __init__.py with no real content
            if len(content.strip()) < 10:
                continue

            python_files.append({
                "file_path": relative_path,
                "content": content,
            })

    print(f"Found {len(python_files)} Python files in {repo_dir}")
    return python_files

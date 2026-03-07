"""
Application configuration.
Loads settings from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Vector Store
CHROMA_PERSIST_DIR = "./chroma_data"
CHROMA_COLLECTION_NAME = "repo_chunks"

# Retrieval
TOP_K = 5

# LLM
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.1

# Repo storage
REPOS_DIR = "./repos"

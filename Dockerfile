# ============================================================
# Dockerfile for RAG Repo Copilot
# ============================================================
#
# What is Docker?
# - Docker packages your app + all its dependencies into a "container"
# - A container is like a lightweight virtual machine
# - Anyone can run your app with one command, without installing Python,
#   ChromaDB, or any other dependency manually
#
# Think of it like shipping a product:
# - Without Docker: "Here's my code. You need Python 3.11, these 15 packages,
#   this version of git..." → things break on different machines
# - With Docker: "Run `docker compose up` and it just works" → same everywhere
#
# How to use:
#   docker compose up --build
#   Then visit http://localhost:8000/docs
# ============================================================

# ---- Stage 1: Base image ----
# We start from an official Python image
# python:3.11-slim is a lightweight version (smaller download, faster build)
FROM python:3.11-slim

# ---- Stage 2: Install system dependencies ----
# git is needed for cloning repos (our app clones GitHub repos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# ---- Stage 3: Set up working directory ----
# All our code will live in /app inside the container
WORKDIR /app

# ---- Stage 4: Install Python dependencies ----
# We copy requirements.txt FIRST (before the rest of the code)
# Why? Docker caches each step. If requirements.txt hasn't changed,
# Docker reuses the cached packages → much faster rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 5: Copy application code ----
# Now copy the rest of our code
COPY . .

# ---- Stage 6: Create directories for data ----
# These directories store cloned repos and vector data
RUN mkdir -p /app/repos /app/chroma_data

# ---- Stage 7: Expose port ----
# Tell Docker that our app listens on port 8000
EXPOSE 8000

# ---- Stage 8: Start the app ----
# uvicorn is the ASGI server that runs our FastAPI app
# --host 0.0.0.0 means "accept connections from outside the container"
# (by default it only accepts localhost, which wouldn't work in Docker)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

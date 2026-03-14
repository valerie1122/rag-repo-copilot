"""
FastAPI application entry point.
Day 1: Hello world skeleton.
Day 5: Full API with /repos and /ask endpoints.
Day 8: Docker-ready + hybrid search + reranking integration.

This is the "restaurant front desk" — it receives requests from users
and routes them to the right code (the "kitchen").
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.loader import clone_repo, collect_python_files
from src.ingestion.chunker import chunk_repo
from src.embedding.embedder import embed_chunks
from src.embedding.store import store_embeddings, search, clear_collection
from src.retrieval.hybrid import build_bm25_index
from src.retrieval.qa_chain import ask


# ---- FastAPI app setup ----

app = FastAPI(
    title="RAG Repo Copilot",
    description="A code repository Q&A system powered by RAG. "
                "Submit any GitHub repo and ask questions about the code. "
                "Uses hybrid search (semantic + BM25) with LLM reranking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request/Response models ----
# Pydantic models define what data the API expects and returns.
# FastAPI uses these to:
#   1. Validate incoming data (reject bad requests automatically)
#   2. Generate API documentation (Swagger UI)
#   3. Provide type hints for your IDE

class RepoRequest(BaseModel):
    """What the user sends when submitting a repo."""
    repo_url: str  # e.g. "https://github.com/tiangolo/fastapi"

class AskRequest(BaseModel):
    """What the user sends when asking a question."""
    question: str       # e.g. "How does authentication work?"
    top_k: int = 5      # How many code chunks to retrieve (optional, default 5)
    use_hybrid: bool = True   # Use hybrid search? (default: yes)
    use_rerank: bool = True   # Use GPT reranking? (default: yes)

class RepoResponse(BaseModel):
    """What we return after ingesting a repo."""
    status: str
    repo_url: str
    files_found: int
    chunks_created: int
    chunks_embedded: int

class AskResponse(BaseModel):
    """What we return when answering a question."""
    question: str
    answer: str
    sources: list[dict]
    search_method: str


# ---- Endpoints ----

@app.get("/health")
async def health_check():
    """Health check endpoint. Used by Docker to verify the app is running."""
    return {"status": "healthy", "version": "0.8.0"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RAG Repo Copilot",
        "description": "Ask questions about any code repository",
        "version": "0.8.0",
        "features": [
            "AST-based code chunking",
            "OpenAI embeddings",
            "Hybrid search (semantic + BM25)",
            "LLM reranking",
            "GPT-4o answer generation",
        ],
        "usage": {
            "step_1": "POST /repos with a GitHub repo URL to ingest it",
            "step_2": "POST /ask with a question to get an answer",
        },
    }


@app.post("/repos", response_model=RepoResponse)
async def ingest_repo(request: RepoRequest):
    """
    Clone and ingest a GitHub repository.

    This runs the full ingestion pipeline:
    1. Clone the repo
    2. Collect all Python files
    3. Chunk code by function/class (AST)
    4. Embed all chunks with OpenAI
    5. Store vectors in ChromaDB
    6. Build BM25 index for hybrid search

    After this, you can ask questions with POST /ask.
    """
    try:
        # Step 1: Clone
        repo_dir = clone_repo(request.repo_url)

        # Step 2: Collect Python files
        python_files = collect_python_files(repo_dir)

        if not python_files:
            raise HTTPException(
                status_code=400,
                detail="No Python files found in this repo."
            )

        # Step 3: Chunk
        chunks = chunk_repo(python_files)

        # Step 4: Embed
        embedded = embed_chunks(chunks)

        # Step 5: Store (clear old data first so we start fresh)
        clear_collection()
        store_embeddings(embedded)

        # Step 6: Build BM25 index for hybrid search (Day 6)
        build_bm25_index(chunks)

        return RepoResponse(
            status="success",
            repo_url=request.repo_url,
            files_found=len(python_files),
            chunks_created=len(chunks),
            chunks_embedded=len(embedded),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about the ingested code repository.

    Make sure you've ingested a repo first with POST /repos.
    The system will:
    1. Search for relevant code (hybrid: semantic + BM25)
    2. Rerank results with GPT for better accuracy
    3. Send top results to GPT with your question
    4. Return an answer with source references

    Options:
    - use_hybrid: Set to false to use vector search only
    - use_rerank: Set to false to skip GPT reranking (faster but less accurate)
    """
    try:
        result = ask(
            question=request.question,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_rerank=request.use_rerank,
        )

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            search_method=result["search_method"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
FastAPI application entry point.
Day 1: Hello world skeleton.
Day 5: Full API with /repos and /ask endpoints.

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
from src.retrieval.qa_chain import ask


# ---- FastAPI app setup ----

app = FastAPI(
    title="RAG Repo Copilot",
    description="A code repository Q&A system powered by RAG. "
                "Submit any GitHub repo and ask questions about the code.",
    version="0.5.0",
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
    question: str  # e.g. "How does authentication work?"
    top_k: int = 5  # How many code chunks to retrieve (optional, default 5)

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


# ---- Endpoints ----

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.5.0"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RAG Repo Copilot",
        "description": "Ask questions about any code repository",
        "usage": {
            "step_1": "POST /repos with a GitHub repo URL to ingest it",
            "step_2": "POST /ask with a question to get an answer",
        },
        "endpoints": {
            "POST /repos": "Submit a repo URL for ingestion",
            "POST /ask": "Ask a question about the ingested repo",
            "GET /health": "Health check",
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
    1. Search for relevant code chunks
    2. Send them to GPT with your question
    3. Return an answer with source references
    """
    try:
        result = ask(request.question, top_k=request.top_k)

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

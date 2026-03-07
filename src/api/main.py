"""
FastAPI application entry point.
Day 1: Hello world skeleton.
Day 5: Will add /repos and /ask endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RAG Repo Copilot",
    description="A code repository Q&A system powered by RAG",
    version="0.1.0",
)

# CORS — allow frontend access later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RAG Repo Copilot",
        "description": "Ask questions about any code repository",
        "endpoints": {
            "POST /repos": "Submit a repo URL for ingestion (coming Day 5)",
            "POST /ask": "Ask a question about the ingested repo (coming Day 5)",
            "GET /health": "Health check",
        },
    }


# ---- Day 5: Uncomment and implement these ----
# from pydantic import BaseModel
#
# class RepoRequest(BaseModel):
#     repo_url: str
#
# class AskRequest(BaseModel):
#     question: str
#     repo_id: str | None = None
#
# @app.post("/repos")
# async def ingest_repo(request: RepoRequest):
#     """Clone and ingest a GitHub repository."""
#     pass
#
# @app.post("/ask")
# async def ask_question(request: AskRequest):
#     """Ask a question about the ingested code."""
#     pass

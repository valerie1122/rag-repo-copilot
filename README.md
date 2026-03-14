# RAG Repo Copilot

A **Retrieval-Augmented Generation** system for code repositories. Submit any GitHub repo URL and ask natural language questions — get accurate answers with specific code references.

Built with a **three-stage retrieval pipeline**: hybrid search (semantic + BM25), LLM reranking, and GPT-4o answer generation.

## Features

- **AST-based code chunking** — intelligently splits code by functions, classes, and methods (not arbitrary line breaks)
- **Hybrid search** — combines semantic understanding (OpenAI embeddings) with keyword matching (BM25) using Reciprocal Rank Fusion
- **LLM reranking** — GPT re-scores search results for higher relevance accuracy
- **RESTful API** — FastAPI with auto-generated Swagger docs
- **Docker-ready** — one-command deployment with `docker compose up`

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    RAG Repo Copilot                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  POST /repos (Ingestion Pipeline)                        │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐ │
│  │  Clone   │ → │  AST     │ → │  OpenAI  │ → │ Chroma│ │
│  │  Repo    │   │  Chunker │   │  Embed   │   │  + BM25│ │
│  └─────────┘   └──────────┘   └──────────┘   └───────┘ │
│                                                          │
│  POST /ask (Query Pipeline)                              │
│  ┌──────────────┐   ┌──────────┐   ┌─────────────────┐  │
│  │ Hybrid Search │ → │ Reranker │ → │ GPT-4o Answer   │  │
│  │ Vector + BM25 │   │ (GPT)    │   │ + Code Refs     │  │
│  └──────────────┘   └──────────┘   └─────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Local Development

```bash
# Clone the repo
git clone https://github.com/valerie1122/rag-repo-copilot.git
cd rag-repo-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run the server
uvicorn src.api.main:app --reload

# Visit http://localhost:8000/docs for Swagger UI
```

### Option 2: Docker

```bash
# Set your API key
export OPENAI_API_KEY=your_key_here

# Build and run
docker compose up --build

# Visit http://localhost:8000/docs
```

## Usage

### Step 1: Ingest a repository

```bash
curl -X POST http://localhost:8000/repos \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/postmanlabs/httpbin"}'
```

Response:
```json
{
  "status": "success",
  "repo_url": "https://github.com/postmanlabs/httpbin",
  "files_found": 8,
  "chunks_created": 174,
  "chunks_embedded": 174
}
```

### Step 2: Ask questions

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the app handle redirect requests?"}'
```

Response:
```json
{
  "question": "How does the app handle redirect requests?",
  "answer": "The httpbin app handles redirects by decrementing the redirect count...",
  "sources": [
    {
      "file_path": "httpbin/core.py",
      "name": "_redirect",
      "relevance_score": 10
    }
  ],
  "search_method": "hybrid+rerank"
}
```

### Search Options

The `/ask` endpoint supports different search strategies:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `question` | required | Your question about the code |
| `top_k` | 5 | Number of code chunks to retrieve |
| `use_hybrid` | true | Use hybrid search (semantic + BM25) |
| `use_rerank` | true | Use GPT reranking for better accuracy |

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | Async API with auto-docs |
| LLM | GPT-4o | Answer generation + reranking |
| Embeddings | text-embedding-3-small | 1536-dim code embeddings |
| Vector Store | ChromaDB | Cosine similarity search |
| Keyword Search | BM25 (rank-bm25) | Term-frequency matching |
| Code Parsing | Python AST | Function/class-level chunking |
| Containerization | Docker | One-command deployment |

## Project Structure

```
rag-repo-copilot/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI endpoints
│   ├── ingestion/
│   │   ├── loader.py            # Git clone + file collection
│   │   └── chunker.py           # AST-based code chunking
│   ├── embedding/
│   │   ├── embedder.py          # OpenAI embedding API
│   │   └── store.py             # ChromaDB vector store
│   ├── retrieval/
│   │   ├── prompts.py           # Prompt templates
│   │   ├── qa_chain.py          # RAG pipeline orchestrator
│   │   ├── hybrid.py            # BM25 + semantic + RRF fusion
│   │   └── reranker.py          # LLM-based result reranking
│   └── config.py                # Configuration
├── scripts/
│   └── evaluate.py              # Search quality evaluation
├── tests/
│   ├── test_ingestion.py        # Chunker tests
│   ├── test_embedding.py        # Embedding + storage tests
│   ├── test_qa.py               # QA pipeline tests
│   ├── test_hybrid.py           # Hybrid search tests
│   └── test_reranker.py         # Reranker + full pipeline tests
├── Dockerfile                   # Container build instructions
├── docker-compose.yml           # Container orchestration
├── requirements.txt             # Python dependencies
└── .env.example                 # Environment variable template
```

## How It Works

### Ingestion Pipeline (POST /repos)

1. **Clone** — shallow clone (`depth=1`) of the GitHub repo
2. **Collect** — find all `.py` files, skip `venv/`, `__pycache__/`, etc.
3. **Chunk** — use Python's AST module to split code into functions, classes, and methods. Each chunk includes metadata (file path, name, line numbers, docstring)
4. **Embed** — convert each chunk to a 1536-dim vector using OpenAI's `text-embedding-3-small`
5. **Store** — save vectors to ChromaDB + build BM25 index

### Query Pipeline (POST /ask)

1. **Hybrid Search** — run both semantic search (ChromaDB cosine similarity) and BM25 keyword search in parallel
2. **RRF Fusion** — combine results using Reciprocal Rank Fusion: `score = 1/(k + rank)`. Chunks found by both methods get the highest scores
3. **Reranking** — GPT evaluates each candidate's relevance (0-10 score) and re-sorts
4. **Generation** — top results + question are sent to GPT-4o, which generates an answer with specific code references

## Evaluation

Run the evaluation script to compare search methods:

```bash
python -m scripts.evaluate
```

This tests 5 query types across 4 methods (vector-only, BM25-only, hybrid, hybrid+rerank) and reports Hit Rate and MRR (Mean Reciprocal Rank).

## Roadmap

- [x] Day 1: Environment + project skeleton
- [x] Day 2: Repo ingestion + AST-based code chunking
- [x] Day 3: Embedding + vector store (ChromaDB)
- [x] Day 4: LLM answer generation (GPT-4o)
- [x] Day 5: FastAPI endpoints + end-to-end integration
- [x] Day 6: Hybrid search (semantic + BM25 + RRF)
- [x] Day 7: LLM reranking + evaluation framework
- [x] Day 8: Docker containerization
- [x] Day 9: Documentation + architecture diagram
- [ ] Day 10: Final polish + resume preparation

## License

MIT

# RAG Repo Copilot

A code repository Q&A system powered by RAG (Retrieval-Augmented Generation). Ask natural language questions about any GitHub repository and get answers with specific code references.

> **Status:** Day 1 — Project skeleton + FastAPI hello world

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-repo-copilot.git
cd rag-repo-copilot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Run the server
uvicorn src.api.main:app --reload

# 6. Visit http://localhost:8000
```

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python | AI ecosystem standard |
| Web Framework | FastAPI | Async, auto-docs, type hints |
| LLM | OpenAI GPT-4o / Claude | Compare both in interviews |
| Embedding | text-embedding-3-small | Best cost/performance for code |
| Vector DB | Chroma → Pinecone | Local dev → production |
| RAG Framework | LangChain | Industry standard |

## Architecture

```
User Question
     ↓
  FastAPI
     ↓
  LangChain
     ↓
┌─────────────────┐
│  Embedding       │ → Chroma (Vector Store)
│  + Retrieval     │ → BM25 (Keyword Search)
└─────────────────┘
     ↓
  LLM (GPT-4o)
     ↓
  Answer + Code References
```

## Roadmap

- [x] Day 1: Environment + project skeleton
- [ ] Day 2: Repo ingestion + AST-based code chunking
- [ ] Day 3: Embedding + vector store
- [ ] Day 4: LLM answer generation
- [ ] Day 5: FastAPI endpoints + end-to-end integration
- [ ] Day 6: Hybrid search (semantic + BM25)
- [ ] Day 7: Reranking + quality evaluation
- [ ] Day 8: Docker + deployment
- [ ] Day 9: README + architecture docs
- [ ] Day 10: Final polish + resume update

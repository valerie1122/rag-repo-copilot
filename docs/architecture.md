# Architecture

## System Overview

```mermaid
flowchart TB
    subgraph User["👤 User"]
        Q[Ask a Question]
        R[Submit Repo URL]
    end

    subgraph API["FastAPI (src/api/main.py)"]
        EP1["POST /repos"]
        EP2["POST /ask"]
        EP3["GET /health"]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        L["loader.py<br/>Git Clone + Collect .py"]
        C["chunker.py<br/>AST Parsing"]
    end

    subgraph Embedding["Embedding"]
        E["embedder.py<br/>OpenAI text-embedding-3-small"]
        S["store.py<br/>ChromaDB"]
    end

    subgraph Search["Search (3-Stage Pipeline)"]
        direction TB
        VS["Semantic Search<br/>(ChromaDB cosine)"]
        BM["BM25 Search<br/>(rank-bm25)"]
        RRF["RRF Fusion<br/>(hybrid.py)"]
        RR["LLM Reranker<br/>(reranker.py)"]

        VS --> RRF
        BM --> RRF
        RRF --> RR
    end

    subgraph Generation["Answer Generation"]
        P["prompts.py<br/>Build Context"]
        LLM["GPT-4o<br/>(qa_chain.py)"]
    end

    R --> EP1
    EP1 --> L --> C --> E --> S

    Q --> EP2
    EP2 --> Search
    S -.->|vectors| VS
    RR --> P --> LLM
    LLM -->|Answer + Sources| User
```

## Data Flow: Ingestion

```mermaid
flowchart LR
    A["GitHub Repo URL"] --> B["git clone<br/>(depth=1)"]
    B --> C["Collect .py files<br/>(skip venv, __pycache__)"]
    C --> D["AST Parse<br/>(functions, classes, methods)"]
    D --> E["174 Code Chunks<br/>(with metadata)"]
    E --> F["OpenAI Embed<br/>(batch of 50)"]
    F --> G["ChromaDB<br/>(cosine similarity)"]
    E --> H["BM25 Index<br/>(tokenized)"]
```

## Data Flow: Query

```mermaid
flowchart LR
    Q["User Question"] --> S1["Semantic Search<br/>top_k × 2 results"]
    Q --> S2["BM25 Search<br/>top_k × 2 results"]
    S1 --> RRF["RRF Fusion<br/>score = 1/(60 + rank)"]
    S2 --> RRF
    RRF --> RR["GPT Reranker<br/>score 0-10"]
    RR --> TOP["Top K Results"]
    TOP --> P["Build Prompt<br/>(code + question)"]
    P --> GPT["GPT-4o<br/>temperature=0.1"]
    GPT --> A["Answer with<br/>Code References"]
```

## Component Details

| Component | File | Purpose | Key Tech |
|-----------|------|---------|----------|
| API Gateway | `src/api/main.py` | HTTP endpoints, request validation | FastAPI, Pydantic |
| Repo Loader | `src/ingestion/loader.py` | Clone repos, collect Python files | GitPython |
| Code Chunker | `src/ingestion/chunker.py` | Split code into semantic units | Python AST |
| Embedder | `src/embedding/embedder.py` | Convert text → vectors | OpenAI API |
| Vector Store | `src/embedding/store.py` | Store & search vectors | ChromaDB |
| Hybrid Search | `src/retrieval/hybrid.py` | BM25 + semantic fusion | rank-bm25, RRF |
| Reranker | `src/retrieval/reranker.py` | LLM-based re-scoring | GPT-4o |
| QA Chain | `src/retrieval/qa_chain.py` | Pipeline orchestration | OpenAI API |
| Prompts | `src/retrieval/prompts.py` | Prompt engineering | Template strings |
| Config | `src/config.py` | Centralized settings | python-dotenv |

## Design Decisions

### Why AST chunking instead of fixed-size chunks?

Fixed-size chunks (e.g., 500 tokens) often split functions in half, losing context. AST-based chunking ensures each chunk is a complete function, class, or method — preserving the semantic boundary of the code.

### Why hybrid search instead of just vector search?

Vector search understands meaning ("connect database" → finds "init_db_connection") but misses exact matches. BM25 excels at exact matching ("test_digest_auth" → finds that exact function). Combining both with RRF gives the best of both worlds.

### Why LLM reranking?

RRF only considers rank positions, not actual content. The LLM reranker reads both the question and each code snippet, then scores relevance on a 0-10 scale. This catches cases where a high-ranked result is actually irrelevant.

### Why ChromaDB?

Simple, runs locally, no server needed. Perfect for development and portfolio projects. For production, you'd switch to Pinecone or Weaviate for cloud hosting and scalability.

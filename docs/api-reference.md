# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

## Endpoints

### GET /health

Health check. Returns status and version.

**Response:**
```json
{"status": "healthy", "version": "0.8.0"}
```

### GET /

API information and usage guide.

### POST /repos

Ingest a GitHub repository into the system.

**Request Body:**
```json
{
  "repo_url": "https://github.com/postmanlabs/httpbin"
}
```

**Response (200):**
```json
{
  "status": "success",
  "repo_url": "https://github.com/postmanlabs/httpbin",
  "files_found": 8,
  "chunks_created": 174,
  "chunks_embedded": 174
}
```

**Errors:**
- `400`: No Python files found in the repo
- `500`: Clone failed or embedding error

### POST /ask

Ask a question about the ingested repository.

**Request Body:**
```json
{
  "question": "How does authentication work?",
  "top_k": 5,
  "use_hybrid": true,
  "use_rerank": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| question | string | required | Natural language question |
| top_k | int | 5 | Number of code chunks to retrieve |
| use_hybrid | bool | true | Enable hybrid search (semantic + BM25) |
| use_rerank | bool | true | Enable GPT reranking |

**Response (200):**
```json
{
  "question": "How does authentication work?",
  "answer": "The authentication system uses...",
  "sources": [
    {
      "file_path": "httpbin/core.py",
      "name": "digest_auth",
      "chunk_type": "function",
      "start_line": 45,
      "end_line": 78,
      "relevance_score": 9,
      "rrf_score": 0.0328
    }
  ],
  "search_method": "hybrid+rerank"
}
```

**Search Methods:**
- `vector`: semantic search only
- `hybrid`: semantic + BM25 + RRF fusion
- `hybrid+rerank`: full pipeline with LLM reranking

**Errors:**
- `500`: No repo ingested yet, or LLM error

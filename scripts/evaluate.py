"""
Evaluation script: 3-mode ablation on a hand-labeled query set.

Modes compared:
  A) dense              — semantic vector search only (cosine similarity)
  B) hybrid             — dense + BM25 fused with Reciprocal Rank Fusion (RRF)
  C) hybrid_rerank      — hybrid (top-2K candidates) → GPT-4o LLM reranker → top-K

Metrics:
  Hit Rate@K  — fraction of queries where ANY relevant chunk is in top-K results
  MRR         — mean of 1 / (rank of first relevant result), 0 if not found in top-K

The query set lives in eval/queries_<repo>.json; each query lists its relevant
chunks by '{file_path}::{name}' (the same id format the chunker emits).

This eval is self-contained: it embeds the repo into a local .npz cache,
then does cosine search in numpy. No ChromaDB/HNSW required — exact cosine
matches what Chroma's HNSW approximates for ~thousands of chunks. The result
of the eval reflects the same retrieval quality the user gets from production.

Usage:
    # First time (does indexing + embedding + eval):
    python -m scripts.evaluate --repo-path repos/fastapi/fastapi --queries eval/queries_fastapi.json

    # Re-runs use the cached embeddings (~30s instead of ~5min):
    python -m scripts.evaluate --queries eval/queries_fastapi.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make src.* importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import collect_python_files
from src.ingestion.chunker import chunk_repo
from src.embedding.embedder import client as openai_client, embed_text
from src.config import EMBEDDING_MODEL
from src.retrieval.hybrid import build_bm25_index, bm25_search, reciprocal_rank_fusion
from src.retrieval.reranker import rerank


# -----------------------------------------------------------------------------
# Embedding cache (numpy-backed, swaps in for ChromaDB during eval)
# -----------------------------------------------------------------------------

def build_or_load_index(repo_path: str, cache_path: str) -> tuple[list, np.ndarray, list[str]]:
    """
    Returns (chunks, embeddings_matrix, chunk_ids).

    First run: chunks the repo, embeds via OpenAI, caches to .npz.
    Later runs: loads from cache instantly.
    """
    chunks_meta_path = cache_path.replace(".npz", "_chunks.json")

    if os.path.exists(cache_path) and os.path.exists(chunks_meta_path):
        print(f"Loading cached index from {cache_path}")
        data = np.load(cache_path, allow_pickle=False)
        embeddings = data["embeddings"]
        with open(chunks_meta_path) as f:
            chunks_data = json.load(f)
        # Re-build minimal CodeChunk-like dicts for BM25
        chunks = chunks_data["chunks"]
        chunk_ids = [c["id"] for c in chunks]
        print(f"Loaded {len(chunks)} cached chunks ({embeddings.shape[1]}-dim embeddings)")
        return chunks, embeddings, chunk_ids

    # ---- Index from scratch ----
    print(f"Indexing repo at {repo_path} (this calls OpenAI embeddings — ~3-5 min for ~400 chunks)")
    files = collect_python_files(repo_path)
    code_chunks = chunk_repo(files)

    # Convert CodeChunks → dicts (this is what we'll cache + use everywhere downstream)
    chunks = []
    for c in code_chunks:
        chunk_id = f"{c.file_path}::{c.name}"
        chunks.append({
            "id": chunk_id,
            "content": c.content,
            "metadata": c.to_dict() | {"id": chunk_id},
        })

    # Build the text we send to the embedder (mirrors src/embedding/embedder.py).
    # text-embedding-3-small has an 8192-token input cap; some chunks (e.g. the
    # whole FastAPI class spanning 4600 lines) bust it. Truncate by character —
    # ~3.5 chars/token for code → 24000 chars is a safe ceiling. The truncated
    # tail rarely matters for retrieval since the signature, docstring, and
    # opening lines (which we keep) carry most of the semantic signal.
    MAX_EMBED_CHARS = 24000
    n_truncated = 0
    texts = []
    for c in chunks:
        m = c["metadata"]
        s = f"File: {m['file_path']}\nType: {m['chunk_type']}\nName: {m['name']}\n"
        if m.get("docstring"):
            s += f"Description: {m['docstring']}\n"
        s += f"\n{c['content']}"
        if len(s) > MAX_EMBED_CHARS:
            s = s[:MAX_EMBED_CHARS] + "\n... (truncated for embedding)"
            n_truncated += 1
        texts.append(s)
    if n_truncated:
        print(f"  (truncated {n_truncated}/{len(texts)} oversized chunks before embedding)")

    # Batch-embed
    embeddings = []
    batch = 50
    for i in range(0, len(texts), batch):
        chunk_batch = texts[i:i + batch]
        resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=chunk_batch)
        embeddings.extend([d.embedding for d in resp.data])
        print(f"  Embedded {min(i + batch, len(texts))}/{len(texts)}")

    embeddings = np.array(embeddings, dtype=np.float32)
    # L2-normalize so dot product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    # Persist
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    with open(chunks_meta_path, "w") as f:
        json.dump({"chunks": chunks}, f)
    print(f"Cached index to {cache_path} and {chunks_meta_path}")

    return chunks, embeddings, [c["id"] for c in chunks]


# -----------------------------------------------------------------------------
# Retrieval modes
# -----------------------------------------------------------------------------

def dense_search(query: str, embeddings: np.ndarray, chunks: list, top_k: int) -> list[dict]:
    """Pure semantic search via cosine similarity over the embedding matrix."""
    q_vec = np.array(embed_text(query), dtype=np.float32)
    q_vec /= max(np.linalg.norm(q_vec), 1e-12)
    sims = embeddings @ q_vec  # cosine since both sides normalized
    idxs = np.argsort(-sims)[:top_k]
    return [
        {
            "id": chunks[i]["id"],
            "content": chunks[i]["content"],
            "metadata": chunks[i]["metadata"],
            "score": float(sims[i]),
        }
        for i in idxs
    ]


def hybrid_search_local(query: str, embeddings: np.ndarray, chunks: list, top_k: int) -> list[dict]:
    """Dense + BM25 fused with RRF. Mirrors src/retrieval/hybrid.py but uses our local dense search."""
    fetch_k = top_k * 2
    dense = dense_search(query, embeddings, chunks, fetch_k)
    bm25 = bm25_search(query, top_k=fetch_k)
    fused = reciprocal_rank_fusion(dense, bm25)
    return fused[:top_k]


def hybrid_rerank_search(query: str, embeddings: np.ndarray, chunks: list, top_k: int) -> list[dict]:
    """Hybrid → grab 2K candidates → LLM rerank → top-K."""
    candidates = hybrid_search_local(query, embeddings, chunks, top_k * 2)
    return rerank(query, candidates, top_k=top_k)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def evaluate_mode(name: str, search_fn, queries: list[dict], k: int) -> dict:
    """Compute Hit Rate@k and MRR over the query set for one retrieval mode."""
    print(f"\n{'='*70}\nEvaluating: {name}\n{'='*70}")
    hits = 0
    rr_sum = 0.0
    per_query = []
    t0 = time.time()

    for q in queries:
        relevant = set(q["relevant_chunks"])
        results = search_fn(q["query"], k)
        retrieved_ids = [r["id"] for r in results]

        hit = any(rid in relevant for rid in retrieved_ids[:k])
        first_rank = next((i + 1 for i, rid in enumerate(retrieved_ids) if rid in relevant), None)
        rr = 1.0 / first_rank if first_rank else 0.0

        hits += int(hit)
        rr_sum += rr
        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "kind": q["kind"],
            "hit": hit,
            "first_rank": first_rank,
            "rr": rr,
            "retrieved": retrieved_ids,
            "relevant": list(relevant),
        })
        marker = "✓" if hit else "✗"
        rank_str = f"rank {first_rank}" if first_rank else "miss"
        print(f"  {marker} [{q['id']}] {rank_str:>8}   {q['query'][:60]}")

    n = len(queries)
    elapsed = time.time() - t0
    metrics = {
        "mode": name,
        "n_queries": n,
        "hit_rate_at_k": hits / n,
        "mrr": rr_sum / n,
        "elapsed_s": round(elapsed, 1),
        "per_query": per_query,
    }
    print(f"\n  Hit Rate@{k}: {metrics['hit_rate_at_k']:.3f}   MRR: {metrics['mrr']:.3f}   ({elapsed:.1f}s)")
    return metrics


def breakdown_by_kind(per_query: list[dict], k: int) -> dict:
    """Split metrics by query kind (explicit vs fuzzy)."""
    out = {}
    for kind in ("explicit", "fuzzy"):
        sub = [q for q in per_query if q["kind"] == kind]
        if not sub:
            continue
        out[kind] = {
            "n": len(sub),
            "hit_rate_at_k": sum(q["hit"] for q in sub) / len(sub),
            "mrr": sum(q["rr"] for q in sub) / len(sub),
        }
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3-mode RAG retrieval ablation")
    parser.add_argument("--repo-path", default="repos/fastapi/fastapi",
                        help="Path to the repo source dir to index")
    parser.add_argument("--queries", default="eval/queries_fastapi.json",
                        help="Path to the labeled query set JSON")
    parser.add_argument("--cache", default="eval/cache_fastapi.npz",
                        help="Path to embedding cache (.npz)")
    parser.add_argument("--k", type=int, default=5, help="Top-K (default 5)")
    parser.add_argument("--out", default="eval/results.json",
                        help="Where to dump full results JSON")
    parser.add_argument("--skip-rerank", action="store_true",
                        help="Skip the LLM reranker mode (saves OpenAI cost)")
    args = parser.parse_args()

    # Load queries
    with open(args.queries) as f:
        qset = json.load(f)
    queries = qset["queries"]
    print(f"Loaded {len(queries)} queries from {args.queries}")

    # Build/load index
    chunks, embeddings, chunk_ids = build_or_load_index(args.repo_path, args.cache)

    # BM25 needs the chunks (it uses metadata["name"] + content + file_path internally)
    # Convert to format hybrid.build_bm25_index expects
    build_bm25_index([
        {"content": c["content"], "metadata": c["metadata"]}
        for c in chunks
    ])

    # Run all 3 modes
    results = {
        "config": {
            "repo": qset.get("repo", args.repo_path),
            "k": args.k,
            "n_queries": len(queries),
            "n_chunks": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
        },
        "modes": {},
    }

    results["modes"]["A_dense"] = evaluate_mode(
        "A) Dense only (semantic baseline)",
        lambda q, k: dense_search(q, embeddings, chunks, k),
        queries, args.k,
    )

    results["modes"]["B_hybrid"] = evaluate_mode(
        "B) Hybrid (dense + BM25 + RRF)",
        lambda q, k: hybrid_search_local(q, embeddings, chunks, k),
        queries, args.k,
    )

    if not args.skip_rerank:
        results["modes"]["C_hybrid_rerank"] = evaluate_mode(
            "C) Hybrid + LLM rerank (full system)",
            lambda q, k: hybrid_rerank_search(q, embeddings, chunks, k),
            queries, args.k,
        )

    # Breakdowns
    for mode_key, mode_data in results["modes"].items():
        mode_data["by_kind"] = breakdown_by_kind(mode_data["per_query"], args.k)

    # Summary table
    print(f"\n{'='*70}\nSUMMARY  (k={args.k}, {len(queries)} queries)\n{'='*70}")
    print(f"{'Mode':<45}{'Hit Rate@'+str(args.k):>14}{'MRR':>8}")
    print("-" * 70)
    for key, data in results["modes"].items():
        print(f"{data['mode']:<45}{data['hit_rate_at_k']:>14.3f}{data['mrr']:>8.3f}")
    print()

    # Per-kind breakdown
    print(f"{'Mode':<45}{'kind':>10}{'HR@'+str(args.k):>10}{'MRR':>8}")
    print("-" * 70)
    for key, data in results["modes"].items():
        for kind, m in data.get("by_kind", {}).items():
            print(f"{data['mode']:<45}{kind:>10}{m['hit_rate_at_k']:>10.3f}{m['mrr']:>8.3f}")
    print()

    # Improvement from baseline
    if "A_dense" in results["modes"]:
        a = results["modes"]["A_dense"]
        print(f"Improvements over dense-only baseline:")
        for key in ("B_hybrid", "C_hybrid_rerank"):
            if key in results["modes"]:
                m = results["modes"][key]
                d_hr = m["hit_rate_at_k"] - a["hit_rate_at_k"]
                d_mrr = m["mrr"] - a["mrr"]
                print(f"  {m['mode']:<45} ΔHR={d_hr:+.3f}  ΔMRR={d_mrr:+.3f}")

    # Dump full results
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results → {args.out}")


if __name__ == "__main__":
    main()

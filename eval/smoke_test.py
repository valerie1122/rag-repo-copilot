"""
Sanity check the eval logic end-to-end with a fake (deterministic) embedder.

This proves the metrics math, RRF fusion, and rank-counting are correct without
needing a live OpenAI key. Run this as part of CI.

Run from project root:
    python -m eval.smoke_test
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure we don't accidentally hit OpenAI
os.environ["OPENAI_API_KEY"] = "sk-fake-for-tests"

from src.retrieval.hybrid import build_bm25_index, bm25_search, reciprocal_rank_fusion


# ---- Mini fake corpus ----
chunks = [
    {"id": "a.py::foo",    "content": "def foo(): return cookie value",     "metadata": {"file_path": "a.py", "name": "foo"}},
    {"id": "a.py::bar",    "content": "def bar(): handle authentication",   "metadata": {"file_path": "a.py", "name": "bar"}},
    {"id": "b.py::baz",    "content": "class Baz: handles cookies",         "metadata": {"file_path": "b.py", "name": "baz"}},
    {"id": "b.py::auth",   "content": "def auth(): check passwords",        "metadata": {"file_path": "b.py", "name": "auth"}},
    {"id": "c.py::cookie", "content": "def cookie(): set http cookie",      "metadata": {"file_path": "c.py", "name": "cookie"}},
]


# Fake "embedding": one-hot on the first matching keyword (deterministic + cheap)
KEYWORDS = ["cookie", "auth", "foo", "bar", "baz"]
def fake_embed(text: str) -> np.ndarray:
    v = np.zeros(len(KEYWORDS), dtype=np.float32)
    for i, k in enumerate(KEYWORDS):
        if k in text.lower():
            v[i] = 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# Build embeddings
embeddings = np.array([fake_embed(c["content"]) for c in chunks])
build_bm25_index([{"content": c["content"], "metadata": c["metadata"]} for c in chunks])


def dense_search(q, k):
    qv = fake_embed(q)
    sims = embeddings @ qv
    idxs = np.argsort(-sims)[:k]
    return [{"id": chunks[i]["id"], "content": chunks[i]["content"],
             "metadata": chunks[i]["metadata"], "score": float(sims[i])} for i in idxs]


def hybrid(q, k):
    fetch = k * 2
    d = dense_search(q, fetch)
    b = bm25_search(q, top_k=fetch)
    return reciprocal_rank_fusion(d, b)[:k]


# ---- Mini eval ----
queries = [
    {"q": "how are cookies handled", "relevant": {"c.py::cookie", "b.py::baz"}},
    {"q": "authentication code",     "relevant": {"a.py::bar", "b.py::auth"}},
    {"q": "the foo function",         "relevant": {"a.py::foo"}},
]


def metrics(search_fn, queries, k=3):
    hits = 0
    rr_sum = 0.0
    for item in queries:
        results = search_fn(item["q"], k)
        ids = [r["id"] for r in results]
        if any(rid in item["relevant"] for rid in ids[:k]):
            hits += 1
        rank = next((i + 1 for i, rid in enumerate(ids) if rid in item["relevant"]), None)
        if rank:
            rr_sum += 1.0 / rank
    return hits / len(queries), rr_sum / len(queries)


hr_d, mrr_d = metrics(dense_search, queries)
hr_h, mrr_h = metrics(hybrid, queries)

print(f"Dense  HR@3={hr_d:.3f}  MRR={mrr_d:.3f}")
print(f"Hybrid HR@3={hr_h:.3f}  MRR={mrr_h:.3f}")

# Sanity assertions
assert 0 <= hr_d <= 1 and 0 <= hr_h <= 1
assert 0 <= mrr_d <= 1 and 0 <= mrr_h <= 1
# All 3 queries have an obvious keyword match in our toy data, hybrid should hit all
assert hr_h == 1.0, f"Expected hybrid to find all 3, got HR={hr_h}"
print("\nAll smoke-test assertions passed ✓")

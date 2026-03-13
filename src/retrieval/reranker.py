"""
Reranker: Re-score search results using a LLM to improve ranking quality.

Why rerank?
- Hybrid search (Day 6) finds good candidates, but the ranking isn't perfect
- Vector search scores (cosine distance) and BM25 scores measure different things
- RRF combines them, but it only looks at RANK positions, not actual content
- A reranker looks at the actual CONTENT of each result and the question,
  then scores how relevant each result actually is

How it works:
1. Hybrid search returns top_k candidates (e.g., 10 results)
2. Reranker takes each result + the user's question
3. Asks GPT to score each result's relevance (0-10)
4. Re-sorts results by relevance score
5. Returns the top results — now in truly the best order

This is a common pattern in production RAG systems:
- First stage: fast but rough (vector search + BM25)
- Second stage: slow but accurate (LLM reranking)

Think of it like a job interview:
- Stage 1 (hybrid search): Resume screening — fast, filters out obviously bad candidates
- Stage 2 (reranker): Actual interview — slower, but finds the truly best candidate
"""

import json
from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL


# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# The prompt that tells GPT how to score relevance
RERANK_PROMPT = """You are a code search relevance evaluator. Your task is to score how relevant each code snippet is to the user's question.

## User's Question:
{question}

## Code Snippets to Evaluate:
{snippets}

## Instructions:
For each snippet, provide a relevance score from 0 to 10:
- 10: Directly answers the question, contains the exact code being asked about
- 7-9: Highly relevant, closely related to the question
- 4-6: Somewhat relevant, partially related
- 1-3: Slightly relevant, only tangentially related
- 0: Completely irrelevant

Respond with ONLY a JSON array of objects, each with "index" and "score".
Example: [{{"index": 0, "score": 8}}, {{"index": 1, "score": 3}}]
"""


def rerank(question: str, search_results: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank search results using GPT to score relevance.

    Takes the rough results from hybrid search and uses GPT to
    determine which results are truly most relevant to the question.

    Args:
        question: The user's question
        search_results: Results from hybrid_search() or vector search
        top_k: Number of top results to return after reranking

    Returns:
        Reranked results (best first), each with an added "relevance_score" field
    """
    if not search_results:
        return []

    # If we have fewer results than requested, just rerank what we have
    if len(search_results) <= top_k:
        top_k = len(search_results)

    # Build the snippets text for the prompt
    snippets_text = ""
    for i, result in enumerate(search_results):
        metadata = result.get("metadata", {})
        content = result.get("content", "")

        # Truncate very long code chunks to save tokens
        if len(content) > 500:
            content = content[:500] + "\n... (truncated)"

        snippets_text += (
            f"\n### Snippet {i}:\n"
            f"File: {metadata.get('file_path', 'unknown')}\n"
            f"Name: {metadata.get('name', 'unknown')}\n"
            f"Type: {metadata.get('chunk_type', 'unknown')}\n"
            f"```\n{content}\n```\n"
        )

    # Build the complete prompt
    prompt = RERANK_PROMPT.format(question=question, snippets=snippets_text)

    try:
        # Ask GPT to score each result
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise code relevance evaluator. Respond ONLY with valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,  # We want consistent, deterministic scoring
            max_tokens=500,
        )

        # Parse the GPT response
        response_text = response.choices[0].message.content.strip()

        # Handle case where GPT wraps response in markdown code block
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]  # Remove first line
            response_text = response_text.rsplit("```", 1)[0]  # Remove last ```

        scores = json.loads(response_text)

        # Apply scores to results
        scored_results = []
        for score_item in scores:
            idx = score_item["index"]
            if 0 <= idx < len(search_results):
                result = search_results[idx].copy()
                result["relevance_score"] = score_item["score"]
                scored_results.append(result)

        # Sort by relevance score (highest first)
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        print(f"Reranked {len(scored_results)} results (top scores: "
              f"{[r['relevance_score'] for r in scored_results[:3]]})")

        return scored_results[:top_k]

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # If GPT returns invalid JSON, fall back to original order
        print(f"Reranking failed ({type(e).__name__}: {e}), using original order")
        return search_results[:top_k]


def rerank_with_details(question: str, search_results: list[dict], top_k: int = 5) -> dict:
    """
    Same as rerank() but returns additional details for evaluation.

    Useful for comparing before/after reranking.

    Returns:
        Dict with:
        - results: reranked results
        - original_order: list of names in original order
        - reranked_order: list of names in new order
        - scores: dict mapping name → relevance_score
    """
    original_order = [r.get("metadata", {}).get("name", "unknown") for r in search_results]

    reranked = rerank(question, search_results, top_k)

    reranked_order = [r.get("metadata", {}).get("name", "unknown") for r in reranked]
    scores = {r.get("metadata", {}).get("name", "unknown"): r.get("relevance_score", 0) for r in reranked}

    return {
        "results": reranked,
        "original_order": original_order,
        "reranked_order": reranked_order,
        "scores": scores,
    }

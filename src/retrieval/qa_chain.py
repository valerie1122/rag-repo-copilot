"""
QA Chain: The "brain" of the RAG system.

This is where everything comes together:
1. User asks a question
2. We search the vector store for relevant code (Day 3)
3. We build a prompt with the code + question (prompts.py)
4. We send it to GPT and get an answer (this file)

Evolution of this file:
- Day 4: Basic pipeline — vector search → prompt → GPT
- Day 7: Upgraded pipeline — hybrid search → rerank → prompt → GPT

The pipeline now has 3 stages:
1. Retrieval: hybrid_search() combines vector + BM25 (Day 6)
2. Reranking: rerank() uses GPT to re-score results (Day 7)
3. Generation: ask_llm() gets GPT to write the answer (Day 4)
"""

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K
from src.embedding.store import search as vector_search
from src.retrieval.prompts import build_prompt
from src.retrieval.reranker import rerank


# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to GPT and get a response.

    Args:
        prompt: The complete prompt (question + code context)

    Returns:
        GPT's answer as a string
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert code analyst. Answer questions about code repositories accurately and concisely.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=1500,
    )

    return response.choices[0].message.content


def ask(question: str, top_k: int = TOP_K, use_hybrid: bool = True, use_rerank: bool = True) -> dict:
    """
    The main function: ask a question about the code repository.

    This is the complete RAG pipeline in one function:
    1. Search for relevant code chunks (hybrid or vector-only)
    2. Optionally rerank results with GPT
    3. Build a prompt with the code + question
    4. Send to GPT for an answer

    Args:
        question: User's question in natural language
        top_k: Number of code chunks to use for the answer
        use_hybrid: If True, use hybrid search (Day 6). If False, use vector search only.
        use_rerank: If True, rerank results with GPT (Day 7). If False, skip reranking.

    Returns:
        Dict with:
        - answer: GPT's response
        - sources: the code chunks that were used
        - question: the original question
        - search_method: which search method was used
    """
    # Step 1: Retrieve relevant code
    print(f"Searching for code related to: \"{question}\"")

    if use_hybrid:
        # Import here to avoid circular imports when hybrid isn't needed
        from src.retrieval.hybrid import hybrid_search
        # Fetch more results for reranking to have a better pool
        fetch_k = top_k * 2 if use_rerank else top_k
        search_results = hybrid_search(question, top_k=fetch_k)
        search_method = "hybrid"
    else:
        search_results = vector_search(question, top_k=top_k * 2 if use_rerank else top_k)
        search_method = "vector"

    print(f"Found {len(search_results)} relevant code chunks via {search_method} search")

    # Step 2: Rerank (optional, Day 7)
    if use_rerank and len(search_results) > 0:
        print("Reranking results with GPT...")
        search_results = rerank(question, search_results, top_k=top_k)
        search_method += "+rerank"
    else:
        search_results = search_results[:top_k]

    # Step 3: Build the prompt
    prompt = build_prompt(question, search_results)

    # Step 4: Generate answer
    print("Asking GPT for an answer...")
    answer = ask_llm(prompt)

    # Step 5: Package the response
    sources = []
    for r in search_results:
        source_info = {
            "file_path": r["metadata"].get("file_path", ""),
            "name": r["metadata"].get("name", ""),
            "chunk_type": r["metadata"].get("chunk_type", ""),
            "start_line": r["metadata"].get("start_line", ""),
            "end_line": r["metadata"].get("end_line", ""),
        }
        # Include whatever score is available
        if "distance" in r and r["distance"] is not None:
            source_info["distance"] = r["distance"]
        if "rrf_score" in r:
            source_info["rrf_score"] = r["rrf_score"]
        if "relevance_score" in r:
            source_info["relevance_score"] = r["relevance_score"]

        sources.append(source_info)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "search_method": search_method,
    }

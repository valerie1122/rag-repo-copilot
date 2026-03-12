"""
QA Chain: The "brain" of the RAG system.

This is where everything comes together:
1. User asks a question
2. We search the vector store for relevant code (Day 3)
3. We build a prompt with the code + question (prompts.py)
4. We send it to GPT and get an answer (this file)

This completes the RAG pipeline: Retrieval + Augmented + Generation
- Retrieval: search() finds relevant code
- Augmented: build_prompt() adds the code to the prompt
- Generation: ask_llm() gets GPT to write the answer
"""

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K
from src.embedding.store import search
from src.retrieval.prompts import build_prompt


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


def ask(question: str, top_k: int = TOP_K) -> dict:
    """
    The main function: ask a question about the code repository.

    This is the complete RAG pipeline in one function:
    1. Search for relevant code chunks
    2. Build a prompt with the code + question
    3. Send to GPT for an answer

    Args:
        question: User's question in natural language
        top_k: Number of code chunks to retrieve

    Returns:
        Dict with:
        - answer: GPT's response
        - sources: the code chunks that were used
        - question: the original question
    """
    # Step 1: Retrieve relevant code
    print(f"Searching for code related to: \"{question}\"")
    search_results = search(question, top_k=top_k)
    print(f"Found {len(search_results)} relevant code chunks")

    # Step 2: Build the prompt
    prompt = build_prompt(question, search_results)

    # Step 3: Generate answer
    print("Asking GPT for an answer...")
    answer = ask_llm(prompt)

    # Step 4: Package the response
    sources = []
    for r in search_results:
        sources.append({
            "file_path": r["metadata"]["file_path"],
            "name": r["metadata"]["name"],
            "chunk_type": r["metadata"]["chunk_type"],
            "start_line": r["metadata"]["start_line"],
            "end_line": r["metadata"]["end_line"],
            "distance": r["distance"],
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }

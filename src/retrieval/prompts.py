"""
Prompt Templates: Instructions that tell GPT how to answer questions about code.

What is a prompt?
- A prompt is the instruction you send to GPT along with the data
- A good prompt = good answers. A bad prompt = garbage answers.
- This is called "Prompt Engineering" — one of the most important skills in AI

Why do we need a template?
- We don't just send the user's question to GPT
- We also send the relevant code chunks we found (from Day 3's search)
- The template tells GPT: "Here's some code. Answer the question based on this code."
- This is the core idea of RAG: ground the LLM's answer in real data
"""

# The main QA prompt template
# {context} will be replaced with the retrieved code chunks
# {question} will be replaced with the user's question
QA_PROMPT_TEMPLATE = """You are an expert code assistant. Your job is to answer questions about a code repository based on the code snippets provided below.

## Rules:
1. ONLY answer based on the provided code snippets. Do not make up information.
2. If the code snippets don't contain enough information to answer, say so honestly.
3. Always reference specific file paths and line numbers when discussing code.
4. Format code references like: `filename.py:L10-L20`
5. Include relevant code snippets in your answer using markdown code blocks.
6. Be concise but thorough.

## Code Snippets from the Repository:

{context}

## Question:
{question}

## Answer:
"""


def build_context(search_results: list[dict]) -> str:
    """
    Build the context string from search results.

    Takes the code chunks returned by vector search and formats them
    into a readable string that gets inserted into the prompt.

    Args:
        search_results: Output from store.search()

    Returns:
        A formatted string with all code chunks and their metadata
    """
    context_parts = []

    for i, result in enumerate(search_results, 1):
        metadata = result["metadata"]
        code = metadata.get("raw_code", result.get("content", ""))

        context_parts.append(
            f"### Snippet {i}: {metadata['name']} "
            f"({metadata['file_path']}:L{metadata['start_line']}-L{metadata['end_line']})\n"
            f"Type: {metadata['chunk_type']}\n"
            f"```python\n{code}\n```\n"
        )

    return "\n".join(context_parts)


def build_prompt(question: str, search_results: list[dict]) -> str:
    """
    Build the complete prompt to send to GPT.

    Combines the template + retrieved code + user's question.

    Args:
        question: User's question
        search_results: Retrieved code chunks from vector search

    Returns:
        The complete prompt string
    """
    context = build_context(search_results)
    return QA_PROMPT_TEMPLATE.format(context=context, question=question)

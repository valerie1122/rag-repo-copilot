"""
Test the complete RAG pipeline: search code → build prompt → get GPT answer.

Prerequisites:
- Day 3 test must have run first (so Chroma has data in it)
- Valid OPENAI_API_KEY in .env
- This calls GPT-4o, so it costs a few cents per question
"""

from src.retrieval.qa_chain import ask


def test_qa_pipeline():
    """
    Test the complete question-answering pipeline.
    Uses the httpbin data that was indexed in Day 3's test.
    """
    print("\n" + "=" * 60)
    print("RAG Q&A PIPELINE TEST")
    print("=" * 60)

    questions = [
        "How does this app handle HTTP GET requests?",
        "What test cases are there for authentication?",
        "How is the Flask app structured and initialized?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Question {i}: {question}")
        print(f"{'─' * 60}")

        result = ask(question)

        print(f"\nAnswer:\n{result['answer']}")

        print(f"\nSources used:")
        for s in result["sources"]:
            print(f"  [{s['chunk_type']}] {s['name']} ({s['file_path']}:L{s['start_line']}-L{s['end_line']}) distance={s['distance']:.4f}")

    print("\n" + "=" * 60)
    print("Q&A PIPELINE TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_qa_pipeline()

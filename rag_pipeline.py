from llm import LLMService
from vector_store import VectorStore


def answer_question(
    question,
    chunks,
    index
):

    vector_store = VectorStore()

    retrieved_chunks = vector_store.search(
        question,
        index,
        chunks
    )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    llm = LLMService()

    return llm.generate(prompt)
from langchain_core.messages import HumanMessage, SystemMessage
from rag.models import get_llm


llm = get_llm()


SYSTEM_PROMPT = """
You are a developer documentation assistant.

Answer ONLY using the provided context.

Rules:
- Only use the context provided to answer the question
- Do NOT use external knowledge
- If answer not found, say:
  "I cannot find relevant information in the provided documents."

- For every claim you make, cite the specific Source ID (e.g., [Source 1]) from the context below.
"""


def generate_answer(query, reranked_docs, chat_history):

    if not reranked_docs:

        return {

            "answer": "I cannot find relevant information in the provided documents.",
            "sources": [],
            "reranked_docs": []

        }

    context_blocks = []

    sources = []

    for i, doc in enumerate(reranked_docs):

        context_blocks.append(
            f"[Source {i+1}]\n{doc['content']}"
        )

        sources.append(doc["source"])

    context = "\n\n".join(context_blocks)

    user_prompt = f"""
Context:
{context}

Chat history:
{chat_history}

Question:
{query}
"""

    messages = [

        SystemMessage(content=SYSTEM_PROMPT),

        HumanMessage(content=user_prompt)

    ]

    response = llm.invoke(messages)

    answer = response.content

    return {

        "answer": answer,
        "sources": list(set(sources)),
        "reranked_docs": reranked_docs

    }
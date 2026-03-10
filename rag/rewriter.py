from langchain_core.messages import HumanMessage, SystemMessage
from rag.models import get_llm
from rag.logger import logger

llm = get_llm()

SYSTEM_PROMPT = """
You are a query rewriting assistant for a developer documentation RAG system.
Your task is to take a user's latest query and the conversation history (summary and recent messages), 
and rewrite the user's latest query into a fully standalone, unambiguous query that contains all necessary context for a vector search.
Do not answer the query. Just return the rewritten query text.
If the query is already standalone and does not rely on history, return it exactly as is.
"""

def rewrite_query(query: str, summary: str, buffer: list) -> str:
    logger.info(f"Rewriting query: {query}")
    if not summary and not buffer:
        logger.info("No summary or buffer, returning original query")
        return query

    user_prompt = f"""
Conversation Summary:
{summary}

Recent Messages:
{buffer}

Latest User Query:
{query}

Rewrite the query to be standalone:
"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    response = llm.invoke(messages)
    logger.info(f"Rewritten query: {response.content.strip()}")
    return response.content.strip()

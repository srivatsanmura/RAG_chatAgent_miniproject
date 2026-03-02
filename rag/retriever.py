from rag.config import RETRIEVAL_TOP_K
from rag.logger import logger


def retrieve(query, vectordb):
    """
    Retrieves documents from the vector database based on the query.

    Args:
        query (str): The query to search for.
        vectordb (Chroma): The vector database.

    Returns:
        list: A list of dictionaries containing the retrieved documents.
    """
    docs = vectordb.similarity_search_with_score(
        query,
        k=RETRIEVAL_TOP_K
    )

    retrieved = []

    for doc, score in docs:

        retrieved.append({

            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "vector_score": float(score)

        })
    
    logger.info(f"Retrieved {len(retrieved)} documents")

    return retrieved
from rag.config import RETRIEVAL_TOP_K, RETRIEVAL_FETCH_K, RETRIEVAL_LAMBDA_MULT
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
    retriever = vectordb.as_retriever(
        search_type="mmr",
         search_kwargs={
        "k": RETRIEVAL_TOP_K, # Final docs returned
        "fetch_k": RETRIEVAL_FETCH_K,        # Initial pool for diversity filtering
        "lambda_mult": RETRIEVAL_LAMBDA_MULT    # Diversity vs Relevance balance
    }
    )
    
    docs = retriever.invoke(query)

    retrieved = []

    for doc in docs:

        retrieved.append({

            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            # MMR search via generic retriever.invoke does not return scores
            "vector_score": 0.0 
        })
    
    logger.info(f"Retrieved {len(retrieved)} documents")

    return retrieved
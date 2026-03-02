from langchain_chroma import Chroma
from rag.embeddings import get_embeddings
from rag.config import CHROMA_PATH, COLLECTION_NAME


def get_vectordb():
    """
    Returns the vector database.

    Returns:
        Chroma: The vector database.
    """
    embeddings = get_embeddings()

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectordb

##
# This is a redundant function, we can use the add_documents method of the vector database directly
def add_chunks(vectordb, chunks):
    """
    Adds chunks to the vector database.

    Args:
        vectordb (Chroma): The vector database.
        chunks (list): A list of dictionaries containing the chunked text and metadata.

    Returns:
        Chroma: The vector database with the added chunks.
    """
    vectordb.add_documents(chunks)

    return vectordb
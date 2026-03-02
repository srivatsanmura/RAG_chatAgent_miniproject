####
## Code for chunking the documents
###

from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents):
    """
    Chunks the documents into smaller chunks.

    Args:
        documents (list): A list of dictionaries containing the extracted text and source URLs.

    Returns:
        list: A list of dictionaries containing the chunked text and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = [doc["content"] for doc in documents]

    metadatas = [
        {"source": doc["source"]}
        for doc in documents
    ]

    chunks = splitter.create_documents(
        texts,
        metadatas=metadatas
    )

    return chunks
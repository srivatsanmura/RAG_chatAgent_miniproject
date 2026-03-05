####
## Code for chunking the documents
###

from langchain_text_splitters import RecursiveCharacterTextSplitter
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

    # Support both LangChain Document objects and legacy dicts
    from langchain_core.documents import Document as LCDocument
    if documents and isinstance(documents[0], LCDocument):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
    else:
        texts = [doc["content"] for doc in documents]
        metadatas = [{"source": doc["source"]} for doc in documents]

    chunks = splitter.create_documents(
        texts,
        metadatas=metadatas
    )

    return chunks
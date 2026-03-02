from rag.extract_docs import extract_url
from rag.chunking import chunk_documents
from rag.vectordb import get_vectordb
from rag.config import EMBEDDING_BATCH_SIZE

from rag.logger import logger

"""
Implmenting scalable vector db ingestion pattern

"""

def build_index():
    urls = [...]
    vectordb = get_vectordb()

    batch = []
    for i, url in enumerate(urls):

        logger.info(f"Ingesting: {url}; {i+1}/{len(urls)}")
        doc = extract_url(url)

        if not doc:
            logger.error(f"Failed to ingest: {url}")
            continue

        chunks = chunk_documents([doc])
        batch.extend(chunks)

        if len(batch) >= EMBEDDING_BATCH_SIZE:
            logger.info(f"Adding batch of {len(batch)} chunks to vector db")
            vectordb.add_documents(batch)
        batch = []

    if batch:
        logger.info(f"Adding batch of {len(batch)} chunks to vector db")
        vectordb.add_documents(batch)

    logger.info("Chroma index built successfully")

if __name__ == "__main__":
    build_index()
###
# This will be a variation of build_index.py, in that, we will use
# github docs repository of langchain-ai to build the index
# we will embed all the relevant mdx files and store them in ChromaDB


### Flow of execution is:
## For every markdown file:
##  file ---> UnstructuredMarkdownLoader --> MdSplitter --> chunking --> vector db
###
##

from langchain_community.document_loaders import  UnstructuredMarkdownLoader
from langchain_core.documents import Document
from pathlib import Path
from pathlib import PureWindowsPath
from langchain_text_splitters import MarkdownHeaderTextSplitter

from rag.chunking import chunk_documents
from rag.vectordb import get_vectordb
from rag.config import EMBEDDING_BATCH_SIZE
from rag.logger import logger

try:

    p = Path(PureWindowsPath(r"C:/Users/msriv/OneDrive/Srivatsan/Srivatsan/Learning and Dev/gitRepos/langchain_ai_docs"))


    headers_to_split_on = [("#", "Header 1"), 
                            ("##", "Header 2"),
                            ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    ## create a list of files based on glob

    files = list(p.glob("**/*.mdx"))

    logger.info(f"No. of files to ingest into vector db = {len(files)}")

    batch = []
    vectordb = get_vectordb()

    # now for each file use UnstructuredMarkdownLoader to load the file
    for file in files:
        logger.info(f"Processing file: {file}")
        loader = UnstructuredMarkdownLoader(file, strategy="fast")
        docs = loader.load()
        all_splits = []

        # Now split the loaded markdown doc by the Markdownheader splitter
        all_splits.extend(markdown_splitter.split_text(docs[0].page_content))

        # since we feed only the page_content to the above splitter, their metadata is lost
        # Add metadata to the splits explicitly
        for split in all_splits:
            split.metadata.update(docs[0].metadata)
            
        logger.info(f"No. of splits created from file: {len(all_splits)}")

        chunks = chunk_documents(all_splits)
        logger.info(f"No. of chunks created from file: {len(chunks)}")
        batch.extend(chunks)
        logger.info(f"No. of chunks in batch: {len(batch)}")

        if len(batch) >= EMBEDDING_BATCH_SIZE:
            logger.info(f"Adding batch of {len(batch)} chunks to vector db")
            vectordb.add_documents(batch)
            batch = []

    # Catch any remaining documents in the final batch after the loop ends
    if batch:
        logger.info(f"Adding batch of {len(batch)} chunks to vector db")
        vectordb.add_documents(batch)

    logger.info("Chroma index built successfully")

except Exception as e:
    logger.error(f"Error: {e}")
    logger.shutdown()

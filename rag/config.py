###
# Config setting for the RAG pipeline
###

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "gpt-4o-mini"
)

VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    "data/vectordb"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5

SIMILARITY_THRESHOLD = 0.3

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "rag_docs"

EMBEDDING_BATCH_SIZE = 100

LOG_FILE = "logs/rag_agent.log"
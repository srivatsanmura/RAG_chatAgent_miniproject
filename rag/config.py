###
# Config setting for the RAG pipeline
###

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SOURCE_PATH = os.getenv(
    "SOURCE_PATH"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
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
RETRIEVAL_FETCH_K = 50
RETRIEVAL_LAMBDA_MULT = 0.5

RERANK_TOP_K = 5
RERANK_THRESHOLD = 0.4
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

SIMILARITY_THRESHOLD = 0.3

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "langchain_tech_docs"

EMBEDDING_BATCH_SIZE = 100

LOG_FILE = "logs/rag_agent.log"
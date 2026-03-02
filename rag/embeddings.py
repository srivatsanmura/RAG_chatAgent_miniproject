from langchain.embeddings.openai import OpenAIEmbeddings
from rag.config import EMBEDDING_MODEL


def get_embeddings():

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )
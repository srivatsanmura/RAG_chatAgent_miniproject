from langchain_openai import ChatOpenAI
from rag.config import GENERATION_MODEL


def get_llm(temperature=0):

    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        temperature=temperature,
        max_tokens=1000,
    )

    return llm
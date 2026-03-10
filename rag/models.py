from langchain_openai import ChatOpenAI
from rag.config import GENERATION_MODEL

# Module‑level singleton for the LLM
_llm_instance: ChatOpenAI | None = None

def get_llm(temperature=0):
    """Return a single shared LLM instance.
    The first call creates the instance; subsequent calls reuse it.
    The `temperature` argument is only respected on the first creation.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=GENERATION_MODEL,
            temperature=temperature,
            max_tokens=1000,
        )
    return _llm_instance
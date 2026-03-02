from langgraph.graph import StateGraph, END
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import generate_answer
from typing import TypedDict


class RAGState(TypedDict):
    query: str
    vectordb: Chroma
    retrieved_docs: list
    reranked_docs: list
    chat_history: list
    answer: str
    sources: list


def retrieve_node(state: RAGState) -> RAGState:

    docs = retrieve(
        state["query"],
        state["vectordb"]
    )

    state["retrieved_docs"] = docs

    return state


def rerank_node(state: RAGState) -> RAGState:

    reranked = rerank(
        state["query"],
        state["retrieved_docs"]
    )

    state["reranked_docs"] = reranked

    return state


def generate_node(state: RAGState) -> RAGState:

    answer, sources = generate_answer(
        state["query"],
        state["reranked_docs"],
        state["chat_history"]
    )

    state["answer"] = answer
    state["sources"] = sources

    return state


def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
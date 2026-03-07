from langgraph.graph import StateGraph, END
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.compressor import sentence_filter
from rag.generator import generate_answer
from langchain_chroma import Chroma
from typing import TypedDict


class RAGState(TypedDict):
    query: str
    vectordb: Chroma
    retrieved_docs: list
    reranked_docs: list
    compressed_docs: list
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


def compress_node(state: RAGState) -> RAGState:

    compressed_docs = []
    
    for doc in state["reranked_docs"]:
        compressed_text = sentence_filter(
            state["query"],
            doc["content"]
        )
        
        # Only keep the document if it still has content after compression
        if compressed_text.strip():
            # Create a shallow copy to avoid mutating the original
            new_doc = doc.copy()
            new_doc["content"] = compressed_text
            compressed_docs.append(new_doc)

    state["compressed_docs"] = compressed_docs

    return state


def generate_node(state: RAGState) -> RAGState:

    result = generate_answer(
        state["query"],
        state["compressed_docs"],
        state["chat_history"]
    )

    state["answer"] = result["answer"]
    state["sources"] = result["sources"]

    return state


def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("compress", compress_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
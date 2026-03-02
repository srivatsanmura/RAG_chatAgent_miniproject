from sentence_transformers import CrossEncoder
from rag.config import RERANK_TOP_K

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, docs):

    doc_objects = [doc for doc, source, score in docs]

    pairs = [
        (query, doc.page_content)
        for doc in doc_objects
    ]

    scores = reranker.predict(pairs)

    ranked = list(zip(doc_objects, scores))

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:RERANK_TOP_K]
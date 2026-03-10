####
## Reranker
#### The objective of this component is to rerank the retrieved documents 
#### using crossEncoder model and return the top k documents
#### activation function Sigmoid is used to compute the score in the probability range of 0-1
#### Threshold for selecting the documents is set to 0.5 (at this time)
#### Top k documents are selected based on the rerank score
from sentence_transformers import CrossEncoder
from rag.config import RERANK_TOP_K, CROSS_ENCODER_MODEL
from rag.config import RERANK_THRESHOLD
import torch

from rag.logger import logger

reranker = CrossEncoder(
    CROSS_ENCODER_MODEL,
    activation_fn=torch.nn.Sigmoid()
)   


def rerank(query, docs):
    logger.info(f"Reranking query: {query} against {len(docs)} documents")
    if not docs:
        return []

    pairs = [
        (query, doc["content"])
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    for i, doc in enumerate(docs):
        doc["rerank_score"] = float(scores[i])
    
    filtered_docs = [
        doc for doc in docs if doc["rerank_score"] >= RERANK_THRESHOLD
    ]

    ### If filtered docs are less than 5, return the top 5 docs
    if len(filtered_docs) < RERANK_TOP_K:
        filtered_docs = docs[:RERANK_TOP_K]

    filtered_docs.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    logger.info(f"Reranked {len(filtered_docs)} documents")
    return filtered_docs[:RERANK_TOP_K]
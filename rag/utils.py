import pandas as pd


def prepare_visualization_data(reranked_docs):

    rows = []

    for i, doc in enumerate(reranked_docs):

        rows.append({

            "chunk_id": i,
            "source": doc["source"],
            "rerank_score": doc["rerank_score"],
            "preview": doc["content"][:200]

        })

    return pd.DataFrame(rows)
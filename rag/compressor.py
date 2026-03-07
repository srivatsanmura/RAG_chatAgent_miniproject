###
# Sentence Compressor that comes after reranking to reduce the number of tokens
# to save on LLM cost and reduce hallucinations
###

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from rag.embeddings import get_embeddings
from rag.logger import logger


def sentence_filter(query, text, top_k=5):

    logger.info(f"Compressing text: {text}")
    # Download punkt tokenizer data if not already present
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)


    punk_param = PunktParameters()
    punk_abbrev_list = ["dr", "mr", "vs", "d.c", "u.s.a", "u.k", "etc", "e.g", "i.e",
    "p.m", "a.m", "v.p"]
    punk_param.abbrev_types.update(punk_abbrev_list)
    punkt_tokenizer = PunktSentenceTokenizer(punk_param)

    # Get the existing embedding model to avoid duplication
    embeddings = get_embeddings()
    sentences = punkt_tokenizer.tokenize(text)

    if not sentences:
        return ""

    query_emb = np.array([embeddings.embed_query(query)])

    sent_emb = np.array(embeddings.embed_documents(sentences))

    sims = cosine_similarity(query_emb, sent_emb)[0]

    ranked = sorted(
        zip(sentences, sims),
        key=lambda x: x[1],
        reverse=True
    )

    selected = [s for s, score in ranked[:top_k]]

    compressed_text = " ".join(selected)
    logger.info(f"Compressed text: {compressed_text}")

    return compressed_text
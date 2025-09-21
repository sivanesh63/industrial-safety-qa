# reranker.py
from rank_bm25 import BM25Okapi
import numpy as np

def rerank(query, results, alpha=0.5):
    # Extract candidate docs
    docs = results["documents"][0]
    ids = results["ids"][0]

    # BM25
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.split())

    # Embedding scores (from Chroma)
    emb_scores = results["distances"][0]  # smaller = closer
    emb_scores = 1 - np.array(emb_scores)  # convert to similarity

    # Combine
    final_scores = alpha * emb_scores + (1 - alpha) * bm25_scores

    reranked = sorted(zip(ids, docs, final_scores), key=lambda x: -x[2])
    return reranked

# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

import numpy as np

from reranker import rerank
from answer import make_answer

app = FastAPI()

client = chromadb.PersistentClient(path="safety_db")
collection = client.get_collection("safety_docs")
model = SentenceTransformer("all-MiniLM-L6-v2")

class Query(BaseModel):
    q: str
    k: int = 3
    mode: str = "rerank"  # or "rerank"

@app.post("/ask")
def ask(query: Query):
    q_emb = model.encode(query.q).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=query.k)

    if query.mode == "rerank":
        reranked = rerank(query.q, results)
        answer = make_answer(query.q, reranked)
        return {"answer": answer, "contexts": reranked, "reranker_used": True}
    else:
        docs = results["documents"][0]
        ids = results["ids"][0]
        scores = 1 - np.array(results["distances"][0])
        contexts = list(zip(ids, docs, scores))
        answer = make_answer(query.q, contexts)
        return {"answer": answer, "contexts": contexts, "reranker_used": False}

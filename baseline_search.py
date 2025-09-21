# baseline_search.py
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="safety_db")
collection = client.get_collection("safety_docs")
model = SentenceTransformer("all-MiniLM-L6-v2")

def baseline_search(query, k=3):
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nResult {i+1}: {doc[:200]}...")
    return results

# Example
print(baseline_search("What are the main causes of industrial accidents?", k=3))
#===================================================================
# import chromadb
# from sentence_transformers import SentenceTransformer

# client = chromadb.PersistentClient(path="safety_db")
# collection = client.get_collection("safety_docs")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Quick search
# query = "industrial accident prevention"
# embedding = model.encode(query).tolist()
# results = collection.query(query_embeddings=[embedding], n_results=3)

# print(f"🔍 Results for: {query}")
# for i, doc in enumerate(results['documents'][0]):
#     print(f"\n📄 Result {i+1}: {doc[:200]}...")
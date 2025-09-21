# # answer.py
# def make_answer(query, reranked, threshold=0.3):
#     if not reranked:
#         return None
    
#     top_id, top_doc, top_score = reranked[0]
#     if top_score < threshold:
#         return None
    
#     # Short extractive answer (first sentence)
#     answer = top_doc.split(".")[0] + "."
#     return answer
# answer.py
def make_answer(query, reranked, threshold=0.1):
    if not reranked:
        return "No relevant information found."
         
    top_id, top_doc, top_score = reranked[0]
    if top_score < threshold:
        return "Found information but relevance score is too low."
         
    # Short extractive answer (first sentence)
    answer = top_doc.split(".")[0] + "."
    return answer
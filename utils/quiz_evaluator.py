from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate(student, reference):
    vec1 = model.encode([student])
    vec2 = model.encode([reference])
    sim = cosine_similarity(vec1, vec2)[0][0]
    score = round(sim * 10, 2)
    feedback = "Very close!" if score > 8 else "Needs improvement."
    return score, feedback
# rag_utils.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage for demo (replace with DB in production)
text_chunks = []
embeddings = []

def chunk_text(text, chunk_size=500):
    """Split text into semantic chunks"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def store_embeddings(chunks):
    """Store text chunks and their embeddings"""
    global text_chunks, embeddings
    text_chunks = chunks
    embeddings = embedder.encode(chunks)

def get_similar_chunks(question, top_k=3):
    """Retrieve relevant chunks using semantic search"""
    if not text_chunks:
        return []
        
    question_embedding = embedder.encode(question)
    similarities = cosine_similarity(
        [question_embedding],
        embeddings
    )[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [text_chunks[i] for i in top_indices]

def get_full_text():
    """Get all text chunks concatenated"""
    return "\n".join(text_chunks)















# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")
# chunks = []
# index = faiss.IndexFlatL2(384)

# def chunk_text(text, size=300):
#     global chunks
#     chunks = [text[i:i+size] for i in range(0, len(text), size)]
#     return chunks

# def store_embeddings(chunks):
#     vectors = model.encode(chunks)
#     index.add(np.array(vectors))

# def get_similar_chunks(query, k=3):
#     q_vec = model.encode([query])
#     D, I = index.search(np.array(q_vec), k)
#     return [chunks[i] for i in I[0]]

# def get_full_text():
#     return ' '.join(chunks)
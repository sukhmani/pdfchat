from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return model.encode(text)

from sklearn.metrics.pairwise import cosine_similarity

def find_best_match(question, chunks):
    chunk_embeddings = embed_text(chunks)
    question_embedding = embed_text([question])

    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    best_idx = similarities.argmax()
    return chunks[best_idx]


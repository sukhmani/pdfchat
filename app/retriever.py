from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch


retrieval_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
retrieval_model = AutoModel.from_pretrained("distilbert-base-uncased")

def embed_question(question):
    inputs = retrieval_tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = retrieval_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def retrieve_top_chunks(question, chunk_embeddings, chunks, k=3):
    q_embed = embed_question(question)
    scores = cosine_similarity([q_embed], chunk_embeddings)[0]
    top_indices = np.argsort(scores)[-k:][::-1]
    return [chunks[i] for i in top_indices]

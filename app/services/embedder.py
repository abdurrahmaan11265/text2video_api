from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def get_embeddings(prompts: list[str]) -> list[list[float]]:
    embeddings = model.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()
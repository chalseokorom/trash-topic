"""Embedding utilities so sentence-transformers loads only on first call."""
import os
import numpy as np

EMBEDDINGS_CACHE_FILE_PATH = "data/embeddings_cache.npy"


def get_embeddings(docs: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    if os.path.exists(EMBEDDINGS_CACHE_FILE_PATH):
        return np.load(EMBEDDINGS_CACHE_FILE_PATH)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, show_progress_bar=True)
    np.save(EMBEDDINGS_CACHE_FILE_PATH, embeddings)

    return embeddings

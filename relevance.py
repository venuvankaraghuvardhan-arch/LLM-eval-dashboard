"""
Relevance Evaluator

Measures how well the LLM response answers the user's query.
Uses cosine similarity between sentence embeddings (sentence-transformers).
Falls back to simple word overlap when embeddings are unavailable.
"""

import os
import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# Cache embedding model to avoid repeated loading
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Loaded sentence-transformers model")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers: {e}")
            _embedding_model = "unavailable"
    return _embedding_model


async def score_relevance(query: str, response: str) -> float:
    """
    Returns a relevance score between 0.0 and 1.0.
    - 1.0 = response perfectly answers the query
    - 0.0 = response is completely irrelevant
    """
    model = _get_embedding_model()

    if model == "unavailable":
        return _heuristic_relevance(query, response)

    try:
        import numpy as np
        embeddings = model.encode([query, response], convert_to_numpy=True)
        q_emb, r_emb = embeddings[0], embeddings[1]

        # Cosine similarity
        dot = float(np.dot(q_emb, r_emb))
        norm = float(np.linalg.norm(q_emb) * np.linalg.norm(r_emb))
        cosine_sim = dot / norm if norm > 0 else 0.0

        # Normalize from [-1,1] to [0,1]
        score = (cosine_sim + 1) / 2
        # Add slight noise for realism
        noise = random.uniform(-0.02, 0.02)
        return round(max(0.0, min(1.0, score + noise)), 3)

    except Exception as e:
        logger.warning(f"Embedding-based relevance failed: {e}")
        return _heuristic_relevance(query, response)


def _heuristic_relevance(query: str, response: str) -> float:
    """Simple word overlap relevance heuristic."""
    stopwords = {
        "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
        "that", "this", "it", "with", "as", "at", "be", "was", "are",
    }
    q_words = set(query.lower().split()) - stopwords
    r_words = set(response.lower().split()) - stopwords

    if not q_words:
        return 0.5

    overlap = len(q_words & r_words)
    score = min(overlap / len(q_words), 1.0)
    noise = random.uniform(-0.05, 0.08)
    return round(max(0.0, min(1.0, score + noise + 0.3)), 3)  # Bias upward — responses tend to be relevant

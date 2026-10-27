"""Dense embeddings with BAAI bge-small-en-v1.5, run locally through fastembed (ONNX, no API quota).

Why local: the Gemini embedding API has a free-tier daily cap, and one full ingest plus a few evals can use
it up. Local bge-small (384 dims) has no quota, is deterministic, and is the model the team discussed.
Queries get bge's retrieval instruction prefix; documents do not, as bge's model card specifies.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from app.config import Settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, settings: Settings):
        self.model = TextEmbedding(model_name=settings.embedding_model)

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        """task is 'RETRIEVAL_DOCUMENT' for indexing or 'RETRIEVAL_QUERY' for questions."""
        prefix = QUERY_PREFIX if task == "RETRIEVAL_QUERY" else ""
        vectors = np.array(list(self.model.embed([prefix + t for t in texts], batch_size=32)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)

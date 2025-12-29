# ml/training/embeddings.py
"""
Sentence-Transformer embeddings helper with deterministic behaviour where possible.

This wrapper exists so training and inference can share a single encoding path.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class MiniLMEncoder:
    """
    Thin wrapper that caches a SentenceTransformer model and encodes text batches.

    Notes:
    - Uses CPU by default for predictability and CI friendliness.
    - Returns a 2D numpy array of shape (n_samples, embedding_dim).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        """
        Lazy-load the encoder model. This avoids importing and loading weights
        in code paths that do not require embeddings.
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """
        Encode an iterable of strings into a 2D numpy array.

        Args:
            texts: iterable of input strings

        Returns:
            numpy array of embeddings
        """
        model = self._get_model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors)

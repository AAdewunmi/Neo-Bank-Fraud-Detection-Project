# tests/test_embeddings_encoder_unit.py
# import numpy as np

# from ml.training.embeddings import MiniLMEncoder


# class _FakeSentenceTransformer:
#     """
#     Test double that avoids network downloads and heavy model loads.
#     """
#     def __init__(self, model_name: str, device: str = "cpu") -> None:
#         self.model_name = model_name
#         self.device = device

#     def encode(self, texts, convert_to_numpy: bool = True, normalize_embeddings: bool = True):
#         assert convert_to_numpy is True
#         assert normalize_embeddings is True
#         n = len(list(texts))
#         return np.zeros((n, 384), dtype=np.float32)


# def test_minilm_encoder_outputs_vectors_without_downloading(monkeypatch) -> None:
#     import ml.training.embeddings as embeddings_mod

#     monkeypatch.setattr(embeddings_mod, "SentenceTransformer", _FakeSentenceTransformer)

#     enc = MiniLMEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     X = enc.encode(["hello world", "coffee shop"])

#     assert X.ndim == 2
#     assert X.shape[0] == 2
#     assert X.shape[1] == 384

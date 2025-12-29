# tests/test_embeddings_encoder_integration.py
# import os
# import pytest

# from ml.training.embeddings import MiniLMEncoder


# @pytest.mark.ml
# @pytest.mark.slow
# def test_minilm_encoder_outputs_vectors_real_model() -> None:
#     """
#     Integration test. Runs only when explicitly enabled.

#     Local run:
#       RUN_SLOW_ML_TESTS=1 pytest -q -m "ml"

#     CI:
#       wired into the workflow_dispatch job only.
#     """
#     if os.getenv("RUN_SLOW_ML_TESTS") != "1":
#         pytest.skip("Set RUN_SLOW_ML_TESTS=1 to enable slow ML integration tests")

#     enc = MiniLMEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     X = enc.encode(["hello world", "coffee shop"])

#     assert X.ndim == 2
#     assert X.shape[0] == 2
#     assert X.shape[1] > 10

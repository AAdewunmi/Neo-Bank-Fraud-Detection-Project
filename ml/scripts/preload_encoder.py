"""
Preload the sentence-transformers encoder into Hugging Face cache.
Used by Docker build to avoid long first-time downloads at runtime.
"""
from __future__ import annotations
import os


def main() -> None:
    encoder_name = os.environ.get(
        "LEDGERGUARD_ENCODER_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    print(f"[LedgerGuard] Preloading encoder: {encoder_name}")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(encoder_name, device="cpu")
    except Exception as exc:
        print(f"[LedgerGuard] Encoder preload failed: {exc}")
    else:
        print("[LedgerGuard] Encoder preload complete.")


if __name__ == "__main__":
    main()

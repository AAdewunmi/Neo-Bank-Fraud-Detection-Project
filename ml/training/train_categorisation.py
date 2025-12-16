"""
Train a deterministic categorisation baseline (TF-IDF + Logistic Regression).

Usage:
  source .venv/bin/activate
  python ml/training/train_categorisation.py \
    --input data/sample_transactions.csv \
    --target_col category \
    --text_cols merchant description \
    --registry model_registry.json

Output:
- artefacts/cat_lr_<timestamp>.joblib
- model_registry.json updated (categorisation.latest -> version)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.training.utils import schema_hash, load_registry, save_registry


def build_pipeline() -> Pipeline:
    """
    Build a deterministic baseline pipeline.

    Returns:
        Sklearn Pipeline with TF-IDF vectorizer and Logistic Regression classifier.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,  # ensures solver behavior stays stable
                    solver="liblinear",
                ),
            ),
        ]
    )

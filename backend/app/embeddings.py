"""
embeddings.py
-------------

Hugging Face Sentence Transformer embeddings.

Model is intentionally unchanged:
    sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _get_model():

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        config.EMBEDDING_MODEL_NAME,
        token=(
            config.HF_TOKEN
            if config.HF_TOKEN
            else None
        ),
    )


def embed_texts(
    texts: List[str],
) -> np.ndarray:
    """
    Generate normalized embeddings.
    """

    if not texts:

        return np.zeros(
            (
                0,
                config.EMBEDDING_DIM,
            ),
            dtype="float32",
        )

    model = _get_model()

    vectors = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        vectors,
        dtype="float32",
    )


def embed_query(
    query: str,
) -> np.ndarray:

    if not query.strip():

        return np.zeros(
            config.EMBEDDING_DIM,
            dtype="float32",
        )

    return embed_texts(
        [query]
    )[0]
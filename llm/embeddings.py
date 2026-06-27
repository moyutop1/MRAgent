import os
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_LOCAL_EMBEDDING_MODEL = "/autodl-pub/models/bge-large-en-v1.5"
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
LOCAL_EMBEDDING_BATCH_SIZE = int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "32"))


@lru_cache(maxsize=1)
def get_embedding_model():
    model_path = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
    if not os.path.exists(model_path):
        raise RuntimeError(
            "Local embedding model not found. Download it first, or set "
            f"LOCAL_EMBEDDING_MODEL. Expected path: {model_path}"
        )
    device = os.getenv("LOCAL_EMBEDDING_DEVICE")
    return SentenceTransformer(model_path, device=device)


def get_local_embedding(texts: Sequence[str], batch_size: int = None) -> np.ndarray:
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list/tuple of strings")
    clean_texts = [("" if t is None else str(t)).replace("\n", " ").strip() for t in texts]
    if not clean_texts:
        return np.empty((0, 0), dtype=np.float32)

    model = get_embedding_model()
    embeddings = model.encode(
        clean_texts,
        batch_size=batch_size or LOCAL_EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32, copy=False)

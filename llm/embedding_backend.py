import os
from functools import lru_cache
from threading import Lock

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "/autodl-pub/models/bge-large-en-v1.5")
BATCH_SIZE = int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "32"))
MODEL_LOCK = Lock()


@lru_cache(maxsize=1)
def local_model():
    return SentenceTransformer(LOCAL_MODEL, device=os.getenv("LOCAL_EMBEDDING_DEVICE"))


def get_embedding(texts):
    texts = [str(text).replace("\n", " ").strip() for text in texts]
    if PROVIDER == "local":
        with MODEL_LOCK:
            return local_model().encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

    if PROVIDER == "openrouter":
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large")
    else:
        client = OpenAI(
            api_key=os.getenv("OFOX_API_KEY"),
            base_url=os.getenv("OFOX_BASE_URL"),
        )
        model = os.getenv("OFOX_EMBEDDING_MODEL", "text-embedding-3-large")
    vectors = np.asarray([
        item.embedding for item in client.embeddings.create(model=model, input=texts).data
    ], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

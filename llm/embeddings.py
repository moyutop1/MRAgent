import os
import time
from functools import lru_cache
from typing import Sequence

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from openai._exceptions import OpenAIError, RateLimitError, APIStatusError
from sentence_transformers import SentenceTransformer

load_dotenv()

DEFAULT_LOCAL_EMBEDDING_MODEL = "/autodl-pub/models/bge-large-en-v1.5"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
LOCAL_EMBEDDING_BATCH_SIZE = int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "32"))
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-large")
OPENROUTER_EMBEDDING_BATCH_SIZE = int(os.getenv("OPENROUTER_EMBEDDING_BATCH_SIZE", "96"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_BATCH_SIZE = OPENROUTER_EMBEDDING_BATCH_SIZE if EMBEDDING_PROVIDER == "openrouter" else LOCAL_EMBEDDING_BATCH_SIZE


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


def get_openrouter_embedding(
        texts: Sequence[str],
        batch_size: int = None,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
) -> np.ndarray:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is empty. Set it in .env for text-embedding-3-large.")
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list/tuple of strings")
    clean_texts = [("" if t is None else str(t)).replace("\n", " ").strip() for t in texts]
    if not clean_texts:
        return np.empty((0, 0), dtype=np.float32)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, timeout=60.0)
    out = []
    step = batch_size or OPENROUTER_EMBEDDING_BATCH_SIZE
    for start in range(0, len(clean_texts), step):
        batch = clean_texts[start:start + step]
        attempt = 0
        backoff = initial_backoff
        while True:
            try:
                resp = client.embeddings.create(model=OPENROUTER_EMBEDDING_MODEL, input=batch)
                out.extend(item.embedding for item in resp.data)
                break
            except (RateLimitError, APIStatusError, OpenAIError, TimeoutError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"OpenRouter embedding request failed after {max_retries} retries "
                        f"at batch [{start}:{start + len(batch)}]: {e}"
                    ) from e
                time.sleep(backoff)
                backoff *= 2.0
    return np.asarray(out, dtype=np.float32)


def get_embedding(texts: Sequence[str], batch_size: int = None) -> np.ndarray:
    if EMBEDDING_PROVIDER == "openrouter":
        return get_openrouter_embedding(texts, batch_size=batch_size)
    return get_local_embedding(texts, batch_size=batch_size)

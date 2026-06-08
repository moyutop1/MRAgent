# global_methods.py

import os
import time
from typing import List, Sequence, Optional

# OpenAI Python SDK v1.x
# pip install openai>=1.0.0
from openai import OpenAI
from openai._exceptions import OpenAIError, RateLimitError, APIStatusError


# embedding via OpenRouter (proxies /embeddings; text-embedding-3-large returns 3072-d)
from dotenv import load_dotenv
load_dotenv()  # read API key from .env
EMBED_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_BASE_URL = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = EMBED_API_KEY or ""  # for set_openai_key() validation

# optional helper
def set_openai_key(key_env: str = "OPENAI_API_KEY") -> None:
    """
    Read the OpenAI key from the environment; you may also set os.environ[key_env] beforehand.
    """

    api_key = os.getenv(key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{key_env} is empty. Please export your OpenAI API key, e.g. "
            f'export {key_env}="sk-..."'
        )


def get_openai_embedding(
    texts: Sequence[str],
    model: str = "text-embedding-3-large",
    *,
    batch_size: int = 96,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    timeout: Optional[float] = 60.0,
) -> List[List[float]]:
    """
    Embed a batch of texts; returns a 2D array aligned 1:1 with the input (list of list of float).

    Args:
    ----
    texts : Sequence[str]
        List of texts to encode; sent in batches, order preserved.
    model : str
        OpenAI embedding model name, common choices:
        - "text-embedding-3-small"   # 1536-d, cheaper
        - "text-embedding-3-large"   # 3072-d, higher quality
    batch_size : int
        Items per request batch; tune to your rate/memory/timeout.
    max_retries : int
        Max retries per batch (exponential backoff).
    initial_backoff : float
        Initial retry wait in seconds, doubled each time.
    timeout : Optional[float]
        Per-request HTTP timeout in seconds; None means no limit.

    Returns:
    ----
    List[List[float]]
        Embeddings of shape (len(texts), dim).
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list/tuple of strings")

    # preprocess: replace newlines with spaces to avoid length/format issues
    clean_texts = [("" if t is None else str(t)).replace("\n", " ").strip() for t in texts]

    client = OpenAI(timeout=timeout, api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)  # embedding via OpenRouter

    embeddings: List[List[float]] = []
    n = len(clean_texts)
    if n == 0:
        return embeddings

    # request in batches, preserving order
    for start in range(0, n, batch_size):
        batch = clean_texts[start : start + batch_size]

        # exponential-backoff retry
        attempt = 0
        backoff = initial_backoff
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                # resp.data order matches the input
                for item in resp.data:
                    embeddings.append(item.embedding)
                break  # success; exit the retry loop

            except (RateLimitError, APIStatusError, OpenAIError, TimeoutError) as e:
                attempt += 1
                if attempt > max_retries:
                    # return partial results and the error position for the caller to debug
                    raise RuntimeError(
                        f"OpenAI embedding request failed after {max_retries} retries "
                        f"at batch [{start}:{start+len(batch)}]: {e}"
                    ) from e
                # wait then retry
                time.sleep(backoff)
                backoff *= 2.0  # exponential backoff

    # assert length match (defensive)
    if len(embeddings) != n:
        raise RuntimeError(
            f"Embedding count mismatch: got {len(embeddings)} for {n} inputs."
        )
    return embeddings

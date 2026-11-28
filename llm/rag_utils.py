from llm.embedding_backend import get_embedding


def get_embeddings(inputs, mode="context"):
    return get_embedding(inputs)

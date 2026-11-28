import json
import pickle

import numpy as np

from llm.rag_utils import get_embeddings


def embed_sample(qa_list, rewrite_path, embedding_path):
    with open(rewrite_path, encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]

    sentence_ids = []
    sentence_texts = []
    topic_ids = []
    topic_texts = []
    for session_index, record in enumerate(records, start=1):
        session = next(iter(record.values()))
        if session is None:
            continue
        for sentence in session["sentence"]:
            sentence_ids.append(sentence["id"])
            sentence_texts.append(sentence["text"])
        for topic_id, summary in session["topics"].items():
            topic_ids.append(f"D{session_index}:{topic_id}")
            topic_texts.append(summary)

    question_embeddings = get_embeddings([qa["question"] for qa in qa_list])
    dimension = question_embeddings.shape[1]
    sentence_embeddings = (
        get_embeddings(sentence_texts)
        if sentence_texts else np.empty((0, dimension), dtype=np.float32)
    )
    topic_embeddings = (
        get_embeddings(topic_texts)
        if topic_texts else np.empty((0, dimension), dtype=np.float32)
    )
    with open(embedding_path, "wb") as output:
        pickle.dump({
            "embeddings": sentence_embeddings,
            "topic": topic_embeddings,
            "sentence_id": sentence_ids,
            "topic_list": topic_ids,
            "question_embeddings": question_embeddings,
        }, output)

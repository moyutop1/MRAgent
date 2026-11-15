import pickle
from llm.rag_utils import get_embeddings
import numpy as np


def embed_question(qa_list):
    question_list = []
    for qa in qa_list:
        question = qa.get("question")
        question_list.append(question)
    embeddings = get_embeddings(question_list, 'query')
    return embeddings

def embed_sample(qa_list, FILE_EMBEDDING):
    question_embeddings = embed_question(qa_list)
    embedding_dim = question_embeddings.shape[1] if len(question_embeddings.shape) > 1 else 0

    # Child retrieval embeddings are built once from tag + rewrite_content by
    # MemoryController. This cache only needs the raw-question embeddings.
    database = {'embeddings': np.empty((0, embedding_dim)),
                'topic': np.empty((0, embedding_dim)),
                'sentence_id': [],
                'topic_list': [],
                'question_embeddings': question_embeddings}

    with open(FILE_EMBEDDING, 'wb') as f:
        pickle.dump(database, f)



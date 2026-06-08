import json, pickle
from llm.rag_utils import get_embeddings
import numpy as np
from pathlib import Path

def embed_session(sentence_list):
    embeddings = get_embeddings(sentence_list, "context")
    assert embeddings.shape[0] == len(sentence_list), "Lengths of embeddings and dialogs do not match"
    return embeddings


def embed_question(qa_list):
    question_list = []
    for qa in qa_list:
        question = qa.get("question")
        question_list.append(question)
    embeddings = get_embeddings(question_list, 'query')
    return embeddings

def embed_sample(qa_list, rewrite_path, FILE_EMBEDDING):

    file_name = rewrite_path  # "result_rewrite.json"
    record_rewrite = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            record_rewrite.append(record)

    sample_embedding_list = []
    topic_embedding_list = []
    sentence_id_list = []
    topic_id_list = []

    for j in range(len(record_rewrite)):
        # a rewrite record key may be session_j or session_first-session_last,
        # take the record's single value (equivalent to .get(session_{j+1}) for plain-key files, no breakage).
        # note: topic_id still uses f"D{j+1}:" (record index), consistent with store_event_new's session_id=i+1.
        _sd = record_rewrite[j]
        session_data = next(iter(_sd.values())) if _sd else None
        if session_data is None:
            continue
        sentences = session_data.get("sentence")
        topics = session_data.get("topics")
        sentence_list = []
        for s in sentences:
            sentence_list.append(s.get("text"))
            sentence_id_list.append(s.get("id"))
        # skip empty sessions (no sentences) entirely: no sentence_id/topic_id, no embedding,
        # to avoid embed_session([]) errors and misalignment with the store side.
        if len(sentence_list) == 0:
            continue
        topic_list = []
        if topics is not None:
            for t,text in topics.items():
                topic_list.append(text)
                topic_id_list.append(f"D{j + 1}:" + t)
        session_embedding = embed_session(sentence_list)
        sample_embedding_list.append(session_embedding)
        if len(topic_list) != 0:
            topic_embedding = embed_session(topic_list)
            topic_embedding_list.append(topic_embedding)
    sample_embedding_con = np.vstack(sample_embedding_list)
    topic_embedding_con = np.vstack(topic_embedding_list)
    question_embeddings = embed_question(qa_list)

    database = {'embeddings': sample_embedding_con,
                'topic': topic_embedding_con,
                'sentence_id': sentence_id_list,
                'topic_list': topic_id_list,
                'question_embeddings': question_embeddings}

    with open(FILE_EMBEDDING, 'wb') as f:
        pickle.dump(database, f)



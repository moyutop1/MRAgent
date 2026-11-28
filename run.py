import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

from agent.agent import Agent
from common import config
from common.logging_utils import per_sample_log
from data.embed_rewrite import embed_sample
from data.get_data import get_data
from llm.controller import LLM
from memory.controller import MemoryController
from memory.system import MemorySystem
from utility.evaluator import get_utility_evaluator


logger = logging.getLogger(__name__)


def load_embeddings(path):
    with open(path, "rb") as source:
        database = pickle.load(source)
    event_embeddings = {
        event_id: database["embeddings"][index]
        for index, event_id in enumerate(database["sentence_id"])
    }
    return (
        event_embeddings,
        database["question_embeddings"],
        database["topic_list"],
        database["topic"],
    )


def answer_questions(
    dataset, questions, sample_id, memory, result_path, embeddings, utility,
):
    indexed = [
        (index, qa) for index, qa in enumerate(questions)
        if str(qa.get("category")) not in config.EXCLUDED_CATEGORIES
    ]
    if config.MAX_QUESTIONS:
        indexed = indexed[:config.MAX_QUESTIONS]

    def answer(item):
        index, qa = item
        llm = LLM()
        agent = Agent(llm, memory, MemoryController(memory), utility)
        question = Agent.question_format(dataset, qa)
        current_date = None
        if dataset == "LM" and qa.get("question_date"):
            current_date = qa["question_date"].split(" ")[0].replace("/", "-")
        prediction, origins = agent.answer_question(
            question,
            qa.get("category"),
            embeddings[index],
            lm_current_date=current_date,
        )
        return index, {
            "answer": qa.get("answer"),
            "prediction": prediction,
            "category": qa.get("category"),
            "evidence": qa.get("evidence"),
            "question": qa.get("question"),
            "prediction_context": origins,
            "sample": sample_id,
            "retrieval": agent.last_retrieval,
        }

    with ThreadPoolExecutor(max_workers=config.QUESTION_WORKERS) as executor:
        results = sorted(executor.map(answer, indexed), key=lambda item: item[0])
    with open(result_path, "w", encoding="utf-8") as output:
        for _, result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    conversations, questions, _, raw_texts = get_data(config.dataset, config.datapath)
    for sample_id, sessions in conversations.items():
        if config.dataset == "locomo" and config.sample_id is not None:
            if int(sample_id.split("-")[1]) != config.sample_id:
                continue
        if config.dataset == "LM":
            category = questions[sample_id][0]["category"]
            categories = [
                "multi-session", "single-session-user", "temporal-reasoning",
                "single-session-preference", "knowledge-update", "single-session-assistant",
            ]
            if category != categories[config.ca]:
                continue

        with per_sample_log(sample_id=sample_id, dataset=config.dataset):
            llm = LLM()
            memory = MemorySystem()
            utility = None if config.DUMP_CANDIDATES else get_utility_evaluator()
            agent = Agent(llm, memory, MemoryController(memory), utility)
            rewrite_path = config.rewrite_template.format(dataset=config.dataset, sample_id=sample_id)
            keyword_path = config.keyword_template.format(dataset=config.dataset, sample_id=sample_id)
            embedding_path = config.embedding_template.format(dataset=config.dataset, sample_id=sample_id)
            result_path = config.result_template.format(dataset=config.dataset, sample_id=sample_id)

            if not os.path.exists(rewrite_path):
                agent.rewrite_sample(sessions, rewrite_path)
            if not os.path.exists(keyword_path):
                agent.extract_keyword_sample(keyword_path, rewrite_path)
            if not os.path.exists(embedding_path):
                embed_sample(questions[sample_id], rewrite_path, embedding_path)

            event_embeddings, question_embeddings, topic_ids, topic_embeddings = load_embeddings(embedding_path)
            agent.store_raw_text(raw_texts[sample_id], event_embeddings, topic_ids, topic_embeddings)
            agent.store_keyword(keyword_path, rewrite_path)
            answer_questions(
                config.dataset, questions[sample_id], sample_id, memory,
                result_path, question_embeddings, utility,
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

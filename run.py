"""
Main entry: tool-calling QA over a graph-structured episodic memory of conversations.

Pipeline (per sample):
  rewrite (per-session rewrite/sentence extraction) -> embed (sentence/topic vectors) -> extract_keyword (keywords)
  -> store (build the graph memory) -> per question answer_question (coarse K1 -> fine K2 -> tool-calling loop)

Usage:
  python run.py --data locomo --model gemini --file <tag> [--sample 42]
  python run.py --data LM --model gemini --file <tag> --ca {0|1|2}   # LM's three categories
"""

from agent.tools import TOOLS
import os
import json
import re
from llm.controller import LLM
from memory.controller import MemoryController
from memory.system import MemorySystem
from agent.agent import Agent
from common import config
from pathlib import Path
from data.get_data import get_data
import numpy as np
import logging
from common.logging_utils import per_sample_log
from data.embed_rewrite import embed_sample

logger = logging.getLogger(__name__)


def _select_question_rows(question_list, sample_id):
    rows = list(enumerate(question_list[sample_id], start=1))
    if config.EXCLUDED_CATEGORIES:
        before = len(rows)
        rows = [
            (question_index, qa)
            for question_index, qa in rows
            if str(qa.get("category")) not in config.EXCLUDED_CATEGORIES
        ]
        logger.info(
            f"Excluded categories {sorted(config.EXCLUDED_CATEGORIES)} for {sample_id}: "
            f"kept {len(rows)}/{before} questions."
        )
    if config.MAX_QUESTIONS is not None:
        rows = rows[:config.MAX_QUESTIONS]
        logger.info(f"Limiting {sample_id} to first {len(rows)} selected questions.")
    return rows

def get_question(dataset, agent, question_list, sample_id, memory, result_path, question_embeddings=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logger.info(f"---------------{sample_id}-------------------")

    question_rows = _select_question_rows(question_list, sample_id)
    memory_system = agent.memory  # shared read-only after store_raw_text / store_keyword

    # resumable: use the line count of result_path as the cursor, skip already-done questions
    done_count = 0
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as _f:
            done_count = sum(1 for line in _f if line.strip())
    if done_count >= len(question_rows):
        logger.info(f"All {len(question_rows)} questions already done for {sample_id}, skipping.")
        return
    if done_count > 0:
        logger.info(f"Resuming {sample_id} from question {done_count + 1} (already done: {done_count})")

    remaining = question_rows[done_count:]

    def _run_one_question(i, qa):
        category = qa.get("category")
        question = Agent.question_format(dataset, qa)
        evidence_labels = qa.get("evidence")

        # each thread gets its own LLM + MemoryController + Agent
        q_llm = LLM()
        q_mc = MemoryController(memory_system, q_llm)
        q_agent = Agent(q_llm, memory_system, q_mc)

        # For LM temporal questions, inject question_date as the current_date anchor
        # (leave question_time empty so retrieval keeps the main path + navigation).
        override_question_time = None
        lm_current_date = None
        if dataset == "LM" and category == "temporal-reasoning":
            qdr = qa.get("question_date")  # "2023/04/01 (Sat) 08:09"
            if qdr:
                lm_current_date = qdr.split(" ")[0].replace("/", "-")  # "2023-04-01"
        try:
            question_emb = question_embeddings[i - 1]
            results, evidence_support = q_agent.answer_question(
                question, category, question_emb, override_question_time, lm_current_date)
        except Exception as e:
            logger.error(f"question{i} failed: {e}", exc_info=True)
            return i, {
                "answer": qa.get("answer"), "prediction": "ERROR", "category": category,
                "evidence": evidence_labels, "question": qa.get("question"),
                "prediction_context": [], "sample": sample_id,
                "question_index": i,
            }

        evaluation = {
            "answer": qa.get("answer"), "prediction": results, "category": category,
            "evidence": evidence_labels, "question": qa.get("question"),
            "prediction_context": evidence_support, "sample": sample_id,
            "question_index": i,
        }
        return i, evaluation

    # multithreaded execution: store results in a dict by index i, then read in order when writing
    results_dict: dict = {}

    with ThreadPoolExecutor(max_workers=config.QUESTION_WORKERS) as executor:
        future_to_i = {executor.submit(_run_one_question, i, qa): i for i, qa in remaining}
        for fut in as_completed(future_to_i):
            i = future_to_i[fut]
            try:
                results_dict[i] = fut.result()
            except Exception as e:
                logger.error(f"question{i} future raised: {e}", exc_info=True)
                results_dict[i] = None  # mark failure; skip when writing

    # write results in submission order to keep file line order consistent with question numbers
    for i, qa in remaining:
        result_tuple = results_dict.get(i)
        if result_tuple is None:
            logger.error(f"question{i} has no result, skipping write.")
            continue

        i, evaluation = result_tuple
        logger.info(f"---------------question{i}-------------------")
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evaluation, ensure_ascii=False, default=list) + "\n")


def _normalize_evidence_ids(items):
    out = []
    for item in items or []:
        for m in re.findall(r"D\d+:\d+(?:-\d+)?", str(item)):
            out.append(m.split("-", 1)[0])
    seen = set()
    deduped = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _retrieval_metrics(gold_evidence, retrieved_origins):
    gold = _normalize_evidence_ids(gold_evidence)
    retrieved = _normalize_evidence_ids(retrieved_origins)
    if not gold:
        return {
            "gold_evidence_norm": gold,
            "retrieved_origins_norm": retrieved,
            "hit": None,
            "recall": None,
            "exact_cover": None,
            "mrr": None,
        }
    gold_set = set(gold)
    retrieved_set = set(retrieved)
    covered = gold_set & retrieved_set
    first_rank = None
    for idx, item in enumerate(retrieved, start=1):
        if item in gold_set:
            first_rank = idx
            break
    return {
        "gold_evidence_norm": gold,
        "retrieved_origins_norm": retrieved,
        "hit": 1 if covered else 0,
        "recall": len(covered) / len(gold_set),
        "exact_cover": 1 if gold_set.issubset(retrieved_set) else 0,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
    }


def get_question_retrieval(dataset, agent, question_list, sample_id, result_path, question_embeddings=None):
    logger.info(f"---------------retrieval-only {sample_id}-------------------")
    question_rows = _select_question_rows(question_list, sample_id)

    done_count = 0
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as _f:
            done_count = sum(1 for line in _f if line.strip())
    if done_count >= len(question_rows):
        logger.info(f"All {len(question_rows)} retrieval rows already done for {sample_id}, skipping.")
        return
    if done_count > 0:
        logger.info(f"Resuming retrieval {sample_id} from question {done_count + 1} (already done: {done_count})")

    metric_rows = []
    for i, qa in question_rows[done_count:]:
        category = qa.get("category")
        question = Agent.question_format(dataset, qa)
        override_question_time = None
        lm_current_date = None
        if dataset == "LM" and category == "temporal-reasoning":
            qdr = qa.get("question_date")
            if qdr:
                lm_current_date = qdr.split(" ")[0].replace("/", "-")
        try:
            question_emb = question_embeddings[i - 1]
            retrieval = agent.retrieve_question_evidence(
                question, category, question_emb, override_question_time, lm_current_date)
            gold_memory_diagnostics = None
            if retrieval.get("mode") == "eaes":
                gold_memory_diagnostics = agent.diagnose_eaes_gold_memories(
                    qa.get("evidence"), retrieval, question_emb)
            metrics = _retrieval_metrics(qa.get("evidence"), retrieval.get("retrieved_origins"))
            graph_metrics = _retrieval_metrics(qa.get("evidence"), retrieval.get("graph_origins"))
            dense_metrics = _retrieval_metrics(qa.get("evidence"), retrieval.get("dense_origins"))
            row = {
                "sample": sample_id,
                "question_index": i,
                "question": qa.get("question"),
                "category": category,
                "answer": qa.get("answer"),
                "evidence": qa.get("evidence"),
                **metrics,
                "combined_metrics": metrics,
                "graph_metrics": graph_metrics,
                "dense_metrics": dense_metrics,
                "gold_memory_diagnostics": gold_memory_diagnostics,
                "retrieval": retrieval,
            }
            metric_rows.append(row)
        except Exception as e:
            logger.error(f"retrieval question{i} failed: {e}", exc_info=True)
            row = {
                "sample": sample_id,
                "question_index": i,
                "question": qa.get("question"),
                "category": category,
                "answer": qa.get("answer"),
                "evidence": qa.get("evidence"),
                "error": str(e),
            }

        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=list) + "\n")

    scored = [r for r in metric_rows if r.get("hit") is not None]
    if scored:
        hit = sum(r["hit"] for r in scored) / len(scored)
        recall = sum(r["recall"] for r in scored) / len(scored)
        exact = sum(r["exact_cover"] for r in scored) / len(scored)
        mrr = sum(r["mrr"] for r in scored) / len(scored)
        logger.info(
            f"[retrieval-only] {sample_id}: n={len(scored)} "
            f"Hit@K={hit:.4f} Recall@K={recall:.4f} ExactCover@K={exact:.4f} MRR={mrr:.4f}"
        )






import pickle
def get_conv_embeddings(embedding_path):
    database = pickle.load(open(embedding_path, 'rb'))
    embeddings = database.get("embeddings")
    sentence_id = database.get("sentence_id")
    topic_embeddings = database.get("topic")
    topic_id = database.get("topic_list")
    question_embeddings = database.get("question_embeddings")
    id2emb = {i: embeddings[r] for r, i in enumerate(sentence_id)}
    tid2emb = {i: topic_embeddings[r] for r, i in enumerate(topic_id)}
    return id2emb, question_embeddings, topic_id, topic_embeddings



        # print(top_embs[0])


def main():

    dataset = config.dataset
    datapath = config.datapath
    conversation_list, question_list, raw_conversation_list, raw_text_list = get_data(dataset, datapath)
    i=0
    # category labels per dataset. LM: filter samples by category via --ca (each sample is one category).
    # locomo: reference labels only — a conversation mixes all categories, so run.py runs every question (no --ca filter).
    category_dict = {
        "LM": {
            0: "multi-session",
            1: "single-session-user",
            2: "temporal-reasoning",
            3: "single-session-preference",
            4: "knowledge-update",
            5: "single-session-assistant",
        },
        "locomo": {
            1: "multi-hop",
            2: "temporal",
            3: "open-domain",
            4: "single-hop",
            5: "adversarial",
        },
    }

    for sample_id, sample in conversation_list.items():
        llm = LLM()
        memory_system = MemorySystem()
        memory_controller = MemoryController(memory_system, llm)
        agent = Agent(llm, memory_system, memory_controller)
        # i+=1

        # [LM] for LM, select samples by category (sample_id is hex, cannot use the split('-') scheme)
        if dataset == "LM":
            cat = question_list[sample_id][0].get("category")
            if cat != category_dict["LM"][config.ca]:
                continue
        else:
            if config.sample_id is not None:
                num = int(sample_id.split('-')[1])
                if num != config.sample_id:
                    continue
        with per_sample_log(sample_id=sample_id, dataset=dataset):
            logging.info(f"=== Start processing sample {sample_id} ===")
            rewrite_path = config.rewrite_template.format(dataset=dataset, sample_id=sample_id)
            if not os.path.exists(rewrite_path):
                agent.rewrite_sample(sample, rewrite_path)
            else:
                logging.info(f"Rewrite for sample {sample_id} already exists, skipping.")

            keyword_path = config.keyword_template.format(dataset=dataset, sample_id=sample_id)
            if not os.path.exists(keyword_path):
                agent.extract_keyword_sample(keyword_path, rewrite_path)
            else:
                logging.info(f"Keyword for sample {sample_id} already exists, skipping.")

            embedding_path = config.embedding_template.format(dataset=dataset, sample_id=sample_id)
            if not os.path.exists(embedding_path):
                embed_sample(question_list[sample_id], rewrite_path, embedding_path)
            else:
                logging.info(f"Embedding for sample {sample_id} already exists, skipping.")

            raw_text = raw_text_list[sample_id]

            conv_embeddings, question_embeddings, topic_id_list, topic_embeddings = get_conv_embeddings(embedding_path)
            agent.store_raw_text(raw_text, conv_embeddings, topic_id_list, topic_embeddings)

            agent.store_keyword(keyword_path, rewrite_path)

            result_path = config.result_template.format(dataset=dataset, sample_id=sample_id)
            if config.RETRIEVAL_ONLY:
                get_question_retrieval(dataset, agent, question_list, sample_id, result_path, question_embeddings)
            else:
                get_question(dataset, agent, question_list, sample_id, memory_system, result_path, question_embeddings)

def log_config(config_module, exclude=("API_KEY", "CHAT_BASE_URL", "DEEPSEEK_URL")):
    logging.info("========== CONFIGURATION ==========")
    for name in dir(config_module):
        if not re.match(r'^[A-Z0-9_]+$', name):
            continue
        if any(kw in name.lower() for kw in ["key", "url", "secret", "password"]):
            # logging.info(f"{name} = [HIDDEN]")
            continue
        value = getattr(config_module, name)
        logging.info(f"{name} = {value}")
    logging.info("===================================")


# logging_utils.py
import os
import logging
from contextlib import contextmanager



if __name__ == "__main__":
    # init logging
    global_file_handler = logging.FileHandler(
        f"log/run_{config.DATASET}{config.ADDITIONAL_TK}{config.ADDITIONAL_RE}.log",
        encoding="utf-8"
    )
    stream_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,  # log INFO and above
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[global_file_handler, stream_handler]
    )

    logging.info("=== Program start ===")
    log_config(config)
    # IMPORTANT: right after configuring, detach the 'aggregate log file' handler
    root_logger = logging.getLogger()
    root_logger.removeHandler(global_file_handler)
    global_file_handler.close()
    main()

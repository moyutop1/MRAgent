"""
Main entry: EAES retrieval and answering over reconstructed conversational memory.

Pipeline (per sample):
  rewrite -> embed -> extract_keyword (memory-index hints) -> build EAES notes
  -> parse query plan -> retrieve/rerank evidence -> final reader

Usage:
  python run.py --data locomo --model gemini --file <tag> --eaes [--sample 42]
  python run.py --data LM --model gemini --file <tag> --eaes --ca {0|1|2}
"""

import os
import json
import re
from llm.controller import LLM
from memory.controller import MemoryController
from memory.system import MemorySystem
from agent.agent import Agent
from agent.retrieval import compact_eaes_retrieval
from common import config
from pathlib import Path
from data.get_data import get_data
import numpy as np
import logging
from common.logging_utils import per_sample_log
from data.embed_rewrite import embed_sample
from eval.retrieval_metrics import retrieval_metrics as _retrieval_metrics

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


def _metric_scalars(metrics):
    return {
        key: (metrics or {}).get(key)
        for key in ("hit", "recall", "exact_cover", "mrr")
    }


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
            stage_metrics = {}
            raw_stage_metrics = {}
            for stage, groups in (
                    retrieval.get("stage_origin_groups") or {}).items():
                raw_metrics = _retrieval_metrics(
                    qa.get("evidence"),
                    (retrieval.get("stage_origins") or {}).get(stage),
                    groups,
                )
                raw_stage_metrics[stage] = raw_metrics
                stage_metrics[stage] = _metric_scalars(raw_metrics)
            final_metrics = raw_stage_metrics.get("final_combined") or (
                _retrieval_metrics(
                    qa.get("evidence"),
                    retrieval.get("retrieved_origins"),
                    retrieval.get("retrieved_origin_groups"),
                )
            )
            row = {
                "sample": sample_id,
                "question_index": i,
                "question": qa.get("question"),
                "category": category,
                "answer": qa.get("answer"),
                "evidence": qa.get("evidence"),
                "gold_evidence_norm": final_metrics.get("gold_evidence_norm"),
                "metrics": stage_metrics,
                "gold_memory_diagnostics": gold_memory_diagnostics,
                "retrieval": compact_eaes_retrieval(retrieval),
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

    count_fields = {
        "global_child": "global_pool_k",
        "global_plus_local": "global_plus_local_k",
        "prefilter_child": "prefilter_k",
        "final_child": "final_child_k",
        "selected_parent": "final_parent_k",
        "final_combined": "final_total_k",
    }
    for stage, count_field in count_fields.items():
        valid_rows = [
            row for row in metric_rows
            if row.get("metrics", {}).get(stage, {}).get("hit") is not None
        ]
        if not valid_rows:
            continue
        stage_rows = [row["metrics"][stage] for row in valid_rows]
        k_values = {
            row.get("retrieval", {}).get("counts", {}).get(count_field)
            for row in valid_rows
        }
        k_values.discard(None)
        k_label = str(next(iter(k_values))) if len(k_values) == 1 else "K"
        label = "retrieval-only" if stage == "final_combined" else stage
        logger.info(
            f"[{label}] {sample_id}: n={len(stage_rows)} "
            f"Hit@{k_label}={sum(r['hit'] for r in stage_rows) / len(stage_rows):.4f} "
            f"Recall@{k_label}={sum(r['recall'] for r in stage_rows) / len(stage_rows):.4f} "
            f"ExactCover@{k_label}={sum(r['exact_cover'] for r in stage_rows) / len(stage_rows):.4f} "
            f"MRR@{k_label}={sum(r['mrr'] for r in stage_rows) / len(stage_rows):.4f}"
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
    if not config.EAES_MODE:
        raise ValueError(
            "--eaes is required because the non-EAES keyword retrieval path has been removed."
        )
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
            if config.EAES_MODE and config.SEMANTIC_HIERARCHY:
                memory_controller.prepare_eaes_parent_embeddings()

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

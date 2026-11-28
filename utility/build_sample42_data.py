import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from agent.agent import Agent
from data.get_data import get_data
from llm.controller import LLM
from llm.rag_utils import get_embeddings
from memory.controller import MemoryController
from memory.system import MemorySystem
from utility.evaluator import node_text, temporal_features


def encode(texts, times):
    return np.concatenate((
        get_embeddings(texts), np.asarray(times, dtype=np.float32)
    ), axis=1).astype(np.float32)


def split_questions(indexed_questions, seed):
    groups = defaultdict(list)
    for index, question in indexed_questions:
        groups[str(question["category"])].append(index)
    rng = random.Random(seed)
    splits = {}
    for indices in groups.values():
        rng.shuffle(indices)
        train_end = int(len(indices) * 0.7)
        validation_end = train_end + max(1, int(len(indices) * 0.15))
        for index in indices[:train_end]:
            splits[index] = "train"
        for index in indices[train_end:validation_end]:
            splits[index] = "validation"
        for index in indices[validation_end:]:
            splits[index] = "test"
    return splits


def load_question_keys(cache_path):
    if not cache_path.exists():
        return {}
    with cache_path.open(encoding="utf-8") as source:
        return {
            record["qa_index"]: record["question_keys"]
            for record in (json.loads(line) for line in source if line.strip())
        }


def build_memory(agent, raw_text, rewrite_path, keyword_path):
    with rewrite_path.open(encoding="utf-8") as source:
        rewrites = [json.loads(line) for line in source if line.strip()]
    sentence_ids = []
    sentence_texts = []
    topic_ids = []
    topic_texts = []
    for session_index, record in enumerate(rewrites, start=1):
        session = next(iter(record.values()))
        for sentence in session["sentence"]:
            sentence_ids.append(sentence["id"])
            sentence_texts.append(sentence["text"])
        for topic_id, summary in session["topics"].items():
            topic_ids.append(f"D{session_index}:{topic_id}")
            topic_texts.append(summary)
    sentence_embeddings = get_embeddings(sentence_texts)
    topic_embeddings = get_embeddings(topic_texts)
    agent.store_raw_text(
        raw_text,
        dict(zip(sentence_ids, sentence_embeddings)),
        topic_ids,
        topic_embeddings,
    )
    agent.store_keyword(str(keyword_path), str(rewrite_path))


def write_examples(sample_id, sessions, train_questions):
    gold = {
        origin
        for question in train_questions
        for origin in question.get("evidence") or []
    }
    context = encode(
        ["operation: retain this raw session before rewriting"],
        [temporal_features()],
    )
    selected_features = []
    selected_ids = []
    tensors = []
    metadata = []
    for index, (session_id, session) in enumerate(sessions.items(), start=1):
        lines = [line for line in session.splitlines() if line.strip()]
        conversation_time = lines[0].removeprefix("time:")
        candidate = encode(
            lines,
            [temporal_features(conversation_time, conversation_time)] * len(lines),
        )
        label = int(any(origin.startswith(f"D{index}:") for origin in gold))
        case_id = f"{sample_id}:WRITE:{session_id}"
        tensors.append({
            "operation": 0,
            "candidate": candidate,
            "selected": np.stack(selected_features) if selected_features else np.empty((0, candidate.shape[1]), np.float32),
            "context": context,
            "label": label,
            "split": "train",
            "sample_id": sample_id,
            "case_id": case_id,
        })
        metadata.append({
            "case_id": case_id,
            "operation": "WRITE",
            "split": "train",
            "label": label,
            "candidate_session": session_id,
            "selected_sessions": list(selected_ids),
            "gold_origins_in_session": sorted(
                origin for origin in gold if origin.startswith(f"D{index}:")
            ),
        })
        if label:
            selected_features.append(candidate.mean(axis=0))
            selected_ids.append(session_id)
    return tensors, metadata


def retrieval_examples(
    agent, sample_id, indexed_questions, question_embeddings, question_keys,
    splits, negative_ratio, seed,
):
    rng = random.Random(seed)
    tensors = []
    metadata = []
    coverage = Counter()
    for qa_index, question in indexed_questions:
        query = Agent.question_format("locomo", question)
        candidates, coarse_context, personal_routes = agent.retrieve_candidates(
            query, question_embeddings[qa_index], question_keys[qa_index]
        )
        split = splits[qa_index]
        gold = set(question.get("evidence") or [])
        covered = {candidate["origin"] for candidate in candidates} & gold
        coverage[(split, "questions")] += 1
        coverage[(split, "covered")] += int(covered == gold)
        if not candidates:
            continue

        context_texts = [f"question: {query}"]
        for item in coarse_context["keys_candidates"]:
            context_texts.extend(
                f"cue: {item['key']}; tag: {tag}" for tag in item["tags"]
            )
        context_texts.extend(
            f"topic: {item['topic_id']}; summary: {item['summary']}"
            for item in coarse_context["similar_topic"]
        )
        context = encode(context_texts, [temporal_features()] * len(context_texts))
        candidate_features = encode(
            [node_text(candidate) for candidate in candidates],
            [
                temporal_features(
                    candidate.get("event_time"), candidate.get("conversation_time")
                )
                for candidate in candidates
            ],
        )
        positive_indices = []
        seen_origins = set()
        negative_indices = []
        for index, candidate in enumerate(candidates):
            if candidate["origin"] in gold:
                if candidate["origin"] not in seen_origins:
                    positive_indices.append(index)
                    seen_origins.add(candidate["origin"])
            else:
                negative_indices.append(index)
        question_vector = context[0, :-6]
        negative_indices.sort(
            key=lambda index: float(candidate_features[index, :-6] @ question_vector),
            reverse=True,
        )

        selected_indices = []
        for stage, positive_index in enumerate(positive_indices):
            stage_negatives = negative_indices[
                stage * negative_ratio:(stage + 1) * negative_ratio
            ]
            state_indices = [positive_index] + stage_negatives
            for candidate_index in state_indices:
                label = int(candidate_index == positive_index)
                candidate = candidates[candidate_index]
                case_id = f"{sample_id}:RETRIEVE:{qa_index}:{stage}:{candidate['node_id']}"
                tensors.append({
                    "operation": 1,
                    "candidate": candidate_features[candidate_index][None],
                    "selected": candidate_features[selected_indices],
                    "context": context,
                    "label": label,
                    "split": split,
                    "sample_id": sample_id,
                    "case_id": case_id,
                })
                metadata.append({
                    "case_id": case_id,
                    "operation": "RETRIEVE",
                    "split": split,
                    "label": label,
                    "qa_index": qa_index,
                    "category": question["category"],
                    "question": query,
                    "candidate": candidate,
                    "selected_node_ids": [
                        candidates[index]["node_id"] for index in selected_indices
                    ],
                    "keys_candidates": coarse_context["keys_candidates"],
                    "similar_topic": coarse_context["similar_topic"],
                    "personal_routes": personal_routes,
                    "gold_evidence": sorted(gold),
                })
            selected_indices.append(positive_index)
        if not positive_indices:
            for candidate_index in negative_indices[:negative_ratio]:
                candidate = candidates[candidate_index]
                case_id = f"{sample_id}:RETRIEVE:{qa_index}:0:{candidate['node_id']}"
                tensors.append({
                    "operation": 1,
                    "candidate": candidate_features[candidate_index][None],
                    "selected": np.empty((0, candidate_features.shape[1]), np.float32),
                    "context": context,
                    "label": 0,
                    "split": split,
                    "sample_id": sample_id,
                    "case_id": case_id,
                })
                metadata.append({
                    "case_id": case_id,
                    "operation": "RETRIEVE",
                    "split": split,
                    "label": 0,
                    "qa_index": qa_index,
                    "category": question["category"],
                    "question": query,
                    "candidate": candidate,
                    "selected_node_ids": [],
                    "keys_candidates": coarse_context["keys_candidates"],
                    "similar_topic": coarse_context["similar_topic"],
                    "personal_routes": personal_routes,
                    "gold_evidence": sorted(gold),
                })
    return tensors, metadata, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_locomo.json")
    parser.add_argument("--baseline_dir", default="data/locomo/MRAgent_baseline")
    parser.add_argument("--sample", default="conv-42")
    parser.add_argument("--exclude_categories", default="5")
    parser.add_argument("--output_dir", default="data/utility/sample42")
    parser.add_argument("--negative_ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    conversations, questions, _, raw_texts = get_data("locomo", args.dataset)
    excluded = {value.strip() for value in args.exclude_categories.split(",") if value.strip()}
    indexed_questions = [
        (index, question)
        for index, question in enumerate(questions[args.sample])
        if str(question["category"]) not in excluded
    ]
    splits = split_questions(indexed_questions, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir)
    rewrite_path = baseline_dir / f"{args.sample}_rewrite.jsonl"
    keyword_path = baseline_dir / f"{args.sample}_keyword.jsonl"

    llm = LLM()
    memory = MemorySystem()
    agent = Agent(llm, memory, MemoryController(memory), utility=object())
    build_memory(agent, raw_texts[args.sample], rewrite_path, keyword_path)
    question_embeddings = get_embeddings([
        question["question"] for question in questions[args.sample]
    ])

    question_key_path = output_dir / f"{args.sample}_question_keys.jsonl"
    question_keys = load_question_keys(question_key_path)
    with question_key_path.open("a", encoding="utf-8") as output:
        for qa_index, question in indexed_questions:
            if qa_index not in question_keys:
                query = Agent.question_format("locomo", question)
                question_keys[qa_index] = agent.extract_question_keys(query)
                output.write(json.dumps({
                    "qa_index": qa_index,
                    "question_keys": question_keys[qa_index],
                }, ensure_ascii=False) + "\n")
                output.flush()

    train_questions = [
        question for index, question in indexed_questions if splits[index] == "train"
    ]
    write_tensors, write_metadata = write_examples(
        args.sample, conversations[args.sample], train_questions
    )
    retrieve_tensors, retrieve_metadata, coverage = retrieval_examples(
        agent, args.sample, indexed_questions, question_embeddings,
        question_keys, splits, args.negative_ratio, args.seed,
    )
    tensors = write_tensors + retrieve_tensors
    metadata_path = output_dir / f"{args.sample}_utility_examples.jsonl"
    with metadata_path.open("w", encoding="utf-8") as output:
        for record in write_metadata + retrieve_metadata:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    tensor_path = output_dir / f"{args.sample}_set_utility.pt"
    torch.save(tensors, tensor_path)

    print(f"saved metadata: {metadata_path}")
    print(f"saved tensors: {tensor_path}")
    print(f"records: {len(tensors)} labels={Counter(record['label'] for record in tensors)}")
    print(f"splits: {Counter(record['split'] for record in tensors)}")
    for split in ("train", "validation", "test"):
        total = coverage[(split, "questions")]
        exact = coverage[(split, "covered")]
        print(f"candidate exact coverage {split}: {exact}/{total}={exact / total:.4f}")


if __name__ == "__main__":
    main()

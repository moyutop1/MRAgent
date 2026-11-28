import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch

from data.get_data import get_data
from llm.rag_utils import get_embeddings
from utility.evaluator import node_text, temporal_features


def features(texts, times):
    return np.concatenate((
        get_embeddings(texts), np.asarray(times, dtype=np.float32)
    ), axis=1).astype(np.float32)


def gold_origins(questions):
    return {
        origin
        for question in questions
        for origin in question.get("evidence") or []
    }


def build_write_records(dataset_path):
    conversations, questions, _, _ = get_data("locomo", dataset_path)
    records = []
    for sample_id, sessions in conversations.items():
        gold = gold_origins(questions[sample_id])
        selected = []
        context = features(
            ["operation: retain this raw session before rewriting"],
            [temporal_features()],
        )
        for index, session in enumerate(sessions.values(), start=1):
            lines = [line for line in session.splitlines() if line.strip()]
            conversation_time = lines[0].removeprefix("time:")
            candidate = features(
                lines,
                [temporal_features(conversation_time, conversation_time)] * len(lines),
            )
            label = int(any(origin.startswith(f"D{index}:") for origin in gold))
            records.append({
                "operation": 0,
                "candidate": candidate,
                "selected": np.stack(selected) if selected else np.empty((0, candidate.shape[1]), np.float32),
                "context": context,
                "label": label,
                "sample_id": sample_id,
                "case_id": f"WRITE:D{index}",
            })
            if label:
                selected.append(candidate.mean(axis=0))
    return records


def retrieval_context(question, retrieval):
    texts = [f"question: {question}"]
    for item in retrieval["keys_candidates"]:
        texts.extend(f"cue: {item['key']}; tag: {tag}" for tag in item["tags"])
    texts.extend(
        f"topic: {item['topic_id']}; summary: {item['summary']}"
        for item in retrieval["similar_topic"]
    )
    return features(texts, [temporal_features()] * len(texts))


def build_retrieval_records(result_paths, negative_ratio, seed):
    rng = random.Random(seed)
    records = []
    for path in [match for pattern in result_paths for match in glob.glob(pattern)]:
        with open(path, encoding="utf-8") as source:
            cases = [json.loads(line) for line in source if line.strip()]
        for case_index, case in enumerate(cases):
            retrieval = case["retrieval"]
            candidates = retrieval["candidates"]
            if not candidates:
                continue
            candidate_features = features(
                [node_text(candidate) for candidate in candidates],
                [
                    temporal_features(
                        candidate.get("event_time"), candidate.get("conversation_time")
                    )
                    for candidate in candidates
                ],
            )
            gold = set(case.get("evidence") or [])
            positives = []
            seen_origins = set()
            negatives = []
            for index, candidate in enumerate(candidates):
                if candidate["origin"] in gold and candidate["origin"] not in seen_origins:
                    positives.append((candidate, candidate_features[index]))
                    seen_origins.add(candidate["origin"])
                else:
                    negatives.append((candidate, candidate_features[index]))
            rng.shuffle(negatives)
            selected = []
            context = retrieval_context(retrieval["query"], retrieval)
            staged_negatives = negatives[:max(negative_ratio, len(positives) * negative_ratio)]
            for candidate, feature in positives:
                candidate_feature = feature[None]
                records.append({
                    "operation": 1,
                    "candidate": candidate_feature,
                    "selected": np.stack(selected) if selected else np.empty((0, candidate_feature.shape[1]), np.float32),
                    "context": context,
                    "label": 1,
                    "sample_id": case["sample"],
                    "case_id": f"RETRIEVE:{case_index}:{candidate['node_id']}",
                })
                selected.append(candidate_feature[0])
            for candidate, feature in staged_negatives:
                candidate_feature = feature[None]
                records.append({
                    "operation": 1,
                    "candidate": candidate_feature,
                    "selected": np.stack(selected) if selected else np.empty((0, candidate_feature.shape[1]), np.float32),
                    "context": context,
                    "label": 0,
                    "sample_id": case["sample"],
                    "case_id": f"RETRIEVE:{case_index}:{candidate['node_id']}",
                })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_locomo.json")
    parser.add_argument("--candidate_results", nargs="+", required=True)
    parser.add_argument("--output", default="data/utility/locomo_set_utility.pt")
    parser.add_argument("--negative_ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    records = build_write_records(args.dataset)
    records.extend(build_retrieval_records(
        args.candidate_results, args.negative_ratio, args.seed
    ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output)
    print(f"saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()

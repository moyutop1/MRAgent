"""Offline scorer capability probe (no reader, training, or model download).

Run from the repository: python -m eval.evaluate_selector --cases ...
    --selector_model_path /path/to/Qwen3-Reranker-4B
Outputs one JSON object per case on stdout. Labels are evaluated after scoring;
they are never provided to the scorer. See docs/local_selector.md for the format.
"""

import argparse
import json

from llm.evidence_scorer import get_local_evidence_scorer


def evaluate_case(scorer, case):
    question = case["question"]
    selected = case.get("selected", [])
    candidates = case["candidates"]
    ids = [node.get("memory_id") or node.get("parent_id") for node in candidates]
    if any(not node_id for node_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Probe candidates require unique nonempty IDs")
    before = scorer.score_added_value(question, selected, candidates)
    expanded = case.get("expanded_selected")
    after = (scorer.score_added_value(question, expanded, candidates)
             if expanded is not None else None)
    before_by_id = dict(zip(ids, before["scores"]))
    after_by_id = dict(zip(ids, after["scores"])) if after else {}
    checks = []
    for high, low in case.get("expected_order", []):
        high_score, low_score = before_by_id[high], before_by_id[low]
        checks.append({"type": "order", "higher": high, "lower": low,
                       "passed": (high_score > low_score)
                       if high_score is not None and low_score is not None else None})
    for node_id in case.get("expected_decrease", []):
        if after is None:
            raise ValueError("expected_decrease requires expanded_selected")
        old, new = before_by_id[node_id], after_by_id[node_id]
        checks.append({"type": "decrease", "id": node_id,
                       "passed": (new < old) if old is not None and new is not None else None})
    return {"case_id": case.get("case_id"), "before": before, "after": after,
            "candidate_ids": ids, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="JSONL capability cases")
    # The shared config parses selector_* options, keeping all model settings
    # identical to run.py. Do not turn probe expectation labels into model input.
    parser.add_argument("--selector_model_path", required=True)
    parser.add_argument("--selector_device", default="cuda")
    parser.add_argument("--selector_batch_size", type=int, default=2)
    parser.add_argument("--selector_max_length", type=int, default=8192)
    args = parser.parse_args()
    scorer = get_local_evidence_scorer()
    with open(args.cases, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                print(json.dumps(evaluate_case(scorer, json.loads(line)), ensure_ascii=False))


if __name__ == "__main__":
    main()

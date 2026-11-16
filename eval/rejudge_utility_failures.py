import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ORIGIN_RE = re.compile(r"D\d+:\d+")
WRITE_RE = re.compile(
    r"\[write-utility\] decision=(KEEP|DROP) selected=\d+ candidate=(\{.*\})$"
)

WRITE_CONDITION = {
    "scope": "current session only",
    "task": "strict duplicate detection only",
    "default": "KEEP",
    "drop_only_if": (
        "the candidate describes the same event or state at the same time or "
        "stage and every factual proposition is already present in the kept children"
    ),
    "keep_if": (
        "it adds any new entity, attribute, value, time, stage, reason, result, "
        "opinion, relation, process, or detail; sharing a person or broad topic "
        "is not duplication"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rejudge Gold-related Qwen utility failures with the main model."
    )
    parser.add_argument("--model", required=True, help="Main model alias, e.g. deepseek-pro")
    parser.add_argument("--rewrite", required=True, type=Path)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def origin_ids(value):
    return list(dict.fromkeys(ORIGIN_RE.findall(str(value or ""))))


def load_rewrite_index(path):
    children = {}
    parent_origins = {}
    events = {}
    for record in load_jsonl(path):
        session = next(iter(record.values()))
        conversation_time = session.get("conversation_time")
        for child in session.get("sentence") or []:
            event_id = child.get("id")
            memory_id = "M_" + re.sub(r"[^A-Za-z0-9]+", "_", event_id).strip("_")
            events[event_id] = child
            children[memory_id] = {
                "conversation_time": conversation_time,
                "origin": child.get("origin"),
            }
        for parent in session.get("parent_nodes") or []:
            parent_origins[parent["parent_id"]] = list(dict.fromkeys(
                origin
                for child_id in parent.get("child_ids") or []
                for origin in origin_ids((events.get(child_id) or {}).get("origin"))
            ))
    return children, parent_origins


def related_questions(rows):
    missing_by_origin = defaultdict(list)
    for row in rows:
        for gold in (row.get("gold_memory_diagnostics") or {}).get("gold_origins") or []:
            if gold.get("memory_ids"):
                continue
            missing_by_origin[gold["origin"]].append({
                "sample": row.get("sample"),
                "question_index": row.get("question_index"),
                "question": row.get("question"),
                "answer": row.get("answer"),
            })
    return missing_by_origin


def build_write_cases(log_path, missing_by_origin):
    kept = defaultdict(list)
    cases = []
    with log_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            match = WRITE_RE.search(line.rstrip())
            if not match:
                continue
            decision, raw_candidate = match.groups()
            candidate = json.loads(raw_candidate)
            candidate_origins = origin_ids(candidate.get("origin"))
            session = candidate_origins[0].split(":", 1)[0]
            matched = [
                item
                for origin in candidate_origins
                for item in missing_by_origin.get(origin, [])
            ]
            if decision == "DROP" and matched:
                questions = {
                    (item["sample"], item["question_index"]): item
                    for item in matched
                }
                cases.append({
                    "case_id": f"WRITE:{line_number}:{candidate.get('id')}",
                    "operation": "WRITE",
                    "condition": WRITE_CONDITION,
                    "selected_set": list(kept[session]),
                    "candidate": candidate,
                    "qwen_decision": decision,
                    "gold_questions": list(questions.values()),
                })
            if decision == "KEEP":
                kept[session].append(candidate)
    return cases


def child_payload(candidate, child_index):
    metadata = child_index.get(candidate.get("memory_id"), {})
    return {
        "memory_id": candidate.get("memory_id"),
        "parent_id": candidate.get("parent_id"),
        "conversation_time": metadata.get("conversation_time"),
        "rewrite_content": candidate.get("rewrite_content"),
    }


def build_select_cases(rows, child_index, parent_origins):
    cases = []
    for row in rows:
        metrics = row.get("metrics") or {}
        if (metrics.get("pre_selector") or {}).get("hit") != 1:
            continue
        if (metrics.get("post_selector") or {}).get("hit") != 0:
            continue
        retrieval = row.get("retrieval") or {}
        child_candidates = {}
        for key in ("direct_children", "routed_children"):
            for candidate in retrieval.get(key) or []:
                child_candidates[candidate["memory_id"]] = candidate
        parent_candidates = {
            candidate["parent_id"]: candidate
            for candidate in retrieval.get("direct_parents") or []
        }
        selected = []
        gold_origins = set(row.get("gold_evidence_norm") or [])
        for decision in (retrieval.get("selector") or {}).get("decisions") or []:
            kind = decision.get("kind")
            node_id = decision.get("id")
            if kind == "child":
                candidate = child_candidates.get(node_id)
                if candidate is None:
                    continue
                payload = child_payload(candidate, child_index)
                support = set(origin_ids(candidate.get("origin")))
            else:
                candidate = parent_candidates.get(node_id)
                if candidate is None:
                    continue
                payload = {
                    "parent_id": node_id,
                    "rewrite_content": candidate.get("rewrite_content"),
                }
                support = set(parent_origins.get(node_id, []))
            if (
                decision.get("decision") == "DROP"
                and not decision.get("reason")
                and support.intersection(gold_origins)
            ):
                cases.append({
                    "case_id": (
                        f"SELECT:{row.get('sample')}:{row.get('question_index')}:"
                        f"{kind}:{node_id}"
                    ),
                    "operation": "SELECT",
                    "condition": {
                        "query": row.get("question"),
                        "criterion": "new query-useful information",
                    },
                    "selected_set": list(selected),
                    "candidate": payload,
                    "qwen_decision": "DROP",
                    "gold_questions": [{
                        "sample": row.get("sample"),
                        "question_index": row.get("question_index"),
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                    }],
                })
            if decision.get("decision") == "KEEP":
                selected.append(payload)
    return cases


def main():
    args = parse_args()
    rows = load_jsonl(args.retrieval)
    child_index, parent_origins = load_rewrite_index(args.rewrite)
    write_cases = build_write_cases(args.log, related_questions(rows))
    select_cases = build_select_cases(rows, child_index, parent_origins)
    cases = write_cases + select_cases
    print(
        f"planned cases={len(cases)} "
        f"WRITE={len(write_cases)} SELECT={len(select_cases)}"
    )
    if args.dry_run:
        return

    from common import config
    from llm.controller import LLM
    from llm.unified_utility_judge import UTILITY_SYSTEM_PROMPT

    if not config.API_KEY:
        raise RuntimeError(f"API key is empty for model {args.model}")
    llm = LLM()
    output_path = args.output or args.retrieval.with_name(
        f"{args.retrieval.stem}_utility_rejudge.jsonl"
    )
    main_keep = 0
    with output_path.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            payload = {
                "operation": case["operation"],
                "condition": case["condition"],
                "selected_set": case["selected_set"],
                "candidate": case["candidate"],
            }
            response = llm.chat_with_tool(
                messages=[
                    {"role": "system", "content": UTILITY_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                model=config.MODEL,
                use_tool=False,
                temperature=0.0,
                max_tokens=8,
            )
            answer = str(response.choices[0].message.content or "").strip().upper()
            if not answer.startswith(("KEEP", "DROP")):
                raise ValueError(f"invalid utility decision for {case['case_id']}: {answer!r}")
            decision = "KEEP" if answer.startswith("KEEP") else "DROP"
            main_keep += decision == "KEEP"
            result = {
                key: value for key, value in case.items()
                if key not in {"condition", "selected_set", "candidate"}
            }
            result.update({
                "main_model": config.MODEL,
                "main_decision": decision,
                "selected_count": len(case["selected_set"]),
                "candidate": case["candidate"],
            })
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index}/{len(cases)}] {case['case_id']} -> {decision}")
    print(
        f"saved {output_path} | main KEEP={main_keep}/{len(cases)} "
        f"({main_keep / len(cases):.1%})"
    )


if __name__ == "__main__":
    main()

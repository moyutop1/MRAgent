import argparse
import glob
import json
import numbers
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from eval.evaluation import f1_score
except Exception:
    import string
    from collections import Counter

    def _normalize_answer(text):
        text = str(text or "").replace(",", "").lower()
        text = re.sub(r"\b(a|an|the|and)\b", " ", text)
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        return " ".join(text.split())

    def f1_score(prediction, ground_truth):
        prediction_tokens = _normalize_answer(prediction).split()
        ground_truth_tokens = _normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = num_same / max(1, len(prediction_tokens))
        recall = num_same / max(1, len(ground_truth_tokens))
        return (2 * precision * recall) / (precision + recall)


def parse_args():
    p = argparse.ArgumentParser(
        description="Create wrong-case diagnostics from answer, judge, and optional retrieval-only files."
    )
    p.add_argument("--data", type=str, default="locomo")
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--file", type=str, required=True, help="Answer result tag used by evaluate_reasoning.")
    p.add_argument("--allfile", action="store_true", help="Load all matching answer files.")
    p.add_argument("--sample", type=str, default=None, help="Single sample id when --allfile is not set.")
    p.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated answer categories to include, e.g. 1,3.",
    )
    p.add_argument("--judge_file", type=str, default=None, help="Existing result_judge_*.jsonl file.")
    p.add_argument(
        "--hit20_only",
        action="store_true",
        help="Only include wrong rows whose answer-run judge record has hit_at_20=true.",
    )
    p.add_argument(
        "--retrieval_tag",
        type=str,
        default=None,
        help="Exact retrieval filename tag after '<sample>_result_<model>_'. Defaults to '<file>_retrieval'.",
    )
    p.add_argument(
        "--retrieval_file",
        action="append",
        default=[],
        help="Retrieval jsonl path or glob. Can be passed multiple times.",
    )
    p.add_argument("--f1_threshold", type=float, default=1.0, help="Fallback wrong threshold when no judge row exists.")
    p.add_argument("--topk", type=int, default=8, help="Top retrieval candidates shown per wrong case.")
    p.add_argument("--out_md", type=str, default=None)
    p.add_argument("--out_jsonl", type=str, default=None)
    return p.parse_args()


def read_jsonl(path):
    rows = []
    if not path or not Path(path).exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def result_dirs(data):
    roots = [Path("result") / data, Path("result")]
    out = []
    for root in roots:
        if root.exists() and root not in out:
            out.append(root)
    return out


def answer_files(args):
    names = []
    if args.allfile:
        pattern = f"*_result_{args.model}_{args.file}.jsonl"
        for root in result_dirs(args.data):
            names.extend(sorted(root.glob(pattern)))
    else:
        if not args.sample:
            raise ValueError("Pass --sample when --allfile is not set.")
        filename = f"{args.sample}_result_{args.model}_{args.file}.jsonl"
        for root in result_dirs(args.data):
            fp = root / filename
            if fp.exists():
                names.append(fp)
    return sorted(dict.fromkeys(names))


def load_answers(args):
    rows = []
    for fp in answer_files(args):
        with fp.open(encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_file"] = str(fp)
                row["question_index"] = row.get("question_index", i)
                rows.append(row)
    return rows


def norm_text(text):
    return " ".join(str(text or "").split())


def norm_case_question(text):
    question = norm_text(text)
    for suffix in ("No extra explanations.", "Give reasons with original text."):
        if question.endswith(suffix):
            return question[:-len(suffix)].rstrip()
    return question


def filter_categories(rows, categories):
    selected = {
        item.strip()
        for item in str(categories or "").split(",")
        if item.strip()
    }
    if not selected:
        return rows
    return [row for row in rows if str(row.get("category")) in selected]


def case_keys(row):
    sample = str(row.get("sample") or "")
    keys = []
    question_index = row.get("question_index")
    if question_index is not None and str(question_index).strip():
        keys.append(("index", sample, str(question_index).strip()))
    question = norm_case_question(row.get("question"))
    if question:
        keys.append(("question", sample, question))
    return keys


def case_key(row):
    keys = case_keys(row)
    return keys[0] if keys else ("question", str(row.get("sample") or ""), "")


def add_case(by_key, row):
    for key in case_keys(row):
        by_key[key] = row


def find_case(by_key, row):
    for key in case_keys(row):
        if key in by_key:
            return by_key[key]
    return None


def load_judge(args):
    judge_path = args.judge_file or f"result_judge_{args.data}_{args.model}_{args.file}.jsonl"
    rows = read_jsonl(judge_path)
    by_key = {}
    for row in rows:
        add_case(by_key, row)
    return by_key, judge_path, rows


def retrieval_files(args):
    files = []
    if args.retrieval_file:
        for pattern in args.retrieval_file:
            matches = glob.glob(pattern)
            files.extend(Path(m) for m in matches)
    retrieval_tag = args.retrieval_tag or f"{args.file}_retrieval"
    pattern = f"*_result_{args.model}_{retrieval_tag}.jsonl"
    for root in result_dirs(args.data):
        files.extend(sorted(root.glob(pattern)))
    return sorted(dict.fromkeys(fp for fp in files if fp.exists()))


def load_retrieval(args):
    by_key = {}
    files = retrieval_files(args)
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_file"] = str(fp)
                add_case(by_key, row)
    return by_key, files


def row_f1(row):
    pred = row.get("prediction")
    ref = row.get("answer")
    if isinstance(pred, numbers.Number):
        pred = str(pred)
    if isinstance(ref, numbers.Number):
        ref = str(ref)
    return f1_score(str(pred), str(ref))


def is_wrong(row, judge_row, f1_threshold):
    if str(row.get("prediction")) == "ERROR":
        return True
    if judge_row is not None and "llm_score" in judge_row:
        return int(judge_row.get("llm_score") or 0) == 0
    return row_f1(row) < f1_threshold


def is_hit20(judge_row):
    if judge_row is None or "hit_at_20" not in judge_row:
        return False
    value = judge_row.get("hit_at_20")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return value is True or value == 1


def compact_candidate(cand, rank):
    return {
        "rank": rank,
        "memory_id": cand.get("memory_id"),
        "event_id": cand.get("event_id"),
        "parent_id": cand.get("parent_id"),
        "origin": cand.get("origin"),
        "tag": cand.get("tag"),
        "max_phrase_similarity": cand.get("max_phrase_similarity"),
        "rrf_score": cand.get("rrf_score"),
        "candidate_score": cand.get("candidate_score"),
        "candidate_sources": cand.get("candidate_sources"),
        "prefilter_rank": cand.get("prefilter_rank"),
        "rerank_rank": cand.get("rerank_rank"),
        "rerank_source": cand.get("rerank_source"),
        "rewrite_content": cand.get("rewrite_content"),
    }


def compact_gold_diag(retrieval_row):
    diag = retrieval_row.get("gold_memory_diagnostics") or {}
    out = []
    for gold in diag.get("gold_origins") or []:
        memories = []
        for mem in gold.get("memories") or []:
            memories.append({
                "event_id": mem.get("event_id"),
                "memory_id": mem.get("memory_id"),
                "parent_id": mem.get("parent_id"),
                "indexed": mem.get("indexed"),
                "prefilter_rank": mem.get("prefilter_rank"),
                "rerank_rank": mem.get("rerank_rank"),
                "in_global_child": mem.get("in_global_child"),
                "in_parent_local": mem.get("in_parent_local"),
                "in_merged_pool": mem.get("in_merged_pool"),
                "in_prefilter": mem.get("in_prefilter"),
                "in_final_child": mem.get("in_final_child"),
                "candidate_score": mem.get("candidate_score"),
                "phrase_hits": mem.get("phrase_hits"),
                "drop_reason": mem.get("drop_reason"),
            })
        out.append({
            "origin": gold.get("origin"),
            "covered_by_retrieval": gold.get("covered_by_retrieval"),
            "covered_by_child": gold.get("covered_by_child"),
            "covered_by_parent": gold.get("covered_by_parent"),
            "matched_parent_ids": gold.get("matched_parent_ids"),
            "drop_reason": gold.get("drop_reason"),
            "best_prefilter_rank": gold.get("best_prefilter_rank"),
            "best_rerank_rank": gold.get("best_rerank_rank"),
            "memories": memories,
        })
    return out


def build_case(row, judge_row, retrieval_row, topk):
    f1 = row_f1(row)
    case = {
        "sample": row.get("sample"),
        "question_index": row.get("question_index"),
        "category": row.get("category"),
        "question": row.get("question"),
        "gold_answer": row.get("answer"),
        "gold_evidence": row.get("evidence"),
        "prediction": row.get("prediction"),
        "prediction_context": row.get("prediction_context"),
        "f1": round(f1, 4),
        "judge_score": None if judge_row is None else judge_row.get("llm_score"),
        "hit_at_20": None if judge_row is None else judge_row.get("hit_at_20"),
        "answer_file": row.get("_file"),
    }
    if retrieval_row:
        retrieval = retrieval_row.get("retrieval") or {}
        candidates = retrieval.get("child_candidates") or []
        by_id = {
            candidate.get("memory_id"): candidate for candidate in candidates
        }
        final_candidates = [
            by_id[memory_id]
            for memory_id in retrieval.get("final_child_ids") or []
            if memory_id in by_id
        ]
        case.update({
            "retrieval_file": retrieval_row.get("_file"),
            "retrieval_metrics": retrieval_row.get("metrics"),
            "query_plan": retrieval.get("query_plan"),
            "top_candidates": [
                compact_candidate(cand, rank)
                for rank, cand in enumerate(final_candidates[:topk], start=1)
            ],
            "parent_candidates": retrieval.get("parent_candidates") or [],
            "embedding_top_candidates": [
                compact_candidate(cand, rank)
                for rank, cand in enumerate(candidates[:topk], start=1)
            ],
            "gold_memory_diagnostics": compact_gold_diag(retrieval_row),
        })
    return case


def write_jsonl(path, cases):
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False, default=list) + "\n")


def md_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=list)


def write_md(path, cases):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Wrong Case Diagnostics\n\nTotal wrong cases: {len(cases)}\n")
        for idx, case in enumerate(cases, start=1):
            f.write(
                f"\n## {idx}. {case.get('sample')} Q{case.get('question_index')} "
                f"cat={case.get('category')} judge={case.get('judge_score')} f1={case.get('f1')}\n\n"
            )
            f.write(f"**Question:** {case.get('question')}\n\n")
            f.write(f"**Gold:** {case.get('gold_answer')}\n\n")
            f.write(f"**Gold Evidence:** `{case.get('gold_evidence')}`\n\n")
            f.write(f"**Prediction:** {case.get('prediction')}\n\n")
            f.write(f"**Answer-run Hit@20:** `{case.get('hit_at_20')}`\n\n")
            f.write(f"**Prediction Context:** `{case.get('prediction_context')}`\n\n")
            if "retrieval_metrics" in case:
                f.write(f"**Retrieval Metrics:** `{case.get('retrieval_metrics')}`\n\n")
                f.write("<details><summary>Query Plan</summary>\n\n")
                f.write(f"```json\n{md_json(case.get('query_plan'))}\n```\n\n</details>\n\n")
                f.write("<details><summary>Top Candidates</summary>\n\n")
                f.write(f"```json\n{md_json(case.get('top_candidates'))}\n```\n\n</details>\n\n")
                f.write("<details><summary>Unified Reranker Pool</summary>\n\n")
                f.write(f"```json\n{md_json(case.get('embedding_top_candidates'))}\n```\n\n</details>\n\n")
                f.write("<details><summary>Gold Memory Diagnostics</summary>\n\n")
                f.write(f"```json\n{md_json(case.get('gold_memory_diagnostics'))}\n```\n\n</details>\n\n")
            else:
                f.write("No retrieval-only diagnostics were matched for this case.\n\n")


def main():
    args = parse_args()
    answers = filter_categories(load_answers(args), args.categories)
    judge_by_key, judge_path, judge_rows = load_judge(args)
    retrieval_by_key, retrieval_paths = load_retrieval(args)

    if args.hit20_only and not any("hit_at_20" in row for row in judge_rows):
        raise ValueError(
            "--hit20_only requires a judge file containing hit_at_20. "
            "Run eval/evaluate_reasoning.py first or pass the correct --judge_file."
        )

    wrong_cases = []
    for row in answers:
        judge_row = find_case(judge_by_key, row)
        if not is_wrong(row, judge_row, args.f1_threshold):
            continue
        if args.hit20_only and not is_hit20(judge_row):
            continue
        retrieval_row = find_case(retrieval_by_key, row)
        wrong_cases.append(build_case(row, judge_row, retrieval_row, args.topk))

    out_dir = Path("result") / args.data
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = (
        f"wrong_cases_{args.model}_{args.file}"
        f"{'_hit20' if args.hit20_only else ''}"
    )
    out_jsonl = args.out_jsonl or str(out_dir / f"{out_base}.jsonl")
    out_md = args.out_md or str(out_dir / f"{out_base}.md")
    write_jsonl(out_jsonl, wrong_cases)
    write_md(out_md, wrong_cases)

    print(f"loaded answers={len(answers)}")
    print(f"loaded judge rows={len(judge_rows)} from {judge_path}")
    print(f"loaded retrieval files={len(retrieval_paths)}")
    for fp in retrieval_paths:
        print(f"  retrieval: {fp}")
    print(f"wrong cases={len(wrong_cases)}")
    print(f"wrote {out_jsonl}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

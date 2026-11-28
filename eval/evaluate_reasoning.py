import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path for standalone runs
import json
import argparse
import numbers
from pathlib import Path
from collections import defaultdict
from eval.evaluation import f1_score
from eval.judge import evaluate_llm_judge


def parse_args():
    p = argparse.ArgumentParser(description="F1 + LLM-judge evaluation (dataset-agnostic: locomo / LM).")
    p.add_argument("--data", type=str, default="locomo", help="Dataset name (locomo / LM)")
    p.add_argument("--model", type=str, default="gemini", help="Chat model short name")
    p.add_argument("--file", type=str, default="0", help="Run/experiment tag")
    p.add_argument("--allfile", action="store_true", help="Aggregate all result files for the run")
    p.add_argument("--sample", type=str, default=None, help="Single sample id (used when --allfile is not set)")
    return p.parse_args()


def load_results(data, model, file, allfile, sample):
    """Read result jsonl into a flat list. Pattern matches both locomo (conv-*) and LM (hex) sample ids."""
    root = Path(f"result/{data}")
    if allfile:
        files = sorted(root.glob(f"*_result_{model}_{file}.jsonl"))
    else:
        files = [root / f"{sample}_result_{model}_{file}.jsonl"]
    rows = []
    for fp in files:
        if not fp.exists():
            continue
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def is_adversarial(category):
    # locomo category 5 = adversarial: gold answer is "not mentioned"; scored by string match, not F1/LLM-judge
    return category == 5


def main():
    args = parse_args()
    data = load_results(args.data, args.model, args.file, args.allfile, args.sample)
    print(f"loaded {len(data)} results from result/{args.data}")
    if not data:
        return

    # ---- F1 by category (categories may be int (locomo) or str (LM)) ----
    f1_by_cat = defaultdict(list)
    for r in data:
        prediction, reference, category = r["prediction"], r["answer"], r["category"]
        if is_adversarial(category):
            f1_by_cat[category].append(1 if "Not mentioned" in str(prediction) else 0)
            continue
        if isinstance(prediction, numbers.Number):
            prediction = str(prediction)
        if isinstance(reference, numbers.Number):
            reference = str(reference)
        f1_by_cat[category].append(f1_score(prediction, reference))

    print("\n== F1 by category ==")
    for cat in sorted(f1_by_cat, key=str):
        v = f1_by_cat[cat]
        print(f"  {cat}: n={len(v)} F1={sum(v) / len(v):.4f}")

    # ---- LLM-judge by category (skip adversarial: scored by string match above) ----
    judge_by_cat = defaultdict(list)
    out_path = f"result_judge_{args.data}_{args.model}_{args.file}.jsonl"
    with open(out_path, "a", encoding="utf-8") as of:
        for r in data:
            category = r["category"]
            if is_adversarial(category):
                continue
            score = evaluate_llm_judge(r["question"], r["answer"], r["prediction"])
            judge_by_cat[category].append(score)
            of.write(json.dumps({
                "llm_score": score, "question": r["question"], "prediction": r["prediction"],
                "reference": r["answer"], "category": category, "sample": r.get("sample"),
            }, ensure_ascii=False, default=list) + "\n")

    print("\n== LLM-judge accuracy by category ==")
    total_ok = total = 0
    for cat in sorted(judge_by_cat, key=str):
        v = judge_by_cat[cat]
        total_ok += sum(v); total += len(v)
        print(f"  {cat}: n={len(v)} acc={sum(v) / len(v):.4f}")
    if total:
        print(f"  OVERALL: {total_ok}/{total} = {total_ok / total:.4f}")


if __name__ == "__main__":
    main()

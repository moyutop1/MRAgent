import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate retrieval-only evidence metrics.")
    p.add_argument("--data", type=str, default="locomo", help="Dataset name")
    p.add_argument("--model", type=str, default="ofox", help="Model short name")
    p.add_argument("--file", type=str, default="0", help="Run/experiment tag")
    p.add_argument("--eaes", action="store_true", help="Read EAES retrieval files")
    p.add_argument("--allfile", action="store_true", help="Aggregate all matching samples")
    p.add_argument("--sample", type=str, default=None, help="Single sample id when --allfile is not set")
    return p.parse_args()


def load_rows(args):
    suffix = f"{args.model}_{args.file}{'_eaes' if args.eaes else ''}_retrieval"
    root = Path(f"result/{args.data}")
    if args.allfile:
        files = sorted(root.glob(f"*_result_{suffix}.jsonl"))
    else:
        if not args.sample:
            raise ValueError("Pass --sample when --allfile is not set.")
        files = [root / f"{args.sample}_result_{suffix}.jsonl"]

    rows = []
    for fp in files:
        if not fp.exists():
            continue
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_file"] = str(fp)
                    rows.append(row)
    return rows


def print_group(name, rows, metric_key=None):
    if metric_key:
        scored = [r[metric_key] for r in rows if r.get(metric_key, {}).get("hit") is not None and "error" not in r]
    else:
        scored = [r for r in rows if r.get("hit") is not None and "error" not in r]
    if not scored:
        print(f"{name}: n=0")
        return
    hit = sum(r["hit"] for r in scored) / len(scored)
    recall = sum(r["recall"] for r in scored) / len(scored)
    exact = sum(r["exact_cover"] for r in scored) / len(scored)
    mrr = sum(r["mrr"] for r in scored) / len(scored)
    print(
        f"{name}: n={len(scored)} "
        f"Hit@K={hit:.4f} Recall@K={recall:.4f} "
        f"ExactCover@K={exact:.4f} MRR={mrr:.4f}"
    )


def main():
    args = parse_args()
    rows = load_rows(args)
    print(f"loaded {len(rows)} retrieval rows")
    if not rows:
        return

    print_group("OVERALL", rows)
    if any("prefilter_metrics" in r for r in rows):
        print_group("OVERALL combined prefilter", rows, "prefilter_metrics")
        print_group("OVERALL LLM rerank", rows, "combined_metrics")
    if any("graph_metrics" in r for r in rows):
        print_group("OVERALL graph", rows, "graph_metrics")
        print_group("OVERALL dense", rows, "dense_metrics")
        print_group("OVERALL combined", rows, "combined_metrics")
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row.get("category")].append(row)
    for cat in sorted(by_cat, key=str):
        print_group(f"category={cat}", by_cat[cat])
        if any("prefilter_metrics" in r for r in by_cat[cat]):
            print_group(f"category={cat} combined prefilter", by_cat[cat], "prefilter_metrics")
            print_group(f"category={cat} LLM rerank", by_cat[cat], "combined_metrics")
        if any("graph_metrics" in r for r in by_cat[cat]):
            print_group(f"category={cat} graph", by_cat[cat], "graph_metrics")
            print_group(f"category={cat} dense", by_cat[cat], "dense_metrics")
            print_group(f"category={cat} combined", by_cat[cat], "combined_metrics")

    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"errors={len(errors)}")


if __name__ == "__main__":
    main()

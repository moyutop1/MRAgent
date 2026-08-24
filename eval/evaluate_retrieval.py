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
    p.add_argument(
        "--semantic_score",
        action="store_true",
        help="Read EAES retrieval files produced with --eaes_semantic_score",
    )
    p.add_argument(
        "--semantic_hierarchy",
        action="store_true",
        help="Read retrieval files produced with --semantic_hierarchy",
    )
    p.add_argument("--allfile", action="store_true", help="Aggregate all matching samples")
    p.add_argument("--sample", type=str, default=None, help="Single sample id when --allfile is not set")
    return p.parse_args()


def load_rows(args):
    if args.semantic_score and not args.eaes:
        raise ValueError("--semantic_score requires --eaes.")
    suffix = (
        f"{args.model}_{args.file}"
        f"{'_eaes' if args.eaes else ''}"
        f"{'_semantic' if args.semantic_score else ''}"
        f"{'_hierarchy' if args.semantic_hierarchy else ''}"
        "_retrieval"
    )
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


def print_group(name, rows, metric_key=None, k_field="retrieval_k"):
    if metric_key:
        valid_rows = [
            r for r in rows
            if r.get(metric_key, {}).get("hit") is not None and "error" not in r
        ]
        scored = [r[metric_key] for r in valid_rows]
    else:
        valid_rows = [
            r for r in rows if r.get("hit") is not None and "error" not in r
        ]
        scored = valid_rows
    if not scored:
        print(f"{name}: n=0")
        return
    k_values = {
        int(row[k_field]) for row in valid_rows
        if row.get(k_field) is not None
    }
    k_label = str(next(iter(k_values))) if len(k_values) == 1 else "K"
    hit = sum(r["hit"] for r in scored) / len(scored)
    recall = sum(r["recall"] for r in scored) / len(scored)
    exact = sum(r["exact_cover"] for r in scored) / len(scored)
    mrr = sum(r["mrr"] for r in scored) / len(scored)
    print(
        f"{name}: n={len(scored)} "
        f"Hit@{k_label}={hit:.4f} Recall@{k_label}={recall:.4f} "
        f"ExactCover@{k_label}={exact:.4f} MRR@{k_label}={mrr:.4f}"
    )


def print_stage_group(name, rows, stage, count_field):
    valid_rows = [
        row for row in rows
        if row.get("metrics", {}).get(stage, {}).get("hit") is not None
        and "error" not in row
    ]
    if not valid_rows:
        print(f"{name}: n=0")
        return
    scored = [row["metrics"][stage] for row in valid_rows]
    k_values = {
        row.get("retrieval", {}).get("counts", {}).get(count_field)
        for row in valid_rows
    }
    k_values.discard(None)
    k_label = str(next(iter(k_values))) if len(k_values) == 1 else "K"
    print(
        f"{name}: n={len(scored)} "
        f"Hit@{k_label}={sum(r['hit'] for r in scored) / len(scored):.4f} "
        f"Recall@{k_label}={sum(r['recall'] for r in scored) / len(scored):.4f} "
        f"ExactCover@{k_label}={sum(r['exact_cover'] for r in scored) / len(scored):.4f} "
        f"MRR@{k_label}={sum(r['mrr'] for r in scored) / len(scored):.4f}"
    )


def main():
    args = parse_args()
    rows = load_rows(args)
    print(f"loaded {len(rows)} retrieval rows")
    if not rows:
        return

    stages = {
        "global_child": "global_pool_k",
        "global_plus_local": "global_plus_local_k",
        "prefilter_child": "prefilter_k",
        "final_child": "final_child_k",
        "selected_parent": "final_parent_k",
        "final_combined": "final_total_k",
    }
    if any("metrics" in row for row in rows):
        for stage, count_field in stages.items():
            print_stage_group(f"OVERALL {stage}", rows, stage, count_field)
    else:
        print_group("OVERALL", rows)
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row.get("category")].append(row)
    for cat in sorted(by_cat, key=str):
        if any("metrics" in row for row in by_cat[cat]):
            for stage, count_field in stages.items():
                print_stage_group(
                    f"category={cat} {stage}", by_cat[cat], stage, count_field
                )
        else:
            print_group(f"category={cat}", by_cat[cat])

    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"errors={len(errors)}")


if __name__ == "__main__":
    main()

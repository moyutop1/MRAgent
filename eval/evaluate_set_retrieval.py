import argparse
import json
from collections import defaultdict
from pathlib import Path


def metrics(row):
    gold = list(dict.fromkeys(row.get("evidence") or []))
    retrieved = list(dict.fromkeys(row.get("prediction_context") or []))
    positions = [retrieved.index(origin) + 1 for origin in gold if origin in retrieved]
    covered = len(positions)
    return {
        "hit": int(covered > 0),
        "recall": covered / len(gold) if gold else 1.0,
        "exact_cover": int(covered == len(gold)),
        "mrr": 1 / min(positions) if positions else 0.0,
        "k": len(retrieved),
    }


def report(name, rows):
    values = [metrics(row) for row in rows]
    count = len(values)
    print(
        f"{name}: n={count} "
        f"Hit@K={sum(value['hit'] for value in values) / count:.4f} "
        f"Recall@K={sum(value['recall'] for value in values) / count:.4f} "
        f"ExactCover@K={sum(value['exact_cover'] for value in values) / count:.4f} "
        f"MRR@K={sum(value['mrr'] for value in values) / count:.4f} "
        f"MeanK={sum(value['k'] for value in values) / count:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    rows = []
    for pattern in args.paths:
        for path in Path().glob(pattern):
            with path.open(encoding="utf-8") as source:
                rows.extend(json.loads(line) for line in source if line.strip())
    report("OVERALL", rows)
    categories = defaultdict(list)
    for row in rows:
        categories[row.get("category")].append(row)
    for category in sorted(categories, key=str):
        report(f"category={category}", categories[category])


if __name__ == "__main__":
    main()

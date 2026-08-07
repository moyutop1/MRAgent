import re


def normalize_evidence_ids(items):
    out = []
    for item in items or []:
        for match in re.findall(r"D\d+:\d+(?:-\d+)?", str(item)):
            origin = match.split("-", 1)[0]
            if origin not in out:
                out.append(origin)
    return out


def retrieval_metrics(gold_evidence, retrieved_origins, origin_groups=None):
    """Score provenance coverage, using memory-node groups for rank when present."""
    gold = normalize_evidence_ids(gold_evidence)
    normalized_groups = None
    if origin_groups is not None:
        normalized_groups = [
            normalize_evidence_ids(group) for group in origin_groups
        ]
        retrieved = []
        for group in normalized_groups:
            for origin in group:
                if origin not in retrieved:
                    retrieved.append(origin)
    else:
        retrieved = normalize_evidence_ids(retrieved_origins)

    if not gold:
        return {
            "gold_evidence_norm": gold,
            "retrieved_origins_norm": retrieved,
            "hit": None,
            "recall": None,
            "exact_cover": None,
            "mrr": None,
        }

    gold_set = set(gold)
    retrieved_set = set(retrieved)
    covered = gold_set & retrieved_set
    first_rank = None
    if normalized_groups is not None:
        for rank, group in enumerate(normalized_groups, start=1):
            if set(group) & gold_set:
                first_rank = rank
                break
    else:
        for rank, origin in enumerate(retrieved, start=1):
            if origin in gold_set:
                first_rank = rank
                break

    return {
        "gold_evidence_norm": gold,
        "retrieved_origins_norm": retrieved,
        "hit": 1 if covered else 0,
        "recall": len(covered) / len(gold_set),
        "exact_cover": 1 if gold_set.issubset(retrieved_set) else 0,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
    }

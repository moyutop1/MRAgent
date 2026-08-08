from collections import defaultdict

from eval.retrieval_metrics import normalize_evidence_ids


def _context_nodes(value, limit=20):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)[:limit]
    return [value]


def hit_at_20(row):
    """Return complete gold-evidence coverage in the first 20 reader nodes.

    The project reports this complete-coverage event as Hit@20 for the
    conditional answer metric. Multi-origin child/parent nodes remain one
    list item but all origins inside that item contribute to coverage.
    """
    if not isinstance(row, dict):
        return None
    gold = normalize_evidence_ids(row.get("evidence"))
    if not gold:
        return None
    retrieved = normalize_evidence_ids(
        _context_nodes(row.get("prediction_context"), limit=20)
    )
    return 1 if set(gold).issubset(set(retrieved)) else 0


def _empty_stats():
    return {
        "judged": 0,
        "hit20_metric_available": 0,
        "hit20": 0,
        "correct_and_hit20": 0,
        "wrong_and_hit20": 0,
        "probability": None,
    }


def compute_wrong_given_hit20(judged_rows):
    """Compute the LLM-judge error rate conditioned on context Hit@20."""
    overall = _empty_stats()
    by_category = defaultdict(_empty_stats)

    for row in judged_rows or []:
        if row.get("llm_score") is None:
            continue
        category_stats = by_category[row.get("category")]
        overall["judged"] += 1
        category_stats["judged"] += 1

        hit = hit_at_20(row)
        if hit is None:
            continue
        overall["hit20_metric_available"] += 1
        category_stats["hit20_metric_available"] += 1
        if not hit:
            continue

        correct = 1 if int(row.get("llm_score") or 0) == 1 else 0
        wrong = 1 - correct
        overall["hit20"] += 1
        overall["correct_and_hit20"] += correct
        overall["wrong_and_hit20"] += wrong
        category_stats["hit20"] += 1
        category_stats["correct_and_hit20"] += correct
        category_stats["wrong_and_hit20"] += wrong

    for stats in [overall, *by_category.values()]:
        if stats["hit20"]:
            stats["probability"] = (
                stats["wrong_and_hit20"] / stats["hit20"]
            )
    return {"overall": overall, "by_category": dict(by_category)}


def format_wrong_given_hit20(stats):
    lines = ["== P(wrong | Hit@20) by category =="]
    for category in sorted(stats["by_category"], key=str):
        values = stats["by_category"][category]
        probability = values["probability"]
        rendered = "NA" if probability is None else f"{probability:.4f}"
        lines.append(
            f"  {category}: n_hit={values['hit20']} "
            f"correct={values['correct_and_hit20']} "
            f"wrong={values['wrong_and_hit20']} "
            f"P(wrong | Hit@20)={rendered}"
        )
    overall = stats["overall"]
    probability = overall["probability"]
    rendered = "NA" if probability is None else f"{probability:.4f}"
    lines.append(
        f"  OVERALL: {overall['wrong_and_hit20']}/{overall['hit20']} "
        f"= {rendered} (correct={overall['correct_and_hit20']})"
    )
    lines.append(
        "  coverage: "
        f"hit20_metric_available={overall['hit20_metric_available']}/"
        f"{overall['judged']} judged rows"
    )
    return "\n".join(lines)

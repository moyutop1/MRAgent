from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
def extract_json_from_content(text: str):
    import json, re
    t = (text or "").strip()

    # strip ```json ... ``` fences
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M).strip()

    # prefer the JSON after an assistantfinal / final marker
    m = re.search(r"(assistantfinal|final)\s*{", t, flags=re.I)
    if m:
        start = m.end() - 1  # point at '{'
        # match braces with a stack to capture the full JSON block
        depth, i = 0, start
        while i < len(t):
            if t[i] == '{':
                depth += 1
            elif t[i] == '}':
                depth -= 1
                if depth == 0:
                    block = t[start:i + 1]
                    return json.loads(block)
            i += 1
        raise ValueError("Unbalanced braces after assistantfinal/final")

    # fallback: grab the largest brace block in the text (not the last one)
    # still use brace matching to avoid grabbing an inner sub-object
    best = None
    stack = []
    for i, ch in enumerate(t):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            left = stack.pop()
            candidate = t[left:i + 1]
            # pick the longest (more likely the top-level object)
            if best is None or len(candidate) > len(best):
                best = candidate
    if best:
        return json.loads(best)

    raise ValueError(f"No JSON object found. head={t[:300]!r}")


def topk_answers_by_similarity(
        question_emb,
        answers_embs,
        id_list: List[str],
        k: int = 5,
        *,
        similarity: str = "dot",  # "cosine" or "dot"
        answer_texts: Optional[List[str]] = None,  # optional: candidate answer texts
) -> Tuple[List[str], List[float], np.ndarray, Optional[List[str]]]:
    """
    Returns:
      - top_ids:    list of <=k ids (most-to-least relevant, deduped by hyphen prefix)
      - top_scores: aligned with top_ids
      - top_embs:   embedding matrix of shape (<=k, d) aligned with top_ids
      - top_texts:  (optional) list of <=k texts (only if answer_texts is given)
    """
    assert len(id_list) == answers_embs.shape[0], "id_list length must match answers_embs rows!"
    if answer_texts is not None:
        assert len(answer_texts) == answers_embs.shape[0], "answer_texts length must match answers_embs rows!"

    question_emb = question_emb.reshape(-1)

    if similarity == "cosine":
        scores = _cosine_sim(question_emb, answers_embs)  # (N,)
    elif similarity == "dot":
        scores = np.dot(question_emb, answers_embs.T)  # (N,)
    else:
        raise ValueError("similarity must be 'cosine' or 'dot'")

    # dedup by hyphen prefix (keep only the highest-scoring item per prefix)
    # e.g. D1:1-3 and D1:1-7 share prefix D1:1; keep only the top-scoring one
    def base_prefix(x: str) -> str:
        # split once; left part is the prefix
        return x.split('-', 1)[0]

    best_idx_by_prefix = {}   # prefix -> (idx, score)
    for i, (id_str, sc) in enumerate(zip(id_list, scores)):
        prefix = base_prefix(id_str)
        if (prefix not in best_idx_by_prefix) or (sc > best_idx_by_prefix[prefix][1]):
            best_idx_by_prefix[prefix] = (i, sc)

    # collect the indices and scores of all representatives
    dedup_indices = np.array([v[0] for v in best_idx_by_prefix.values()], dtype=int)
    dedup_scores  = np.array([v[1] for v in best_idx_by_prefix.values()], dtype=float)

    # take top-k among the representatives
    if dedup_indices.size == 0:
        # edge case: no candidates
        return [], [], np.empty((0, answers_embs.shape[1])), ([] if answer_texts is not None else None)

    k_eff = min(k, dedup_indices.size)
    top_dedup_idx_unsorted = np.argpartition(-dedup_scores, kth=k_eff - 1)[:k_eff]
    order = np.argsort(dedup_scores[top_dedup_idx_unsorted])[::-1]
    top_dedup_idx = top_dedup_idx_unsorted[order]

    final_indices = dedup_indices[top_dedup_idx]
    final_scores  = dedup_scores[top_dedup_idx]

    top_ids = [id_list[i] for i in final_indices]
    top_scores = final_scores.tolist()
    top_embs = answers_embs[final_indices]

    top_texts = None
    if answer_texts is not None:
        top_texts = [answer_texts[i] for i in final_indices]

    return top_ids, top_scores, top_embs, top_texts

from typing import List, Dict, Optional


import re
import unicodedata
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple


NormalizedKeyword = Tuple[str, FrozenSet[str]]


def normalize_keyword(value: Any) -> NormalizedKeyword:
    """Normalize one keyword while retaining its underscore-delimited tokens."""
    text = str(value or "").lower()
    normalized_chars = []
    for char in text:
        if char in {"/", ":"}:
            normalized_chars.append(" ")
        elif char == "_" or char == "-" or unicodedata.category(char) == "Pd":
            normalized_chars.append("_")
        elif unicodedata.category(char).startswith("P"):
            continue
        else:
            normalized_chars.append(char)
    normalized = re.sub(r"[\s_]+", "_", "".join(normalized_chars)).strip("_")
    tokens = frozenset(token for token in normalized.split("_") if token)
    return normalized, tokens


def normalize_keyword_entries(values: Iterable[Any]) -> List[NormalizedKeyword]:
    entries = []
    seen = set()
    for value in values or []:
        entry = normalize_keyword(value)
        if not entry[0] or entry[0] in seen:
            continue
        seen.add(entry[0])
        entries.append(entry)
    return entries


def append_keyword_if_missing(values: List[Any], value: Any) -> bool:
    """Append a non-empty value unless its normalized exact form already exists."""
    normalized = normalize_keyword(value)[0]
    if not normalized:
        return False
    existing = {normalize_keyword(item)[0] for item in values or []}
    if normalized in existing:
        return False
    values.append(value)
    return True


def keyword_entries_match(left: NormalizedKeyword, right: NormalizedKeyword) -> bool:
    left_text, left_tokens = left
    right_text, right_tokens = right
    if not left_text or not right_text or not left_tokens or not right_tokens:
        return False
    if left_text == right_text:
        return True
    if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
        return True
    union = left_tokens | right_tokens
    return bool(union) and len(left_tokens & right_tokens) / len(union) >= 0.6


def query_keyword_groups(query_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return aligned original/alternative groups, accepting legacy string keywords."""
    if not isinstance(query_plan, dict):
        return []
    raw_groups = query_plan.get("keyword_groups")
    if not isinstance(raw_groups, list):
        raw_groups = query_plan.get("keywords")
    if not isinstance(raw_groups, list):
        raw_groups = [] if raw_groups is None else [raw_groups]

    groups = []
    for item in raw_groups:
        if isinstance(item, dict):
            keyword = str(item.get("keyword") or "").strip()
            alternatives = item.get("alternatives")
            if not isinstance(alternatives, list):
                alternatives = [] if alternatives is None else [alternatives]
            alternatives = [
                str(value).strip() for value in alternatives
                if str(value or "").strip()
            ][:3]
        else:
            keyword = str(item or "").strip()
            alternatives = []
        if keyword:
            groups.append({"keyword": keyword, "alternatives": alternatives})
    return groups


def matched_query_keyword_count(
        query_plan: Dict[str, Any],
        memory_keywords: Iterable[NormalizedKeyword],
) -> int:
    memory_entries = list(memory_keywords or [])
    if not memory_entries:
        return 0
    count = 0
    seen_originals = set()
    for group in query_keyword_groups(query_plan):
        original = normalize_keyword(group["keyword"])
        if not original[0] or original[0] in seen_originals:
            continue
        seen_originals.add(original[0])
        query_entries = normalize_keyword_entries(
            [group["keyword"], *group.get("alternatives", [])]
        )
        if any(
                keyword_entries_match(query_entry, memory_entry)
                for query_entry in query_entries
                for memory_entry in memory_entries
        ):
            count += 1
    return count


def select_keyword_gated_memory_ids(
        keyword_index: Mapping[str, Iterable[NormalizedKeyword]],
        query_plan: Dict[str, Any],
        minimum_matches: int = 2,
) -> Set[str]:
    return {
        memory_id
        for memory_id, memory_keywords in (keyword_index or {}).items()
        if matched_query_keyword_count(query_plan, memory_keywords) >= minimum_matches
    }


def keyword_gate_allowed_ids(
        keyword_index: Mapping[str, Iterable[NormalizedKeyword]],
        query_plan: Dict[str, Any],
        excluded_memory_ids=None,
        minimum_matches: int = 2,
) -> Optional[Set[str]]:
    """Return gated IDs, or None to signal the required full-memory fallback."""
    matched_ids = select_keyword_gated_memory_ids(
        keyword_index,
        query_plan,
        minimum_matches=minimum_matches,
    )
    matched_ids.difference_update(set(excluded_memory_ids or []))
    return matched_ids or None

import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta
from prompts import schema as json_scheme
from prompts.prompts import Prompts
from agent.semantic_segmentation import (
    child_window_turns,
    parse_session_turns,
    plan_child_windows,
)


_MONTH_NAMES = (
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_WEEKDAY_NAMES = {
    "mon": ("Monday", 0), "monday": ("Monday", 0),
    "tue": ("Tuesday", 1), "tuesday": ("Tuesday", 1),
    "wed": ("Wednesday", 2), "wednesday": ("Wednesday", 2),
    "thu": ("Thursday", 3), "thursday": ("Thursday", 3),
    "fri": ("Friday", 4), "friday": ("Friday", 4),
    "sat": ("Saturday", 5), "saturday": ("Saturday", 5),
    "sun": ("Sunday", 6), "sunday": ("Sunday", 6),
}
_DAY_COUNTS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
}
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:last|next)\s+(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|"
    r"Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?|weekend|week)"
    r"\b(?!\s+of\b)|"
    r"\b(?:yesterday|today|tomorrow)\b|"
    r"\b(?:a|an|one|two|three|four|five|six|seven|\d+)\s+days?\s+"
    r"(?:ago|before|after)\b|"
    r"\b(?:last|next)\s+(?:month|year)\b",
    re.IGNORECASE,
)


def origin_ids(origin):
    return re.findall(r"D\d+:\d+", str(origin or ""))


def first_origin(origin):
    ids = origin_ids(origin)
    return ids[0] if ids else str(origin or "")


def normalize_sentence_ids(rewrite_out):
    if not isinstance(rewrite_out, dict):
        return
    sents = rewrite_out.get("sentence")
    if not isinstance(sents, list):
        return
    cnt = defaultdict(int)
    for sentence in sents:
        if not isinstance(sentence, dict) or not sentence.get("origin"):
            continue
        ids = origin_ids(sentence.get("origin"))
        if not ids:
            continue
        primary = ids[0]
        sentence["origin"] = ",".join(ids)
        cnt[primary] += 1
        sentence["id"] = f"{primary}-{cnt[primary]}"


def finalize_child_memory_ids(memories, turns):
    """Assign session-global IDs from each node's earliest supporting turn."""
    ordered_origins = [
        getattr(turn, "origin", None)
        for turn in turns or []
        if getattr(turn, "origin", None)
    ]
    origin_positions = {
        origin: position for position, origin in enumerate(ordered_origins)
    }
    counters = defaultdict(int)
    for index, memory in enumerate(memories or []):
        if not isinstance(memory, dict):
            raise ValueError(f"child memory[{index}] must be an object")
        ids = list(dict.fromkeys(origin_ids(memory.get("origin"))))
        if not ids:
            raise ValueError(f"child memory[{index}] must use at least one origin")
        unknown = [origin for origin in ids if origin not in origin_positions]
        if unknown:
            raise ValueError(
                f"child memory[{index}] uses unknown origins: {unknown!r}"
            )
        ids.sort(key=origin_positions.__getitem__)
        memory["origin"] = ",".join(ids)
        primary = ids[0]
        counters[primary] += 1
        memory["id"] = f"{primary}-{counters[primary]}"
    return list(memories or [])


def inherit_adjacent_question_origins(rewrite_out, turns):
    """Attach an immediately preceding question to its answer memories."""
    if not isinstance(rewrite_out, dict):
        return 0
    sentences = rewrite_out.get("sentence")
    ordered_turns = list(turns or [])
    if not isinstance(sentences, list) or len(ordered_turns) < 2:
        return 0

    positions = {
        getattr(turn, "origin", None): index
        for index, turn in enumerate(ordered_turns)
    }
    changed = 0
    for index, question_turn in enumerate(ordered_turns[:-1]):
        if "?" not in str(getattr(question_turn, "line", "") or ""):
            continue
        question_origin = getattr(question_turn, "origin", None)
        answer_origin = getattr(ordered_turns[index + 1], "origin", None)
        if not question_origin or not answer_origin:
            continue
        for sentence in sentences:
            if not isinstance(sentence, dict):
                continue
            ids = origin_ids(sentence.get("origin"))
            if answer_origin not in ids or question_origin in ids:
                continue
            ids.append(question_origin)
            ids = sorted(
                dict.fromkeys(ids),
                key=lambda origin: positions.get(origin, len(positions)),
            )
            sentence["origin"] = ",".join(ids)
            changed += 1
    return changed


def _compose_child_tag_components(rewrite_out):
    """Validate generated prefix/facet pairs and compose stored child tags."""
    if not isinstance(rewrite_out, dict):
        return False, rewrite_out, "output_structure", (
            "child window rewrite must be a JSON object"
        )
    sentences = rewrite_out.get("sentence")
    if not isinstance(sentences, list):
        return True, deepcopy(rewrite_out), "", ""

    output = deepcopy(rewrite_out)
    for sentence_index, sentence in enumerate(output["sentence"]):
        if not isinstance(sentence, dict):
            continue
        components = sentence.get("tag")
        if not isinstance(components, list):
            return False, rewrite_out, "tag_structure", (
                f"sentence[{sentence_index}].tag must be an array of "
                "prefix/facet objects"
            )
        if not 1 <= len(components) <= 4:
            return False, rewrite_out, "tag_cardinality", (
                f"sentence[{sentence_index}].tag must contain 1-4 unique "
                f"prefix/facet objects; got {len(components)}"
            )

        complete_tags = []
        for tag_index, component in enumerate(components):
            location = f"sentence[{sentence_index}].tag[{tag_index}]"
            if not isinstance(component, dict):
                return False, rewrite_out, "tag_structure", (
                    f"{location} must be an object containing prefix and facet"
                )
            if "prefix" not in component or "facet" not in component:
                return False, rewrite_out, "tag_structure", (
                    f"{location} must contain both prefix and facet"
                )
            if set(component) != {"prefix", "facet"}:
                return False, rewrite_out, "tag_structure", (
                    f"{location} must contain exactly prefix and facet"
                )

            prefix = (
                re.sub(r"\s+", " ", component["prefix"]).strip()
                if isinstance(component["prefix"], str)
                else component["prefix"]
            )
            facet = (
                re.sub(r"\s+", " ", component["facet"]).strip()
                if isinstance(component["facet"], str)
                else component["facet"]
            )
            prefix_ok, prefix_error = (
                json_scheme.check_generated_tag_prefix(prefix)
            )
            if not prefix_ok:
                return False, rewrite_out, "tag_prefix", (
                    f"{location}.prefix {prefix_error}: {component['prefix']!r}"
                )
            facet_ok, facet_error = json_scheme.check_tag_facet(facet)
            if not facet_ok:
                return False, rewrite_out, "tag_facet", (
                    f"{location}.facet {facet_error}: {component['facet']!r}"
                )
            complete_tags.append(f"{prefix}.{facet}")

        normalized_tags = [tag.casefold() for tag in complete_tags]
        if len(normalized_tags) != len(set(normalized_tags)):
            return False, rewrite_out, "tag_duplicate", (
                f"sentence[{sentence_index}].tag values must be unique after "
                "whitespace and case normalization"
            )
        sentence["tag"] = complete_tags

    return True, output, "", ""


_TAG_HEAD_SEMANTIC_PROPERTY_MAP = {
    "activity": "event_action",
    "plan": "event_action",
    "profile": "personal_profile",
    "possession": "personal_profile",
    "relationship": "relation_social",
}

def normalize_rewrite_semantic_properties(rewrite_out):
    """Repair canonical tag heads leaked into ``semantic_properties``."""
    if not isinstance(rewrite_out, dict):
        return 0
    sentences = rewrite_out.get("sentence")
    if not isinstance(sentences, list):
        return 0

    changed = 0
    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        properties = sentence.get("semantic_properties")
        if not isinstance(properties, list):
            continue
        normalized = []
        for value in properties:
            if not isinstance(value, str):
                normalized.append(value)
                continue
            clean_value = value.strip().lower()
            mapped_value = _TAG_HEAD_SEMANTIC_PROPERTY_MAP.get(
                clean_value, clean_value
            )
            if mapped_value != value:
                changed += 1
            if mapped_value in normalized:
                changed += 1
                continue
            normalized.append(mapped_value)
        sentence["semantic_properties"] = normalized
    return changed


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _conversation_time_from_text(text: str):
    m = re.search(r"^\s*time\s*:\s*(.+?)\s*$", str(text or ""), re.MULTILINE)
    if not m:
        return None
    raw_time = m.group(1).strip()
    iso = re.search(r"\b([0-9]{4}-[0-9]{2}-[0-9]{2})\b", raw_time)
    if iso:
        return iso.group(1)
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M%p on %d %B, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(raw_time, fmt).date().isoformat()
        except ValueError:
            pass
    m2 = re.search(r"\bon\s+(\d{1,2}\s+[A-Za-z]+,\s+\d{4})\b", raw_time, re.IGNORECASE)
    if m2:
        try:
            return datetime.strptime(m2.group(1), "%d %B, %Y").date().isoformat()
        except ValueError:
            return None
    return None


def _dialogue_by_origin(text: str):
    dialogue = {}
    for line in str(text or "").splitlines():
        match = re.search(r"\bdia_id\s*:\s*(D\d+:\d+)\b", line, re.IGNORECASE)
        if match:
            dialogue[match.group(1)] = line.strip()
    return dialogue


def _human_date(value: date):
    return f"{value.day} {_MONTH_NAMES[value.month]} {value.year}"


def _shift_month(anchor: date, offset: int):
    month_index = anchor.year * 12 + anchor.month - 1 + offset
    return date(month_index // 12, month_index % 12 + 1, 1)


def _relative_time_rendering(cue: str, conversation_time: str):
    """Return (display text, normalized index start) for supported relative cues."""
    try:
        anchor = date.fromisoformat(str(conversation_time or "")[:10])
    except (TypeError, ValueError):
        return None
    normalized_cue = re.sub(r"\s+", " ", str(cue or "")).strip().lower()
    anchor_text = _human_date(anchor)

    named = re.fullmatch(
        r"(last|next)\s+(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|"
        r"thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?|weekend|week)",
        normalized_cue,
    )
    if named:
        direction, unit = named.groups()
        relation = "before" if direction == "last" else "after"
        if unit in _WEEKDAY_NAMES:
            display_unit, weekday = _WEEKDAY_NAMES[unit]
            if direction == "last":
                days = (anchor.weekday() - weekday) % 7 or 7
                index_start = anchor - timedelta(days=days)
            else:
                days = (weekday - anchor.weekday()) % 7 or 7
                index_start = anchor + timedelta(days=days)
        elif unit == "week":
            display_unit = "week"
            current_week_start = anchor - timedelta(days=anchor.weekday())
            index_start = current_week_start + timedelta(
                days=-7 if direction == "last" else 7)
        else:
            display_unit = "weekend"
            current_week_start = anchor - timedelta(days=anchor.weekday())
            index_start = current_week_start + timedelta(
                days=-2 if direction == "last" else 12)
        return f"the {display_unit} {relation} {anchor_text}", index_start

    exact_day = re.fullmatch(
        r"(a|an|one|two|three|four|five|six|seven|\d+)\s+days?\s+"
        r"(ago|before|after)",
        normalized_cue,
    )
    if exact_day:
        raw_amount, relation = exact_day.groups()
        amount = _DAY_COUNTS.get(raw_amount, int(raw_amount) if raw_amount.isdigit() else None)
        if amount is None:
            return None
        index_start = anchor + timedelta(
            days=amount if relation == "after" else -amount)
        return _human_date(index_start), index_start

    day_offsets = {"yesterday": -1, "today": 0, "tomorrow": 1}
    if normalized_cue in day_offsets:
        index_start = anchor + timedelta(days=day_offsets[normalized_cue])
        return _human_date(index_start), index_start

    month = re.fullmatch(r"(last|next)\s+month", normalized_cue)
    if month:
        index_start = _shift_month(anchor, -1 if month.group(1) == "last" else 1)
        return f"{_MONTH_NAMES[index_start.month]} {index_start.year}", index_start

    year = re.fullmatch(r"(last|next)\s+year", normalized_cue)
    if year:
        index_start = date(anchor.year + (-1 if year.group(1) == "last" else 1), 1, 1)
        return str(index_start.year), index_start
    return None


def _date_text_variants(value: date):
    month = _MONTH_NAMES[value.month]
    return (
        value.isoformat(),
        f"{value.day} {month} {value.year}",
        f"{month} {value.day}, {value.year}",
        f"{month} {value.day} {value.year}",
    )


def _replace_or_append_time(text: str, cue: str, display: str, index_start: date):
    value = str(text or "").strip()
    if not value:
        return value
    if re.fullmatch(r"\d{4}", display):
        already_rendered = re.search(
            rf"\b{re.escape(display)}\b(?!-\d{{2}}-\d{{2}})", value)
    else:
        already_rendered = display.lower() in value.lower()
    if already_rendered:
        return value
    replaced, count = re.subn(
        re.escape(cue), display, value, count=1, flags=re.IGNORECASE)
    if count:
        return replaced
    for variant in _date_text_variants(index_start):
        replaced, count = re.subn(
            rf"\b{re.escape(variant)}\b", display, value, count=1,
            flags=re.IGNORECASE,
        )
        if count:
            return replaced
    if re.fullmatch(r"(?:last|next)\s+year", cue, re.IGNORECASE):
        replaced, count = re.subn(
            rf"\b{index_start.year}\b", display, value, count=1)
        if count:
            return replaced
    suffix = value[-1] if value[-1] in ".!?" else ""
    body = value[:-1].rstrip() if suffix else value
    return f"{body} ({display}){suffix}"


def normalize_rewrite_temporal_granularity(rewrite_out, dialogue_text: str):
    """Normalize source-supported temporal wording directly in rewrite text."""
    if not isinstance(rewrite_out, dict):
        return
    sentences = rewrite_out.get("sentence") or []
    # Event time is represented in the self-contained memory text. Strip the
    # retired structured field even if a model emits the legacy schema.
    for sentence in sentences:
        if isinstance(sentence, dict):
            sentence.pop("time", None)
    conversation_time = _conversation_time_from_text(dialogue_text)
    if not conversation_time:
        return
    dialogue = _dialogue_by_origin(dialogue_text)
    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        source = "\n".join(
            dialogue[origin]
            for origin in origin_ids(sentence.get("origin"))
            if origin in dialogue
        )
        matches = list(_RELATIVE_TIME_RE.finditer(source))
        if not matches:
            continue
        # Several distinct temporal cues may apply to different clauses. Leave
        # their placement to the rewrite model instead of attaching one cue to
        # the whole compressed memory and changing its meaning.
        unique_cues = list(dict.fromkeys(
            re.sub(r"\s+", " ", match.group(0)).strip().lower()
            for match in matches
        ))
        if len(unique_cues) != 1:
            continue
        cue = matches[0].group(0)
        rendering = _relative_time_rendering(cue, conversation_time)
        if not rendering:
            continue
        display, index_start = rendering
        sentence["text"] = _replace_or_append_time(
            sentence.get("text"), cue, display, index_start)


def _empty_rewrite(conversation_time=None):
    return {
        "conversation_time": conversation_time or "1970-01-01",
        "sentence": [],
        "topics": {},
        "personal_sentences": [],
    }


def _utility_child(memory):
    return {
        "id": memory.get("id"),
        "origin": memory.get("origin"),
        "text": memory.get("text"),
    }


def _keep_new_children(candidates, kept_children, logger=None):
    from llm.unified_utility_judge import get_unified_utility_judge

    judge = get_unified_utility_judge()
    accepted = []
    for candidate in candidates:
        keep = judge.keep(
            "WRITE",
            {
                "scope": "current session only",
                "task": "strict duplicate detection only",
                "default": "KEEP",
                "drop_only_if": (
                    "the candidate describes the same event or state at the "
                    "same time or stage and every factual proposition is "
                    "already present in the kept children"
                ),
                "keep_if": (
                    "it adds any new entity, attribute, value, time, stage, "
                    "reason, result, opinion, relation, process, or detail; "
                    "sharing a person or broad topic is not duplication"
                ),
            },
            [_utility_child(memory) for memory in kept_children],
            _utility_child(candidate),
        )
        if logger:
            logger.info(
                "[write-utility] decision=%s selected=%d candidate=%s",
                "KEEP" if keep else "DROP",
                len(kept_children),
                json.dumps(_utility_child(candidate), ensure_ascii=False),
            )
        if keep:
            kept_children.append(candidate)
            accepted.append(candidate)
    return accepted


def _plan_parent_groups(llm, turns, children):
    if not children:
        return []
    payload = {
        "complete_raw_session": [turn.line for turn in turns],
        "accepted_children": [_utility_child(child) for child in children],
    }
    output = llm.chat_text(messages=[
        {"role": "system", "content": Prompts.PARENT_GROUP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": Prompts.extract_parent_group_prompt(
                json.dumps(payload, ensure_ascii=False)
            ),
        },
    ], temperature=0.0)
    groups = output.get("parent_groups") if isinstance(output, dict) else None
    if not isinstance(groups, list):
        raise ValueError("parent grouping must return parent_groups")

    valid_ids = {child["id"] for child in children}
    used = set()
    result = []
    for group in groups:
        child_ids = group.get("child_ids") if isinstance(group, dict) else None
        if not isinstance(child_ids, list) or not child_ids:
            raise ValueError("every parent group must contain child_ids")
        if any(child_id not in valid_ids for child_id in child_ids):
            raise ValueError("parent group contains an unknown child id")
        if len(child_ids) != len(set(child_ids)) or used.intersection(child_ids):
            raise ValueError("parent groups must be disjoint")
        used.update(child_ids)
        result.append(child_ids)
    return result


def _build_session_parents(llm, turns, children):
    from llm.unified_utility_judge import get_unified_utility_judge

    groups = _plan_parent_groups(llm, turns, children)
    if not groups:
        return []
    child_by_id = {child["id"]: child for child in children}
    raw_session = [turn.line for turn in turns]
    session_prefix = turns[0].origin.split(":", 1)[0]
    judge = get_unified_utility_judge()
    parents = []
    for number, child_ids in enumerate(groups, start=1):
        parent_id = f"{session_prefix}:P{number}"
        group = [child_by_id[child_id] for child_id in child_ids]
        payload = {
            "parent_id": parent_id,
            "complete_raw_session": raw_session,
            "child_group": [_utility_child(child) for child in group],
        }
        output = llm.chat_text(messages=[
            {"role": "system", "content": Prompts.PARENT_REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": Prompts.extract_parent_rewrite_prompt(
                    json.dumps(payload, ensure_ascii=False)
                ),
            },
        ], temperature=0.0)
        if not isinstance(output, dict) or output.get("parent_id") != parent_id:
            raise ValueError(f"invalid parent rewrite for {parent_id}")
        rewrite_content = str(output.get("rewrite_content") or "").strip()
        if not rewrite_content:
            raise ValueError(f"empty parent rewrite for {parent_id}")
        candidate = {
            "parent_id": parent_id,
            "rewrite_content": rewrite_content,
            "child_ids": child_ids,
        }
        if not judge.keep(
            "ABSTRACT",
            {
                "scope": "current session only",
                "criterion": "supported, distinct, useful high-level abstraction",
            },
            [
                {
                    "parent_id": parent["parent_id"],
                    "rewrite_content": parent["rewrite_content"],
                }
                for parent in parents
            ],
            candidate,
            {"complete_raw_session": raw_session, "child_group": payload["child_group"]},
        ):
            continue
        parents.append(candidate)
        for child in group:
            child["parent_id"] = parent_id
    return parents


def _child_window_source_text(window, turns, conversation_time):
    dialogue = "\n".join(
        turn.line for turn in child_window_turns(window, turns)
    )
    if conversation_time:
        return f"time:{conversation_time}\n{dialogue}"
    return dialogue


def _previous_child_rewrite_context(generated_memories, limit=2):
    return [
        {"rewrite_content": memory.get("text")}
        for memory in (generated_memories or [])[-limit:]
        if isinstance(memory, dict) and memory.get("text")
    ]


def _rewrite_child_window(
        llm,
        window,
        turns,
        conversation_time,
        previous_rewrites=None,
        logger=None,
):
    current_turns = child_window_turns(window, turns)
    user_prompt = Prompts.extract_child_window_rewrite_prompt(
        json.dumps({
            "conversation_time": conversation_time,
            "start_origin": window.start_origin,
            "end_origin": window.end_origin,
            "current_window_turns": [turn.line for turn in current_turns],
        }, ensure_ascii=False),
        json.dumps(previous_rewrites or [], ensure_ascii=False),
    )
    source_text = _child_window_source_text(
        window, turns, conversation_time
    )
    last_error = ""
    for attempt in range(2):
        system_prompt = Prompts.CHILD_WINDOW_REWRITE_SYSTEM_PROMPT
        if attempt:
            system_prompt += (
                f"\nRepair the complete JSON. Validation error: {last_error}"
            )
        raw_output = llm.chat_text(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        tags_valid, output, _, compose_error = (
            _compose_child_tag_components(raw_output)
        )
        if not tags_valid:
            last_error = compose_error
            continue
        output.pop("topics", None)
        inherit_adjacent_question_origins(output, current_turns)
        normalize_sentence_ids(output)
        normalize_rewrite_temporal_granularity(output, source_text)
        normalize_rewrite_semantic_properties(output)
        valid, last_error = json_scheme.check_child_window_rewrite_json(
            output, window, list(turns), source_text
        )
        if valid:
            output["conversation_time"] = conversation_time or output.get(
                "conversation_time"
            )
            return output
    raise ValueError(
        f"child window rewrite failed for {window.start_origin} through "
        f"{window.end_origin}: {last_error}"
    )


def rewrite_semantic_hierarchy_session(llm, text: str, logger=None):
    conversation_time = _conversation_time_from_text(text)
    turns = parse_session_turns(text)
    if not turns:
        empty = _empty_rewrite(conversation_time)
        empty.pop("topics", None)
        empty["parent_nodes"] = []
        return empty

    child_windows = plan_child_windows(llm, turns, conversation_time)
    window_outputs = []
    generated_memories = []
    for window in child_windows:
        previous_rewrites = _previous_child_rewrite_context(
            generated_memories, limit=2
        )
        output = _rewrite_child_window(
            llm,
            window,
            turns,
            conversation_time,
            previous_rewrites=previous_rewrites,
            logger=logger,
        )
        if logger:
            logger.info(
                "[child-candidates] window=%s..%s candidates=%s",
                window.start_origin,
                window.end_origin,
                json.dumps(
                    [_utility_child(item) for item in _as_list(output.get("sentence"))],
                    ensure_ascii=False,
                ),
            )
        output["sentence"] = _keep_new_children(
            _as_list(output.get("sentence")), generated_memories, logger=logger
        )
        window_outputs.append(output)

    # IDs become stable only after every candidate has passed WRITE utility.
    finalize_child_memory_ids(generated_memories, turns)
    parent_outputs = _build_session_parents(llm, turns, generated_memories)

    merged = _empty_rewrite(conversation_time)
    merged.pop("topics", None)
    merged["parent_nodes"] = parent_outputs
    for output in window_outputs:
        for sentence in output.get("sentence") or []:
            if not isinstance(sentence, dict):
                continue
            item = dict(sentence)
            item["origin"] = ",".join(origin_ids(item.get("origin")))
            item["topic"] = []
            merged["sentence"].append(item)
    return merged


def rewrite_windowed_session(llm, text: str, logger=None):
    return rewrite_semantic_hierarchy_session(llm, text, logger=logger)

import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import List, NamedTuple

from common import config
from prompts import schema as json_scheme
from prompts.prompts import Prompts


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


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _dialogue_turn_lines(text: str):
    return [
        line.strip()
        for line in str(text or "").splitlines()
        if re.search(r"\bdia_id\s*:\s*D\d+:\d+", line)
    ]


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
    """Normalize rewrite text granularity and its ISO date index from raw sources."""
    if not isinstance(rewrite_out, dict):
        return
    conversation_time = _conversation_time_from_text(dialogue_text)
    if not conversation_time:
        return
    dialogue = _dialogue_by_origin(dialogue_text)
    for sentence in rewrite_out.get("sentence") or []:
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
        # A compressed memory with several distinct temporal cues cannot be represented
        # by the schema's single time field, so leave it for the model instead of guessing.
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
        sentence["time"] = index_start.isoformat()


class _RewriteWindow(NamedTuple):
    previous_context: List[str]
    current_turns: List[str]


def _window_turns(turns: List[str]):
    if not turns:
        return []
    window_size = config.REWRITE_WINDOW_SIZE
    windows = []
    for start in range(0, len(turns), window_size):
        context_size = min(start, config.REWRITE_OVERLAP_SIZE)
        windows.append(_RewriteWindow(
            previous_context=turns[start - context_size:start],
            current_turns=turns[start:start + window_size],
        ))
    return windows


def _empty_rewrite(conversation_time=None):
    return {
        "conversation_time": conversation_time or "1970-01-01",
        "sentence": [],
        "topics": {},
        "personal_sentences": [],
    }


def _filter_context_only_outputs(rewrite_out, current_window_text: str):
    """Drop context-only items, retrying only when every returned item is dropped."""
    current_ids = set(_dialogue_by_origin(current_window_text))
    if not current_ids:
        return False, "current dialogue window has no dia_id values"
    invalid_count = 0
    compliant_count = 0
    for field in ("sentence", "personal_sentences"):
        compliant_items = []
        for item in rewrite_out.get(field) or []:
            if not isinstance(item, dict):
                compliant_items.append(item)
                compliant_count += 1
                continue
            ids = set(origin_ids(item.get("origin")))
            if ids and ids.isdisjoint(current_ids):
                invalid_count += 1
                continue
            compliant_items.append(item)
            compliant_count += 1
        rewrite_out[field] = compliant_items
    if not invalid_count:
        return True, ""
    if not compliant_count:
        return (
            False,
            "all returned memory items are supported only by "
            "PREVIOUS_DIALOGUE_CONTEXT; omit them and extract information "
            "from CURRENT_DIALOGUE_WINDOW",
        )

    used_topics = {
        str(topic)
        for sentence in rewrite_out.get("sentence") or []
        if isinstance(sentence, dict)
        for topic in _as_list(sentence.get("topic"))
    }
    if isinstance(rewrite_out.get("topics"), dict):
        rewrite_out["topics"] = {
            topic_id: topic_text
            for topic_id, topic_text in rewrite_out["topics"].items()
            if str(topic_id) in used_topics
        }
    return True, ""


def _rewrite_window(
        llm,
        current_window_text: str,
        source_text: str,
        previous_dialogue_context: str,
        logger=None,
):
    rewrite_prompt = Prompts.extract_rewrite_prompt(
        json.dumps(current_window_text, ensure_ascii=False),
        json.dumps(previous_dialogue_context, ensure_ascii=False),
    )
    rewrite_out = llm.chat_text(
        messages=[
            {"role": "system", "content": Prompts.REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": rewrite_prompt},
        ],
    )
    normalize_sentence_ids(rewrite_out)
    normalize_rewrite_temporal_granularity(rewrite_out, source_text)
    flag, err = json_scheme.check_rewrite_json(rewrite_out, source_text)
    if flag:
        flag, err = _filter_context_only_outputs(
            rewrite_out, current_window_text)
    last_err = err
    if not flag:
        for _ in range(1, 4):
            rewrite_out = llm.chat_text(
                messages=[
                    {
                        "role": "system",
                        "content": Prompts.REWRITE_SYSTEM_PROMPT
                        + " The previous run failed with the following error: "
                        + last_err,
                    },
                    {"role": "user", "content": rewrite_prompt},
                ],
                temperature=1.0,
            )
            normalize_sentence_ids(rewrite_out)
            normalize_rewrite_temporal_granularity(rewrite_out, source_text)
            flag, err = json_scheme.check_rewrite_json(rewrite_out, source_text)
            if flag:
                flag, err = _filter_context_only_outputs(
                    rewrite_out, current_window_text)
            if flag:
                break
            last_err = err
    if not flag:
        if logger:
            logger.warning(f"rewrite window failed validation after retries: {last_err}")
        return _empty_rewrite(_conversation_time_from_text(current_window_text))
    return rewrite_out


def _merge_window_rewrites(window_outputs: List[dict], conversation_time: str):
    merged = _empty_rewrite(conversation_time)
    topic_counter = 1
    personal_counter = 1
    seen_memory_keys = set()
    for out in window_outputs:
        if not isinstance(out, dict):
            continue
        # Keep the session-level anchor parsed from the raw session text.
        # Window LLM outputs may confuse event dates with conversation_time.
        topic_map = {}
        for old_tid, topic_text in (out.get("topics") or {}).items():
            if not topic_text:
                continue
            new_tid = f"t{topic_counter}"
            topic_counter += 1
            topic_map[str(old_tid)] = new_tid
            merged["topics"][new_tid] = topic_text
        for sentence in out.get("sentence") or []:
            if not isinstance(sentence, dict):
                continue
            ids = origin_ids(sentence.get("origin"))
            text = str(sentence.get("text") or "").strip()
            if not ids or not text:
                continue
            sentence = dict(sentence)
            sentence["origin"] = ",".join(ids)
            sentence["topic"] = [
                topic_map.get(str(topic), str(topic))
                for topic in _as_list(sentence.get("topic"))
                if topic_map.get(str(topic), str(topic)) in merged["topics"]
            ]
            key = (text.lower(), sentence["origin"])
            if key in seen_memory_keys:
                continue
            seen_memory_keys.add(key)
            merged["sentence"].append(sentence)
        for personal_sentence in out.get("personal_sentences") or []:
            if not isinstance(personal_sentence, dict) or not personal_sentence.get("text"):
                continue
            item = dict(personal_sentence)
            item["id"] = f"p{personal_counter}"
            personal_counter += 1
            if item.get("origin"):
                item["origin"] = ",".join(origin_ids(item.get("origin"))) or item.get("origin")
            merged["personal_sentences"].append(item)
    normalize_sentence_ids(merged)
    return merged


def rewrite_windowed_session(llm, text: str, logger=None):
    conversation_time = _conversation_time_from_text(text)
    turns = _dialogue_turn_lines(text)
    if not turns:
        return _empty_rewrite(conversation_time)
    window_outputs = []
    for window in _window_turns(turns):
        previous_dialogue_context = "\n".join(window.previous_context)
        current_dialogue = "\n".join(window.current_turns)
        source_dialogue = "\n".join(
            window.previous_context + window.current_turns)
        if conversation_time:
            current_window_text = "time:" + conversation_time + "\n" + current_dialogue
            source_text = "time:" + conversation_time + "\n" + source_dialogue
        else:
            current_window_text = current_dialogue
            source_text = source_dialogue
        out = _rewrite_window(
            llm,
            current_window_text,
            source_text,
            previous_dialogue_context,
            logger=logger,
        )
        window_outputs.append(out)
    return _merge_window_rewrites(window_outputs, conversation_time)

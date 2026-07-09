import json
import re
from collections import defaultdict
from typing import List

from common import config
from prompts import schema as json_scheme
from prompts.prompts import Prompts


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
    m = re.search(r"^\s*time\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", str(text or ""), re.MULTILINE)
    return m.group(1) if m else None


def _window_turns(turns: List[str]):
    if not turns:
        return []
    window_size = config.REWRITE_WINDOW_SIZE
    step_size = window_size - config.REWRITE_OVERLAP_SIZE
    windows = []
    start = 0
    while start < len(turns):
        windows.append(turns[start:start + window_size])
        if start + window_size >= len(turns):
            break
        start += step_size
    return windows


def _empty_rewrite(conversation_time=None):
    return {
        "conversation_time": conversation_time or "1970-01-01",
        "sentence": [],
        "topics": {},
        "personal_sentences": [],
    }


def _rewrite_window(llm, window_text: str, previous_memories: List[dict], logger=None):
    previous_slice = (
        previous_memories[-config.REWRITE_PREVIOUS_LIMIT:]
        if config.REWRITE_PREVIOUS_LIMIT
        else []
    )
    previous_payload = [
        {
            "id": memory.get("id"),
            "text": memory.get("text"),
            "origin": memory.get("origin"),
            "time": memory.get("time"),
            "tag": memory.get("tag"),
        }
        for memory in previous_slice
        if isinstance(memory, dict)
    ]
    rewrite_prompt = Prompts.extract_rewrite_prompt(
        json.dumps(window_text, ensure_ascii=False),
        json.dumps(previous_payload, ensure_ascii=False),
    )
    rewrite_out = llm.chat_text(
        messages=[
            {"role": "system", "content": Prompts.REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": rewrite_prompt},
        ],
    )
    normalize_sentence_ids(rewrite_out)
    flag, err = json_scheme.check_rewrite_json(rewrite_out, window_text)
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
            flag, err = json_scheme.check_rewrite_json(rewrite_out, window_text)
            if flag:
                break
            last_err = err
    if not flag:
        if logger:
            logger.warning(f"rewrite window failed schema validation after retries: {last_err}")
        return _empty_rewrite(_conversation_time_from_text(window_text))
    return rewrite_out


def _merge_window_rewrites(window_outputs: List[dict], conversation_time: str):
    merged = _empty_rewrite(conversation_time)
    topic_counter = 1
    personal_counter = 1
    seen_memory_keys = set()
    for out in window_outputs:
        if not isinstance(out, dict):
            continue
        if out.get("conversation_time"):
            merged["conversation_time"] = out.get("conversation_time")
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
    previous_memories = []
    for window in _window_turns(turns):
        if conversation_time:
            window_text = "time:" + conversation_time + "\n" + "\n".join(window)
        else:
            window_text = "\n".join(window)
        out = _rewrite_window(llm, window_text, previous_memories, logger=logger)
        window_outputs.append(out)
        previous_memories.extend(_as_list(out.get("sentence")) if isinstance(out, dict) else [])
    return _merge_window_rewrites(window_outputs, conversation_time)

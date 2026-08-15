import json
import math
import re
from dataclasses import dataclass, field
from typing import List, Sequence

from common import config
from prompts.prompts import Prompts


_ORIGIN_RE = re.compile(r"\bdia_id\s*:\s*(D\d+:\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TurnRecord:
    origin: str
    line: str
    index: int


@dataclass
class ParentSegment:
    parent_id: str
    start_index: int
    end_index: int
    current_turns: List[TurnRecord]
    previous_context: List[TurnRecord]
    child_ids: List[str] = field(default_factory=list)
    rewrite_content: str = ""


@dataclass(frozen=True)
class ChildWindow:
    start_origin: str
    end_origin: str


def parse_session_turns(text: str) -> List[TurnRecord]:
    turns = []
    for line in str(text or "").splitlines():
        match = _ORIGIN_RE.search(line)
        if match:
            turns.append(TurnRecord(match.group(1), line.strip(), len(turns)))
    return turns


def _request_complete_plan(llm, system_prompt, user_prompt, validate, label):
    last_error = ""
    last_output = None
    for attempt in range(4):
        prompt = system_prompt
        request = user_prompt
        if attempt:
            prompt += (
                "\nThe previous complete plan was invalid. Return the entire plan "
                "again. The validation error below identifies the exact invalid "
                "segment or boundary. Do not return that invalid boundary unchanged. "
                "Recalculate every segment length with the inclusive formula "
                "end_position - start_position + 1 before responding. "
                f"Validation error: {last_error}"
            )
            request += (
                "\n\nREPAIR_REQUIRED:\n"
                f"Validation error: {last_error}\n"
                "Correct the exact segment or boundary identified above. Do not "
                "repeat it unchanged. After repairing it, verify that every segment "
                "satisfies all minimum/maximum limits and that the complete session "
                "is still covered contiguously with no gaps or overlap. Return the "
                "entire plan, not only the changed segment. The following was the "
                "invalid complete plan:\n"
                f"{json.dumps(last_output, ensure_ascii=False)}"
            )
        output = llm.chat_text(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": request},
            ],
            temperature=0.0 if attempt == 0 else 0.7,
        )
        last_output = output
        valid, last_error, value = validate(output)
        if valid:
            return value
    raise ValueError(f"{label} failed validation after retries: {last_error}")


def _turn_payload(turns: Sequence[TurnRecord]):
    return [
        {
            "position": turn.index + 1,
            "origin": turn.origin,
            "text": turn.line,
        }
        for turn in turns
    ]


def _validate_parent_plan(output, turns: Sequence[TurnRecord]):
    if not isinstance(output, dict):
        return False, "parent plan must be a JSON object", None
    raw_segments = output.get("parent_segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return False, "parent_segments must be a non-empty list", None
    minimum_segment_count = math.ceil(len(turns) / config.PARENT_MAX_TURNS)
    if len(raw_segments) < minimum_segment_count:
        return False, (
            f"this {len(turns)}-turn session must return at least "
            f"{minimum_segment_count} parent segments because the hard maximum is "
            f"{config.PARENT_MAX_TURNS}; received {len(raw_segments)}"
        ), None
    origin_to_index = {turn.origin: turn.index for turn in turns}
    expected_start = 0
    parents = []
    session_prefix = turns[0].origin.split(":", 1)[0]
    for number, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, dict):
            return False, f"parent_segments[{number - 1}] must be an object", None
        start_origin = str(raw.get("start_origin") or "")
        end_origin = str(raw.get("end_origin") or "")
        if start_origin not in origin_to_index or end_origin not in origin_to_index:
            return False, f"parent segment has unknown boundary: {start_origin}, {end_origin}", None
        start = origin_to_index[start_origin]
        end = origin_to_index[end_origin]
        if start != expected_start or end < start:
            expected_origin = turns[expected_start].origin
            return False, (
                f"parent segment {number} has invalid boundaries: received "
                f"{start_origin} (position {start + 1}) through {end_origin} "
                f"(position {end + 1}), but this segment must start at "
                f"{expected_origin} (position {expected_start + 1}) to preserve "
                "contiguous ordered coverage; end position must not precede start "
                "position"
            ), None
        length = end - start + 1
        is_final = number == len(raw_segments)
        if length > config.PARENT_MAX_TURNS:
            required_parts = math.ceil(length / config.PARENT_MAX_TURNS)
            return False, (
                f"parent segment {number} has the invalid range {start_origin} "
                f"through {end_origin}. Its inclusive positions are {start + 1} "
                f"through {end + 1}, so its actual length is "
                f"{end + 1} - {start + 1} + 1 = {length} turns. This exceeds "
                f"the hard maximum of {config.PARENT_MAX_TURNS}. Do not return "
                f"{start_origin} through {end_origin} unchanged; split this exact "
                f"range into at least {required_parts} contiguous segments, each "
                f"containing at most {config.PARENT_MAX_TURNS} turns"
            ), None
        if length < config.PARENT_MIN_TURNS and not is_final:
            return False, (
                f"non-final parent segment {number} has the invalid range "
                f"{start_origin} through {end_origin}. Its inclusive positions are "
                f"{start + 1} through {end + 1}, so its actual length is "
                f"{end + 1} - {start + 1} + 1 = {length} turns. The hard "
                f"minimum is {config.PARENT_MIN_TURNS}; move this segment's end "
                "boundary or an adjacent boundary so every non-final segment meets "
                "the minimum"
            ), None
        context_start = max(0, start - config.PARENT_CONTEXT_TURNS)
        parents.append(ParentSegment(
            parent_id=f"{session_prefix}:t{number}",
            start_index=start,
            end_index=end,
            current_turns=list(turns[start:end + 1]),
            previous_context=list(turns[context_start:start]),
        ))
        expected_start = end + 1
    if expected_start != len(turns):
        return False, (
            f"the complete plan stops before {turns[expected_start].origin} "
            f"(position {expected_start + 1}); it must continue through "
            f"{turns[-1].origin} (position {len(turns)}) with contiguous legal "
            "segments"
        ), None
    return True, "", parents


def plan_parent_segments(llm, turns: Sequence[TurnRecord], conversation_time=None):
    if not turns:
        return []
    user_prompt = Prompts.extract_parent_segment_prompt(
        json.dumps({
            "conversation_time": conversation_time,
            "minimum_turns": config.PARENT_MIN_TURNS,
            "maximum_turns": config.PARENT_MAX_TURNS,
            "minimum_segment_count": math.ceil(
                len(turns) / config.PARENT_MAX_TURNS
            ),
            "total_turns": len(turns),
            "turns": _turn_payload(turns),
        }, ensure_ascii=False)
    )
    return _request_complete_plan(
        llm,
        Prompts.PARENT_SEGMENT_SYSTEM_PROMPT,
        user_prompt,
        lambda output: _validate_parent_plan(output, turns),
        "parent semantic plan",
    )


def _validate_child_window_plan(output, turns: Sequence[TurnRecord]):
    if not isinstance(output, dict):
        return False, "child window plan must be a JSON object", None
    raw_segments = output.get("child_segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return False, "child_segments must be a non-empty list", None
    origin_to_index = {turn.origin: turn.index for turn in turns}
    minimum_segment_count = math.ceil(len(turns) / config.CHILD_MAX_TURNS)
    if len(raw_segments) < minimum_segment_count:
        return False, (
            f"this {len(turns)}-turn session must return at least "
            f"{minimum_segment_count} child windows because the hard maximum is "
            f"{config.CHILD_MAX_TURNS}; received {len(raw_segments)}"
        ), None
    expected_start = 0
    windows = []
    for number, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, dict):
            return False, f"child_segments[{number - 1}] must be an object", None
        start_origin = str(raw.get("start_origin") or "")
        end_origin = str(raw.get("end_origin") or "")
        if start_origin not in origin_to_index or end_origin not in origin_to_index:
            return False, (
                f"child window has unknown boundary: {start_origin}, {end_origin}"
            ), None
        start = origin_to_index[start_origin]
        end = origin_to_index[end_origin]
        if expected_start >= len(turns):
            return False, (
                f"child window {number} starts after the session was already "
                f"fully covered through {turns[-1].origin}"
            ), None
        if start != expected_start or end < start:
            expected_origin = turns[expected_start].origin
            return False, (
                f"child window {number} has invalid boundaries: received "
                f"{start_origin} through {end_origin}, but it must start at "
                f"{expected_origin} to preserve contiguous ordered coverage"
            ), None
        length = end - start + 1
        if length > config.CHILD_MAX_TURNS:
            required_parts = math.ceil(length / config.CHILD_MAX_TURNS)
            return False, (
                f"child window {number} has the invalid range {start_origin} "
                f"through {end_origin}. Its inclusive length is {length} turns, "
                f"which exceeds the hard maximum of {config.CHILD_MAX_TURNS}; "
                f"split this range into at least {required_parts} contiguous windows"
            ), None
        windows.append(ChildWindow(
            start_origin=start_origin,
            end_origin=end_origin,
        ))
        expected_start = end + 1
    if expected_start != len(turns):
        return False, (
            f"the child window plan stops before {turns[expected_start].origin}; "
            f"it must continue through {turns[-1].origin}"
        ), None
    return True, "", windows


def plan_child_windows(llm, turns: Sequence[TurnRecord], conversation_time=None):
    if not turns:
        return []
    user_prompt = Prompts.extract_child_window_prompt(
        json.dumps({
            "conversation_time": conversation_time,
            "maximum_turns": config.CHILD_MAX_TURNS,
            "minimum_segment_count": math.ceil(
                len(turns) / config.CHILD_MAX_TURNS
            ),
            "total_turns": len(turns),
            "turns": _turn_payload(turns),
        }, ensure_ascii=False)
    )
    return _request_complete_plan(
        llm,
        Prompts.CHILD_WINDOW_SYSTEM_PROMPT,
        user_prompt,
        lambda output: _validate_child_window_plan(output, turns),
        "child window plan",
    )


def child_window_turns(
        window: ChildWindow,
        turns: Sequence[TurnRecord],
) -> List[TurnRecord]:
    origin_to_index = {turn.origin: turn.index for turn in turns}
    try:
        start = origin_to_index[window.start_origin]
        end = origin_to_index[window.end_origin]
    except KeyError as exc:
        raise ValueError(f"child window contains unknown origin {exc.args[0]}") from exc
    if end < start:
        raise ValueError(
            f"child window ends before it starts: "
            f"{window.start_origin}, {window.end_origin}"
        )
    return list(turns[start:end + 1])


def attach_child_memory_ids_to_parents(
        parent_segments: Sequence[ParentSegment],
        child_memories: Sequence[dict],
):
    parent_by_turn = {}
    for parent in parent_segments:
        for turn in parent.current_turns:
            parent_by_turn[turn.origin] = parent
    for memory in child_memories:
        origins = re.findall(r"D\d+:\d+", str(memory.get("origin") or ""))
        if not origins or not memory.get("id"):
            raise ValueError("child memory needs a valid id and origin")
        owner = parent_by_turn.get(origins[0])
        if owner is None:
            raise ValueError(
                f"no parent core contains child first origin {origins[0]}"
            )
        owner.child_ids.append(memory["id"])
    return list(parent_segments)

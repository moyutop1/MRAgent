import json
import math
import re
from dataclasses import dataclass
from typing import List, Sequence

from common import config
from prompts.prompts import Prompts


_ORIGIN_RE = re.compile(r"\bdia_id\s*:\s*(D\d+:\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TurnRecord:
    origin: str
    line: str
    index: int


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


def _validate_child_plan(output, turns: Sequence[TurnRecord]):
    segments = output.get("child_segments") if isinstance(output, dict) else None
    if not isinstance(segments, list) or not segments:
        return None, "child_segments must be a non-empty list"
    if len(segments) < math.ceil(len(turns) / config.CHILD_MAX_TURNS):
        return None, "too few windows for CHILD_MAX_TURNS"

    positions = {turn.origin: turn.index for turn in turns}
    expected_start = 0
    windows = []
    for segment in segments:
        if not isinstance(segment, dict):
            return None, "every child segment must be an object"
        start_origin = str(segment.get("start_origin") or "")
        end_origin = str(segment.get("end_origin") or "")
        if start_origin not in positions or end_origin not in positions:
            return None, "child segment contains an unknown origin"
        start, end = positions[start_origin], positions[end_origin]
        if start != expected_start or end < start:
            return None, "child windows must be contiguous and ordered"
        if end - start + 1 > config.CHILD_MAX_TURNS:
            return None, "a child window exceeds CHILD_MAX_TURNS"
        windows.append(ChildWindow(start_origin, end_origin))
        expected_start = end + 1
    if expected_start != len(turns):
        return None, "child windows do not cover the complete session"
    return windows, ""


def plan_child_windows(llm, turns: Sequence[TurnRecord], conversation_time=None):
    if not turns:
        return []
    payload = {
        "conversation_time": conversation_time,
        "maximum_turns": config.CHILD_MAX_TURNS,
        "minimum_segment_count": math.ceil(len(turns) / config.CHILD_MAX_TURNS),
        "total_turns": len(turns),
        "turns": [
            {
                "position": turn.index + 1,
                "origin": turn.origin,
                "text": turn.line,
            }
            for turn in turns
        ],
    }
    error = ""
    for attempt in range(2):
        system_prompt = Prompts.CHILD_WINDOW_SYSTEM_PROMPT
        if error:
            system_prompt += f"\nRepair the plan. Validation error: {error}"
        output = llm.chat_text(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": Prompts.extract_child_window_prompt(
                        json.dumps(payload, ensure_ascii=False)
                    ),
                },
            ],
            temperature=0.0,
        )
        windows, error = _validate_child_plan(output, turns)
        if windows is not None:
            return windows
    raise ValueError(f"child window plan failed validation: {error}")


def child_window_turns(
        window: ChildWindow, turns: Sequence[TurnRecord]
) -> List[TurnRecord]:
    positions = {turn.origin: turn.index for turn in turns}
    start = positions[window.start_origin]
    end = positions[window.end_origin]
    return list(turns[start:end + 1])

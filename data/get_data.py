from pathlib import Path
import json, re
import logging
from collections import defaultdict
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

def get_data(dataset: str, data_path: str):
    path = Path(data_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    sample_ids = [s.get("sample_id") for s in data if isinstance(s, dict)]
    n_samples_total = len(data)

    conversation_list = {}
    question_list = {}
    raw_conversation_list = {}
    raw_text_list = {}

    if not isinstance(data, list):
        raise ValueError(f"expected data to be a list but got {type(data)}; check {data_path}")

    pat = re.compile(r"^session_(\d+)$")
    for i, sample in enumerate(data):
        if not isinstance(sample, dict):
            logger.warning(f"sample {i} is not a dict; skipping")
            continue
        sample_id = sample.get("sample_id")
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            logger.warning(f"sample {sample_id} has no dict conversation; skipping")
            continue
        session_keys = []
        for k in conversation.keys():
            m = pat.match(k)
            if m and isinstance(conversation.get(k), list):
                session_keys.append((int(m.group(1)), k))
        n_sessions = len(session_keys)
        session_keys.sort()
        session_list = {}
        raw_text: Dict[str, dict] = defaultdict(dict)
        for idx, k in session_keys:
            turns = conversation.get(k, [])
            session_time = conversation.get(f"{k}_date_time")
            session_id = f"D{idx}"

            lines = []

            lines.append(f"time:{session_time}".strip())

            # expand turn by turn
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = (turn.get("speaker") or "UNKNOWN").strip()
                dia_id = (turn.get("dia_id") or f"{k}:{len(lines)}").strip()
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                # [LM] for LM, keep only the user's turns
                if dataset == "LM" and speaker != "user":
                    continue
                if dataset == "locomo":
                    if turn.get("blip_caption") != None:
                        lines.append(f"dia_id:{dia_id} {speaker}:{text} and shared {turn.get('blip_caption')}")
                        raw_text[session_id].update({dia_id: f"{speaker}:{text} and shared {turn.get('blip_caption')}"})

                    else:
                        lines.append(f"dia_id:{dia_id} {speaker}:{text}")
                        raw_text[session_id].update({dia_id: f"{speaker}:{text}"})

                else:
                    lines.append(f"dia_id:{dia_id} {speaker}:{text}")
                    raw_text[session_id].update({dia_id: f"{speaker}:{text}"})

            # join into one session text (line-separated, for the LLM)
            session_text = "\n".join(lines)
            session_list[k] = session_text

        conversation_list[sample_id] = session_list
        questions = sample.get("qa")
        # inject metadata.question_date into each qa (temporal time anchor)
        _md = sample.get("metadata", {})
        _qdate = _md.get("question_date") if isinstance(_md, dict) else None
        if questions and _qdate:
            for _qa in questions:
                if isinstance(_qa, dict) and "question_date" not in _qa:
                    _qa["question_date"] = _qdate
        question_list[sample_id] = questions
        raw_conversation_list[sample_id] = conversation
        raw_text_list[sample_id] = raw_text

    out_path = Path(f"data/conversation_list_{dataset}.json")
    out_path.write_text(
        json.dumps(conversation_list, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    out_path = Path(f"data/question_list_{dataset}.json")
    out_path.write_text(
        json.dumps(question_list, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    logger.info(f"Wrote {out_path.resolve()}  | samples: {len(conversation_list)}")
    return conversation_list, question_list, raw_conversation_list, raw_text_list


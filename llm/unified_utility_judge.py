import json
import logging
import re
import threading

from common import config
from prompts.schema import check_composite_tag


_JUDGE = None
_JUDGE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)

UTILITY_SYSTEM_PROMPT = (
    "You are a unified set-conditioned marginal-utility judge. "
    "Decide whether CANDIDATE adds useful information under CONDITION "
    "given SELECTED_SET. Positive conditional marginal value means the "
    "candidate contributes at least one supported fact not already in the set.\n"
    "WRITE is strict duplicate detection, not importance filtering. Default "
    "to KEEP. DROP only when the candidate describes the same event or state "
    "at the same time or stage and every factual proposition is already "
    "contained in SELECTED_SET. The same person, topic, or activity is not "
    "enough to DROP. Any new entity, attribute, value, time, stage, reason, "
    "result, opinion, relation, process, or concrete detail requires KEEP.\n"
    "ABSTRACT: KEEP only if the abstraction is supported by SOURCE, is "
    "more useful than merely repeating its children, and adds a distinct "
    "profile, event synopsis, entity relation, or stable high-level state.\n"
    "SELECT: KEEP only if it adds information useful for answering the query "
    "that is not already supplied by SELECTED_SET.\n"
    "Return exactly KEEP or DROP."
)

CUE_SYSTEM_PROMPT = """You generate one main retrieval cue for a question using exactly the same grammar as a child memory tag. Return only valid JSON:
{"prefix": "Person description", "facet": "concrete noun phrase"}
Rules:
- Output exactly one prefix and one facet.
- prefix starts with an explicit person's name from the question, followed by a concise activity, plan, profile, possession, or relationship description.
- facet is a concrete noun phrase of at most five words.
- Do not place periods inside either field.
- Describe the information being requested; never guess the answer.
- Preserve named entities, relation direction, event, and temporal constraints from the question.
- For list or aggregate questions, describe the shared relation broadly enough to retrieve all supporting memories."""


class UnifiedUtilityJudge:
    """Use DeepSeek Flash for the main Cue and unified utility decisions."""

    def __init__(self):
        from openai import OpenAI

        if not config.UTILITY_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is required by the utility judge")
        self.client = OpenAI(
            api_key=config.UTILITY_API_KEY,
            base_url=config.UTILITY_BASE_URL,
            timeout=config.UTILITY_TIMEOUT,
            max_retries=config.LLM_REQUEST_MAX_RETRIES,
        )

    @staticmethod
    def count_tokens(value):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        # Stable local proxy for the evidence budget; API billing uses server tokens.
        return max(1, (len(text) + 3) // 4)

    def keep(self, operation, condition, selected_set, candidate, source=None):
        payload = {
            "operation": operation,
            "condition": condition,
            "selected_set": selected_set,
            "candidate": candidate,
        }
        if source is not None:
            payload["source"] = source
        response = self.client.chat.completions.create(
            model=config.UTILITY_MODEL,
            messages=[
                {"role": "system", "content": UTILITY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=8,
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = str(response.choices[0].message.content or "").strip().upper()
        if answer.startswith("KEEP"):
            return True
        if answer.startswith("DROP"):
            return False
        raise ValueError(f"invalid utility decision: {answer!r}")

    def generate_cue(self, question):
        messages = [
            {"role": "system", "content": CUE_SYSTEM_PROMPT},
            {"role": "user", "content": str(question).strip()},
        ]
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=config.UTILITY_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=96,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception as error:
                logger.error(
                    "Cue API retries exhausted; using raw-query fallback: %s",
                    error,
                )
                return None

            output = str(response.choices[0].message.content or "").strip()
            try:
                data = json.loads(output)
                if not isinstance(data, dict):
                    raise ValueError("Cue output must be a JSON object")
                prefix = re.sub(
                    r"\s+", " ", str(data.get("prefix") or "")
                ).strip()
                facet = re.sub(
                    r"\s+", " ", str(data.get("facet") or "")
                ).strip()
                cue = f"{prefix}.{facet}"
                valid, error = check_composite_tag(cue)
                if valid:
                    return cue
                raise ValueError(error)
            except (json.JSONDecodeError, ValueError) as error:
                logger.warning(
                    "Invalid Cue on attempt %d/3: %s", attempt + 1, error
                )
                if attempt == 2:
                    break
                messages.extend([
                    {"role": "assistant", "content": output},
                    {
                        "role": "user",
                        "content": (
                            "Your previous Cue was invalid. "
                            f"Validation error: {error}. "
                            "Correct it and return only the required JSON."
                        ),
                    },
                ])
        logger.error("Cue correction retries exhausted; using raw-query fallback")
        return None


def get_unified_utility_judge():
    global _JUDGE
    if _JUDGE is None:
        with _JUDGE_LOCK:
            if _JUDGE is None:
                _JUDGE = UnifiedUtilityJudge()
    return _JUDGE

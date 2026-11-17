import json
import threading

from common import config


_JUDGE = None
_JUDGE_LOCK = threading.Lock()

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


class UnifiedUtilityJudge:
    """Use DeepSeek Flash for WRITE, ABSTRACT, and SELECT utility."""

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


def get_unified_utility_judge():
    global _JUDGE
    if _JUDGE is None:
        with _JUDGE_LOCK:
            if _JUDGE is None:
                _JUDGE = UnifiedUtilityJudge()
    return _JUDGE

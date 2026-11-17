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
    "Return exactly KEEP or DROP."
)

SELECT_SYSTEM_PROMPT = """You estimate two binary components of a candidate memory's set-conditioned marginal answer utility. Return only valid JSON:
{"new_answer_unit": true, "necessary_constraint": false}

Definitions:
- new_answer_unit is true only when CANDIDATE supplies a query-requested answer value or event instance that SELECTED_SET does not already supply.
- For list, "what/which things", and multi-event questions, each distinct requested value or event is one answer unit.
- For count questions, only a distinct countable event instance is a new answer unit. Extra details about an already represented event are not new units.
- For a single-fact question, a candidate is a new answer unit only when it fills the requested answer slot and that slot is not already filled.
- necessary_constraint is true only when CANDIDATE supplies a missing entity, relation direction, time, quantity, cause, update, conflict resolution, or disambiguating fact required to derive or verify an answer unit.
- A related detail, explanation, emotion, example, paraphrase, or Parent/Child restatement is not necessary unless the query explicitly asks for it.
- The same session or topic does not imply duplication, and different sessions do not imply diversity. Judge query-dependent answer units, not surface topic diversity.
- Both fields must be false when the candidate is merely relevant but adds no new answer unit or required constraint.
Do not output KEEP/DROP, scores, explanations, or extra fields."""


class UnifiedUtilityJudge:
    """Use DeepSeek Flash to estimate conditional marginal-utility terms."""

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

    def select(self, query, selected_set, candidate):
        payload = {
            "query": query,
            "selected_set": selected_set,
            "candidate": candidate,
        }
        response = self.client.chat.completions.create(
            model=config.UTILITY_MODEL,
            messages=[
                {"role": "system", "content": SELECT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=48,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        result = json.loads(response.choices[0].message.content)
        for field in ("new_answer_unit", "necessary_constraint"):
            if not isinstance(result.get(field), bool):
                raise ValueError(f"invalid SELECT utility component: {field}")
        marginal_utility = (
            int(result["new_answer_unit"])
            + int(result["necessary_constraint"])
        )
        return {
            **result,
            "marginal_utility": marginal_utility,
            "keep": marginal_utility > 0,
        }


def get_unified_utility_judge():
    global _JUDGE
    if _JUDGE is None:
        with _JUDGE_LOCK:
            if _JUDGE is None:
                _JUDGE = UnifiedUtilityJudge()
    return _JUDGE

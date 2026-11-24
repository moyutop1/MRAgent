import json
import threading

from common import config


_JUDGE = None
_JUDGE_LOCK = threading.Lock()

UTILITY_SYSTEM_PROMPT = """You are a shared set-conditioned marginal-utility estimator. Use CONDITION to identify the operation's target semantic units. Compare CANDIDATE with SELECTED_SET and use SOURCE to verify CANDIDATE. For SELECT, CANDIDATE and SELECTED_SET are retrieved factual evidence. Return only valid JSON:
{"new_target_unit": true, "necessary_constraint": false}

Rules:
- new_target_unit is true only when CANDIDATE adds a SOURCE-supported target unit defined by CONDITION that SELECTED_SET does not already contain.
- necessary_constraint is true only when CANDIDATE adds a SOURCE-supported entity, relation direction, time, quantity, cause, update, or disambiguating fact required to construct or verify a target unit.
- A related detail, explanation, emotion, example, paraphrase, or restatement has no marginal utility unless CONDITION explicitly makes it a target or required constraint.
- The same topic does not imply duplication, and a different session does not imply a new target unit.
- Both fields are false when CANDIDATE is unsupported, redundant, or merely relevant.
Do not output KEEP/DROP, scores, explanations, or extra fields."""

UTILITY_CONDITIONS = {
    "WRITE": {
        "goal": "preserve supported atomic facts without exact repetition",
        "target_unit": (
            "an atomic factual proposition distinguished by its entity, event "
            "or state, time or stage, attribute, value, relation, reason, "
            "process, result, or opinion"
        ),
    },
    "ABSTRACT": {
        "goal": "retain supported nonredundant high-level abstractions",
        "target_unit": (
            "a person profile, event synopsis, entity relation, or stable "
            "high-level state that is not a paraphrase of its source children"
        ),
    },
    "SELECT": {
        "goal": "construct a minimal sufficient evidence set for the query",
        "target_unit": (
            "a query-requested answer value or distinct countable event instance"
        ),
        "comparison": (
            "candidate is tested for removal from the active evidence set; "
            "selected_set contains all other active evidence"
        ),
    },
}


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

    def estimate(
            self, operation, source, selected_set, candidate, context=None
    ):
        condition = {"operation": operation, **UTILITY_CONDITIONS[operation]}
        if context:
            condition["context"] = context
        payload = {
            "condition": condition,
            "source": source,
            "selected_set": selected_set,
            "candidate": candidate,
        }
        result = self._request_json(UTILITY_SYSTEM_PROMPT, payload, max_tokens=48)
        for field in ("new_target_unit", "necessary_constraint"):
            if not isinstance(result.get(field), bool):
                raise ValueError(f"invalid utility component: {field}")
        marginal_utility = int(
            result["new_target_unit"] or result["necessary_constraint"]
        )
        return {
            "new_target_unit": result["new_target_unit"],
            "necessary_constraint": result["necessary_constraint"],
            "marginal_utility": marginal_utility,
            "keep": marginal_utility > 0,
        }

    def missing_fact_indices(self, facts, drafts):
        result = self._request_json(
            "Compare each numbered source fact against the Child drafts. "
            "A fact is covered only if the drafts express its specific entity, "
            "event, relation and value. Preserve distinct emotions, exact names "
            "and nicknames, and temporal scope in this comparison. Sharing an "
            "origin or topic does not establish coverage. Return only JSON "
            'with a missing array of 0-based fact indices: {"missing": [0]}.',
            {
                "facts": [
                    {"index": index, "text": fact["text"]}
                    for index, fact in enumerate(facts)
                ],
                "drafts": [draft["text"] for draft in drafts],
            },
            max_tokens=16 + 5 * len(facts),
        )
        return set(result["missing"])

    def _request_json(self, prompt, payload, max_tokens):
        response = self.client.chat.completions.create(
            model=config.UTILITY_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        return json.loads(response.choices[0].message.content)


def get_unified_utility_judge():
    global _JUDGE
    if _JUDGE is None:
        with _JUDGE_LOCK:
            if _JUDGE is None:
                _JUDGE = UnifiedUtilityJudge()
    return _JUDGE

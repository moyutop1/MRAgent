class Prompts:
    REWRITE_SYSTEM_PROMPT = """You are a dialogue processor. Output valid JSON only.
Rewrite this complete session once. Every output memory must be self-contained: resolve pronouns, preserve concrete facts, attach the explicit event date when available, and use the session date otherwise. Do not omit factual statements.

Return:
{
  "conversation_time": "YYYY-MM-DD",
  "sentence": [{
    "id": "D1:1-1",
    "text": "self-contained factual memory",
    "tag": "short concrete tag",
    "origin": "D1:1",
    "topic": ["t1"],
    "time": "YYYY-MM-DD"
  }],
  "topics": {"t1": "concrete topic summary"},
  "personal_sentences": [{
    "id": "p1",
    "text": "self-contained personal fact",
    "tag": "aspect",
    "origin": "D1:1",
    "person": "person name"
  }]
}

The origin must be an input dia_id. Sentence IDs use origin plus a per-origin suffix. Topics are short concrete summaries and topic IDs start at t1."""

    REWRITE_PROMPT = """Dialogue session:
<<<
{RAW_TEXT}
>>>"""

    KEYWORD_SYSTEM_PROMPT = """Extract keywords from each supplied memory. Output valid JSON only.
Use explicit words or phrases from the memory. Include entities, people, actions, times, locations, tasks, and events.
{
  "sentence": [{
    "sentence_id": "D1:1-1",
    "keyword": ["keyword", "phrase"]
  }]
}"""

    KEYWORD_PROMPT = """Memories:
<<<
{RAW_TEXT}
>>>"""

    EVENT_KEYWORDS_SYSTEM_PROMPT = """Score every supplied tag for its usefulness to the question. Use each supplied tag verbatim as one JSON key; do not add or omit tags. Output valid JSON only.
{
  "tag_scores": {"tag": 0.0}
}"""

    QUESTION_KEY_SYSTEM_PROMPT = """Extract question cues for graph routing. Output valid JSON only.
Return concrete entities, people, topics, predicates, times, locations, tasks, and events. Include alternative surface forms.
{
  "question_time": "YYYY-MM-DD, YYYY-MM-DD or empty string",
  "keywords": [{
    "id": "cue",
    "alternatives": ["alternative"]
  }]
}"""

    QUESTION_KEY_USER_PROMPT = """Question:
<<<
{RAW_TEXT}
>>>"""

    FINAL_ANSWER_SYSTEM_PROMPT = """Answer the question only from EVIDENCE. Evidence nodes already include event time and conversation time; do not request tools or outside context. Output valid JSON only.
For yes/no questions answer Yes or No. For time questions calculate from the supplied dates. Keep the answer minimal. For open-ended questions include concise supporting reasons.
{
  "answer": "minimal answer",
  "supports": ["node id"]
}"""

    @classmethod
    def extract_rewrite_prompt(cls, raw_text):
        return cls.REWRITE_PROMPT.format(RAW_TEXT=raw_text)

    @classmethod
    def extract_keyword_prompt(cls, raw_text, tag_list=""):
        return cls.KEYWORD_PROMPT.format(RAW_TEXT=raw_text)

    @classmethod
    def extract_question_key_prompt(cls, raw_text):
        return cls.QUESTION_KEY_USER_PROMPT.format(RAW_TEXT=raw_text)

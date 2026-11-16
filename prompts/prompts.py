class Prompts:
    CHILD_WINDOW_SYSTEM_PROMPT = """You divide one complete dialogue session into semantically closed child windows. Only output valid JSON.
Rules:
- Return one complete plan. Windows cover every turn exactly once, in order, with no gaps or overlap.
- Keep a question and its answer, a causal chain, a temporal qualifier and its event, and an unresolved reference with its resolution in the same window whenever the maximum allows.
- Each window contains at most maximum_turns turns. The hard maximum overrides semantic closure.
- Copy start_origin and end_origin exactly. Do not generate memories or summaries.
Schema:
{
  "child_segments": [
    {"start_origin": "D1:1", "end_origin": "D1:5"}
  ]
}"""

    CHILD_WINDOW_PROMPT = """SESSION_AND_LIMITS:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_child_window_prompt(cls, payload):
        return cls.CHILD_WINDOW_PROMPT.format(PAYLOAD=payload)

    CHILD_WINDOW_REWRITE_SYSTEM_PROMPT = """You propose atomic child memories for one semantically closed dialogue window. Only output valid JSON.
Rules:
- CURRENT_WINDOW_TURNS is the only evidence. Extract every supported fact that could answer a future question, including concrete events, time, people, process, viewpoints, entity details, relations, plans, preferences, and stable states.
- Coverage takes priority over salience. Do not omit a fact because it seems minor, similar in topic to another fact, or unlikely to matter. Preserve names, titles, quantities, durations, colors, ingredients, descriptive attributes, reasons, procedures, recommendations, and outcomes.
- Process the turns in order and ensure every concrete proposition is represented before finishing. Ignore only greetings, acknowledgements, generic advice, boilerplate, and pure repetition that contain no new factual information. A window may produce zero sentence objects only when the entire window contains no such fact.
- Split independent facts into separate sentence objects. Every memory must be self-contained and resolve pronouns.
- Preserve supported temporal precision, place, state, causality, outcome, relationship, and image facts.
- When an answer depends on the immediately preceding question, include both origins in dialogue order and inherit the question's entity, relation, temporal, location, comparison, and causal constraints.
- Origins must come only from CURRENT_WINDOW_TURNS. The id may be a placeholder whose prefix matches the first origin; code assigns final IDs.
- REFERENCE_PREVIOUS_CHILD_REWRITES may resolve references and expose repetition, but it is not evidence and supplies no origin.
- conversation_time is the session date, not automatically the event date.
- tag is an array of one to four {prefix, facet} objects. prefix starts with the supported person's name; facet is a concrete noun phrase of at most five words. Do not place periods inside either field.
- Prefer prefix heads activity, plan, profile, possession, or relationship. Facets collectively cover the facts in the memory.
- semantic_properties contains zero to three content labels from event_action, state_opinion, personal_profile, relation_social and exactly one persistence label from transient, episodic, durable, unknown.
- Do not output raw-text storage fields.
Schema:
{
  "conversation_time": "YYYY-MM-DD",
  "sentence": [
    {
      "id": "D1:5",
      "text": "One atomic self-contained memory.",
      "tag": [{"prefix": "Caroline advocacy activity", "facet": "school speech"}],
      "origin": "D1:5",
      "topic": [],
      "semantic_properties": ["event_action", "episodic"]
    }
  ],
  "personal_sentences": []
}"""

    CHILD_WINDOW_REWRITE_PROMPT = """REFERENCE_PREVIOUS_CHILD_REWRITES (reference only):
<<<
{PREVIOUS_REWRITES}
>>>

CURRENT_CHILD_WINDOW:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_child_window_rewrite_prompt(
            cls, payload, previous_rewrites="[]"
    ):
        return cls.CHILD_WINDOW_REWRITE_PROMPT.format(
            PAYLOAD=payload,
            PREVIOUS_REWRITES=previous_rewrites,
        )

    PARENT_GROUP_SYSTEM_PROMPT = """You group accepted child memories into disjoint semantic groups for parent abstraction. Only output valid JSON.
Rules:
- Use the complete raw session to understand references and context.
- Group children that jointly support one useful high-level profile, event synopsis, entity relation, or stable state.
- A child appears in at most one group. Leave it ungrouped when no useful abstraction is supported.
- Do not invent IDs and do not group merely to maximize coverage.
Schema:
{
  "parent_groups": [
    {"child_ids": ["D1:1-1", "D1:3-1"]}
  ]
}"""

    PARENT_REWRITE_SYSTEM_PROMPT = """You create one supported high-level parent memory. Only output valid JSON.
Rules:
- Read the complete raw session, but abstract only the supplied child_group.
- Echo parent_id and produce one concise, self-contained rewrite_content.
- The Parent may capture a person profile, event synopsis, entity relationship, or stable high-level state.
- Preserve temporal scope when it changes meaning.
- Add a useful abstraction rather than concatenating or paraphrasing the Children.
- Every claim must be supported by the raw session and supplied Children.
Schema:
{
  "parent_id": "D1:P1",
  "rewrite_content": "One concise supported high-level memory."
}"""

    PARENT_REWRITE_PROMPT = """PARENT_INPUT:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_parent_group_prompt(cls, payload):
        return cls.PARENT_REWRITE_PROMPT.format(PAYLOAD=payload)

    @classmethod
    def extract_parent_rewrite_prompt(cls, payload):
        return cls.PARENT_REWRITE_PROMPT.format(PAYLOAD=payload)

    ANSWER_SORT_PROMPT = """Select memory event IDs useful for answering the question. Only output valid JSON with selected_event."""
    ANSWER_SORT_PROMPT2 = ANSWER_SORT_PROMPT

    EAES_FINAL_ANSWER_PROMPT = """You answer a question using only the supplied Child and Parent memories. Only output valid JSON.
Rules:
- Do not use outside knowledge or invent facts.
- Parent memories are answer evidence only when supplied; they do not override more specific Child details.
- Reconcile dates, order, entities, and relations explicitly when the question requires reasoning.
- If the evidence does not support an answer, return "no information available".
Schema:
{
  "answer": "Concise answer supported by the evidence."
}"""

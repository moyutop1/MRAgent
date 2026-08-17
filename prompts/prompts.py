# prompts.py
import json

class Prompts:
    REWRITE_SYSTEM_PROMPT = """You are a memory compressor for long-term conversational memory. Only output valid JSON.
TASK:
- Convert the dialogue window into compact rewrite memories, not a sentence-by-sentence transcript.
- Keep only answer-bearing information: user-specific facts, preferences, plans, completed events, times, places, people, relationships, decisions, task outcomes, and image/caption facts.
- Drop low-value content: greetings, acknowledgements, boilerplate, generic advice, repeated confirmations, and assistant text that contains no user-specific fact or task result.
- Each memory in "sentence" must be self-contained, explicit, and useful without the raw dialogue context.
- Resolve all pronouns ("I", "you", "he", "she", "it", "they", "we", "this", "that", "these", "those") into concrete people, objects, events, or noun phrases from the window.
- Keep top-level "conversation_time" equal to the dialogue's session date. Never replace it with an event date.
- For every memory, regardless of its semantic properties, if the occurrence time or validity time of the remembered information can be obtained from its source dialogue, include that time directly in the memory "text".
- For memories classified as "episodic", occurrence time is especially important: actively preserve it whenever it is available from any contributing source turn, and never omit it during compression or merging.
- A temporal qualifier contained in a question or surrounding turn must be carried into the memory text when it applies to the corresponding answer.
- Do not invent an occurrence time when the source dialogue does not provide one. Top-level conversation_time is an anchor for resolving relative expressions, not evidence that an event occurred on that date.
- Preserve temporal granularity in memory "text" with these exact rules (anchor = conversation_time):
  - Named weekdays, weeks, and weekends stay anchored-relative instead of becoming a calendar date: "last Friday" + anchor 2023-07-22 -> "the Friday before 22 July 2023"; "last week" -> "the week before 22 July 2023"; "last weekend" -> "the weekend before 22 July 2023".
  - Exact-day expressions become human-readable absolute dates: with anchor 2023-07-22, "yesterday" -> "21 July 2023" and "two days ago" -> "20 July 2023".
  - Month expressions keep month precision: "last month" -> "June 2023".
  - Year expressions keep year precision in text: "last year" -> "2022".
- If several adjacent turns describe the same fact/event, merge them into one dense memory, but do not merge otherwise similar events that occurred at different times.
- PREVIOUS_DIALOGUE_CONTEXT contains the tail of the preceding raw-dialogue window. Use it to resolve cross-window questions and answers, ellipsis, pronouns, entities, and qualifiers such as time and place.
- Create a memory only when CURRENT_DIALOGUE_WINDOW adds answer-bearing information. Never create a memory supported only by PREVIOUS_DIALOGUE_CONTEXT.
- Use "origin" as a comma-separated list of every source dia_id that contributes information to the memory, from either dialogue section. A cross-window question carrying a time/place/entity constraint and its answer must both be included, e.g. "D1:40,D1:41". Do not invent source ids.
- Use a short concrete noun phrase for "tag", e.g. Movie Preference, Support Group, Travel Plan. No more than three words.
- Classify every memory with one orthogonal "semantic_properties" array in the same rewrite call. Do not create a separate classification response.
  - Content properties (choose zero to three):
    - "event_action": a concrete action, event, plan, decision, or task outcome.
    - "state_opinion": a reaction, emotion, opinion, evaluation, or temporary state.
    - "personal_profile": person-centered characteristics such as interests, hobbies, occupation, education, skills, traits, residence, possessions, pets, or stable goals. It is not limited to preferences.
    - "relation_social": an interpersonal relationship, social role, membership, support, or interaction pattern.
  - Persistence property (choose exactly one):
    - "transient": momentary or short-lived state/reaction.
    - "episodic": bounded occurrence tied to a particular event or period.
    - "durable": relatively stable profile, preference, possession, role, relationship, or long-lived condition.
    - "unknown": persistence cannot be determined from the dialogue.
  - Use only the eight labels above, never "profile_preference" or "fact_background". Do not repeat a label.
- The "id" field may be any valid placeholder matching the first source id, because code will rewrite ids deterministically after validation.
- Use PREVIOUS_REWRITE_MEMORIES only to avoid repeating already-written memories; do not copy them unless CURRENT_DIALOGUE_WINDOW adds new information.
- Topics: derive concrete topic summaries from the memories in this window. Assign topic IDs (t1..tn). In each memory, fill "topic" with topic IDs that apply; use [] if none.
- Personal information: extract person-related stable facts into "personal_sentences". If a fact is already in a memory, also duplicate a concise normalized version here.
Schema:
{
  "conversation_time":"YYYY-MM-DD",
  "sentence":[
    {
      "id":"D1:1-1", 
      "text":"sentence.", 
      "tag":"short concrete tag",
      "origin":"D1:1",
      "topic": ["t1","t3"],
      "semantic_properties":["personal_profile","durable"]
    }
  ],
  "topics":{
    "t1": "Nate plans the charity race route",
    "t2": "Joanna discusses aquarium maintenance"
    }
  "personal_sentences":[{
  "id":"p1",
  "text":"Nate enjoys long-distance running.",
  "tag":"preference",
  "origin":"D1:1",
  "person": "Nate"
  }]
}
    """

    REWRITE_PROMPT = """PREVIOUS_REWRITE_MEMORIES:
<<<
{PREVIOUS_MEMORIES}
>>>

PREVIOUS_DIALOGUE_CONTEXT (context only; do not rewrite by itself):
<<<
{PREVIOUS_DIALOGUE_CONTEXT}
>>>

CURRENT_DIALOGUE_WINDOW (produce memories for new information here):
<<<
{RAW_TEXT}
>>>"""

    @classmethod
    def extract_rewrite_prompt(
            cls,
            raw_text: str,
            previous_memories: str = "[]",
            previous_dialogue_context: str = "[]",
    ) -> str:
        return cls.REWRITE_PROMPT.format(
            RAW_TEXT=raw_text,
            PREVIOUS_MEMORIES=previous_memories,
            PREVIOUS_DIALOGUE_CONTEXT=previous_dialogue_context,
        )

    PARENT_SEGMENT_SYSTEM_PROMPT = """You plan coarse semantic parent segments for one dialogue session. Only output valid JSON.
Rules:
- Return the complete plan for the entire session in one response.
- Parent segments must cover every input turn exactly once, in dialogue order, with no gaps or core overlap.
- Choose boundaries where a broad topic, event episode, or discourse unit is semantically closed.
- Avoid cutting an unresolved question/answer pair, pronoun reference, temporal qualifier, or causal explanation when a legal alternative exists.
- Each turn has a one-based position. Segment length is end position minus start position plus one; count with these positions rather than estimating from the text.
- Return at least minimum_segment_count segments. If total_turns exceeds maximum_turns, returning the whole session as one segment is invalid even when it is semantically coherent.
- Never place more than 10 dialogue turns in one parent segment. The supplied maximum_turns may be lower but can never be higher than 10.
- Every non-final segment must respect minimum_turns and every segment must respect maximum_turns.
- Only the final segment may be shorter than minimum_turns.
- Hard length limits override the preference for semantic closure. Within the legal length range, place the boundary at the best semantic closure.
- Copy start_origin and end_origin exactly from the input; do not generate summaries or new IDs.
Schema:
{
  "parent_segments": [
    {"start_origin": "D1:1", "end_origin": "D1:10"}
  ]
}"""

    PARENT_SEGMENT_PROMPT = """SESSION_AND_LIMITS:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_parent_segment_prompt(cls, payload: str) -> str:
        return cls.PARENT_SEGMENT_PROMPT.format(PAYLOAD=payload)

    CHILD_WINDOW_SYSTEM_PROMPT = """You divide one complete dialogue session into semantically closed child windows. Only output valid JSON.
Rules:
- Return the complete window plan for the entire session in one response.
- Parent planning is independent and is not provided here.
- Child windows must cover every input turn exactly once, in dialogue order, with no gaps or core overlap.
- A window may contain one or more consecutive turns. Choose its end boundary from semantic closure, topic continuity, completed question/answer structure, resolved references, temporal qualifiers, and completed causal explanations.
- Avoid cutting an unresolved question/answer pair, pronoun reference, temporal qualifier, or causal explanation when a legal alternative exists.
- Every window must contain at most maximum_turns turns. Hard length limits override semantic closure.
- Return at least minimum_segment_count windows. Returning the complete session as one window is invalid when total_turns exceeds maximum_turns.
- Copy start_origin and end_origin exactly from the input. Do not generate window IDs, summaries, focuses, source lists, or memories.
Schema:
{
  "child_segments": [
    {
      "start_origin": "D1:1",
      "end_origin": "D1:5"
    }
  ]
}"""

    CHILD_WINDOW_PROMPT = """SESSION_AND_LIMITS:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_child_window_prompt(cls, payload: str) -> str:
        return cls.CHILD_WINDOW_PROMPT.format(PAYLOAD=payload)

    PARENT_REWRITE_SYSTEM_PROMPT = """You create one person-centric profile memory from one dialogue segment. Only output valid JSON.
Rules:
- Produce exactly one concise, self-contained rewrite_content for PARENT_DIALOGUE_WINDOW and echo parent_id exactly.
- rewrite_content is a PERSON PROFILE MEMORY, not a dialogue summary, event memory, timeline, or turn-by-turn recap. Raw events are only evidence from which to extract profile-level information.
- Prioritize explicitly supported personality traits and recurring tendencies; likes, dislikes, interests, hobbies, and values; occupation, skills, long-term goals, and sustained pursuits; stable possessions or pets; and interpersonal relationships or recurring support patterns.
- Prefer direct person-centered clauses such as "Joanna is...", "Joanna likes...", "Nate has...", or "Nate supports...". Name the person instead of using ambiguous pronouns.
- Convert event evidence into a durable or ongoing profile fact when justified. For example, repeated writing effort and difficulty switching off may support "Joanna is a dedicated screenwriter who struggles to disengage from her work"; do not retell which script she finished or what happened next.
- Do not narrate completed actions, conversations, reactions, congratulations, photos, task steps, event outcomes, or sequences of who said or did what. Omit such details unless they directly establish a stable preference, possession, relationship, or sustained pursuit, and then state only that profile fact.
- Do not record any temporal information. Omit dates, years, months, weekdays, clock times, ages, durations, relative-time expressions, conversation_time, event ordering, and temporal calculations even when they appear in the dialogue.
- Do not turn a one-off action, temporary emotion, isolated statement, or single polite response into a durable personality trait, broad preference, or relationship claim. Keep claims as narrow as the evidence requires; for example, evidence about turtles supports liking turtles, not necessarily liking all animals.
- If one participant has no supported profile information in this window, omit that participant instead of filling the rewrite with their event reactions. When profile evidence is sparse, return only the narrowest supported profile fact rather than an event recap.
- Ignore greetings, acknowledgements, boilerplate, generic advice, and repeated confirmations.
- PREVIOUS_DIALOGUE_CONTEXT may resolve references but cannot independently support a claim.
- Do not invent information and do not generate child IDs, attributes, topics, or semantic properties.
Style calibration:
- INVALID event-summary style: "On 2022-03-18, Joanna finished her second script, felt anxious, and Nate congratulated her and shared a tortoise photo."
- VALID profile-memory style: "Joanna is a dedicated screenwriter who struggles to switch off from her work and balances ambition with self-doubt. Nate is drawn to turtles, keeps them as calming pets, and consistently supports Joanna's writing ambitions."
Schema:
{
  "parent_id": "D1:t1",
  "rewrite_content": "One concise person-centric profile memory."
}"""

    PARENT_REWRITE_PROMPT = """PARENT_INPUT:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_parent_rewrite_prompt(cls, payload: str) -> str:
        return cls.PARENT_REWRITE_PROMPT.format(PAYLOAD=payload)

    CHILD_WINDOW_REWRITE_SYSTEM_PROMPT = """You create exhaustive atomic child memories for one semantically closed dialogue window. Only output valid JSON.
Rules:
- CURRENT_WINDOW_TURNS is the only evidence section. Rewrite every turn and every piece of information in that section; nothing may be omitted, even greetings, questions, acknowledgements, generic advice, repeated confirmations, repeated facts, or image/caption information.
- A single turn containing several independent pieces of information must produce several sentence objects. A window may therefore produce one or many sentence objects.
- Preserve every repeated occurrence as a separate memory. Never deduplicate against another sentence or against REFERENCE_PREVIOUS_CHILD_REWRITES. If the current window repeats a referenced fact, write a new memory using only the current occurrence's origin.
- REFERENCE_PREVIOUS_CHILD_REWRITES contains only the two most recently generated child rewrite texts. Use it only to resolve people, objects, topics, pronouns, and ellipsis in the current window. It is not evidence, must not independently produce a memory, and must never supply an origin.
- Every sentence must be self-contained. Resolve pronouns into concrete entities and preserve all source-supported people, relationships, time, place, state, causality, task outcomes, questions, responses, and image facts.
- Every origin must come from CURRENT_WINDOW_TURNS, must list all current-window turns contributing to that memory in dialogue order, and must never cite a reference-only rewrite.
- Across the complete sentence list, every turn in CURRENT_WINDOW_TURNS must appear in at least one origin.
- The id may be any placeholder whose prefix matches the first origin; code assigns deterministic final IDs after generation.
- Keep conversation_time equal to the supplied session date; it is not automatically an event occurrence date.
- Preserve source-supported temporal information directly in text using the same precision as the dialogue.
- Use a short concrete tag of at most three words and set topic to [].
- semantic_properties may contain zero to three content labels from event_action, state_opinion, personal_profile, relation_social and exactly one persistence label from transient, episodic, durable, unknown.
- Do not output raw_text, raw_content, source_text, current_turns, dialogue text, or any other raw-text storage field.
Schema:
{
  "conversation_time": "YYYY-MM-DD",
  "sentence": [
    {
      "id": "D1:5",
      "text": "One atomic self-contained memory.",
      "tag": "short tag",
      "origin": "D1:5",
      "topic": [],
      "semantic_properties": ["event_action", "episodic"]
    }
  ],
  "topics": {},
  "personal_sentences": []
}"""

    CHILD_WINDOW_REWRITE_PROMPT = """REFERENCE_PREVIOUS_CHILD_REWRITES (reference only; never use as origin):
<<<
{PREVIOUS_REWRITES}
>>>

CURRENT_CHILD_WINDOW:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_child_window_rewrite_prompt(
            cls, payload: str, previous_rewrites: str = "[]"
    ) -> str:
        return cls.CHILD_WINDOW_REWRITE_PROMPT.format(
            PAYLOAD=payload,
            PREVIOUS_REWRITES=previous_rewrites,
        )


    KEYWORD_SYSTEM_PROMPT = """You are an information extraction system. Only output valid JSON.
Keyword Extraction
- For each input sentence, extract 2–30 keywords DIRECTLY from the original text, such as "drew", "park", "lake sunrise". Do not invent , paraphrase, or generalize. Do not include inferred words unless they explicitly appear in the text.
- Keyword types to consider: entity | topic | verb | time | location | task | event | people.
- For each sentence, extract ALL words/phrases that match these types if they are explicitly present.
- "sentence_id" must be same with "id" in TEXT. Do not create or invent new ids.
Prefer using single quotes (e.g., 'at a time'), or use escaped double quotes (e.g., \"at a time\")
Generate a JSON object strictly following the given schema, no extra text.
Schema:
{
  "sentence":[
    {
      "sentence_id":"D1:1-1",
      "keyword":["Coraline","park"]
    }
  ]
}

    """

    KEYWORD_PROMPT = """TEXT:
        <<<
        {RAW_TEXT}
        >>>
        """

    @classmethod
    def extract_keyword_prompt(cls, raw_text: str) -> str:
        return cls.KEYWORD_PROMPT.format(
            RAW_TEXT=raw_text
        )

    """   -  For time-related questions (e.g., "When…", "What date…"), call query_conversation_time, output the answer only as an absolute date or relative date grounded to query conversation time. Format must be: '7 May 2023', 'May 2023', '2023','The week/Sunday before 25 May 2023' and no extra word.
    """

    ANSWER_SORT_PROMPT = """You are a careful QA reasoner working over a memory of timestamped events. For every event in top_texts, produce a relevance score in [0.0, 1.0] reflecting how useful it is for answering question, do not make up event id:
    {
      "mode": "score",
      "relevance_scores": {
        "D1:1": 0.0-1.0,
      },
    } DO NOT output extra explanation."""

    ANSWER_SORT_PROMPT2 = """You are a careful QA reasoner working over a memory of timestamped events. For every event in top_texts, select at most 20 relevant events for answering question, do not make up event id:
        {
          "mode": "sort", 
          "events": ["D1:1","D1:2"]
        } DO NOT output extra explanation."""


    EAES_QUERY_SYSTEM_PROMPT = """You are a query parser for long-term conversational memory. Only output valid JSON.
Extract fields for answer-oriented evidence selection.
Schema:
{
  "entities": ["person or entity names"],
  "query_attributes": ["semantic.path: question-side relation clause"],
  "answer_type": "event_list | time | person | location | reason | state | fact | yes_no | unknown",
  "temporal_intent": "historical_event | planned_event | current_state | relative_time | time_answer | none",
  "required_lifecycle": "planned | current | historical | unknown",
  "no_time_limit": true,
  "keywords": [
    {
      "keyword": "important lexical constraint",
      "alternatives": ["synonym", "tense-aware synonym", "alternate wording"]
    }
  ]
}
Rules:
- Use "historical" when the question asks what happened, what someone did, or what events someone attended.
- Use "planned" when the question asks about intentions, plans, scheduled future events, or going to do something.
- Use "current" when the question asks about now, currently, still, preferences, roles, residence, or ongoing state.
- Use "unknown" with no_time_limit=true for stable fact/profile questions without an explicit temporal or event constraint, such as identity, relationship status, preferences, interests, activities, membership, allyship, career fields, or kinds/types of art.
- Generate 1-3 query_attributes using only the question. Never use or assume an answer.
- Each query_attribute must be a compact retrieval intent with a semantic path and an answer-slot relation clause, e.g. "object.symbolism: symbolism of Caroline's necklace" or "event.activity: activities Melanie's family did while camping".
- Keep named entities and concrete relation words from the question. Do not output bare keywords.
- For each keyword that is not a likely person name, generate exactly three distinct alternatives. Alternatives may use synonyms or synonymous verb forms in different tenses, but must preserve the original keyword's meaning and must not guess the answer.
- If a keyword is likely a person's name, keep it as the keyword and return an empty alternatives list for it. Do not generate aliases, nicknames, or other alternatives for likely person names.
- Do not answer the question."""

    EAES_SEMANTIC_QUERY_EXTENSION = """

Additionally infer the semantic memory properties required by the question.
Add this field to the JSON object:
  "required_semantic_properties": ["content property", "persistence property"]
Rules:
- Content properties are: "event_action", "state_opinion", "personal_profile", "relation_social". Select only properties that the answer evidence needs.
- "personal_profile" covers person-centered interests, hobbies, occupation, education, skills, traits, residence, possessions, pets, preferences, and stable goals.
- Persistence properties are: "transient", "episodic", "durable". Select the best required persistence when the question supports one.
- Never output "unknown", "profile_preference", or "fact_background".
- Use only the seven allowed query labels above and do not repeat a label.
- This field describes evidence requirements; it must not answer the question."""

    EAES_INDEX_SYSTEM_PROMPT = """You build an entity-attribute-memory index for long-term conversational memory. Only output valid JSON.
For each memory sentence, identify:
- entities: people, organizations, communities, named objects, or concrete concepts central to retrieving the memory.
- attributes: small answer-bearing relation clauses connecting an entity to the memory. Each attribute must include a compact semantic path and a natural-language description.

Rules:
- Use only information present in the given memory sentence/raw text.
- Keep entity names explicit, e.g. "Caroline", not pronouns.
- Attribute names should be short dotted paths, e.g. career.interest, education.field, mental_health.counseling, adoption.plan, event.attendance.
- Attribute descriptions should be concise clauses preserving important nouns and verbs, e.g. "Caroline is interested in counseling and mental health as a career."
- Do not output bare keywords, tags, topic ids, or one-word attributes. Every attribute must be useful as a small standalone evidence sentence.
- Include 1-6 entities and 1-8 attributes per memory.
- Copy event_id exactly from input.
- event_lifecycle is one of: planned, current, historical.

Schema:
{
  "memories": [
    {
      "event_id": "D1:9-1",
      "entities": ["Caroline"],
      "attributes": [
        {"name": "career.interest", "description": "Caroline is interested in counseling and mental health as a career."}
      ],
      "event_lifecycle": "current"
    }
  ]
}"""

    EAES_INDEX_USER_PROMPT = """MEMORY_SENTENCES:
{MEMORIES}"""

    @classmethod
    def eaes_index_prompt(cls, memories: str) -> str:
        return cls.EAES_INDEX_USER_PROMPT.format(MEMORIES=memories)

    EAES_ATTRIBUTE_RERANK_PROMPT = """You rerank memory candidates using structured attributes. Only output valid JSON.
Use only the question, query_attributes, memory attribute_paths, and prefilter rank/score.
Do not answer the question. Do not invent memory IDs.
Prefer memories whose attributes directly contain the relation needed to fill the question's answer slot.
Keep complementary attribute evidence for multi-hop and list questions.
Return memory IDs in descending relevance order, with at most the requested limit.
Schema:
{
  "ranked_memory_ids": ["M_D1_2_1"]
}"""

    EAES_ROLLBACK_QUERY_PROMPT = """The input also contains the current query plan and only the rewrite contents of the 20 memories retrieved in the first pass.
Use those rewrite contents only to identify answer-relevant evidence that may still be missing. Produce one replacement query plan that searches for complementary evidence outside the current memories.
Keep the same query-plan schema and constraints. Do not add fields, answer the question, assume that a retrieved memory is true, or put a guessed answer into a query attribute.
Prefer alternate entities, relations, wording, temporal constraints, or answer-slot descriptions only when they are justified by the original question and address a real gap in the current memories."""

    EAES_ROLLBACK_SUPPLEMENT_RERANK_PROMPT = """You select complementary memory nodes for a retrieval rollback check. Only output valid JSON.
The input contains two separately prefiltered groups: child_candidates has up to 27 child memories and parent_candidates has up to 3 parent memories. Their scores are meaningful only within the same node type; never compare child and parent numeric scores directly.
Use the question, current query plan, rollback query plan, current top rewrite contents, and candidate contents. Select the requested total number of nodes that add the strongest missing answer evidence. Do not select a node merely because it repeats evidence already present. Do not invent IDs.
Return exactly limit distinct nodes when at least limit candidates are provided.
Schema:
{
  "ranked_nodes": [
    {"node_type": "child", "node_id": "M_D1_2_1"},
    {"node_type": "parent", "node_id": "D1:t2"}
  ]
}"""

    EAES_ROLLBACK_FINAL_RERANK_PROMPT = """You rerank one merged child candidate pool and one merged parent candidate pool. Each pool contains the first-pass memories plus any supplemental rollback memories of that type. Only output valid JSON.
Rank child and parent memories separately. Retrieval ranks and scores are only weak hints because candidates may have been retrieved by different query plans. Prefer nodes that directly answer the question, preserve complementary evidence, and avoid redundancy. A rollback node may replace a first-pass node only when it is more useful for answering the question. Do not invent IDs.
Return exactly child_limit distinct child IDs and exactly parent_limit distinct parent IDs when each pool is large enough.
Schema:
{
  "ranked_child_ids": ["M_D1_2_1"],
  "ranked_parent_ids": ["D1:t2"]
}"""

    EAES_EVIDENCE_SELECTION_PROMPT = """You select compact answer evidence from retrieved memory notes. Only output valid JSON.
Goal: select valid answer evidence, not merely related memories.
Consider entity match, attribute match, answer type, lifecycle compatibility, temporal usability, facet specificity, answer density, low redundancy, and coverage.
Be recall-friendly: if at least one candidate plausibly helps answer the question, select it. Do not return an empty answer_items list merely because the evidence is imperfect.
Use planned/current/historical carefully:
- planned evidence can support plan/future questions.
- current evidence can support current-state questions.
- historical evidence can support happened/attended/did questions.
- For list-answer questions, cluster memories by possible answer item.
Output schema:
{
  "need_raw_expansion": true,
  "memory_ids_to_expand": ["M_D1_2_1"],
  "reason": "short reason",
  "answer_items": [
    {
      "item": "candidate answer item",
      "score": 0.0,
      "evidence": [
        {
          "memory_id": "M_D1_2_1",
          "role": "direct_evidence | specificity_evidence | temporal_anchor | lifecycle_evidence | background",
          "rationale": "short reason"
        }
      ]
    }
  ]
}
Limits:
- Select at most 8 answer_items.
- Select at most 3 memories per answer_item.
- Prefer direct evidence; use complementary pairs only when one memory supplies specificity and another supplies lifecycle/completion."""

    TEMPORAL_ANSWER_POLICY = """Temporal output policy:
- Preserve the temporal granularity and relation stated by the source memory. The conversation anchor explains what a relative phrase refers to; it is not automatically the answer.
- After identifying the event, call query_conversation_time when its conversation anchor is not already available.
- For named weekdays, weeks, and weekends, use an anchored-relative phrase instead of calculating the calendar date: "last Friday" + anchor 2023-07-15 becomes "The Friday before 15 July 2023"; "last week" becomes "The week before 15 July 2023". Do not output an ISO date for these expressions.
- For day-exact expressions such as "yesterday" or "two days ago", an explicit source date, or a question explicitly asking for the exact date, return a human-readable absolute date such as "7 May 2023", never YYYY-MM-DD.
- For month/year expressions, preserve their natural precision, such as "June 2023" or "2022". For duration questions, return the duration as stated, such as "10 years ago"; do not turn it into a date.
- Return one minimal time expression and no explanation."""

    EAES_FINAL_ANSWER_PROMPT = """You answer from an EAES evidence package. Only output valid JSON.
Use the structured evidence package as the primary context.
Rules:
- Give the minimal answer requested by the question.
- State the answer directly. Never preface it with phrases such as "The original text states", "the memory says", "the rewrite says", "according to the evidence", or similar source-reporting language.
- Each child evidence object contains only memory_id, conversation_time, and rewrite_content. memory_id is only for supports; conversation_time is a dialogue anchor, not automatically the event occurrence time.
- parent_memories, when present, are independently retrieved coarse-grained rewrite memories. They are direct supporting context and do not restrict or rank the child evidence package.
- A relevant parent memory may support the answer even when its children are absent from the child candidate list. Cite its parent_id in supports when used.
- For list questions, return a concise comma-separated list.
- Treat evidence_package as primary evidence. Use backup_candidates only when evidence_package is empty or clearly insufficient.
- For time questions, preserve the source wording and its precision. Use conversation_time only when a relative expression needs that reference point.
- For a single-time question, return exactly one best time expression, not a list of multiple candidate dates.
- If the question asks for an exact date, output a human-readable absolute date like "10 July 2023".
- When multiple candidates mention similar events, choose the one whose entity, event type, month/season, and wording best match the question; do not merge conflicting times.
- Do not use planned-only evidence to answer a historical/completed question unless paired with historical evidence.
- If evidence_package has answer_items or backup_candidates, make the best answer supported by them instead of saying "no information available".
- Use "no information available" only when there is no relevant evidence at all.
- If the exact wording differs from the gold answer, prefer a short normalized phrase over a full sentence.
Schema:
{
  "mode": "answer",
  "answer": "...",
  "supports": ["memory_id"],
  "confidence": 0.0
}"""

    # -------- fact extraction --------



    # -------- multi-perspective key generation --------



    # -------- Questions → keys --------







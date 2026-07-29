# prompts.py
import json

class Prompts:

    EVENT_KEYWORDS_SYSTEM_PROMPT = """You are going to answer a question with keyword and corresponding tags(fact summary).  For every tag of key, produce a relevance score in [0.0, 1.0] reflecting how useful it is for answering question:
    {
      "keyword": "Caroline",
      "tag_scores": {
        "Plan": 0.0-1.0,
        "Conference": 0.0-1.0
      },
    }"""




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

    CHILD_SEGMENT_SYSTEM_PROMPT = """You plan atomic, answer-bearing child semantic segments for one dialogue session. Only output valid JSON.
Rules:
- Return the complete child plan for the entire session in one response.
- Parent planning is independent and is not provided here.
- Each child segment describes exactly one fact, event, plan, state, preference, relationship, decision, task result, or image/caption fact.
- source_origins must contain the real dia_ids whose text supplies that semantic unit, in dialogue order.
- The raw turn span from the first through last source origin must not exceed maximum_turns.
- Several turns may form one semantic unit. One turn containing several independent facts may appear in several child segments with different focus values.
- Keep child segments ordered by their first source origin. Do not duplicate an identical source_origins plus focus pair.
- Omit greetings, acknowledgements, boilerplate, generic advice, repeated confirmations, and other low-value turns rather than creating child segments for them.
- Copy source origins exactly. Do not rewrite the memories and do not invent IDs.
Schema:
{
  "child_segments": [
    {
      "source_origins": ["D1:3", "D1:5"],
      "focus": "Caroline was inspired by transgender stories at the support group"
    }
  ]
}"""

    CHILD_SEGMENT_PROMPT = """SESSION_AND_LIMITS:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_child_segment_prompt(cls, payload: str) -> str:
        return cls.CHILD_SEGMENT_PROMPT.format(PAYLOAD=payload)

    PARENT_REWRITE_SYSTEM_PROMPT = """You create one coarse-grained parent memory from one dialogue segment. Only output valid JSON.
Rules:
- Produce exactly one self-contained parent rewrite for PARENT_DIALOGUE_WINDOW and echo parent_id exactly.
- Produce a high-level dialogue overview centered on the people involved, their explicitly supported personality characteristics, preferences or values, and their relationships or social dynamics.
- Describe the main discussion theme and durable interpersonal context rather than listing atomic events, actions, task steps, or detailed outcomes.
- Do not record any temporal information. Omit dates, years, months, weekdays, clock times, ages, durations, relative-time expressions, conversation_time, event ordering, and temporal calculations even when they appear in the dialogue.
- Do not turn a one-off action, temporary emotion, or isolated statement into a durable personality trait or relationship claim. Use only explicit or repeatedly supported high-level information.
- Ignore greetings, acknowledgements, boilerplate, generic advice, and repeated confirmations.
- PREVIOUS_DIALOGUE_CONTEXT may resolve references but cannot independently support a claim.
- Do not invent information and do not generate child IDs, attributes, topics, or semantic properties.
Schema:
{
  "parent_id": "D1:t1",
  "rewrite_content": "One self-contained coarse-grained memory."
}"""

    PARENT_REWRITE_PROMPT = """PARENT_INPUT:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_parent_rewrite_prompt(cls, payload: str) -> str:
        return cls.PARENT_REWRITE_PROMPT.format(PAYLOAD=payload)

    CHILD_BATCH_REWRITE_SYSTEM_PROMPT = """You create atomic child memories from pre-segmented dialogue windows. Only output valid JSON.
Rules:
- Produce exactly one sentence object for every input child, preserve input order, and echo each child_id exactly.
- Never omit, duplicate, merge, split, reorder, or add a child.
- Each rewrite must express only its child's focus as one self-contained answer-bearing memory.
- Resolve pronouns into concrete entities and preserve source-supported people, relationships, time, place, state, causality, and task outcomes.
- origin is the comma-separated list of all dia_ids that directly contribute to the memory. Use only IDs present in that child's current_dialogue_window.
- origin must start with the child window's first source origin so its deterministic child_id remains provenance-aligned. Previous context may resolve a reference but cannot be cited as independent support; the child planner must place every contributing turn inside the current child window.
- Keep conversation_time equal to the supplied session date; it is not automatically an event occurrence date.
- Preserve source-supported temporal information directly in text using the same precision as the dialogue.
- Use a short concrete tag of at most three words and set topic to [].
- semantic_properties may contain zero to three content labels from event_action, state_opinion, personal_profile, relation_social and exactly one persistence label from transient, episodic, durable, unknown.
- personal_sentences may duplicate stable person facts, but it does not change the required one-to-one sentence output.
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

    CHILD_BATCH_REWRITE_PROMPT = """PREVIOUS_REWRITE_MEMORIES:
<<<
{PREVIOUS_MEMORIES}
>>>

CHILD_BATCH (at most 15 children):
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_child_batch_rewrite_prompt(
            cls, payload: str, previous_memories: str = "[]"
    ) -> str:
        return cls.CHILD_BATCH_REWRITE_PROMPT.format(
            PAYLOAD=payload,
            PREVIOUS_MEMORIES=previous_memories,
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
    def extract_keyword_prompt(cls, raw_text: str, tag_list:str) -> str:
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


    ANSWER_SYSTEM_PROMPT_FINAL = """
       You must answer the question with queried contents.
          Rules:
          -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'.    
          -  For "where / location / place" questions, the answer should be a concrete and specific place name. If no exact name is mentioned, describe it instead.
          -  For "what / which" questions, try to respond with one specific, concrete item directly asked for or descriptions of the answer. 
          -  For other questions, output only the minimal answer (key phrase or entity) without extra context.

           Format:
           - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.   
           {
             "mode": "answer",
             "answer": "...", 
             "supports": ["D1:1","D1:2"],
             "confidence": 0.0-1.0
           }  """

    # tool-loop system prompt: one shared base template + each variant declares only its differences (locomo / category-3 / LM).
    # placeholders <<INTRO>>/<<WHERE>>/<<WHAT>>/<<EXTRA>>/<<NAV>> are filled per variant.
    _ANSWER_TOOL_BASE = """You are a diligent question-answering agent. You always want to gather and verify all relevant information before producing your final answer.
<<INTRO>>
   Rules:
   -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'.
   -  For "where / location / place" questions, <<WHERE>>
   -  For "what / which" questions, <<WHAT>>
<<EXTRA>>   -  For other questions, output only the minimal answer (key phrase or entity) without extra context.
   -  There may be multiple answers, you should try to explore more relevant information.

    Decide ONE mode of:
    - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.
   If the information is vague or incomplete, you may further query_personal_information, query_topic_events, query_event_keywords or query_event_context.
    {
      "mode": "answer",
      "answer": "...",
      "supports": ["D1:1","D1:2"],
      "confidence": 0.0-1.0
    }

    - "navigate": If evidence is insufficient, <<NAV>>immediately call the tools with the proper argument. Do NOT describe the call in text or JSON. In each round, you must call as many relevant tools as possible, rather than skipping potential ones. Only avoid calls that are clearly irrelevant."""

    _WHERE_STRONG = "the answer must be a concrete and specific place name. If the sentence only provides a vague or ambiguous location, call query_event_keywords to further explore and identify a more specific place."
    _WHERE_SIMPLE = "the answer should be a concrete and specific place name."
    _WHAT_STRONG = "respond with one specific, concrete item (an event, subject, person, organization, place, or titled work) directly asked for\u2014not a category, type, or class."
    _WHAT_SIMPLE = "respond with one specific, concrete item (an event, subject, person, organization, place, or titled work) directly asked for."
    _EXTRA_LM = ('   -  For "how many" questions, the answer must be the number of tasks/objects, not the number of physical categories.\n'
                 '   -  For temporal questions (e.g., "How many days/weeks/months ago...", "How many days passed between..."), use "current_date" in the input as TODAY\'s date to calculate the time difference. Example: if current_date is "2023-02-01" and an event happened on "2023-01-25", then "7 days ago" is the answer.\n')

    # locomo: strict (exact words / strong where+what)
    ANSWER_SYSTEM_TOOL_PROMPT = (_ANSWER_TOOL_BASE
        .replace("<<INTRO>>", "You need to answer a question with key candidates and corresponding tags and similar_sentence. Write short answer with exact words from event whenever possible.")
        .replace("<<WHERE>>", _WHERE_STRONG).replace("<<WHAT>>", _WHAT_STRONG)
        .replace("<<EXTRA>>", "").replace("<<NAV>>", ""))

    # category-3 (adversarial): lenient (inferred / simple where+what)
    ANSWER_SYSTEM_TOOL_PROMPT3 = (_ANSWER_TOOL_BASE
        .replace("<<INTRO>>", "You need to answer a question with key candidates and corresponding tags and similar_sentence. Write short answer infered from sentences.")
        .replace("<<WHERE>>", _WHERE_SIMPLE).replace("<<WHAT>>", _WHAT_SIMPLE)
        .replace("<<EXTRA>>", "").replace("<<NAV>>", ""))

    # LM: key_sentence + how-many/temporal rules + forced navigate
    ANSWER_SYSTEM_TOOL_PROMPT_LM = (_ANSWER_TOOL_BASE
        .replace("<<INTRO>>", "You need to either answer a question or call tools to get more information with key candidates and corresponding tags and key_sentence. Write short answer with exact words from event whenever possible.")
        .replace("<<WHERE>>", _WHERE_STRONG).replace("<<WHAT>>", _WHAT_STRONG)
        .replace("<<EXTRA>>", _EXTRA_LM).replace("<<NAV>>", "you must "))




    """Requirements:
    - In navigate mode, you must have  multiple tool calls. 
    - For "what / which / about" questions, the final answer must contain a specific topic or subject, not just a restatement like "do some research". If no concrete topic is available in current evidence, you must switch to "navigate" mode and query event context/keywords and edges_by_tag with tags like "Research", "Topic", "Subject", or "Project".
    - Only after exhausting all relevant tool queries (event context, keywords, edges_by_tag across related keys and tags) and still finding no clear evidence, you may finally answer with "unknown" or "cannot be determined".
    """














    QUESTION_KEY_SYSTEM_PROMPT = """You are a keyword extractor. Only output valid JSON. 
    Given a question, extract keywords to find answers. Keyword types (all must be extracted for each question): entity | topic | predicate | time | location | task | event | people. For each keyword, also provide possible alternative expressions, including Synonyms, different form, different tense. Different tense of word is mandatory.
    If the question contains a time limit, return it in "question_time" as: "YYYY-MM-DD, YYYY-MM-DD". If no time info, set "question_time" as "". If no year, then write "MM-DD, MM-DD".If a single day, repeat the same date(e.g., "YYYY-MM-DD, YYYY-MM-DD").
    If no year appears, DO NOT guess or infer a year. Use only 'MM-DD, MM-DD'.
    Schema:
    {
      "question_time": "YYYY-MM-DD, YYYY-MM-DD or '' or MM-DD, MM-DD",
      "keywords": [
        {
          "id": "Extracted keyword",
          "alternatives": ["Possible tense", "Different Synonyms", "Different form"]
        }
      ] 
    }"""

    QUESTION_KEY_USER_PROMPT = """QUESTION:
        <<<
        {RAW_TEXT}
        >>>"""

    @classmethod
    def extract_question_key_prompt(cls, raw_text: str) -> str:
        return cls.QUESTION_KEY_USER_PROMPT.format(
            RAW_TEXT=raw_text,
        )

    QUESTION_KEY_INVENTORY_SYSTEM_PROMPT = """You select retrieval keys for a question from an existing memory-key inventory. Only output valid JSON.
Rules:
- You MUST choose keywords only from the provided candidates' "key" values. Copy the key exactly.
- Do not invent, paraphrase, translate, stem, or normalize keys.
- Prefer keys that are likely to retrieve answer-bearing evidence, including entity keys and concrete field/topic/action keys.
- Avoid selecting too many generic keys. Select 2-12 keys when useful.
- If no candidate is useful, return an empty keywords list.
- If the question contains a time limit, return it in "question_time" as "YYYY-MM-DD, YYYY-MM-DD". If no time info, set "question_time" as "".
- If no year appears, do not guess a year. Use only "MM-DD, MM-DD".
Schema:
{
  "question_time": "YYYY-MM-DD, YYYY-MM-DD or '' or MM-DD, MM-DD",
  "keywords": [
    {
      "id": "exact candidate key",
      "alternatives": []
    }
  ]
}"""

    QUESTION_KEY_INVENTORY_USER_PROMPT = """QUESTION:
<<<
{QUESTION}
>>>

CANDIDATE_KEYS:
{CANDIDATES}"""

    @classmethod
    def select_question_key_prompt(cls, question: str, candidates: str) -> str:
        return cls.QUESTION_KEY_INVENTORY_USER_PROMPT.format(
            QUESTION=question,
            CANDIDATES=candidates,
        )

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
  "keywords": ["important lexical constraints"]
}
Rules:
- Use "historical" when the question asks what happened, what someone did, or what events someone attended.
- Use "planned" when the question asks about intentions, plans, scheduled future events, or going to do something.
- Use "current" when the question asks about now, currently, still, preferences, roles, residence, or ongoing state.
- Use "unknown" with no_time_limit=true for stable fact/profile questions without an explicit temporal or event constraint, such as identity, relationship status, preferences, interests, activities, membership, allyship, career fields, or kinds/types of art.
- Generate 1-3 query_attributes using only the question. Never use or assume an answer.
- Each query_attribute must be a compact retrieval intent with a semantic path and an answer-slot relation clause, e.g. "object.symbolism: symbolism of Caroline's necklace" or "event.activity: activities Melanie's family did while camping".
- Keep named entities and concrete relation words from the question. Do not output bare keywords.
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







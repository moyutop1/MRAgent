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
- Normalize relative time using conversation_time and output "time" as YYYY-MM-DD. If no event time is mentioned, use conversation_time.
- If several adjacent turns describe the same fact/event, merge them into one dense memory.
- Use "origin" as a comma-separated list of the exact source dia_id values copied from this window, e.g. "D1:12,D1:13". Do not invent source ids.
- Use a short concrete noun phrase for "tag", e.g. Movie Preference, Support Group, Travel Plan. No more than three words.
- The "id" field may be any valid placeholder matching the first source id, because code will rewrite ids deterministically after validation.
- Use PREVIOUS_REWRITE_MEMORIES only to avoid repeating already-written memories; do not copy them unless this window adds new information.
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
      "time":"YYYY-MM-DD"
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

Dialogue:
<<<
{RAW_TEXT}
>>>"""

    @classmethod
    def extract_rewrite_prompt(cls, raw_text: str, previous_memories: str = "[]") -> str:
        return cls.REWRITE_PROMPT.format(
            RAW_TEXT=raw_text,
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

    EAES_FINAL_ANSWER_PROMPT = """You answer from an EAES evidence package. Only output valid JSON.
Use the structured evidence package as the primary context.
Rules:
- Give the minimal answer requested by the question.
- For list questions, return a concise comma-separated list.
- Treat evidence_package as primary evidence. Use backup_candidates only when evidence_package is empty or clearly insufficient.
- For time questions, always normalize the answer using rewrite_content plus the evidence time_interval.start anchor.
- For time questions, do not answer only with relative phrases such as "yesterday", "last Friday", "last week", "last year", "next month", or "two days ago" when time_interval.start is available.
- For a single-time question, return exactly one best time expression, not a list of multiple candidate dates.
- If the question asks for an exact date, output an absolute date like "2023-07-10" or "10 July 2023".
- If the gold-style answer is naturally relative to a known anchor, normalize it, e.g. "the Friday before 14 August 2023" or "the week before 3 July 2023".
- Use these conversions with time_interval.start as the conversation date: yesterday = start - 1 day; two days ago = start - 2 days; last week = the week before start; last Friday = the nearest Friday before start; next month = the month after start.
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







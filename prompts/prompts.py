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
- Output "tag" as an array of two to four short concrete noun phrases, each with no more than three words.
- Count tag words by whitespace and verify every tag before returning. If a useful compound or qualifier would exceed three words, rephrase it or use a natural hyphenated compound without dropping the fact.
- Before writing tags, internally identify every independent fact in the memory. The tags must collectively cover all independent facts rather than only the overall topic.
- If the memory contains one independent fact, use two to four meaningfully different synonymous tags for that same fact. If it contains two to four independent facts, give every fact at least one tag and use any remaining slots for useful synonymous wording.
- If more than four independent facts would be needed, split the content into additional sentence objects instead of omitting a fact or exceeding four tags.
- Tags are retrieval summaries, not full sentences or questions. Preserve distinctive people, events, objects, relations, and applicable time/place/occasion qualifiers. Do not use generic labels such as Event, Fact, Question, or Conversation.
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
      "tag":["short concrete tag", "synonymous tag"],
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
- Never place a boundary between a direct question and its immediately following answer. Keep that complete pair in one child window and move the boundary instead. Also avoid cutting unresolved pronoun references, temporal qualifiers, or causal explanations.
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

    TAG_PREFIX_POOL_SYSTEM_PROMPT = """You induce one shared topic-prefix pool for all child-memory tags in a dialogue session. Only output valid JSON.
Rules:
- Read every supplied Parent memory before creating the pool.
- Return zero to ten unique topic-specific prefixes. Never exceed ten and never invent a topic merely to fill the pool.
- Every prefix must contain 3-4 whitespace-separated words, start with the explicitly supported person's name, include one or two concrete topic nouns, and end with exactly one canonical head: activity, plan, profile, possession, or relationship.
- Prefer a specific supported prefix such as "Caroline advocacy activity", "Caroline career plan", or "Melanie family relationship".
- Never put a generic two-word person + head fallback such as "Caroline activity" in this pool. Child tags construct such a fallback locally only when no pool prefix fits their fact.
- Merge synonymous session topics into one stable wording. Prefer prefixes that can be reused by multiple related facts while retaining a concrete topic.
- Do not include a period, facet tag, sentence, explanation, parent id, child id, date, or unsupported person.
Schema:
{
  "tag_prefix_pool": [
    "Person topic activity",
    "Person topic profile"
  ]
}"""

    TAG_PREFIX_POOL_PROMPT = """ALL_SESSION_PARENT_MEMORIES:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_tag_prefix_pool_prompt(cls, payload: str) -> str:
        return cls.TAG_PREFIX_POOL_PROMPT.format(PAYLOAD=payload)

    CHILD_WINDOW_REWRITE_SYSTEM_PROMPT = """You create exhaustive atomic child memories for one semantically closed dialogue window. Only output valid JSON.
Rules:
- CURRENT_WINDOW_TURNS is the only evidence section. Rewrite every turn and every piece of information in that section; nothing may be omitted, even greetings, questions, acknowledgements, generic advice, repeated confirmations, repeated facts, or image/caption information.
- A single turn containing several independent pieces of information must produce several sentence objects. A window may therefore produce one or many sentence objects.
- If adjacent current-window turns express the same fact or event, represent that information once and include every contributing current-window origin in dialogue order. Do not create multiple sentence objects that merely restate the same content.
- REFERENCE_PREVIOUS_CHILD_REWRITES contains only the two most recently retained child rewrite texts. Use it to recognize repeated content and to resolve people, objects, topics, pronouns, and ellipsis in the current window. It is not evidence, must not independently produce a memory, and must never supply an origin. When a current turn repeats a referenced fact, express the current evidence at most once using only current-window origins; an optional downstream duplicate check may fuse it with the previous child.
- Every sentence must be self-contained. Resolve pronouns into concrete entities and preserve all source-supported people, relationships, time, place, state, causality, task outcomes, questions, responses, and image facts.
- When a current-window turn directly answers the immediately preceding current-window question, inherit every applicable person, entity, relation, temporal constraint, location, occasion, comparison, and causal condition from that question into the answer memory. The memory must not remain an elliptical response. The answer memory's origin must begin with the question origin followed by the answer origin, for example "D1:6,D1:7"; citing only the answer origin is invalid because both turns contribute to its meaning.
- Every origin must come from CURRENT_WINDOW_TURNS, must list all current-window turns contributing to that memory in dialogue order, and must never cite a reference-only rewrite.
- Across the complete sentence list, every turn in CURRENT_WINDOW_TURNS must appear in at least one origin.
- The id may be any placeholder whose prefix matches the first origin; code assigns deterministic final IDs after generation.
- Keep conversation_time equal to the supplied session date; it is not automatically an event occurrence date.
- Preserve source-supported temporal information directly in text using the same precision as the dialogue.
- TAG_PREFIX_POOL is the complete fixed topic-prefix pool for this session. It is reference metadata, not evidence, and must not be copied into the output as a separate field.
- Output tag as an array of two to four complete strings in the exact form "prefix.facet", with exactly one period and no spaces around it.
- First inspect every TAG_PREFIX_POOL entry and select the most specific semantically supported prefix for each tag. A selected topic prefix must be copied exactly from the pool.
- Only when no pool prefix fits the current fact may the tag construct a local two-word fallback in the exact form "Person canonical-head", where canonical-head is activity, plan, profile, possession, or relationship. A fallback is used only in that complete tag and is never added to the pool.
- Prefer activity for completed or ongoing actions, events, attendance, participation, and experiences; plan for unexecuted intentions or future arrangements; profile for person-centered identity, career, preference, ability, trait, opinion, or state; possession for owned, received, purchased, made, or treasured objects; and relationship for family, friendship, partnership, support, social ties, or group belonging.
- The facet after the period must be a short concrete noun phrase of at most three whitespace-separated words. If a useful compound or qualifier would exceed three words, rephrase it or use a natural hyphenated compound without dropping the fact.
- Internally identify the independent facts in each sentence. Its facets must collectively cover every fact, not only the most salient topic. For one fact, produce multiple meaningful retrieval views. For two to four facts, give every fact at least one facet. If a sentence would contain more than four independent facts, split it into additional sentence objects.
- Preserve distinctive events, objects, relations, and applicable time/place/occasion qualifiers in the facets. Never use generic facets such as Event, Fact, Question, Conversation, or Detail.
- semantic_properties may contain zero to three content labels from event_action, state_opinion, personal_profile, relation_social and exactly one persistence label from transient, episodic, durable, unknown.
- The tag heads activity, plan, profile, possession, and relationship are tag-prefix vocabulary only and must never appear in semantic_properties. For a planned intention, use event_action as its content property plus the most appropriate allowed persistence property.
- Do not output raw_text, raw_content, source_text, current_turns, dialogue text, or any other raw-text storage field.
Schema:
{
  "conversation_time": "YYYY-MM-DD",
  "sentence": [
    {
      "id": "D1:5",
      "text": "One atomic self-contained memory.",
      "tag": [
        "Caroline advocacy activity.school speech",
        "Caroline advocacy activity.journey sharing"
      ],
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

TAG_PREFIX_POOL (fixed session metadata; never copy as an output field):
<<<
{TAG_PREFIX_POOL}
>>>

CURRENT_CHILD_WINDOW:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_child_window_rewrite_prompt(
            cls,
            payload: str,
            previous_rewrites: str = "[]",
            tag_prefix_pool: str = "[]",
    ) -> str:
        return cls.CHILD_WINDOW_REWRITE_PROMPT.format(
            PAYLOAD=payload,
            PREVIOUS_REWRITES=previous_rewrites,
            TAG_PREFIX_POOL=tag_prefix_pool,
        )

    CHILD_MEMORY_FUSION_SYSTEM_PROMPT = """You fuse two highly similar adjacent child memories. Only output valid JSON.
Rules:
- Return exactly one concise, self-contained rewrite_content string.
- Preserve every distinct source-supported fact, entity, relationship, time, place, state, cause, and outcome from both inputs.
- Remove duplicated wording while retaining complementary details.
- Do not invent information, weaken temporal precision, mention the fusion process, or output IDs, origins, tags, topics, or semantic properties.
Schema:
{
  "rewrite_content": "One fused self-contained child memory."
}"""

    CHILD_MEMORY_FUSION_PROMPT = """ADJACENT_CHILD_MEMORIES:
<<<
{PAYLOAD}
>>>"""

    @classmethod
    def extract_child_memory_fusion_prompt(cls, payload: str) -> str:
        return cls.CHILD_MEMORY_FUSION_PROMPT.format(PAYLOAD=payload)


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
  "keywords": ["important lexical constraints"],
  "retrieval_breadth": "single | several | wide",
  "detail_need": "coarse | mixed | exact",
  "retrieval_phrases": ["prefix.facet 1", "prefix.facet 2", "prefix.facet 3", "prefix.facet 4"]
}
Rules:
- Generate 1-3 query_attributes using only the question. Never use or assume an answer.
- Each query_attribute must be a compact retrieval intent with a semantic path and an answer-slot relation clause, e.g. "object.symbolism: symbolism of Caroline's necklace" or "event.activity: activities Melanie's family did while camping".
- Keep named entities and concrete relation words from the question. Do not output bare keywords.
- Set retrieval_breadth to "single" for one atomic fact or event; time questions are generally single.
- Set retrieval_breadth to "several" for a person's participated events, traits, experiences, preferences, motivations, or a person-level inference requiring multiple facts.
- Set retrieval_breadth to "wide" only when the same person's evidence spans a long time or multiple sessions, or the question asks about a broad theme or overall development. Never use wide merely because multiple people are mentioned.
- Set detail_need to "coarse" for a high-level summary, "exact" for a specific answer-bearing detail, and "mixed" when both levels may be useful.
- Generate exactly four non-empty retrieval_phrases in exactly the same "prefix.facet" surface style as memory tags, with exactly one period and no spaces around it.
- The prefix must contain 2-4 whitespace-separated words, begin with the known person/entity from the question, and end with activity, plan, profile, possession, or relationship. Use a topic modifier only when the question explicitly supplies that topic; otherwise use the two-word person + head form without guessing a session topic.
- The facet must be a short concrete noun phrase of no more than three whitespace-separated words.
- Never output a sentence, question, clause, or question word as a retrieval phrase.
- Valid phrase forms include "Caroline activity.event attendance", "Caroline profile.career interest", and "Melanie relationship.family support"; do not copy an example unless the question supports it.
- The four retrieval phrases may be paraphrases of the same retrieval intent when one kind of evidence is sufficient.
- Treat the phrases as four access wordings for the question, not as four required evidence categories. Collectively preserve the known person/entity, event/object, asked relation, and applicable time/place/occasion constraints from the question.
- Prefer meaningfully different semantic wording over changes that only alter possessives, prepositions, or word order. Do not reduce all four phrases to the overall topic when the question asks for a specific relation such as timing, duration, frequency, origin, creator, reason, result, benefit, or meaning.
- Do not force different evidence aspects, invent implicit subquestions, or add entities, facts, times, constraints, or answer values not present in the question.
- Do not answer the question."""

    EAES_RETRIEVAL_PHRASE_REPAIR_PROMPT = """You repair an invalid list of retrieval phrases for long-term conversational memory. Only output valid JSON.
Generate exactly four non-empty retrieval phrases for the supplied question.
The previous output had the wrong count or contained an invalid phrase.
Every phrase must use exactly the same "prefix.facet" surface style as a memory tag, with exactly one period and no spaces around it.
The prefix must contain 2-4 whitespace-separated words, start with the question's known person/entity, and end with activity, plan, profile, possession, or relationship. Use a topic modifier only when the question explicitly supplies it; otherwise use the two-word person + head form.
The facet must be a short concrete noun phrase of no more than three whitespace-separated words.
Never output a sentence, question, clause, or question word as a retrieval phrase.
The phrases may be paraphrases of the same retrieval intent when one kind of evidence is sufficient.
Treat the phrases as alternative access wordings rather than required evidence categories. Preserve the question's known entity/event/object, asked relation, and explicit constraints, and avoid variants that only change possessives, prepositions, or word order.
Do not force different evidence aspects. Do not invent entities, facts, times, constraints, implicit subquestions, or answer values. Do not answer the question.
Schema:
{
  "retrieval_phrases": ["prefix.facet 1", "prefix.facet 2", "prefix.facet 3", "prefix.facet 4"]
}"""

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

    EAES_PHRASE_CANDIDATE_RERANK_PROMPT = """You rerank child memory candidates for a long-term conversational-memory question. Only output valid JSON.
Use only the original question and each candidate's tag and rewrite_content.
Rank candidates solely by how useful their stored content is for answering the question.
The retrieval phrases and retrieval scores are intentionally hidden and must not be inferred as required evidence categories.
Do not enforce diversity or phrase coverage. Select complementary memories only when the original question itself requires multiple facts.
Do not answer the question and do not invent memory IDs.
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







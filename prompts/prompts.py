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
- Output "tag" as an array of one to four short concrete noun phrases, each with no more than five words.
- Count tag words by whitespace and verify every tag before returning. If a useful compound or qualifier would exceed five words, rephrase it or use a natural hyphenated compound without dropping the fact.
- Before writing tags, internally identify every independent fact in the memory. The tags must collectively cover all independent facts rather than only the overall topic.
- If the memory contains one independent fact, use one to four meaningful tags for that fact. If it contains two to four independent facts, give every fact at least one tag and use any remaining slots for useful synonymous wording.
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
- Output tag as an array of one to four objects. Every object must generate its own prefix and facet together, using exactly the keys prefix and facet.
- Every prefix must start with the explicitly supported person's name and end with exactly one canonical head: activity, plan, profile, possession, or relationship. Optional concrete descriptive words may appear between the person and canonical head, with no prefix word-count limit. For example, "Caroline caring profile" is valid.
- Never use generic placeholders such as Person, Speaker, User, Assistant, Entity, or Someone as the person name.
- Do not include a period in either prefix or facet. Code joins each valid pair into the final stored string "prefix.facet".
- Prefer activity for completed or ongoing actions, events, attendance, participation, and experiences; plan for unexecuted intentions or future arrangements; profile for person-centered identity, career, preference, ability, trait, opinion, or state; possession for owned, received, purchased, made, or treasured objects; and relationship for family, friendship, partnership, support, social ties, or group belonging.
- Every facet must be a short concrete noun phrase of at most five whitespace-separated words. If a useful compound or qualifier would exceed five words, rephrase it or use a natural hyphenated compound without dropping the fact.
- Internally identify the independent facts in each sentence. Its facets must collectively cover every fact, not only the most salient topic. For one fact, produce one or more meaningful retrieval views. For two to four facts, give every fact at least one facet. If a sentence would contain more than four independent facts, split it into additional sentence objects.
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
        {
          "prefix": "Caroline advocacy activity",
          "facet": "school speech"
        },
        {
          "prefix": "Caroline advocacy activity",
          "facet": "journey sharing"
        }
      ],
      "origin": "D1:5",
      "topic": [],
      "semantic_properties": ["event_action", "episodic"]
    }
  ],
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
            cls,
            payload: str,
            previous_rewrites: str = "[]",
    ) -> str:
        return cls.CHILD_WINDOW_REWRITE_PROMPT.format(
            PAYLOAD=payload,
            PREVIOUS_REWRITES=previous_rewrites,
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
  "retrieval_phrases": ["short phrase 1", "short phrase 2", "short phrase 3", "short phrase 4"]
}
Rules:
- Generate 1-3 query_attributes using only the question. Never use or assume an answer.
- Each query_attribute must be a compact retrieval intent with a semantic path and an answer-slot relation clause, e.g. "object.symbolism: symbolism of Caroline's necklace" or "event.activity: activities Melanie's family did while camping".
- Keep named entities and concrete relation words from the question. Do not output bare keywords.
- Set retrieval_breadth to "single" for one atomic fact or event; time questions are generally single.
- Set retrieval_breadth to "several" for a person's participated events, traits, experiences, preferences, motivations, or a person-level inference requiring multiple facts.
- Set retrieval_breadth to "wide" only when the same person's evidence spans a long time or multiple sessions, or the question asks about a broad theme or overall development. Never use wide merely because multiple people are mentioned.
- Set detail_need to "coarse" for a high-level summary, "exact" for a specific answer-bearing detail, and "mixed" when both levels may be useful.
- Generate exactly four non-empty retrieval_phrases. Each phrase must be a normal short retrieval expression containing no more than three whitespace-separated words.
- Retrieval phrases do not use the child-memory "prefix.facet" format. Do not add an artificial canonical head or period delimiter merely to imitate a memory tag.
- Retrieval phrases are matched against complete child-memory tags describing people's actions, objects, attributes, states, and relationships. Express the kind of fact a relevant memory would state, using concrete relation wording supported by the question rather than merely summarizing its broad topic.
- Prioritize the specific action or relation asked about together with its known person/entity, event, or object. Preserve distinctions such as adoption versus ownership, joining versus attending, occurrence date versus duration, and celebration versus winning. Do not replace a concrete relation with a broader topic.
- Use compact noun phrases or telegraphic entity-action-object expressions, such as "Leo adoption date" or "Maya adopted Leo" when supported by the question. Do not output questions, explanations, narrative sentences, or question words.
- Each phrase must independently describe a useful retrieval target. Anchor it with a known person, distinctive entity, event, object, or specific relation when available. Separate query-plan fields do not automatically supply missing context to a phrase. Avoid disconnected name-only, date-only, or generic-topic fragments.
- For one fact or relation, use the four phrases as alternative access wordings for that same evidence need. Prefer meaningful wording differences over changes only to possessives, prepositions, or word order; do not change the retrieval target merely to make the phrases different.
- For questions explicitly requiring comparison, shared attributes, or multiple conditions, distribute the four phrases across the required entities or relations. Preserve the question's relation in each branch; do not drop one participant or condition by spending all four phrases on near-identical overall-topic wording.
- Do not force every question into four evidence categories or invent additional evidence branches. Only decompose requirements explicitly supported by the question.
- Across the four phrases, preserve useful known entities, events, objects, relations, and explicit time/place/occasion constraints. Within the three-word limit, prioritize the specific relation and its identifying entity/event/object; express other useful constraints in another meaningful phrase when possible. Do not replace event retrieval with a bare date or assume that a conversation date is an event occurrence date.
- Avoid generic words such as "information", "detail", "mention", or "action" when the question supplies a more specific relation. Broad words such as "activity" remain valid when the question itself is broad.
- If the question is underspecified, preserve that uncertainty. Do not guess a concrete activity, object, place, or answer to make the phrases more specific. Do not invent entities, facts, times, constraints, or answer values. Do not answer the question.
- Before returning the JSON, silently check the phrase count and three-word limit, preservation of the specific retrieval target, coverage of explicitly required entities or evidence branches, meaningful wording differences, and absence of unsupported answers or assumptions.
Examples illustrate retrieval_phrases only; return the full query-plan schema above and never copy example facts into an unrelated question:
- Question: "When did Maya adopt her cat Leo?"
  retrieval_phrases: ["Maya adopted Leo", "Leo adoption date", "Maya cat adoption", "Leo adoption time"]
- Question: "What hobbies do Alice and Ben share?"
  retrieval_phrases: ["Alice hobbies", "Alice leisure activities", "Ben hobbies", "Ben leisure activities"]
- For an adoption-date question, "Leo adoption date" preserves the relation; "Maya pet information" loses it. For a question about celebrating a tournament win, "tournament victory celebration" preserves the relation; "tournament information" loses it.
- "What new activity did Lena start?" does not justify guessing volunteering or a shelter. "What did Alex do for Riley?" does not justify guessing a gift unless the question mentions giving or receiving one."""

    EAES_RETRIEVAL_PHRASE_REPAIR_PROMPT = """You repair an invalid list of retrieval phrases for long-term conversational memory. Only output valid JSON.
Generate exactly four non-empty retrieval phrases for the supplied question.
The previous output had the wrong count or contained an invalid phrase.
Read the supplied validation_error and repair that exact error once.
Every phrase must be a normal short retrieval expression containing no more than three whitespace-separated words.
Retrieval phrases do not use the child-memory "prefix.facet" format. Do not add an artificial canonical head or period delimiter merely to imitate a memory tag.
Retrieval phrases are matched against complete child-memory tags describing people's actions, objects, attributes, states, and relationships. Express the kind of fact a relevant memory would state, using concrete relation wording supported by the question rather than merely summarizing its broad topic.
Prioritize the specific action or relation asked about together with its known person/entity, event, or object. Preserve distinctions such as adoption versus ownership, joining versus attending, occurrence date versus duration, and celebration versus winning. Do not replace a concrete relation with a broader topic.
Use compact noun phrases or telegraphic entity-action-object expressions, such as "Leo adoption date" or "Maya adopted Leo" when supported by the question. Do not output questions, explanations, narrative sentences, or question words.
Each phrase must independently describe a useful retrieval target. Anchor it with a known person, distinctive entity, event, object, or specific relation when available. Separate query-plan fields do not automatically supply missing context to a phrase. Avoid disconnected name-only, date-only, or generic-topic fragments.
For one fact or relation, use the four phrases as alternative access wordings for that same evidence need. Prefer meaningful wording differences over changes only to possessives, prepositions, or word order; do not change the retrieval target merely to make the phrases different.
For questions explicitly requiring comparison, shared attributes, or multiple conditions, distribute the four phrases across the required entities or relations. Preserve the question's relation in each branch; do not drop one participant or condition by spending all four phrases on near-identical overall-topic wording.
Do not force every question into four evidence categories or invent additional evidence branches. Only decompose requirements explicitly supported by the question.
Across the four phrases, preserve useful known entities, events, objects, relations, and explicit time/place/occasion constraints. Within the three-word limit, prioritize the specific relation and its identifying entity/event/object; express other useful constraints in another meaningful phrase when possible. Do not replace event retrieval with a bare date or assume that a conversation date is an event occurrence date.
Avoid generic words such as "information", "detail", "mention", or "action" when the question supplies a more specific relation. Broad words such as "activity" remain valid when the question itself is broad.
If the question is underspecified, preserve that uncertainty. Do not guess a concrete activity, object, place, or answer to make the phrases more specific. Do not invent entities, facts, times, constraints, or answer values. Do not answer the question.
Before returning the JSON, silently check the phrase count and three-word limit, preservation of the specific retrieval target, coverage of explicitly required entities or evidence branches, meaningful wording differences, and absence of unsupported answers or assumptions.
Examples illustrate retrieval_phrases only; return the repair schema below and never copy example facts into an unrelated question:
- Question: "When did Maya adopt her cat Leo?"
  retrieval_phrases: ["Maya adopted Leo", "Leo adoption date", "Maya cat adoption", "Leo adoption time"]
- Question: "What hobbies do Alice and Ben share?"
  retrieval_phrases: ["Alice hobbies", "Alice leisure activities", "Ben hobbies", "Ben leisure activities"]
- For an adoption-date question, "Leo adoption date" preserves the relation; "Maya pet information" loses it. For a question about celebrating a tournament win, "tournament victory celebration" preserves the relation; "tournament information" loses it.
- "What new activity did Lena start?" does not justify guessing volunteering or a shelter. "What did Alex do for Riley?" does not justify guessing a gift unless the question mentions giving or receiving one.
Schema:
{
  "retrieval_phrases": ["short phrase 1", "short phrase 2", "short phrase 3", "short phrase 4"]
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

    EAES_ROLLBACK_QUERY_PROMPT = """You are an evidence-sufficiency controller for long-term conversational memory. Only output valid JSON.
The input contains the original question, only the rewrite_content strings currently available as evidence, the four initial query phases, any previous rollback decisions, and the number of rollback retrievals still available.

First decide whether the selected rewrite contents already contain all evidence needed to answer the original question correctly. Judge evidence sufficiency, not whether a language model could guess a plausible answer.

Output schema:
{
  "state": "no_need_more | need_more",
  "semantic_properties": ["property required by the missing evidence"],
  "query_phase": ["short phase 1", "short phase 2", "short phase 3", "short phase 4"]
}

State rules:
- Use "no_need_more" only when the current rewrite contents explicitly provide every necessary fact, entity binding, relation, and time/detail constraint needed by the question.
- Use "need_more" when any necessary evidence is absent, only topically related, attached to the wrong entity/event, too imprecise, or in unresolved conflict.
- For "no_need_more", semantic_properties and query_phase must both be empty arrays.
- For "need_more", describe only the still-missing evidence. Do not repeat evidence already present merely to increase confidence.

semantic_properties rules:
- Allowed content properties are: "event_action", "state_opinion", "personal_profile", "relation_social".
- Allowed persistence properties are: "transient", "episodic", "durable".
- Content and persistence properties may each contain multiple applicable labels.
- If the missing evidence's semantic properties are unknown, output an empty array. An empty array means no memory receives a semantic-property bonus.
- Never output the literal label "unknown", tag-prefix heads, or any unlisted property. Do not repeat a property.

query_phase rules:
- For "need_more", generate exactly four distinct non-empty phases, each containing no more than three whitespace-separated words. Distinctness is case-insensitive.
- Every phase must target the missing evidence rather than restating the overall question or evidence already present.
- The rollback phases are deliberately allowed more aggressive synonymy than the first query plan. Use meaningfully different synonyms, paraphrased relations, event roles, nominalizations, or inverse relation wording that could retrieve the same missing fact.
- Preserve the missing fact's known person/entity, event, object, and explicit temporal or relational constraint where useful. Evidence and entities explicitly present in the supplied rewrites may be used as bridge terms.
- Do not invent an answer, unknown entity, event, object, date, or unsupported constraint. Do not use generic words such as "information", "detail", or "mention" when a concrete missing relation is available.
- Phases are ordinary retrieval expressions, not child-memory prefix.facet tags. Do not output questions or explanations.
- Previous rollback decisions are context: do not repeat an unsuccessful phase set without a meaningfully different access path.
- On a repair attempt, use validation_error and previous_invalid_output to correct the invalid fields while preserving the intended missing-evidence target.

Return exactly the three fields in the schema and nothing else."""

    EAES_ROLLBACK_SUPPLEMENT_RERANK_PROMPT = """You select complementary memory nodes for a retrieval rollback check. Only output valid JSON.
The input contains two separately prefiltered groups: child_candidates has up to 27 child memories and parent_candidates has up to 3 parent memories. Their scores are meaningful only within the same node type; never compare child and parent numeric scores directly.
Use the question, rollback decision, current evidence rewrite contents, and candidate contents. Select up to limit distinct nodes that add the strongest missing answer evidence. Do not select a node merely because it repeats evidence already present. Do not invent IDs.
Returning fewer than limit nodes, including an empty array, is valid when no additional candidate provides useful missing evidence.
Schema:
{
  "ranked_nodes": [
    {"node_type": "child", "node_id": "M_D1_2_1"},
    {"node_type": "parent", "node_id": "D1:t2"}
  ]
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
- State the answer directly. Never preface it with phrases such as "The original text states", "the memory says", "the rewrite says", "according to the evidence", or similar source-reporting language.
- Each child evidence object contains only memory_id, conversation_time, and rewrite_content. memory_id is only for supports; conversation_time is a dialogue anchor, not automatically the event occurrence time.
- parent_memories, when present, are independently retrieved coarse-grained rewrite memories. They are direct supporting context and do not restrict or rank the child evidence package.
- A relevant parent memory may support the answer even when its children are absent from the child candidate list. Cite its parent_id in supports when used.
- For list questions, return a concise comma-separated list.
- Treat evidence_package as primary evidence. Use backup_candidates only when evidence_package is empty or clearly insufficient.
- When multiple candidates mention similar events, choose the evidence whose entities, relations, constraints, and wording best match the question; do not merge conflicting facts.
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







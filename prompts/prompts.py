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




    REWRITE_SYSTEM_PROMPT = """You are a dialogue processor. Only output valid JSON.
TASK:
- For each sentence:
  1. Replace ALL pronouns ("I", "you", "he", "she", "it", "they", "we", "this", "that", "these", "those", "the", "the xx", "one") with explicit entities, events or noun phrases from the conversation context, such as "the event"->"charity race".  
  2. Do NOT modify verbs, adjectives, or other words. Only replace pronouns. 
  3. Use a short concrete noun to describe what the speaker is talking about in "tag", e.g. Movie Preference, Hobbies. No more than two words. 
  4. If a sentence uses a relative time (e.g., 'yesterday', 'next Monday', 'in two days', 'last summer'), compute the absolute calendar date based on conversation_time and output 'YYYY-MM-DD'. If the sentence includes a precise absolute date already, output that date. If time refers to a period, output the midpoint date 'YYYY-MM-DD'. If no time is mentioned, output the conversation date (YYYY-MM-DD).
  5. If a sentence ends with a question, merge the question content into the next sentence that provides an answer to complete sentence information.
  6. Never omit or miss any sentence.
- Topics:derive at least ten concrete topics overall (short sentences). Assign topic IDs (t1..tn). In each sentence, fill 'topic' with a list of topic IDs that apply; use [] if none
- Personal information: extract person-related facts (preferences, roles, schedules, background, relations, attributes) into 'personal_sentences'. If a fact is already in a sentence, also duplicate a concise normalized version here.
- "id" in "sentence" is combination of "origin" and number. The "origin" must exactly correspond to the "dia_id" field in the "Dialogue". Do not create or invent new ids.
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

    REWRITE_PROMPT = """Dialogue:
        <<<
        {RAW_TEXT}
        >>>"""

    @classmethod
    def extract_rewrite_prompt(cls, raw_text: str) -> str:
        return cls.REWRITE_PROMPT.format(
            RAW_TEXT=raw_text,
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

    # -------- fact extraction --------



    # -------- multi-perspective key generation --------



    # -------- Questions → keys --------







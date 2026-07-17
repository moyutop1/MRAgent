import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add repo root to path for standalone runs
import argparse
import json
from collections import defaultdict

import numpy as np
from openai import OpenAI
from common.utils import extract_json_from_content
from common.openrouter import OPENROUTER_BASE_URL, get_openrouter_headers

from dotenv import load_dotenv
import os
load_dotenv()  # read API key from .env
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", os.getenv("CHAT_PROVIDER", "openrouter")).lower()
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "300" if JUDGE_PROVIDER == "deepseek" else "60"))
JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "5" if JUDGE_PROVIDER == "deepseek" else "2"))
DEEPSEEK_THINKING_MODE = os.getenv("DEEPSEEK_THINKING_MODE", "disabled").lower()
if JUDGE_PROVIDER == "ofox":
    API_KEY = os.getenv("OFOX_API_KEY")
    JUDGE_MODEL = os.getenv("OFOX_JUDGE_MODEL", os.getenv("OFOX_MODEL", "gpt-4o-mini"))
    JUDGE_BASE_URL = os.getenv("OFOX_BASE_URL", "").rstrip("/")
elif JUDGE_PROVIDER == "deepseek":
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    JUDGE_MODEL = os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash")
    JUDGE_BASE_URL = "https://api.deepseek.com"
else:
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    JUDGE_MODEL = os.getenv("OPENROUTER_JUDGE_MODEL", "openai/gpt-4o-mini")
    JUDGE_BASE_URL = OPENROUTER_BASE_URL
client = OpenAI(
    api_key=API_KEY,
    base_url=JUDGE_BASE_URL,
    timeout=JUDGE_TIMEOUT,
    max_retries=JUDGE_MAX_RETRIES,
    default_headers=get_openrouter_headers() if JUDGE_PROVIDER == "openrouter" else None,
)

ACCURACY_PROMPT = """
Your task is to label a generated answer to a question as CORRECT or WRONG. You will receive:
    (1) a question posed by one user to another,
    (2) a gold (ground-truth) answer,
    (3) a generated answer.

Judge semantic equivalence, not lexical overlap. First identify the answer-bearing fact or event requested by the question, then compare the meaning expressed by the gold and generated answers.

Label CORRECT when the generated answer conveys the same answer-bearing proposition as the gold answer, including when it uses:
- synonyms or near-synonyms;
- a paraphrase or different grammatical form;
- a noun phrase instead of a verb phrase, or vice versa;
- a longer answer with extra details that do not contradict the gold answer.

Do not require the same wording or shared keywords. For example:
- Gold: "school speech"; Generated: "gave a talk at a school event" -> CORRECT. "Speech" and "gave a talk" describe the same action in this context.
- Gold: "a shell necklace"; Generated: "They brought back a necklace made of shells from Hawaii" -> CORRECT.

However, sharing only a broad topic is not enough. The generated answer must entail the answer-bearing fact requested by the question. For example:
- Gold: "gave a speech at the school event"; Generated: "attended a school event" -> WRONG. Attendance does not establish that a speech was given.

Label WRONG when the generated answer contradicts the gold answer, describes a different person/event/object, omits the requested fact, or is too vague to establish semantic equivalence. Treat differences in negation, person, quantity, completion status (planned versus completed), and answer-critical time or place as meaningful. Minor details that the question does not ask for need not match.

For time-related questions, accept different surface forms that identify the same date or time period, including equivalent absolute and relative expressions when they are grounded to the same time. For example, "May 7th" and "7 May" are equivalent. Do not accept genuinely different dates or periods.

Now evaluate the real example:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return only valid JSON. Do not include explanations, markdown, or extra text.
The JSON schema is exactly:
{{"label": "CORRECT"}}
or
{{"label": "WRONG"}}
"""


def _parse_judge_label(content):
    import re
    try:
        label = json.loads(content)["label"]
    except Exception:
        try:
            label = extract_json_from_content(content).get("label")
        except Exception:
            text = str(content or "")
            if re.search(r"\bCORRECT\b", text) and not re.search(r"\bWRONG\b", text):
                label = "CORRECT"
            elif re.search(r"\bWRONG\b", text) and not re.search(r"\bCORRECT\b", text):
                label = "WRONG"
            else:
                return None
    label = str(label or "").strip().upper()
    return label if label in {"CORRECT", "WRONG"} else None


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    if not API_KEY:
        raise RuntimeError(
            "Judge API key is empty. Set OPENROUTER_API_KEY, DEEPSEEK_API_KEY, "
            "or OFOX_API_KEY in .env before running LLM judge."
        )
    if not JUDGE_BASE_URL:
        raise RuntimeError("Judge base URL is empty. Set OFOX_BASE_URL in .env when JUDGE_PROVIDER=ofox.")
    req = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question, gold_answer=gold_answer, generated_answer=generated_answer
                ),
            }
        ],
        "temperature": 0.0,
    }
    if JUDGE_PROVIDER == "deepseek" and JUDGE_MODEL.startswith("deepseek-v4") and DEEPSEEK_THINKING_MODE == "disabled":
        req["extra_body"] = {"thinking": {"type": "disabled"}}
    last_content = ""
    for _ in range(2):
        try:
            response = client.chat.completions.create(**req, response_format={"type": "json_object"})
        except Exception:
            response = client.chat.completions.create(**req)
        last_content = response.choices[0].message.content
        label = _parse_judge_label(last_content)
        if label is not None:
            return 1 if label == "CORRECT" else 0
    print(f"[judge] Could not parse label; counting WRONG. head={str(last_content)[:120]!r}", file=sys.stderr)
    return 0


# def main():
#     """Main function to evaluate RAG results using LLM judge."""
#     parser.add_argument(
#         "--input_file",
#     )


#     output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

#     with open(dataset_path, "r") as f:


#             question = x["question"]
#             gold_answer = x["answer"]
#             generated_answer = x["response"]
#             category = x["category"]

#             # Skip category 5

#             # Evaluate the answer
#             LLM_JUDGE[category].append(label)

#             # Store the results
#             RESULTS[index].append(
#                 {
#                     "question": question,
#                     "gt_answer": gold_answer,
#                     "response": generated_answer,
#                     "category": category,
#                     "llm_label": label,
#                 }
#             )

#             # Save intermediate results
#             with open(output_path, "w") as f:

#             # Print current accuracy for all categories
#         index += 1

#     # Save final results
#     with open(output_path, "w") as f:

#     # Print final summary



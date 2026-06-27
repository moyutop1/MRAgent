import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add repo root to path for standalone runs
import argparse
import json
from collections import defaultdict

import numpy as np
from openai import OpenAI
from common.utils import extract_json_from_content

from dotenv import load_dotenv
import os
load_dotenv()  # read API key from .env
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", os.getenv("CHAT_PROVIDER", "deepseek")).lower()
if JUDGE_PROVIDER == "ofox":
    API_KEY = os.getenv("OFOX_API_KEY")
    JUDGE_MODEL = os.getenv("OFOX_JUDGE_MODEL", os.getenv("OFOX_MODEL", "gpt-4o-mini"))
    JUDGE_BASE_URL = os.getenv("OFOX_BASE_URL", "").rstrip("/")
else:
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    JUDGE_MODEL = os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash")
    JUDGE_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=API_KEY, base_url=JUDGE_BASE_URL)

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    if not API_KEY:
        raise RuntimeError("Judge API key is empty. Set DEEPSEEK_API_KEY or OFOX_API_KEY in .env before running LLM judge.")
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
    try:
        response = client.chat.completions.create(**req, response_format={"type": "json_object"})
    except Exception:
        response = client.chat.completions.create(**req)
    # print([
    #         {
    #             "role": "user",
    #             "content": ACCURACY_PROMPT.format(
    #             ),
    #         }
    #     ])
    # print(response.choices[0].message.content)
    content = response.choices[0].message.content
    try:
        label = json.loads(content)["label"]
    except Exception:
        label = extract_json_from_content(content).get("label")
    return 1 if label == "CORRECT" else 0


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



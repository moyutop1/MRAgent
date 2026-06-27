import logging
import os
import argparse
from dotenv import load_dotenv
load_dotenv()  # read API key from .env
parser = argparse.ArgumentParser(description="Configure dataset and model parameters.")
parser.add_argument("--data", type=str, default="locomo", help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--model", type=str, default="deepseek", help="Model name, e.g., deepseek / deepseek-pro")
parser.add_argument("--file", type=str, default="0", help="Run/experiment tag appended to result filenames")
parser.add_argument("--sample", type=int, default=None, help="Sample id to run (e.g. 42). Omit to run all samples.")
parser.add_argument("--qu", type=int, default=0, help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--re_model", type=str, default=None, help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--ca", type=int, default=1, help="LM category index: 0=multi-session,1=single-session-user,2=temporal-reasoning,3=single-session-preference,4=knowledge-update,5=single-session-assistant")
parser.add_argument("--lm_batch", type=int, default=1, help="LM: sessions merged per rewrite call. 1=per-session (key=session_i, compatible with existing files/per-session readers); >1=merged (key=session_first-session_last)")
parser.add_argument("--eaes", action="store_true", help="Use EAES-Mem answer-oriented evidence selection instead of the default graph tool loop.")

# parse_known_args (not parse_args) so importing this module under a foreign argv
# (pytest, notebooks, helper scripts) does not crash on unrecognized arguments.
args, _ = parser.parse_known_args()

DEEPSEEK_MODEL_ALIASES = {"deepseek", "deepseek-pro", "deepseek-chat", "deepseek-reasoner"}
if args.model not in DEEPSEEK_MODEL_ALIASES:
    raise ValueError("This fork is configured for DeepSeek official API. Use --model deepseek or --model deepseek-pro.")
if args.re_model and args.re_model not in DEEPSEEK_MODEL_ALIASES:
    raise ValueError("This fork is configured for DeepSeek official API. Use --re_model deepseek or --re_model deepseek-pro.")

DEEPSEEK_URL = "https://api.deepseek.com"
CHAT_BASE_URL = DEEPSEEK_URL
API_PROVIDER = "deepseek"
if args.model == "gpt4.1mini":
    MODEL = "openai/gpt-4.1-mini"
elif args.model == "gpt4omini":
    MODEL = "gpt-4o-mini-2024-07-18"
elif args.model == "claude":
    MODEL = "anthropic/claude-sonnet-4.5"
elif args.model == "gpt4o":
    MODEL = "openai/gpt-4o"
elif args.model == "claude3.5":
    MODEL = "anthropic/claude-3.5-haiku"
elif args.model == "qwen":
    MODEL = "qwen/qwen3-max"
elif args.model == "gemini":
    MODEL = "google/gemini-2.5-flash"
elif args.model == "deepseek":
    API_PROVIDER = "deepseek"
    CHAT_BASE_URL = DEEPSEEK_URL
    MODEL = "deepseek-v4-flash"
elif args.model == "deepseek-pro":
    API_PROVIDER = "deepseek"
    CHAT_BASE_URL = DEEPSEEK_URL
    MODEL = "deepseek-v4-pro"
elif args.model == "deepseek-chat":
    API_PROVIDER = "deepseek"
    CHAT_BASE_URL = DEEPSEEK_URL
    MODEL = "deepseek-chat"
elif args.model == "deepseek-reasoner":
    API_PROVIDER = "deepseek"
    CHAT_BASE_URL = DEEPSEEK_URL
    MODEL = "deepseek-reasoner"
else:
    raise ValueError("This fork is configured for DeepSeek official API. Use --model deepseek or --model deepseek-pro.")
CHOOSE_MODEL = MODEL
MODEL_NAME = args.model  # short name (gemini/claude/...), used by the LM temporal method answer_question_with_time_lm
if args.re_model:
    if args.re_model == "gpt4.1mini":
        RE_MODEL = "openai/gpt-4.1-mini"
    elif args.re_model == "gpt4omini":
        RE_MODEL = "gpt-4o-mini-2024-07-18"
    elif args.re_model == "claude":
        RE_MODEL = "anthropic/claude-sonnet-4.5"
    elif args.re_model == "gpt4o":
        RE_MODEL = "openai/gpt-4o"
    elif args.re_model == "claude3.5":
        RE_MODEL = "anthropic/claude-3.5-haiku"
    elif args.re_model == "qwen":
        RE_MODEL = "qwen/qwen3-max"
    elif args.re_model == "gemini":
        RE_MODEL = "google/gemini-2.5-flash"
    elif args.re_model == "deepseek":
        RE_MODEL = "deepseek-v4-flash"
    elif args.re_model == "deepseek-pro":
        RE_MODEL = "deepseek-v4-pro"
    elif args.re_model == "deepseek-chat":
        RE_MODEL = "deepseek-chat"
    elif args.re_model == "deepseek-reasoner":
        RE_MODEL = "deepseek-reasoner"
    else:
        RE_MODEL = MODEL
else:
    RE_MODEL = MODEL
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_SORT = MODEL #"anthropic/claude-sonnet-4.5"
K1=80                 # coarse retrieval breadth (embedding similarity)
K2=20                 # fine retrieval breadth (LLM re-ranking)
TAG_MAX=15            # select_key_tag: re-rank a key's tags only when it has more than this many
TAG_LIMIT=10         # select_key_tag: keep at most this many tags after re-ranking
TIME_EVENT_LIMIT=50  # answer_question_with_time: dense-time fast path threshold (locomo)
TOPIC_K=8            # select_topic: number of topic candidates
RERANK_LIMIT=20      # event_by_tag: re-rank events only when more than this many match
MAX_ROUNDS=8         # tool-calling loop: max assistant rounds
MAX_TOOL_CALLS=50    # tool-calling loop: safety cap on total tool calls
EAES_MODE = args.eaes
EAES_CANDIDATE_LIMIT = 60
EAES_SELECTION_LIMIT = 30
EAES_RAW_EXPANSION_LIMIT = 3
sample_id = args.sample
qu = args.qu
ca = args.ca
LM_REWRITE_BATCH = args.lm_batch  # sessions merged per LM rewrite call

dataset = args.data
DATASET = dataset
datapath = f"data/dataset_{dataset}.json"
ADDITIONAL_TK = f"_{args.model}"#"_gpt4o-mini"
ADDITIONAL_EM = f"_{args.model}_local_bge"#
ADDITIONAL_RE = f"_{args.model}_{args.file}{'_eaes' if EAES_MODE else ''}" #"_gpt4o-mini"
base_dir_t = f"data/{{dataset}}/rewrite{ADDITIONAL_TK}/"
base_dir_k = f"data/{{dataset}}/keyword{ADDITIONAL_TK}/"
base_dir_emb = f"data/{{dataset}}/embedding/gpt{ADDITIONAL_EM}/"
# auto-create all output dirs so a fresh checkout runs without manual mkdir
os.makedirs(base_dir_t.format(dataset=dataset), exist_ok=True)      # data/<ds>/rewrite_<model>/
os.makedirs(base_dir_k.format(dataset=dataset), exist_ok=True)      # data/<ds>/keyword_<model>/
os.makedirs(base_dir_emb.format(dataset=dataset), exist_ok=True)    # data/<ds>/embedding/gpt_<model>/
os.makedirs(f"result/{dataset}", exist_ok=True)                    # prediction outputs
os.makedirs(f"log/{dataset}", exist_ok=True)                       # logs (also creates log/ for the run-level handler)

rewrite_template = f"data/{{dataset}}/rewrite{ADDITIONAL_TK}/{{sample_id}}_rewrite.json"
keyword_template = f"data/{{dataset}}/keyword{ADDITIONAL_TK}/{{sample_id}}_keyword.json"
embedding_template = f"data/{{dataset}}/embedding/gpt{ADDITIONAL_EM}/{{sample_id}}_embedding.pkl"
result_template = f"result/{{dataset}}/{{sample_id}}_result{ADDITIONAL_RE}.jsonl"

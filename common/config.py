import logging
import os
import argparse
from dotenv import load_dotenv
from common.openrouter import OPENROUTER_BASE_URL, get_openrouter_headers
load_dotenv()  # read API key from .env
parser = argparse.ArgumentParser(description="Configure dataset and model parameters.")
parser.add_argument("--data", type=str, default="locomo", help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--model", type=str, default="gemini", help="Model name, e.g., gemini / deepseek / ofox")
parser.add_argument("--file", type=str, default="0", help="Run/experiment tag appended to result filenames")
parser.add_argument("--sample", type=int, default=None, help="Sample id to run (e.g. 42). Omit to run all samples.")
parser.add_argument("--max_questions", type=int, default=None, help="Run at most the first N questions per selected sample.")
parser.add_argument("--qu", type=int, default=0, help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--re_model", type=str, default=None, help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--ca", type=int, default=1, help="LM category index: 0=multi-session,1=single-session-user,2=temporal-reasoning,3=single-session-preference,4=knowledge-update,5=single-session-assistant")
parser.add_argument("--lm_batch", type=int, default=1, help="LM: sessions merged per rewrite call. 1=per-session (key=session_i, compatible with existing files/per-session readers); >1=merged (key=session_first-session_last)")
parser.add_argument("--eaes", action="store_true", help="Use EAES-Mem answer-oriented evidence selection instead of the default graph tool loop.")
parser.add_argument("--retrieval_only", action="store_true", help="Only evaluate retrieval evidence; skip final answer generation and LLM judge.")

# parse_known_args (not parse_args) so importing this module under a foreign argv
# (pytest, notebooks, helper scripts) does not crash on unrecognized arguments.
args, _ = parser.parse_known_args()

OPENROUTER_MODEL_ALIASES = {"gpt4.1mini", "gpt4omini", "claude", "gpt4o", "claude3.5", "qwen", "gemini"}
DEEPSEEK_MODEL_ALIASES = {"deepseek", "deepseek-pro", "deepseek-chat", "deepseek-reasoner"}
SUPPORTED_MODEL_ALIASES = OPENROUTER_MODEL_ALIASES | DEEPSEEK_MODEL_ALIASES | {"ofox"}
if args.model not in SUPPORTED_MODEL_ALIASES:
    raise ValueError("Use --model gemini, --model deepseek, --model deepseek-pro, or --model ofox.")
if args.re_model and args.re_model not in SUPPORTED_MODEL_ALIASES:
    raise ValueError("Use --re_model gemini, --re_model deepseek, --re_model deepseek-pro, or --re_model ofox.")

OPENROUTER_URL = OPENROUTER_BASE_URL
DEEPSEEK_URL = "https://api.deepseek.com"
CHAT_BASE_URL = OPENROUTER_URL
API_PROVIDER = "openrouter"
OPENAI_COMPAT_DEFAULT_HEADERS = None
if args.model == "gpt4.1mini":
    API_PROVIDER = "openrouter"
    CHAT_BASE_URL = OPENROUTER_URL
    MODEL = "openai/gpt-4.1-mini"
elif args.model == "gpt4omini":
    API_PROVIDER = "openrouter"
    CHAT_BASE_URL = OPENROUTER_URL
    MODEL = "openai/gpt-4o-mini"
elif args.model == "claude":
    API_PROVIDER = "openrouter"
    CHAT_BASE_URL = OPENROUTER_URL
    MODEL = "anthropic/claude-sonnet-4.5"
elif args.model == "gpt4o":
    API_PROVIDER = "openrouter"
    CHAT_BASE_URL = OPENROUTER_URL
    MODEL = "openai/gpt-4o"
elif args.model == "claude3.5":
    API_PROVIDER = "openrouter"
    CHAT_BASE_URL = OPENROUTER_URL
    MODEL = "anthropic/claude-3.5-haiku"
elif args.model == "qwen":
    API_PROVIDER = "openrouter"
    CHAT_BASE_URL = OPENROUTER_URL
    MODEL = "qwen/qwen3-max"
elif args.model == "gemini":
    API_PROVIDER = "openrouter"
    CHAT_BASE_URL = OPENROUTER_URL
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
elif args.model == "ofox":
    API_PROVIDER = "ofox"
    CHAT_BASE_URL = os.getenv("OFOX_BASE_URL", "").rstrip("/")
    MODEL = os.getenv("OFOX_MODEL", "gpt-4o-mini")
else:
    raise ValueError("Use --model gemini, --model deepseek, --model deepseek-pro, or --model ofox.")
CHOOSE_MODEL = MODEL
MODEL_NAME = args.model  # short name (gemini/claude/...), used by the LM temporal method answer_question_with_time_lm
if args.re_model:
    if args.re_model == "gpt4.1mini":
        RE_MODEL = "openai/gpt-4.1-mini"
    elif args.re_model == "gpt4omini":
        RE_MODEL = "openai/gpt-4o-mini"
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
    elif args.re_model == "ofox":
        RE_MODEL = os.getenv("OFOX_RE_MODEL", os.getenv("OFOX_MODEL", "gpt-4o-mini"))
    else:
        RE_MODEL = MODEL
else:
    RE_MODEL = MODEL
if API_PROVIDER == "ofox":
    API_KEY = os.getenv("OFOX_API_KEY")
elif API_PROVIDER == "deepseek":
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
else:
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_COMPAT_DEFAULT_HEADERS = get_openrouter_headers()
if API_PROVIDER == "ofox" and not CHAT_BASE_URL:
    raise ValueError("OFOX_BASE_URL is empty. Set it in .env when using --model ofox.")
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
RETRIEVAL_ONLY = args.retrieval_only
EAES_CANDIDATE_LIMIT = 60
EAES_SELECTION_LIMIT = 30
EAES_RAW_EXPANSION_LIMIT = 3
sample_id = args.sample
MAX_QUESTIONS = args.max_questions
if MAX_QUESTIONS is not None and MAX_QUESTIONS <= 0:
    raise ValueError("--max_questions must be a positive integer.")
qu = args.qu
ca = args.ca
LM_REWRITE_BATCH = args.lm_batch  # sessions merged per LM rewrite call

dataset = args.data
DATASET = dataset
datapath = f"data/dataset_{dataset}.json"
ADDITIONAL_TK = f"_{args.model}"#"_gpt4o-mini"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_TAG = os.getenv(
    "EMBEDDING_TAG",
    "text_embedding_3_large" if EMBEDDING_PROVIDER in {"openrouter", "ofox"} else "local_bge"
)
ADDITIONAL_EM = f"_{args.model}_{EMBEDDING_TAG}"#
ADDITIONAL_RE = (
    f"_{args.model}_{args.file}"
    f"{'_q' + str(MAX_QUESTIONS) if MAX_QUESTIONS is not None else ''}"
    f"{'_eaes' if EAES_MODE else ''}"
    f"{'_retrieval' if RETRIEVAL_ONLY else ''}"
) #"_gpt4o-mini"
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

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
parser.add_argument("--max_questions", type=int, default=None, help="Run at most the first N questions per selected sample after category filtering.")
parser.add_argument("--exclude_categories", type=str, default=os.getenv("EXCLUDE_CATEGORIES", ""), help="Comma-separated question categories to skip, e.g. 5 or 3,5.")
parser.add_argument("--qu", type=int, default=0, help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--re_model", type=str, default=None, help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--ca", type=int, default=1, help="LM category index: 0=multi-session,1=single-session-user,2=temporal-reasoning,3=single-session-preference,4=knowledge-update,5=single-session-assistant")
parser.add_argument("--semantic_hierarchy", action="store_true", help="Build session-local Child memories and end-of-session Parent abstractions.")
parser.add_argument("--child_max_turns", type=int, default=int(os.getenv("CHILD_MAX_TURNS", "8")), help="Maximum raw turns in one semantic child segment.")
parser.add_argument(
    "--utility_token_budget",
    type=int,
    default=int(os.getenv("UTILITY_TOKEN_BUDGET", "2048")),
    help="Hard token budget for evidence selected by the unified utility judge.",
)
parser.add_argument("--parent_top_k", type=int, default=int(os.getenv("PARENT_TOP_K", "10")), help="Direct Parent retrieval count.")
parser.add_argument("--workers", type=int, default=int(os.getenv("MRA_WORKERS", "10")), help="Concurrent question workers per selected sample.")
parser.add_argument("--eaes_child_candidate_limit", type=int, default=int(os.getenv("EAES_CHILD_CANDIDATE_LIMIT", "30")), help="Direct Child retrieval count.")
parser.add_argument(
    "--eaes_parent_child_route_limit",
    type=int,
    default=int(os.getenv("EAES_PARENT_CHILD_ROUTE_LIMIT", "30")),
    help="Maximum children recovered through retrieved Parents.",
)
parser.add_argument("--eaes", action="store_true", help="Enable the required EAES-Mem retrieval and answer pipeline.")
parser.add_argument(
    "--disable_evidence_selector",
    action="store_true",
    help="Bypass SELECT and pass every pre-selector Child and Parent candidate directly to the final reader.",
)
parser.add_argument("--retrieval_only", action="store_true", help="Save retrieval evidence without running the final answer reader.")
parser.add_argument(
    "--parent_reader_passthrough",
    action="store_true",
    help="Pass recalled Parents directly to the reader and run SELECT on Children only.",
)
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
    MODEL = "deepseek-flash"
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
MODEL_NAME = args.model
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
        RE_MODEL = "deepseek-flash"
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
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300" if API_PROVIDER == "deepseek" else "120"))
LLM_CLIENT_MAX_RETRIES = int(os.getenv("LLM_CLIENT_MAX_RETRIES", "3"))
LLM_REQUEST_MAX_RETRIES = int(os.getenv("LLM_REQUEST_MAX_RETRIES", "8" if API_PROVIDER == "deepseek" else "3"))
LLM_BACKOFF = float(os.getenv("LLM_BACKOFF", "1.8" if API_PROVIDER == "deepseek" else "1.5"))
DEEPSEEK_THINKING_MODE = os.getenv("DEEPSEEK_THINKING_MODE", "disabled").lower()
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "deepseek-flash")
UTILITY_API_KEY = os.getenv("DEEPSEEK_API_KEY")
UTILITY_BASE_URL = DEEPSEEK_URL
UTILITY_TIMEOUT = float(os.getenv("UTILITY_TIMEOUT", "300"))
MODEL_SORT = MODEL #"anthropic/claude-sonnet-4.5"
K1=80                 # coarse retrieval breadth (embedding similarity)
K2=20                 # fine retrieval breadth (LLM re-ranking)
TOPIC_K=8            # select_topic: number of topic candidates
EAES_MODE = args.eaes
SEMANTIC_HIERARCHY = args.semantic_hierarchy
RETRIEVAL_ONLY = args.retrieval_only
DISABLE_EVIDENCE_SELECTOR = args.disable_evidence_selector
PARENT_READER_PASSTHROUGH = args.parent_reader_passthrough
UTILITY_TOKEN_BUDGET = args.utility_token_budget
if UTILITY_TOKEN_BUDGET <= 0:
    raise ValueError("Utility evidence budget must be positive.")
if DISABLE_EVIDENCE_SELECTOR and not EAES_MODE:
    raise ValueError("--disable_evidence_selector requires --eaes.")
if SEMANTIC_HIERARCHY and not EAES_MODE:
    raise ValueError("--semantic_hierarchy requires --eaes.")
EAES_CHILD_CANDIDATE_LIMIT = args.eaes_child_candidate_limit
EAES_PARENT_CHILD_ROUTE_LIMIT = args.eaes_parent_child_route_limit
if min(EAES_CHILD_CANDIDATE_LIMIT, EAES_PARENT_CHILD_ROUTE_LIMIT) <= 0:
    raise ValueError("EAES Child retrieval limits must be positive.")
sample_id = args.sample
MAX_QUESTIONS = args.max_questions
if MAX_QUESTIONS is not None and MAX_QUESTIONS <= 0:
    raise ValueError("--max_questions must be a positive integer.")
EXCLUDED_CATEGORIES = {
    item.strip() for item in str(args.exclude_categories or "").split(",") if item.strip()
}
QUESTION_WORKERS = args.workers
if QUESTION_WORKERS <= 0:
    raise ValueError("--workers must be a positive integer.")
qu = args.qu
ca = args.ca
CHILD_MAX_TURNS = args.child_max_turns
PARENT_TOP_K = args.parent_top_k
if CHILD_MAX_TURNS <= 0:
    raise ValueError("--child_max_turns must be positive.")
if PARENT_TOP_K <= 0:
    raise ValueError("--parent_top_k must be positive.")
dataset = args.data
DATASET = dataset
datapath = f"data/dataset_{dataset}.json"
HIERARCHY_SUFFIX = "_hierarchy" if SEMANTIC_HIERARCHY else ""
ADDITIONAL_TK = f"_{args.model}{HIERARCHY_SUFFIX}"#"_gpt4o-mini"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_TAG = os.getenv(
    "EMBEDDING_TAG",
    "text_embedding_3_large" if EMBEDDING_PROVIDER in {"openrouter", "ofox"} else "local_bge"
)
ADDITIONAL_EM = f"_{args.model}_{EMBEDDING_TAG}{HIERARCHY_SUFFIX}"#
ADDITIONAL_RE = (
    f"_{args.model}_{args.file}"
    f"{'_q' + str(MAX_QUESTIONS) if MAX_QUESTIONS is not None else ''}"
    f"{'_xcat' + '-'.join(sorted(EXCLUDED_CATEGORIES)) if EXCLUDED_CATEGORIES else ''}"
    f"{'_eaes' if EAES_MODE else ''}"
    f"{HIERARCHY_SUFFIX}"
    f"{'_no_selector' if DISABLE_EVIDENCE_SELECTOR else ''}"
    f"{'_parent_passthrough' if PARENT_READER_PASSTHROUGH else ''}"
    f"{'_retrieval' if RETRIEVAL_ONLY else ''}"
) #"_gpt4o-mini"
base_dir_t = f"data/{{dataset}}/rewrite{ADDITIONAL_TK}/"
base_dir_emb = f"data/{{dataset}}/embedding/gpt{ADDITIONAL_EM}/"
# auto-create all output dirs so a fresh checkout runs without manual mkdir
os.makedirs(base_dir_t.format(dataset=dataset), exist_ok=True)      # data/<ds>/rewrite_<model>/
os.makedirs(base_dir_emb.format(dataset=dataset), exist_ok=True)    # data/<ds>/embedding/gpt_<model>/
os.makedirs(f"result/{dataset}", exist_ok=True)                    # prediction outputs
os.makedirs(f"log/{dataset}", exist_ok=True)                       # logs (also creates log/ for the run-level handler)

rewrite_template = f"data/{{dataset}}/rewrite{ADDITIONAL_TK}/{{sample_id}}_rewrite.json"
embedding_template = f"data/{{dataset}}/embedding/gpt{ADDITIONAL_EM}/{{sample_id}}_embedding.pkl"
result_template = f"result/{{dataset}}/{{sample_id}}_result{ADDITIONAL_RE}.jsonl"

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
parser.add_argument("--lm_batch", type=int, default=1, help="LM: sessions merged per rewrite call. 1=per-session (key=session_i, compatible with existing files/per-session readers); >1=merged (key=session_first-session_last)")
parser.add_argument("--rewrite_window_size", type=int, default=int(os.getenv("REWRITE_WINDOW_SIZE", "40")), help="Current dialogue turns per session-local rewrite window, excluding previous context.")
parser.add_argument("--rewrite_overlap_size", type=int, default=int(os.getenv("REWRITE_OVERLAP_SIZE", "2")), help="Tail turns from the preceding window shown as context for cross-window completion.")
parser.add_argument("--rewrite_previous_limit", type=int, default=int(os.getenv("REWRITE_PREVIOUS_LIMIT", "3")), help="Previous compressed memories shown to the next rewrite window for deduplication.")
parser.add_argument("--workers", type=int, default=int(os.getenv("MRA_WORKERS", "10")), help="Concurrent question workers per selected sample.")
parser.add_argument("--dense_k", type=int, default=int(os.getenv("DENSE_RETRIEVAL_K", "80")), help="Global dense retrieval candidates mixed into retrieval-only diagnostics.")
parser.add_argument("--query_key_mode", choices=["inventory", "extract"], default=os.getenv("QUERY_KEY_MODE", "inventory"), help="Question-key strategy: select from stored keys or freely extract keywords.")
parser.add_argument("--key_candidate_dense_k", type=int, default=int(os.getenv("KEY_CANDIDATE_DENSE_K", "40")), help="Dense events used to build the stored-key candidate pool.")
parser.add_argument("--key_candidate_limit", type=int, default=int(os.getenv("KEY_CANDIDATE_LIMIT", "120")), help="Maximum stored-key candidates shown to the LLM.")
parser.add_argument("--eaes_index_mode", choices=["llm", "heuristic"], default=os.getenv("EAES_INDEX_MODE", "llm"), help="EAES memory index construction strategy.")
parser.add_argument("--eaes_prefilter_limit", type=int, default=int(os.getenv("EAES_PREFILTER_LIMIT", "120")), help="Combined-score candidates kept before EAES LLM reranking.")
parser.add_argument("--eaes_rerank_limit", type=int, default=int(os.getenv("EAES_RERANK_LIMIT", "30")), help="Memories kept by the EAES attribute reranker for evidence selection.")
parser.add_argument("--eaes", action="store_true", help="Use EAES-Mem answer-oriented evidence selection instead of the default graph tool loop.")
parser.add_argument(
    "--disable_evidence_selector",
    action="store_true",
    help="EAES answer ablation: bypass the evidence selector and pass all reranked candidates directly to the final reader.",
)
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
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300" if API_PROVIDER == "deepseek" else "120"))
LLM_CLIENT_MAX_RETRIES = int(os.getenv("LLM_CLIENT_MAX_RETRIES", "3"))
LLM_REQUEST_MAX_RETRIES = int(os.getenv("LLM_REQUEST_MAX_RETRIES", "8" if API_PROVIDER == "deepseek" else "3"))
LLM_BACKOFF = float(os.getenv("LLM_BACKOFF", "1.8" if API_PROVIDER == "deepseek" else "1.5"))
DEEPSEEK_THINKING_MODE = os.getenv("DEEPSEEK_THINKING_MODE", "disabled").lower()
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
DISABLE_EVIDENCE_SELECTOR = args.disable_evidence_selector
if DISABLE_EVIDENCE_SELECTOR and not EAES_MODE:
    raise ValueError("--disable_evidence_selector requires --eaes.")
if DISABLE_EVIDENCE_SELECTOR and RETRIEVAL_ONLY:
    raise ValueError(
        "--disable_evidence_selector cannot change --retrieval_only metrics because "
        "retrieval-only stops before the evidence selector. Run the answer ablation without "
        "--retrieval_only and compare F1/LLM-judge instead."
    )
EAES_CANDIDATE_LIMIT = args.eaes_prefilter_limit
if EAES_CANDIDATE_LIMIT <= 0:
    raise ValueError("--eaes_prefilter_limit must be a positive integer.")
EAES_RERANK_LIMIT = args.eaes_rerank_limit
if EAES_RERANK_LIMIT <= 0 or EAES_RERANK_LIMIT > EAES_CANDIDATE_LIMIT:
    raise ValueError("--eaes_rerank_limit must be positive and no larger than --eaes_prefilter_limit.")
EAES_RAW_EXPANSION_LIMIT = 3
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
DENSE_RETRIEVAL_K = args.dense_k
if DENSE_RETRIEVAL_K <= 0:
    raise ValueError("--dense_k must be a positive integer.")
QUERY_KEY_MODE = args.query_key_mode
KEY_CANDIDATE_DENSE_K = args.key_candidate_dense_k
if KEY_CANDIDATE_DENSE_K <= 0:
    raise ValueError("--key_candidate_dense_k must be a positive integer.")
KEY_CANDIDATE_LIMIT = args.key_candidate_limit
if KEY_CANDIDATE_LIMIT <= 0:
    raise ValueError("--key_candidate_limit must be a positive integer.")
EAES_INDEX_MODE = args.eaes_index_mode
qu = args.qu
ca = args.ca
LM_REWRITE_BATCH = args.lm_batch  # sessions merged per LM rewrite call
REWRITE_WINDOW_SIZE = args.rewrite_window_size
if REWRITE_WINDOW_SIZE <= 0:
    raise ValueError("--rewrite_window_size must be a positive integer.")
REWRITE_OVERLAP_SIZE = args.rewrite_overlap_size
if REWRITE_OVERLAP_SIZE < 0 or REWRITE_OVERLAP_SIZE >= REWRITE_WINDOW_SIZE:
    raise ValueError("--rewrite_overlap_size must be >= 0 and smaller than --rewrite_window_size.")
REWRITE_PREVIOUS_LIMIT = args.rewrite_previous_limit
if REWRITE_PREVIOUS_LIMIT < 0:
    raise ValueError("--rewrite_previous_limit must be non-negative.")

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
    f"{'_xcat' + '-'.join(sorted(EXCLUDED_CATEGORIES)) if EXCLUDED_CATEGORIES else ''}"
    f"{'_eaes' if EAES_MODE else ''}"
    f"{'_no_selector' if DISABLE_EVIDENCE_SELECTOR else ''}"
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

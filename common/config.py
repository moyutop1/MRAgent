import argparse
import os

from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="MRAgent with shared set utility")
parser.add_argument("--data", default="locomo")
parser.add_argument("--model", default="deepseek-pro")
parser.add_argument("--re_model", default=None)
parser.add_argument("--file", default="0")
parser.add_argument("--sample", type=int, default=None)
parser.add_argument("--max_questions", type=int, default=None)
parser.add_argument("--exclude_categories", default=os.getenv("EXCLUDE_CATEGORIES", ""))
parser.add_argument("--workers", type=int, default=int(os.getenv("MRA_WORKERS", "8")))
parser.add_argument("--ca", type=int, default=1)
parser.add_argument("--lm_batch", type=int, default=1)
parser.add_argument("--retrieval_only", action="store_true")
parser.add_argument("--dump_candidates", action="store_true")
parser.add_argument("--utility_checkpoint", default=os.getenv("UTILITY_CHECKPOINT", "checkpoints/set_utility.pt"))
parser.add_argument("--write_threshold", type=float, default=float(os.getenv("WRITE_THRESHOLD", "0.5")))
parser.add_argument("--retrieve_threshold", type=float, default=float(os.getenv("RETRIEVE_THRESHOLD", "0.5")))
args, _ = parser.parse_known_args()

OPENROUTER_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEEPSEEK_URL = "https://api.deepseek.com"
MODEL_MAP = {
    "gemini": ("openrouter", "google/gemini-2.5-flash"),
    "gpt4.1mini": ("openrouter", "openai/gpt-4.1-mini"),
    "qwen": ("openrouter", "qwen/qwen3-max"),
    "deepseek": ("deepseek", "deepseek-flash"),
    "deepseek-pro": ("deepseek", "deepseek-v4-pro"),
    "deepseek-chat": ("deepseek", "deepseek-chat"),
    "deepseek-reasoner": ("deepseek", "deepseek-reasoner"),
}

API_PROVIDER, MODEL = MODEL_MAP[args.model]
CHAT_BASE_URL = DEEPSEEK_URL if API_PROVIDER == "deepseek" else OPENROUTER_URL
API_KEY = os.getenv("DEEPSEEK_API_KEY") if API_PROVIDER == "deepseek" else os.getenv("OPENROUTER_API_KEY")
RE_MODEL = MODEL_MAP[args.re_model][1] if args.re_model else MODEL
MODEL_NAME = args.model
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
DEEPSEEK_THINKING_MODE = os.getenv("DEEPSEEK_THINKING_MODE", "disabled")

TAG_MAX = 15
TAG_LIMIT = 10
TOPIC_K = 8
LM_REWRITE_BATCH = args.lm_batch
QUESTION_WORKERS = args.workers
RETRIEVAL_ONLY = args.retrieval_only
DUMP_CANDIDATES = args.dump_candidates
UTILITY_CHECKPOINT = args.utility_checkpoint
UTILITY_DEVICE = os.getenv("UTILITY_DEVICE")
WRITE_THRESHOLD = args.write_threshold
RETRIEVE_THRESHOLD = args.retrieve_threshold
sample_id = args.sample
MAX_QUESTIONS = args.max_questions
EXCLUDED_CATEGORIES = {
    item.strip() for item in args.exclude_categories.split(",") if item.strip()
}
ca = args.ca
dataset = args.data
DATASET = dataset
datapath = f"data/dataset_{dataset}.json"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_TAG = os.getenv("EMBEDDING_TAG", "local_bge" if EMBEDDING_PROVIDER == "local" else "remote")
CANDIDATE_SUFFIX = "_candidate_dump" if DUMP_CANDIDATES else ""
ADDITIONAL_TK = f"_{args.model}_set_utility{CANDIDATE_SUFFIX}"
ADDITIONAL_EM = f"_{args.model}_{EMBEDDING_TAG}_set_utility{CANDIDATE_SUFFIX}"
ADDITIONAL_RE = (
    f"_{args.model}_{args.file}_set_utility"
    f"{'_q' + str(MAX_QUESTIONS) if MAX_QUESTIONS else ''}"
    f"{'_retrieval' if RETRIEVAL_ONLY else ''}"
    f"{'_candidates' if DUMP_CANDIDATES else ''}"
)

base_dir_t = f"data/{{dataset}}/rewrite{ADDITIONAL_TK}/"
base_dir_k = f"data/{{dataset}}/keyword{ADDITIONAL_TK}/"
base_dir_emb = f"data/{{dataset}}/embedding/gpt{ADDITIONAL_EM}/"
for path in (
    base_dir_t.format(dataset=dataset),
    base_dir_k.format(dataset=dataset),
    base_dir_emb.format(dataset=dataset),
    f"result/{dataset}",
    f"log/{dataset}",
):
    os.makedirs(path, exist_ok=True)

rewrite_template = f"data/{{dataset}}/rewrite{ADDITIONAL_TK}/{{sample_id}}_rewrite.json"
keyword_template = f"data/{{dataset}}/keyword{ADDITIONAL_TK}/{{sample_id}}_keyword.json"
embedding_template = f"data/{{dataset}}/embedding/gpt{ADDITIONAL_EM}/{{sample_id}}_embedding.pkl"
result_template = f"result/{{dataset}}/{{sample_id}}_result{ADDITIONAL_RE}.jsonl"

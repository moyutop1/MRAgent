# MRAgent

Code for hierarchical conversational memory on LoCoMo and LongMemEval.

## Core pipeline

The implementation uses one set-conditioned marginal-utility rule in memory
construction and retrieval. DeepSeek Flash returns only `KEEP` or `DROP` for
three operations:

- `WRITE`: inside the current session, retain a proposed Child only when it adds
  a new event/state, time/stage, or concrete detail.
- `ABSTRACT`: at session end, retain a Parent only when it is a supported and
  useful high-level abstraction of its Child group and the complete raw session.
- `SELECT`: for one query, retain a retrieved Child or Parent only when it adds
  answer-useful information beyond the already selected evidence.

Construction is isolated by session. Parent nodes are create-only and never
cross session boundaries. Query-time retrieval sees every stored session.

Retrieval uses the raw question directly:

1. Child Top-K by cosine over `tag + rewrite_content`.
2. Parent Top-K by cosine over Parent `rewrite_content`.
3. Expand retrieved Parents and rescore routed Children only by query-Child
   cosine; Parent scores never boost Child scores.
4. Merge Children by ID, sort Child and Parent candidates by query score, and run
   sequential `SELECT` under a hard evidence-token budget.
5. Send only selected evidence to the answer model.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set at least:

```dotenv
DEEPSEEK_API_KEY=...
UTILITY_MODEL=deepseek-flash
UTILITY_TIMEOUT=300
UTILITY_TOKEN_BUDGET=2048
LOCAL_EMBEDDING_MODEL=/autodl-pub/models/bge-large-en-v1.5
```

## Run

Full LoCoMo answer run with DeepSeek Pro:

```bash
python run.py \
  --data locomo \
  --model deepseek-pro \
  --file unified_utility \
  --workers 8 \
  --eaes \
  --semantic_hierarchy
```

Retrieval-only run and evaluation:

```bash
python run.py \
  --data locomo \
  --model deepseek-pro \
  --file unified_retrieval \
  --workers 8 \
  --retrieval_only \
  --eaes \
  --semantic_hierarchy

python eval/evaluate_retrieval.py \
  --data locomo \
  --model deepseek-pro \
  --file unified_retrieval \
  --allfile \
  --eaes \
  --semantic_hierarchy
```

Reasoning evaluation after the full answer run:

```bash
python eval/evaluate_reasoning.py \
  --data locomo \
  --model deepseek-pro \
  --file unified_utility_eaes_hierarchy \
  --allfile
```

Use `--sample 42` for one LoCoMo conversation and `--max_questions 50` for a
short run. Delete that sample's rewrite and embedding caches before regenerating
memories with a changed construction pipeline.

# MRAgent

> This repository contains the code for the paper
> **"Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"** ([arXiv:2606.06036](https://arxiv.org/abs/2606.06036)).

A retrieval-augmented question-answering system that builds a **graph-structured
episodic memory** from long multi-session dialogues and answers questions through an
**LLM tool-calling reasoning loop**. The system is evaluated on the **LoCoMo** and
**LongMemEval (LM)** benchmarks.

The pipeline has two phases:

**Phase 1 — Build the graph memory** (once per conversation sample):

- **rewrite** — compress dialogue into self-contained memories: resolve pronouns, render relative times at their source granularity, store normalized `YYYY-MM-DD` start dates for indexing, attach topic tags, and extract topics and person-level facts.
- **extract_keyword** — extract salient keywords for each rewritten sentence.
- **store** — build the in-memory graph from the above: key nodes, episode / topic / personal events, and the links between them.

**Phase 2 — Answer questions** (per question):

- **answer** — run a tool-calling reasoning loop (keyword / topic / personal / temporal / context tools) to produce a short final answer.

---

## 1. Repository Structure

```
run.py                    # main entry point
common/
    config.py             # CLI args, models, paths
    utils.py              # JSON + similarity helpers
    logging_utils.py      # per-sample logging
memory/
    system.py             # in-memory graph store
    controller.py         # graph query tools
llm/
    controller.py         # LLM tool-calling wrapper
    embeddings.py         # text-embedding client
    rag_utils.py          # batched embedding helper
agent/
    agent.py              # pipeline orchestration
    tools.py              # tool schemas + dispatch
prompts/
    prompts.py            # all LLM prompts
    schema.py             # output JSON validators
data/
    get_data.py           # load benchmark dataset
    embed_rewrite.py      # build embedding files
    dataset_locomo.json   # LoCoMo benchmark data
    dataset_LM.json       # LongMemEval benchmark data
eval/
    judge.py              # LLM-as-judge scorer
    evaluation.py         # F1 / EM helpers
    evaluate_reasoning.py # eval entry (F1 + judge)
```

---

## 2. Installation

Python 3.9+. The LongMemEval dataset (`data/dataset_LM.json`) is stored with **Git LFS**,
so install Git LFS *before* cloning, otherwise that file arrives as a small pointer stub.

```bash
# install Git LFS once: https://git-lfs.com   (e.g. `apt install git-lfs` / `brew install git-lfs`)
git lfs install
git clone https://github.com/Ji-shuo/MRAgent.git
cd MRAgent

pip install -r requirements.txt
```

> `torch` is used only for embedding tensor ops (L2 normalization); a CPU build is
> sufficient. `nltk` is required only by the `eval/` scripts.

---

## 3. Configuration

All components — the chat LLM, the text-embedding model, and the LLM-as-judge
evaluator — are accessed through a single **OpenRouter** key (OpenAI-compatible API).
The key is read from a `.env` file at the repository root; **no key is hard-coded**.

Copy the template and fill in your key:

```bash
cp .env.example .env
# then edit .env:
# OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
```

`.env` is git-ignored. The same `OPENROUTER_API_KEY` is used everywhere:

| Component | File | Model |
| --- | --- | --- |
| Chat / reasoning | `common/config.py` (`--model` → `MODEL`) | e.g. `gemini` → `google/gemini-2.5-flash` |
| Embedding | `llm/embeddings.py` | `openai/text-embedding-3-large` (3072-d) |
| LLM-as-judge | `eval/judge.py` | `openai/gpt-4o-mini` |

Quick connectivity check:

```bash
python try.py
```

If the account has no balance yet, this command should still distinguish config/network
errors from OpenRouter billing or authentication errors.

---

## 4. Data Layout

The two benchmarks shipped in `data/` come from:

- **LoCoMo** (`dataset_locomo.json`) — Maharana et al., *Evaluating Very Long-Term
  Conversational Memory of LLM Agents*, ACL 2024. [arXiv:2402.17753](https://arxiv.org/abs/2402.17753)
- **LongMemEval** (`dataset_LM.json`) — Wu et al., *LongMemEval: Benchmarking Chat
  Assistants on Long-Term Interactive Memory*, ICLR 2025. [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)

Place the benchmark file at `data/dataset_<name>.json`. Generated intermediate
artifacts and results are written under per-dataset subfolders:

```
data/<dataset>/rewrite_<model>/<sample_id>_rewrite.json   # stage 1 output
data/<dataset>/keyword_<model>/<sample_id>_keyword.json       # stage 3 output
data/<dataset>/embedding/gpt_<model>/<sample_id>_embedding.pkl# stage 2 output
result/<dataset>/<sample_id>_result_<model>_<file>.jsonl      # predictions (the only run output)
```

The run writes a single output per sample — the `_result_*.jsonl` predictions file
(one JSON line per question: gold answer, prediction, category, evidence labels,
retrieved support). A stage is **skipped if its output file already exists**, so
generation runs once and subsequent runs reuse the cached `rewrite` / `keyword` /
`embedding` files.

---

## 5. Usage

The single entry point is `run.py`, invoked from the repository root.

### 5.1 Arguments

| Argument | Meaning | Default |
| --- | --- | --- |
| `--data` | dataset name (`locomo` / `LM`) | `locomo` |
| `--model` | chat model short name (`gemini` / `claude` / `gpt4o` / `qwen`) | `gemini` |
| `--file` | run/experiment tag appended to result filenames | `0` |
| `--sample` | (LoCoMo) run a single sample id, e.g. `42`; omit to run all | `None` |
| `--max_questions` | run at most the first N questions per selected sample after category filtering | `None` |
| `--exclude_categories` | comma-separated question categories to skip, e.g. `5` | none |
| `--ca` | (LM) category index: `0`=multi-session, `1`=single-session-user, `2`=temporal-reasoning | `1` |
| `--lm_batch` | (LM) sessions merged per rewrite call (`1` recommended) | `1` |
| `--rewrite_window_size` | Current raw turns per rewrite window, excluding overlap context | `40` |
| `--rewrite_overlap_size` | Tail turns from the preceding window supplied as raw context | `2` |
| `--rewrite_previous_limit` | Previous compressed memories supplied for deduplication | `3` |
| `--workers` | concurrent question workers per selected sample | `10` |
| `--query_key_mode` | question-key strategy (`inventory` selects from stored keys; `extract` uses free extraction) | `inventory` |
| `--eaes_index_mode` | EAES memory index strategy (`llm` builds entity/attribute notes; `heuristic` uses keyword-derived notes) | `llm` |
| `--eaes_prefilter_limit` | combined-score candidates retained before LLM reranking | `120` |
| `--eaes_rerank_limit` | memories retained by the attribute LLM reranker for evidence selection | `30` |
| `--eaes_typed_memory` | enable orthogonal semantic memory types and persistence as weak EAES retrieval bonuses | off |
| `--eaes_type_weight` | semantic-type compatibility bonus weight | `0.15` |
| `--eaes_persistence_weight` | persistence compatibility bonus weight | `0.05` |
| `--disable_evidence_selector` | EAES answer ablation: pass all reranked candidates directly to the final reader | off |

### 5.2 LoCoMo

```bash
# all conversations
python run.py --data locomo --model gemini --file myrun

# a single conversation
python run.py --data locomo --model gemini --file myrun --sample 42

# a quick partial run over the first 50 questions of one conversation
python run.py --data locomo --model qwen --file smoke50 --sample 26 --max_questions 50

# a conservative DeepSeek run for unstable networks
python run.py --data locomo --model deepseek --file smoke50 --sample 26 --max_questions 50 --workers 1

# retrieval-only diagnostics with global dense fallback mixed in
python run.py --data locomo --model deepseek-chat --file retr50 --sample 26 --max_questions 50 --workers 1 --retrieval_only
python eval/evaluate_retrieval.py --data locomo --model deepseek-chat --file retr50_q50 --sample conv-26

# entity-attribute-memory retrieval diagnostics
python run.py --data locomo --model deepseek-chat --file eaes50 --sample 26 --max_questions 50 --workers 1 --retrieval_only --eaes --eaes_index_mode llm --eaes_prefilter_limit 120 --eaes_rerank_limit 30
python eval/evaluate_retrieval.py --data locomo --model deepseek-chat --file eaes50_q50_eaes --sample conv-26

# answer-stage ablation: bypass the EAES evidence selector (do not combine with --retrieval_only)
python run.py --data locomo --model deepseek-chat --file no_selector --sample 26 --workers 1 --eaes --disable_evidence_selector --eaes_index_mode llm --eaes_prefilter_limit 120 --eaes_rerank_limit 30

# typed-memory retrieval ablation (adds the _typed suffix)
python run.py --data locomo --model deepseek-chat --file typed_retrieval --sample 26 --workers 1 --retrieval_only --eaes --eaes_typed_memory --eaes_index_mode llm --eaes_prefilter_limit 120 --eaes_rerank_limit 30
python eval/evaluate_retrieval.py --data locomo --model deepseek-chat --file typed_retrieval --sample conv-26 --eaes --typed_memory

# compare against the older free keyword extraction path
python run.py --data locomo --model deepseek-chat --file retr50_extract --sample 26 --max_questions 50 --workers 1 --retrieval_only --query_key_mode extract
```

### 5.3 LongMemEval (LM)

`--ca` selects the question category (one run per category):

```bash
python run.py --data LM --model gemini --file myrun --ca 0 --lm_batch 10   # multi-session
python run.py --data LM --model gemini --file myrun --ca 1 --lm_batch 10   # single-session-user
python run.py --data LM --model gemini --file myrun --ca 2 --lm_batch 10   # temporal-reasoning
```

`--lm_batch` controls rewrite granularity. `--lm_batch 1` (default) rewrites one
session per LLM call and produces per-session records compatible with all downstream
readers. Values `>1` merge multiple sessions per call (range-keyed records, handled by
the robust readers and the origin-prefixed graph store).

Each question is answered concurrently (10 worker threads); predictions stream to the
`result/<dataset>/` files and runs are resumable (already-answered questions are
skipped on restart).

---

## 6. Evaluation

```bash
# F1 + LLM-as-judge accuracy (writes result_judge_<data>_<model>_<file>.jsonl)
python eval/evaluate_reasoning.py --data locomo --model gemini --file myrun --allfile
```

The LLM judge (`eval/judge.py`) grades a prediction `CORRECT`/`WRONG` against the gold
answer with `gpt-4o-mini`, using lenient matching (topic overlap; date equivalence for
temporal questions).

---

## 7. Notes

- The pipeline is **cache-based**: delete the corresponding `rewrite` / `keyword` /
  `embedding` files to force regeneration of a sample.
- Per-sample reasoning traces are logged under `log/<dataset>/`.
- Tool inventory (7 tools): `edges_by_tag`, `query_conversation_time`,
  `query_event_keywords`, `query_event_context`, `query_personal_information`,
  `query_personal_aspect`, `query_topic_events`.

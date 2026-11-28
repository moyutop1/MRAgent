# MRAgent with unified set-conditioned utility

This implementation keeps MRAgent's Cue–Tag, Topic, and Personal memory
structure, but uses one learned marginal-utility estimator for both memory
writing and retrieval.

## Mechanism

### WRITE

Each raw session is evaluated once before rewriting:

```text
U(raw session | retained sessions, WRITE)
  -> DROP
  -> KEEP -> one session rewrite -> keywords/topics/personal memory -> store
```

There is no utility decision after rewriting.

### RETRIEVE

The original coarse routing context contains only:

- `keys_candidates`: question Cue–Tag pairs;
- `similar_topic`: the eight closest Topic summaries.

`key_sentences` and `key_tag_sentences` are not used. Person–Aspect routes are
enumerated once from people recalled in the question. Every fixed route is
expanded exactly once through:

- `edges_by_tag(Cue, Tag)`;
- `query_topic_events(Topic)`;
- `query_personal_aspect(Person, Aspect)`.

All returned Content nodes form one fixed candidate set. Accepted nodes do not
create new routes. There is no second retrieval, pruning, context tool, time
tool, or keyword-expansion tool.

For every candidate, the estimator sees the question, the current selected
Content set, all recalled Cue–Tag pairs, all recalled Topic summaries, and the
candidate/current-set time fields.

## Utility model

`utility/model.py` implements four attention blocks:

1. two self-attention blocks over the current set;
2. candidate-to-set cross attention;
3. candidate-to-context cross attention.

The default hidden size is 384 with six heads. The same parameters serve WRITE
and RETRIEVE; a learned operation embedding distinguishes them.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

## Build LoCoMo training data

First dump fixed retrieval candidates. Candidate-dump mode keeps every raw
session only to construct the supervised training corpus; normal runs always
apply WRITE utility.

```bash
python run.py \
  --data locomo --model deepseek-pro --file utility_candidates \
  --dump_candidates
```

Then build unified WRITE/RETRIEVE records. WRITE labels retain a session when
it contains at least one LoCoMo gold evidence origin. RETRIEVE labels retain the
first candidate covering each still-uncovered gold origin; other sampled
candidates are negatives.

```bash
python -m utility.build_locomo_data \
  --candidate_results result/locomo/*_utility_candidates*_candidates.jsonl \
  --output data/utility/locomo_set_utility.pt
```

Train the four-block model:

```bash
python -m utility.train \
  --data data/utility/locomo_set_utility.pt \
  --output checkpoints/set_utility.pt
```

## Run

```bash
python run.py \
  --data locomo --model deepseek-pro --file set_utility \
  --utility_checkpoint checkpoints/set_utility.pt \
  --workers 8
```

Retrieval-only evaluation uses the same WRITE and RETRIEVE utility path:

```bash
python run.py \
  --data locomo --model deepseek-pro --file set_utility \
  --utility_checkpoint checkpoints/set_utility.pt \
  --retrieval_only

python eval/evaluate_set_retrieval.py \
  "result/locomo/*_set_utility_retrieval.jsonl"
```

## Tests

```bash
python -m unittest tests.test_set_utility_pipeline -v
```

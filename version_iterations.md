# Version Iterations

## v106-20260630

### Goal

Reduce lifecycle-related retrieval misses in EAES memory retrieval without adding finer lifecycle categories.

### Changes

- Treat `event_lifecycle` as a weak rerank bonus instead of a strong ranking signal.
  - Matching `planned/current/historical` now adds only a small bonus.
  - Mismatched lifecycle no longer receives a negative penalty.
- Add deterministic EAES query-plan postprocessing for stable fact/profile questions.
  - Questions without explicit temporal or event constraints can be normalized to:
    - `required_lifecycle = "unknown"`
    - `temporal_intent = "none"`
    - `no_time_limit = true`
  - Targeted examples include identity, relationship status, preferences, interests, activities, membership, allyship, career fields, and kinds/types of art.
- Update EAES query prompts to expose `no_time_limit` and encourage `unknown` lifecycle for stable fact/profile questions.

### Expected Effect

- Improve recall for answer-bearing memories whose sentence lifecycle differs from the question-level lifecycle inferred by the LLM.
- Reduce cases where current-state questions incorrectly suppress historical evidence that supports stable facts.
- Preserve the three lifecycle labels (`planned`, `current`, `historical`) while making their use uncertainty-tolerant.

### Suggested Comparison

Run v106 against the same conv-26 retrieval-only setup used for v105, then compare:

- `hit`
- `recall`
- `exact_cover`
- `mrr`
- distribution of `query_plan.required_lifecycle`
- count of `query_plan.no_time_limit = true`

Suggested output tag: `v106_20260630`.

# Version Iterations

## v107-20260702

### Goal

Improve final-answer reliability for EAES runs, especially temporal questions, without changing memory construction or retrieval breadth.

### Changes

- Strengthen the EAES final-answer prompt for time questions.
  - Require relative time phrases to be normalized with `time_interval.start`.
  - Prefer a single best time for single-time questions.
  - Forbid merging conflicting dates from multiple similar candidates.
  - Treat `evidence_package` as primary evidence and use `backup_candidates` only when needed.
- Make LLM-judge parsing more robust.
  - The judge prompt now asks for JSON only, with no explanation.
  - Malformed or truncated judge responses are retried once.
  - If parsing still fails, the item is counted as wrong instead of crashing the whole evaluation.

### Expected Effect

- Reduce answers like `last Friday`, `last year`, or multiple conflicting dates when an absolute or anchored time answer is required.
- Prevent evaluation runs from stopping on malformed judge outputs such as a truncated `{"`.

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

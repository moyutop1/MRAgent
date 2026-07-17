# Version Iterations

## v112-20260717

### Goal

Restore previous rewrite memories after the no-previous-memory ablation reduced performance, while making LLM-judge grading coverage-oriented for complete answers that include additional candidates.

### Changes

- Restore accumulation of prior window rewrite memories and expose up to `--rewrite_previous_limit` entries to the next rewrite call.
- Keep previous raw-dialogue context and the v111 retry behavior: mixed outputs discard only context-only items, while all-context-only outputs retry at most three times.
- Restore the `--rewrite_previous_limit` option with default `3`.
- Change the LLM-judge to treat the gold answer as a sufficient reference rather than an exhaustive list.
- Grade a generated answer as correct when it affirmatively covers every gold-answer element, even if it also contains other answers, alternatives, or extra details.
- Treat extra content as disqualifying only when it explicitly denies or directly contradicts the covered gold answer.

### Expected Effect

- Recover the rewrite quality and deduplication benefit of previous compressed memories without reintroducing whole-window loss from mixed context-only output.
- Reduce judge false negatives for complete but non-exclusive generated answers.

## v111-20260717

### Goal

Evaluate previous raw-dialogue context without exposing earlier compressed rewrite memories to the rewrite model, while avoiding whole-window loss from isolated context-only outputs.

### Changes

- Remove `PREVIOUS_REWRITE_MEMORIES` from the rewrite prompt and stop accumulating prior window memories for later rewrite calls.
- Remove the unused `--rewrite_previous_limit` configuration option.
- Keep previous raw-dialogue context and the fixed-size current-window behavior introduced in v109.
- When an output mixes current-supported and context-only memories, discard only the context-only items without retrying.
- Retry when every returned memory item is context-only, sharing the existing limit of at most three retries with schema-validation failures.
- Remove topics that become unreferenced after context-only sentences are discarded.
- Keep final exact `(lowercase text, normalized origin)` merge deduplication unchanged.

### Expected Effect

- Isolate the effect of raw cross-window context from model-level deduplication using previous compressed memories.
- Preserve valid current-window memories when the same LLM output also contains an overlap-only duplicate.

## v110-20260716

### Goal

Improve LLM-judge accuracy for semantically equivalent answers expressed with different wording.

### Changes

- Judge the answer-bearing proposition instead of requiring lexical overlap.
- Explicitly accept synonyms, paraphrases, noun/verb alternations, and longer non-contradictory formulations.
- Add `school speech` versus `gave a talk at a school event` as a positive calibration example.
- Add a same-topic counterexample so merely mentioning or attending a school event does not count as giving a speech.
- Preserve strict handling of contradictions and answer-critical differences in person, negation, quantity, completion status, time, and place.

### Expected Effect

- Reduce false negatives caused by synonymous or paraphrased answers without broadly accepting answers that only share the same topic.

## v109-20260716

### Goal

Preserve answer-bearing qualifiers when a question and its answer straddle adjacent rewrite windows.

### Changes

- Keep the original fixed-size current windows and prepend up to `--rewrite_overlap_size` preceding raw turns as explicit previous-dialogue context.
- Tell the rewrite model to use previous raw turns for cross-window question/answer completion, reference resolution, and time/place/entity qualifiers.
- Require cross-window memories to cite all contributing dialogue origins, including a context question and its current-window answer.
- Reject outputs supported only by overlap context so repeated raw turns do not create duplicate memories.
- Validate and normalize each output against the combined context plus current source, allowing temporal cues such as `last week` in the preceding question to be restored deterministically.

### Expected Effect

- A boundary pair such as `Where did you go last week?` followed by `I went to the national park with my kids` becomes one self-contained memory that preserves the week-level time constraint and both source origins.
- Each rewrite call contains up to `rewrite_window_size + rewrite_overlap_size` raw turns; overlap no longer reduces the current window's capacity.
- Existing `--rewrite_window_size`, `--rewrite_overlap_size`, and `--rewrite_previous_limit` controls remain compatible.

## v108-20260708

### Goal

Adopt a SimpleMem-style memory creation stage while preserving the existing keyword, EAES attribute, and memory retrieval layers.

### Changes

- Replace per-session sentence-preserving rewrite with session-local windowed memory compression.
  - Each session is processed with sliding windows controlled by `--rewrite_window_size` and `--rewrite_overlap_size`.
  - Windows never cross session boundaries.
  - Each window receives up to `--rewrite_previous_limit` previously generated rewrite memories to reduce duplicate memories.
- Allow low-value dialogue turns to be omitted during rewrite.
  - The rewrite prompt now asks for compact, self-contained memories instead of preserving every sentence.
  - Greetings, acknowledgements, generic advice, and repeated confirmations can be dropped.
- Preserve LoCoMo evidence alignment with multi-origin compressed memories.
  - A memory can now use comma-separated source origins such as `D1:12,D1:13`.
  - Final memory ids are generated deterministically from the first origin, e.g. `D1:12-1`.
  - Schema validation checks that every source origin exists in the source dialogue window.
- Use rewrite memory as the stored event text.
  - `EpisodeEvent.text` now stores the compressed rewrite memory text.
  - Raw dialogue ids remain only as provenance through `origin`.
- Update multi-origin compatibility in retrieval/evaluation support paths.
  - Gold-origin diagnostics can map `D1:13` to a compressed memory whose origin is `D1:12,D1:13`.
  - Time-filtered graph retrieval checks all source origins in a compressed memory.
  - Event-context expansion handles compressed memories with multiple source origins.
- Split windowed rewrite creation into `agent/rewrite_memory.py`.
  - `agent.py` keeps a thin `rewrite()` entrypoint and origin helper wrappers.
  - The windowing, previous-memory prompt context, schema retry, and session merge logic live in the new module.
- Split large Agent responsibilities into mixins so no agent file exceeds 1000 lines.
  - `agent/eaes.py` contains EAES memory indexing, query parsing, evidence selection, and EAES answering.
  - `agent/retrieval.py` contains retrieval-only diagnostics, dense retrieval, answer routing, and query-key inventory selection.
  - `agent/agent.py` now focuses on orchestration, tool helpers, rewrite/store entrypoints, and shared utilities.

### Expected Effect

- Reduce memory noise and retrieval clutter from low-information turns.
- Improve answer density by storing higher-level, self-contained memories.
- Keep retrieval-only evidence scoring compatible with LoCoMo `D?:?` gold ids.
- Preserve the downstream keyword, EAES attribute, and retrieval architecture while changing only the rewrite-memory creation stage.

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

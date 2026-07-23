# Version Iterations

> Evaluation recording convention: whenever a new experiment result is reported, append its scope, metrics, and diagnosis to the corresponding version entry.

## v113-20260721

### Goal

Test whether the EAES evidence selector is an answer-stage bottleneck when the reranked candidate list already contains complete gold evidence.

### Changes

- Add `--disable_evidence_selector` as an opt-in EAES answer ablation; the default pipeline remains unchanged.
- When enabled, bypass the LLM evidence selector and expose every reranked candidate directly to the final reader in rerank order.
- Add `_no_selector` to result and log filenames for the ablation.
- Reject `--disable_evidence_selector` together with `--retrieval_only`, because retrieval-only already stops before evidence selection and its ExactCover/MRR cannot be affected by this switch.

### Expected Effect

- If answer accuracy improves, the selector is discarding or over-compressing useful complementary evidence.
- If answer accuracy remains unchanged while final-context gold coverage is complete, the primary bottleneck is the final reader model or its reasoning prompt.
- Retrieval-only ExactCover and MRR are expected to remain unchanged because they measure the reranked candidate list before evidence selection.

### Evaluation Result

Pending. Compare the same category-1/category-3 question set with selector enabled versus disabled using F1 and LLM-judge accuracy.

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

### Evaluation Result

Scope: LoCoMo `conv-26`, `deepseek-chat`, first 100 questions after excluding category 5, with EAES enabled.

#### F1 by Category

| Category | n | F1 |
| --- | ---: | ---: |
| 1 | 32 | 0.4517 |
| 2 | 37 | 0.8788 |
| 3 | 13 | 0.2282 |
| 4 | 18 | 0.4757 |

#### LLM-Judge Accuracy by Category

| Category | n | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| 1 | 32 | 26 | 0.8125 |
| 2 | 37 | 32 | 0.8649 |
| 3 | 13 | 9 | 0.6923 |
| 4 | 18 | 13 | 0.7222 |
| Overall | 100 | 80 | 0.8000 |

#### Category 4 Diagnosis

- The lower category-4 judge score does not yet establish that single-hop questions are intrinsically harder in this system. Category 4 has only 18 questions and 5 errors, so each error changes its accuracy by 5.56 percentage points. Its 95% Wilson interval is approximately `[0.491, 0.875]`, overlapping category 1 (`[0.647, 0.911]`) and category 2 (`[0.720, 0.941]`). Two-sided Fisher tests also do not show a significant difference from category 1 (`p≈0.494`) or category 2 (`p≈0.268`).
- `--max_questions 100` creates a category-order sampling bias. All 18 evaluated category-4 questions occupy selected positions 83-100. They are only 18 of the 70 category-4 questions in `conv-26`, and their gold evidence is concentrated in sessions D2 (8 questions), D3 (1), and D4 (9), while categories 1 and 2 cover many more sessions. This small cluster is not representative of all single-hop questions.
- Single-hop describes evidence-chain length, not answer atomicity or retrieval difficulty. The evaluated category-4 gold answers average about 6.89 words, compared with 3.44 for category 1 and 4.05 for category 2. Several require complete lists, opinions, or causal explanations rather than a short entity.
- Multiple category-4 questions ask for different facets of the same source turn. For example, D4:3 supports the necklace's symbolism, origin country, and gift identity, while D4:13 supports the counseling population, workshop identity, and workshop content. Rewrite compression can place all facets in one broad memory; retrieval may find the correct memory but the final answer model can select the wrong facet or omit one required list element.
- Several D2 adoption questions are semantically very close: summer plans, agency population, reason for choosing the agency, and excitement about adoption. This increases same-topic distractor competition even though every question is technically single-hop.
- There is at least one source/question entity inconsistency: the question asks about “Melanie's hand-painted bowl,” but D4:5 attributes the bowl to Caroline. Entity-aware EAES indexing or reranking can therefore demote the gold memory.
- Category 4's F1 (0.4757) is slightly higher than category 1's (0.4517) even though its judge accuracy is lower. This suggests that a few semantic-coverage failures, answer-facet errors, or residual judge errors may be moving the judge metric more than a general collapse in answer overlap.

Next diagnosis should inspect the five category-4 judge failures together with their retrieved candidates and final evidence package, then rerun either all 70 category-4 questions or a category-stratified sample. This will separate rewrite loss, retrieval/reranking errors, final-answer facet selection, and judge false negatives.

### Full Evaluation Result

Scope: all 1,540 non-adversarial questions from the ten LoCoMo conversations (`conv-26`, `conv-30`, `conv-41` through `conv-44`, and `conv-47` through `conv-50`), using `deepseek-chat` result tag `rewrite_overlap_and_new_llmjudge`. The run used one worker, rewrite window size 40 with overlap 2, and EAES with LLM indexing, prefilter limit 120, and rerank limit 30.

#### F1 by Category

| Category | n | F1 |
| --- | ---: | ---: |
| 1 | 282 | 0.4929 |
| 2 | 321 | 0.7163 |
| 3 | 96 | 0.3335 |
| 4 | 841 | 0.6269 |

#### LLM-Judge Accuracy by Category

| Category | n | Correct | Wrong | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 282 | 208 | 74 | 0.7376 |
| 2 | 321 | 264 | 57 | 0.8224 |
| 3 | 96 | 57 | 39 | 0.5938 |
| 4 | 841 | 704 | 137 | 0.8371 |
| Overall | 1,540 | 1,233 | 307 | 0.8006 |

#### Full-Set Diagnosis

- The full evaluation reverses the small `conv-26` pilot's apparent category-4 weakness. Category 4 reaches the highest judge accuracy (`0.8371`) over 841 questions, confirming that its earlier `13/18` result was not representative.
- Category 3 is the clearest rate-level weakness: judge accuracy is `0.5938` and F1 is `0.3335`. Category 1 is the next-lowest by both judge accuracy (`0.7376`) and F1 (`0.4929`).
- Category 1 contributes more judge failures than category 3 in absolute terms (74 versus 39), so it remains a high-impact target even though its error rate is lower.
- Category 4 comprises 54.6% of the evaluated questions. The overall micro accuracy (`0.8006`) is therefore dominated by the strongest and largest category; per-category metrics should remain the primary basis for diagnosis.
- The next diagnostic run should evaluate retrieval only for categories 1 and 3, then join those rows with the existing judge failures. This will distinguish rewrite/index loss and retrieval/reranking misses from final-answer reasoning or residual judge errors.

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

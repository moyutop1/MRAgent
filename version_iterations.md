# Version Iterations

## v120-20260808

### Goal

Add an opt-in retrieval rollback check that searches outside the first Top20 for complementary child and parent evidence while preserving the final 16-child plus 4-parent reader budget.

### Changes

- Add `--eaes_rollback_check`, requiring semantic hierarchy and the existing `16 child + 4 parent` limits.
- After first-pass retrieval, give the question, original query plan, and only the 20 first-pass `rewrite_content` strings to an LLM to produce a complementary query plan using the existing schema.
- Exclude only the exact first-pass node IDs, then independently prefilter 27 child nodes and 3 parent nodes with the complementary plan.
- Ask one LLM to select three total supplemental nodes from the combined 30-node pool without numerically comparing child and parent scores.
- Merge first-pass and supplemental nodes into one child pool and one parent pool, then rerank them to exactly 16 children and 4 parents before evidence selection and final reading.
- Apply the same rollback flow to normal answering, selector-disabled answering, and retrieval-only evaluation; malformed rollback outputs retain the original Top20.
- Compute retrieval-only Hit/MRR from this final 16-child plus 4-parent set, so a rollback supplement counts only when the final reranker retains it in the reader Top20.
- Save both query plans, first/rollback prefilter information, supplemental Top3 IDs, and final Top20 IDs inside retrieval-only diagnostics; do not derive or save an `applied` indicator.

### Expected Effect

- Recover relevant memories missed by the first query formulation without increasing final reader context size.
- Measure whether a complementary second query improves category-4 evidence coverage and downstream answer accuracy.
- Keep rollback experiments directly comparable to the v118/v119 Top20 baseline.

### Verification

- Add tests for exact first-pass ID exclusion, the 27-child plus 3-parent second pool, total Top3 supplement selection, strict final 16+4 composition, failure fallback, and both answer/retrieval-only entry points.

## v119-20260808

### Goal

Measure answer error conditioned on complete final-context evidence coverage and align saved answer provenance with the 20 memory nodes actually exposed to the reader.

### Changes

- Save `prediction_context` as at most 20 memory-node provenance entries in reader order rather than flattening parent nodes into all linked child origins.
- Keep all origins linked to one parent grouped inside one `prediction_context` entry, so a 16-child plus 4-parent reader input remains exactly 20 entries.
- Report `P(wrong | Hit@20)` overall and by category directly from answer rows, defining Hit@20 as complete coverage of all normalized gold `evidence` origins by the first 20 `prediction_context` nodes.
- Use all Hit@20 rows as the denominator and Hit@20 rows with LLM-judge score `0` as the numerator, so a larger value indicates a stronger reader-stage bottleneck.
- Persist `evidence`, `prediction_context`, and `hit_at_20` in newly written judge rows for downstream diagnosis.

### Expected Effect

- Separate answer-stage failures from missing final-context evidence without requiring a separate retrieval-only result file.
- Prevent expanded parent provenance from making a 20-node reader context appear to contain roughly 40 independently retrieved memories.

### Verification

- Add regression coverage for full-gold rather than any-gold Hit@20, wrong answers inside the Hit@20 denominator, the hard first-20 boundary, and the 16-child plus 4-parent saved context budget.

## v118-20260807

### Goal

Align retrieval-only evaluation with the 20-memory context used by the semantic-hierarchy no-selector reader.

### Changes

- Change the default final context budget from 30 child plus 8 parent memories to 16 child plus 4 parent memories.
- Make retrieval-only retrieve the same independently ranked 16 children and 4 parents instead of evaluating children alone.
- Compute the headline metrics over 20 memory-node provenance groups and report explicit `Hit@20`, `Recall@20`, `ExactCover@20`, and `MRR@20` labels.
- Keep separate diagnostics for child `@16`, parent `@4`, and embedding prefilter `@120` so gains from broad parent provenance remain visible.
- Rank a multi-origin child or parent as one memory node for MRR; a parent's provenance group consists of the origins linked through its child IDs.
- Add semantic-hierarchy filename support to the standalone retrieval evaluator.

### Expected Effect

- Make retrieval metrics directly comparable to the evidence capacity of the final reader while retaining stage-level visibility.
- Avoid reporting the former child-only `Hit@30` after reducing and restructuring the reader context.

### Verification

- Add tests for the 16-child plus 4-parent reader budget, retrieval-only composition, parent-only keyword routing, and grouped memory-node MRR.

## v117-20260803

### Goal

Keep query-generated keywords exclusively as the semantic-hierarchy parent retrieval signal, and remove the obsolete non-EAES keyword graph path.

### Changes

- Continue generating raw `query_plan["keywords"]` values without deduplication and embedding them against each parent's own `rewrite_content`.
- Derive a child query plan with `keywords` removed before child prefilter scoring, attribute reranking, evidence selection, and final reading; also hide the parent's matched keyword from the reader payload.
- Remove the child lexical-overlap keyword score and its diagnostic field; child retrieval continues to use entities, query-attribute embeddings, lifecycle, original embeddings, and optional semantic-property bonuses.
- Retain build-side rewrite keyword files only as hints for EAES entity/attribute indexing.
- Remove the non-EAES query-key inventory/extraction flow, keyword graph nodes/links, keyword tools, tool loop, prompts, and CLI configuration.
- Require `--eaes` for `run.py` invocations and update usage documentation accordingly.

### Expected Effect

- Prevent noisy lexical keyword overlap from changing child recall or answer evidence while preserving keyword-based coarse parent context.
- Keep parent retrieval independent of child retrieval and eliminate two competing query-key mechanisms.

### Verification

- Add coverage that query keywords reach parent embedding retrieval but are absent from child scoring, reranking/selection inputs, and the final-reader query plan.
- Add coverage that changing query keywords cannot change a child candidate score or create a child `keyword` score component.

## v116-20260728

### Goal

Replace fixed-length rewrite windows with an opt-in semantic parent/child hierarchy while preserving the existing EAES child retrieval path and all behavior when the feature is disabled.

### Changes

- Add `--semantic_hierarchy` plus explicit parent/child size and retrieval parameters.
  - Parent core segments use a hard 4-10 turn range by default; only the final session segment may be shorter than the minimum.
  - Child semantic spans use a hard maximum of 8 turns, and child rewrite calls contain at most 15 segments.
  - Hierarchical rewrite, embedding, and result paths receive a separate `_hierarchy` suffix so legacy caches remain reusable.
- Plan parent and child boundaries with two independent full-session LLM calls.
  - Parent planning creates contiguous coarse topic/episode segments and returns all boundaries at once.
  - Child planning independently emits answer-bearing semantic units and omits greetings, acknowledgements, generic advice, and other low-value turns.
  - A child belongs to the parent whose core contains the child's first source `dia_id`; later child evidence may cross the parent boundary.
- Use separate parent and child rewrite prompts.
  - Each child segment produces exactly one atomic rewrite sentence, validated one-to-one and in order.
  - A unique child uses its first source `dia_id` as its ID; repeated first origins receive only a local defensive suffix such as `D1:5-1` and `D1:5-2`.
  - Each parent stores its own coarse `rewrite_content`, all linked child IDs, and all attributes already stored on those child nodes.
  - Parent rewrites omit all temporal information and focus on dialogue overview, well-supported personality characteristics, and interpersonal relationships; child rewrites retain the existing time-sensitive behavior.
  - Parent rewrite prompting treats raw events only as evidence for person-centric profile facts, prioritizing traits, preferences, sustained pursuits, pets or stable possessions, and recurring relationship patterns while prohibiting event recaps.
- Add independent parent retrieval without changing EAES child recall.
  - Parent embeddings are computed only from the parent's own `rewrite_content`.
  - Every raw `query_plan["keywords"]` item is embedded without deduplication; each parent is ranked by its maximum keyword cosine similarity.
  - The top eight parents are sent directly to the final reader even when the selector is disabled, while child candidates continue through the existing global EAES path without parent filtering.
- Keep parent internals out of the final-reader payload.
  - Each child exposes only `conversation_time` and `rewrite_content` as answer-bearing information; `memory_id` remains solely for support citation, while retrieval scores, ranks, attributes, entities, lifecycle, origins, and rationales stay internal.
  - The reader receives only each parent's ID, rewrite content, score, rank, and matched keyword; `child_attributes` and `child_ids` remain internal.
  - All retrieved parent IDs are added to final supports, and support resolution retains their linked child dialogue origins for provenance evaluation.
- Treat the child planner's `source_origins` as immutable construction provenance: child rewrite calls copy them exactly instead of choosing origins again, while cardinality, order, and `child_id` validation remain strict.

### Expected Effect

- Produce topic-coherent coarse memories and atomic answer-bearing child memories without coupling their retrieval recall.
- Give the final reader broader relationship, personality, and episode context from parents while preserving fine-grained child evidence.
- Avoid fixed-window boundary artifacts and retain the legacy rewrite/retrieval behavior unless `--semantic_hierarchy` is explicitly enabled.

### Verification

- Added semantic-hierarchy unit coverage for independent planning, first-origin ownership, duplicate-origin IDs, the 15-child batch limit, one-to-one rewrites, hidden parent attributes, and support expansion.
- Relevant rewrite, temporal-answer, selector-ablation, semantic-score, and judge-prompt regression suites pass.

> Evaluation recording convention: whenever a new experiment result is reported, append its scope, metrics, and diagnosis to the corresponding version entry.

## v115-20260726

### Goal

Make rewritten memories temporally self-contained without retaining an ambiguous per-memory structured `time` field, expose the EAES dialogue anchor explicitly as `conversation_time`, and remove duplicate final-reader backup input when the evidence selector is disabled.

### Changes

- Remove `time` from the rewrite sentence schema and rewrite prompt output format; keep the session-level `conversation_time` field.
- Require every rewrite memory to include source-supported occurrence or validity time directly in its text whenever available, with a stronger preservation requirement for `episodic` memories.
- Preserve the existing anchored rendering rules for relative weekdays, weeks, weekends, months, years, and exact-day expressions, but stop writing a separate normalized event-time field.
- Strip legacy sentence-level `time` values during rewrite normalization so regenerated memories follow the new schema even if the rewrite model emits the retired field.
- Replace EAES note/candidate `time_interval` with the scalar `conversation_time` dialogue anchor and update the final-reader temporal instruction accordingly.
- When `--disable_evidence_selector` is enabled, omit `backup_candidates` because every reranked candidate is already present in `evidence_package`; keep the existing top-12 backup path when the selector is enabled.
- Retain read compatibility for legacy cached rewrite memories that still contain `time`, while avoiding empty event-time indexing for newly generated memories.

### Expected Effect

- Help the final reader distinguish similar memories from different times using the answer-bearing rewrite text itself, especially for repeated episodic events and count questions.
- Remove duplicated top-12 candidates from no-selector reader input, reducing token use and evidence repetition while keeping the selector-enabled path unchanged.
- Regenerated memories no longer populate the structured event timeline from rewrite output; experiments that depend on non-EAES time-filtered retrieval must account for this tradeoff.

### Evaluation Result

Pending. First reuse the existing v114 memories to isolate the no-selector backup removal. Then regenerate rewrite, keyword, embedding, and EAES index files to evaluate the temporal-text change separately.

## v114-20260723

### Goal

Add query-adaptive semantic-property scoring to the existing EAES retrieval path without introducing a new memory type, cache, embedding, reranker, or filtering module.

### Changes

- Generate and permanently save `semantic_properties` with every rewrite memory in the existing rewrite LLM call.
  - Content axis: `event_action`, `state_opinion`, `personal_profile`, `relation_social` (zero to three labels).
  - Persistence axis: `transient`, `episodic`, `durable`, `unknown` (exactly one label).
  - `personal_profile` covers preferences plus interests, hobbies, occupation, education, skills, traits, residence, possessions, pets, and stable goals.
  - The abandoned labels `profile_preference` and `fact_background` are rejected by schema validation.
- Store the field on the existing `EpisodeEvent`; leave `EAESMemoryNote`, its cached retrieval embedding, and EAES index construction unchanged.
- Add `--eaes_semantic_score` as an opt-in EAES retrieval-scoring ablation.
  - When enabled, the existing query parser additionally predicts `required_semantic_properties`.
  - Query labels are whitelist-filtered, deduplicated, and never include `unknown`.
  - Each exact query-memory property match adds `0.1`, capped at three matches (`0.3` total).
  - Mismatches and missing properties receive no penalty and never filter a candidate.
- Add semantic match count, matched labels, and bonus to candidate score diagnostics.
- Add `_semantic` to result filenames and `--semantic_score` to the retrieval evaluator for locating those files.
- Keep the original query prompt, query-plan fields, and retrieval score unchanged when the flag is disabled.

### Expected Effect

- Prefer memories whose content granularity and persistence match the evidence needs of the query while retaining the current entity, attribute, keyword, lifecycle, embedding, LLM reranker, and evidence-selector pipeline.
- Improve retrieval ranking for category-1 and category-3 questions without reducing recall through hard type filtering.

### Evaluation Result

Scope: LoCoMo `conv-26`, `deepseek-chat`, first 100 questions after excluding category 5, with regenerated semantic-property memories, EAES LLM indexing, and `--eaes_semantic_score` enabled. Retrieval-only metrics cover 98 questions because two questions have no normalized gold evidence and therefore receive `None` retrieval metrics.

#### Answer Metrics and v112 Comparison

| Category | n | F1 | Delta F1 vs v112 | Judge Correct | Judge Accuracy | Delta Accuracy vs v112 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 0.3855 | -0.0662 | 21 | 0.6562 | -0.1563 |
| 2 | 37 | 0.8764 | -0.0024 | 32 | 0.8649 | 0.0000 |
| 3 | 13 | 0.2758 | +0.0476 | 9 | 0.6923 | 0.0000 |
| 4 | 18 | 0.4864 | +0.0107 | 14 | 0.7778 | +0.0556 |
| Overall | 100 | 0.5710 (weighted) | -0.0140 | 76 | 0.7600 | -0.0400 |

The end-to-end result is lower than v112. Category 1 loses five judge-correct answers (`26 -> 21`), category 4 gains one (`13 -> 14`), and categories 2 and 3 are unchanged; therefore the net four-answer decline comes entirely from category 1 after the category-4 offset. Category 3's F1 rises while its judge correctness stays at `9/13`, indicating improved lexical overlap rather than more semantically correct answers.

This is an end-to-end version comparison, not a clean estimate of the semantic score's causal effect: v114 regenerated memories with a changed rewrite prompt, whereas v112 used the previous rewrite memories. A controlled ablation must reuse the exact same v114 rewrite/keyword/embedding files and change only `--eaes_semantic_score`.

#### Retrieval-Only Metrics

| Stage | n | Hit@K | Recall@K | ExactCover@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined prefilter | 98 | 0.9490 | 0.9243 | 0.8878 | 0.6178 |
| LLM rerank output | 98 | 0.9184 | 0.8639 | 0.8061 | 0.6546 |
| Delta (rerank - prefilter) | - | -0.0306 | -0.0604 | -0.0817 | +0.0368 |

The LLM rerank/narrowing stage improves the first relevant evidence rank (`MRR +0.0368`) but reduces evidence coverage: approximately 87 of 98 questions have full gold-origin coverage before reranking, versus 79 of 98 afterward. This relevance-versus-coverage tradeoff is especially risky for category-1 multi-hop questions that need complementary memories.

#### Diagnosis: Why Top-10 Gold Retrieval Can Still Produce a Wrong Answer

- Retrieval evaluation matches normalized gold dialogue IDs against memory `origin`. A compressed memory can cite a gold origin while omitting or diluting the exact answer-bearing facet from that source turn, so origin-level ExactCover can overestimate usable textual evidence.
- Retrieval-only stops after the attribute reranker. The answer path then runs an evidence selector that keeps at most eight answer items and at most three memories per item. A gold memory in the reranked top 10 can still be discarded, grouped under the wrong candidate answer, or separated from complementary multi-hop evidence.
- The final reader treats the selector's `evidence_package` as primary and consults only the first 12 reranked candidates as backup when the package is empty or clearly insufficient. It can therefore ignore a correct backup candidate when the selected package looks plausible but supports the wrong facet.
- Even when all required facts reach the final reader, the model may fail answer-slot selection, multi-hop composition, list completeness, or planned-versus-completed distinctions. The category-1 judge drop is consistent with a composition bottleneck, but per-question stage alignment is required before assigning causality.
- One global `required_semantic_properties` set is applied independently to every memory. A heterogeneous multi-hop question may require different types for different hops; global match bonuses can promote several redundant same-type memories instead of preserving one memory for each required hop.

#### Next Controlled Tests

1. Reuse the regenerated v114 memory files and rerun the same questions without `--eaes_semantic_score`. This isolates semantic scoring from rewrite variation.
2. On the same memories and semantic setting, run the existing `--disable_evidence_selector` answer ablation. Improvement would locate the bottleneck in selector coverage; no improvement when the correct textual facts are present would point to the final reader/prompt.
3. Join answer and retrieval rows per question and audit four checkpoints: gold origin in prefilter, gold origin after rerank, answer-bearing text in the retrieved memory, and required memories in the final evidence package. Do not classify an error as a reader failure from retrieval rank alone.
4. If the selector is responsible, change the existing selector from compact relevance selection to requirement coverage: retain at least one evidence item for every query sub-requirement/hop and preserve complementary evidence before removing redundancy. As a simple diagnostic, union the selector output with a small fixed top-k reranked set rather than immediately adding a new reranker module.
5. If the correct facts already reach the final package, strengthen the existing final-answer prompt to explicitly resolve the requested answer slot, compose all required facts, and check list completeness before emitting the minimal answer. Consider a stronger reader model only after this stage-specific test.

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

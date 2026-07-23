# Typed Evidence Hierarchy: Implementation Design

## 1. Purpose

This document fixes the implementation contract for the next three MRAgent research priorities:

1. add orthogonal memory properties;
2. change the EAES evidence selector from free-form relevance selection to explicit requirement coverage;
3. add hierarchical window memories while preserving the existing session-local windowed rewrite flow.

The priorities are intentionally sequential. Each stage must be evaluated before starting the next one so that gains and regressions remain attributable.

The target research question is:

> Can a long-term conversational memory system infer the evidence structure required by a query and compose a minimal sufficient evidence set under semantic-type, persistence, temporal, and provenance constraints?

### Implementation status

- 2026-07-22: Priority 1A and 1B are implemented behind `--eaes_typed_memory`.
- Implemented scope: typed EAES index/query prompts, normalized storage, heuristic fallback, label-based soft compatibility bonuses, result diagnostics, feature flags, and unit tests.
- Evaluation status: pending retrieval-only comparison on the same questions as the current EAES baseline.

## 2. Fixed Decisions

The following decisions are treated as settled unless experiments provide contrary evidence.

1. Memory type and persistence are orthogonal properties.
2. Types and persistence provide soft bonuses; they are never hard retrieval filters.
3. `unknown` means missing or uncertain information, not mismatch.
4. A query may require several memory types at once.
5. The selector must cover explicit query requirements, not merely choose individually relevant memories.
6. No manually labelled reranker dataset is required for the first implementation.
7. The existing LLM attribute reranker remains in place until an ablation shows that it is unnecessary.
8. Hierarchical memory reuses the current session-local window and overlap process.
9. The hierarchy does not run independent raw-dialogue rewrite pipelines with window sizes such as 5, 10, and 20.
10. One existing rewrite window produces one window-level parent and multiple variable-span child memories.
11. A parent is a substantive summary memory as well as a routing node. Supported parent summaries may be answer-bearing evidence for high-level, durable, relation, preference, and event-overview questions.
12. A parent can never be the only evidence in a final evidence package. Every selected parent must be accompanied by at least one valid, linked child that supports the selected parent claim.
13. Every child must preserve at least one valid raw-dialogue origin such as `D12:3`; a child supported by several turns stores all origins in canonical comma-separated form such as `D12:3,D12:4`.
14. Every new behavior is opt-in and receives a result filename suffix so existing experiments remain reproducible.

## 3. Non-Goals

The first implementation will not:

- train a new neural reranker;
- use memory properties as negative filters;
- replace the existing entity, attribute, keyword, embedding, lifecycle, or temporal signals;
- introduce an additional multi-agent architecture;
- let any parent summary, supported or unsupported, become the only evidence in a final evidence package;
- change the LLM-judge during a method ablation;
- run several independent rewrite passes over the same raw dialogue at different fixed sizes.

## 4. Terminology

### 4.1 Construction window

The existing `rewrite_window_size` block of current dialogue turns. Previous overlap turns are context for boundary completion and do not form a second owner window.

### 4.2 Child memory

A self-contained, source-grounded memory produced from one construction window. Its real span is determined by its `origin` IDs and may cover one or several dialogue turns. Every child must have at least one valid origin. The stored format remains compatible with the current code: `D12:3` for one source turn and `D12:3,D12:4` for several source turns.

### 4.3 Parent memory

A substantive window-level summary memory covering the current construction window. It stores a coherent summary, durable information, and the event outline; points to all child memories produced from that window; and also acts as a routing node. It is not merely an index. A supported parent may be selected as answer-bearing evidence for high-level questions, but it can never appear alone: the final package must contain at least one linked supporting child with raw-dialogue origins.

### 4.4 Memory type

The semantic function of a memory, such as an action, an opinion, a preference, or a social relation.

### 4.5 Persistence

How strongly the memory is expected to remain valid beyond its source moment.

### 4.6 Evidence requirement

One information need that must or should be satisfied to answer a query, for example an answer-bearing state, a bridge event, or a temporal anchor.

### 4.7 Evidence composition

Selecting a set of complementary memories that collectively cover the query requirements, rather than selecting each memory independently by relevance.

## 5. Priority 1: Orthogonal Memory Properties

### 5.1 Controlled memory types

The initial controlled vocabulary is:

| Value | Meaning | Examples |
| --- | --- | --- |
| `event_action` | A bounded action, occurrence, attendance, plan execution, or event | attended a support group; travelled to a park |
| `state_opinion` | A reaction, feeling, opinion, intention, decision, or temporary state | felt inspired; decided to apply |
| `profile_preference` | A personal attribute, stable interest, occupation, goal, or preference | interested in counseling; enjoys pottery |
| `relation_social` | A social, family, organizational, or interpersonal relation | Melanie is Caroline's friend; member of a group |
| `fact_background` | A descriptive fact about an object, place, situation, or external background | the photo contains a dog and a mural |

A memory may have one to three types. Multi-label output is necessary because one supported sentence may describe both an event and the reaction it produced.

Recommended stored form:

```json
{
  "memory_types": ["event_action", "state_opinion"]
}
```

Unknown or malformed labels are discarded. If no valid label remains, the memory uses:

```json
{
  "memory_types": []
}
```

An empty list means unclassified, not irrelevant.

### 5.2 Persistence vocabulary

The initial persistence vocabulary is:

| Value | Meaning |
| --- | --- |
| `transient` | A momentary reaction, feeling, or short-lived state |
| `episodic` | Information bound to a particular event or occurrence |
| `durable` | A relation, preference, profile, goal, or fact expected to remain useful across time |
| `unknown` | The source does not establish persistence |

Recommended stored form:

```json
{
  "persistence": "durable"
}
```

Persistence has one primary value. If one generated memory contains clauses with incompatible persistence, the rewrite/index prompt should prefer splitting it into separate memories when source alignment permits.

### 5.3 Classification location

Priority 1 adds these fields in the EAES index stage, not the rewrite stage.

Reasons:

- the existing rewrite cache remains reusable;
- `EAES_INDEX_SYSTEM_PROMPT` already classifies entities, attributes, and lifecycle;
- the first experiment can isolate typed indexing and retrieval from rewrite changes;
- a failed experiment can be reverted without regenerating rewrite memories.

The first affected structures are:

- `prompts/prompts.py`: EAES index and query schemas;
- `agent/eaes.py`: index parsing and query-plan parsing;
- `memory/system.py`: `EAESMemoryNote` storage and serialization;
- `memory/controller.py`: typed compatibility scoring.

### 5.4 Query-side typed intent

Priority 1 extends the query plan with type and persistence preferences:

```json
{
  "required_memory_types": ["event_action", "state_opinion"],
  "preferred_persistence": ["episodic", "durable"]
}
```

The words `required` and `preferred` do not imply hard filtering. `required_memory_types` identify the semantic evidence needs; `preferred_persistence` remains a weak ranking preference unless Priority 2 explicitly marks a requirement as mandatory.

Fallback query plans use empty type requirements and `unknown` persistence so legacy behavior is preserved.

### 5.5 Three-state compatibility

Each property comparison has three semantic states:

- `match`: the candidate explicitly matches;
- `conflict`: the candidate explicitly expresses an incompatible property;
- `unknown`: the candidate is unclassified or uncertain.

`unknown` must never be treated as `conflict`.

Initial compatibility values:

| Comparison | Value |
| --- | ---: |
| exact type match | `1.0` |
| related type match | `0.5` |
| unknown type | `0.0` |
| explicit type mismatch | `0.0` |
| exact persistence match | `1.0` |
| unknown persistence | `0.5` |
| adjacent/usable persistence | `0.3` |
| explicit persistence mismatch | `0.0` |

No negative compatibility is used in the first version.

Example:

```text
Query: relation_social + durable
Memory: relation_social + unknown

type compatibility        = 1.0
persistence compatibility = 0.5
```

The memory remains a strong candidate because the primary semantic type matches.

### 5.6 Candidate score

Typed properties extend the current EAES score rather than replacing it:

```text
typed_score = current_eaes_score
            + lambda_type * type_compatibility
            + lambda_persistence * persistence_compatibility
```

Initial experimental values:

```text
lambda_type        = 0.15
lambda_persistence = 0.05
```

These values are starting points, not claims. They must be evaluated against the actual score distribution. The type signal must remain weaker than entity and attribute relevance.

The result diagnostics must expose:

```json
{
  "score_parts": {
    "type": 1.0,
    "persistence": 0.5
  },
  "matched_memory_types": ["relation_social"],
  "persistence_match_state": "unknown"
}
```

### 5.7 Backward compatibility

Legacy `EAESMemoryNote` data without typed fields must deserialize as:

```text
memory_types = []
persistence = unknown
```

Typed retrieval must be controlled by an opt-in experiment flag. When disabled, ranking must match the current implementation.

Suggested result suffix:

```text
_typed
```

### 5.8 Priority 1 acceptance criteria

Priority 1 is complete when:

1. controlled labels are generated and cached for every EAES note;
2. malformed or absent labels fall back without failing a run;
3. typed fields appear in retrieval diagnostics;
4. exact matches receive weak bonuses;
5. `unknown` never causes a negative penalty or removal;
6. disabling the experiment flag reproduces the existing score path;
7. unit tests cover multi-label types, unknown persistence, legacy notes, and score decomposition;
8. retrieval-only ExactCover/MRR is evaluated before changing the selector.

## 6. Priority 2: Requirement-Coverage Evidence Selector

### 6.1 Problem with the current selector

The current selector is instructed to consider relevance, coverage, specificity, temporal usability, and redundancy, but coverage is implicit. The model is free to decide what information the question requires, which evidence covers it, and whether the final set is sufficient in a single unverified step.

Priority 2 separates these responsibilities:

1. the query parser declares evidence requirements;
2. candidates are mapped to requirements;
3. the selector composes a set;
4. deterministic code validates coverage and memory IDs;
5. the final reader receives requirement-aligned evidence.

### 6.2 Evidence requirement schema

The query plan gains:

```json
{
  "evidence_requirements": [
    {
      "requirement_id": "R1",
      "role": "answer_bearing",
      "description": "Caroline's resulting career interest",
      "memory_types": ["state_opinion", "profile_preference"],
      "preferred_persistence": ["durable"],
      "mandatory": true,
      "weight": 1.0
    },
    {
      "requirement_id": "R2",
      "role": "bridge_context",
      "description": "The support-group experience that influenced the interest",
      "memory_types": ["event_action"],
      "preferred_persistence": ["episodic"],
      "mandatory": true,
      "weight": 0.7
    }
  ],
  "evidence_constraints": [
    {
      "type": "entity_consistency",
      "requirement_ids": ["R1", "R2"]
    },
    {
      "type": "temporal_order",
      "from": "R2",
      "to": "R1",
      "relation": "before"
    }
  ]
}
```

Initial requirement roles:

| Role | Meaning |
| --- | --- |
| `answer_bearing` | Contains the answer value or the decisive relation |
| `bridge_context` | Connects entities, events, causes, or stages in a multi-hop question |
| `temporal_anchor` | Establishes date, order, duration, or relative-time reference |
| `disambiguation` | Distinguishes similar entities, events, facets, or candidate answers |
| `background` | Useful supporting context that is not independently sufficient |

The parser should prefer one to three requirements. It must not invent an answer value.

### 6.3 Selector input

The selector receives:

```json
{
  "question": "...",
  "query_plan": {
    "evidence_requirements": [],
    "evidence_constraints": []
  },
  "candidates": [
    {
      "memory_id": "M_D1_3_1",
      "memory_types": ["event_action"],
      "persistence": "episodic",
      "event_lifecycle": "historical",
      "attribute_paths": [],
      "rewrite_content": "...",
      "rerank_rank": 4,
      "score_parts": {}
    }
  ]
}
```

### 6.4 Selector output

The selector must explicitly map requirements to evidence:

```json
{
  "coverage": [
    {
      "requirement_id": "R1",
      "covered": true,
      "memory_ids": ["M_D1_8_1"],
      "rationale": "The memory states the resulting career interest."
    },
    {
      "requirement_id": "R2",
      "covered": true,
      "memory_ids": ["M_D1_3_1", "M_D1_4_1"],
      "rationale": "These memories establish the triggering support-group experience."
    }
  ],
  "uncovered_requirement_ids": [],
  "selected_memory_ids": ["M_D1_8_1", "M_D1_3_1", "M_D1_4_1"],
  "constraint_checks": [
    {
      "type": "entity_consistency",
      "satisfied": true,
      "memory_ids": ["M_D1_8_1", "M_D1_3_1"]
    }
  ],
  "reason": "The set covers the triggering event and resulting career interest."
}
```

### 6.5 Deterministic validation

Code must validate all selector outputs:

1. remove memory IDs not present in the candidate set;
2. remove duplicate IDs while preserving order;
3. reject unknown requirement IDs;
4. recompute `uncovered_requirement_ids` from valid coverage rows;
5. enforce the evidence budget;
6. ensure every selected ID appears in at least one coverage row or has an allowed supporting role;
7. verify that every selected child has at least one syntactically valid and source-resolvable `D<number>:<turn>` origin;
8. for every selected parent, verify that the final package also contains at least one valid child linked to that parent and supporting the selected parent claim;
9. if a selected parent has valid support children but none was selected, attach the highest-relevance supporting child within the evidence budget;
10. if a selected parent has no valid supporting child, remove the parent and mark its requirement uncovered unless other evidence covers it;
11. retain the current fallback package if the selector output is unusable.

When a mandatory requirement is uncovered:

- if compatible candidates exist, add the highest-scoring compatible candidate as a recall-oriented fallback;
- if a high-scoring candidate has uncertain persistence or insufficient rewrite detail, the existing raw expansion path may inspect it;
- if no compatible candidate exists, pass the uncovered status to the final reader instead of fabricating coverage.

### 6.6 Set objective

The conceptual objective is:

```text
F(S | q) = requirement_coverage(S, q)
         + bridge_coverage(S, q)
         + temporal_constraint_support(S, q)
         - redundancy(S)
         - size_cost(S)
```

The first implementation may use the LLM selector plus deterministic validation. A later deterministic or greedy composer is allowed only if the LLM version demonstrates that explicit requirements improve selected evidence quality.

This sequence prevents premature algorithmic complexity.

### 6.7 Final reader input

The final evidence package should be grouped by requirement rather than only by possible answer item:

```json
{
  "requirements": [
    {
      "requirement_id": "R1",
      "role": "answer_bearing",
      "description": "...",
      "evidence": []
    },
    {
      "requirement_id": "R2",
      "role": "bridge_context",
      "description": "...",
      "evidence": []
    }
  ],
  "uncovered_requirement_ids": []
}
```

The final reader must answer from the evidence and must not treat requirement descriptions as facts. A valid package may contain only children, but it may never contain only parents. If it contains a parent, at least one linked supporting child with raw-dialogue origins must be present in the same package.

### 6.8 Selector-stage metrics

Retrieval-only metrics currently stop before evidence selection. Priority 2 must add separate selector diagnostics rather than reinterpret existing metrics.

Required stage metrics:

```text
Prefilter Hit / Recall / ExactCover / MRR
Rerank Hit / Recall / ExactCover / MRR
Selected Hit / Recall / ExactCover
Requirement Coverage Rate
Mandatory Requirement Coverage Rate
Selected Evidence Count
Final F1 / LLM-judge
```

MRR is optional for an unordered selected set. If the selector returns an ordered list, selected MRR may be reported but must be labelled separately.

### 6.9 Priority 2 acceptance criteria

Priority 2 is complete when:

1. every parsed requirement has a stable ID and controlled role;
2. selector output contains an explicit coverage map;
3. invalid IDs and fabricated requirement coverage cannot reach the reader;
4. uncovered mandatory requirements are visible in diagnostics;
5. selected-stage gold evidence metrics are reported independently of rerank metrics;
6. the selector-on/off comparison remains available;
7. unit tests cover complete coverage, partial coverage, invalid IDs, duplicates, unknown persistence, and fallback filling;
8. Cat 1/3 results are compared with the same query set and retrieval budget.

## 7. Priority 3: Window-Native Hierarchical Memory

### 7.1 Correct interpretation

Priority 3 preserves the existing rewrite process:

```text
session dialogue
-> current fixed-size window plus previous overlap context
-> one rewrite operation for the current window
-> one parent plus several child memories
```

It does not mean:

```text
raw dialogue -> size 5 rewrite
raw dialogue -> size 10 rewrite
raw dialogue -> size 20 rewrite
```

The memory hierarchy has multiple information spans even though the raw construction scheduler remains the existing one:

- parent span: the whole current construction window;
- child span: the source origins needed for one self-contained fact, event, or state.

### 7.2 Rewrite output schema

The rewrite output gains a window-level object while retaining `sentence` children:

```json
{
  "window_memory": {
    "local_id": "parent",
    "source_span": {
      "start": "D1:1",
      "end": "D1:11"
    },
    "routing_summary": "Caroline received support from an LGBTQ group and developed plans related to counseling.",
    "durable_claims": [
      {
        "text": "Caroline became interested in further study and counseling.",
        "supporting_child_local_ids": ["c2", "c4"]
      }
    ],
    "event_outline": [
      {
        "text": "Caroline attended an LGBTQ support group.",
        "supporting_child_local_ids": ["c1"]
      },
      {
        "text": "Caroline shared a photo with Melanie.",
        "supporting_child_local_ids": ["c3"]
      }
    ]
  },
  "sentence": [
    {
      "local_id": "c1",
      "text": "Caroline attended an LGBTQ support group on 7 May 2023.",
      "origin": "D1:3"
    },
    {
      "local_id": "c2",
      "text": "Caroline heard an inspiring transgender story at the support group.",
      "origin": "D1:4"
    },
    {
      "local_id": "c3",
      "text": "Caroline shared a photo showing a dog beside a wall painted with a woman.",
      "origin": "D1:8"
    }
  ]
}
```

Local IDs are normalized to deterministic global IDs after validation.

### 7.3 Parent coverage inventory

A parent summary is lossy and cannot be the only routing representation. After child indexing, code derives a coverage inventory from validated child memories:

```json
{
  "coverage_inventory": [
    {
      "description": "LGBTQ support group attendance",
      "memory_types": ["event_action"],
      "child_ids": ["M_D1_3_1"]
    },
    {
      "description": "inspiring transgender story",
      "memory_types": ["event_action", "state_opinion"],
      "child_ids": ["M_D1_4_1"]
    },
    {
      "description": "dog and woman mural photo",
      "memory_types": ["event_action", "fact_background"],
      "child_ids": ["M_D1_8_1"]
    }
  ]
}
```

The inventory must be derived from children so every valid child has at least one route-visible entry. It is not allowed to depend only on what the parent summary happened to mention.

### 7.4 Parent-child provenance rules

1. Every child has exactly one primary parent.
2. Every child has a non-empty canonical `origin` string containing one or more raw-dialogue IDs matching `D<number>:<turn>`.
3. Multiple child origins are comma-separated without spaces, for example `D12:3,D12:4`; their order follows the source dialogue and duplicates are removed.
4. Every child origin must resolve to an actual source turn available to the construction window, including allowed previous-overlap context.
5. The primary parent is the construction window whose `current_turns` produced the child.
6. Previous overlap is context, not a second primary owner.
7. A cross-boundary child may cite both context and current origins.
8. A cross-boundary child is marked `cross_window = true` and may record a context parent, but is stored once.
9. Every parent claim must cite one or more supporting child IDs.
10. Every child cited by a parent claim must belong to that parent and have valid raw-dialogue origins.
11. Parent claims unsupported by valid children are removed during validation.
12. A final evidence package containing a parent must also contain at least one linked child supporting that parent claim.
13. Previous rewrite memories shown for deduplication must not silently become new parent claims without current source support.

Example ownership:

```text
P1 current origins: D1:1-D1:40
P2 context origins: D1:39-D1:40
P2 current origins: D1:41-D1:80

Child origin: D1:40,D1:41
primary_parent = P2
context_parent = P1
cross_window = true
```

### 7.5 Storage structures

Add a parent structure separate from child `EAESMemoryNote` records:

```text
WindowMemoryNode
  parent_id
  session_id
  source_start
  source_end
  routing_summary
  durable_claims
  event_outline
  child_ids
  coverage_inventory
  embedding
```

Recommended indexes:

```text
window_memories[parent_id] -> WindowMemoryNode
children_by_parent[parent_id] -> set(child_id)
parent_by_child[child_id] -> parent_id
```

Child memories remain the existing EAES notes with the Priority 1 typed fields.

### 7.6 Hierarchical retrieval

The query plan chooses a preferred route:

```json
{
  "preferred_granularity": "coarse | fine | mixed",
  "routing_strategy": "parent_first | child_first | hybrid"
}
```

Initial policy:

| Query need | Preferred route |
| --- | --- |
| durable profile, preference, or relation | `parent_first` |
| exact event, date, object, or momentary reaction | `child_first` or immediate drill-down |
| multi-hop, causal, comparison, or cross-event query | `mixed` |
| ambiguous routing intent | `hybrid` |

Safe candidate generation is dual-route:

```text
parent_candidates = top_parent_routes
hierarchical_child_candidates = children(top_parent_routes)
global_candidates = global_child_retrieval(top_k)
candidate_union = parent_candidates
                union hierarchical_child_candidates
                union global_candidates
```

Parent summaries remain candidates instead of being discarded after routing. The global child route is a safety mechanism against parent-summary routing false negatives.

### 7.7 Parent usage policy

From the first hierarchical version:

- parent memories contain a real summary description and are both retrievable memories and routing nodes;
- supported parent summaries may be answer-bearing evidence for durable profile, preference, relation, cross-event trend, and event-overview questions, but never the only evidence;
- a selected parent is passed to the coverage selector together with its supporting child IDs;
- for a coarse question, the final package may use the parent as `answer_bearing` evidence and must include at least one linked child as `supporting_provenance`;
- for a fine-grained question, the parent normally acts as `bridge_context` or `routing_context`, while children provide `answer_bearing` evidence;
- parent-only evidence packages are forbidden for every question type, not only fine-grained questions;
- an unsupported generated parent claim cannot be used as answer evidence.

Example final package for a high-level question:

```json
{
  "requirement_id": "R1",
  "role": "answer_bearing",
  "parent_evidence": {
    "parent_id": "P_D1_W1",
    "summary": "Caroline received support from an LGBTQ group and developed an interest in counseling.",
    "supporting_child_ids": ["M_D1_3_1", "M_D1_4_1", "M_D1_8_1"]
  },
  "child_evidence": [
    {"memory_id": "M_D1_3_1", "role": "supporting_provenance"},
    {"memory_id": "M_D1_8_1", "role": "supporting_provenance"}
  ]
}
```

### 7.8 Routing metrics

Priority 3 adds:

```text
Parent Routing Recall
Gold Child Reachability
Hierarchy-only Child Recall
Global-fallback Child Recall
Union Child Recall
Average Parents Selected
Average Children Expanded
Parent Evidence Selection Rate
Supported Parent Claim Rate
Routing Input Tokens
Total Retrieval Tokens
Latency and LLM Calls
```

Definitions:

```text
Parent Routing Recall
= questions where a selected parent contains at least one gold-support child
 / questions with a mapped gold-support child

Gold Child Reachability
= gold-support children under selected parents
 / all mapped gold-support children
```

Parent routing must be evaluated before connecting the hierarchy to final answering.

### 7.9 Priority 3 acceptance criteria

Priority 3 is complete when:

1. existing current-window and overlap semantics remain unchanged;
2. one rewrite window produces one validated parent and zero or more validated children;
3. every child retains at least one valid raw-dialogue origin, and multi-turn children preserve all supporting origins in canonical form such as `D12:3,D12:4`;
4. parent claims cannot survive without supporting children;
5. every valid child is represented in the parent coverage inventory;
6. overlap does not create duplicate primary ownership;
7. hierarchy-only and global-fallback candidates are separately diagnosable;
8. flat retrieval remains available as an ablation;
9. routing recall and token cost are reported before final-answer claims;
10. supported parent summaries can enter the evidence package for high-level questions only together with at least one linked supporting child;
11. no question type can produce a parent-only evidence package;
12. fine-grained questions retain child answer evidence and do not rely only on a parent summary;
13. unit tests cover single-window sessions, multi-window sessions, one-origin and multi-origin children, invalid or empty origins, cross-boundary children, supported and unsupported parent claims, parent evidence selection, parent-only rejection/repair, and missed-parent global fallback.

## 8. Implementation Order

### Stage 0: Finish the selector-off diagnosis

Before adding new method behavior:

- compare selector-on and selector-off on the same Cat 1/3 questions;
- record F1 and LLM-judge;
- do not interpret retrieval-only ExactCover/MRR as selector metrics;
- identify whether the main bottleneck is retrieval, selection, or final reading.

### Stage 1A: Typed storage only

- add memory type and persistence fields;
- index and serialize them;
- expose them in diagnostics;
- do not change ranking.

Purpose: validate label coverage and consistency without accuracy confounds.

### Stage 1B: Typed retrieval bonus

- extend the query plan;
- add weak compatibility bonuses;
- compare retrieval-only metrics and type distributions.

Purpose: determine whether typed signals improve retrieval before changing selection.

### Stage 2A: Explicit requirements only

- add evidence requirements to the query plan;
- log requirements;
- keep the existing selector behavior.

Purpose: inspect requirement quality without changing selected evidence.

### Stage 2B: Coverage selector

- add requirement-to-memory mapping;
- validate output deterministically;
- report selected-stage metrics;
- preserve selector-off ablation.

### Stage 3A: Parent construction and routing test

- preserve child groups per existing rewrite window;
- generate substantive parent summary memories;
- derive coverage inventories;
- validate that every parent claim has child support;
- evaluate parent routing and parent-summary quality before changing final answer generation.

### Stage 3B: Safe hierarchical retrieval

- retain selected parent summaries as candidates;
- expand their children;
- union with global child fallback;
- feed parents and children into the typed rerank and coverage selector;
- allow supported parents to fill high-level evidence requirements.

### Stage 3C: Parent-child answer policy ablation

- compare parent plus child evidence against child-only evidence;
- report results separately for coarse/durable and fine-grained questions;
- verify that any parent gain is not caused by unsupported summary hallucination.

## 9. Required Experiment Matrix

| Variant | Typed fields | Typed score | Explicit requirements | Coverage selector | Parent routing | Parent evidence | Global child fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current EAES | no | no | no | no | no | no | n/a |
| Typed storage | yes | no | no | no | no | no | n/a |
| Typed retrieval | yes | yes | no | no | no | no | n/a |
| Requirement logging | yes | yes | yes | no | no | no | n/a |
| Coverage composition | yes | yes | yes | yes | no | no | n/a |
| Hierarchy child-only ablation | yes | yes | yes | yes | yes | no | yes |
| Safe full method | yes | yes | yes | yes | yes | yes | yes |

Every comparison must use:

- the same questions;
- the same backbone and judge;
- the same rerank candidate limit where applicable;
- unchanged rewrite data unless the stage explicitly changes rewrite;
- unique result tags;
- per-category as well as overall metrics.

## 10. Suggested Feature Flags and Suffixes

Exact names may change during implementation, but each behavior must be independently switchable.

```text
--eaes_typed_memory
--eaes_requirement_selector
--hierarchical_window_memory
--hierarchical_global_fallback
```

Suggested suffixes:

```text
_typed
_reqsel
_hier
_globalfb
```

Do not create one flag that enables all three priorities. Independent flags are required for clean ablations.

## 11. Test Plan

### Priority 1 tests

- one memory type;
- multiple memory types;
- malformed label removal;
- missing typed fields in legacy notes;
- exact persistence match;
- unknown persistence remains neutral;
- typed mode disabled reproduces the legacy score;
- type score cannot remove candidates.

### Priority 2 tests

- all mandatory requirements covered;
- one mandatory requirement uncovered;
- optional requirement uncovered;
- fabricated memory ID removed;
- fabricated requirement ID removed;
- duplicate selected IDs removed;
- unknown persistence evidence may cover an exact type requirement;
- compatible fallback fills an uncovered requirement;
- evidence budget enforcement;
- selector-off behavior remains unchanged.

### Priority 3 tests

- session shorter than one construction window;
- session spanning multiple construction windows;
- previous overlap used only as context;
- cross-window child receives the current parent as primary owner;
- one-origin child stores `D12:3`;
- multi-origin child stores canonical `D12:3,D12:4`;
- empty, malformed, duplicate, or source-unresolvable child origins are rejected or normalized as appropriate;
- parent claim with missing child support is removed;
- parent-only selector output is repaired by attaching a linked supporting child or rejected when no valid child exists;
- every child appears in the coverage inventory;
- parent routing miss recovered by global child fallback;
- flat retrieval remains available;
- rewrite IDs remain deterministic.

## 12. Failure Interpretation

### Typed retrieval does not improve retrieval metrics

Possible conclusions:

- types are redundant with existing attribute paths;
- query type prediction is noisy;
- score weights are poorly calibrated;
- Cat 1/3 failures occur after retrieval.

Do not proceed directly to a trained reranker. First inspect score distributions and gold-memory type labels.

### Coverage selector improves selected ExactCover but not final answers

Likely bottleneck:

- final reader reasoning;
- final prompt organization;
- context overload;
- answer/judge mismatch.

### Coverage selector reduces selected ExactCover

Likely bottleneck:

- requirements are incomplete;
- mandatory/optional distinction is too strict;
- selector budget is too small;
- validation fallback is insufficient.

### Parent routing recall is low

Do not connect hierarchy-only retrieval to the reader. Compare:

- summary-only routing;
- summary plus coverage inventory;
- hierarchy plus global fallback.

### Hierarchy improves cost but not accuracy

This can still be useful only if retrieval recall is preserved and efficiency gains are substantial. It is not evidence that the full typed composition method improves reasoning.

## 13. Novelty Positioning

The individual components are not sufficient novelty claims:

- memory labels alone are metadata engineering;
- hierarchical memory alone overlaps with existing multi-granularity systems;
- generic set selection alone overlaps with prior RAG work;
- an LLM selector prompt alone may be judged as prompt engineering.

The intended combined contribution is:

1. a dual-axis semantic-type and persistence representation for conversational memories;
2. query-generated evidence requirements with explicit answer, bridge, temporal, and disambiguation roles;
3. evidence composition under requirement coverage, temporal constraints, and provenance;
4. a single-pass, provenance-aligned parent-child memory hierarchy;
5. evidence-preserving parent routing through child-derived coverage inventories and a safe global fallback;
6. stage-wise diagnostics that separate available, ranked, selected, and used evidence.

The paper claim should focus on the formulation and verified mechanism:

> Long-term conversational QA is a constrained evidence-composition problem, not only a similarity-ranking problem.

## 14. Non-Blocking Experimental Choices

The following values are deliberately left as tunable experimental parameters and do not require architectural clarification:

- `lambda_type` and `lambda_persistence`;
- maximum query requirements;
- evidence budget;
- number of parents selected;
- global child fallback size;
- threshold for raw expansion;
- whether parent summaries use the answering model or the rewrite model;
- whether deterministic greedy composition is added after the LLM coverage selector.

These values must be chosen on a development subset and reported. They must not be silently tuned on the full test set.

## 15. Immediate Next Action

Run the typed-memory retrieval-only comparison on the same Cat 1/3 questions as the current EAES baseline. Inspect field distributions and gold-memory score parts before changing the selector. Keep the selector requirement-coverage work disabled until Priority 1 retrieval effects are recorded.

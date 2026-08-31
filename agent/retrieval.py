import json
import logging
import re
from typing import Any, Dict, List

import numpy as np

from common import config
from common.utils import topk_answers_by_similarity

logger = logging.getLogger(__name__)


def compact_eaes_retrieval(retrieval):
    """Serialize one non-duplicated retrieval-only diagnostic payload."""
    child_fields = (
        "memory_id", "event_id", "parent_id", "origin", "tag",
        "rewrite_content", "max_phrase_similarity", "rrf_score",
        "base_score", "parent_boost", "candidate_score",
        "candidate_sources", "prefilter_rank", "rerank_rank",
        "rerank_source", "matched_tag", "phrase_matches",
    )
    parent_fields = (
        "parent_id", "rewrite_content", "raw_similarity",
        "parent_probability", "child_support", "posterior_score",
        "selected", "rank",
    )
    routing = {
        key: value for key, value in (retrieval.get("routing") or {}).items()
        if key != "parent_candidates"
    }
    query_plan = dict(retrieval.get("query_plan") or {})
    query_plan.pop("breadth_value", None)
    query_plan.pop("detail_value", None)
    rollback = retrieval.get("rollback_check") or {"enabled": False}
    compact_rollback = {"enabled": bool(rollback.get("enabled"))}
    for key in (
            "reader_gate", "failure_reason", "first_pass",
            "selected_supplements", "final"):
        if key in rollback:
            compact_rollback[key] = rollback[key]
    return {
        "mode": "eaes",
        "query_plan": query_plan,
        "routing": routing,
        "phrase_retrieval": {
            "phrases": (retrieval.get("phrase_retrieval") or {}).get(
                "phrases", []
            ),
        },
        "parent_local_retrieval": retrieval.get("parent_local_retrieval"),
        "parent_candidates": [
            {key: item.get(key) for key in parent_fields if key in item}
            for item in retrieval.get("parent_candidate_scores") or []
        ],
        "child_candidates": [
            {key: item.get(key) for key in child_fields if key in item}
            for item in retrieval.get("prefilter_candidates") or []
        ],
        "final_child_ids": retrieval.get("final_child_ids") or [],
        "counts": retrieval.get("counts") or {},
        "rollback_check": compact_rollback,
    }


class RetrievalMixin:
    @staticmethod
    def _unique_keep_order(items):
        seen = set()
        out = []
        for item in items or []:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _origins_for_event_ids(self, event_ids):
        origins = []
        for eid in event_ids or []:
            if eid in self.memory.episode_events:
                origins.append(self.memory.episode_events[eid].origin)
            elif re.match(r"^D\d+:\d+$", str(eid)) and f"{eid}-1" in self.memory.episode_events:
                origins.append(self.memory.episode_events[f"{eid}-1"].origin)
            else:
                origins.append(eid)
        return self._unique_keep_order(origins)

    @staticmethod
    def _event_ids_from_texts(texts):
        event_ids = []
        for text in texts or []:
            m = re.search(r"\b(D\d+:\d+(?:-\d+)?)\s*:", str(text))
            if m:
                event_ids.append(m.group(1))
        return event_ids

    @staticmethod
    def _normalize_evidence_ids(items):
        out = []
        for item in items or []:
            for match in re.findall(r"D\d+:\d+(?:-\d+)?", str(item)):
                out.append(match.split("-", 1)[0])
        return RetrievalMixin._unique_keep_order(out)

    def _episode_ids_for_origin(self, origin):
        origin = str(origin or "")
        if not origin:
            return []
        event_ids = []
        for event_id, event in self.memory.episode_events.items():
            event_origin = getattr(event, "origin", None)
            event_origin_ids = set(self._origin_ids(event_origin))
            if origin in event_origin_ids or event_origin == origin or event_id == origin or event_id.startswith(f"{origin}-"):
                event_ids.append(event_id)
        return self._unique_keep_order(event_ids)

    def diagnose_eaes_gold_memories(self, gold_evidence, retrieval, question_emb=None, window=2):
        if not isinstance(retrieval, dict) or retrieval.get("mode") != "eaes":
            return None
        phrase_retrieval = retrieval.get("phrase_retrieval") or {}
        prefilter_candidates = self._as_list(retrieval.get("prefilter_candidates"))
        ranked = sorted(
            (
                item for item in prefilter_candidates
                if isinstance(item, dict) and item.get("memory_id")
            ),
            key=lambda item: (
                int(item.get("prefilter_rank") or 10**9),
                str(item.get("memory_id") or ""),
            ),
        )
        by_memory_id = {item.get("memory_id"): item for item in ranked if item.get("memory_id")}
        by_event_id = {item.get("event_id"): item for item in ranked if item.get("event_id")}
        phrase_hits = {}
        for phrase in self._as_list(phrase_retrieval.get("phrases")):
            phrase_index = phrase.get("phrase_index") if isinstance(phrase, dict) else None
            for item in self._as_list(
                    phrase.get("candidates") if isinstance(phrase, dict) else []):
                if not isinstance(item, dict) or not item.get("memory_id"):
                    continue
                phrase_hits.setdefault(item["memory_id"], []).append({
                    "phrase_index": phrase_index,
                    "rank": item.get("rank"),
                    "similarity": item.get("similarity"),
                })
        prefilter_memory_ids = {
            item.get("memory_id") for item in prefilter_candidates if isinstance(item, dict)
        }
        global_memory_ids = {
            item.get("memory_id")
            for item in self._as_list(retrieval.get("global_candidates"))
            if isinstance(item, dict)
        }
        local_memory_ids = {
            item.get("memory_id")
            for item in self._as_list(retrieval.get("local_candidates"))
            if isinstance(item, dict)
        }
        merged_memory_ids = set(
            (retrieval.get("merge_retrieval") or {}).get(
                "global_plus_local_ids", []
            )
        )
        pool_dropped_ids = set(
            (retrieval.get("merge_retrieval") or {}).get(
                "dropped_by_pool_limit_ids", []
            )
        )
        reranked_candidates = self._as_list(retrieval.get("candidates"))
        reranked_by_memory_id = {
            item.get("memory_id"): item for item in reranked_candidates
            if isinstance(item, dict) and item.get("memory_id")
        }
        retrieved_memory_ids = {
            item.get("memory_id") for item in reranked_candidates if isinstance(item, dict)
        }
        parent_ids = set(self._as_list(retrieval.get("parent_ids")))
        parent_origin_groups = self._as_list(
            retrieval.get("parent_origin_groups")
        )

        diagnostics = {
            "prefilter_limit": getattr(config, "EAES_PHRASE_UNION_LIMIT", 60),
            "rerank_limit": getattr(config, "EAES_PHRASE_RERANK_LIMIT", 15),
            "total_scored_memories": len(self.memory.eaes_notes),
            "gold_origins": [],
        }
        for origin in self._normalize_evidence_ids(gold_evidence):
            event_ids = self._episode_ids_for_origin(origin)
            memory_ids = [
                self.memory.eaes_event_to_memory.get(event_id)
                for event_id in event_ids
                if self.memory.eaes_event_to_memory.get(event_id)
            ]
            memory_entries = []
            if not event_ids:
                memory_entries.append({
                    "event_id": None,
                    "memory_id": None,
                    "parent_id": None,
                    "indexed": False,
                    "in_global_child": False,
                    "in_parent_local": False,
                    "in_prefilter": False,
                    "in_final_child": False,
                    "prefilter_rank": None,
                    "rerank_rank": None,
                    "candidate_score": None,
                    "phrase_hits": [],
                    "drop_reason": "gold_origin_not_in_episode_events",
                })
            for event_id in event_ids:
                memory_id = self.memory.eaes_event_to_memory.get(event_id)
                note = self.memory.get_eaes_note(memory_id) if memory_id else None
                scored = by_memory_id.get(memory_id) or by_event_id.get(event_id)
                rank = scored.get("prefilter_rank") if scored else None
                reranked = reranked_by_memory_id.get(memory_id)
                rerank_rank = reranked.get("rerank_rank") if reranked else None
                covered_by_selected_parent = bool(
                    note is not None and note.parent_id in parent_ids
                )
                if note is None:
                    drop_reason = "not_built_in_eaes_memory"
                elif memory_id in reranked_by_memory_id:
                    drop_reason = "inside_llm_top15"
                elif covered_by_selected_parent:
                    drop_reason = "inside_dynamic_parent"
                elif memory_id in pool_dropped_ids:
                    drop_reason = "dropped_by_merged_pool_limit"
                elif memory_id in prefilter_memory_ids:
                    drop_reason = "dropped_by_llm_reranker"
                elif memory_id in local_memory_ids:
                    drop_reason = "added_by_parent_local"
                else:
                    drop_reason = "not_in_any_dynamic_phrase_topk"
                memory_entries.append({
                    "event_id": event_id,
                    "memory_id": memory_id,
                    "parent_id": note.parent_id if note is not None else None,
                    "indexed": note is not None,
                    "in_global_child": memory_id in global_memory_ids,
                    "in_parent_local": memory_id in local_memory_ids,
                    "in_merged_pool": memory_id in merged_memory_ids,
                    "in_prefilter": memory_id in prefilter_memory_ids,
                    "in_final_child": memory_id in retrieved_memory_ids,
                    "prefilter_rank": rank,
                    "rerank_rank": rerank_rank,
                    "candidate_score": scored.get("candidate_score") if scored else None,
                    "phrase_hits": phrase_hits.get(memory_id, []),
                    "drop_reason": drop_reason,
                })
            covered_by_child = any(
                entry["in_final_child"] for entry in memory_entries
            )
            matched_parent_ids = [
                parent_id
                for parent_id, parent_origins in zip(
                    self._as_list(retrieval.get("parent_ids")), parent_origin_groups
                )
                if origin in self._normalize_evidence_ids(parent_origins)
            ]
            covered_by_parent = bool(matched_parent_ids)
            covered_by_retrieval = covered_by_child or covered_by_parent
            rank_values = [
                entry["prefilter_rank"] for entry in memory_entries
                if entry["prefilter_rank"] is not None
            ]
            rerank_values = [
                entry["rerank_rank"] for entry in memory_entries
                if entry["rerank_rank"] is not None
            ]
            if not event_ids:
                origin_drop_reason = "gold_origin_not_in_episode_events"
            elif not memory_ids:
                origin_drop_reason = "no_gold_memory_built_for_origin"
            elif covered_by_child:
                origin_drop_reason = "inside_llm_top15"
            elif covered_by_parent:
                origin_drop_reason = "inside_dynamic_parent"
            elif rank_values:
                origin_drop_reason = "dropped_by_llm_reranker"
            elif any(memory_id in pool_dropped_ids for memory_id in memory_ids):
                origin_drop_reason = "dropped_by_merged_pool_limit"
            elif any(memory_id in local_memory_ids for memory_id in memory_ids):
                origin_drop_reason = "added_by_parent_local"
            else:
                origin_drop_reason = "not_in_any_dynamic_phrase_topk"

            diagnostics["gold_origins"].append({
                "origin": origin,
                "event_ids": event_ids,
                "memory_ids": memory_ids,
                "covered_by_retrieval": covered_by_retrieval,
                "covered_by_child": covered_by_child,
                "covered_by_parent": covered_by_parent,
                "matched_parent_ids": matched_parent_ids,
                "drop_reason": origin_drop_reason,
                "best_prefilter_rank": min(rank_values, default=None),
                "best_rerank_rank": min(rerank_values, default=None),
                "memories": memory_entries,
            })
        return diagnostics

    def _dense_episode_retrieval(self, question_emb, k=None):
        if question_emb is None:
            return [], [], [], []
        ids, embs, texts = [], [], []
        for event_id, event in self.memory.episode_events.items():
            if event.embedding is None:
                continue
            ids.append(event_id)
            embs.append(event.embedding)
            texts.append(f"{event_id}:{event.text}")
        if not embs:
            return [], [], [], []
        emb_matrix = np.vstack(embs)
        top_ids, top_scores, _, top_texts = topk_answers_by_similarity(
            question_emb, emb_matrix, ids, k=k or config.DENSE_RETRIEVAL_K, answer_texts=texts)
        return top_ids, top_scores, top_texts or [], self._origins_for_event_ids(top_ids)

    def retrieve_question_evidence(self, question: str, category=0, question_emb=None,
                                   override_question_time=None, lm_current_date=None) -> Dict[str, Any]:
        """Return retrieval candidates without generating a final answer."""
        if config.EAES_MODE:
            first_pass = self._retrieve_eaes_first_pass(question, question_emb)
            query_plan = first_pass["query_plan"]
            child_query_plan = self._eaes_child_query_plan(query_plan)
            global_candidates = first_pass["global_children"]
            prefilter_candidates = first_pass["prefilter_children"]
            candidates = first_pass["final_children"]
            parent_candidates = first_pass["selected_parents"]
            rollback_metadata = {"enabled": False}
            if getattr(config, "EAES_ROLLBACK_CHECK", False):
                first_parent_candidates = list(parent_candidates)
                try:
                    _, _, internal_raw_answer = self._read_eaes_candidates(
                        question,
                        child_query_plan,
                        candidates,
                        parent_candidates,
                        category,
                        lm_current_date,
                    )
                    returned_no_information = (
                        self._eaes_is_no_information_answer(
                            internal_raw_answer
                        )
                    )
                    if returned_no_information:
                        candidates, parent_candidates, rollback_metadata = (
                            self.apply_eaes_rollback_check(
                                question,
                                query_plan,
                                candidates,
                                parent_candidates,
                                question_emb,
                            )
                        )
                    else:
                        rollback_metadata = {
                            "enabled": True,
                            "first_query_plan": query_plan,
                            "first_pass": {
                                "child_ids": [
                                    candidate.get("memory_id")
                                    for candidate in candidates
                                ],
                                "parent_ids": [
                                    candidate.get("parent_id")
                                    for candidate in parent_candidates
                                ],
                            },
                        }
                    rollback_metadata["reader_gate"] = {
                        "returned_no_information_available": (
                            returned_no_information
                        ),
                    }
                    rollback_metadata["first_prefilter"] = {
                        "child_ids": [
                            candidate.get("memory_id")
                            for candidate in prefilter_candidates
                        ],
                        "parent_ids": [
                            candidate.get("parent_id")
                            for candidate in first_parent_candidates
                        ],
                    }
                except Exception:
                    logger.warning(
                        "EAES rollback reader gate/check failed; retaining "
                        "the first-pass reader set.",
                        exc_info=True,
                    )
                    rollback_metadata = {
                        "enabled": True,
                        "first_query_plan": query_plan,
                        "first_prefilter": {
                            "child_ids": [
                                candidate.get("memory_id")
                                for candidate in prefilter_candidates
                            ],
                            "parent_ids": [
                                candidate.get("parent_id")
                                for candidate in first_parent_candidates
                            ],
                        },
                        "failure_reason": "reader_gate_or_rollback_exception",
                    }
            def child_origin_groups(items):
                return [
                    [candidate.get("origin")]
                    if candidate.get("origin") else
                    self._origins_for_event_ids([candidate.get("event_id")])
                    for candidate in items or []
                ]

            def origins_from_groups(groups):
                return self._unique_keep_order([
                    origin for group in groups for origin in group if origin
                ])

            event_ids = self._unique_keep_order([
                candidate.get("event_id") for candidate in candidates
            ])
            child_groups = child_origin_groups(candidates)
            child_origins = origins_from_groups(child_groups)
            parent_ids = self._unique_keep_order([
                parent.get("parent_id") for parent in parent_candidates
            ])
            parent_origin_groups = [
                self.memory.get_eaes_support_origin([parent_id])
                for parent_id in parent_ids
            ]
            parent_origins = self._unique_keep_order([
                origin
                for group in parent_origin_groups
                for origin in group
            ])
            final_groups = child_groups + parent_origin_groups
            final_origins = origins_from_groups(final_groups)

            global_groups = child_origin_groups(global_candidates)
            global_plus_local_ids = first_pass["merge_retrieval"].get(
                "global_plus_local_ids", []
            )
            global_plus_local_candidates = []
            for memory_id in global_plus_local_ids:
                note = self.memory.get_eaes_note(memory_id)
                if note is not None:
                    global_plus_local_candidates.append(note.to_dict())
            global_plus_local_groups = child_origin_groups(
                global_plus_local_candidates
            )
            prefilter_groups = child_origin_groups(prefilter_candidates)
            stage_origin_groups = {
                "global_child": global_groups,
                "global_plus_local": global_plus_local_groups,
                "prefilter_child": prefilter_groups,
                "final_child": child_groups,
                "selected_parent": parent_origin_groups,
                "final_combined": final_groups,
            }
            stage_origins = {
                key: origins_from_groups(groups)
                for key, groups in stage_origin_groups.items()
            }
            first_pass["counts"]["final_child_k"] = len(candidates)
            first_pass["counts"]["final_parent_k"] = len(parent_candidates)
            first_pass["counts"]["final_total_k"] = (
                len(candidates) + len(parent_candidates)
            )
            return {
                "mode": "eaes",
                "query_plan": query_plan,
                "routing": first_pass["routing"],
                "counts": first_pass["counts"],
                "phrase_retrieval": first_pass["phrase_retrieval"],
                "parent_local_retrieval": first_pass["local_retrieval"],
                "merge_retrieval": first_pass["merge_retrieval"],
                "parent_candidate_scores": first_pass["routing"].get(
                    "parent_candidates", []
                ),
                "global_candidates": global_candidates,
                "local_candidates": first_pass["local_children"],
                "prefilter_candidates": prefilter_candidates,
                "candidates": candidates,
                "parent_candidates": parent_candidates,
                "final_child_ids": [
                    candidate.get("memory_id") for candidate in candidates
                ],
                "stage_origins": stage_origins,
                "stage_origin_groups": stage_origin_groups,
                "retrieved_event_ids": event_ids,
                "retrieved_memory_ids": self._unique_keep_order(
                    [candidate.get("memory_id") for candidate in candidates]
                    + parent_ids
                ),
                "retrieved_origins": final_origins,
                "retrieved_origin_groups": final_groups,
                "retrieval_k": len(candidates) + len(parent_candidates),
                "child_k": len(candidates),
                "parent_k": len(parent_candidates),
                "child_origins": child_origins,
                "child_origin_groups": child_groups,
                "parent_ids": parent_ids,
                "parent_origins": parent_origins,
                "parent_origin_groups": parent_origin_groups,
                "prefilter_origins": stage_origins["prefilter_child"],
                "prefilter_origin_groups": prefilter_groups,
                "prefilter_k": len(prefilter_candidates),
                "rollback_check": rollback_metadata,
            }

        raise RuntimeError(
            "Non-EAES keyword retrieval has been removed; run with --eaes."
        )


    def answer_question(self, question: str, category=0, question_emb=None, override_question_time=None, lm_current_date=None) -> Dict[str, Any]:
        if not config.EAES_MODE:
            raise RuntimeError("Non-EAES keyword retrieval has been removed; run with --eaes.")
        return self.answer_question_eaes(
            question, category, question_emb, lm_current_date
        )


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
        "rrf_score_ratio", "rrf_rank", "question_relevance",
        "local_relevance_score", "weighted_rrf_sum",
        "best_view_contribution", "normalized_consensus",
        "normalized_best_view", "fused_score", "fused_score_ratio",
        "fused_rank", "normalized_mass", "cumulative_mass",
        "mass_target", "inside_fused_safety_cap",
        "inside_adaptive_prefix", "adaptive_k",
        "base_score", "candidate_score", "score", "score_parts",
        "candidate_sources", "prefilter_rank", "rerank_rank",
        "rerank_source", "matched_tag", "phrase_matches",
        "matched_query_phase",
    )
    parent_fields = (
        "parent_id", "rewrite_content", "raw_similarity",
        "rrf_score", "rrf_score_ratio", "rrf_rank",
        "question_relevance", "inside_adaptive_prefix", "adaptive_k",
        "local_relevance_score", "weighted_rrf_sum",
        "best_view_contribution", "normalized_consensus",
        "normalized_best_view", "fused_score", "fused_score_ratio",
        "fused_rank", "normalized_mass", "cumulative_mass",
        "mass_target", "inside_fused_safety_cap",
        "parent_probability", "child_support", "posterior_score",
        "selected", "rank", "score", "matched_query_phase",
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
            "terminal_reason", "rollback_count", "initial_query_phase",
            "first_pass", "initial_retrieval_pool", "rounds",
            "selected_supplements", "last_s2g_decision",
            "post_second_rollback_sufficiency", "final"):
        if key in rollback:
            compact_rollback[key] = rollback[key]
    return {
        "mode": "eaes",
        "query_plan": query_plan,
        "routing": routing,
        "phrase_retrieval": retrieval.get("phrase_retrieval") or {},
        "parent_candidates": [
            {key: item.get(key) for key in parent_fields if key in item}
            for item in retrieval.get("parent_candidate_scores") or []
        ],
        "child_candidates": [
            {key: item.get(key) for key in child_fields if key in item}
            for item in retrieval.get("initial_candidates") or []
        ],
        "child_probe_candidates": [
            {key: item.get(key) for key in child_fields if key in item}
            for item in retrieval.get("child_probe_candidates") or []
        ],
        "selected_view_child_candidates": [
            {key: item.get(key) for key in child_fields if key in item}
            for item in retrieval.get("prefilter_candidates") or []
        ],
        "parent_probe_candidates": [
            {key: item.get(key) for key in parent_fields if key in item}
            for item in retrieval.get("parent_probe_candidates") or []
        ],
        "selected_view_parent_candidates": [
            {key: item.get(key) for key in parent_fields if key in item}
            for item in retrieval.get("selected_parent_pool") or []
        ],
        "final_child_candidates": [
            {key: item.get(key) for key in child_fields if key in item}
            for item in retrieval.get("candidates") or []
        ],
        "final_parent_candidates": [
            {key: item.get(key) for key in parent_fields if key in item}
            for item in retrieval.get("parent_candidates") or []
        ],
        "final_child_ids": retrieval.get("final_child_ids") or [],
        "final_parent_ids": [
            item.get("parent_id")
            for item in retrieval.get("parent_candidates") or []
            if item.get("parent_id")
        ],
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

    def _diagnose_eaes_adaptive_gold_memories(
            self, gold_evidence, retrieval, question_emb=None
    ):
        phrases = self._as_list(
            (retrieval.get("query_plan") or {}).get("retrieval_phrases")
        )
        phrase_retrieval = retrieval.get("phrase_retrieval") or {}
        selected_phrase_indices = self._as_list(
            phrase_retrieval.get("selected_phrase_indices")
        )
        gold_origins = self._normalize_evidence_ids(gold_evidence)

        origin_event_ids = {
            origin: self._episode_ids_for_origin(origin)
            for origin in gold_origins
        }
        origin_memory_ids = {}
        requested_memory_ids = []
        for origin, event_ids in origin_event_ids.items():
            memory_ids = self._unique_keep_order([
                self.memory.eaes_event_to_memory.get(event_id)
                for event_id in event_ids
                if self.memory.eaes_event_to_memory.get(event_id)
            ])
            origin_memory_ids[origin] = memory_ids
            requested_memory_ids.extend(memory_ids)

        origin_parent_ids = {origin: [] for origin in gold_origins}
        requested_parent_ids = []
        for parent in self.memory.eaes_parent_nodes.values():
            linked_origins = set()
            for child_id in parent.child_ids:
                child = self.memory.eaes_notes.get(child_id)
                if child is not None:
                    linked_origins.update(
                        self._normalize_evidence_ids([child.origin])
                    )
            for origin in gold_origins:
                if origin in linked_origins:
                    origin_parent_ids[origin].append(parent.parent_id)
                    requested_parent_ids.append(parent.parent_id)

        node_scores = self.memory_controller.diagnose_eaes_nodes_against_phrases(
            phrases,
            selected_phrase_indices,
            question_emb=question_emb,
            child_memory_ids=self._unique_keep_order(requested_memory_ids),
            parent_ids=self._unique_keep_order(requested_parent_ids),
        )

        def by_id(items, field):
            return {
                item.get(field): item for item in self._as_list(items)
                if isinstance(item, dict) and item.get(field)
            }

        child_probe = by_id(
            retrieval.get("child_probe_candidates"), "memory_id"
        )
        child_selected = by_id(
            retrieval.get("prefilter_candidates"), "memory_id"
        )
        child_adaptive = by_id(
            retrieval.get("initial_candidates"), "memory_id"
        )
        child_final = by_id(retrieval.get("candidates"), "memory_id")
        parent_probe = by_id(
            retrieval.get("parent_probe_candidates"), "parent_id"
        )
        parent_selected = by_id(
            retrieval.get("selected_parent_pool"), "parent_id"
        )
        parent_final = by_id(
            retrieval.get("parent_candidates"), "parent_id"
        )

        def best_phrase_index(scores):
            if not scores:
                return None
            return max(
                scores,
                key=lambda row: (
                    float(row.get("diagnostic_support_potential") or 0.0),
                    -int(row.get("phrase_index") or 0),
                ),
            ).get("phrase_index")

        diagnostics = {
            "candidate_phrase_count": len(phrases),
            "selected_phrase_indices": selected_phrase_indices,
            "child_phrase_top_k": getattr(
                config, "EAES_PHRASE_INITIAL_TOP_K", 30
            ),
            "parent_phrase_top_k": getattr(
                config, "EAES_PARENT_PHRASE_TOP_K", 10
            ),
            "question_relevance_weight": getattr(
                config, "EAES_QUESTION_RELEVANCE_WEIGHT", 0.7
            ),
            "fusion_consensus_weight": getattr(
                config, "EAES_FUSION_CONSENSUS_WEIGHT", 0.5
            ),
            "child_mass_target": getattr(
                config, "EAES_CHILD_MASS_TARGET", 0.85
            ),
            "parent_mass_target": getattr(
                config, "EAES_PARENT_MASS_TARGET", 0.85
            ),
            "gold_origins": [],
        }
        for origin in gold_origins:
            child_nodes = []
            for memory_id in origin_memory_ids.get(origin, []):
                note = self.memory.get_eaes_note(memory_id)
                phrase_info = (node_scores.get("child") or {}).get(
                    memory_id, {}
                )
                phrase_scores = phrase_info.get("phrase_scores") or []
                fused = child_selected.get(memory_id) or {}
                adaptive = child_adaptive.get(memory_id) or {}
                final = child_final.get(memory_id) or {}
                any_top30 = any(
                    row.get("inside_top30") for row in phrase_scores
                )
                selected_top30 = any(
                    row.get("inside_top30") and row.get("phrase_selected")
                    for row in phrase_scores
                )
                if note is None:
                    drop_reason = "no_child_memory_built"
                elif memory_id in child_final:
                    drop_reason = "inside_final_child"
                elif not any_top30:
                    drop_reason = "outside_local_relevance_top30"
                elif not selected_top30:
                    drop_reason = "retrieved_only_by_unselected_phrase"
                elif memory_id not in child_adaptive:
                    drop_reason = (
                        "outside_child_fused_safety_cap"
                        if not fused.get("inside_fused_safety_cap")
                        else "outside_child_cumulative_mass_prefix"
                    )
                else:
                    drop_reason = "missing_after_child_rerank"
                child_nodes.append({
                    "memory_id": memory_id,
                    "event_id": note.event_id if note is not None else None,
                    "parent_id": note.parent_id if note is not None else None,
                    "rewrite_content": (
                        note.rewrite_content if note is not None else None
                    ),
                    "question_relevance": phrase_info.get(
                        "question_relevance"
                    ),
                    "best_phrase_index": best_phrase_index(phrase_scores),
                    "phrase_scores": phrase_scores,
                    "inside_all_view_probe": memory_id in child_probe,
                    "inside_selected_view_union": memory_id in child_selected,
                    "rrf_score": fused.get("rrf_score", 0.0),
                    "rrf_score_ratio": fused.get("rrf_score_ratio", 0.0),
                    "rrf_rank": fused.get("rrf_rank"),
                    "weighted_rrf_sum": fused.get(
                        "weighted_rrf_sum", 0.0
                    ),
                    "best_view_contribution": fused.get(
                        "best_view_contribution", 0.0
                    ),
                    "normalized_consensus": fused.get(
                        "normalized_consensus", 0.0
                    ),
                    "normalized_best_view": fused.get(
                        "normalized_best_view", 0.0
                    ),
                    "fused_score": fused.get("fused_score", 0.0),
                    "fused_score_ratio": fused.get(
                        "fused_score_ratio", 0.0
                    ),
                    "fused_rank": fused.get("fused_rank"),
                    "normalized_mass": fused.get("normalized_mass", 0.0),
                    "cumulative_mass": fused.get("cumulative_mass"),
                    "mass_target": fused.get("mass_target"),
                    "inside_fused_safety_cap": fused.get(
                        "inside_fused_safety_cap", False
                    ),
                    "adaptive_child_k": fused.get(
                        "adaptive_k",
                        (retrieval.get("counts") or {}).get(
                            "adaptive_child_k"
                        ),
                    ),
                    "inside_adaptive_prefix": memory_id in child_adaptive,
                    "pre_rerank_rank": fused.get("fused_rank"),
                    "rerank_rank": final.get("rerank_rank"),
                    "inside_final_child": memory_id in child_final,
                    "drop_reason": drop_reason,
                })

            parent_nodes = []
            for parent_id in origin_parent_ids.get(origin, []):
                parent = self.memory.get_eaes_parent_node(parent_id)
                phrase_info = (node_scores.get("parent") or {}).get(
                    parent_id, {}
                )
                phrase_scores = phrase_info.get("phrase_scores") or []
                fused = parent_selected.get(parent_id) or {}
                final = parent_final.get(parent_id) or {}
                any_top10 = any(
                    row.get("inside_top10") for row in phrase_scores
                )
                selected_top10 = any(
                    row.get("inside_top10") and row.get("phrase_selected")
                    for row in phrase_scores
                )
                if parent is None:
                    drop_reason = "no_parent_memory_built"
                elif parent_id in parent_final:
                    drop_reason = "inside_final_parent"
                elif not any_top10:
                    drop_reason = "outside_local_relevance_parent_top10"
                elif not selected_top10:
                    drop_reason = "retrieved_only_by_unselected_phrase"
                else:
                    drop_reason = (
                        "outside_parent_fused_safety_cap"
                        if not fused.get("inside_fused_safety_cap")
                        else "outside_parent_cumulative_mass_prefix"
                    )
                parent_nodes.append({
                    "parent_id": parent_id,
                    "rewrite_content": (
                        parent.rewrite_content if parent is not None else None
                    ),
                    "origin_association": "child_membership_proxy",
                    "question_relevance": phrase_info.get(
                        "question_relevance"
                    ),
                    "best_phrase_index": best_phrase_index(phrase_scores),
                    "phrase_scores": phrase_scores,
                    "inside_all_view_probe": parent_id in parent_probe,
                    "inside_selected_view_union": parent_id in parent_selected,
                    "rrf_score": fused.get("rrf_score", 0.0),
                    "rrf_score_ratio": fused.get("rrf_score_ratio", 0.0),
                    "rrf_rank": fused.get("rrf_rank"),
                    "weighted_rrf_sum": fused.get(
                        "weighted_rrf_sum", 0.0
                    ),
                    "best_view_contribution": fused.get(
                        "best_view_contribution", 0.0
                    ),
                    "normalized_consensus": fused.get(
                        "normalized_consensus", 0.0
                    ),
                    "normalized_best_view": fused.get(
                        "normalized_best_view", 0.0
                    ),
                    "fused_score": fused.get("fused_score", 0.0),
                    "fused_score_ratio": fused.get(
                        "fused_score_ratio", 0.0
                    ),
                    "fused_rank": fused.get("fused_rank"),
                    "normalized_mass": fused.get("normalized_mass", 0.0),
                    "cumulative_mass": fused.get("cumulative_mass"),
                    "mass_target": fused.get("mass_target"),
                    "inside_fused_safety_cap": fused.get(
                        "inside_fused_safety_cap", False
                    ),
                    "adaptive_parent_k": fused.get(
                        "adaptive_k",
                        (retrieval.get("counts") or {}).get(
                            "adaptive_parent_k"
                        ),
                    ),
                    "inside_adaptive_prefix": parent_id in parent_final,
                    "inside_final_parent": parent_id in parent_final,
                    "drop_reason": drop_reason,
                })

            covered_by_child = any(
                node["inside_final_child"] for node in child_nodes
            )
            covered_by_parent = any(
                node["inside_final_parent"] for node in parent_nodes
            )
            if covered_by_child and covered_by_parent:
                final_path = "child_and_parent"
            elif covered_by_child:
                final_path = "child"
            elif covered_by_parent:
                final_path = "parent"
            else:
                final_path = None
            if final_path:
                origin_drop_reason = None
            elif not origin_event_ids.get(origin):
                origin_drop_reason = "gold_origin_not_in_memory"
            elif not child_nodes and not parent_nodes:
                origin_drop_reason = "no_memory_node_built_for_origin"
            else:
                reasons = [
                    node.get("drop_reason")
                    for node in child_nodes + parent_nodes
                    if node.get("drop_reason")
                ]
                origin_drop_reason = reasons[0] if reasons else "not_retrieved"
            diagnostics["gold_origins"].append({
                "origin": origin,
                "event_ids": origin_event_ids.get(origin, []),
                "memory_ids": origin_memory_ids.get(origin, []),
                "linked_parent_ids": origin_parent_ids.get(origin, []),
                "covered_by_retrieval": bool(final_path),
                "covered_by_final": bool(final_path),
                "covered_by_child": covered_by_child,
                "covered_by_parent": covered_by_parent,
                "final_path": final_path,
                "drop_reason": origin_drop_reason,
                "child_nodes": child_nodes,
                "parent_nodes": parent_nodes,
            })
        return diagnostics

    def diagnose_eaes_gold_memories(self, gold_evidence, retrieval, question_emb=None, window=2):
        if not isinstance(retrieval, dict) or retrieval.get("mode") != "eaes":
            return None
        if "child_rankings" in retrieval:
            return self._diagnose_eaes_adaptive_gold_memories(
                gold_evidence, retrieval, question_emb
            )
        phrase_retrieval = retrieval.get("phrase_retrieval") or {}
        prefilter_candidates = self._as_list(retrieval.get("prefilter_candidates"))
        initial_candidates = self._as_list(retrieval.get("initial_candidates"))
        ranked = sorted(
            (
                item for item in initial_candidates
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
        initial_memory_ids = {
            item.get("memory_id")
            for item in initial_candidates
            if isinstance(item, dict)
        }
        pool_dropped_ids = set(
            (retrieval.get("initial_retrieval") or {}).get(
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
            "initial_limit": getattr(config, "EAES_PHRASE_UNION_LIMIT", 60),
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
                    "in_prefilter_child": False,
                    "in_initial_child": False,
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
                    drop_reason = "dropped_by_initial_pool_limit"
                elif memory_id in initial_memory_ids:
                    drop_reason = "dropped_by_llm_reranker"
                else:
                    drop_reason = "not_in_any_dynamic_phrase_topk"
                memory_entries.append({
                    "event_id": event_id,
                    "memory_id": memory_id,
                    "parent_id": note.parent_id if note is not None else None,
                    "indexed": note is not None,
                    "in_prefilter_child": memory_id in prefilter_memory_ids,
                    "in_initial_child": memory_id in initial_memory_ids,
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
                origin_drop_reason = "dropped_by_initial_pool_limit"
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
            prefilter_candidates = first_pass["prefilter_children"]
            initial_candidates = first_pass["initial_children"]
            candidates = first_pass["final_children"]
            parent_candidates = first_pass["selected_parents"]
            child_probe_candidates = first_pass.get(
                "child_probe_candidates", prefilter_candidates
            )
            parent_probe_candidates = first_pass.get(
                "parent_probe_candidates", parent_candidates
            )
            selected_parent_pool = first_pass.get(
                "selected_parent_pool", parent_candidates
            )
            rollback_metadata = {"enabled": False}
            if getattr(config, "EAES_ROLLBACK_CHECK", False):
                first_parent_candidates = list(parent_candidates)
                candidates, parent_candidates, rollback_metadata = (
                    self.apply_eaes_rollback_check(
                        question,
                        query_plan,
                        candidates,
                        parent_candidates,
                        question_emb,
                    )
                )
                rollback_metadata["initial_retrieval_pool"] = {
                    "child_ids": [
                        candidate.get("memory_id")
                        for candidate in initial_candidates
                    ],
                    "parent_ids": [
                        candidate.get("parent_id")
                        for candidate in first_parent_candidates
                    ],
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

            def parent_origin_groups_for(items):
                return [
                    self.memory.get_eaes_support_origin([
                        parent.get("parent_id")
                    ])
                    for parent in items or []
                    if parent.get("parent_id")
                ]

            event_ids = self._unique_keep_order([
                candidate.get("event_id") for candidate in candidates
            ])
            child_groups = child_origin_groups(candidates)
            child_origins = origins_from_groups(child_groups)
            parent_ids = self._unique_keep_order([
                parent.get("parent_id") for parent in parent_candidates
            ])
            parent_origin_groups = parent_origin_groups_for(parent_candidates)
            parent_origins = self._unique_keep_order([
                origin
                for group in parent_origin_groups
                for origin in group
            ])
            final_groups = child_groups + parent_origin_groups
            final_origins = origins_from_groups(final_groups)

            child_probe_groups = child_origin_groups(child_probe_candidates)
            parent_probe_groups = parent_origin_groups_for(
                parent_probe_candidates
            )
            prefilter_groups = child_origin_groups(prefilter_candidates)
            selected_parent_groups = parent_origin_groups_for(
                selected_parent_pool
            )
            initial_groups = child_origin_groups(initial_candidates)
            stage_origin_groups = {
                "child_probe_all_views": child_probe_groups,
                "parent_probe_all_views": parent_probe_groups,
                "child_selected_views": prefilter_groups,
                "parent_selected_views": selected_parent_groups,
                "child_pre_rerank": initial_groups,
                "parent_final": parent_origin_groups,
                "prefilter_child": prefilter_groups,
                "initial_child": initial_groups,
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
                "child_rankings": first_pass.get("child_rankings", []),
                "parent_rankings": first_pass.get("parent_rankings", []),
                "initial_retrieval": first_pass["initial_retrieval"],
                "parent_candidate_scores": first_pass["routing"].get(
                    "parent_candidates", []
                ),
                "prefilter_candidates": prefilter_candidates,
                "initial_candidates": initial_candidates,
                "child_probe_candidates": child_probe_candidates,
                "parent_probe_candidates": parent_probe_candidates,
                "selected_parent_pool": selected_parent_pool,
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
                "initial_origins": stage_origins["initial_child"],
                "initial_origin_groups": initial_groups,
                "initial_k": len(initial_candidates),
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


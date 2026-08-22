import json
import logging
import re
from typing import Any, Dict, List

import numpy as np

from common import config
from common.utils import topk_answers_by_similarity

logger = logging.getLogger(__name__)


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
                int(item.get("_rrf_rank") or 10**9),
                str(item.get("memory_id") or ""),
            ),
        )
        by_memory_id = {item.get("memory_id"): item for item in ranked if item.get("memory_id")}
        by_event_id = {item.get("event_id"): item for item in ranked if item.get("event_id")}
        phrase_hits = {}
        for phrase_index, phrase_ranking in enumerate(
                self._as_list(phrase_retrieval.get("phrase_top15"))):
            for item in self._as_list(phrase_ranking):
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
        reranked_candidates = self._as_list(retrieval.get("candidates"))
        reranked_by_memory_id = {
            item.get("memory_id"): item for item in reranked_candidates
            if isinstance(item, dict) and item.get("memory_id")
        }
        retrieved_memory_ids = {
            item.get("memory_id") for item in reranked_candidates if isinstance(item, dict)
        }
        retrieved_event_ids = {
            item for item in self._as_list(retrieval.get("retrieved_event_ids")) if item
        }
        parent_ids = self._as_list(retrieval.get("parent_ids"))
        parent_origin_groups = self._as_list(
            retrieval.get("parent_origin_groups")
        )

        diagnostics = {
            "prefilter_limit": getattr(config, "EAES_PHRASE_UNION_LIMIT", 40),
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
                    "indexed": False,
                    "in_prefilter_topk": False,
                    "in_llm_topk": False,
                    "in_retrieved_candidates": False,
                    "candidate_rank": None,
                    "prefilter_rank": None,
                    "rerank_rank": None,
                    "candidate_score": None,
                    "score_parts": None,
                    "entities": [],
                    "attribute_paths": [],
                    "rewrite_content": None,
                    "nearby_ranked_candidates": [],
                    "drop_reason": "gold_origin_not_in_episode_events",
                })
            for event_id in event_ids:
                memory_id = self.memory.eaes_event_to_memory.get(event_id)
                note = self.memory.get_eaes_note(memory_id) if memory_id else None
                scored = by_memory_id.get(memory_id) or by_event_id.get(event_id)
                rank = scored.get("_rrf_rank") if scored else None
                reranked = reranked_by_memory_id.get(memory_id)
                rerank_rank = reranked.get("rerank_rank") if reranked else None
                neighbor_window = []
                if rank is not None:
                    start = max(0, rank - window - 1)
                    end = min(len(ranked), rank + window)
                    for neighbor in ranked[start:end]:
                        neighbor_window.append({
                            "rank": neighbor.get("rank"),
                            "memory_id": neighbor.get("memory_id"),
                            "event_id": neighbor.get("event_id"),
                            "origin": neighbor.get("origin"),
                            "score": neighbor.get("_rrf_score"),
                            "phrase_hits": phrase_hits.get(
                                neighbor.get("memory_id"), []
                            ),
                        })
                memory_entries.append({
                    "event_id": event_id,
                    "memory_id": memory_id,
                    "indexed": note is not None,
                    "in_prefilter_topk": memory_id in prefilter_memory_ids,
                    "in_llm_topk": memory_id in reranked_by_memory_id,
                    "in_retrieved_candidates": memory_id in retrieved_memory_ids or event_id in retrieved_event_ids,
                    "candidate_rank": rank,
                    "prefilter_rank": rank,
                    "rerank_rank": rerank_rank,
                    "candidate_score": scored.get("_rrf_score") if scored else None,
                    "score_parts": {
                        "phrase_hits": phrase_hits.get(memory_id, []),
                    },
                    "entities": note.entities if note is not None else [],
                    "attribute_paths": note.attribute_paths if note is not None else [],
                    "rewrite_content": note.rewrite_content if note is not None else None,
                    "nearby_ranked_candidates": neighbor_window,
                    "drop_reason": (
                        "not_built_in_eaes_memory" if note is None else
                        "not_in_any_phrase_top15" if memory_id not in phrase_hits else
                        "dropped_by_per_phrase_rrf" if scored is None else
                        "dropped_by_llm_reranker" if memory_id not in reranked_by_memory_id else
                        "inside_llm_topk"
                    ),
                })
            covered_by_child = any(
                entry["in_retrieved_candidates"] for entry in memory_entries
            )
            matched_parent_ids = [
                parent_id
                for parent_id, parent_origins in zip(
                    parent_ids, parent_origin_groups
                )
                if origin in self._normalize_evidence_ids(parent_origins)
            ]
            covered_by_parent = bool(matched_parent_ids)
            covered_by_retrieval = covered_by_child or covered_by_parent
            rank_values = [
                entry["candidate_rank"] for entry in memory_entries
                if entry["candidate_rank"] is not None
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
                origin_drop_reason = "inside_llm_topk"
            elif covered_by_parent:
                origin_drop_reason = "inside_parent_topk"
            elif rank_values:
                origin_drop_reason = "dropped_by_llm_reranker"
            elif any(memory_id in phrase_hits for memory_id in memory_ids):
                origin_drop_reason = "dropped_by_per_phrase_rrf"
            else:
                origin_drop_reason = "not_in_any_phrase_top15"

            diagnostics["gold_origins"].append({
                "origin": origin,
                "event_ids": event_ids,
                "memory_ids": memory_ids,
                "covered_by_retrieval": covered_by_retrieval,
                "covered_by_child": covered_by_child,
                "covered_by_parent": covered_by_parent,
                "matched_parent_ids": matched_parent_ids,
                "drop_reason": origin_drop_reason,
                "best_rank": min(rank_values, default=None),
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
            query_plan = self.parse_eaes_query(question, question_emb)
            child_query_plan = self._eaes_child_query_plan(query_plan)
            parent_candidates = []
            if getattr(config, "SEMANTIC_HIERARCHY", False):
                parent_candidates = self.memory_controller.retrieve_eaes_parent_candidates(
                    query_plan, question_emb, limit=config.PARENT_TOP_K
                )
            embedding_candidates, phrase_retrieval = (
                self.memory_controller.retrieve_eaes_phrase_candidates(
                    query_plan["retrieval_phrases"],
                    include_diagnostics=True,
                )
            )
            candidates = self.rerank_eaes_phrase_candidates(
                question,
                embedding_candidates,
                top_k=getattr(config, "EAES_PHRASE_RERANK_LIMIT", 15),
            )
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
                            for candidate in embedding_candidates
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
                                for candidate in embedding_candidates
                            ],
                            "parent_ids": [
                                candidate.get("parent_id")
                                for candidate in first_parent_candidates
                            ],
                        },
                        "failure_reason": "reader_gate_or_rollback_exception",
                    }
            event_ids = self._unique_keep_order([c.get("event_id") for c in candidates])
            child_origins = self._unique_keep_order(
                [c.get("origin") for c in candidates] or self._origins_for_event_ids(event_ids)
            )
            if not child_origins:
                child_origins = self._origins_for_event_ids(event_ids)
            child_origin_groups = [
                [candidate.get("origin")]
                if candidate.get("origin") else
                self._origins_for_event_ids([candidate.get("event_id")])
                for candidate in candidates
            ]
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
            origins = self._unique_keep_order(child_origins + parent_origins)
            origin_groups = child_origin_groups + parent_origin_groups
            prefilter_event_ids = self._unique_keep_order(
                [c.get("event_id") for c in embedding_candidates]
            )
            prefilter_origins = self._unique_keep_order(
                [c.get("origin") for c in embedding_candidates]
                or self._origins_for_event_ids(prefilter_event_ids)
            )
            prefilter_origin_groups = [
                [candidate.get("origin")]
                if candidate.get("origin") else
                self._origins_for_event_ids([candidate.get("event_id")])
                for candidate in embedding_candidates
            ]
            child_k = len(candidates)
            parent_k = len(parent_candidates)
            return {
                "mode": "eaes",
                "query_plan": query_plan,
                "retrieved_event_ids": event_ids,
                "retrieved_memory_ids": self._unique_keep_order(
                    [candidate.get("memory_id") for candidate in candidates]
                    + parent_ids
                ),
                "retrieved_origins": origins,
                "retrieved_origin_groups": origin_groups,
                "retrieval_k": child_k + parent_k,
                "child_k": child_k,
                "parent_k": parent_k,
                "candidates": candidates,
                "child_origins": child_origins,
                "child_origin_groups": child_origin_groups,
                "parent_candidates": parent_candidates,
                "parent_ids": parent_ids,
                "parent_origins": parent_origins,
                "parent_origin_groups": parent_origin_groups,
                "prefilter_event_ids": prefilter_event_ids,
                "prefilter_origins": prefilter_origins,
                "prefilter_origin_groups": prefilter_origin_groups,
                "prefilter_k": getattr(config, "EAES_PHRASE_UNION_LIMIT", 40),
                "prefilter_candidates": embedding_candidates,
                "phrase_retrieval": {
                    "retrieval_phrases": query_plan["retrieval_phrases"],
                    **phrase_retrieval,
                },
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


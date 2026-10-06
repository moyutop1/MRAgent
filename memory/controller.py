import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import json
import re
import threading
from datetime import date
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from nltk.stem import PorterStemmer
from common import config
from llm.controller import LLM
from llm.embeddings import get_embedding
from memory.system import MemorySystem
from prompts.schema import check_composite_tag
import logging
logger = logging.getLogger(__name__)

_EAES_EMBEDDING_LOCK = threading.Lock()
_EAES_STEMMER = PorterStemmer()

class MemoryController:
    """Wraps your storage; replace with your own DB/vector/graph implementation."""

    def __init__(self, store: MemorySystem, llm: Optional[LLM] = None):
        self.memory = store
        self.llm = llm
        self.question_emb = None
        self.queried_event = []
        self._eaes_query_embedding_cache = {}
        self._eaes_parent_query_embedding_cache = {}
        self._eaes_tag_embedding_cache = {}
        self._eaes_phrase_embedding_cache = {}


    def query_conversation_time(self, event_id):
        return f"Conversation_time:{event_id}:{self.memory.query_conversation_time(event_id)}"

    def query_event_context(self, event_id):
        return self.memory.query_event_context(event_id)

    def query_topic_events(self, topic):
        return self.memory.query_topic_events(topic)


    def query_personal_information(self, person):
        return self.memory.query_personal_information(person)

    def query_personal_aspect(self, person, aspect):
        return self.memory.query_personal_aspect( person, aspect)

    @staticmethod
    def _eaes_words(text: str) -> Set[str]:
        text = MemoryController._snake_norm(text).replace("_", " ")
        return {_EAES_STEMMER.stem(token) for token in text.split() if token}

    @staticmethod
    def _eaes_overlap_score(left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, len(left))

    @staticmethod
    def _normalize_embedding_rows(values):
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @staticmethod
    def _eaes_retrieval_text(note):
        attributes = "\n".join(str(attr) for attr in note.attribute_paths or [] if str(attr).strip())
        return f"ATTRIBUTES:\n{attributes}\nREWRITE:\n{note.rewrite_content or ''}".strip()

    def prepare_eaes_retrieval_embeddings(self):
        pending = [note for note in self.memory.eaes_notes.values() if note.retrieval_embedding is None]
        if not pending:
            return
        with _EAES_EMBEDDING_LOCK:
            pending = [note for note in self.memory.eaes_notes.values() if note.retrieval_embedding is None]
            if not pending:
                return
            vectors = self._normalize_embedding_rows(
                get_embedding([self._eaes_retrieval_text(note) for note in pending])
            )
            if len(vectors) != len(pending):
                raise RuntimeError(
                    f"EAES retrieval embedding count mismatch: {len(vectors)} != {len(pending)}"
                )
            for note, vector in zip(pending, vectors):
                note.retrieval_embedding = vector

    def prepare_eaes_parent_embeddings(self):
        pending = [
            parent for parent in self.memory.eaes_parent_nodes.values()
            if parent.retrieval_embedding is None and parent.rewrite_content
        ]
        if not pending:
            return
        with _EAES_EMBEDDING_LOCK:
            pending = [
                parent for parent in self.memory.eaes_parent_nodes.values()
                if parent.retrieval_embedding is None and parent.rewrite_content
            ]
            if not pending:
                return
            vectors = self._normalize_embedding_rows(
                get_embedding([parent.rewrite_content for parent in pending])
            )
            if len(vectors) != len(pending):
                raise RuntimeError(
                    f"EAES parent embedding count mismatch: {len(vectors)} != {len(pending)}"
                )
            for parent, vector in zip(pending, vectors):
                parent.retrieval_embedding = vector

    def _eaes_child_tags(self, note):
        event = self.memory.episode_events[note.event_id]
        tags = event.tag_t
        if not isinstance(tags, list) or not 1 <= len(tags) <= 4:
            raise ValueError(
                f"EAES child {note.event_id} requires a tag array with 1-4 items"
            )
        clean_tags = []
        for tag in tags:
            clean_tag = re.sub(r"\s+", " ", str(tag or "")).strip()
            valid, error = check_composite_tag(clean_tag)
            if not valid:
                raise ValueError(
                    f"EAES child {note.event_id} has an invalid tag "
                    f"{tag!r}: {error}"
                )
            clean_tags.append(clean_tag)
        if len({tag.casefold() for tag in clean_tags}) != len(clean_tags):
            raise ValueError(
                f"EAES child {note.event_id} has duplicate normalized tags"
            )
        return clean_tags

    def _prepare_eaes_tag_embeddings(self):
        pending = []
        for note in self.memory.eaes_notes.values():
            tags = tuple(self._eaes_child_tags(note))
            cached = self._eaes_tag_embedding_cache.get(note.memory_id)
            if cached is None or cached[0] != tags:
                pending.append((note, tags))
        if not pending:
            return

        with _EAES_EMBEDDING_LOCK:
            pending = []
            for note in self.memory.eaes_notes.values():
                tags = tuple(self._eaes_child_tags(note))
                cached = self._eaes_tag_embedding_cache.get(note.memory_id)
                if cached is None or cached[0] != tags:
                    pending.append((note, tags))
            if not pending:
                return

            flat_tags = [tag for _, tags in pending for tag in tags]
            vectors = self._normalize_embedding_rows(
                get_embedding(flat_tags)
            )
            if len(vectors) != len(flat_tags):
                raise RuntimeError(
                    "EAES tag embedding count mismatch: "
                    f"{len(vectors)} != {len(flat_tags)}"
                )
            offset = 0
            for note, tags in pending:
                count = len(tags)
                self._eaes_tag_embedding_cache[note.memory_id] = (
                    tags, vectors[offset:offset + count]
                )
                offset += count

    def _eaes_phrase_embeddings(self, retrieval_phrases):
        cache_key = tuple(retrieval_phrases)
        vectors = self._eaes_phrase_embedding_cache.get(cache_key)
        if vectors is not None:
            return vectors
        with _EAES_EMBEDDING_LOCK:
            vectors = self._eaes_phrase_embedding_cache.get(cache_key)
            if vectors is None:
                vectors = self._normalize_embedding_rows(
                    get_embedding(list(retrieval_phrases))
                )
                self._eaes_phrase_embedding_cache[cache_key] = vectors
        return vectors

    def rank_eaes_children_per_phrase(
            self,
            retrieval_phrases,
            top_k=None,
            exclude_memory_ids=None,
            question_relevance_by_id=None,
    ):
        phrases = [str(value).strip() for value in retrieval_phrases or []]
        top_k = top_k or config.EAES_PHRASE_INITIAL_TOP_K
        excluded = set(exclude_memory_ids or [])

        self._prepare_eaes_tag_embeddings()
        phrase_vectors = self._eaes_phrase_embeddings(phrases)
        notes = sorted(
            (
                note for note in self.memory.eaes_notes.values()
                if note.memory_id not in excluded
            ),
            key=lambda note: note.memory_id,
        )
        if not notes:
            return [[] for _ in phrases]

        relevance_by_id = question_relevance_by_id or {}
        beta = float(config.EAES_QUESTION_RELEVANCE_WEIGHT)
        phrase_rankings = []
        for phrase_index, (phrase, phrase_vector) in enumerate(
                zip(phrases, phrase_vectors)):
            scored = []
            for note in notes:
                tags, tag_vectors = self._eaes_tag_embedding_cache[note.memory_id]
                similarities = np.dot(tag_vectors, phrase_vector)
                best_tag_index = int(np.argmax(similarities))
                phrase_similarity = max(
                    0.0,
                    min(1.0, float(similarities[best_tag_index])),
                )
                question_relevance = float(
                    relevance_by_id.get(note.memory_id, 0.0)
                )
                local_relevance = self._eaes_joint_relevance(
                    question_relevance,
                    phrase_similarity,
                    beta,
                )
                scored.append((
                    note,
                    local_relevance,
                    phrase_similarity,
                    question_relevance,
                    tags[best_tag_index],
                    best_tag_index,
                ))
            scored = sorted(
                scored, key=lambda item: (-item[1], item[0].memory_id)
            )[:top_k]
            ranking = []
            for phrase_rank, (
                    note, local_relevance, similarity, question_relevance,
                    matched_tag, matched_tag_index
            ) in enumerate(scored, start=1):
                ranking.append({
                    **note.to_dict(include_raw=False),
                    "tag": self._eaes_child_tags(note),
                    "phrase_index": phrase_index,
                    "phrase": phrase,
                    "phrase_rank": phrase_rank,
                    "phrase_similarity": similarity,
                    "question_relevance": question_relevance,
                    "joint_relevance": local_relevance,
                    "matched_tag": matched_tag,
                    "matched_tag_index": matched_tag_index,
                })
            phrase_rankings.append(ranking)
        return phrase_rankings

    def rank_eaes_parents_per_phrase(
            self,
            retrieval_phrases,
            top_k=None,
            exclude_parent_ids=None,
            question_relevance_by_id=None,
    ):
        """Probe parent summaries independently with the shared phrase pool."""
        phrases = [str(value).strip() for value in retrieval_phrases or []]
        top_k = top_k or config.EAES_PARENT_PHRASE_TOP_K
        excluded = set(exclude_parent_ids or [])
        if not config.SEMANTIC_HIERARCHY:
            return [[] for _ in phrases]

        self.prepare_eaes_parent_embeddings()
        phrase_vectors = self._eaes_phrase_embeddings(phrases)
        parents = sorted(
            (
                parent for parent in self.memory.eaes_parent_nodes.values()
                if parent.parent_id not in excluded
                and parent.retrieval_embedding is not None
            ),
            key=lambda parent: parent.parent_id,
        )
        relevance_by_id = question_relevance_by_id or {}
        beta = float(config.EAES_QUESTION_RELEVANCE_WEIGHT)
        rankings = []
        for phrase_index, (phrase, phrase_vector) in enumerate(
                zip(phrases, phrase_vectors)):
            scored = []
            for parent in parents:
                parent_vector = self._normalize_embedding_rows(
                    parent.retrieval_embedding
                )[0]
                phrase_similarity = self._eaes_clipped_cosine(
                    phrase_vector, parent_vector
                )
                question_relevance = float(
                    relevance_by_id.get(parent.parent_id, 0.0)
                )
                local_relevance = self._eaes_joint_relevance(
                    question_relevance,
                    phrase_similarity,
                    beta,
                )
                scored.append((
                    parent,
                    local_relevance,
                    phrase_similarity,
                    question_relevance,
                ))
            scored.sort(key=lambda row: (-row[1], row[0].parent_id))
            rankings.append([
                {
                    **parent.to_reader_dict(),
                    "phrase_index": phrase_index,
                    "phrase": phrase,
                    "phrase_rank": rank,
                    "phrase_similarity": similarity,
                    "question_relevance": question_relevance,
                    "joint_relevance": local_relevance,
                }
                for rank, (
                    parent, local_relevance, similarity, question_relevance
                ) in enumerate(
                    scored[:top_k], start=1
                )
            ])
        return rankings

    def _eaes_question_vector(self, question_emb=None, question_text=None):
        if question_emb is not None:
            rows = self._normalize_embedding_rows(question_emb)
            return rows[0] if len(rows) else None
        question_text = str(question_text or "").strip()
        if not question_text:
            return None
        rows = self._normalize_embedding_rows(get_embedding([question_text]))
        return rows[0] if len(rows) else None

    @staticmethod
    def _eaes_clipped_cosine(left, right):
        if left is None or right is None:
            return 0.0
        return max(0.0, min(1.0, float(np.dot(left, right))))

    @staticmethod
    def _eaes_joint_relevance(
            question_relevance, phrase_similarity, beta=None
    ):
        """Require joint question and phrase support via a geometric gate."""
        beta = (
            float(config.EAES_QUESTION_RELEVANCE_WEIGHT)
            if beta is None else float(beta)
        )
        question_relevance = max(
            0.0, min(1.0, float(question_relevance or 0.0))
        )
        phrase_similarity = max(
            0.0, min(1.0, float(phrase_similarity or 0.0))
        )
        if question_relevance <= 0.0 or phrase_similarity <= 0.0:
            return 0.0
        return (
            question_relevance ** beta
            * phrase_similarity ** (1.0 - beta)
        )

    @staticmethod
    def _eaes_phrase_node_contribution(
            phrase_fidelity, joint_relevance, rank, rrf_k=None
    ):
        """Return the bounded phrase-to-node coverage contribution A_i(m)."""
        rank = int(rank or 0)
        if rank <= 0:
            return 0.0
        rrf_k = (
            float(config.EAES_PHRASE_RRF_K)
            if rrf_k is None else float(rrf_k)
        )
        fidelity = max(0.0, min(1.0, float(phrase_fidelity or 0.0)))
        relevance = max(0.0, min(1.0, float(joint_relevance or 0.0)))
        rank_discount = (rrf_k + 1.0) / (rrf_k + rank)
        return max(
            0.0,
            min(1.0, fidelity * relevance * rank_discount),
        )

    def _eaes_question_relevance_maps(
            self, question_emb=None, question_text=None
    ):
        """Score every node against the original question before Top-K."""
        self.prepare_eaes_retrieval_embeddings()
        if config.SEMANTIC_HIERARCHY:
            self.prepare_eaes_parent_embeddings()
        question_vector = self._eaes_question_vector(
            question_emb, question_text
        )
        child_relevance = {}
        for note in self.memory.eaes_notes.values():
            vector = (
                self._normalize_embedding_rows(note.retrieval_embedding)[0]
                if note.retrieval_embedding is not None else None
            )
            child_relevance[note.memory_id] = self._eaes_clipped_cosine(
                vector, question_vector
            )
        parent_relevance = {}
        for parent in self.memory.eaes_parent_nodes.values():
            vector = (
                self._normalize_embedding_rows(parent.retrieval_embedding)[0]
                if parent.retrieval_embedding is not None else None
            )
            parent_relevance[parent.parent_id] = self._eaes_clipped_cosine(
                vector, question_vector
            )
        return question_vector, child_relevance, parent_relevance

    def _score_eaes_view_rankings(
            self,
            child_rankings,
            parent_rankings,
            retrieval_phrases,
            question_emb=None,
            question_text=None,
    ):
        """Attach fidelity, question relevance, and MEG support potential."""
        phrase_vectors = self._eaes_phrase_embeddings(retrieval_phrases)
        question_vector = self._eaes_question_vector(
            question_emb, question_text
        )
        fidelities = [
            self._eaes_clipped_cosine(vector, question_vector)
            for vector in phrase_vectors
        ]
        rrf_k = float(config.EAES_PHRASE_RRF_K)

        def attach(rankings):
            for phrase_index, ranking in enumerate(rankings):
                fidelity = fidelities[phrase_index]
                for item in ranking:
                    rank = int(item.get("phrase_rank") or 0)
                    local_relevance = float(
                        item.get("joint_relevance") or 0.0
                    )
                    support = self._eaes_phrase_node_contribution(
                        fidelity,
                        local_relevance,
                        rank,
                        rrf_k,
                    )
                    item["phrase_fidelity"] = fidelity
                    item["support_potential"] = support

        attach(child_rankings)
        attach(parent_rankings)
        return fidelities

    @staticmethod
    def _eaes_view_contribution_maps(rankings, id_field):
        return [
            {
                item[id_field]: float(item.get("support_potential") or 0.0)
                for item in ranking
                if item.get(id_field)
            }
            for ranking in rankings
        ]

    @staticmethod
    def _eaes_marginal_gain(contributions, coverage):
        """Sum saturating node gains A_i(m) * (1 - C(m | S))."""
        return sum(
            value * (1.0 - coverage.get(node_id, 0.0))
            for node_id, value in contributions.items()
        )

    def _select_eaes_channel_views_by_meg(
            self,
            retrieval_phrases,
            fidelities,
            rankings,
            id_field,
    ):
        """Greedily maximize one channel's saturating evidence coverage."""
        contribution_maps = self._eaes_view_contribution_maps(
            rankings, id_field
        )
        scale = max(
            (sum(values.values()) for values in contribution_maps),
            default=0.0,
        )
        min_views = int(config.EAES_MIN_SELECTED_VIEWS)
        max_views = int(config.EAES_MAX_SELECTED_VIEWS)
        threshold = float(config.EAES_VIEW_GAIN_THRESHOLD)
        if scale <= 0.0:
            for ranking in rankings:
                for item in ranking:
                    item["phrase_selected"] = False
            return [], {
                "phrases": [
                    {
                        "phrase_index": phrase_index,
                        "phrase": phrase,
                        "fidelity": float(fidelities[phrase_index]),
                        "selected": False,
                        "selection_step": None,
                        "marginal_gain": 0.0,
                        "normalized_marginal_gain": 0.0,
                        "remaining_marginal_gain": 0.0,
                        "remaining_normalized_marginal_gain": 0.0,
                        "decision": "no_channel_evidence",
                    }
                    for phrase_index, phrase in enumerate(retrieval_phrases)
                ],
                "selection_history": [],
                "min_views": min_views,
                "max_views": max_views,
                "gain_threshold": threshold,
                "initial_gain_scale": 0.0,
            }
        coverage = {}
        selected = []
        history = []
        remaining = set(range(len(retrieval_phrases)))

        def normalized_gain(raw_gain):
            return raw_gain / scale if scale > 0.0 else 0.0

        while remaining and len(selected) < max_views:
            rows = []
            for phrase_index in sorted(remaining):
                raw_gain = self._eaes_marginal_gain(
                    contribution_maps[phrase_index], coverage
                )
                rows.append({
                    "phrase_index": phrase_index,
                    "marginal_gain": raw_gain,
                    "normalized_marginal_gain": normalized_gain(raw_gain),
                })
            best = max(
                rows,
                key=lambda row: (
                    row["normalized_marginal_gain"],
                    -row["phrase_index"],
                ),
            )
            must_select = len(selected) < min_views
            if (
                    not must_select
                    and best["normalized_marginal_gain"] < threshold
            ):
                history.append({
                    "step": len(selected) + 1,
                    "selected_phrase_index": None,
                    "stop_reason": "below_marginal_gain_threshold",
                    "candidates": rows,
                })
                break

            phrase_index = best["phrase_index"]
            selected.append(phrase_index)
            remaining.remove(phrase_index)
            for node_id, value in contribution_maps[phrase_index].items():
                old_coverage = coverage.get(node_id, 0.0)
                coverage[node_id] = (
                    old_coverage + value * (1.0 - old_coverage)
                )
            history.append({
                "step": len(selected),
                "selected_phrase_index": phrase_index,
                "stop_reason": None,
                "candidates": rows,
            })

        selected_step = {
            phrase_index: step
            for step, phrase_index in enumerate(selected, start=1)
        }
        selected_gain = {}
        for step in history:
            chosen = step.get("selected_phrase_index")
            if chosen is None:
                continue
            selected_gain[chosen] = next(
                (
                    row for row in step.get("candidates", [])
                    if row.get("phrase_index") == chosen
                ),
                {},
            )

        phrase_rows = []
        for phrase_index, phrase in enumerate(retrieval_phrases):
            raw_gain = self._eaes_marginal_gain(
                contribution_maps[phrase_index], coverage
            )
            remaining_gain = normalized_gain(raw_gain)
            is_selected = phrase_index in selected_step
            if is_selected:
                decision = "selected"
            elif len(selected) >= max_views:
                decision = "max_views_reached"
            elif raw_gain <= 1e-12:
                decision = "redundant_with_selected_views"
            else:
                decision = "below_marginal_gain_threshold"
            reported = selected_gain.get(phrase_index) or {
                "marginal_gain": raw_gain,
                "normalized_marginal_gain": remaining_gain,
            }
            phrase_rows.append({
                "phrase_index": phrase_index,
                "phrase": phrase,
                "fidelity": float(fidelities[phrase_index]),
                "selected": is_selected,
                "selection_step": selected_step.get(phrase_index),
                "marginal_gain": reported.get("marginal_gain", 0.0),
                "normalized_marginal_gain": reported.get(
                    "normalized_marginal_gain", 0.0
                ),
                "remaining_marginal_gain": raw_gain,
                "remaining_normalized_marginal_gain": remaining_gain,
                "decision": decision,
            })

        for ranking in rankings:
            for item in ranking:
                item["phrase_selected"] = (
                    item.get("phrase_index") in selected_step
                )
        return selected, {
            "phrases": phrase_rows,
            "selection_history": history,
            "min_views": min_views,
            "max_views": max_views,
            "gain_threshold": threshold,
            "initial_gain_scale": scale,
        }

    def select_eaes_views_by_meg(
            self,
            retrieval_phrases,
            fidelities,
            child_rankings,
            parent_rankings,
    ):
        """Select shared phrases independently for child and parent channels."""
        child_selected, child_diagnostics = (
            self._select_eaes_channel_views_by_meg(
                retrieval_phrases,
                fidelities,
                child_rankings,
                "memory_id",
            )
        )
        parent_selected, parent_diagnostics = (
            self._select_eaes_channel_views_by_meg(
                retrieval_phrases,
                fidelities,
                parent_rankings,
                "parent_id",
            )
        )
        return {
            "child_selected_phrase_indices": child_selected,
            "parent_selected_phrase_indices": parent_selected,
            "child": child_diagnostics,
            "parent": parent_diagnostics,
            "question_relevance_weight": float(
                config.EAES_QUESTION_RELEVANCE_WEIGHT
            ),
        }

    @staticmethod
    def fuse_eaes_channel_rankings(
            phrase_rankings,
            selected_phrase_indices,
            id_field,
    ):
        """Fuse a channel with the same saturating coverage used by MEG."""
        selected_set = set(selected_phrase_indices)
        fused = {}
        coverage_by_id = {}
        for phrase_index, ranking in enumerate(phrase_rankings):
            if phrase_index not in selected_set:
                continue
            for item in ranking:
                node_id = item.get(id_field)
                rank = int(item.get("phrase_rank") or 0)
                if not node_id or rank <= 0:
                    continue
                if node_id not in fused:
                    fused[node_id] = {
                        key: value for key, value in item.items()
                        if key not in {
                            "phrase_index", "phrase", "phrase_rank",
                            "phrase_similarity", "phrase_fidelity",
                            "support_potential", "phrase_selected",
                        }
                    }
                    fused[node_id]["phrase_matches"] = []
                contribution = max(
                    0.0,
                    min(
                        1.0,
                        float(item.get("support_potential") or 0.0),
                    ),
                )
                old_coverage = coverage_by_id.get(node_id, 0.0)
                coverage_by_id[node_id] = (
                    old_coverage
                    + contribution * (1.0 - old_coverage)
                )
                fused[node_id]["phrase_matches"].append({
                    "phrase_index": phrase_index,
                    "phrase": item.get("phrase"),
                    "phrase_rank": rank,
                    "phrase_similarity": float(
                        item.get("phrase_similarity") or 0.0
                    ),
                    "phrase_fidelity": float(
                        item.get("phrase_fidelity") or 0.0
                    ),
                    "question_relevance": float(
                        item.get("question_relevance") or 0.0
                    ),
                    "joint_relevance": float(
                        item.get("joint_relevance") or 0.0
                    ),
                    "support_potential": contribution,
                })
        rows = list(fused.values())
        for item in rows:
            node_id = item.get(id_field)
            fused_score = coverage_by_id.get(node_id, 0.0)
            item["fused_score"] = fused_score
            item["candidate_score"] = fused_score
            item["_candidate_score"] = fused_score
            item["candidate_sources"] = ["saturating_evidence_coverage"]
        rows.sort(key=lambda item: (
            -float(item.get("fused_score") or 0.0),
            str(item.get(id_field) or ""),
        ))
        top_score = float(rows[0]["fused_score"]) if rows else 0.0
        for rank, item in enumerate(rows, start=1):
            ratio = (
                float(item["fused_score"]) / top_score
                if top_score > 0 else 0.0
            )
            item["fused_rank"] = rank
            item["prefilter_rank"] = rank
            item["fused_score_ratio"] = ratio
        return rows

    @staticmethod
    def select_eaes_adaptive_prefix(
            candidates,
            max_k,
            mass_target=None,
            min_k=1,
            relevance_floor=None,
            ratio_threshold=None,
    ):
        """Keep the smallest fused-score prefix reaching target mass."""
        annotated = [dict(item) for item in candidates]
        if not annotated or max_k <= 0:
            return [], annotated
        mass_target = float(
            mass_target if mass_target is not None else 0.85
        )
        if not 0.0 < mass_target <= 1.0:
            raise ValueError("mass_target must be in (0, 1]")
        eligible = True
        if relevance_floor is not None:
            eligible = float(
                annotated[0].get("question_relevance") or 0.0
            ) >= float(relevance_floor)
        selected_k = 0
        safety_k = min(int(max_k), len(annotated))
        all_score_values = [
            max(
                0.0,
                float(
                    item.get("fused_score")
                    if item.get("fused_score") is not None
                    else item.get("candidate_score")
                    if item.get("candidate_score") is not None
                    else 0.0
                ),
            )
            for item in annotated
        ]
        total_score = sum(all_score_values)
        if eligible:
            if total_score <= 1e-12:
                selected_k = safety_k
            else:
                cumulative = 0.0
                selected_k = safety_k
                for index, score in enumerate(
                        all_score_values[:safety_k], start=1):
                    cumulative += score / total_score
                    if cumulative + 1e-12 >= mass_target:
                        selected_k = index
                        break
            selected_k = min(safety_k, max(min_k, selected_k))
        cumulative_mass = 0.0
        for index, item in enumerate(annotated):
            normalized_mass = (
                all_score_values[index] / total_score
                if total_score > 1e-12 else 1.0 / len(annotated)
            )
            cumulative_mass += normalized_mass
            item["normalized_mass"] = normalized_mass
            item["cumulative_mass"] = min(1.0, cumulative_mass)
            item["mass_target"] = mass_target
            item["inside_fused_safety_cap"] = index < safety_k
            item["inside_adaptive_prefix"] = index < selected_k
            item["adaptive_k"] = selected_k
        return annotated[:selected_k], annotated

    @staticmethod
    def _eaes_compact_phrase_rankings(rankings, id_field):
        return [
            {
                "phrase_index": phrase_index,
                "candidates": [
                    {
                        id_field: item.get(id_field),
                        "rank": item.get("phrase_rank"),
                        "similarity": item.get("phrase_similarity"),
                        "question_relevance": item.get("question_relevance"),
                        "joint_relevance": item.get("joint_relevance"),
                        "support_potential": item.get("support_potential"),
                        "selected": item.get("phrase_selected"),
                    }
                    for item in ranking
                ],
            }
            for phrase_index, ranking in enumerate(rankings)
        ]

    def retrieve_eaes_adaptive_views(
            self,
            retrieval_phrases,
            question_emb=None,
            question_text=None,
    ):
        """Run shared-view, independent-channel retrieval and adaptive depths."""
        phrases = [str(value).strip() for value in retrieval_phrases or []]
        _, child_relevance, parent_relevance = (
            self._eaes_question_relevance_maps(
                question_emb=question_emb,
                question_text=question_text,
            )
        )
        child_rankings = self.rank_eaes_children_per_phrase(
            phrases,
            top_k=config.EAES_PHRASE_INITIAL_TOP_K,
            question_relevance_by_id=child_relevance,
        )
        parent_rankings = self.rank_eaes_parents_per_phrase(
            phrases,
            top_k=config.EAES_PARENT_PHRASE_TOP_K,
            question_relevance_by_id=parent_relevance,
        )
        fidelities = self._score_eaes_view_rankings(
            child_rankings,
            parent_rankings,
            phrases,
            question_emb=question_emb,
            question_text=question_text,
        )
        selection = self.select_eaes_views_by_meg(
            phrases, fidelities, child_rankings, parent_rankings
        )
        child_selected_indices = selection[
            "child_selected_phrase_indices"
        ]
        parent_selected_indices = selection[
            "parent_selected_phrase_indices"
        ]
        all_indices = list(range(len(phrases)))
        child_probe = self.fuse_eaes_channel_rankings(
            child_rankings, all_indices, "memory_id"
        )
        parent_probe = self.fuse_eaes_channel_rankings(
            parent_rankings, all_indices, "parent_id"
        )
        child_fused = self.fuse_eaes_channel_rankings(
            child_rankings,
            child_selected_indices,
            "memory_id",
        )
        parent_fused = self.fuse_eaes_channel_rankings(
            parent_rankings,
            parent_selected_indices,
            "parent_id",
        )
        adaptive_children, child_scored = self.select_eaes_adaptive_prefix(
            child_fused,
            max_k=config.EAES_RERANK_LIMIT,
            mass_target=config.EAES_CHILD_MASS_TARGET,
            min_k=1,
        )
        adaptive_parents, parent_scored = self.select_eaes_adaptive_prefix(
            parent_fused,
            max_k=config.PARENT_TOP_K,
            mass_target=config.EAES_PARENT_MASS_TARGET,
            min_k=1,
            relevance_floor=config.PARENT_RELEVANCE_FLOOR,
        )
        for rank, parent in enumerate(adaptive_parents, start=1):
            parent["rank"] = rank
            parent["score"] = float(parent.get("fused_score") or 0.0)
        return {
            "child_probe_candidates": child_probe,
            "parent_probe_candidates": parent_probe,
            "selected_child_candidates": child_scored,
            "selected_parent_candidates": parent_scored,
            "adaptive_children": adaptive_children,
            "adaptive_parents": adaptive_parents,
            "child_rankings": child_rankings,
            "parent_rankings": parent_rankings,
            "phrase_retrieval": {
                **selection,
                "child_mass_target": float(config.EAES_CHILD_MASS_TARGET),
                "parent_mass_target": float(config.EAES_PARENT_MASS_TARGET),
                "child": {
                    **selection["child"],
                    "top_k": config.EAES_PHRASE_INITIAL_TOP_K,
                    "rankings": self._eaes_compact_phrase_rankings(
                        child_rankings, "memory_id"
                    ),
                },
                "parent": {
                    **selection["parent"],
                    "top_k": config.EAES_PARENT_PHRASE_TOP_K,
                    "rankings": self._eaes_compact_phrase_rankings(
                        parent_rankings, "parent_id"
                    ),
                },
            },
        }

    def diagnose_eaes_nodes_against_phrases(
            self,
            retrieval_phrases,
            selected_child_phrase_indices,
            selected_parent_phrase_indices,
            question_emb=None,
            question_text=None,
            child_memory_ids=None,
            parent_ids=None,
    ):
        """Score requested gold-linked nodes without changing retrieval state."""
        phrases = [str(value).strip() for value in retrieval_phrases or []]
        selected_children = set(selected_child_phrase_indices or [])
        selected_parents = set(selected_parent_phrase_indices or [])
        requested_children = set(child_memory_ids or [])
        requested_parents = set(parent_ids or [])
        self._prepare_eaes_tag_embeddings()
        self.prepare_eaes_retrieval_embeddings()
        if config.SEMANTIC_HIERARCHY:
            self.prepare_eaes_parent_embeddings()
        phrase_vectors = self._eaes_phrase_embeddings(phrases)
        question_vector = self._eaes_question_vector(
            question_emb, question_text
        )
        fidelities = [
            self._eaes_clipped_cosine(vector, question_vector)
            for vector in phrase_vectors
        ]
        beta = float(config.EAES_QUESTION_RELEVANCE_WEIGHT)
        rrf_k = float(config.EAES_PHRASE_RRF_K)

        child_rows = {}
        child_notes = sorted(
            self.memory.eaes_notes.values(), key=lambda note: note.memory_id
        )
        for phrase_index, (phrase, phrase_vector) in enumerate(
                zip(phrases, phrase_vectors)):
            scored = []
            for note in child_notes:
                tags, tag_vectors = self._eaes_tag_embedding_cache[note.memory_id]
                similarities = np.dot(tag_vectors, phrase_vector)
                tag_index = int(np.argmax(similarities))
                similarity = max(
                    0.0,
                    min(1.0, float(similarities[tag_index])),
                )
                node_vector = self._normalize_embedding_rows(
                    note.retrieval_embedding
                )[0]
                question_relevance = self._eaes_clipped_cosine(
                    node_vector, question_vector
                )
                local_relevance = self._eaes_joint_relevance(
                    question_relevance,
                    similarity,
                    beta,
                )
                scored.append((
                    local_relevance,
                    similarity,
                    question_relevance,
                    note.memory_id,
                    tags[tag_index],
                ))
            scored.sort(key=lambda row: (-row[0], row[3]))
            for full_rank, (
                    local_relevance, similarity, question_relevance,
                    memory_id, matched_tag
            ) in enumerate(
                    scored, start=1):
                if memory_id not in requested_children:
                    continue
                inside = full_rank <= int(config.EAES_PHRASE_INITIAL_TOP_K)
                diagnostic_support = self._eaes_phrase_node_contribution(
                    fidelities[phrase_index],
                    local_relevance,
                    full_rank,
                    rrf_k,
                )
                actual_support = diagnostic_support if inside else 0.0
                row = child_rows.setdefault(memory_id, {
                    "question_relevance": question_relevance,
                    "phrase_scores": [],
                })
                row["phrase_scores"].append({
                    "phrase_index": phrase_index,
                    "phrase": phrase,
                    "phrase_fidelity": fidelities[phrase_index],
                    "phrase_node_similarity": similarity,
                    "joint_relevance": local_relevance,
                    "matched_tag": matched_tag,
                    "full_channel_rank": full_rank,
                    "inside_top30": inside,
                    "inside_phrase_topk": inside,
                    "phrase_selected": phrase_index in selected_children,
                    "support_potential": actual_support,
                    "diagnostic_support_potential": diagnostic_support,
                    "rank_discount": (
                        (rrf_k + 1.0) / (rrf_k + full_rank)
                    ),
                })

        parent_rows = {}
        parents = sorted(
            self.memory.eaes_parent_nodes.values(),
            key=lambda parent: parent.parent_id,
        )
        for phrase_index, (phrase, phrase_vector) in enumerate(
                zip(phrases, phrase_vectors)):
            scored = []
            for parent in parents:
                if parent.retrieval_embedding is None:
                    continue
                vector = self._normalize_embedding_rows(
                    parent.retrieval_embedding
                )[0]
                similarity = self._eaes_clipped_cosine(
                    phrase_vector, vector
                )
                question_relevance = self._eaes_clipped_cosine(
                    vector, question_vector
                )
                local_relevance = self._eaes_joint_relevance(
                    question_relevance,
                    similarity,
                    beta,
                )
                scored.append((
                    local_relevance,
                    similarity,
                    question_relevance,
                    parent.parent_id,
                ))
            scored.sort(key=lambda row: (-row[0], row[3]))
            for full_rank, (
                    local_relevance, similarity, question_relevance, parent_id
            ) in enumerate(
                    scored, start=1):
                if parent_id not in requested_parents:
                    continue
                inside = full_rank <= int(config.EAES_PARENT_PHRASE_TOP_K)
                diagnostic_support = self._eaes_phrase_node_contribution(
                    fidelities[phrase_index],
                    local_relevance,
                    full_rank,
                    rrf_k,
                )
                actual_support = diagnostic_support if inside else 0.0
                row = parent_rows.setdefault(parent_id, {
                    "question_relevance": question_relevance,
                    "phrase_scores": [],
                })
                row["phrase_scores"].append({
                    "phrase_index": phrase_index,
                    "phrase": phrase,
                    "phrase_fidelity": fidelities[phrase_index],
                    "phrase_node_similarity": similarity,
                    "joint_relevance": local_relevance,
                    "full_channel_rank": full_rank,
                    "inside_top10": inside,
                    "inside_phrase_topk": inside,
                    "phrase_selected": phrase_index in selected_parents,
                    "support_potential": actual_support,
                    "diagnostic_support_potential": diagnostic_support,
                    "rank_discount": (
                        (rrf_k + 1.0) / (rrf_k + full_rank)
                    ),
                })
        return {"child": child_rows, "parent": parent_rows}

    def _eaes_parent_keyword_embeddings(self, query_plan, question_emb=None):
        keywords = [
            str(value).strip()
            for value in self._as_list((query_plan or {}).get("keywords"))
            if str(value).strip()
        ]
        if keywords:
            cache_key = tuple(keywords)
            vectors = self._eaes_parent_query_embedding_cache.get(cache_key)
            if vectors is None:
                with _EAES_EMBEDDING_LOCK:
                    vectors = self._normalize_embedding_rows(get_embedding(keywords))
                self._eaes_parent_query_embedding_cache[cache_key] = vectors
            return vectors, keywords
        if question_emb is not None:
            return self._normalize_embedding_rows(question_emb), [
                "__question_embedding_fallback__"
            ]
        return np.empty((0, 0), dtype=np.float32), []

    def _eaes_query_embeddings(self, query_plan, question_emb=None):
        query_attributes = [
            str(value).strip()
            for value in self._as_list((query_plan or {}).get("query_attributes"))
            if str(value).strip()
        ]
        if query_attributes:
            cache_key = tuple(query_attributes)
            vectors = self._eaes_query_embedding_cache.get(cache_key)
            if vectors is None:
                with _EAES_EMBEDDING_LOCK:
                    vectors = self._normalize_embedding_rows(get_embedding(query_attributes))
                self._eaes_query_embedding_cache[cache_key] = vectors
            return vectors, query_attributes
        if question_emb is not None:
            return self._normalize_embedding_rows(question_emb), ["__question_embedding_fallback__"]
        return np.empty((0, 0), dtype=np.float32), []

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

    def score_eaes_candidates(self, query_plan: Dict[str, Any], question_emb=None, limit: int = None,
                              include_rank: bool = False, exclude_memory_ids=None):
        if not isinstance(query_plan, dict):
            query_plan = {}
        self.prepare_eaes_retrieval_embeddings()
        query_vectors, query_attributes = self._eaes_query_embeddings(query_plan, question_emb)
        query_entities = self._as_list(query_plan.get("entities"))
        required_semantic_properties = []
        if config.EAES_SEMANTIC_SCORE:
            for value in self._as_list(query_plan.get("required_semantic_properties")):
                value = str(value or "").lower().strip()
                if value and value not in required_semantic_properties:
                    required_semantic_properties.append(value)

        entity_words = [self._eaes_words(entity) for entity in query_entities]
        excluded = set(exclude_memory_ids or [])
        scored = []
        for note in self.memory.eaes_notes.values():
            if note.memory_id in excluded:
                continue
            if note.retrieval_embedding is None or query_vectors.size == 0:
                continue
            note_vector = self._normalize_embedding_rows(note.retrieval_embedding)[0]
            similarities = np.dot(query_vectors, note_vector)
            best_index = int(np.argmax(similarities))
            raw_attribute_score = float(similarities[best_index])
            attribute_score = max(0.0, raw_attribute_score)

            note_entity_words = set()
            for entity in self._as_list(note.entities):
                note_entity_words |= self._eaes_words(entity)
            note_text_words = self._eaes_words(note.rewrite_content)

            if entity_words:
                entity_score = max(
                    self._eaes_overlap_score(words, note_entity_words | note_text_words)
                    for words in entity_words
                )
            else:
                entity_score = 0.2
            original_embedding_score = 0.0
            if question_emb is not None and note.embedding is not None:
                try:
                    original_embedding_score = float(
                        np.dot(question_emb.reshape(-1), note.embedding.reshape(-1))
                    )
                except Exception:
                    original_embedding_score = 0.0

            # Semantic properties live on EpisodeEvent, not EAESMemoryNote. A
            # missing event/field, disabled flag, or empty query requirement is
            # deliberately neutral. Exact intersections receive a capped,
            # tiered positive bonus; mismatches are never penalized or filtered.
            matched_semantic_properties = []
            semantic_match_count = 0
            semantic_bonus = 0.0
            if config.EAES_SEMANTIC_SCORE and required_semantic_properties:
                event = self.memory.episode_events.get(note.event_id)
                memory_properties = set(
                    self._as_list(getattr(event, "semantic_properties", []))
                ) if event is not None else set()
                matched_semantic_properties = [
                    value for value in required_semantic_properties
                    if value in memory_properties
                ]
                semantic_match_count = len(matched_semantic_properties)
                semantic_bonus = (
                    min(semantic_match_count, 3) * config.SEMANTIC_MATCH_WEIGHT
                )

            score = (
                2.0 * entity_score
                + 1.4 * attribute_score
                + 0.2 * original_embedding_score
                + semantic_bonus
            )
            scored.append((score, {
                **note.to_dict(include_raw=False),
                "score": round(score, 4),
                "score_parts": {
                    "entity": round(entity_score, 3),
                    "attribute": round(attribute_score, 4),
                    "attribute_embedding_raw": round(raw_attribute_score, 4),
                    "embedding": round(original_embedding_score, 3),
                    "semantic_match_count": semantic_match_count,
                    "matched_semantic_properties": matched_semantic_properties,
                    "semantic_bonus": round(semantic_bonus, 3),
                    "query_attribute_count": len(query_attributes),
                },
                "matched_query_attribute": query_attributes[best_index],
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = []
        for rank, (_, item) in enumerate(scored, start=1):
            if include_rank:
                item = {**item, "rank": rank}
            ranked.append(item)
        if limit is not None:
            return ranked[:limit]
        return ranked

    def retrieve_eaes_candidates(
            self, query_plan: Dict[str, Any], question_emb=None, limit: int = None,
            exclude_memory_ids=None
    ):
        limit = limit or config.EAES_CANDIDATE_LIMIT
        return self.score_eaes_candidates(
            query_plan,
            question_emb,
            limit=limit,
            include_rank=True,
            exclude_memory_ids=exclude_memory_ids,
        )

    def retrieve_eaes_rollback_children(
            self,
            query_phases,
            semantic_properties,
            entities,
            question_emb=None,
            exclude_memory_ids=None,
            limit=None,
    ):
        """Retrieve unseen children for one evidence-gap rollback round."""
        phases = [
            str(value).strip() for value in query_phases or []
            if str(value).strip()
        ]
        if not phases:
            return []

        self.prepare_eaes_retrieval_embeddings()
        phase_vectors = self._eaes_phrase_embeddings(phases)
        entity_words = [
            self._eaes_words(value) for value in entities or []
            if str(value).strip()
        ]
        required_properties = []
        for value in semantic_properties or []:
            normalized = str(value or "").lower().strip()
            if normalized and normalized not in required_properties:
                required_properties.append(normalized)

        excluded = set(exclude_memory_ids or [])
        scored = []
        for note in self.memory.eaes_notes.values():
            if note.memory_id in excluded or note.retrieval_embedding is None:
                continue

            note_vector = self._normalize_embedding_rows(
                note.retrieval_embedding
            )[0]
            phase_similarities = np.dot(phase_vectors, note_vector)
            best_phase_index = int(np.argmax(phase_similarities))
            raw_phase_score = float(phase_similarities[best_phase_index])
            phase_score = max(0.0, raw_phase_score)

            note_entity_words = set()
            for entity in self._as_list(note.entities):
                note_entity_words |= self._eaes_words(entity)
            note_text_words = self._eaes_words(note.rewrite_content)
            if entity_words:
                entity_score = max(
                    self._eaes_overlap_score(
                        words, note_entity_words | note_text_words
                    )
                    for words in entity_words
                )
            else:
                entity_score = 0.2

            question_score = 0.0
            if question_emb is not None and note.embedding is not None:
                try:
                    question_score = float(np.dot(
                        np.asarray(question_emb).reshape(-1),
                        np.asarray(note.embedding).reshape(-1),
                    ))
                except Exception:
                    question_score = 0.0

            event = self.memory.episode_events.get(note.event_id)
            memory_properties = {
                str(value or "").lower().strip()
                for value in self._as_list(
                    getattr(event, "semantic_properties", [])
                    if event is not None else []
                )
                if str(value or "").strip()
            }
            matched_properties = [
                value for value in required_properties
                if value in memory_properties
            ]
            semantic_bonus = 0.0
            if config.EAES_SEMANTIC_SCORE and required_properties:
                semantic_bonus = (
                    min(len(matched_properties), 3)
                    * config.SEMANTIC_MATCH_WEIGHT
                )

            score = (
                2.0 * entity_score
                + 1.4 * phase_score
                + 0.2 * question_score
                + semantic_bonus
            )
            scored.append({
                **note.to_dict(include_raw=False),
                "score": round(score, 4),
                "score_parts": {
                    "entity": round(entity_score, 3),
                    "rollback_phase": round(phase_score, 4),
                    "rollback_phase_raw": round(raw_phase_score, 4),
                    "question_embedding": round(question_score, 3),
                    "semantic_match_count": len(matched_properties),
                    "matched_semantic_properties": matched_properties,
                    "semantic_bonus": round(semantic_bonus, 3),
                },
                "matched_query_phase": phases[best_phase_index],
            })

        scored.sort(key=lambda item: (
            -float(item.get("score") or 0.0),
            str(item.get("memory_id") or ""),
        ))
        selected = scored[:limit] if limit is not None else scored
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected

    @staticmethod
    def _eaes_probability_entropy(probabilities):
        values = np.asarray(probabilities, dtype=np.float64)
        support_size = len(values)
        if support_size <= 1:
            return 0.0
        values = values[values > 0]
        if not len(values):
            return 0.0
        values = values / float(np.sum(values))
        return float(
            -np.sum(values * np.log(values)) / np.log(support_size)
        )

    @staticmethod
    def _eaes_js_divergence(left, right):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if len(left) == 0 or len(left) != len(right):
            return 0.0
        if float(np.sum(left)) <= 0 or float(np.sum(right)) <= 0:
            return 0.0
        left = left / float(np.sum(left))
        right = right / float(np.sum(right))
        middle = 0.5 * (left + right)

        def kl(source, target):
            mask = source > 0
            return float(np.sum(source[mask] * np.log(source[mask] / target[mask])))

        return (0.5 * kl(left, middle) + 0.5 * kl(right, middle)) / np.log(2.0)

    def score_eaes_parent_candidates(
            self, query_plan: Dict[str, Any], question_emb=None,
            exclude_parent_ids=None,
    ):
        """Score every parent rewrite; selection is handled by the router."""
        if not config.SEMANTIC_HIERARCHY:
            return []
        self.prepare_eaes_parent_embeddings()
        query_vectors, keywords = self._eaes_parent_keyword_embeddings(
            query_plan, question_emb
        )
        if query_vectors.size == 0:
            return []
        excluded = set(exclude_parent_ids or [])
        scored = []
        for parent in self.memory.eaes_parent_nodes.values():
            if parent.parent_id in excluded or parent.retrieval_embedding is None:
                continue
            vector = self._normalize_embedding_rows(parent.retrieval_embedding)[0]
            similarities = np.dot(query_vectors, vector)
            best_index = int(np.argmax(similarities))
            scored.append({
                **parent.to_reader_dict(),
                "raw_similarity": float(similarities[best_index]),
                "matched_keyword": keywords[best_index],
            })
        scored.sort(key=lambda item: (
            -float(item["raw_similarity"]), item["parent_id"]
        ))
        return scored

    def route_eaes_parent_candidates(
            self, query_plan, prefilter_children, question_emb=None,
    ):
        """Combine parent similarity and direct child support, then select 0-6."""
        parents = self.score_eaes_parent_candidates(query_plan, question_emb)
        if not parents:
            return [], {
                "breadth_value": float((query_plan or {}).get("breadth_value", 0.5)),
                "detail_value": float((query_plan or {}).get("detail_value", 0.5)),
                "parent_entropy": 0.0,
                "child_parent_dispersion": 0.0,
                "parent_child_jsd": 0.0,
                "retrieval_uncertainty": 0.0,
                "target_parent_mass": 0.0,
                "selected_parent_mass": 0.0,
                "selected_parent_k": 0,
                "parent_candidates": [],
            }

        raw = np.asarray([
            parent["raw_similarity"] for parent in parents
        ], dtype=np.float64)
        exp_raw = np.exp(raw - float(np.max(raw)))
        parent_probabilities = exp_raw / float(np.sum(exp_raw))
        parent_index = {
            parent["parent_id"]: index for index, parent in enumerate(parents)
        }
        child_support = np.zeros(len(parents), dtype=np.float64)
        child_mass = np.zeros(len(parents), dtype=np.float64)
        for child in prefilter_children or []:
            index = parent_index.get(child.get("parent_id"))
            if index is None:
                continue
            score = max(0.0, float(child.get("base_score") or 0.0))
            child_support[index] = max(child_support[index], score)
            child_mass[index] += score
        support_distribution = (
            child_support / float(np.sum(child_support))
            if float(np.sum(child_support)) > 0 else
            np.zeros(len(parents), dtype=np.float64)
        )
        child_distribution = (
            child_mass / float(np.sum(child_mass))
            if float(np.sum(child_mass)) > 0 else
            np.zeros(len(parents), dtype=np.float64)
        )
        posterior = 0.8 * parent_probabilities + 0.2 * support_distribution
        posterior = posterior / max(float(np.sum(posterior)), 1e-12)

        parent_entropy = self._eaes_probability_entropy(parent_probabilities)
        child_dispersion = self._eaes_probability_entropy(child_distribution)
        parent_child_jsd = self._eaes_js_divergence(
            parent_probabilities, support_distribution
        )
        uncertainty = min(
            1.0,
            0.4 * parent_entropy
            + 0.3 * child_dispersion
            + 0.3 * parent_child_jsd,
        )
        breadth_value = min(1.0, max(
            0.0, float((query_plan or {}).get("breadth_value", 0.5))
        ))
        detail_value = min(1.0, max(
            0.0, float((query_plan or {}).get("detail_value", 0.5))
        ))
        target_mass = min(0.95, 0.55 + 0.30 * breadth_value + 0.10 * uncertainty)

        rows = []
        for index, parent in enumerate(parents):
            rows.append({
                **parent,
                "parent_probability": float(parent_probabilities[index]),
                "child_support": float(child_support[index]),
                "posterior_score": float(posterior[index]),
                "selected": False,
            })
        rows.sort(key=lambda item: (
            -item["posterior_score"], -item["raw_similarity"], item["parent_id"]
        ))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        eligible = [
            row for row in rows
            if row["raw_similarity"] >= config.PARENT_RELEVANCE_FLOOR
        ]
        selected = []
        selected_mass = 0.0
        for row in eligible[:config.PARENT_TOP_K]:
            item = dict(row)
            item["selected"] = True
            item["rank"] = len(selected) + 1
            item["score"] = item["posterior_score"]
            selected.append(item)
            selected_mass += item["posterior_score"]
            if selected_mass >= target_mass:
                break
        selected_ids = {item["parent_id"] for item in selected}
        for row in rows:
            row["selected"] = row["parent_id"] in selected_ids
        return selected, {
            "breadth_value": breadth_value,
            "detail_value": detail_value,
            "parent_entropy": parent_entropy,
            "child_parent_dispersion": child_dispersion,
            "parent_child_jsd": parent_child_jsd,
            "retrieval_uncertainty": uncertainty,
            "target_parent_mass": target_mass,
            "selected_parent_mass": selected_mass,
            "selected_parent_k": len(selected),
            "parent_candidates": rows,
        }

    def retrieve_eaes_parent_candidates(
            self, query_plan: Dict[str, Any], question_emb=None, limit: int = None,
            exclude_parent_ids=None
    ):
        """Retrieve parents independently; this never filters child candidates."""
        scored = self.score_eaes_parent_candidates(
            query_plan, question_emb, exclude_parent_ids
        )
        selected = scored[:limit or config.PARENT_TOP_K]
        return [
            {
                **parent,
                "score": round(parent["raw_similarity"], 4),
                "rank": rank,
            }
            for rank, parent in enumerate(selected, start=1)
        ]

    def retrieve_eaes_rollback_parents(
            self,
            query_phases,
            exclude_parent_ids=None,
            limit=None,
    ):
        """Retrieve unseen parents by gap-phase similarity without a semantic bonus."""
        phases = [
            str(value).strip() for value in query_phases or []
            if str(value).strip()
        ]
        if not phases or not config.SEMANTIC_HIERARCHY:
            return []

        self.prepare_eaes_parent_embeddings()
        phase_vectors = self._eaes_phrase_embeddings(phases)
        excluded = set(exclude_parent_ids or [])
        scored = []
        for parent in self.memory.eaes_parent_nodes.values():
            if parent.parent_id in excluded or parent.retrieval_embedding is None:
                continue
            vector = self._normalize_embedding_rows(
                parent.retrieval_embedding
            )[0]
            similarities = np.dot(phase_vectors, vector)
            best_index = int(np.argmax(similarities))
            raw_similarity = float(similarities[best_index])
            scored.append({
                **parent.to_reader_dict(),
                "raw_similarity": raw_similarity,
                "score": round(raw_similarity, 4),
                "matched_query_phase": phases[best_index],
            })

        scored.sort(key=lambda item: (
            -float(item.get("raw_similarity") or 0.0),
            str(item.get("parent_id") or ""),
        ))
        selected = scored[:limit] if limit is not None else scored
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected

    def expand_eaes_raw_text(self, memory_ids: List[str]):
        expanded = []
        for mid in memory_ids[:config.EAES_RAW_EXPANSION_LIMIT]:
            note = self.memory.get_eaes_note(mid)
            if note is not None:
                expanded.append(note.to_dict(include_raw=True))
        return expanded

    def _parse_md(self,s: str):
        MD_RE = re.compile(r"^\s*(\d{2})-(\d{2})\s*$")  # MM-DD
        YMD_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
        m = MD_RE.match(s or "")
        if not m:
            return None
        mm, dd = int(m.group(1)), int(m.group(2))
        # basic validity check
        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            return None
        return mm, dd

    def _parse_ymd(self, s: str):
        try:
            return date.fromisoformat(s)
        except Exception:
            return None

    def _expand_to_ranges_by_timeline(self, start_str: str, end_str: str) -> List[Tuple[date, date]]:
        """
        Expand start_str/end_str into one or more (start_date, end_date) ranges:
          - if both are YYYY-MM-DD: return a single range
          - if both are MM-DD: generate one range per year seen in the timeline
          - other combinations (one with year, one without) -> treated as invalid, return empty
        """
        s_ymd = self._parse_ymd(start_str)
        e_ymd = self._parse_ymd(end_str)
        if s_ymd and e_ymd:
            return [(s_ymd, e_ymd)]

        s_md = self._parse_md(start_str)
        e_md = self._parse_md(end_str)
        if s_md and e_md:
            years = self.memory._years_from_timeline_keys()
            mm1, dd1 = s_md
            mm2, dd2 = e_md
            ranges = []
            for y in years:
                try:
                    s = date(y, mm1, dd1)
                    e = date(y, mm2, dd2)
                except ValueError:
                    # e.g. illegal dates like 02-30; skip that year
                    continue
                # if the month-day spans the year boundary (e.g. 12-20 ~ 01-10), it can be split into two parts; here a two-part scheme:
                if e < s:
                    # part 1: this year s ~ this year 12-31; part 2: this year 01-01 ~ e (needs next year; since timeline years backfill,
                    # the 'next year' key usually is not queried, so it is safer to keep only s~12-31, or check whether y+1 exists in years before adding the second part)
                    ranges.append((s, date(y, 12, 31)))
                    # if the timeline also has year y+1, add the second part
                    if (y + 1) in years:
                        ranges.append((date(y + 1, 1, 1), e))
                else:
                    ranges.append((s, e))
            return ranges

        # mixed case (one YMD, one MD): return empty here (inference logic could be added)
        return []


    def get_time_event(self, time):
        start_str, end_str = [x.strip() for x in time.split(",", 1)]
        time_ranges = self._expand_to_ranges_by_timeline(start_str, end_str)
        event_list = []
        for start_date, end_date in time_ranges:
            event_list.extend(self.memory.get_time_event(start_date, end_date))
        return event_list


    @staticmethod
    def _snake_norm(s: str) -> str:
        s = (s or "").lower().strip()
        # normalize punctuation/separators to spaces, then to underscores
        s = re.sub(r"[^\w\s:/-]+", " ", s)
        s = s.replace("/", " ").replace(":", " ")
        s = re.sub(r"[\s\-]+", "_", s)
        return s.strip("_")

    def extract_json_from_content(self, text: str):
        import json, re
        t = (text or "").strip()

        # strip ```json ... ``` fences
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M).strip()

        def _escape_inner_quotes_in_text_fields(s: str) -> str:
            """
            Only fix unescaped double quotes inside a "text": " ... " value -> \"
            Keep already-escaped content; do not touch other fields, to avoid over-replacing.
            """
            pattern = r'("text"\s*:\s*")((?:\\.|[^"\\])*)"'

            def _fix(m):
                body = m.group(2)
                # turn "unescaped" " inside body into \"
                body_fixed = re.sub(r'(?<!\\)"', r'\"', body)
                return m.group(1) + body_fixed + '"'

            return re.sub(pattern, _fix, s)

        def _loads_with_repair(s: str):
            # try a direct parse first
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            # on failure, repair the text field once, then retry
            s2 = _escape_inner_quotes_in_text_fields(s)
            return json.loads(s2)

        # prefer the JSON after an assistantfinal / final marker
        m = re.search(r"(assistantfinal|final)\s*{", t, flags=re.I)
        if m:
            start = m.end() - 1  # point at '{'
            # match braces with a stack to capture the full JSON block
            depth, i = 0, start
            while i < len(t):
                if t[i] == '{':
                    depth += 1
                elif t[i] == '}':
                    depth -= 1
                    if depth == 0:
                        block = t[start:i + 1]
                        return json.loads(block)
                i += 1
            raise ValueError("Unbalanced braces after assistantfinal/final")

        # fallback: grab the largest brace block in the text (not the last one)
        # still use brace matching to avoid grabbing an inner sub-object
        best = None
        stack = []
        for i, ch in enumerate(t):
            if ch == '{':
                stack.append(i)
            elif ch == '}' and stack:
                left = stack.pop()
                candidate = t[left:i + 1]
                # pick the longest (more likely the top-level object)
                if best is None or len(candidate) > len(best):
                    best = candidate
        if best:
            return json.loads(best)

        raise ValueError(f"No JSON object found. head={t[:300]!r}")


    def set_queried_events(self, query_events):
        self.queried_event = query_events

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
        if not isinstance(tags, list) or not 2 <= len(tags) <= 4:
            raise ValueError(
                f"EAES child {note.event_id} requires a tag array with 2-4 items"
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

        phrase_rankings = []
        for phrase_index, (phrase, phrase_vector) in enumerate(
                zip(phrases, phrase_vectors)):
            scored = []
            for note in notes:
                tags, tag_vectors = self._eaes_tag_embedding_cache[note.memory_id]
                similarities = np.dot(tag_vectors, phrase_vector)
                best_tag_index = int(np.argmax(similarities))
                scored.append((
                    note,
                    float(similarities[best_tag_index]),
                    tags[best_tag_index],
                    best_tag_index,
                ))
            scored = sorted(
                scored,
                key=lambda item: (-item[1], item[0].memory_id),
            )[:top_k]
            ranking = []
            for phrase_rank, (
                    note, similarity, matched_tag, matched_tag_index
            ) in enumerate(scored, start=1):
                ranking.append({
                    **note.to_dict(include_raw=False),
                    "tag": self._eaes_child_tags(note),
                    "phrase_index": phrase_index,
                    "phrase": phrase,
                    "phrase_rank": phrase_rank,
                    "phrase_similarity": similarity,
                    "matched_tag": matched_tag,
                    "matched_tag_index": matched_tag_index,
                })
            phrase_rankings.append(ranking)
        return phrase_rankings

    @staticmethod
    def fuse_eaes_phrase_rankings(
            phrase_rankings,
            rrf_k=10.0,
            protected_top_k=None,
            final_per_phrase=None,
    ):
        """Fuse already-selected dynamic phrase lists without hard Top10 pruning."""
        rrf_scores = {}
        max_similarities = {}
        phrase_matches = {}
        for ranking in phrase_rankings:
            for item in ranking:
                memory_id = item.get("memory_id")
                rank = int(item.get("phrase_rank") or 0)
                if not memory_id or rank <= 0:
                    continue
                rrf_scores[memory_id] = (
                    rrf_scores.get(memory_id, 0.0)
                    + 1.0 / (float(rrf_k) + rank)
                )
                max_similarities[memory_id] = max(
                    max_similarities.get(memory_id, -1.0),
                    float(item.get("phrase_similarity") or 0.0),
                )
                phrase_matches.setdefault(memory_id, []).append({
                    "phrase_index": item.get("phrase_index"),
                    "phrase": item.get("phrase"),
                    "phrase_rank": rank,
                    "phrase_similarity": float(
                        item.get("phrase_similarity") or 0.0
                    ),
                    "matched_tag": item.get("matched_tag"),
                    "matched_tag_index": item.get("matched_tag_index"),
                })

        fused_by_id = {}
        for ranking in phrase_rankings:
            for item in ranking:
                memory_id = item.get("memory_id")
                if not memory_id or memory_id in fused_by_id:
                    continue
                public_item = {
                    key: value for key, value in item.items()
                    if key not in {
                        "phrase_index", "phrase", "phrase_rank",
                        "phrase_similarity", "matched_tag",
                        "matched_tag_index",
                    }
                }
                public_item["max_phrase_similarity"] = max_similarities.get(
                    memory_id, 0.0
                )
                public_item["rrf_score"] = rrf_scores.get(memory_id, 0.0)
                public_item["phrase_matches"] = phrase_matches.get(
                    memory_id, []
                )
                if public_item["phrase_matches"]:
                    best_match = max(
                        public_item["phrase_matches"],
                        key=lambda match: match["phrase_similarity"],
                    )
                    public_item["matched_tag"] = best_match["matched_tag"]
                fused_by_id[memory_id] = public_item

        fused = list(fused_by_id.values())
        max_values = np.asarray([
            item["max_phrase_similarity"] for item in fused
        ], dtype=np.float32)
        rrf_values = np.asarray([
            item["rrf_score"] for item in fused
        ], dtype=np.float32)

        def minmax(values):
            if len(values) == 0:
                return values
            low, high = float(np.min(values)), float(np.max(values))
            if high - low <= 1e-12:
                return np.ones_like(values) if high > 0 else np.zeros_like(values)
            return (values - low) / (high - low)

        max_norm = minmax(max_values)
        rrf_norm = minmax(rrf_values)
        for item, normalized_similarity, normalized_rrf in zip(
                fused, max_norm, rrf_norm):
            base_score = 0.75 * float(normalized_similarity) + 0.25 * float(
                normalized_rrf
            )
            item["base_score"] = base_score
            item["candidate_score"] = base_score
            item["_candidate_score"] = base_score
            item["candidate_sources"] = ["global_phrase"]
        fused.sort(key=lambda item: (
            -float(item.get("candidate_score") or 0.0),
            str(item.get("memory_id") or ""),
        ))
        for rank, item in enumerate(fused, start=1):
            item["prefilter_rank"] = rank

        diagnostics = {
            "global_candidate_ids": [item.get("memory_id") for item in fused],
        }
        return fused, diagnostics

    @staticmethod
    def select_eaes_dynamic_phrase_rankings(phrase_rankings):
        selected_rankings = []
        phrase_diagnostics = []
        temperature = float(config.EAES_PHRASE_SOFTMAX_TEMPERATURE)
        target_mass = float(config.EAES_PHRASE_TARGET_MASS)
        for phrase_index, ranking in enumerate(phrase_rankings):
            similarities = np.asarray([
                float(item.get("phrase_similarity") or 0.0)
                for item in ranking
            ], dtype=np.float64)
            if len(similarities):
                logits = (similarities - float(np.max(similarities))) / temperature
                weights = np.exp(logits)
                probabilities = weights / max(float(np.sum(weights)), 1e-12)
                cumulative = np.cumsum(probabilities)
                mass_k = int(np.searchsorted(cumulative, target_mass) + 1)
                selected_k = min(
                    len(ranking),
                    max(
                        config.EAES_PHRASE_MIN_TOP_K,
                        min(config.EAES_PHRASE_MAX_TOP_K, mass_k),
                    ),
                )
                mass_at_15 = float(cumulative[min(14, len(cumulative) - 1)])
                selected_mass = float(cumulative[selected_k - 1])
            else:
                probabilities = np.asarray([], dtype=np.float64)
                selected_k = 0
                mass_at_15 = 0.0
                selected_mass = 0.0
            selected = list(ranking[:selected_k])
            selected_rankings.append(selected)
            phrase_diagnostics.append({
                "phrase_index": phrase_index,
                "selected_k": selected_k,
                "mass_at_15": mass_at_15,
                "selected_mass": selected_mass,
                "candidates": [
                    {
                        "memory_id": item.get("memory_id"),
                        "rank": item.get("phrase_rank"),
                        "similarity": item.get("phrase_similarity"),
                        "probability": float(probabilities[index]),
                        "rrf_contribution": 1.0 / (
                            float(config.EAES_PHRASE_RRF_K)
                            + int(item.get("phrase_rank") or 0)
                        ),
                    }
                    for index, item in enumerate(selected)
                ],
            })
        return selected_rankings, phrase_diagnostics

    def _rescore_eaes_global_phrase_pool(self, candidates, retrieval_phrases):
        """Compute the maximum phrase-to-any-tag score without new embeddings."""
        if not candidates:
            return []
        phrase_vectors = self._eaes_phrase_embeddings(retrieval_phrases)
        for candidate in candidates:
            tags, tag_vectors = self._eaes_tag_embedding_cache[
                candidate["memory_id"]
            ]
            similarities = np.dot(phrase_vectors, tag_vectors.T)
            best_flat_index = int(np.argmax(similarities))
            _, best_tag_index = np.unravel_index(
                best_flat_index, similarities.shape
            )
            candidate["max_phrase_similarity"] = float(
                np.max(similarities)
            )
            candidate["matched_tag"] = tags[int(best_tag_index)]
        similarities = np.asarray([
            candidate["max_phrase_similarity"] for candidate in candidates
        ], dtype=np.float64)
        rrf_scores = np.asarray([
            float(candidate.get("rrf_score") or 0.0)
            for candidate in candidates
        ], dtype=np.float64)

        def minmax(values):
            low, high = float(np.min(values)), float(np.max(values))
            if high - low <= 1e-12:
                return np.ones_like(values) if high > 0 else np.zeros_like(values)
            return (values - low) / (high - low)

        for candidate, sim_norm, rrf_norm in zip(
                candidates, minmax(similarities), minmax(rrf_scores)):
            base_score = 0.75 * float(sim_norm) + 0.25 * float(rrf_norm)
            candidate["base_score"] = base_score
            candidate["candidate_score"] = base_score
            candidate["_candidate_score"] = base_score
        candidates.sort(key=lambda item: (
            -float(item.get("candidate_score") or 0.0),
            str(item.get("memory_id") or ""),
        ))
        for rank, candidate in enumerate(candidates, start=1):
            candidate["prefilter_rank"] = rank
        return candidates

    def retrieve_eaes_phrase_candidates(
            self,
            retrieval_phrases,
            exclude_memory_ids=None,
            include_diagnostics=False,
    ):
        rankings = self.rank_eaes_children_per_phrase(
            retrieval_phrases,
            top_k=config.EAES_PHRASE_INITIAL_TOP_K,
            exclude_memory_ids=exclude_memory_ids,
        )
        selected_rankings, phrase_diagnostics = (
            self.select_eaes_dynamic_phrase_rankings(rankings)
        )
        candidates, diagnostics = self.fuse_eaes_phrase_rankings(
            selected_rankings,
            rrf_k=config.EAES_PHRASE_RRF_K,
        )
        candidates = self._rescore_eaes_global_phrase_pool(
            candidates, retrieval_phrases
        )
        diagnostics["global_candidate_ids"] = [
            candidate.get("memory_id") for candidate in candidates
        ]
        diagnostics["phrases"] = phrase_diagnostics
        if include_diagnostics:
            return candidates, diagnostics
        return candidates

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
            self, query_plan, global_children, question_emb=None,
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
        for child in global_children or []:
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

    def retrieve_eaes_parent_local_children(
            self, retrieval_phrases, selected_parents, detail_value,
    ):
        """Add a small tag-only child scan inside every selected parent."""
        if not selected_parents or not retrieval_phrases:
            return [], {"per_parent_k": 0, "parents": []}
        self._prepare_eaes_tag_embeddings()
        phrase_vectors = self._eaes_phrase_embeddings(retrieval_phrases)
        local_k = 2 + int(round(2 * min(1.0, max(0.0, float(detail_value)))))
        local_candidates = []
        diagnostics = []
        seen = set()
        for selected_parent in selected_parents:
            parent_id = selected_parent.get("parent_id")
            notes = sorted(
                (
                    note for note in self.memory.eaes_notes.values()
                    if note.parent_id == parent_id
                ),
                key=lambda note: note.memory_id,
            )
            scored = []
            for note in notes:
                tags, tag_vectors = self._eaes_tag_embedding_cache[
                    note.memory_id
                ]
                similarities = np.dot(phrase_vectors, tag_vectors.T)
                best_flat_index = int(np.argmax(similarities))
                _, best_tag_index = np.unravel_index(
                    best_flat_index, similarities.shape
                )
                scored.append((
                    float(np.max(similarities)),
                    note,
                    tags[int(best_tag_index)],
                ))
            scored.sort(key=lambda item: (-item[0], item[1].memory_id))
            parent_ids = []
            for similarity, note, matched_tag in scored[:local_k]:
                parent_ids.append(note.memory_id)
                if note.memory_id in seen:
                    continue
                seen.add(note.memory_id)
                local_candidates.append({
                    **note.to_dict(include_raw=False),
                    "tag": self._eaes_child_tags(note),
                    "matched_tag": matched_tag,
                    "max_phrase_similarity": similarity,
                    "rrf_score": 0.0,
                    "candidate_sources": ["parent_local"],
                })
            diagnostics.append({
                "parent_id": parent_id,
                "selected_memory_ids": parent_ids,
            })
        return local_candidates, {"per_parent_k": local_k, "parents": diagnostics}

    @staticmethod
    def merge_eaes_hierarchical_candidates(
            global_candidates, local_candidates, selected_parents, limit=None,
    ):
        """Union global/local children, rescore once, and cap the reranker pool."""
        limit = limit or config.EAES_PHRASE_UNION_LIMIT
        merged = {}
        global_ids = []
        for candidate in global_candidates or []:
            item = dict(candidate)
            memory_id = item.get("memory_id")
            if not memory_id:
                continue
            global_ids.append(memory_id)
            item["candidate_sources"] = list(dict.fromkeys(
                item.get("candidate_sources") or ["global_phrase"]
            ))
            merged[memory_id] = item
        local_ids = []
        local_added_ids = []
        for candidate in local_candidates or []:
            memory_id = candidate.get("memory_id")
            if not memory_id:
                continue
            local_ids.append(memory_id)
            if memory_id in merged:
                item = merged[memory_id]
                item["candidate_sources"] = list(dict.fromkeys(
                    list(item.get("candidate_sources") or []) + ["parent_local"]
                ))
                item["max_phrase_similarity"] = max(
                    float(item.get("max_phrase_similarity") or 0.0),
                    float(candidate.get("max_phrase_similarity") or 0.0),
                )
            else:
                item = dict(candidate)
                item["rrf_score"] = 0.0
                item["candidate_sources"] = ["parent_local"]
                merged[memory_id] = item
                local_added_ids.append(memory_id)

        items = list(merged.values())
        similarities = np.asarray([
            float(item.get("max_phrase_similarity") or 0.0) for item in items
        ], dtype=np.float64)
        rrf_scores = np.asarray([
            float(item.get("rrf_score") or 0.0) for item in items
        ], dtype=np.float64)

        def minmax(values):
            if len(values) == 0:
                return values
            low, high = float(np.min(values)), float(np.max(values))
            if high - low <= 1e-12:
                return np.ones_like(values) if high > 0 else np.zeros_like(values)
            return (values - low) / (high - low)

        parent_scores = {
            item.get("parent_id"): float(item.get("posterior_score") or 0.0)
            for item in selected_parents or []
        }
        for item, sim_norm, rrf_norm in zip(
                items, minmax(similarities), minmax(rrf_scores)):
            base_score = 0.75 * float(sim_norm) + 0.25 * float(rrf_norm)
            parent_boost = 0.1 * parent_scores.get(item.get("parent_id"), 0.0)
            candidate_score = base_score + parent_boost
            item["base_score"] = base_score
            item["parent_boost"] = parent_boost
            item["candidate_score"] = candidate_score
            item["_candidate_score"] = candidate_score
        items.sort(key=lambda item: (
            -float(item.get("candidate_score") or 0.0),
            -float(item.get("max_phrase_similarity") or 0.0),
            str(item.get("memory_id") or ""),
        ))
        prefilter = items[:limit]
        for rank, item in enumerate(prefilter, start=1):
            item["prefilter_rank"] = rank
        return prefilter, {
            "global_candidate_ids": global_ids,
            "local_candidate_ids": local_ids,
            "local_added_ids": local_added_ids,
            "global_plus_local_ids": [item.get("memory_id") for item in items],
            "prefilter_candidate_ids": [item.get("memory_id") for item in prefilter],
            "dropped_by_pool_limit_ids": [
                item.get("memory_id") for item in items[limit:]
            ],
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

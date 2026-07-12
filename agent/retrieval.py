import json
import logging
import re
from typing import Any, Dict, List

import numpy as np

from common import config
from common.utils import topk_answers_by_similarity
from prompts.prompts import Prompts

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
        query_plan = retrieval.get("query_plan") or {}
        ranked = self.memory_controller.score_eaes_candidates(
            query_plan, question_emb, limit=None, include_rank=True)
        by_memory_id = {item.get("memory_id"): item for item in ranked if item.get("memory_id")}
        by_event_id = {item.get("event_id"): item for item in ranked if item.get("event_id")}
        prefilter_candidates = self._as_list(retrieval.get("prefilter_candidates"))
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

        diagnostics = {
            "prefilter_limit": config.EAES_CANDIDATE_LIMIT,
            "rerank_limit": config.EAES_RERANK_LIMIT,
            "total_scored_memories": len(ranked),
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
                rank = scored.get("rank") if scored else None
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
                            "score": neighbor.get("score"),
                            "score_parts": neighbor.get("score_parts"),
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
                    "candidate_score": scored.get("score") if scored else None,
                    "score_parts": scored.get("score_parts") if scored else None,
                    "entities": note.entities if note is not None else [],
                    "attribute_paths": note.attribute_paths if note is not None else [],
                    "rewrite_content": note.rewrite_content if note is not None else None,
                    "nearby_ranked_candidates": neighbor_window,
                    "drop_reason": (
                        "not_built_in_eaes_memory" if note is None else
                        "not_scored_by_query_attributes" if scored is None else
                        "rank_beyond_prefilter_topk" if rank and rank > config.EAES_CANDIDATE_LIMIT else
                        "dropped_by_llm_reranker" if memory_id not in reranked_by_memory_id else
                        "inside_llm_topk"
                    ),
                })
            covered_by_retrieval = any(
                entry["in_retrieved_candidates"] for entry in memory_entries
            )
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
            elif covered_by_retrieval:
                origin_drop_reason = "inside_llm_topk"
            elif rank_values and min(rank_values) > config.EAES_CANDIDATE_LIMIT:
                origin_drop_reason = "rank_beyond_prefilter_topk"
            elif rank_values:
                origin_drop_reason = "dropped_by_llm_reranker"
            else:
                origin_drop_reason = "not_scored_by_query_attributes"

            diagnostics["gold_origins"].append({
                "origin": origin,
                "event_ids": event_ids,
                "memory_ids": memory_ids,
                "covered_by_retrieval": covered_by_retrieval,
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
            embedding_candidates = self.memory_controller.retrieve_eaes_candidates(
                query_plan, question_emb, limit=config.EAES_CANDIDATE_LIMIT)
            candidates = self.rerank_eaes_candidates(question, query_plan, embedding_candidates)
            event_ids = self._unique_keep_order([c.get("event_id") for c in candidates])
            origins = self._unique_keep_order(
                [c.get("origin") for c in candidates] or self._origins_for_event_ids(event_ids)
            )
            if not origins:
                origins = self._origins_for_event_ids(event_ids)
            prefilter_event_ids = self._unique_keep_order(
                [c.get("event_id") for c in embedding_candidates]
            )
            prefilter_origins = self._unique_keep_order(
                [c.get("origin") for c in embedding_candidates]
                or self._origins_for_event_ids(prefilter_event_ids)
            )
            return {
                "mode": "eaes",
                "query_plan": query_plan,
                "retrieved_event_ids": event_ids,
                "retrieved_origins": origins,
                "candidates": candidates,
                "prefilter_event_ids": prefilter_event_ids,
                "prefilter_origins": prefilter_origins,
                "prefilter_candidates": embedding_candidates,
            }

        self.memory_controller.question_emb = question_emb
        question_keys = self.extract_question_keys(question, question_emb)
        self.memory_controller.set_queried_keywords(question_keys.get("keywords"))
        query_answers = self.memory_controller.evaluate_relations_over_graph(question_keys.get("keywords"))

        question_time = override_question_time if override_question_time else question_keys.get("question_time")
        if question_time:
            question_time = self.get_time(question_time)

        ans_input = {}
        retrieved_ids = []
        retrieved_texts = []
        _lm_time_done = False
        if question_time:
            if config.dataset == "LM":
                ans_input, retrieved_ids = self.answer_question_with_time_lm(
                    question, question_emb, query_answers, question_time, question_keys, category)
                _lm_time_done = len(ans_input.get("key_sentences", [])) > 0
                retrieved_texts = ans_input.get("key_sentences", [])
            else:
                ans_input = self.answer_question_with_time(question, question_time, question_keys, category)
                retrieved_texts = ans_input.get("similar_sentence", [])
                retrieved_ids = self._event_ids_from_texts(retrieved_texts)

        if (not _lm_time_done) and ((not question_time) or len(ans_input.get("similar_sentence", [])) == 0):
            similar_sentence = []
            similar_sentence_embs = []
            similar_sentence_ids = []

            def _collect(m):
                similar_sentence.append(m.get("value_id") + ":" + m.get("value_text"))
                similar_sentence_embs.append(self.memory.episode_events[m.get("value_id")].embedding)
                similar_sentence_ids.append(m.get("value_id"))

            for m in query_answers.get("full_matches"):
                _collect(m)
            for m in query_answers.get("partial_matches"):
                if m.get("matched_keys") >= 2:
                    _collect(m)

            if len(similar_sentence_embs) > config.K1:
                similar_sentence_embs = np.vstack(similar_sentence_embs)
                top_ids, _, top_embs, top_texts = topk_answers_by_similarity(
                    question_emb, similar_sentence_embs, similar_sentence_ids,
                    k=config.K1, answer_texts=similar_sentence)
            else:
                top_texts = similar_sentence
                top_ids = similar_sentence_ids
                top_embs = similar_sentence_embs

            seen_prefixes = set()
            pattern = re.compile(r'^(.+)-\d+$')
            deduplicated_ids, deduplicated_texts, deduplicated_embs = [], [], []
            for idx, tid in enumerate(top_ids):
                match = pattern.match(tid)
                prefix = match.group(1) if match else tid
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                deduplicated_ids.append(tid)
                deduplicated_texts.append(top_texts[idx])
                if len(top_embs) > 0:
                    deduplicated_embs.append(top_embs[idx])
            top_ids, top_texts, top_embs = deduplicated_ids, deduplicated_texts, deduplicated_embs

            top_topic_texts = self.select_topic(question_emb)

            if len(top_ids) > config.K2:
                top_ids, top_embs, top_texts = self.select_finegrained_sentence(
                    question, question_emb, top_texts, top_ids, top_embs)
                top_ids2, top_embs2, top_texts2 = self.select_finegrained_sentence_sort(
                    question, question_emb, top_texts, top_ids, top_embs)
                merged = {}
                for i, t in zip(top_ids + top_ids2, top_texts + top_texts2):
                    merged.setdefault(i, t)
                top_ids = list(merged.keys())
                top_texts = list(merged.values())

            queried_similar_sentence_ids = self.extract_id_prefixes(top_ids)
            key_candidates, key_tag_ids, key_tag_sentences = self.select_key_tag(
                question, question_keys, queried_similar_sentence_ids)

            retrieved_ids = self._unique_keep_order(top_ids + key_tag_ids)
            retrieved_texts = top_texts + key_tag_sentences
            ans_input = {
                "question": question,
                "key_sentences": top_texts,
                "keys_candidates": key_candidates,
                "key_tag_sentences": key_tag_sentences,
                "similar_topic": top_topic_texts
            }
            if lm_current_date:
                ans_input["current_date"] = lm_current_date

        graph_ids = self._unique_keep_order(retrieved_ids)
        graph_texts = list(retrieved_texts or [])
        graph_origins = self._origins_for_event_ids(graph_ids)
        dense_ids, dense_scores, dense_texts, dense_origins = self._dense_episode_retrieval(question_emb)
        combined_ids = self._unique_keep_order(graph_ids + dense_ids)
        combined_texts = graph_texts + [t for i, t in zip(dense_ids, dense_texts) if i not in set(graph_ids)]
        origins = self._origins_for_event_ids(combined_ids)
        return {
            "mode": "graph",
            "question_keys": question_keys,
            "retrieved_event_ids": combined_ids,
            "retrieved_origins": origins,
            "retrieved_texts": combined_texts,
            "graph_event_ids": graph_ids,
            "graph_origins": graph_origins,
            "graph_texts": graph_texts,
            "dense_event_ids": dense_ids,
            "dense_origins": dense_origins,
            "dense_scores": dense_scores,
            "dense_texts": dense_texts,
            "answer_input": ans_input,
        }


    def answer_question(self, question: str, category=0, question_emb=None, override_question_time=None, lm_current_date=None) -> Dict[str, Any]:
        if config.EAES_MODE:
            return self.answer_question_eaes(question, category, question_emb, lm_current_date)
        self.memory_controller.question_emb = question_emb
        question_keys = self.extract_question_keys(question, question_emb)
        self.memory_controller.set_queried_keywords(question_keys.get("keywords"))
        query_answers = self.memory_controller.evaluate_relations_over_graph(question_keys.get("keywords"))
        # if an override is given (LM temporal question_date anchor) use it; otherwise use the LLM-extracted time
        if override_question_time:
            question_time = override_question_time
        else:
            question_time = question_keys.get("question_time")

        if question_time:
            question_time = self.get_time(question_time)

        _lm_time_done = False
        if question_time:
            if config.dataset == "LM":
                # LM temporal handling
                ans_input, _ = self.answer_question_with_time_lm(
                    question, question_emb, query_answers, question_time, question_keys, category)
                _lm_time_done = len(ans_input.get("key_sentences", [])) > 0
            else:
                # locomo temporal handling
                ans_input = self.answer_question_with_time(question, question_time, question_keys, category)

        if (not _lm_time_done) and ((not question_time) or len(ans_input.get("similar_sentence", [])) == 0):
            similar_sentence = []
            full_ma = query_answers.get("full_matches")
            part_ma = query_answers.get("partial_matches")
            similar_sentence_embs = []
            similar_sentence_ids = []

            def _collect(m):
                similar_sentence.append(m.get("value_id") + ":" + m.get("value_text"))
                similar_sentence_embs.append(self.memory.episode_events[m.get("value_id")].embedding)
                similar_sentence_ids.append(m.get("value_id"))

            for m in full_ma:
                _collect(m)
            for m in part_ma:
                if m.get("matched_keys") >= 2:
                    _collect(m)


            # coarse select
            if len(similar_sentence_embs) > config.K1:
                similar_sentence_embs = np.vstack(similar_sentence_embs)
                top_ids, _, top_embs, top_texts = topk_answers_by_similarity(question_emb, similar_sentence_embs, similar_sentence_ids,
                                                                                  k=config.K1, answer_texts=similar_sentence)
            else:
                top_texts = similar_sentence
                top_ids = similar_sentence_ids
                top_embs = similar_sentence_embs

            # dedup: keep only the first item per distinct id-prefix
            seen_prefixes = set()
            pattern = re.compile(r'^(.+)-\d+$')
            deduplicated_ids, deduplicated_texts, deduplicated_embs = [], [], []
            for idx, tid in enumerate(top_ids):
                match = pattern.match(tid)
                prefix = match.group(1) if match else tid
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                deduplicated_ids.append(tid)
                deduplicated_texts.append(top_texts[idx])
                if len(top_embs) > 0:
                    deduplicated_embs.append(top_embs[idx])
            top_ids, top_texts, top_embs = deduplicated_ids, deduplicated_texts, deduplicated_embs

            top_topic_texts = self.select_topic(question_emb)

            if len(top_ids) > config.K2:
                top_ids, top_embs, top_texts = self.select_finegrained_sentence(question, question_emb, top_texts, top_ids, top_embs)
                top_ids2, top_embs2, top_texts2 = self.select_finegrained_sentence_sort(question, question_emb, top_texts, top_ids, top_embs)
                all_ids = top_ids + top_ids2
                all_texts = top_texts + top_texts2

                # dedup via dict (key = id), keeping the first occurrence
                merged = {}
                for i, t in zip(all_ids, all_texts):
                    merged.setdefault(i, t)
                top_ids = list(merged.keys())
                top_texts = list(merged.values())

            queried_similar_sentence_ids = self.extract_id_prefixes(top_ids)
            key_candidates, _, key_tag_sentences = self.select_key_tag(question,question_keys,queried_similar_sentence_ids)

            self.memory_controller.set_queried_events(top_ids)
            ans_input = {
                "question": question,
                "key_sentences": top_texts,
                "keys_candidates": key_candidates,
                "key_tag_sentences": key_tag_sentences,
                "similar_topic": top_topic_texts
            }
            # inject current_date (=question_date) on the main path as the "now" anchor for temporal questions
            if lm_current_date:
                ans_input["current_date"] = lm_current_date

        if str(category) == "2" and str(config.dataset).lower() == "locomo":
            ans_input["question"] = (
                ans_input["question"] + " " + Prompts.TEMPORAL_ANSWER_POLICY
            )
        elif category == 3:
            # open-ended questions are asked to give supporting reasons
            ans_input["question"] = ans_input["question"] + (" No extra explanations in 'answer'. Give reasons with original text in 'reason'. ")
        # LM uses the LM ANSWER system prompt (how-many/temporal rules + tool navigation); locomo uses the default one
        _answer_prompt = Prompts.ANSWER_SYSTEM_TOOL_PROMPT_LM if config.dataset == "LM" else Prompts.ANSWER_SYSTEM_TOOL_PROMPT
        ans_messages, evidence_support = self._chat_with_tools(
            _answer_prompt, ans_input, category)
        support_origin = self.memory.get_support_origin(evidence_support)
        return ans_messages, support_origin

    @staticmethod
    def _key_tokens(text: str):
        stop = {
            "what", "which", "would", "could", "should", "likely", "the", "and", "for", "with",
            "from", "into", "about", "that", "this", "have", "has", "had", "her", "his", "their",
            "your", "you", "she", "him", "who", "when", "where", "why", "how", "did", "does", "educaton",
        }
        return {
            t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in stop
        }

    def _legacy_extract_question_keys(self, questions: str):
        question_prompt = Prompts.extract_question_key_prompt(json.dumps(questions, ensure_ascii=False))
        question_out = self.llm.chat_text(
            messages=[{"role": "system", "content": Prompts.QUESTION_KEY_SYSTEM_PROMPT},
                      {"role": "user", "content": question_prompt}],
            model=config.RE_MODEL
        )
        return question_out

    def _candidate_key_records(self, question: str, question_emb=None) -> List[Dict[str, Any]]:
        q_tokens = self._key_tokens(question)
        scores: Dict[str, float] = {}
        dense_ids, _, _, _ = self._dense_episode_retrieval(question_emb, k=config.KEY_CANDIDATE_DENSE_K)

        for rank, event_id in enumerate(dense_ids):
            weight = 1.0 / (rank + 1)
            for key in self.memory.event_to_keys.get(event_id, set()):
                scores[key] = scores.get(key, 0.0) + weight

        for key in self.memory.keys.keys():
            key_tokens = self._key_tokens(key)
            if not key_tokens:
                continue
            overlap = len(q_tokens & key_tokens)
            if overlap:
                scores[key] = scores.get(key, 0.0) + 2.0 + overlap / max(len(key_tokens), 1)

        ordered_keys = sorted(scores, key=lambda k: (scores[k], k), reverse=True)[:config.KEY_CANDIDATE_LIMIT]
        records = []
        dense_set = set(dense_ids)
        for key in ordered_keys:
            examples = []
            linked_ids = [eid for eid in dense_ids if key in self.memory.event_to_keys.get(eid, set())]
            if not linked_ids:
                linked_ids = [vid for vid, _ in list(self.memory.key_to_values.get(key, set()))[:3]]
            for event_id in linked_ids:
                if event_id in self.memory.episode_events:
                    text = self.memory.episode_events[event_id].text
                    examples.append(f"{event_id}: {text[:180]}")
                if len(examples) >= 2:
                    break
            records.append({
                "key": key,
                "tags": self.memory.get_tag_list(key)[:8],
                "score": round(scores[key], 4),
                "source": "dense" if any(eid in dense_set for eid in linked_ids) else "lexical",
                "examples": examples,
            })
        return records

    def _select_question_keys_from_inventory(self, questions: str, question_emb=None):
        candidates = self._candidate_key_records(questions, question_emb)
        if not candidates:
            return self._legacy_extract_question_keys(questions)

        prompt = Prompts.select_question_key_prompt(
            questions,
            json.dumps(candidates, ensure_ascii=False)
        )
        out = self.llm.chat_text(
            messages=[
                {"role": "system", "content": Prompts.QUESTION_KEY_INVENTORY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=config.RE_MODEL
        )
        if not isinstance(out, dict):
            return self._legacy_extract_question_keys(questions)

        valid = {c["key"] for c in candidates}
        selected = []
        seen = set()
        for item in self._as_list(out.get("keywords")):
            key = item.get("id") if isinstance(item, dict) else str(item)
            if key in valid and key not in seen:
                seen.add(key)
                selected.append({"id": key, "alternatives": []})

        if not selected:
            logger.info("inventory key selection returned no valid keys; falling back to extracted question keys.")
            return self._legacy_extract_question_keys(questions)

        return {
            "question_time": out.get("question_time", ""),
            "keywords": selected,
            "candidate_count": len(candidates),
            "query_key_mode": "inventory",
        }

    def extract_question_keys(self, questions: str, question_emb=None):
        if config.QUERY_KEY_MODE == "inventory":
            return self._select_question_keys_from_inventory(questions, question_emb)
        return self._legacy_extract_question_keys(questions)


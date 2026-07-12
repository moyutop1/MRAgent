import json
import logging
import re

from common import config
from memory.system import EAESMemoryNote
from prompts.prompts import Prompts

logger = logging.getLogger(__name__)


class EAESMixin:
    @staticmethod
    def _eaes_query_question(question):
        text = str(question or "").strip()
        for suffix in (
            " No extra explanations.",
            " Give reasons with original text.",
        ):
            if text.endswith(suffix.strip()):
                text = text[:-len(suffix.strip())].rstrip()
        return text

    @staticmethod
    def _eaes_memory_id(event_id):
        return "M_" + re.sub(r"[^A-Za-z0-9]+", "_", event_id).strip("_")

    @staticmethod
    def _eaes_infer_lifecycle(text, explicit=None):
        explicit = (explicit or "").lower().strip()
        if explicit in {"planned", "current", "historical"}:
            return explicit
        t = (text or "").lower()
        planned_markers = [
            "will ", "going to", "plans to", "planned to", "planning to", "hopes to",
            "expects to", "scheduled", "upcoming", "next ", "tomorrow", "looking forward"
        ]
        historical_markers = [
            "attended", "went", "visited", "had ", "did ", "was ", "were ", "finished",
            "completed", "joined", "shared", "talked", "met", "bought", "made", "created"
        ]
        current_markers = [
            "currently", "now", "still", "is working", "is living", "lives", "works",
            "likes", "prefers", "enjoys", "has a", "has an", "is a", "are a"
        ]
        if any(m in t for m in planned_markers):
            return "planned"
        if any(m in t for m in historical_markers):
            return "historical"
        if any(m in t for m in current_markers):
            return "current"
        return "historical"

    @staticmethod
    def _eaes_entities_from_keywords(keywords, raw_text):
        stop = {
            "i", "you", "he", "she", "we", "they", "it", "me", "him", "her", "them",
            "user", "assistant", "the", "a", "an", "and", "or", "to", "of", "in", "on",
            "at", "for", "with", "from", "about", "this", "that"
        }
        entities = []
        for kw in keywords or []:
            k = str(kw).strip()
            if not k or k.lower() in stop:
                continue
            if re.search(r"[A-Z][a-z]+", k) or " " in k:
                entities.append(k)
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", raw_text or ""):
            if match.lower() not in stop:
                entities.append(match)
        out = []
        seen = set()
        for ent in entities:
            key = ent.lower()
            if key not in seen:
                seen.add(key)
                out.append(ent)
        return out[:8]

    @staticmethod
    def _eaes_short_text(text, max_chars=180):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0].rstrip(" .,;:") + "..."

    def _eaes_attribute_text(self, attr, fallback_text=None):
        if isinstance(attr, str):
            text = attr.strip()
            if not text:
                return ""
            if ":" in text:
                return text
            fallback = self._eaes_short_text(fallback_text)
            return f"{text}: {fallback}" if fallback else text
        if not isinstance(attr, dict):
            return ""
        name = str(attr.get("name") or "").strip()
        desc = str(attr.get("description") or "").strip()
        if name and desc:
            return f"{name}: {desc}"
        if desc:
            return f"event.detail: {desc}"
        if name:
            fallback = self._eaes_short_text(fallback_text)
            return f"{name}: {fallback}" if fallback else name
        return ""

    def _eaes_llm_index_for_session(self, events, keyword_by_sentence):
        memories = []
        for ee in self._as_list(events.get("sentence")):
            if not isinstance(ee, dict):
                continue
            event_id = ee.get("id")
            if event_id not in self.memory.episode_events:
                continue
            ev = self.memory.episode_events[event_id]
            memories.append({
                "event_id": event_id,
                "rewrite_content": ee.get("text") or ev.text,
                "raw_text": ev.text,
                "tag": ee.get("tag"),
                "keywords": self._as_list(keyword_by_sentence.get(event_id))[:12],
                "time": ee.get("time"),
            })
        if not memories:
            return {}
        out = self.llm.chat_text(
            messages=[
                {"role": "system", "content": Prompts.EAES_INDEX_SYSTEM_PROMPT},
                {"role": "user", "content": Prompts.eaes_index_prompt(json.dumps(memories, ensure_ascii=False))},
            ],
            model=config.RE_MODEL,
        )
        if not isinstance(out, dict):
            logger.warning("EAES LLM index returned non-dict; falling back to heuristic index.")
            return {}
        indexed = {}
        valid_ids = {m["event_id"] for m in memories}
        for item in self._as_list(out.get("memories")):
            if not isinstance(item, dict):
                continue
            event_id = item.get("event_id")
            if event_id not in valid_ids:
                continue
            entities = [str(e).strip() for e in self._as_list(item.get("entities")) if str(e).strip()]
            memory_text = next((m["rewrite_content"] for m in memories if m["event_id"] == event_id), "")
            attributes = [
                self._eaes_attribute_text(attr, memory_text)
                for attr in self._as_list(item.get("attributes"))
            ]
            attributes = [a for a in attributes if a]
            lifecycle = str(item.get("event_lifecycle") or "").lower().strip()
            indexed[event_id] = {
                "entities": list(dict.fromkeys(entities))[:8],
                "attribute_paths": list(dict.fromkeys(attributes))[:12],
                "event_lifecycle": lifecycle if lifecycle in {"planned", "current", "historical"} else None,
            }
        return indexed

    def _eaes_build_notes_for_session(self, events, keyword_by_sentence, conversation_time):
        llm_index = {}
        if config.EAES_INDEX_MODE == "llm":
            try:
                llm_index = self._eaes_llm_index_for_session(events, keyword_by_sentence)
            except Exception as e:
                logger.warning(f"EAES LLM index failed; falling back to heuristic index: {e}", exc_info=True)
                llm_index = {}
        for ee in self._as_list(events.get("sentence")):
            if not isinstance(ee, dict):
                continue
            event_id = ee.get("id")
            if event_id not in self.memory.episode_events:
                continue
            ev = self.memory.episode_events[event_id]
            keywords = self._as_list(keyword_by_sentence.get(event_id))
            index_item = llm_index.get(event_id) or {}
            entities = index_item.get("entities") or self._eaes_entities_from_keywords(keywords, ev.text)
            attribute_paths = list(index_item.get("attribute_paths") or [])
            rewrite_content = ee.get("text") or ev.text
            attribute_paths = [
                self._eaes_attribute_text(attr, rewrite_content)
                for attr in attribute_paths
            ]
            attribute_paths = [attr for attr in attribute_paths if attr and ":" in attr]
            if not attribute_paths:
                short = self._eaes_short_text(rewrite_content)
                if short:
                    attribute_paths.append(f"event.summary: {short}")
            attribute_paths = list(dict.fromkeys(attribute_paths))[:12]
            note = EAESMemoryNote(
                memory_id=self._eaes_memory_id(event_id),
                event_id=event_id,
                entities=entities,
                attribute_paths=attribute_paths,
                raw_text=ev.text,
                rewrite_content=rewrite_content,
                time_interval={
                    "type": "conversation_time",
                    "start": conversation_time,
                    "end": conversation_time,
                },
                event_lifecycle=index_item.get("event_lifecycle") or self._eaes_infer_lifecycle(rewrite_content, ee.get("event_lifecycle")),
                origin=ev.origin,
                embedding=ev.embedding,
            )
            self.memory.add_eaes_memory_note(note)

    @staticmethod
    def _eaes_should_use_unknown_lifecycle(question, query_plan):
        if not isinstance(query_plan, dict):
            return False
        required = str(query_plan.get("required_lifecycle") or "").lower().strip()
        if required not in {"current", "historical"}:
            return False
        answer_type = str(query_plan.get("answer_type") or "").lower().strip()
        temporal_intent = str(query_plan.get("temporal_intent") or "").lower().strip()
        if answer_type == "time" or temporal_intent in {"time_answer", "relative_time", "planned_event"}:
            return False
        q = re.sub(r"\s+", " ", str(question or "").lower()).strip()
        if not q:
            return False
        explicit_event = re.search(
            r"\b(when|what date|which date|where did|what did|who did|how did|"
            r"attended|participated|joined|went|visited|met|shared|recently|"
            r"yesterday|last week|last month|last year|ago)\b",
            q,
        )
        explicit_plan = re.search(
            r"\b(will|going to|plans? to|planning to|intend|intends|scheduled|"
            r"upcoming|tomorrow|next week|next month|next year)\b",
            q,
        )
        if explicit_event or explicit_plan:
            return False
        stable_fact = re.search(
            r"\b(identity|relationship status|status|preference|likes?|"
            r"interested in|kind of|type of|types of|activities|partake|"
            r"considered|member of|ally|field|fields|career|art)\b",
            q,
        )
        if stable_fact:
            return True
        return (
            temporal_intent in {"current_state", "none", ""}
            and answer_type in {"fact", "state", "person", "location", "reason", "yes_no", "unknown"}
        )

    def _postprocess_eaes_query_plan(self, question, query_plan):
        if not isinstance(query_plan, dict):
            return query_plan
        plan = dict(query_plan)
        if self._eaes_should_use_unknown_lifecycle(question, plan):
            plan["required_lifecycle"] = "unknown"
            plan["temporal_intent"] = "none"
            plan["no_time_limit"] = True
        else:
            plan["no_time_limit"] = False
        return plan

    def parse_eaes_query(self, question, question_emb=None):
        query_question = self._eaes_query_question(question)
        query_out = self.llm.chat_text(
            messages=[
                {"role": "system", "content": Prompts.EAES_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"question": query_question}, ensure_ascii=False)},
            ],
            model=config.RE_MODEL
        )
        if isinstance(query_out, dict):
            query_attributes = []
            for value in self._as_list(query_out.get("query_attributes")):
                value = str(value or "").strip()
                if value and value not in query_attributes:
                    query_attributes.append(value)
            plan = {
                "entities": self._as_list(query_out.get("entities")),
                "query_attributes": query_attributes[:3] or [query_question],
                "answer_type": query_out.get("answer_type", "unknown"),
                "temporal_intent": query_out.get("temporal_intent", "none"),
                "required_lifecycle": query_out.get("required_lifecycle", "unknown"),
                "keywords": self._as_list(query_out.get("keywords")),
                "query_mode": "question_only",
            }
            return self._postprocess_eaes_query_plan(query_question, plan)
        fallback_query = {
            "entities": [],
            "query_attributes": [query_question],
            "answer_type": "unknown",
            "temporal_intent": "none",
            "required_lifecycle": "unknown",
            "keywords": [],
            "query_mode": "question_text_fallback",
        }
        return self._postprocess_eaes_query_plan(query_question, fallback_query)

    def rerank_eaes_candidates(self, question, query_plan, candidates):
        if not candidates:
            return []
        limit = min(config.EAES_RERANK_LIMIT, len(candidates))
        rerank_input = {
            "question": self._eaes_query_question(question),
            "query_plan": query_plan,
            "limit": limit,
            "candidates": [
                {
                    "memory_id": candidate.get("memory_id"),
                    "embedding_rank": candidate.get("rank"),
                    "embedding_score": candidate.get("score"),
                    "attribute_paths": candidate.get("attribute_paths"),
                }
                for candidate in candidates
            ],
        }
        out = self.llm.chat_text(
            messages=[
                {"role": "system", "content": Prompts.EAES_ATTRIBUTE_RERANK_PROMPT},
                {"role": "user", "content": json.dumps(rerank_input, ensure_ascii=False)},
            ],
            model=config.RE_MODEL,
        )
        by_id = {candidate.get("memory_id"): candidate for candidate in candidates}
        ordered_ids = []
        if isinstance(out, dict):
            for memory_id in self._as_list(out.get("ranked_memory_ids")):
                if memory_id in by_id and memory_id not in ordered_ids:
                    ordered_ids.append(memory_id)
                if len(ordered_ids) >= limit:
                    break
        llm_selected_ids = set(ordered_ids)
        for candidate in candidates:
            memory_id = candidate.get("memory_id")
            if memory_id and memory_id not in ordered_ids:
                ordered_ids.append(memory_id)
            if len(ordered_ids) >= limit:
                break

        reranked = []
        for rerank_rank, memory_id in enumerate(ordered_ids[:limit], start=1):
            item = dict(by_id[memory_id])
            item["embedding_rank"] = item.get("rank")
            item["rerank_rank"] = rerank_rank
            item["rerank_source"] = "llm" if memory_id in llm_selected_ids else "embedding_fill"
            item["rank"] = rerank_rank
            reranked.append(item)
        return reranked

    def _enrich_eaes_package(self, package):
        if not isinstance(package, dict):
            return {"answer_items": []}
        enriched_items = []
        for item in self._as_list(package.get("answer_items")):
            if not isinstance(item, dict):
                continue
            enriched_evidence = []
            for ev in self._as_list(item.get("evidence")):
                if not isinstance(ev, dict):
                    continue
                mid = ev.get("memory_id")
                note = self.memory.get_eaes_note(mid)
                if note is None:
                    continue
                enriched_evidence.append({**ev, **note.to_dict(include_raw=False)})
            if enriched_evidence:
                enriched_items.append({
                    "item": item.get("item"),
                    "score": item.get("score"),
                    "evidence": enriched_evidence,
                })
        return {
            "need_raw_expansion": package.get("need_raw_expansion", False),
            "reason": package.get("reason", ""),
            "answer_items": enriched_items,
        }

    def _fallback_eaes_package(self, candidates, reason="selector returned no usable evidence"):
        answer_items = []
        for cand in candidates[:8]:
            mid = cand.get("memory_id")
            if not mid:
                continue
            answer_items.append({
                "item": cand.get("rewrite_content", "")[:80],
                "score": cand.get("score", 0.0),
                "evidence": [{
                    "memory_id": mid,
                    "role": "candidate_evidence",
                    "rationale": "Top retrieved memory used as fallback evidence.",
                    **cand,
                }],
            })
        return {
            "need_raw_expansion": False,
            "reason": reason,
            "answer_items": answer_items,
        }

    def select_eaes_evidence(self, question, query_plan, candidates):
        selection_input = {
            "question": question,
            "query_plan": query_plan,
            "candidates": candidates,
        }
        package = self.llm.chat_text(
            messages=[
                {"role": "system", "content": Prompts.EAES_EVIDENCE_SELECTION_PROMPT},
                {"role": "user", "content": json.dumps(selection_input, ensure_ascii=False)},
            ],
            model=config.RE_MODEL
        )
        if not isinstance(package, dict):
            package = {"answer_items": []}
        raw_ids = package.get("memory_ids_to_expand") or []
        if package.get("need_raw_expansion") and raw_ids:
            selection_input["expanded_raw_notes"] = self.memory_controller.expand_eaes_raw_text(raw_ids)
            package2 = self.llm.chat_text(
                messages=[
                    {"role": "system", "content": Prompts.EAES_EVIDENCE_SELECTION_PROMPT},
                    {"role": "user", "content": json.dumps(selection_input, ensure_ascii=False)},
                ],
                model=config.RE_MODEL
            )
            if isinstance(package2, dict):
                package = package2
        enriched = self._enrich_eaes_package(package)
        if not enriched.get("answer_items"):
            return self._fallback_eaes_package(candidates)
        return enriched

    def answer_question_eaes(self, question, category=0, question_emb=None, lm_current_date=None):
        query_plan = self.parse_eaes_query(question, question_emb)
        embedding_candidates = self.memory_controller.retrieve_eaes_candidates(
            query_plan, question_emb, limit=config.EAES_CANDIDATE_LIMIT)
        if not embedding_candidates:
            return "no information available", []
        candidates = self.rerank_eaes_candidates(question, query_plan, embedding_candidates)
        if not candidates:
            return "no information available", []
        evidence_package = self.select_eaes_evidence(question, query_plan, candidates)
        final_input = {
            "question": question,
            "query_plan": query_plan,
            "evidence_package": evidence_package,
            "backup_candidates": candidates[:12],
        }
        if lm_current_date:
            final_input["current_date"] = lm_current_date
        answer_obj = self.llm.chat_text(
            messages=[
                {"role": "system", "content": Prompts.EAES_FINAL_ANSWER_PROMPT},
                {"role": "user", "content": json.dumps(final_input, ensure_ascii=False)},
            ],
            model=config.RE_MODEL
        )
        if not isinstance(answer_obj, dict):
            evidence_package = self._fallback_eaes_package(candidates, reason="final answer JSON parsing failed")
            fallback_input = {
                "question": question,
                "query_plan": query_plan,
                "evidence_package": evidence_package,
                "backup_candidates": candidates[:12],
            }
            answer_obj = self.llm.chat_text(
                messages=[
                    {"role": "system", "content": Prompts.EAES_FINAL_ANSWER_PROMPT},
                    {"role": "user", "content": json.dumps(fallback_input, ensure_ascii=False)},
                ],
                model=config.RE_MODEL
            )
            if not isinstance(answer_obj, dict):
                return "no information available", self.memory.get_eaes_support_origin(
                    [c.get("memory_id") for c in candidates[:3]])
        supports = self._as_list(answer_obj.get("supports"))
        if not supports:
            for item in self._as_list(evidence_package.get("answer_items")):
                if not isinstance(item, dict):
                    continue
                for ev in self._as_list(item.get("evidence")):
                    if not isinstance(ev, dict):
                        continue
                    mid = ev.get("memory_id")
                    if mid and mid not in supports:
                        supports.append(mid)
        return answer_obj.get("answer", "no information available"), self.memory.get_eaes_support_origin(supports)



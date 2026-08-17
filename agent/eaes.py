import json
import logging
import re
from datetime import date

from common import config
from memory.system import EAESMemoryNote, EAESParentNode
from prompts.prompts import Prompts

logger = logging.getLogger(__name__)


class EAESMixin:
    _MONTH_NAMES = (
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )

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

    @staticmethod
    def _eaes_extract_temporal_expression(text):
        """Return the source's relative-time wording without resolving it."""
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if not value:
            return None
        patterns = (
            r"\blast\s+week\s+of\s+[A-Za-z]+(?:\s*,?\s*\d{4})?\b",
            r"\b(?:last|next)\s+(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|"
            r"Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?|weekend|week|month|year)\b",
            r"\b(?:a\s+few|few|a\s+couple\s+of|couple\s+of|an?|one|two|three|four|five|\d+)\s+"
            r"(?:days?|weeks?|weekends?|months?|years?)\s+(?:ago|before|after)\b",
            r"\b(?:yesterday|today|tomorrow)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, value, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _eaes_raw_source_text(self, event):
        """Recover original dialogue text so rewrite normalization is not lossy."""
        parts = []
        for origin in re.findall(r"D\d+:\d+", str(getattr(event, "origin", "") or "")):
            prefix = origin.split(":", 1)[0]
            raw = self.memory.raw_text.get(prefix, {}).get(origin)
            if raw:
                parts.append(str(raw))
        return "\n".join(parts) or str(getattr(event, "text", "") or "")

    @classmethod
    def _eaes_anchor_relative_answer(cls, source_expression, anchor):
        """Render an anchored-relative answer for coarse calendar cues."""
        cue = re.sub(r"\s+", " ", str(source_expression or "")).strip()
        if not cue or re.search(r"\blast\s+week\s+of\b", cue, re.IGNORECASE):
            return None
        try:
            anchor_date = date.fromisoformat(str(anchor or "")[:10])
        except (TypeError, ValueError):
            return None
        anchor_text = f"{anchor_date.day} {cls._MONTH_NAMES[anchor_date.month]} {anchor_date.year}"

        named = re.fullmatch(
            r"(last|next)\s+(Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|"
            r"Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?|weekend|week)",
            cue,
            re.IGNORECASE,
        )
        if named:
            direction, unit = named.groups()
            weekday_names = {
                "mon": "Monday", "monday": "Monday",
                "tue": "Tuesday", "tuesday": "Tuesday",
                "wed": "Wednesday", "wednesday": "Wednesday",
                "thu": "Thursday", "thursday": "Thursday",
                "fri": "Friday", "friday": "Friday",
                "sat": "Saturday", "saturday": "Saturday",
                "sun": "Sunday", "sunday": "Sunday",
            }
            normalized_unit = weekday_names.get(unit.lower(), unit.lower())
            relation = "before" if direction.lower() == "last" else "after"
            return f"The {normalized_unit} {relation} {anchor_text}"

        distance = re.fullmatch(
            r"(a\s+few|few|a\s+couple\s+of|couple\s+of|an?|one|two|three|four|five|\d+)\s+"
            r"(days?|weeks?|weekends?)\s+(ago|before|after)",
            cue,
            re.IGNORECASE,
        )
        if distance:
            amount, unit, relation = distance.groups()
            if unit.lower().startswith("day") and amount.lower() not in {
                "a few", "few", "a couple of", "couple of"
            }:
                return None
            if relation.lower() == "ago":
                relation = "before"
            amount = amount.lower()
            if amount == "few":
                amount = "a few"
            elif amount == "couple of":
                amount = "a couple of"
            return f"{amount.capitalize()} {unit.lower()} {relation.lower()} {anchor_text}"
        return None

    @classmethod
    def _eaes_precise_temporal_answer(cls, source_expression, event_time):
        """Format exact-day/month/year cues from the normalized event time."""
        cue = re.sub(r"\s+", " ", str(source_expression or "")).strip().lower()
        try:
            event_date = date.fromisoformat(str(event_time or "")[:10])
        except (TypeError, ValueError):
            return None
        human_date = f"{event_date.day} {cls._MONTH_NAMES[event_date.month]} {event_date.year}"
        if cue in {"yesterday", "today", "tomorrow"}:
            return human_date
        exact_days = re.fullmatch(
            r"(?:an?|one|two|three|four|five|\d+)\s+days?\s+ago", cue)
        if exact_days:
            return human_date
        if cue in {"last year", "next year"}:
            return str(event_date.year)
        if cue in {"last month", "next month"}:
            return f"{cls._MONTH_NAMES[event_date.month]} {event_date.year}"
        return None

    def _eaes_temporal_answer(self, answer, supports, evidence_package, candidates):
        """Preserve the supported source expression's temporal granularity."""
        supported = set(supports or [])
        evidence_records = []
        for item in (evidence_package or {}).get("answer_items", []):
            if isinstance(item, dict):
                evidence_records.extend(
                    ev for ev in item.get("evidence", []) if isinstance(ev, dict))
        candidate_records = [c for c in (candidates or []) if isinstance(c, dict)]
        records = evidence_records + candidate_records
        supported_records = [
            record for record in records if record.get("memory_id") in supported
        ]
        if supported_records:
            records = supported_records
        elif evidence_records:
            records = evidence_records
        for record in records:
            event = self.memory.episode_events.get(record.get("event_id"))
            if event is None:
                continue
            source_expression = self._eaes_extract_temporal_expression(
                self._eaes_raw_source_text(event))
            if not source_expression:
                source_expression = self._eaes_extract_temporal_expression(
                    record.get("rewrite_content") or event.text)
            rendered = self._eaes_anchor_relative_answer(
                source_expression,
                event.conversation_time or record.get("conversation_time"),
            )
            if not rendered:
                rendered = self._eaes_precise_temporal_answer(
                    source_expression, event.time)
            if rendered:
                return rendered
        iso_answer = re.fullmatch(r"\s*(\d{4})-(\d{2})-(\d{2})\s*", str(answer or ""))
        if iso_answer:
            try:
                answer_date = date.fromisoformat(iso_answer.group(0).strip())
                return f"{answer_date.day} {self._MONTH_NAMES[answer_date.month]} {answer_date.year}"
            except ValueError:
                pass
        return answer

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
                "raw_text": self._eaes_raw_source_text(ev),
                "tag": ee.get("tag"),
                "keywords": self._as_list(keyword_by_sentence.get(event_id))[:12],
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
            raw_source_text = self._eaes_raw_source_text(ev)
            note = EAESMemoryNote(
                memory_id=self._eaes_memory_id(event_id),
                event_id=event_id,
                entities=entities,
                attribute_paths=attribute_paths,
                raw_text=raw_source_text,
                rewrite_content=rewrite_content,
                conversation_time=conversation_time,
                event_lifecycle=index_item.get("event_lifecycle") or self._eaes_infer_lifecycle(rewrite_content, ee.get("event_lifecycle")),
                origin=ev.origin,
                embedding=ev.embedding,
                keywords=keywords,
            )
            self.memory.add_eaes_memory_note(note)

    def _eaes_build_parent_nodes_for_session(self, events):
        if not getattr(config, "SEMANTIC_HIERARCHY", False):
            return
        for spec in self._as_list(events.get("parent_nodes")):
            if not isinstance(spec, dict):
                continue
            parent_id = str(spec.get("parent_id") or "").strip()
            rewrite_content = str(spec.get("rewrite_content") or "").strip()
            if not parent_id or not rewrite_content:
                continue
            child_memory_ids = []
            child_attributes = []
            for event_id in self._as_list(spec.get("child_ids")):
                memory_id = self.memory.eaes_event_to_memory.get(event_id)
                note = self.memory.get_eaes_note(memory_id) if memory_id else None
                if note is None:
                    continue
                child_memory_ids.append(memory_id)
                child_attributes.append({
                    "child_id": memory_id,
                    "attributes": list(note.attribute_paths or []),
                })
            if not child_memory_ids:
                continue
            self.memory.add_eaes_parent_node(EAESParentNode(
                parent_id=parent_id,
                rewrite_content=rewrite_content,
                child_ids=child_memory_ids,
                child_attributes=child_attributes,
            ))

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

    @staticmethod
    def _eaes_child_query_plan(query_plan):
        """Hide keyword-gate fields from child scoring and downstream readers."""
        if not isinstance(query_plan, dict):
            return {}
        plan = dict(query_plan)
        plan.pop("keywords", None)
        plan.pop("keyword_groups", None)
        return plan

    def _eaes_query_keyword_fields(self, query_out):
        keywords = []
        keyword_groups = []
        for item in self._as_list(query_out.get("keywords")):
            if isinstance(item, dict):
                keyword = str(item.get("keyword") or "").strip()
                alternatives = [
                    str(value).strip()
                    for value in self._as_list(item.get("alternatives"))
                    if str(value or "").strip()
                ][:3]
            else:
                # Backward compatibility for cached tests/providers that still
                # return the former list-of-strings query schema.
                keyword = str(item or "").strip()
                alternatives = []
            if not keyword:
                continue
            keywords.append(keyword)
            keyword_groups.append({
                "keyword": keyword,
                "alternatives": alternatives,
            })
        return keywords, keyword_groups

    def _eaes_query_plan_from_output(self, query_question, query_out, query_mode):
        if not isinstance(query_out, dict):
            return None
        query_attributes = []
        for value in self._as_list(query_out.get("query_attributes")):
            value = str(value or "").strip()
            if value and value not in query_attributes:
                query_attributes.append(value)
        keywords, keyword_groups = self._eaes_query_keyword_fields(query_out)
        plan = {
            "entities": self._as_list(query_out.get("entities")),
            "query_attributes": query_attributes[:3] or [query_question],
            "answer_type": query_out.get("answer_type", "unknown"),
            "temporal_intent": query_out.get("temporal_intent", "none"),
            "required_lifecycle": query_out.get("required_lifecycle", "unknown"),
            # Parent retrieval intentionally embeds originals only. Alternatives
            # are kept in aligned groups solely for the child memory-keyword gate.
            "keywords": keywords,
            "keyword_groups": keyword_groups,
            "query_mode": query_mode,
        }
        if config.EAES_SEMANTIC_SCORE:
            # Query requirements use exact labels only. Unknown persistence
            # and unapproved/legacy labels cannot create a scoring bonus.
            allowed_properties = {
                "event_action", "state_opinion", "personal_profile",
                "relation_social", "transient", "episodic", "durable",
            }
            required_properties = []
            for value in self._as_list(
                    query_out.get("required_semantic_properties")):
                value = str(value or "").lower().strip()
                if value in allowed_properties and value not in required_properties:
                    required_properties.append(value)
            plan["required_semantic_properties"] = required_properties
        return self._postprocess_eaes_query_plan(query_question, plan)

    def parse_eaes_query(self, question, question_emb=None):
        query_question = self._eaes_query_question(question)
        # Keep the baseline query prompt byte-for-byte unchanged when semantic
        # scoring is disabled. The extension is opt-in and only asks for the
        # additional evidence-requirement field.
        query_prompt = Prompts.EAES_QUERY_SYSTEM_PROMPT
        if config.EAES_SEMANTIC_SCORE:
            query_prompt += Prompts.EAES_SEMANTIC_QUERY_EXTENSION
        query_out = self.llm.chat_text(
            messages=[
                {"role": "system", "content": query_prompt},
                {"role": "user", "content": json.dumps({"question": query_question}, ensure_ascii=False)},
            ],
            model=config.RE_MODEL
        )
        parsed_plan = self._eaes_query_plan_from_output(
            query_question, query_out, "question_only"
        )
        if parsed_plan is not None:
            return parsed_plan
        fallback_query = {
            "entities": [],
            "query_attributes": [query_question],
            "answer_type": "unknown",
            "temporal_intent": "none",
            "required_lifecycle": "unknown",
            "keywords": [],
            "keyword_groups": [],
            "query_mode": "question_text_fallback",
        }
        if config.EAES_SEMANTIC_SCORE:
            # A failed semantic query parse is neutral: no match and no penalty.
            fallback_query["required_semantic_properties"] = []
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
                    "prefilter_rank": candidate.get("rank"),
                    "prefilter_score": candidate.get("score"),
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
            item["prefilter_rank"] = item.get("rank")
            item["rerank_rank"] = rerank_rank
            item["rerank_source"] = "llm" if memory_id in llm_selected_ids else "embedding_fill"
            item["rank"] = rerank_rank
            reranked.append(item)
        return reranked

    def build_eaes_rollback_query_plan(
            self, question, query_plan, child_candidates, parent_candidates
    ):
        """Use the first-pass 16 + 4 nodes to plan a complementary retrieval."""
        query_question = self._eaes_query_question(question)
        query_prompt = (
            Prompts.EAES_QUERY_SYSTEM_PROMPT
            + (Prompts.EAES_SEMANTIC_QUERY_EXTENSION
               if config.EAES_SEMANTIC_SCORE else "")
            + "\n"
            + Prompts.EAES_ROLLBACK_QUERY_PROMPT
        )
        payload = {
            "question": query_question,
            "current_query_plan": query_plan,
            "current_top_rewrite_contents": [
                candidate.get("rewrite_content")
                for candidate in list(child_candidates or [])
                + list(parent_candidates or [])
                if candidate.get("rewrite_content")
            ],
        }
        out = self.llm.chat_text(
            messages=[
                {"role": "system", "content": query_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            model=config.RE_MODEL,
        )
        return self._eaes_query_plan_from_output(
            query_question, out, "rollback_gap"
        )

    def select_eaes_rollback_supplements(
            self,
            question,
            query_plan,
            rollback_query_plan,
            current_children,
            current_parents,
            child_candidates,
            parent_candidates,
    ):
        """Select three total nodes from the 27-child + 3-parent rollback pool."""
        limit = min(
            config.EAES_ROLLBACK_SUPPLEMENT_LIMIT,
            len(child_candidates or []) + len(parent_candidates or []),
        )
        if limit <= 0:
            return [], []
        payload = {
            "question": self._eaes_query_question(question),
            "current_query_plan": query_plan,
            "rollback_query_plan": rollback_query_plan,
            "current_top_rewrite_contents": [
                candidate.get("rewrite_content")
                for candidate in list(current_children or [])
                + list(current_parents or [])
                if candidate.get("rewrite_content")
            ],
            "limit": limit,
            "child_candidates": child_candidates,
            "parent_candidates": parent_candidates,
        }
        out = self.llm.chat_text(
            messages=[
                {
                    "role": "system",
                    "content": Prompts.EAES_ROLLBACK_SUPPLEMENT_RERANK_PROMPT,
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            model=config.RE_MODEL,
        )
        if not isinstance(out, dict):
            return [], []

        child_by_id = {
            candidate.get("memory_id"): candidate
            for candidate in child_candidates or []
            if candidate.get("memory_id")
        }
        parent_by_id = {
            candidate.get("parent_id"): candidate
            for candidate in parent_candidates or []
            if candidate.get("parent_id")
        }
        selected_keys = []
        for item in self._as_list(out.get("ranked_nodes")):
            if not isinstance(item, dict):
                continue
            node_type = str(item.get("node_type") or "").lower().strip()
            node_id = item.get("node_id")
            key = (node_type, node_id)
            if key in selected_keys:
                continue
            if (
                    (node_type == "child" and node_id in child_by_id)
                    or (node_type == "parent" and node_id in parent_by_id)
            ):
                selected_keys.append(key)
            if len(selected_keys) >= limit:
                break

        # An incomplete or malformed 30-to-3 result is treated as a failed
        # rollback check. This keeps the first-pass Top20 unchanged.
        if len(selected_keys) != limit:
            return [], []
        selected_children = [
            child_by_id[node_id]
            for node_type, node_id in selected_keys
            if node_type == "child"
        ]
        selected_parents = [
            parent_by_id[node_id]
            for node_type, node_id in selected_keys
            if node_type == "parent"
        ]
        return selected_children, selected_parents

    @staticmethod
    def _eaes_unique_candidate_pool(initial, supplements, id_key):
        pool = []
        seen = set()
        for candidate in list(initial or []) + list(supplements or []):
            node_id = candidate.get(id_key)
            if node_id and node_id not in seen:
                seen.add(node_id)
                pool.append(candidate)
        return pool

    @staticmethod
    def _eaes_finalize_rollback_type(
            pool, initial, supplements, id_key, ranked_ids, limit, llm_ids
    ):
        by_id = {candidate.get(id_key): candidate for candidate in pool}
        ordered_ids = []
        for node_id in list(ranked_ids or []) + [
                candidate.get(id_key) for candidate in initial or []
        ] + [
                candidate.get(id_key) for candidate in supplements or []
        ]:
            if node_id in by_id and node_id not in ordered_ids:
                ordered_ids.append(node_id)
            if len(ordered_ids) >= limit:
                break
        supplement_ids = {
            candidate.get(id_key) for candidate in supplements or []
        }
        result = []
        for final_rank, node_id in enumerate(ordered_ids[:limit], start=1):
            item = dict(by_id[node_id])
            item["rank"] = final_rank
            item["rollback_final_rank"] = final_rank
            item["rollback_final_source"] = (
                "llm" if node_id in llm_ids else "first_pass_fill"
            )
            item["rollback_candidate_source"] = (
                "supplement" if node_id in supplement_ids else "first_pass"
            )
            if id_key == "memory_id":
                item["rerank_rank"] = final_rank
            result.append(item)
        return result

    def rerank_eaes_rollback_final(
            self,
            question,
            query_plan,
            rollback_query_plan,
            initial_children,
            initial_parents,
            supplemental_children,
            supplemental_parents,
    ):
        """Rerank the merged pool while enforcing separate child/parent budgets."""
        child_pool = self._eaes_unique_candidate_pool(
            initial_children, supplemental_children, "memory_id"
        )
        parent_pool = self._eaes_unique_candidate_pool(
            initial_parents, supplemental_parents, "parent_id"
        )
        child_limit = min(config.EAES_RERANK_LIMIT, len(child_pool))
        parent_limit = min(config.PARENT_TOP_K, len(parent_pool))
        payload = {
            "question": self._eaes_query_question(question),
            "current_query_plan": query_plan,
            "rollback_query_plan": rollback_query_plan,
            "child_limit": child_limit,
            "parent_limit": parent_limit,
            "child_candidates": child_pool,
            "parent_candidates": parent_pool,
        }
        out = self.llm.chat_text(
            messages=[
                {
                    "role": "system",
                    "content": Prompts.EAES_ROLLBACK_FINAL_RERANK_PROMPT,
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            model=config.RE_MODEL,
        )
        if not isinstance(out, dict):
            return list(initial_children or []), list(initial_parents or [])

        child_ids = []
        child_pool_ids = {candidate.get("memory_id") for candidate in child_pool}
        for node_id in self._as_list(out.get("ranked_child_ids")):
            if node_id in child_pool_ids and node_id not in child_ids:
                child_ids.append(node_id)
        parent_ids = []
        parent_pool_ids = {candidate.get("parent_id") for candidate in parent_pool}
        for node_id in self._as_list(out.get("ranked_parent_ids")):
            if node_id in parent_pool_ids and node_id not in parent_ids:
                parent_ids.append(node_id)

        # The final call defines the actual reader Top20. Partial, duplicate,
        # or invented-ID outputs must not silently alter the first-pass set.
        if len(child_ids) != child_limit or len(parent_ids) != parent_limit:
            return list(initial_children or []), list(initial_parents or [])

        final_children = self._eaes_finalize_rollback_type(
            child_pool,
            initial_children,
            supplemental_children,
            "memory_id",
            child_ids,
            child_limit,
            set(child_ids),
        )
        final_parents = self._eaes_finalize_rollback_type(
            parent_pool,
            initial_parents,
            supplemental_parents,
            "parent_id",
            parent_ids,
            parent_limit,
            set(parent_ids),
        )
        return final_children, final_parents

    def apply_eaes_rollback_check(
            self,
            question,
            query_plan,
            child_candidates,
            parent_candidates,
            question_emb=None,
    ):
        """Run the optional 27-child + 3-parent complementary retrieval."""
        metadata = {
            "enabled": True,
            "first_query_plan": query_plan,
            "first_pass": {
                "child_ids": [
                    candidate.get("memory_id")
                    for candidate in child_candidates or []
                ],
                "parent_ids": [
                    candidate.get("parent_id")
                    for candidate in parent_candidates or []
                ],
            },
        }
        rollback_query_plan = self.build_eaes_rollback_query_plan(
            question, query_plan, child_candidates, parent_candidates
        )
        if not isinstance(rollback_query_plan, dict):
            metadata["failure_reason"] = "invalid_rollback_query_plan"
            return child_candidates, parent_candidates, metadata

        excluded_child_ids = {
            candidate.get("memory_id")
            for candidate in child_candidates or []
            if candidate.get("memory_id")
        }
        excluded_parent_ids = {
            candidate.get("parent_id")
            for candidate in parent_candidates or []
            if candidate.get("parent_id")
        }
        rollback_children = self.memory_controller.retrieve_eaes_candidates(
            self._eaes_child_query_plan(rollback_query_plan),
            question_emb,
            limit=config.EAES_ROLLBACK_CHILD_PREFILTER_LIMIT,
            exclude_memory_ids=excluded_child_ids,
            keyword_query_plan=rollback_query_plan,
        )
        rollback_parents = self.memory_controller.retrieve_eaes_parent_candidates(
            rollback_query_plan,
            question_emb,
            limit=config.EAES_ROLLBACK_PARENT_PREFILTER_LIMIT,
            exclude_parent_ids=excluded_parent_ids,
        )
        supplemental_children, supplemental_parents = (
            self.select_eaes_rollback_supplements(
                question,
                query_plan,
                rollback_query_plan,
                child_candidates,
                parent_candidates,
                rollback_children,
                rollback_parents,
            )
        )
        metadata.update({
            "rollback_query_plan": rollback_query_plan,
            "rollback_prefilter": {
                "child_candidates": rollback_children,
                "parent_candidates": rollback_parents,
            },
            "selected_supplements": {
                "child_ids": [
                    candidate.get("memory_id")
                    for candidate in supplemental_children
                ],
                "parent_ids": [
                    candidate.get("parent_id")
                    for candidate in supplemental_parents
                ],
            },
        })
        if not supplemental_children and not supplemental_parents:
            metadata["failure_reason"] = "invalid_or_empty_supplement_selection"
            return child_candidates, parent_candidates, metadata

        final_children, final_parents = self.rerank_eaes_rollback_final(
            question,
            query_plan,
            rollback_query_plan,
            child_candidates,
            parent_candidates,
            supplemental_children,
            supplemental_parents,
        )
        metadata["final"] = {
            "child_ids": [
                candidate.get("memory_id") for candidate in final_children
            ],
            "parent_ids": [
                candidate.get("parent_id") for candidate in final_parents
            ],
        }
        return final_children, final_parents, metadata

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

    def _fallback_eaes_package(
            self,
            candidates,
            reason="selector returned no usable evidence",
            limit=8,
            role="candidate_evidence",
            rationale="Top retrieved memory used as fallback evidence.",
    ):
        answer_items = []
        selected = candidates if limit is None else candidates[:limit]
        for cand in selected:
            mid = cand.get("memory_id")
            if not mid:
                continue
            answer_items.append({
                "item": cand.get("rewrite_content", "")[:80],
                "score": cand.get("score", 0.0),
                "evidence": [{
                    "memory_id": mid,
                    "role": role,
                    "rationale": rationale,
                    **cand,
                }],
            })
        return {
            "need_raw_expansion": False,
            "reason": reason,
            "answer_items": answer_items,
        }

    @staticmethod
    def _eaes_reader_child_memory(candidate):
        """Expose only answer content plus the ID required for support citation."""
        return {
            "memory_id": candidate.get("memory_id"),
            "conversation_time": candidate.get("conversation_time"),
            "rewrite_content": candidate.get("rewrite_content"),
        }

    def _eaes_reader_evidence_package(self, package):
        """Remove retrieval/index metadata before the final-reader call."""
        answer_items = []
        for item in self._as_list((package or {}).get("answer_items")):
            if not isinstance(item, dict):
                continue
            evidence = []
            for candidate in self._as_list(item.get("evidence")):
                if not isinstance(candidate, dict) or not candidate.get("memory_id"):
                    continue
                evidence.append(self._eaes_reader_child_memory(candidate))
            if evidence:
                answer_items.append({"evidence": evidence})
        return {"answer_items": answer_items}

    @staticmethod
    def _eaes_reader_parent_memory(candidate):
        """Hide the matched query keyword while retaining parent rank context."""
        return {
            "parent_id": candidate.get("parent_id"),
            "rewrite_content": candidate.get("rewrite_content"),
            "score": candidate.get("score"),
            "rank": candidate.get("rank"),
        }

    def _eaes_prediction_context(self, reader_input, candidates, limit=20):
        """Return one provenance entry per unique memory node shown to the reader."""
        visible_child_ids = []

        def add_child(memory_id):
            if memory_id and memory_id not in visible_child_ids:
                visible_child_ids.append(memory_id)

        package = (reader_input or {}).get("evidence_package") or {}
        for item in self._as_list(package.get("answer_items")):
            if not isinstance(item, dict):
                continue
            for evidence in self._as_list(item.get("evidence")):
                if isinstance(evidence, dict):
                    add_child(evidence.get("memory_id"))
        for evidence in self._as_list(
                (reader_input or {}).get("backup_candidates")):
            if isinstance(evidence, dict):
                add_child(evidence.get("memory_id"))

        # Preserve reranker order even when the selector emitted answer items
        # in a different grouping order.
        visible_set = set(visible_child_ids)
        ordered_child_ids = [
            candidate.get("memory_id")
            for candidate in candidates or []
            if candidate.get("memory_id") in visible_set
        ]
        ordered_child_ids.extend(
            memory_id for memory_id in visible_child_ids
            if memory_id not in ordered_child_ids
        )
        parent_ids = [
            parent.get("parent_id")
            for parent in self._as_list(
                (reader_input or {}).get("parent_memories"))
            if isinstance(parent, dict) and parent.get("parent_id")
        ]
        memory_ids = (ordered_child_ids + parent_ids)[:limit]
        candidate_by_id = {
            candidate.get("memory_id"): candidate
            for candidate in candidates or []
            if candidate.get("memory_id")
        }

        context = []
        for memory_id in memory_ids:
            candidate = candidate_by_id.get(memory_id)
            origin = candidate.get("origin") if candidate else None
            if origin:
                context.append(str(origin))
                continue
            expanded = self.memory.get_eaes_support_origin([memory_id])
            # A parent can expand to many child origins. Keep them grouped in
            # one string so prediction_context still has one entry per node.
            origins = [
                str(value) for value in self._as_list(expanded)
                if value and value != memory_id
            ]
            context.append(",".join(origins) if origins else str(memory_id))
        return context

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

    def _read_eaes_candidates(
            self,
            question,
            child_query_plan,
            candidates,
            parent_candidates,
            category=0,
            lm_current_date=None,
    ):
        """Generate one reader answer from an already selected child/parent set."""
        if config.DISABLE_EVIDENCE_SELECTOR:
            evidence_package = self._fallback_eaes_package(
                candidates,
                reason="evidence selector disabled for ablation; using all reranked candidates",
                limit=None,
                role="reranked_candidate",
                rationale="Reranked memory passed directly to the final reader.",
            )
        else:
            evidence_package = self.select_eaes_evidence(
                question, child_query_plan, candidates
            )
        final_input = {
            "question": question,
            "query_plan": child_query_plan,
            "evidence_package": self._eaes_reader_evidence_package(
                evidence_package
            ),
        }
        if parent_candidates:
            final_input["parent_memories"] = [
                self._eaes_reader_parent_memory(parent)
                for parent in parent_candidates
            ]
        if not config.DISABLE_EVIDENCE_SELECTOR:
            final_input["backup_candidates"] = [
                self._eaes_reader_child_memory(candidate)
                for candidate in candidates[:12]
            ]
        use_anchored_temporal_style = (
            str(config.dataset).lower() == "locomo" and str(category) == "2"
        )
        if lm_current_date:
            final_input["current_date"] = lm_current_date
        final_answer_prompt = Prompts.EAES_FINAL_ANSWER_PROMPT
        if use_anchored_temporal_style:
            final_answer_prompt += "\n" + Prompts.TEMPORAL_ANSWER_POLICY
        answer_obj = self.llm.chat_text(
            messages=[
                {"role": "system", "content": final_answer_prompt},
                {"role": "user", "content": json.dumps(final_input, ensure_ascii=False)},
            ],
            model=config.RE_MODEL
        )
        reader_input = final_input
        if not isinstance(answer_obj, dict):
            evidence_package = self._fallback_eaes_package(candidates, reason="final answer JSON parsing failed")
            fallback_input = {
                **final_input,
                "evidence_package": self._eaes_reader_evidence_package(
                    evidence_package
                ),
            }
            reader_input = fallback_input
            answer_obj = self.llm.chat_text(
                messages=[
                    {"role": "system", "content": final_answer_prompt},
                    {"role": "user", "content": json.dumps(fallback_input, ensure_ascii=False)},
                ],
                model=config.RE_MODEL
            )
            if not isinstance(answer_obj, dict):
                context = self._eaes_prediction_context(
                    reader_input, candidates
                )
                return "no information available", context, None
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
        for parent in parent_candidates:
            parent_id = parent.get("parent_id")
            if parent_id and parent_id not in supports:
                supports.append(parent_id)
        raw_answer = answer_obj.get("answer")
        answer = (
            raw_answer
            if raw_answer is not None
            else "no information available"
        )
        if use_anchored_temporal_style:
            answer = self._eaes_temporal_answer(
                answer, supports, evidence_package, candidates)
        return (
            answer,
            self._eaes_prediction_context(reader_input, candidates),
            raw_answer,
        )

    @staticmethod
    def _eaes_is_no_information_answer(answer):
        text = re.sub(r"\s+", " ", str(answer or "")).strip().lower()
        text = text.strip("\"'`")
        text = re.sub(r"[.!?]+$", "", text).strip()
        return text == "no information available"

    @staticmethod
    def _eaes_reader_node_ids(child_candidates, parent_candidates):
        return (
            tuple(
                candidate.get("memory_id")
                for candidate in child_candidates or []
            ),
            tuple(
                candidate.get("parent_id")
                for candidate in parent_candidates or []
            ),
        )

    def answer_question_eaes(
            self, question, category=0, question_emb=None,
            lm_current_date=None
    ):
        query_plan = self.parse_eaes_query(question, question_emb)
        child_query_plan = self._eaes_child_query_plan(query_plan)
        parent_candidates = []
        if getattr(config, "SEMANTIC_HIERARCHY", False):
            parent_candidates = self.memory_controller.retrieve_eaes_parent_candidates(
                query_plan, question_emb, limit=config.PARENT_TOP_K
            )
        embedding_candidates = self.memory_controller.retrieve_eaes_candidates(
            child_query_plan,
            question_emb,
            limit=config.EAES_CANDIDATE_LIMIT,
            keyword_query_plan=query_plan,
        )
        if not embedding_candidates and not parent_candidates:
            return "no information available", []
        candidates = self.rerank_eaes_candidates(
            question, child_query_plan, embedding_candidates
        ) if embedding_candidates else []
        if not candidates and not parent_candidates:
            return "no information available", []

        first_answer, first_context, first_raw_answer = (
            self._read_eaes_candidates(
                question,
                child_query_plan,
                candidates,
                parent_candidates,
                category,
                lm_current_date,
            )
        )
        if (
                not getattr(config, "EAES_ROLLBACK_CHECK", False)
                or not self._eaes_is_no_information_answer(first_raw_answer)
        ):
            return first_answer, first_context

        first_node_ids = self._eaes_reader_node_ids(
            candidates, parent_candidates
        )
        try:
            rollback_children, rollback_parents, _ = (
                self.apply_eaes_rollback_check(
                    question,
                    query_plan,
                    candidates,
                    parent_candidates,
                    question_emb,
                )
            )
        except Exception:
            logger.warning(
                "EAES rollback check failed after a no-information answer; "
                "retaining the first-pass answer and Top20.",
                exc_info=True,
            )
            return first_answer, first_context

        if self._eaes_reader_node_ids(
                rollback_children, rollback_parents
        ) == first_node_ids:
            return first_answer, first_context
        second_answer, second_context, _ = self._read_eaes_candidates(
            question,
            child_query_plan,
            rollback_children,
            rollback_parents,
            category,
            lm_current_date,
        )
        return second_answer, second_context



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
    def _normalize_eaes_parent_id(parent_id):
        """Normalize legacy D<session>:t<n> IDs without rewriting old files."""
        value = str(parent_id or "").strip()
        match = re.fullmatch(r"D(\d+):t(\d+)", value, flags=re.IGNORECASE)
        return f"{match.group(1)}-{match.group(2)}" if match else value

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
                parent_id=self._normalize_eaes_parent_id(ee.get("parent_id")),
            )
            self.memory.add_eaes_memory_note(note)

    def _eaes_build_parent_nodes_for_session(self, events):
        if not getattr(config, "SEMANTIC_HIERARCHY", False):
            return
        for spec in self._as_list(events.get("parent_nodes")):
            if not isinstance(spec, dict):
                continue
            parent_id = self._normalize_eaes_parent_id(spec.get("parent_id"))
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
                # New rewrite files persist this relation directly on every
                # child. This assignment also backfills legacy rewrite files.
                note.parent_id = parent_id
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
    def _eaes_child_query_plan(query_plan):
        """Hide parent-only query keywords from every child/reader stage."""
        if not isinstance(query_plan, dict):
            return {}
        plan = dict(query_plan)
        plan.pop("keywords", None)
        plan.pop("retrieval_phrases", None)
        plan.pop("retrieval_phrase_source", None)
        plan.pop("retrieval_breadth", None)
        plan.pop("breadth_value", None)
        plan.pop("detail_need", None)
        plan.pop("detail_value", None)
        return plan

    @staticmethod
    def _validate_eaes_retrieval_phrases(values, expected_count=4):
        if not isinstance(values, list):
            return None, "retrieval_phrases must be an array"
        phrases = []
        for index, value in enumerate(values):
            if len(phrases) >= expected_count:
                break
            if not isinstance(value, str):
                return None, f"retrieval_phrases[{index}] must be a string"
            phrase = re.sub(r"\s+", " ", value).strip()
            if not phrase:
                return None, f"retrieval_phrases[{index}] must be non-empty"
            if len(phrase.split()) > 3:
                return None, (
                    f"retrieval_phrases[{index}] must contain no more than "
                    f"3 whitespace-separated words: {value!r}"
                )
            phrases.append(phrase)
        if len(phrases) < expected_count:
            return None, (
                f"retrieval_phrases must contain at least {expected_count} "
                f"valid phrases; got {len(phrases)}"
            )
        return phrases[:expected_count], ""

    @staticmethod
    def _normalize_eaes_retrieval_phrases(values, expected_count=4):
        phrases, _ = EAESMixin._validate_eaes_retrieval_phrases(
            values, expected_count=expected_count
        )
        return phrases

    def _regenerate_eaes_retrieval_phrases(
            self, query_question, invalid_phrases, validation_error
    ):
        return self.llm.chat_text(
            messages=[
                {
                    "role": "system",
                    "content": Prompts.EAES_RETRIEVAL_PHRASE_REPAIR_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "question": query_question,
                        "previous_invalid_output": invalid_phrases,
                        "validation_error": validation_error,
                    }, ensure_ascii=False),
                },
            ],
            model=config.RE_MODEL,
        )

    def _eaes_query_plan_from_output(self, query_question, query_out, query_mode):
        if not isinstance(query_out, dict):
            return None
        query_attributes = []
        for value in self._as_list(query_out.get("query_attributes")):
            value = str(value or "").strip()
            if value and value not in query_attributes:
                query_attributes.append(value)
        breadth = str(query_out.get("retrieval_breadth") or "").lower().strip()
        if breadth not in {"single", "several", "wide"}:
            breadth = "several"
        detail = str(query_out.get("detail_need") or "").lower().strip()
        if detail not in {"coarse", "mixed", "exact"}:
            detail = "mixed"
        plan = {
            "entities": self._as_list(query_out.get("entities")),
            "query_attributes": query_attributes[:3] or [query_question],
            "answer_type": query_out.get("answer_type", "unknown"),
            "keywords": self._as_list(query_out.get("keywords")),
            "query_mode": query_mode,
            "retrieval_breadth": breadth,
            "breadth_value": {"single": 0.0, "several": 0.5, "wide": 1.0}[breadth],
            "detail_need": detail,
            "detail_value": {"coarse": 0.0, "mixed": 0.5, "exact": 1.0}[detail],
            "retrieval_phrases": (
                self._normalize_eaes_retrieval_phrases(
                    query_out.get("retrieval_phrases"),
                    expected_count=getattr(config, "EAES_PHRASE_COUNT", 4),
                )
                or []
            ),
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
        return plan

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
        if parsed_plan is None:
            parsed_plan = {
                "entities": [],
                "query_attributes": [query_question],
                "answer_type": "unknown",
                "keywords": [],
                "query_mode": "question_text_fallback",
                "retrieval_phrases": [],
                "retrieval_breadth": "several",
                "breadth_value": 0.5,
                "detail_need": "mixed",
                "detail_value": 0.5,
            }
            if config.EAES_SEMANTIC_SCORE:
                # A failed semantic query parse is neutral: no match and no penalty.
                parsed_plan["required_semantic_properties"] = []

        phrase_count = getattr(config, "EAES_PHRASE_COUNT", 4)
        raw_retrieval_phrases = (
            query_out.get("retrieval_phrases")
            if isinstance(query_out, dict)
            else parsed_plan.get("retrieval_phrases")
        )
        phrases, phrase_error = self._validate_eaes_retrieval_phrases(
            raw_retrieval_phrases,
            expected_count=phrase_count,
        )
        phrase_source = "initial"
        if phrases is None:
            repair_out = self._regenerate_eaes_retrieval_phrases(
                query_question,
                raw_retrieval_phrases,
                phrase_error,
            )
            phrases, phrase_error = self._validate_eaes_retrieval_phrases(
                repair_out.get("retrieval_phrases")
                if isinstance(repair_out, dict) else None,
                expected_count=phrase_count,
            )
            phrase_source = "regenerated"
        if phrases is None:
            raise ValueError(
                "EAES retrieval phrase generation failed after exactly one "
                f"repair attempt: {phrase_error}"
            )
        parsed_plan["retrieval_phrases"] = phrases
        parsed_plan["retrieval_phrase_source"] = phrase_source
        return parsed_plan

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

    def rerank_eaes_phrase_candidates(self, question, candidates, top_k=None):
        if not candidates:
            return []
        requested_limit = top_k or getattr(
            config, "EAES_PHRASE_RERANK_LIMIT", 15
        )
        limit = min(requested_limit, len(candidates))
        fallback_candidates = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate.get("_candidate_score") or candidate.get("candidate_score") or 0.0),
                int(candidate.get("prefilter_rank") or 10**9),
                str(candidate.get("memory_id") or ""),
            ),
        )
        prompt_candidates = sorted(
            candidates,
            key=lambda candidate: str(candidate.get("memory_id") or ""),
        )
        rerank_input = {
            "question": self._eaes_query_question(question),
            "limit": limit,
            "candidates": [
                {
                    "memory_id": candidate.get("memory_id"),
                    "origin": candidate.get("origin"),
                    "tag": candidate.get("tag"),
                    "rewrite_content": candidate.get("rewrite_content"),
                }
                for candidate in prompt_candidates
            ],
        }
        out = self.llm.chat_text(
            messages=[
                {
                    "role": "system",
                    "content": Prompts.EAES_PHRASE_CANDIDATE_RERANK_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(rerank_input, ensure_ascii=False),
                },
            ],
            model=config.RE_MODEL,
        )
        by_id = {
            candidate.get("memory_id"): candidate
            for candidate in candidates
            if candidate.get("memory_id")
        }
        ordered_ids = []
        if isinstance(out, dict):
            for memory_id in self._as_list(out.get("ranked_memory_ids")):
                if memory_id in by_id and memory_id not in ordered_ids:
                    ordered_ids.append(memory_id)
                if len(ordered_ids) >= limit:
                    break
        llm_selected_ids = set(ordered_ids)
        for candidate in fallback_candidates:
            memory_id = candidate.get("memory_id")
            if memory_id and memory_id not in ordered_ids:
                ordered_ids.append(memory_id)
            if len(ordered_ids) >= limit:
                break

        reranked = []
        for rerank_rank, memory_id in enumerate(ordered_ids[:limit], start=1):
            item = {
                key: value for key, value in by_id[memory_id].items()
                if not key.startswith("_")
            }
            item["rerank_rank"] = rerank_rank
            item["rerank_source"] = (
                "llm" if memory_id in llm_selected_ids else "candidate_score_fill"
            )
            item["rank"] = rerank_rank
            reranked.append(item)
        return reranked

    @staticmethod
    def _eaes_rollback_evidence_contents(child_candidates, parent_candidates):
        return [
            str(candidate.get("rewrite_content"))
            for candidate in list(child_candidates or [])
            + list(parent_candidates or [])
            if candidate.get("rewrite_content")
        ]

    @staticmethod
    def _validate_eaes_rollback_decision(output):
        if not isinstance(output, dict):
            raise ValueError("rollback decision must be a JSON object")

        expected_fields = {"state", "semantic_properties", "query_phase"}
        if set(output) != expected_fields:
            raise ValueError(
                "rollback decision must contain exactly state, "
                "semantic_properties, and query_phase"
            )

        state = str(output.get("state") or "").strip().lower()
        if state not in {"no_need_more", "need_more"}:
            raise ValueError(
                "rollback decision state must be no_need_more or need_more"
            )

        raw_properties = output.get("semantic_properties")
        if not isinstance(raw_properties, list):
            raise ValueError("semantic_properties must be an array")
        allowed_properties = {
            "event_action", "state_opinion", "personal_profile",
            "relation_social", "transient", "episodic", "durable",
        }
        semantic_properties = []
        for index, value in enumerate(raw_properties):
            if not isinstance(value, str):
                raise ValueError(
                    f"semantic_properties[{index}] must be a string"
                )
            normalized = value.lower().strip()
            if normalized not in allowed_properties:
                raise ValueError(
                    f"semantic_properties[{index}] is not an allowed label: "
                    f"{value!r}"
                )
            if normalized in semantic_properties:
                raise ValueError(
                    f"semantic_properties contains duplicate label: {value!r}"
                )
            semantic_properties.append(normalized)

        raw_phases = output.get("query_phase")
        if not isinstance(raw_phases, list):
            raise ValueError("query_phase must be an array")
        query_phases = []
        normalized_phases = set()
        for index, value in enumerate(raw_phases):
            if not isinstance(value, str):
                raise ValueError(f"query_phase[{index}] must be a string")
            phase = re.sub(r"\s+", " ", value).strip()
            if not phase:
                raise ValueError(f"query_phase[{index}] must be non-empty")
            if len(phase.split()) > 3:
                raise ValueError(
                    f"query_phase[{index}] must contain no more than 3 "
                    f"whitespace-separated words: {value!r}"
                )
            phase_key = phase.casefold()
            if phase_key in normalized_phases:
                raise ValueError(
                    f"query_phase contains duplicate normalized phase: {value!r}"
                )
            normalized_phases.add(phase_key)
            query_phases.append(phase)

        if state == "no_need_more":
            if semantic_properties or query_phases:
                raise ValueError(
                    "no_need_more requires empty semantic_properties and "
                    "query_phase arrays"
                )
        elif len(query_phases) != 4:
            raise ValueError(
                "need_more requires exactly four distinct query phases; "
                f"got {len(query_phases)}"
            )

        return {
            "state": state,
            "semantic_properties": semantic_properties,
            "query_phase": query_phases,
        }

    def _request_eaes_rollback_decision(
            self,
            question,
            child_candidates,
            parent_candidates,
            initial_query_phases,
            rollback_history,
            remaining_rollbacks,
    ):
        payload = {
            "question": self._eaes_query_question(question),
            "current_evidence": self._eaes_rollback_evidence_contents(
                child_candidates, parent_candidates
            ),
            "initial_query_phase": list(initial_query_phases or []),
            "rollback_history": list(rollback_history or []),
            "remaining_rollbacks": int(remaining_rollbacks),
        }
        last_error = None
        previous_output = None
        for attempt in range(2):
            request_payload = dict(payload)
            if attempt:
                request_payload.update({
                    "previous_invalid_output": previous_output,
                    "validation_error": str(last_error),
                })
            output = self.llm.chat_text(
                messages=[
                    {
                        "role": "system",
                        "content": Prompts.EAES_ROLLBACK_QUERY_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            request_payload, ensure_ascii=False
                        ),
                    },
                ],
                model=config.RE_MODEL,
            )
            previous_output = output
            try:
                return self._validate_eaes_rollback_decision(output)
            except ValueError as error:
                last_error = error
        raise ValueError(
            "EAES rollback decision failed validation after exactly one "
            f"repair attempt: {last_error}"
        )

    def select_eaes_rollback_supplements(
            self,
            question,
            rollback_decision,
            current_children,
            current_parents,
            child_candidates,
            parent_candidates,
    ):
        """Select up to three useful nodes from one rollback candidate pool."""
        limit = min(
            config.EAES_ROLLBACK_SUPPLEMENT_LIMIT,
            len(child_candidates or []) + len(parent_candidates or []),
        )
        if limit <= 0:
            return [], []
        payload = {
            "question": self._eaes_query_question(question),
            "rollback_decision": rollback_decision,
            "current_evidence": self._eaes_rollback_evidence_contents(
                current_children, current_parents
            ),
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

    def apply_eaes_rollback_check(
            self,
            question,
            query_plan,
            child_candidates,
            parent_candidates,
            question_emb=None,
    ):
        """Run up to two evidence-gap decisions and additive retrievals."""
        current_children = list(child_candidates or [])
        current_parents = list(parent_candidates or [])
        initial_query_phases = list(query_plan.get("retrieval_phrases") or [])
        excluded_child_ids = {
            candidate.get("memory_id")
            for candidate in current_children
            if candidate.get("memory_id")
        }
        excluded_parent_ids = {
            candidate.get("parent_id")
            for candidate in current_parents
            if candidate.get("parent_id")
        }
        rollback_history = []
        metadata = {
            "enabled": True,
            "first_query_plan": query_plan,
            "initial_query_phase": initial_query_phases,
            "first_pass": {
                "child_ids": [
                    candidate.get("memory_id")
                    for candidate in current_children
                ],
                "parent_ids": [
                    candidate.get("parent_id")
                    for candidate in current_parents
                ],
            },
            "rounds": [],
            "selected_supplements": {
                "child_ids": [],
                "parent_ids": [],
            },
        }
        # Two rounds are a protocol invariant: after rollback 2 the reader uses
        # the accumulated evidence without a third sufficiency call.
        max_rounds = 2
        for round_index in range(1, max_rounds + 1):
            decision = self._request_eaes_rollback_decision(
                question,
                current_children,
                current_parents,
                initial_query_phases,
                rollback_history,
                remaining_rollbacks=max_rounds - round_index + 1,
            )
            metadata["last_s2g_decision"] = decision
            if decision["state"] == "no_need_more":
                metadata.update({
                    "terminal_reason": "no_need_more",
                    "rollback_count": len(rollback_history),
                    "final": {
                        "child_ids": [
                            candidate.get("memory_id")
                            for candidate in current_children
                        ],
                        "parent_ids": [
                            candidate.get("parent_id")
                            for candidate in current_parents
                        ],
                    },
                })
                return current_children, current_parents, metadata

            rollback_children = (
                self.memory_controller.retrieve_eaes_rollback_children(
                    query_phases=decision["query_phase"],
                    semantic_properties=decision["semantic_properties"],
                    entities=query_plan.get("entities") or [],
                    question_emb=question_emb,
                    exclude_memory_ids=excluded_child_ids,
                    limit=config.EAES_ROLLBACK_CHILD_PREFILTER_LIMIT,
                )
            )
            rollback_parents = (
                self.memory_controller.retrieve_eaes_rollback_parents(
                    query_phases=decision["query_phase"],
                    exclude_parent_ids=excluded_parent_ids,
                    limit=config.EAES_ROLLBACK_PARENT_PREFILTER_LIMIT,
                )
            )
            supplemental_children, supplemental_parents = (
                self.select_eaes_rollback_supplements(
                    question,
                    decision,
                    current_children,
                    current_parents,
                    rollback_children,
                    rollback_parents,
                )
            )

            selected_child_ids = [
                candidate.get("memory_id")
                for candidate in supplemental_children
                if candidate.get("memory_id")
            ]
            selected_parent_ids = [
                candidate.get("parent_id")
                for candidate in supplemental_parents
                if candidate.get("parent_id")
            ]
            current_children.extend(supplemental_children)
            current_parents.extend(supplemental_parents)
            excluded_child_ids.update(selected_child_ids)
            excluded_parent_ids.update(selected_parent_ids)
            metadata["selected_supplements"]["child_ids"].extend(
                selected_child_ids
            )
            metadata["selected_supplements"]["parent_ids"].extend(
                selected_parent_ids
            )

            history_entry = {
                "round": round_index,
                "state": decision["state"],
                "semantic_properties": decision["semantic_properties"],
                "query_phase": decision["query_phase"],
                "selected_supplement_count": (
                    len(supplemental_children) + len(supplemental_parents)
                ),
            }
            rollback_history.append(history_entry)
            metadata["rounds"].append({
                **history_entry,
                "prefilter": {
                    "child_ids": [
                        candidate.get("memory_id")
                        for candidate in rollback_children
                    ],
                    "parent_ids": [
                        candidate.get("parent_id")
                        for candidate in rollback_parents
                    ],
                },
                "selected_supplements": {
                    "child_ids": selected_child_ids,
                    "parent_ids": selected_parent_ids,
                },
            })

        metadata.update({
            "terminal_reason": "max_rollbacks_completed",
            "rollback_count": max_rounds,
            "post_second_rollback_sufficiency": "not_checked",
            "final": {
                "child_ids": [
                    candidate.get("memory_id")
                    for candidate in current_children
                ],
                "parent_ids": [
                    candidate.get("parent_id")
                    for candidate in current_parents
                ],
            },
        })
        return current_children, current_parents, metadata

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

    def _retrieve_eaes_first_pass(self, question, question_emb=None):
        """Shared dynamic first pass for normal answering and retrieval-only."""
        query_plan = self.parse_eaes_query(question, question_emb)
        prefilter_children, phrase_retrieval = (
            self.memory_controller.retrieve_eaes_phrase_candidates(
                query_plan["retrieval_phrases"],
                include_diagnostics=True,
            )
        )
        selected_parents = []
        routing = {
            "breadth_value": float(query_plan.get("breadth_value", 0.5)),
            "detail_value": float(query_plan.get("detail_value", 0.5)),
            "parent_entropy": 0.0,
            "child_parent_dispersion": 0.0,
            "parent_child_jsd": 0.0,
            "retrieval_uncertainty": 0.0,
            "target_parent_mass": 0.0,
            "selected_parent_mass": 0.0,
            "selected_parent_k": 0,
            "parent_candidates": [],
        }
        if getattr(config, "SEMANTIC_HIERARCHY", False):
            selected_parents, routing = (
                self.memory_controller.route_eaes_parent_candidates(
                    query_plan, prefilter_children, question_emb
                )
            )
        initial_limit = getattr(config, "EAES_PHRASE_UNION_LIMIT", 60)
        initial_children = prefilter_children[:initial_limit]
        initial_retrieval = {
            "prefilter_candidate_ids": [
                item.get("memory_id") for item in prefilter_children
            ],
            "initial_candidate_ids": [
                item.get("memory_id") for item in initial_children
            ],
            "dropped_by_pool_limit_ids": [
                item.get("memory_id")
                for item in prefilter_children[initial_limit:]
            ],
        }
        final_children = self.rerank_eaes_phrase_candidates(
            question,
            initial_children,
            top_k=getattr(config, "EAES_PHRASE_RERANK_LIMIT", 15),
        ) if initial_children else []
        return {
            "query_plan": query_plan,
            "prefilter_children": prefilter_children,
            "initial_children": initial_children,
            "final_children": final_children,
            "selected_parents": selected_parents,
            "routing": routing,
            "phrase_retrieval": phrase_retrieval,
            "initial_retrieval": initial_retrieval,
            "counts": {
                "phrase_selected_k": [
                    item.get("selected_k")
                    for item in phrase_retrieval.get("phrases", [])
                ],
                "prefilter_child_k": len(prefilter_children),
                "initial_child_k": len(initial_children),
                "dropped_by_pool_limit_k": len(
                    initial_retrieval.get("dropped_by_pool_limit_ids", [])
                ),
                "final_child_k": len(final_children),
                "final_parent_k": len(selected_parents),
                "final_total_k": len(final_children) + len(selected_parents),
            },
        }

    def answer_question_eaes(
            self, question, category=0, question_emb=None,
            lm_current_date=None
    ):
        first_pass = self._retrieve_eaes_first_pass(question, question_emb)
        query_plan = first_pass["query_plan"]
        child_query_plan = self._eaes_child_query_plan(query_plan)
        parent_candidates = first_pass["selected_parents"]
        candidates = first_pass["final_children"]
        rollback_enabled = bool(
            getattr(config, "EAES_ROLLBACK_CHECK", False)
        )
        if rollback_enabled:
            candidates, parent_candidates, _ = self.apply_eaes_rollback_check(
                question,
                query_plan,
                candidates,
                parent_candidates,
                question_emb,
            )
        if not candidates and not parent_candidates and not rollback_enabled:
            return "no information available", []

        answer, context, _ = self._read_eaes_candidates(
            question,
            child_query_plan,
            candidates,
            parent_candidates,
            category,
            lm_current_date,
        )
        return answer, context



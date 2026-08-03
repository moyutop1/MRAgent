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
                              include_rank: bool = False):
        if not isinstance(query_plan, dict):
            query_plan = {}
        self.prepare_eaes_retrieval_embeddings()
        query_vectors, query_attributes = self._eaes_query_embeddings(query_plan, question_emb)
        query_entities = self._as_list(query_plan.get("entities"))
        required_lifecycle = str(query_plan.get("required_lifecycle") or "").lower().strip()
        required_semantic_properties = []
        if config.EAES_SEMANTIC_SCORE:
            for value in self._as_list(query_plan.get("required_semantic_properties")):
                value = str(value or "").lower().strip()
                if value and value not in required_semantic_properties:
                    required_semantic_properties.append(value)

        entity_words = [self._eaes_words(entity) for entity in query_entities]
        scored = []
        for note in self.memory.eaes_notes.values():
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
            lifecycle_score = 0.0
            if required_lifecycle in {"planned", "current", "historical"}:
                lifecycle_score = 1.0 if note.event_lifecycle == required_lifecycle else 0.0
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
                + 0.1 * lifecycle_score
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
                    "lifecycle": round(lifecycle_score, 3),
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

    def retrieve_eaes_candidates(self, query_plan: Dict[str, Any], question_emb=None, limit: int = None):
        limit = limit or config.EAES_CANDIDATE_LIMIT
        return self.score_eaes_candidates(query_plan, question_emb, limit=limit, include_rank=True)

    def retrieve_eaes_parent_candidates(
            self, query_plan: Dict[str, Any], question_emb=None, limit: int = None
    ):
        """Retrieve parents independently; this never filters child candidates."""
        if not config.SEMANTIC_HIERARCHY:
            return []
        self.prepare_eaes_parent_embeddings()
        query_vectors, keywords = self._eaes_parent_keyword_embeddings(
            query_plan, question_emb
        )
        if query_vectors.size == 0:
            return []
        scored = []
        for parent in self.memory.eaes_parent_nodes.values():
            if parent.retrieval_embedding is None:
                continue
            vector = self._normalize_embedding_rows(parent.retrieval_embedding)[0]
            similarities = np.dot(query_vectors, vector)
            best_index = int(np.argmax(similarities))
            score = float(similarities[best_index])
            scored.append((score, best_index, parent))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:limit or config.PARENT_TOP_K]
        return [
            {
                **parent.to_reader_dict(),
                "score": round(score, 4),
                "rank": rank,
                "matched_keyword": keywords[best_index],
            }
            for rank, (score, best_index, parent) in enumerate(selected, start=1)
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

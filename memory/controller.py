import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import json
import re
import threading
from datetime import date
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from nltk.stem import PorterStemmer
from common import config
from prompts.prompts import Prompts
from common.utils import topk_answers_by_similarity
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
        self.queried_keyword = []
        self._eaes_query_embedding_cache = {}

    # Dispatcher op

    def event_by_tag(self, key, tag, note):

        text, origin, event_ids = self.memory.event_by_tag(key, tag)

        if len(event_ids) > config.RERANK_LIMIT:
            embeddings = []
            for id in event_ids:
                embeddings.append(self.memory.episode_events[id].embedding)
            embeddings = np.vstack(embeddings)

            top_ids, _, top_embs, top_texts = topk_answers_by_similarity(self.question_emb, embeddings, event_ids,
                                                                          k=config.K2, answer_texts=text)

            origin_list = []
            for tid in top_ids:
                origin_list.append(self.memory.episode_events[tid].origin)
                self.queried_event.append(tid)
            # evidence is the retrieved event_ids
            return top_texts, origin_list, top_ids
        return text, origin, event_ids


    def query_conversation_time(self, event_id):
        return f"Conversation_time:{event_id}:{self.memory.query_conversation_time(event_id)}"

    def query_event_keywords_single(self, event_id, key_list):

        key_candidates = self.memory.query_event_keywords(event_id)
        key_candidates_new = []
        text = self.memory.episode_events[event_id].text
        for kdict in key_candidates:
            if kdict['key'] in self.queried_keyword:
                idx = key_candidates.index(kdict)
                del key_candidates[idx]
            else:
                self.set_queried_keywords(kdict, tool=True)
                key = kdict['key']
                tag = kdict['tags']
                if len(tag) > config.TAG_MAX and (key not in key_list):
                    ans_input_tag = {
                        "question": text,
                        "keyword": key,
                        "tags": tag
                    }
                    key_out = self.llm.chat_text(
                        messages=[{"role": "system", "content": Prompts.EVENT_KEYWORDS_SYSTEM_PROMPT},
                                  {"role": "user", "content": json.dumps(ans_input_tag, ensure_ascii=False)}],
                        model=config.RE_MODEL

                    )

                    scores = key_out.get("tag_scores")

                    logger.info(f"[tool-tag-sort] {scores}")

                    tag_dict = sorted(
                        ((k, v) for k, v in scores.items() if v != 0),
                        key=lambda kv: kv[1],
                        reverse=True
                    )[:config.TAG_LIMIT]
                    selected_tags = [k for k, _ in tag_dict]
                    key_candidates_new.append({"key": key, "tags": selected_tags})
                else:
                    key_candidates_new.append({"key": key, "tags": tag})
                key_list.append(key)

        return key_candidates_new, key_list

    def query_event_keywords(self, event_id):
        key_list = []
        key_candidates = []
        pattern = re.compile(r'^(D\d+):(\d+)$')
        pattern_split = re.compile(r'^(D\d+):(\d+)-(\d+)$')
        if pattern_split.match(event_id):
            event_id = event_id.split("-")[0]

        if pattern.match(event_id):
            for i in range(1,10):
                if f"{event_id}-{i}" in self.memory.episode_events:
                    key_tag, key_list = self.query_event_keywords_single(f"{event_id}-{i}", key_list)
                    key_candidates.extend(key_tag)
                else:
                    # for LM, collect all variants of an event (skip numbering gaps) for richer keywords/tags; locomo stops at the first gap
                    if config.dataset == "LM":
                        continue
                    break

        return json.dumps(key_candidates, ensure_ascii=False)

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
        keywords = self._as_list(query_plan.get("keywords"))
        required_lifecycle = str(query_plan.get("required_lifecycle") or "").lower().strip()
        required_semantic_properties = []
        if config.EAES_SEMANTIC_SCORE:
            for value in self._as_list(query_plan.get("required_semantic_properties")):
                value = str(value or "").lower().strip()
                if value and value not in required_semantic_properties:
                    required_semantic_properties.append(value)

        entity_words = [self._eaes_words(entity) for entity in query_entities]
        keyword_words = set()
        for keyword in keywords:
            keyword_words |= self._eaes_words(keyword)

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
            note_attr_words = set()
            for attribute in self._as_list(note.attribute_paths):
                note_attr_words |= self._eaes_words(attribute)

            if entity_words:
                entity_score = max(
                    self._eaes_overlap_score(words, note_entity_words | note_text_words)
                    for words in entity_words
                )
            else:
                entity_score = 0.2
            keyword_score = self._eaes_overlap_score(
                keyword_words, note_text_words | note_attr_words
            )
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
                + 1.2 * keyword_score
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
                    "keyword": round(keyword_score, 3),
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

    @staticmethod
    def _tokens(s: str) -> Set[str]:
        return set([t for t in re.split(r"[_\s]+", (s or "").lower()) if t])

    # ---------------- match a single query atomic key against graph KeyNodes (text-only, no vectors) ----------------
    def match_query_key_to_graph_keys(
            self,
            q_text: str,
            *,
            q_alternative: list,
            jaccard_thresh: float = 0.6
    ) -> Set[str]:
        """
        Return the set of graph key_ids matching the query atomic key, using text similarity only:
        1) types must match
        2) normalized exact-equal  OR  token-subset containment  OR  Jaccard >= threshold
        """
        q_list = [q_text] + (q_alternative or [])
        out: Set[str] = set()

        norm_queries = []
        for q in q_list:
            norm_q = self._snake_norm(q)
            Tq = self._tokens(norm_q)
            norm_queries.append((norm_q, Tq))

            # iterate over all keys in the graph
        for kid, kn in self.memory.keys.items():
            norm_k = self._snake_norm(kn.text)
            Tk = self._tokens(norm_k)

            # compare against any one query phrase
            for norm_q, Tq in norm_queries:
                # exact equal
                if norm_k == norm_q:
                    out.add(kid)
                    break

                # subset containment
                if (Tq and Tq.issubset(Tk)) or (Tk and Tk.issubset(Tq)):
                    out.add(kid)
                    break

                # Jaccard similarity
                inter = len(Tq & Tk)
                union = len(Tq | Tk) if (Tq or Tk) else 1
                jac = inter / union
                if jac >= jaccard_thresh:
                    out.add(kid)
                    break  # one match is enough; no need to try other alternatives

        return out

    # ---------------- compute fully/partially satisfied Values from relations ----------------
    def evaluate_relations_over_graph(
            self,
            query_keys: List[Dict[str, Any]],
            relations: List[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        query_keys: [{"key_id":"k1","text":"caroline","type":"entity", "key_quality_score":...}, ...]
        relations:  [["k1","k2"], ["k3"]]  -> (k1 AND k2) OR (k3)

        Returns:
        {
          "full_matches":   [ {value_id, value_text, via_group, matched_keys, total_keys, coverage, last_used_at}, ...],
          "partial_matches":[ { ... as above (coverage<1.0) ... } ]
        }
        """
        # 1) build a dict: key_id in the question -> {text,type}
        qmap: Dict[str, Dict[str, Any]] = {k["id"]: k for k in query_keys}
        # in LM all events hang under "user"; force-add a "user" query key to recall all user-related events
        if config.dataset == "LM":
            if "user" not in qmap:
                qmap["user"] = {"id": "user", "alternatives": []}

        # 2) for each question key, the set of graph key_ids it matches and the value set they cover
        qid_to_value_ids: Dict[str, Set[str]] = {}
        qid_to_seconds: Dict[str, Dict[str, Set[Any]]] = {}
        #
        for qid, q in qmap.items():
            matched_kids = self.match_query_key_to_graph_keys(q["id"], q_alternative = q["alternatives"])
            vset: Set[str] = set()
            sec_map: Dict[str, Set[Any]] = defaultdict(set)
            for mk in matched_kids:
                for t in self.memory.key_to_values.get(mk, set()):
                    vid, origin = t[0], t[1]
                    vset.add(vid)
                    sec_map[vid].add(origin)
            qid_to_value_ids[qid] = vset
            qid_to_seconds[qid] = sec_map

        if relations is None:
            relations = [[qid for qid in qmap.keys()]]

        full: Dict[str, Dict[str, Any]] = {}
        partial: Dict[str, Dict[str, Any]] = {}

        # 3) for each AND group, count coverage (how many qids in the group a value hits)
        for gi, group in enumerate(relations):
            if not group:
                continue
            total = len(group)
            counts: Dict[str, int] = defaultdict(int)  # value_id -> matched_key_count

            group_seconds: Dict[str, Set[Any]] = defaultdict(set)
            # accumulate values hit by each qid
            for qid in group:
                for vid in qid_to_value_ids.get(qid, set()):
                    counts[vid] += 1
                    for sec in qid_to_seconds.get(qid, {}).get(vid, set()):
                        group_seconds[vid].add(sec)

            # 4) record full / partial
            for vid, c in counts.items():
                cov = c / total
                v = self.memory.episode_events.get(vid)
                seconds = sorted(list(group_seconds.get(vid, set())), key=lambda x: str(x))
                rec = {
                    "value_id": vid,
                    "value_text": v.text,
                    "via_group": gi,  # index of this AND group
                    "matched_keys": c,  # how many atomic keys in the group are hit
                    "total_keys": total,  # total atomic keys in the group
                    "coverage": cov,  # hit ratio
                    "origin": seconds
                    # "last_used_at": v.last_used_at,  # for sorting
                }
                if len(seconds) == 1:
                    rec["second_value"] = seconds[0]
                if c == total:
                    # fully satisfies this AND group
                    prev = full.get(vid)
                    if (prev is None) or (cov > prev["coverage"]) or \
                            (cov == prev["coverage"] and total > prev["total_keys"]):
                        full[vid] = rec
                else:
                    # only partial; keep the record with the highest coverage across all groups for this value
                    prev = partial.get(vid)
                    if (prev is None) or (cov > prev["coverage"]) or \
                            (cov == prev["coverage"] and total > prev["total_keys"]):
                        partial[vid] = rec

        # 5) OR semantics: a value is full if any AND group is full; remove full ones from partial
        for vid in list(partial.keys()):
            if vid in full:
                partial.pop(vid, None)

        # 6) sort: full by group size desc then last-used time desc; partial by coverage then group size then time

        full_list = sorted(
            full.values(),
            key=lambda x: x["total_keys"],
            reverse=True
        )
        partial_list = sorted(
            partial.values(),
            key=lambda x: (x["coverage"], x["total_keys"]),
            reverse=True
        )

        return {"full_matches": full_list, "partial_matches": partial_list}


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

    def set_queried_keywords(self, query_keywords, tool=False):
        if tool:
            self.queried_keyword.append(query_keywords['key'])
        else:
            for k_dict in query_keywords:
                self.queried_keyword.append(k_dict['id'])
            # self.queried_keyword.extend(k_dict['alternatives'])

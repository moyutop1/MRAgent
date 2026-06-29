import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import json
import re
from datetime import datetime, date
from typing import List, Dict, Set, Any
from collections import defaultdict








class KeyNode:
    def __init__(self, key_id: str):
        self.key_id = key_id
        self.text = key_id
        self.tag_list = []
        self.tag_dict = {}

    def add_tag(self, tag, episode_id):
        if tag not in self.tag_list:
            self.tag_list.append(tag)
        if tag not in self.tag_dict:
            self.tag_dict[tag] = []
        self.tag_dict[tag].append(episode_id)

    def get_tag_link(self, tag):
        return self.tag_dict.get(tag, [])


    def get_tag_list(self):
        return self.tag_list


class Topic:
    def __init__(self, topic_id: str, text: str):
        self.topic_id = topic_id
        self.text = text
        self.event_list = []


class PersonalEvent:
    def __init__(self, person: str, id: str, text: str, tag:str, origin: str):
        self.person = person
        self.personal_id = id
        self.text = text
        self.tag = tag
        self.origin = origin


class Persona:
    def __init__(self, person: str):
        self.person = person
        self.persona_dict: Dict[str, PersonalEvent] = defaultdict()
        self.tag_list = []
        self.tag_dict: Dict[str, List[PersonalEvent]] = defaultdict(list)

    def add_information(self, id: str, text: str, tag:str, origin: str):
        pe = PersonalEvent(self.person, id, text, tag, origin)
        self.persona_dict[id] = pe
        if tag not in self.tag_list:
            self.tag_list.append(tag)
        self.tag_dict[tag].append(pe)

    def get_tag_text(self, tag):
        pe_list = self.tag_dict[tag]
        text_list = []
        id_list = []
        for pe in pe_list:
            text_list.append(pe.origin+":"+pe.text)
            id_list.append(pe.origin)
        return text_list, id_list


class EpisodeEvent:
    def __init__(self, event_id: str, text: str, origin: str, embedding = None, time: str = None, conv_time: str = None, true_time: str = None):
        self.event_id = event_id
        self.text = text
        self.time = time
        self.true_time = true_time
        self.tag_t = ""
        self.tag_list = []
        self.tag_dict = {}
        self.origin = origin
        self.embedding = embedding
        self.conversation_time = conv_time


    def add_tag(self, tag, episode_id):
        self.tag_list.append(tag)
        if tag not in self.tag_dict:
            self.tag_dict[tag] = []
        self.tag_dict[tag].append(episode_id)


class Link:
    def __init__(self, key_id: str, event_id: str, event_type: str, tag: str):
        self.key_id = key_id
        self.event_id = event_id
        self.event_type = event_type
        self.tag = tag


class EAESMemoryNote:
    def __init__(
            self,
            memory_id: str,
            event_id: str,
            entities: List[str],
            attribute_paths: List[str],
            raw_text: str,
            rewrite_content: str,
            time_interval: Dict[str, Any],
            event_lifecycle: str,
            origin: str,
            embedding=None,
    ):
        self.memory_id = memory_id
        self.event_id = event_id
        self.entities = entities
        self.attribute_paths = attribute_paths
        self.raw_text = raw_text
        self.rewrite_content = rewrite_content
        self.time_interval = time_interval
        self.event_lifecycle = event_lifecycle
        self.origin = origin
        self.embedding = embedding

    def to_dict(self, include_raw: bool = False):
        data = {
            "memory_id": self.memory_id,
            "event_id": self.event_id,
            "entities": self.entities,
            "attribute_paths": self.attribute_paths,
            "rewrite_content": self.rewrite_content,
            "time_interval": self.time_interval,
            "event_lifecycle": self.event_lifecycle,
        }
        if include_raw:
            data["raw_text"] = self.raw_text
        return data


class MemorySystem:
    def __init__(self):
        self.keys: Dict[str, KeyNode] = {}  # key -> meta {'aliases': set(), ...}
        self.episode_events: Dict[str, EpisodeEvent] = {}
        self.episode_links: Dict[str, Link] = {}
        self.tag_list: List[str] = []
        self.key_to_values: Dict[str, Set[tuple]] = defaultdict(set)
        self.event_to_keys: Dict[str, Set[str]] = defaultdict(set)

        self.by_tag: Dict[str, List[str]] = {}  # tag -> [edge_id]
        self.by_key: Dict[str, List[str]] = {}  # key -> [edge_id]
        self.timeline: Dict[datetime, List[str]] = {} # -> event_id
        self.topic_to_event: Dict[str, List[str]] = {}
        self.persona_list: Dict[str, Persona] = {}
        self.topic_id_list: List[str] = []
        self.topic_embeddings = []
        self.topic_sentence_list = []

        self.raw_text : Dict[str,Dict[str,str]] = {}
        self.embeddings = {}
        self.topic_dict: Dict[str, Topic] = {}
        self.tid2emb = {}
        self.eaes_notes: Dict[str, EAESMemoryNote] = {}
        self.eaes_event_to_memory: Dict[str, str] = {}
        self.eaes_by_entity: Dict[str, Set[str]] = defaultdict(set)
        self.eaes_by_attribute: Dict[str, Set[str]] = defaultdict(set)

    # ----- Node ops -----

    def add_event_time(self, event_id, absolute_time):
        absolute_time = self._to_date(absolute_time)
        self.timeline.setdefault(absolute_time, []).append(event_id)


    def get_time_event(self, start_date, end_date, include_date: bool = False):
        start_date = self._to_date(start_date) if start_date is not None else None
        end_date = self._to_date(end_date) if end_date is not None else None

        keys = sorted(self.timeline.keys())
        take = lambda d: (start_date is None or d >= start_date) and (end_date is None or d <= end_date)

        if include_date:
            return [(d, eid) for d in keys if take(d) for eid in self.timeline[d]]
        else:
            return [eid for d in keys if take(d) for eid in self.timeline[d]]

    def _years_from_timeline_keys(self):
        """Extract the set of years seen in the timeline keys."""
        yrs = set()
        for k in self.timeline.keys():
            if isinstance(k, date):  # datetime is a subclass of date, so this covers both
                yrs.add(k.year)
            elif isinstance(k, str):
                try:
                    yrs.add(date.fromisoformat(k).year)
                except Exception:
                    pass  # extend here if the timeline has other formats
        return sorted(yrs)

    def _to_date(self, x):
        # handle datetime first, then date (datetime is a subclass of date)
        if isinstance(x, datetime):
            return x.date()
        if isinstance(x, date):
            return x
        if isinstance(x, (int, float)):  # unix timestamp (seconds)
            return datetime.fromtimestamp(x).date()

        if isinstance(x, str):
            s = x.strip()

            # (A) normalize ISO-style with 00: YYYY-00-00 / YYYY-00-DD / YYYY-MM-00
            m00 = re.fullmatch(r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})', s)
            if m00:
                y = int(m00.group('y'))
                mm = m00.group('m')
                dd = m00.group('d')
                if mm == '00':  # month unknown -> 01, day also -> 01 if unknown
                    mm, dd = '01', '01'
                elif dd == '00':  # day unknown -> 01
                    dd = '01'
                s_norm = f'{y}-{mm}-{dd}'
                try:
                    return date.fromisoformat(s_norm)
                except Exception:
                    # if still illegal (e.g. 1963-02-31), fall back to the 1st of that month
                    mm_int = max(1, min(12, int(mm) if mm.isdigit() else 1))
                    return date(y, mm_int, 1)

            # (B) try a plain ISO-8601 single day: YYYY-MM-DD
            try:
                return date.fromisoformat(s)
            except Exception:
                pass

            # (C) year range: 1914-1918 / 1914-1918 / 1914-1918
            m = re.fullmatch(r'(\d{4})\s*[–—-]\s*(\d{4})', s)
            if m:
                y1, y2 = int(m.group(1)), int(m.group(2))
                if y1 > y2:
                    y1, y2 = y2, y1
                start = date(y1, 1, 1)
                end = date(y2, 12, 31)
                return start + (end - start) // 2

            # (D) date range: YYYY-MM-DD <sep> YYYY-MM-DD, sep can be to / ~ / - / -
            sep_pat = r'\s*(?:to|~|–|—)\s*'
            dm = re.fullmatch(r'(\d{4}-\d{2}-\d{2})' + sep_pat + r'(\d{4}-\d{2}-\d{2})', s)
            if dm:
                try:
                    d1 = date.fromisoformat(dm.group(1))
                    d2 = date.fromisoformat(dm.group(2))
                    if d1 > d2:
                        d1, d2 = d2, d1
                    return d1 + (d2 - d1) // 2
                except Exception:
                    pass

            # (E) other common formats
            for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    return datetime.strptime(s, fmt).date()
                except Exception:
                    pass

            # (F) two-digit year YY-MM-DD -> 20YY-MM-DD (rewrite occasionally emits '23-03-19')
            m = re.fullmatch(r'(\d{2})-(\d{2})-(\d{2})', s)
            if m:
                yy, mm, dd = m.groups()
                try:
                    return date.fromisoformat(f'20{yy}-{mm}-{dd}')
                except Exception:
                    mm_int = max(1, min(12, int(mm)))
                    return date(2000 + int(yy), mm_int, 1)

        raise TypeError(f"Unsupported date-like type: {type(x)} -> {x!r}")

    def store_raw_text(self, raw_text, conv_embeddings=None, topic_id_list=None, topic_embeddings=None):
        self.raw_text = raw_text
        self.embeddings = conv_embeddings
        self.topic_id_list = topic_id_list
        self.topic_embeddings = topic_embeddings



    def get_tag_list(self, key_id):
        if key_id not in self.keys:
            return []
        return self.keys[key_id].get_tag_list()


    def add_tag(self, tag, eid, key_id):
        if tag not in self.tag_list:
            self.tag_list.append(tag)
            self.by_tag[tag] = [eid]
        else:
            self.by_tag[tag].append(eid)
        self.event_to_keys[eid].add(key_id)


    def add_topics(self, topic_sentences, eid_topic_dict, session_id):
        if not isinstance(topic_sentences, dict):
            topic_sentences = {}

        for ts in topic_sentences:
            tid = f"D{session_id}:"+ts
            ttext = topic_sentences[ts]
            topic = Topic(tid, ttext)
            self.topic_dict[tid] = topic
            self.topic_sentence_list.append(tid + ":" + ttext)
            assert self.topic_id_list.index(tid) == self.topic_sentence_list.index(tid + ":" + ttext)

        for eid, topics in eid_topic_dict.items():
            for tid in topics or []:
                tid = f"D{session_id}:" + tid
                # [guard] some samples' episode topic references a tid not in this session's topics -> get returns None; skip that link
                t = self.topic_dict.get(tid)
                if t is not None:
                    t.event_list.append(eid)


    def add_personal_information(self, pid, ptext, porigin, ptag, person):
        if person not in self.persona_list:
            self.persona_list[person] = Persona(person)
        self.persona_list[person].add_information(pid, ptext, ptag, porigin)

    @staticmethod
    def _eaes_norm(text: str):
        text = (text or "").lower().strip()
        text = re.sub(r"[^a-z0-9_./\s-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def add_eaes_memory_note(self, note: EAESMemoryNote):
        self.eaes_notes[note.memory_id] = note
        self.eaes_event_to_memory[note.event_id] = note.memory_id
        for entity in note.entities or []:
            norm_entity = self._eaes_norm(entity)
            if norm_entity:
                self.eaes_by_entity[norm_entity].add(note.memory_id)
        for attr in note.attribute_paths or []:
            norm_attr = self._eaes_norm(attr)
            if norm_attr:
                self.eaes_by_attribute[norm_attr].add(note.memory_id)

    def get_eaes_note(self, memory_id: str):
        return self.eaes_notes.get(memory_id)

    def get_eaes_support_origin(self, memory_ids):
        origins = []
        for mid in memory_ids or []:
            note = self.eaes_notes.get(mid)
            if note is not None and note.origin not in origins:
                origins.append(note.origin)
        return origins



    def event_by_tag(self, key: str, tag: str):
        if key not in self.keys:
            return [], [], []
        links = self.keys[key].get_tag_link(tag)
        text = []
        origin = []
        event_ids = []
        for link in links:
            episode_event = self.episode_events[link]
            text.append(episode_event.event_id + ":" + episode_event.text)
            origin.append(episode_event.origin)
            event_ids.append(episode_event.event_id)
        return text, origin, event_ids

    def query_conversation_time(self, event_id):
        pattern = re.compile(r'^(D\d+):(\d+)$')
        if pattern.match(event_id):
            return self.episode_events[event_id+"-1"].conversation_time
        return self.episode_events[event_id].conversation_time

    def query_event_keywords(self, event_id):
        return [{"key": k, "tags": self.get_tag_list(k)} for k in self.event_to_keys[event_id]]

    def query_personal_information(self, person):
        return {"person":person, "aspects":self.persona_list[person].tag_list}

    def query_personal_aspect(self, person, aspect):
        return self.persona_list[person].get_tag_text(aspect)


    def query_event_context(self, event_id):
        event_text = []
        EVENT_RE = re.compile(r"^s\d+$")
        origin_list = []
        pattern = re.compile(r'^(D\d+):(\d+)$')
        if not EVENT_RE.match(event_id):
            if pattern.match(event_id):
                event_origin = event_id
            else:
                event_origin = self.episode_events[event_id].origin

            m = pattern.match(event_origin)
            prefix, n = m.group(1), int(m.group(2))
            # [fix, option B] under LM user-only filtering, n+-1 lands on the filtered-out assistant turn (raw_text only has user's odd turns),
            # the original `id + ":" + None` crashed. Instead, find the nearest actually-existing previous/next turn in raw_text (i.e. the previous/next user utterance).
            session_turns = self.raw_text.get(f"{prefix}", {})
            avail_n = sorted(int(k.split(":")[1]) for k in session_turns.keys())
            prev_cands = [x for x in avail_n if x < n]
            next_cands = [x for x in avail_n if x > n]
            prev_id = f"{prefix}:{prev_cands[-1]}" if prev_cands else None
            next_id = f"{prefix}:{next_cands[0]}" if next_cands else None
            if prev_id is not None:
                event_text.append(prev_id + ":" + session_turns.get(prev_id))
                origin_list.append(prev_id)
            _cur = session_turns.get(event_origin)
            if _cur is not None:
                event_text.append(event_origin + ":" + _cur)
                origin_list.append(event_origin)
            if next_id is not None:
                event_text.append(next_id + ":" + session_turns.get(next_id))
                origin_list.append(next_id)

        return json.dumps(event_text, ensure_ascii=False), origin_list



    def query_topic_events(self, topic_id):

        event_text = []
        events = self.topic_dict[topic_id].event_list
        origin_list = []
        for e in events:
            event = self.episode_events[e]
            e_origin = event.origin
            if e_origin in origin_list:
                continue
            event_text.append(event.event_id + ":" + event.text)
            origin_list.append(event.origin)
        return json.dumps(event_text, ensure_ascii=False), origin_list




    def get_support_origin(self, evidence_support):
        if not evidence_support:
            return []
        return [self.episode_events[e].origin if e in self.episode_events else e
                for e in evidence_support]


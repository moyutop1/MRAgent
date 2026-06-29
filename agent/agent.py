import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import json
import random
import re
from typing import Dict, Any, List
import numpy as np
from prompts import schema as json_scheme
from prompts.prompts import Prompts
from common.utils import topk_answers_by_similarity
from common import config
from llm.controller import LLM
from memory.controller import MemoryController
from memory.system import MemorySystem, KeyNode, EpisodeEvent, Link, EAESMemoryNote
from agent.tools import TOOLS, ToolBridge
import logging
logger = logging.getLogger(__name__)
class Agent:
    def __init__(self, llm: LLM, memory_system: MemorySystem, memory_controller: MemoryController):
        self.llm = llm
        self.memory = memory_system
        self.memory_controller = memory_controller
        self.tools = TOOLS
        self.tool_bridge = ToolBridge(memory_controller)

        self.episode_link_num = 0
        self.tags = set()

    # ---------- Core utility: one tool-calling turn (with automatic tool execution) ----------
    def _chat_with_tools(self, system_prompt: str, user_obj: dict, category):
        return self.llm.chat_with_tools_once(
            system_prompt=system_prompt,
            user_obj=user_obj,
            tools=self.tools,
            tool_choice="auto",
            execute_tool=self.tool_bridge.call,  # bind tool executor
            temperature=0.0,
            category=category,
            model=config.RE_MODEL
        )

    @staticmethod
    def question_format(dataset, qa):
        if dataset == "locomo":
            if qa.get("category") == 5:
                question = qa['question'] + " Select the correct answer: {} or {}. "
                if random.random() < 0.5:
                    question = question.format('Not mentioned in the conversation', qa['adversarial_answer'])
                else:
                    question = question.format(qa['adversarial_answer'], 'Not mentioned in the conversation')
            elif qa.get("category") == 2:
                question = qa['question']
            elif qa.get("category") == 1:
                question = qa['question'] + (" No extra explanations. ")
            elif qa.get("category") == 3:
                question = qa['question'] + (" Give reasons with original text. ")
            else:
                question = qa['question']
        else:
            question = qa['question']

        return question

    def answer_question_with_time(self, question, question_time, question_keys, category):
        events_time = self.memory_controller.get_time_event(question_time)
        events_time_originlist = list()
        for time in events_time:
            events_time_originlist.append(time.split("-")[0].strip())
        if len(events_time) < config.TIME_EVENT_LIMIT:
            similar_sentences = []
            queried_origin = []
            for id in events_time:
                id_origin = self.memory.episode_events[id].origin
                if id_origin in queried_origin:
                    continue
                similar_sentences.append("Time:"+self.memory.episode_events[id].time+" "+id+":"+self.memory.episode_events[id].text)
                queried_origin.append(id_origin)

            ans_input = {
                "question": question,
                "similar_sentence": similar_sentences,
            }
            return ans_input

        else:
            query_answers = self.memory_controller.evaluate_relations_over_graph(question_keys.get("keywords"))
            full_ma = query_answers.get("full_matches")
            part_ma = query_answers.get("partial_matches")
            similar_sentence = []
            queried_origin = []

            def _collect(m):
                origin0 = m.get("origin")[0]
                if origin0 not in events_time_originlist or origin0 in queried_origin:
                    return
                ev = self.memory.episode_events[m.get("value_id")]
                similar_sentence.append("Time:" + ev.time + " " + m.get("value_id") + ":" + m.get("value_text"))
                queried_origin.append(origin0)

            for m in full_ma:
                _collect(m)
            for m in part_ma:
                if m.get("matched_keys") >= 1:
                    _collect(m)
            ans_input = {
                "question": question,
                "similar_sentence": similar_sentence,
            }
            return ans_input

    def answer_question_with_time_lm(self, question, question_emb, query_answers, question_time, question_keys, category):
        # LM-specific temporal handling
        events_time = self.memory_controller.get_time_event(question_time)
        events_time_originlist = list()
        for time in events_time:
            events_time_originlist.append(time.split("-")[0].strip())
        if len(events_time) < config.K2:
            similar_sentences = []
            queried_origin = []
            for id in events_time:
                id_origin = self.memory.episode_events[id].origin
                if id_origin in queried_origin:
                    continue
                similar_sentences.append("Time:"+self.memory.episode_events[id].time+" "+id+":"+self.memory.episode_events[id].text)
                queried_origin.append(id_origin)
            ans_input = {
                "question": question,
                "key_sentences": similar_sentences,
                "current_date": question_time.split(",")[0].strip() if question_time else None,  # current date (the "today" at question time)
            }
            return ans_input, events_time
        else:
            full_ma = query_answers.get("full_matches")
            part_ma = query_answers.get("partial_matches")
            similar_sentence_embs = []
            similar_sentence_ids = []
            similar_sentence = []
            queried_origin = []

            def _collect(m):
                origin0 = m.get("origin")[0]
                if origin0 not in events_time_originlist or origin0 in queried_origin:
                    return
                ev = self.memory.episode_events[m.get("value_id")]
                similar_sentence.append("Time:" + ev.time + " " + m.get("value_id") + ":" + m.get("value_text"))
                similar_sentence_embs.append(ev.embedding)
                similar_sentence_ids.append(m.get("value_id"))
                queried_origin.append(origin0)

            for m in full_ma:
                _collect(m)
            for m in part_ma:
                if m.get("matched_keys") >= 1:
                    _collect(m)
            if len(similar_sentence_embs) > config.K1:
                similar_sentence_embs = np.vstack(similar_sentence_embs)
                top_ids, _, top_embs, top_texts = topk_answers_by_similarity(question_emb, similar_sentence_embs, similar_sentence_ids,
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
            if len(top_ids) > config.K2:
                top_ids, top_embs, top_texts = self.select_finegrained_sentence_sort(question, question_emb, top_texts,
                                                                                     top_ids, top_embs)
                if config.MODEL_NAME == "gemini":
                    top_ids2, top_embs2, top_texts2 = self.select_finegrained_sentence(question, question_emb, top_texts,
                                                                                       top_ids, top_embs)
                    all_ids = top_ids + top_ids2
                    all_texts = top_texts + top_texts2
                    merged = {}
                    for i, t in zip(all_ids, all_texts):
                        merged.setdefault(i, t)
                    top_ids = list(merged.keys())
                    top_texts = list(merged.values())
                sort_ids = top_ids
            queried_similar_sentence_ids = self.extract_id_prefixes(top_ids)
            key_candidates, key_tag_sentences_id, key_tag_sentences = self.select_key_tag(question, question_keys,
                                                                                          queried_similar_sentence_ids)
            self.memory_controller.set_queried_events(top_ids)
            ans_input = {
                "question": question,
                "key_sentences": top_texts,
                "current_date": question_time.split(",")[0].strip() if question_time else None,  # current date (the "today" at question time)
            }
            return ans_input, top_ids

    def get_time(self,question_time):
        # 1) "YYYY-MM-DD, YYYY-MM-DD"
        YMD_RANGE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}, \d{4}-\d{2}-\d{2}$")
        # 2) "MM-DD, MM-DD"
        MD_RANGE_RE = re.compile(r"^\d{2}-\d{2}, \d{2}-\d{2}$")
        qt_str = str(question_time).strip()

        valid = (
                qt_str == "" or
                qt_str == "''" or
                YMD_RANGE_RE.match(qt_str) is not None or
                MD_RANGE_RE.match(qt_str) is not None
        )

        if not valid:
            question_time = None
        return question_time


    def extract_id_prefixes(self, id_list):
        """
        Extract the prefix of an id list (drop the -<number> suffix).
        e.g. ['D1:1-1', 'D1:1-2', 'D2:3'] -> ['D1:1', 'D1:1', 'D2:3']
        """
        prefixes = []
        pattern = re.compile(r'^(.+)-\d+$')
        for id_str in id_list:
            match = pattern.match(id_str)
            if match:
                prefixes.append(match.group(1))
            else:
                prefixes.append(id_str)
        return prefixes

    def select_topic(self, question_emb):
        if not self.memory.topic_id_list or len(self.memory.topic_embeddings) == 0:
            return []
        similar_topic_embs = np.vstack(self.memory.topic_embeddings)
        if similar_topic_embs.shape[0] == 0:
            return []
        top_tids, _, top_tembs, top_topic_texts = topk_answers_by_similarity(question_emb, similar_topic_embs,
                                                                             self.memory.topic_id_list,
                                                                             k=config.TOPIC_K,
                                                                             answer_texts=self.memory.topic_sentence_list)
        return top_topic_texts

    def select_finegrained_sentence(self, question, question_emb, coarse_sentences, coarse_ids, coarse_sentence_embs):
        sort_ids = []
        selected_list = []
        emb_list = []
        text_list = []
        # above the number of fine-grained sentence
        if len(coarse_ids) > config.K2:
            ans_input2 = {
                "question": question,
                "similar_sentence": coarse_sentences,
            }

            question_out = self.llm.chat_text(
                messages=[{"role": "system", "content": Prompts.ANSWER_SORT_PROMPT2},
                          {"role": "user", "content": json.dumps(ans_input2, ensure_ascii=False)}],
                model=config.RE_MODEL
            )



            if question_out is None:
                pass  # no LLM ranking returned; fall through to the similarity fallback below
            else:
                sort_ids = question_out.get("events")

                logger.info(f"[sort] {question_out}")

                text_list = []
                emb_list = []
                selected_list = []
                if sort_ids != None:
                    for id in sort_ids:
                        match_idx = None
                        for i, tid in enumerate(coarse_ids):
                            # exact match, or tid prefixed by "id-", e.g. id="D32:6", tid="D32:6-2"
                            if tid == id or tid.startswith(id + "-"):
                                match_idx = i
                                break
                        if match_idx is not None:

                            text_list.append(coarse_sentences[match_idx])
                            emb_list.append(coarse_sentence_embs[match_idx])
                            selected_list.append(coarse_ids[match_idx])
                else:
                    sort_ids = []

            # if select fails
            if len(sort_ids) == 0 or len(selected_list) > config.K2:
                if len(selected_list) != 0:
                    emb_list = np.vstack(emb_list)

                    selected_list, _, emb_list, text_list = topk_answers_by_similarity(question_emb, emb_list,
                                                                                 selected_list,
                                                                                 k=config.K2,
                                                                                 answer_texts=text_list)
                elif len(coarse_sentence_embs) != 0:
                    coarse_sentence_embs = np.vstack(coarse_sentence_embs)
                    selected_list, _, emb_list, text_list = topk_answers_by_similarity(question_emb, coarse_sentence_embs,
                                                                                 coarse_ids,
                                                                                 k=config.K2,
                                                                                 answer_texts=coarse_sentences)
        return selected_list, emb_list, text_list



    def select_finegrained_sentence_sort(self, question, question_emb, coarse_sentences, coarse_ids, coarse_sentence_embs):
        sort_ids = []
        selected_list = []
        emb_list = []
        text_list = []
        # above the number of fine-grained sentence
        if len(coarse_ids) > config.K2:
            ans_input2 = {
                "question": question,
                "similar_sentence": coarse_sentences,
            }

            question_out = self.llm.chat_text(
                messages=[{"role": "system", "content": Prompts.ANSWER_SORT_PROMPT},
                          {"role": "user", "content": json.dumps(ans_input2, ensure_ascii=False)}],
                model=config.RE_MODEL
            )

            if question_out is None:
                pass  # no LLM ranking returned; fall through to the similarity fallback below
            else:
                scores = question_out.get("relevance_scores")

                logger.info(f"[sort] {question_out}")

                evidence_dict = sorted(
                    ((k, v) for k, v in scores.items() if v != 0),
                    key=lambda kv: kv[1],
                    reverse=True
                )[:config.K2]

                sort_ids = [k for k, _ in evidence_dict]


                text_list = []
                emb_list = []
                selected_list = []
                if sort_ids != None:
                    for id in sort_ids:
                        match_idx = None
                        # LM ids may carry a speaker suffix (e.g. D27:34-1:Joanna); match tolerantly by stripping the speaker
                        id_parts = id.rsplit(":", 1)
                        id_without_speaker = id_parts[0] if len(id_parts) > 1 and not id_parts[1].isdigit() else id
                        for i, tid in enumerate(coarse_ids):
                            if config.dataset == "LM":
                                tid_parts = tid.rsplit(":", 1)
                                tid_without_speaker = tid_parts[0] if len(tid_parts) > 1 and not tid_parts[1].isdigit() else tid
                                _match = (tid == id or tid.startswith(id + "-") or
                                          tid == id_without_speaker or
                                          tid_without_speaker == id_without_speaker or
                                          tid.startswith(id_without_speaker + "-"))
                            else:
                                # exact match, or tid prefixed by "id-", e.g. id="D32:6", tid="D32:6-2"
                                _match = (tid == id or tid.startswith(id + "-"))
                            if _match:
                                match_idx = i
                                break
                        if match_idx is not None:
                            text_list.append(coarse_sentences[match_idx])
                            emb_list.append(coarse_sentence_embs[match_idx])
                            selected_list.append(coarse_ids[match_idx])
                else:
                    sort_ids = []


            # if select fails
            if len(sort_ids) == 0 or len(selected_list) > config.K2:
                if len(selected_list) != 0:
                    emb_list = np.vstack(emb_list)

                    selected_list, _, emb_list, text_list = topk_answers_by_similarity(question_emb, emb_list,
                                                                                 selected_list,
                                                                                 k=config.K2,
                                                                                 answer_texts=text_list)
                elif len(coarse_sentence_embs) != 0:
                    coarse_sentence_embs = np.vstack(coarse_sentence_embs)
                    selected_list, _, emb_list, text_list = topk_answers_by_similarity(question_emb, coarse_sentence_embs,
                                                                                 coarse_ids,
                                                                                 k=config.K2,
                                                                                 answer_texts=coarse_sentences)
        return selected_list, emb_list, text_list


    def select_key_tag(self, question, question_keys, queried_similar_sentence_ids):
        key_candidates = []
        key_tag_sentences = []
        key_tag_sentences_id = []
        if not isinstance(question_keys, dict):
            question_keys = {}
        for s in self._as_list(question_keys.get("keywords")):
            if not isinstance(s, dict):
                continue
            # [fix] use the raw key (consistent with stored keys and qmap); lemmatize lowercases + stems,
            # which would mismatch the raw-stored keys (proper nouns / plurals / past tense missed).
            key = s.get("id")
            if not key:
                continue
            tag = self.memory.get_tag_list(key)
            if len(tag) != 0:
                if len(tag) > config.TAG_MAX:
                    ans_input_tag = {
                        "question": question,
                        "keyword": key,
                        "tags": tag
                    }

                    key_out = self.llm.chat_text(
                        messages=[
                            {"role": "system", "content": Prompts.EVENT_KEYWORDS_SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(ans_input_tag, ensure_ascii=False)},
                        ],
                        model=config.RE_MODEL, )


                    if key_out is None:
                        key_candidates.append({"key": key, "tags": tag})
                    else:
                        scores = key_out.get("tag_scores")

                        logger.info(f"[sort] {scores}")

                        tag_dict = sorted(
                            ((k, v) for k, v in scores.items() if v != 0),
                            key=lambda kv: kv[1],
                            reverse=True
                        )[:config.TAG_LIMIT]
                        tag_list = []

                        for k, v in tag_dict:
                            tag_list.append(k)
                            if v >= 0.7:
                                text_queried, tagids, _ = self.memory_controller.event_by_tag(key, k, "")
                                selected_sentences = []
                                for i in range(len(tagids)):
                                    if tagids[i] in queried_similar_sentence_ids:
                                        continue
                                    selected_sentences.append(text_queried[i])
                                    queried_similar_sentence_ids.append(tagids[i])
                                key_tag_sentences.append(f"key:{key},tag:{k}:{selected_sentences}")
                                key_tag_sentences_id.extend(tagids)
                        key_candidates.append({"key": key, "tags": tag_list})
                else:
                    key_candidates.append({"key": key, "tags": tag})
        return key_candidates, key_tag_sentences_id, key_tag_sentences

    @staticmethod
    def _eaes_safe_path(prefix, text):
        text = (text or "").lower().strip()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            return None
        return f"{prefix}.{text}"

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

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
    def _eaes_attribute_text(attr):
        if isinstance(attr, str):
            return attr.strip()
        if not isinstance(attr, dict):
            return ""
        name = str(attr.get("name") or "").strip()
        desc = str(attr.get("description") or "").strip()
        if name and desc:
            return f"{name}: {desc}"
        return name or desc

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
            attributes = [
                self._eaes_attribute_text(attr)
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
            tag_path = self._eaes_safe_path("tag", ee.get("tag"))
            if tag_path:
                attribute_paths.append(tag_path)
            for kw in keywords[:8]:
                kw_path = self._eaes_safe_path("keyword", kw)
                if kw_path:
                    attribute_paths.append(kw_path)
            for topic_id in self._as_list(ee.get("topic")):
                topic_path = self._eaes_safe_path("topic", topic_id)
                if topic_path:
                    attribute_paths.append(topic_path)
            attribute_paths = list(dict.fromkeys(attribute_paths))[:12]
            note = EAESMemoryNote(
                memory_id=self._eaes_memory_id(event_id),
                event_id=event_id,
                entities=entities,
                attribute_paths=attribute_paths,
                raw_text=ev.text,
                rewrite_content=ee.get("text") or ev.text,
                time_interval={
                    "type": "conversation_time",
                    "start": conversation_time,
                    "end": conversation_time,
                },
                event_lifecycle=index_item.get("event_lifecycle") or self._eaes_infer_lifecycle(ee.get("text") or ev.text, ee.get("event_lifecycle")),
                origin=ev.origin,
                embedding=ev.embedding,
            )
            self.memory.add_eaes_memory_note(note)

    def parse_eaes_query(self, question, question_keys, question_emb=None):
        if config.EAES_QUERY_MODE == "inventory":
            inventory_query = self._parse_eaes_query_from_inventory(question, question_emb)
            if inventory_query:
                return inventory_query
        query_out = self.llm.chat_text(
            messages=[
                {"role": "system", "content": Prompts.EAES_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)},
            ],
            model=config.RE_MODEL
        )
        if isinstance(query_out, dict):
            return query_out
        if not isinstance(question_keys, dict):
            question_keys = {}
        key_items = self._as_list(question_keys.get("keywords"))
        return {
            "entities": [k.get("id") for k in key_items if isinstance(k, dict)],
            "attribute_hints": [],
            "answer_type": "unknown",
            "temporal_intent": "none",
            "required_lifecycle": "unknown",
            "keywords": [k.get("id") for k in key_items if isinstance(k, dict)],
        }

    def _dense_eaes_note_ids(self, question_emb, k=None):
        if question_emb is None:
            return []
        ids, embs = [], []
        for mid, note in self.memory.eaes_notes.items():
            if note.embedding is None:
                continue
            ids.append(mid)
            embs.append(note.embedding)
        if not embs:
            return []
        top_ids, _, _, _ = topk_answers_by_similarity(
            question_emb, np.vstack(embs), ids, k=k or config.EAES_QUERY_CANDIDATE_DENSE_K)
        return top_ids

    def _eaes_inventory_candidates(self, question, question_emb=None):
        q_tokens = self._key_tokens(question)
        dense_mids = self._dense_eaes_note_ids(question_emb, config.EAES_QUERY_CANDIDATE_DENSE_K)
        entity_scores, attr_scores = {}, {}
        entity_examples, attr_examples = {}, {}
        entity_dense_hits = set()

        def add_example(target, key, text):
            if not key:
                return
            bucket = target.setdefault(key, [])
            if len(bucket) < 2:
                bucket.append(text[:220])

        for rank, mid in enumerate(dense_mids):
            note = self.memory.get_eaes_note(mid)
            if note is None:
                continue
            weight = 2.0 / (rank + 1)
            example = f"{note.event_id}: {note.rewrite_content}"
            for entity in self._as_list(note.entities):
                entity_scores[entity] = entity_scores.get(entity, 0.0) + weight
                entity_dense_hits.add(entity)
                add_example(entity_examples, entity, example)
            for attr in self._as_list(note.attribute_paths):
                attr_scores[attr] = attr_scores.get(attr, 0.0) + weight
                add_example(attr_examples, attr, example)

        for note in self.memory.eaes_notes.values():
            note_entities = [str(entity) for entity in self._as_list(note.entities) if entity is not None]
            note_attributes = [str(attr) for attr in self._as_list(note.attribute_paths) if attr is not None]
            note_text = f"{note.rewrite_content} {' '.join(note_entities)} {' '.join(note_attributes)}"
            text_tokens = self._key_tokens(note_text)
            overlap = len(q_tokens & text_tokens)
            if not overlap:
                continue
            example = f"{note.event_id}: {note.rewrite_content}"
            for entity in note_entities:
                entity_scores[entity] = entity_scores.get(entity, 0.0) + 1.0 + overlap / max(len(text_tokens), 1)
                add_example(entity_examples, entity, example)
            for attr in note_attributes:
                attr_scores[attr] = attr_scores.get(attr, 0.0) + 1.0 + overlap / max(len(text_tokens), 1)
                add_example(attr_examples, attr, example)

        entity_records = [
            {
                "entity": entity,
                "score": round(score, 4),
                "source": "dense" if entity in entity_dense_hits else "lexical",
                "examples": entity_examples.get(entity, []),
            }
            for entity, score in sorted(entity_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        ][:config.EAES_QUERY_ENTITY_LIMIT]

        attr_records = [
            {
                "attribute": attr,
                "score": round(score, 4),
                "examples": attr_examples.get(attr, []),
            }
            for attr, score in sorted(attr_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        ][:config.EAES_QUERY_ATTRIBUTE_LIMIT]
        return entity_records, attr_records

    def _parse_eaes_query_from_inventory(self, question, question_emb=None):
        entity_records, attr_records = self._eaes_inventory_candidates(question, question_emb)
        if not entity_records and not attr_records:
            return None
        out = self.llm.chat_text(
            messages=[
                {"role": "system", "content": Prompts.EAES_QUERY_INVENTORY_SYSTEM_PROMPT},
                {"role": "user", "content": Prompts.eaes_query_inventory_prompt(
                    question,
                    json.dumps(entity_records, ensure_ascii=False),
                    json.dumps(attr_records, ensure_ascii=False),
                )},
            ],
            model=config.RE_MODEL,
        )
        if not isinstance(out, dict):
            return None
        valid_entities = {r["entity"] for r in entity_records}
        valid_attrs = {r["attribute"] for r in attr_records}
        selected_entities = []
        for entity in self._as_list(out.get("entities")):
            if entity in valid_entities and entity not in selected_entities:
                selected_entities.append(entity)
        selected_attrs = []
        for attr in self._as_list(out.get("attribute_hints")):
            if attr in valid_attrs and attr not in selected_attrs:
                selected_attrs.append(attr)
        return {
            "entities": selected_entities,
            "attribute_hints": selected_attrs,
            "answer_type": out.get("answer_type", "unknown"),
            "temporal_intent": out.get("temporal_intent", "none"),
            "required_lifecycle": out.get("required_lifecycle", "unknown"),
            "keywords": self._as_list(out.get("keywords")),
            "query_mode": "inventory",
            "entity_candidate_count": len(entity_records),
            "attribute_candidate_count": len(attr_records),
        }

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
            "candidates": candidates[:config.EAES_SELECTION_LIMIT],
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
        question_keys = self.extract_question_keys(question, question_emb)
        query_plan = self.parse_eaes_query(question, question_keys, question_emb)
        candidates = self.memory_controller.retrieve_eaes_candidates(
            query_plan, question_emb, limit=config.EAES_CANDIDATE_LIMIT)
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
            question_keys = self.extract_question_keys(question, question_emb)
            query_plan = self.parse_eaes_query(question, question_keys, question_emb)
            candidates = self.memory_controller.retrieve_eaes_candidates(
                query_plan, question_emb, limit=config.EAES_CANDIDATE_LIMIT)
            event_ids = self._unique_keep_order([c.get("event_id") for c in candidates])
            origins = self._unique_keep_order(
                [c.get("origin") for c in candidates] or self._origins_for_event_ids(event_ids)
            )
            if not origins:
                origins = self._origins_for_event_ids(event_ids)
            return {
                "mode": "eaes",
                "question_keys": question_keys,
                "query_plan": query_plan,
                "retrieved_event_ids": event_ids,
                "retrieved_origins": origins,
                "candidates": candidates,
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

        if category == 2:
            ans_input["question"] = (ans_input["question"]
                + " After identifying the corresponding event, "
                + "call query_conversation_time and calculate an absolute date grounded to the query conversation time, 'yesterday' of conversation time '7 May 2023' is '6 May 2023'. "
                + "For 'when' questions, accepted formats include: '7 May 2023', 'May 2023', '2023', "
                "'the week/Sunday before 25 May 2023' or else with no additional words. For 'how long' questions, do not compute or convert; return the duration exactly as written in the conversation.")
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

    @staticmethod
    def _normalize_sentence_ids(rewrite_out):
        # normalize each sentence id to "{origin}-{seq}" (per-origin sequence from 1 in appearance order).
        # fixes the LLM omitting "-seq" under large batches (e.g. origin=D51:3 emits id=D51:3, should be D51:3-1), which makes
        # the id invalid (schema requires ^D\d+:\d+-\d+$). Idempotent on already-valid batch=1 output.
        if not isinstance(rewrite_out, dict):
            return
        sents = rewrite_out.get("sentence")
        if not isinstance(sents, list):
            return
        from collections import defaultdict
        cnt = defaultdict(int)
        for s in sents:
            if isinstance(s, dict) and s.get("origin"):
                cnt[s["origin"]] += 1
                s["id"] = f"{s['origin']}-{cnt[s['origin']]}"

    def rewrite(self, text:str):
        rewrite_prompt = Prompts.extract_rewrite_prompt(json.dumps(text, ensure_ascii=False))
        rewrite_out = self.llm.chat_text(
            messages=[{"role": "system", "content": Prompts.REWRITE_SYSTEM_PROMPT},
                      {"role": "user", "content": rewrite_prompt}],
        )
        # [fix] chat_text already returns a parsed dict; drop the redundant json.loads here (json.loads on a dict raises TypeError);
        # JSON parsing is done inside llm.chat_text.
        self._normalize_sentence_ids(rewrite_out)  # [batch>1] fix the "-seq" of ids
        flag,err = json_scheme.check_rewrite_json(rewrite_out, text)
        max_tries = 3
        last_err = err

        if not flag:
            for attempt in range(1, max_tries + 1):
                rewrite_out = self.llm.chat_text(
                    messages=[
                        {"role": "system", "content": Prompts.REWRITE_SYSTEM_PROMPT + "The previous run failed with the following error:"  + last_err},
                        {"role": "user", "content": rewrite_prompt},
                    ],
                    temperature=1.0,
                )
                self._normalize_sentence_ids(rewrite_out)  # [batch>1] fix the "-seq" of ids
                flag, err = json_scheme.check_rewrite_json(rewrite_out, text)
                if flag:
                    break
                else:
                    last_err = err  # keep the last error message for logging/handling

        return rewrite_out

    def rewrite_sample(self, sample: dict, rewrite_path: str, session_id_ref: int = 0):
        # [LM] for LM, rewrite config.LM_REWRITE_BATCH sessions together per call
        #   batch=1  -> key is session_i (per-session, compatible with existing files + per-session readers)
        #   batch>1  -> key is session_first-session_last (merged batch, needs the robust readers)
        if config.DATASET == "LM":
            session_items = list(sample.items())
            i = 1
            batch_size = config.LM_REWRITE_BATCH
            for batch_start in range(0, len(session_items), batch_size):
                if i < session_id_ref:
                    i += 1
                    continue
                batch = session_items[batch_start:batch_start + batch_size]
                batch_session_ids = [sid for sid, _ in batch]
                batch_text = "\n\n".join([f"Session {sid}:\n{text}" for sid, text in batch])
                if len(batch) == 1:
                    batch_identifier = batch_session_ids[0]
                else:
                    batch_identifier = f"{batch_session_ids[0]}-{batch_session_ids[-1]}"
                self.rewrite_sentence(batch_identifier, batch_text, rewrite_path)
                i += 1
            return
        i=1
        for session_id, session in sample.items():
            #
            if i<session_id_ref:
                i += 1
                continue
            self.rewrite_sentence(session_id, session, rewrite_path)
            i+=1




    def rewrite_sentence(self, session_id: int, text: str, rewrite_path: str):
        rewritten_sentences = self.rewrite(text)
        file_name = rewrite_path # "result_rewrite.json"
        with open(file_name, "a", encoding="utf-8") as f:
            record = {session_id: rewritten_sentences}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    def extract_keyword_sample(self, keyword_path: str, rewrite_path:str, ref_id:int = 0):
        self.extract_keyword(keyword_path, rewrite_path, ref_id)


    def extract_keys(self, text: str):
        keys_prompt = Prompts.extract_keyword_prompt(json.dumps(text, ensure_ascii=False), json.dumps(list(self.tags), ensure_ascii=False))
        keys_out = self.llm.chat_text(
            messages=[{"role": "system", "content": Prompts.KEYWORD_SYSTEM_PROMPT},
                      {"role": "user", "content": keys_prompt}],
        )
        # [fix] chat_text already returns a parsed dict; drop the redundant json.loads here (json.loads on a dict raises TypeError);
        # JSON parsing is done inside llm.chat_text.
        flag, err = json_scheme.check_key_json(keys_out, text)

        max_tries = 3
        last_err = err

        if not flag:
            for attempt in range(1, max_tries + 1):
                keys_out = self.llm.chat_text(
                    messages=[
                        {"role": "system", "content": Prompts.KEYWORD_SYSTEM_PROMPT+ "The previous run failed with the following error:"  + last_err},
                        {"role": "user", "content": keys_prompt},
                    ],
                    temperature=0.5,
                )
                # [fix] chat_text already returns a parsed dict; drop the redundant json.loads
                flag, err = json_scheme.check_key_json(keys_out, text)
                if flag:
                    break
                else:
                    last_err = err  # keep the last error
                    if attempt == max_tries:
                        flag, err = json_scheme.check_key_json(keys_out, text, replace=True)

        # final safety check: ensure we return a dict object, not a string
        if isinstance(keys_out, str):
            logger.warning("extract_keys: keys_out is still a string, attempting final extraction")
            try:
                keys_out = self.memory_controller.extract_json_from_content(keys_out)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"extract_keys: failed to parse final result: {e}")
                # return an empty structure instead of a string
                keys_out = {"sentence": []}

        return keys_out

    def extract_keyword(self, keyword_path: str, rewrite_path:str, ref_id:int):
        file_name = rewrite_path # "result_rewrite.json"
        record_rewrite = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                record_rewrite.append(record)

        file_name = keyword_path # "result_keyword2.json"
        with open(file_name, "a", encoding="utf-8") as f:
            for i in range(len(record_rewrite)):
                if i < ref_id:
                    continue
                # a rewrite record key may be session_i (batch=1) or session_first-session_last (batch>1),
                # so take the record's single value instead of a fixed session_{i+1} key (equivalent for plain-key files, no breakage).
                session_dict = record_rewrite[i]
                session_data = next(iter(session_dict.values())) if session_dict else None
                if session_data is None:
                    # write a null placeholder to keep line-index alignment with store_keyword (store_event_new handles keys=None)
                    logger.warning(f"record_rewrite[{i}] value is None; writing null placeholder")
                    f.write("null\n")
                    continue
                # feed only {id, text} to extract_keys, dropping the rewrite-stage tag/origin/topic/time,
                # to avoid leaking existing tag/topic into the keyword-extraction prompt (the LLM would copy them).
                sentences = session_data.get("sentence") or []
                filtered_sentences = []
                for sentence in sentences:
                    if isinstance(sentence, dict):
                        filtered_sentences.append({"id": sentence.get("id"), "text": sentence.get("text")})
                    else:
                        filtered_sentences.append(sentence)
                keys_out = self.extract_keys(filtered_sentences)
                f.write(json.dumps(keys_out, ensure_ascii=False) + "\n")



    def store_keyword(self, keyword_path: str, rewrite_path:str,) -> None:
        file_name = rewrite_path # "result_rewrite.json"
        records_event = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                records_event.append(record)

        file_name = keyword_path # "result_keyword.json"
        records_key = []

        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                records_key.append(record)

        for i in range(len(records_key)):
            # a rewrite record key may be session_i or session_first-session_last,
            # take the record's single value (equivalent to .get(session_{i+1}) for plain-key files, no breakage).
            ev = records_event[i]
            events = next(iter(ev.values())) if ev else None
            self.store_event_new(events, records_key[i], i+1)


    def store_event_new(self, events, keys,  session_id):
        # [guard] skip the whole session only when events (rewrite) is None: no events means no topic/episode, so no misalignment with embeddings
        if events is None:
            logging.warning(f"store_event_new: session_{session_id} events is None; skipping this session")
            return
        conversation_time = events.get("conversation_time")
        topic_sentences = events.get("topics") or {}
        personal_sentences = self._as_list(events.get("personal_sentences"))
        # [removed] summary->semantic memory build: summary is never queried (query_semantic_information is not in the TOOLS given to the LLM), so the whole block is removed.
        # keys is still used by the keyword-link block below (keys.get("sentence")), so keys=None is guarded there.

        episode_events = events.get("sentence")
        eid_topic_dict = {}

        if isinstance(episode_events, list):
            for ee in episode_events:
                if not isinstance(ee, dict):
                    continue
                id = ee.get("id")
                origin = ee.get("origin")
                if not id or not origin:
                    continue
                time = ee.get("time")
                topics = self._as_list(ee.get("topic"))


                prefix = origin.split(":")[0]  # "D1"
                ids = [x.strip() for x in origin.split(",")]
                embedding = self.memory.embeddings[id]
                if len(ids) != 1:
                    text = ""
                    for i in ids:
                        text = text + self.memory.raw_text[prefix].get(i)
                else:
                    origin = re.findall(r'D\d+:\d+', origin)[0]
                    text = self.memory.raw_text[prefix].get(origin)

                # [LM] for LM, store only the user's sentences
                if config.dataset == "LM":
                    if text is None:
                        continue
                    if text.split(":", 1)[0].strip() != "user":
                        continue
                ee_event = EpisodeEvent(id, text, origin, embedding, time=time, conv_time=conversation_time)
                ee_event.tag_t = ee.get("tag")
                self.memory.episode_events[id] = ee_event
                try:
                    self.memory.add_event_time(id,time)
                except Exception as _e:
                    # guard: some samples rewrite to unparseable dates; skip time indexing instead of crashing
                    logging.warning(f"add_event_time skip {id} time={time!r}: {_e}")
                eid_topic_dict[id] = topics

        self.memory.add_topics(topic_sentences, eid_topic_dict, session_id)

        for ps in personal_sentences:
            if not isinstance(ps, dict):
                continue
            personal_id = ps.get("id")
            if not personal_id:
                continue
            pid = f"D{session_id}:" + str(personal_id)
            ptext = ps.get("text")
            porigin = ps.get("origin")
            ptag = ps.get("tag")
            person = ps.get("person")
            self.memory.add_personal_information(pid,ptext,porigin,ptag,person)

        # [guard] keys=None (the keyword file line is null) means no keyword links; skip this block; topic/episode already registered above
        keywords = keys.get("sentence") if keys is not None else None
        keyword_by_sentence = {}

        i = 0
        for s in (keywords or []):
            sentence_id = s.get("sentence_id")
            ks = s.get("keyword")
            keyword_by_sentence[sentence_id] = ks or []
            if sentence_id not in self.memory.episode_events.keys():
                continue
            tag = self.memory.episode_events[sentence_id].tag_t #s.get("tag")
            origin_add = self.memory.episode_events[sentence_id].origin
            # [batch>1] look up raw text by the origin's own prefix (e.g. D5:3->D5), not the record index session_id;
            # otherwise cross-session sentences in a merged batch are looked up under the wrong D{session_id} -> None.split crash.
            _prefix = origin_add.split(":")[0]
            speaker = self.memory.raw_text.get(_prefix).get(origin_add).split(":", 1)[
                0].strip()
            if speaker not in ks:
                ks.append(speaker)
            for k in ks:
                if k not in self.memory.keys:
                    key_node = KeyNode(k)
                    self.memory.keys[k] = key_node
                link = Link(k, sentence_id, "episode", tag)
                self.memory.episode_links[f"el{i + self.episode_link_num}"] = link
                self.memory.keys[k].add_tag(tag, sentence_id)
                ori = self.memory.episode_events[sentence_id].origin
                self.memory.key_to_values[k].add((sentence_id, ori))
                self.memory.add_tag(tag, sentence_id, k)
                self.memory.episode_events[sentence_id].add_tag(tag, sentence_id)
                i += 1

        self.episode_link_num = self.episode_link_num + i
        if config.EAES_MODE:
            self._eaes_build_notes_for_session(events, keyword_by_sentence, conversation_time)

    def store_raw_text(self, raw_text, conv_embeddings=None, topic_id_list=None, topic_embeddings=None):
        self.memory.store_raw_text(raw_text, conv_embeddings, topic_id_list, topic_embeddings)

def calculate_and(list1, list2):
    if not list2:
        return 0.0
    covered = set(list1) & set(list2)
    return round(len(covered) / len(list2) * 100, 2)

if __name__ == "__main__":

    list1 = ["D1:12", "D28:20", "D2:5", "D28:15", "D9:2", "D9:2", "D9:2", "D1:10", "D1:2", "D15:17", "D21:16", "D26:11", "D3:3", "D2:29", "D26:3", "D15:12", "D7:12", "D14:15", "D13:6", "D13:6", "D28:33", "D26:9", "D28:12", "D12:19", "D28:16", "D2:9", "D7:6", "D8:12", "D18:2", "D9:9", "D9:1", "D9:1", "D21:19", "D13:2", "D6:13", "D22:15", "D19:4", "D17:3", "D17:3", "D17:3", "D6:4", "D22:13", "D16:2", "D14:23", "D18:7", "D17:20", "D17:20", "D17:20", "D6:12", "D12:16", "D1:1", "D11:1", "D7:9", "D25:1", "D12:2", "D4:2", "D28:6", "D6:6", "D16:15", "D5:20", "D5:20", "D13:10", "D20:19", "D7:10", "D13:22", "D13:22", "D17:2", "D17:2", "D17:2", "D25:3", "D28:7", "D17:4", "D17:4", "D17:4", "D11:19", "D20:16", "D1:8", "D6:2", "D4:9", "D10:14"]
    list2 = ["D1:10", "D1:11", "D1:12", "D3:4", "D4:9", "D10:9", "D20:2"]

    calculate_and(list1,list2)



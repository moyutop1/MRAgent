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
from memory.system import MemorySystem, EpisodeEvent
from memory.keyword_matching import append_keyword_if_missing
from agent.rewrite_memory import first_origin, normalize_sentence_ids, origin_ids, rewrite_windowed_session
from agent.eaes import EAESMixin
from agent.retrieval import RetrievalMixin
import logging
logger = logging.getLogger(__name__)
class Agent(EAESMixin, RetrievalMixin):
    def __init__(self, llm: LLM, memory_system: MemorySystem, memory_controller: MemoryController):
        self.llm = llm
        self.memory = memory_system
        self.memory_controller = memory_controller

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
    def _normalize_sentence_ids(rewrite_out):
        normalize_sentence_ids(rewrite_out)

    @staticmethod
    def _origin_ids(origin):
        return origin_ids(origin)

    @staticmethod
    def _first_origin(origin):
        return first_origin(origin)

    def rewrite(self, text:str):
        return rewrite_windowed_session(self.llm, text, logger=logger)

    def rewrite_sample(self, sample: dict, rewrite_path: str, session_id_ref: int = 0):
        # Windowed rewrite is always session-local: do not merge D1/D2/... into one compression window.
        if config.DATASET == "LM":
            session_items = list(sample.items())
            i = 1
            for _, (session_id, session) in enumerate(session_items):
                if i < session_id_ref:
                    i += 1
                    continue
                self.rewrite_sentence(session_id, session, rewrite_path)
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
        keys_prompt = Prompts.extract_keyword_prompt(
            json.dumps(text, ensure_ascii=False)
        )
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
                # feed only {id, text} to extract_keys, dropping rewrite-stage metadata,
                # to avoid leaking existing tag/topic into the keyword-extraction prompt (the LLM would copy them).
                sentences = session_data.get("sentence") or []
                filtered_sentences = []
                for sentence in sentences:
                    if isinstance(sentence, dict):
                        filtered_sentences.append({"id": sentence.get("id"), "text": sentence.get("text")})
                    else:
                        filtered_sentences.append(sentence)
                if not filtered_sentences:
                    keys_out = {"sentence": []}
                else:
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
        # Legacy summaries are not used by the EAES retrieval pipeline.
        # The keyword cache remains an optional hint for EAES indexing.

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
                semantic_properties = self._as_list(ee.get("semantic_properties"))


                first_origin = self._first_origin(origin)
                prefix = first_origin.split(":")[0]  # "D1"
                ids = [x.strip() for x in origin.split(",")]
                embedding = self.memory.embeddings[id]
                if len(ids) != 1:
                    raw_context = ""
                    raw_session = self.memory.raw_text.get(prefix, {})
                    for i in ids:
                        raw_context = raw_context + (raw_session.get(i) or "")
                else:
                    raw_context = self.memory.raw_text.get(prefix, {}).get(first_origin)
                text = ee.get("text") or raw_context

                # [LM] for LM, store only the user's sentences
                if config.dataset == "LM":
                    if raw_context is None:
                        continue
                    if raw_context.split(":", 1)[0].strip() != "user":
                        continue
                # semantic_properties is generated with the rewrite memory and
                # stored on EpisodeEvent; loading it does not create a new cache
                # or recompute the already persisted memory embedding.
                ee_event = EpisodeEvent(
                    id,
                    text,
                    origin,
                    embedding,
                    time=time,
                    conv_time=conversation_time,
                    semantic_properties=semantic_properties,
                )
                ee_event.tag_t = ee.get("tag")
                self.memory.episode_events[id] = ee_event
                # New rewrite memories carry source-supported event time in
                # their text. Keep legacy cached structured times readable,
                # but do not try to index a missing retired field.
                if time:
                    try:
                        self.memory.add_event_time(id, time)
                    except Exception as _e:
                        # guard: some legacy rewrites contain unparseable dates
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

        # Memory-side keywords remain only as optional hints for EAES
        # entity/attribute indexing. The removed non-EAES key/link graph is no
        # longer constructed.
        keywords = keys.get("sentence") if keys is not None else None
        keyword_by_sentence = {}

        for s in (keywords or []):
            if not isinstance(s, dict):
                continue
            sentence_id = s.get("sentence_id")
            raw_keywords = s.get("keyword")
            ks = list(raw_keywords) if isinstance(raw_keywords, list) else []
            keyword_by_sentence[sentence_id] = ks

        # Ensure every stored memory in this session has its first origin's
        # speaker as a keyword. Compare normalized exact values so casing and
        # punctuation variants do not create duplicate speaker entries.
        for ee in self._as_list(events.get("sentence")):
            if not isinstance(ee, dict):
                continue
            sentence_id = ee.get("id")
            if sentence_id not in self.memory.episode_events:
                continue
            ks = keyword_by_sentence.setdefault(sentence_id, [])
            origin_add = self.memory.episode_events[sentence_id].origin
            first_origin = self._first_origin(origin_add)
            _prefix = first_origin.split(":")[0]
            raw_speaker_text = self.memory.raw_text.get(_prefix, {}).get(first_origin, "")
            speaker = raw_speaker_text.split(":", 1)[0].strip()
            append_keyword_if_missing(ks, speaker)

        self._eaes_build_notes_for_session(
            events, keyword_by_sentence, conversation_time
        )
        self._eaes_build_parent_nodes_for_session(events)

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



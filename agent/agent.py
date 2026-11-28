import json
import logging
import random

import numpy as np

from common import config
from memory.system import EpisodeEvent, KeyNode, Link
from prompts import schema
from prompts.prompts import Prompts
from utility.evaluator import get_utility_evaluator


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, llm, memory, memory_controller, utility=None):
        self.llm = llm
        self.memory = memory
        self.memory_controller = memory_controller
        self.utility = utility
        self.episode_link_num = 0
        self.last_retrieval = None

    def utility_evaluator(self):
        if self.utility is None:
            self.utility = get_utility_evaluator()
        return self.utility

    @staticmethod
    def question_format(dataset, qa):
        question = qa["question"]
        if dataset == "locomo" and qa.get("category") == 5:
            options = ["Not mentioned in the conversation", qa["adversarial_answer"]]
            random.shuffle(options)
            return f"{question} Select the correct answer: {options[0]} or {options[1]}."
        if dataset == "locomo" and qa.get("category") == 1:
            return question + " No extra explanations."
        if dataset == "locomo" and qa.get("category") == 3:
            return question + " Give reasons with original text."
        return question

    def extract_question_keys(self, question):
        return self.llm.chat_text(
            [
                {"role": "system", "content": Prompts.QUESTION_KEY_SYSTEM_PROMPT},
                {"role": "user", "content": Prompts.extract_question_key_prompt(question)},
            ],
            model=config.RE_MODEL,
        )

    def select_key_tags(self, question, question_keys):
        candidates = []
        for item in question_keys["keywords"]:
            key = item["id"]
            tags = self.memory.get_tag_list(key)
            if len(tags) > config.TAG_MAX:
                scores = {}
                for start in range(0, len(tags), config.TAG_SCORE_BATCH):
                    batch = tags[start:start + config.TAG_SCORE_BATCH]
                    result = self.llm.chat_text(
                        [
                            {"role": "system", "content": Prompts.EVENT_KEYWORDS_SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps({
                                "question": question, "keyword": key, "tags": batch,
                            }, ensure_ascii=False)},
                        ],
                        model=config.RE_MODEL,
                    )
                    scores.update(result["tag_scores"])
                tags = [
                    tag for tag, score in sorted(
                        scores.items(), key=lambda pair: pair[1], reverse=True
                    )[:config.TAG_LIMIT]
                    if score > 0
                ]
            if tags:
                candidates.append({"key": key, "tags": tags})
        return candidates

    def select_topics(self, question_embedding):
        if not self.memory.topic_id_list:
            return []
        embeddings = np.asarray(self.memory.topic_embeddings)
        scores = embeddings @ np.asarray(question_embedding)
        indices = np.argsort(scores)[::-1][:config.TOPIC_K]
        return [
            {
                "topic_id": self.memory.topic_id_list[index],
                "summary": self.memory.topic_dict[self.memory.topic_id_list[index]].text,
            }
            for index in indices
        ]

    def select_personal_routes(self, question, question_keys):
        query_terms = {question.lower()}
        for item in question_keys["keywords"]:
            query_terms.add(item["id"].lower())
            query_terms.update(value.lower() for value in item.get("alternatives", []))
        routes = []
        for person, persona in self.memory.persona_list.items():
            if any(person.lower() in term for term in query_terms):
                routes.extend(
                    {"person": person, "aspect": aspect}
                    for aspect in persona.tag_list
                )
        return routes

    def retrieve_candidates(self, question, question_embedding, question_keys):
        keys_candidates = self.select_key_tags(question, question_keys)
        similar_topic = self.select_topics(question_embedding)
        personal_routes = self.select_personal_routes(question, question_keys)

        nodes = []
        for item in keys_candidates:
            for tag in item["tags"]:
                nodes.extend(self.memory_controller.edges_by_tag(item["key"], tag))
        for topic in similar_topic:
            nodes.extend(self.memory_controller.query_topic_events(topic["topic_id"]))
        for route in personal_routes:
            nodes.extend(self.memory_controller.query_personal_aspect(**route))

        unique = {}
        for node in nodes:
            if node["node_id"] in unique:
                unique[node["node_id"]]["routes"].append(node["route"])
            else:
                node["routes"] = [node.pop("route")]
                unique[node["node_id"]] = node
        coarse_context = {
            "keys_candidates": keys_candidates,
            "similar_topic": similar_topic,
        }
        return list(unique.values()), coarse_context, personal_routes

    def answer_question(
        self, question, category=0, question_emb=None,
        override_question_time=None, lm_current_date=None,
    ):
        question_keys = self.extract_question_keys(question)
        candidates, coarse_context, personal_routes = self.retrieve_candidates(
            question, question_emb, question_keys
        )
        if config.DUMP_CANDIDATES:
            selected, decisions = [], []
        else:
            selected, decisions = self.utility_evaluator().select_retrieval(
                question, candidates, coarse_context, config.RETRIEVE_THRESHOLD
            )
        self.last_retrieval = {
            "query": question,
            **coarse_context,
            "personal_routes": personal_routes,
            "candidates": candidates,
            "selected": selected,
            "decisions": decisions,
        }
        origins = list(dict.fromkeys(node["origin"] for node in selected))
        if config.RETRIEVAL_ONLY or config.DUMP_CANDIDATES:
            return "", origins

        result = self.llm.chat_text(
            [
                {"role": "system", "content": Prompts.FINAL_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "question": question,
                    "current_date": lm_current_date,
                    "evidence": selected,
                }, ensure_ascii=False)},
            ]
        )
        return result["answer"], origins

    def rewrite(self, text):
        result = self.llm.chat_text(
            [
                {"role": "system", "content": Prompts.REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": Prompts.extract_rewrite_prompt(text)},
            ]
        )
        valid, error = schema.check_rewrite_json(result, text)
        if not valid:
            raise ValueError(error)
        return result

    def rewrite_sample(self, sample, rewrite_path, session_id_ref=0):
        selected_sessions = []
        with open(rewrite_path, "a", encoding="utf-8") as output:
            for index, (session_id, session) in enumerate(sample.items(), start=1):
                if index < session_id_ref:
                    continue
                if config.DUMP_CANDIDATES:
                    utility, representation, keep = 1.0, None, True
                else:
                    utility, representation = self.utility_evaluator().score_write(
                        session, selected_sessions
                    )
                    keep = utility >= config.WRITE_THRESHOLD
                logger.info(
                    "[write-utility] session=%s utility=%.6f decision=%s",
                    session_id, utility, "KEEP" if keep else "DROP",
                )
                rewritten = self.rewrite(session) if keep else None
                if keep and representation is not None:
                    selected_sessions.append(representation)
                output.write(json.dumps({session_id: rewritten}, ensure_ascii=False) + "\n")

    def extract_keys(self, sentences):
        result = self.llm.chat_text(
            [
                {"role": "system", "content": Prompts.KEYWORD_SYSTEM_PROMPT},
                {"role": "user", "content": Prompts.extract_keyword_prompt(
                    json.dumps(sentences, ensure_ascii=False), ""
                )},
            ]
        )
        valid, error = schema.check_key_json(result, sentences)
        if not valid:
            raise ValueError(error)
        return result

    def extract_keyword_sample(self, keyword_path, rewrite_path, ref_id=0):
        with open(rewrite_path, encoding="utf-8") as source:
            rewrites = [json.loads(line) for line in source if line.strip()]
        with open(keyword_path, "a", encoding="utf-8") as output:
            for rewrite in rewrites[ref_id:]:
                session = next(iter(rewrite.values()))
                if session is None:
                    output.write("null\n")
                    continue
                sentences = [
                    {"id": item["id"], "text": item["text"]}
                    for item in session["sentence"]
                ]
                output.write(json.dumps(self.extract_keys(sentences), ensure_ascii=False) + "\n")

    def store_keyword(self, keyword_path, rewrite_path):
        with open(rewrite_path, encoding="utf-8") as source:
            rewrites = [json.loads(line) for line in source if line.strip()]
        with open(keyword_path, encoding="utf-8") as source:
            keywords = [json.loads(line) for line in source if line.strip()]
        for session_id, (rewrite, keys) in enumerate(zip(rewrites, keywords), start=1):
            self.store_event(next(iter(rewrite.values())), keys, session_id)

    def store_event(self, events, keys, session_id):
        if events is None:
            return
        conversation_time = events["conversation_time"]
        event_topics = {}
        for item in events["sentence"]:
            origin = item["origin"]
            raw_text = self.memory.raw_text[origin.split(":")[0]][origin]
            event = EpisodeEvent(
                item["id"], item["text"], raw_text, origin,
                self.memory.embeddings[item["id"]], item["time"], conversation_time,
            )
            event.tag = item["tag"]
            self.memory.episode_events[event.event_id] = event
            event_topics[event.event_id] = item["topic"]

        self.memory.add_topics(events["topics"], event_topics, session_id)
        for item in events["personal_sentences"]:
            self.memory.add_personal_information(
                f"D{session_id}:{item['id']}", item["text"], item["origin"],
                item["tag"], item["person"], conversation_time,
            )

        for item in keys["sentence"]:
            event = self.memory.episode_events[item["sentence_id"]]
            speaker = self.memory.raw_text[event.origin.split(":")[0]][event.origin].split(":", 1)[0]
            for key in list(dict.fromkeys(item["keyword"] + [speaker])):
                if key not in self.memory.keys:
                    self.memory.keys[key] = KeyNode(key)
                self.memory.keys[key].add_tag(event.tag, event.event_id)
                link_id = f"el{self.episode_link_num}"
                self.memory.episode_links[link_id] = Link(key, event.event_id, event.tag)
                self.memory.event_to_keys[event.event_id].add(key)
                self.episode_link_num += 1

    def store_raw_text(self, raw_text, embeddings, topic_ids, topic_embeddings):
        self.memory.store_raw_text(raw_text, embeddings, topic_ids, topic_embeddings)

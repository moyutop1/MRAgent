from collections import defaultdict


class KeyNode:
    def __init__(self, key_id):
        self.key_id = key_id
        self.text = key_id
        self.tag_list = []
        self.tag_dict = defaultdict(list)

    def add_tag(self, tag, episode_id):
        if tag not in self.tag_list:
            self.tag_list.append(tag)
        self.tag_dict[tag].append(episode_id)


class Topic:
    def __init__(self, topic_id, text):
        self.topic_id = topic_id
        self.text = text
        self.event_list = []


class PersonalEvent:
    def __init__(self, person, event_id, text, tag, origin, conversation_time):
        self.person = person
        self.event_id = event_id
        self.text = text
        self.tag = tag
        self.origin = origin
        self.conversation_time = conversation_time


class Persona:
    def __init__(self, person):
        self.person = person
        self.tag_list = []
        self.tag_dict = defaultdict(list)

    def add_information(self, event):
        if event.tag not in self.tag_list:
            self.tag_list.append(event.tag)
        self.tag_dict[event.tag].append(event)


class EpisodeEvent:
    def __init__(
        self, event_id, text, raw_text, origin, embedding, event_time,
        conversation_time,
    ):
        self.event_id = event_id
        self.text = text
        self.raw_text = raw_text
        self.origin = origin
        self.embedding = embedding
        self.event_time = event_time
        self.conversation_time = conversation_time
        self.tag = ""


class Link:
    def __init__(self, key_id, event_id, tag):
        self.key_id = key_id
        self.event_id = event_id
        self.tag = tag


class MemorySystem:
    def __init__(self):
        self.keys = {}
        self.episode_events = {}
        self.episode_links = {}
        self.event_to_keys = defaultdict(set)
        self.persona_list = {}
        self.topic_dict = {}
        self.topic_id_list = []
        self.topic_embeddings = []
        self.topic_sentence_list = []
        self.raw_text = {}
        self.embeddings = {}
        self.tid2emb = {}

    def store_raw_text(self, raw_text, embeddings, topic_ids, topic_embeddings):
        self.raw_text = raw_text
        self.embeddings = embeddings
        self.topic_id_list = list(topic_ids)
        self.topic_embeddings = topic_embeddings
        self.tid2emb = {
            topic_id: topic_embeddings[index]
            for index, topic_id in enumerate(topic_ids)
        }

    def get_tag_list(self, key):
        return list(self.keys[key].tag_list) if key in self.keys else []

    def add_topics(self, topics, event_topics, session_id):
        for topic_name, summary in (topics or {}).items():
            topic_id = f"D{session_id}:{topic_name}"
            self.topic_dict[topic_id] = Topic(topic_id, summary)
            self.topic_sentence_list.append(f"{topic_id}:{summary}")
        for event_id, topic_names in event_topics.items():
            for topic_name in topic_names or []:
                topic_id = f"D{session_id}:{topic_name}"
                if topic_id in self.topic_dict:
                    self.topic_dict[topic_id].event_list.append(event_id)

    def add_personal_information(
        self, event_id, text, origin, tag, person, conversation_time,
    ):
        if person not in self.persona_list:
            self.persona_list[person] = Persona(person)
        self.persona_list[person].add_information(
            PersonalEvent(person, event_id, text, tag, origin, conversation_time)
        )

    @staticmethod
    def _episode_node(event, route):
        return {
            "node_id": event.event_id,
            "node_type": "episode",
            "content": event.text,
            "raw_content": event.raw_text,
            "origin": event.origin,
            "event_time": event.event_time,
            "conversation_time": event.conversation_time,
            "route": route,
        }

    def event_by_tag(self, key, tag):
        route = {"type": "tag", "cue": key, "tag": tag}
        return [
            self._episode_node(self.episode_events[event_id], route)
            for event_id in self.keys[key].tag_dict[tag]
        ]

    def query_topic_events(self, topic_id):
        route = {"type": "topic", "topic_id": topic_id}
        return [
            self._episode_node(self.episode_events[event_id], route)
            for event_id in self.topic_dict[topic_id].event_list
        ]

    def query_personal_aspect(self, person, aspect):
        route = {"type": "personal", "person": person, "aspect": aspect}
        return [
            {
                "node_id": event.event_id,
                "node_type": "personal",
                "content": event.text,
                "raw_content": event.text,
                "origin": event.origin,
                "event_time": event.conversation_time,
                "conversation_time": event.conversation_time,
                "route": route,
            }
            for event in self.persona_list[person].tag_dict[aspect]
        ]

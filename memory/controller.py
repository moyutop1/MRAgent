class MemoryController:
    def __init__(self, memory, llm=None):
        self.memory = memory

    def edges_by_tag(self, cue, tag):
        return self.memory.event_by_tag(cue, tag)

    def query_topic_events(self, topic_id):
        return self.memory.query_topic_events(topic_id)

    def query_personal_aspect(self, person, aspect):
        return self.memory.query_personal_aspect(person, aspect)

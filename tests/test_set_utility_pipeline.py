import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agent.agent import Agent
from utility.model import SetUtilityModel


class FakeUtility:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.selected_sizes = []

    def score_write(self, session, selected):
        self.selected_sizes.append(len(selected))
        return next(self.scores), np.ones(8, dtype=np.float32)


class FakeMemory:
    def __init__(self):
        self.topic_id_list = ["D1:t1", "D1:t2"]
        self.topic_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        self.topic_dict = {
            "D1:t1": type("Topic", (), {"text": "pets"})(),
            "D1:t2": type("Topic", (), {"text": "travel"})(),
        }
        self.persona_list = {
            "Caroline": type("Persona", (), {"tag_list": ["hobby", "pet"]})()
        }

    def get_tag_list(self, key):
        return ["Pet", "Plan"] if key == "Caroline" else []


class FakeController:
    def __init__(self):
        self.calls = []

    @staticmethod
    def node(node_id, route):
        return {
            "node_id": node_id,
            "node_type": "episode",
            "content": node_id,
            "raw_content": node_id,
            "origin": node_id.split("-")[0],
            "event_time": "2023-05-08",
            "conversation_time": "2023-05-08",
            "route": route,
        }

    def edges_by_tag(self, cue, tag):
        self.calls.append(("tag", cue, tag))
        return [self.node("D1:1-1", {"type": "tag", "cue": cue, "tag": tag})]

    def query_topic_events(self, topic_id):
        self.calls.append(("topic", topic_id))
        return [self.node("D1:1-1", {"type": "topic", "topic_id": topic_id})]

    def query_personal_aspect(self, person, aspect):
        self.calls.append(("personal", person, aspect))
        return [self.node(f"D1:p-{aspect}", {
            "type": "personal", "person": person, "aspect": aspect,
        })]


class SetUtilityModelTests(unittest.TestCase):
    def test_selected_set_is_permutation_invariant(self):
        torch.manual_seed(7)
        model = SetUtilityModel(14, hidden_dim=24, num_heads=4, dropout=0.0).eval()
        candidate = torch.randn(2, 1, 14)
        selected = torch.randn(2, 5, 14)
        context = torch.randn(2, 4, 14)
        operation = torch.tensor([0, 1])
        original = model(candidate, selected, context, operation)
        permuted = model(candidate, selected[:, [3, 1, 4, 0, 2]], context, operation)
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)

    def test_fixed_routes_are_all_expanded_and_deduplicated(self):
        memory = FakeMemory()
        controller = FakeController()
        agent = Agent(object(), memory, controller, utility=object())
        question_keys = {
            "keywords": [{"id": "Caroline", "alternatives": ["she"]}]
        }
        nodes, context, personal = agent.retrieve_candidates(
            "What pet does Caroline have?", np.asarray([1.0, 0.0]), question_keys
        )
        self.assertEqual(context["keys_candidates"], [{
            "key": "Caroline", "tags": ["Pet", "Plan"],
        }])
        self.assertEqual(len(context["similar_topic"]), 2)
        self.assertEqual(len(personal), 2)
        self.assertEqual(len(controller.calls), 6)
        event = next(node for node in nodes if node["node_id"] == "D1:1-1")
        self.assertEqual(len(event["routes"]), 4)
        self.assertNotIn("key_sentences", context)
        self.assertNotIn("key_tag_sentences", context)

    def test_write_utility_runs_once_before_each_session_rewrite(self):
        utility = FakeUtility([0.9, 0.1])
        agent = Agent(object(), object(), object(), utility=utility)
        rewritten = []
        agent.rewrite = lambda text: rewritten.append(text) or {"kept": text}
        sessions = {"session_1": "time:2023-01-01\nfirst", "session_2": "time:2023-01-02\nsecond"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rewrite.jsonl"
            agent.rewrite_sample(sessions, str(path))
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(utility.selected_sizes, [0, 1])
        self.assertEqual(rewritten, [sessions["session_1"]])
        self.assertEqual(records[1], {"session_2": None})


if __name__ == "__main__":
    unittest.main()

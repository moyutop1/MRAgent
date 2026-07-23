import json
import sys
import types
import unittest
from unittest.mock import patch

# Keep this unit test independent from the optional python-dotenv dependency
# while retaining the real common.config module for cross-test compatibility.
dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from common import config


class _FakeLLM:
    def __init__(self):
        self.inputs = []

    def chat_text(self, messages, **_kwargs):
        self.inputs.append(json.loads(messages[-1]["content"]))
        return {
            "mode": "answer",
            "answer": "test answer",
            "supports": ["M_1"],
            "confidence": 1.0,
        }


class _FakeMemory:
    @staticmethod
    def get_eaes_support_origin(memory_ids):
        return list(memory_ids)


class _FakeController:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve_eaes_candidates(self, *_args, **_kwargs):
        return list(self.candidates)


class _AblationAgent(EAESMixin):
    def __init__(self, candidates):
        self.llm = _FakeLLM()
        self.memory = _FakeMemory()
        self.memory_controller = _FakeController(candidates)
        self.selector_calls = 0

    @staticmethod
    def parse_eaes_query(_question, _question_emb):
        return {"answer_type": "fact"}

    @staticmethod
    def _as_list(value):
        return value if isinstance(value, list) else []

    @staticmethod
    def rerank_eaes_candidates(_question, _query_plan, candidates):
        return list(candidates)

    def select_eaes_evidence(self, _question, _query_plan, candidates):
        self.selector_calls += 1
        return self._fallback_eaes_package(candidates[:1], reason="selector enabled")


def _candidates(count=20):
    return [
        {
            "memory_id": f"M_{i}",
            "event_id": f"D1:{i}-1",
            "origin": f"D1:{i}",
            "rewrite_content": f"Memory {i}",
            "score": 1.0 / i,
        }
        for i in range(1, count + 1)
    ]


class EvidenceSelectorAblationTests(unittest.TestCase):
    def test_disabled_selector_passes_every_reranked_candidate_to_reader(self):
        agent = _AblationAgent(_candidates())

        with patch.object(config, "DISABLE_EVIDENCE_SELECTOR", True):
            answer, supports = agent.answer_question_eaes("question", category=1)

        self.assertEqual(answer, "test answer")
        self.assertEqual(supports, ["M_1"])
        self.assertEqual(agent.selector_calls, 0)
        package = agent.llm.inputs[0]["evidence_package"]
        self.assertEqual(len(package["answer_items"]), 20)
        self.assertEqual(
            [item["evidence"][0]["memory_id"] for item in package["answer_items"]],
            [f"M_{i}" for i in range(1, 21)],
        )
        self.assertTrue(all(
            item["evidence"][0]["role"] == "reranked_candidate"
            for item in package["answer_items"]
        ))

    def test_enabled_selector_keeps_existing_path(self):
        agent = _AblationAgent(_candidates())

        with patch.object(config, "DISABLE_EVIDENCE_SELECTOR", False):
            agent.answer_question_eaes("question", category=1)

        self.assertEqual(agent.selector_calls, 1)
        package = agent.llm.inputs[0]["evidence_package"]
        self.assertEqual(len(package["answer_items"]), 1)


if __name__ == "__main__":
    unittest.main()

"""One-pass setwise selector tests without model weights or network access."""

from contextlib import ExitStack, nullcontext
import json
import sys
import types
import unittest
from unittest.mock import patch


dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from common import config
from llm import setwise_evidence_selector


class _Tensor:
    def __init__(self, length):
        self.shape = (1, length)

    def to(self, _device):
        return self


class _Generated:
    def __getitem__(self, _key):
        return []


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, output="M_2\nP_1\nUNKNOWN"):
        self.output = output
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "seven token prompt"

    def __call__(self, _prompt, **_kwargs):
        return {"input_ids": _Tensor(7), "attention_mask": _Tensor(7)}

    def decode(self, _tokens, **_kwargs):
        return self.output


class _Model:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        return _Generated()


class _Agent(EAESMixin):
    def __init__(self):
        self.memory = types.SimpleNamespace(
            get_eaes_support_origin=lambda ids: ["origin:" + ids[0]]
        )

    @staticmethod
    def _as_list(value):
        return value if isinstance(value, list) else []


class SetwiseSelectorTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(config, "EVIDENCE_SELECTOR_BACKEND", "setwise")
        )
        self.stack.enter_context(
            patch.object(config, "SELECTOR_MODEL_PATH", "local-instruct-model")
        )
        self.stack.enter_context(
            patch.object(config, "DISABLE_EVIDENCE_SELECTOR", False)
        )

    def test_model_reads_complete_set_once_and_returns_only_real_ids(self):
        selector = setwise_evidence_selector.SetwiseEvidenceSelector.__new__(
            setwise_evidence_selector.SetwiseEvidenceSelector
        )
        selector.max_length = 100
        selector.tokenizer = _Tokenizer()
        selector.model = _Model()
        candidates = [
            {"memory_id": "M_1", "rewrite_content": "distinct fact"},
            {"memory_id": "M_2", "rewrite_content": "covered fact"},
            {"parent_id": "P_1", "rewrite_content": "unrelated summary"},
        ]

        with patch.dict(
            sys.modules,
            {"torch": types.SimpleNamespace(inference_mode=nullcontext)},
        ):
            redundant, input_tokens = selector.select_redundant_ids(
                "question", candidates
            )

        self.assertEqual(redundant, ["M_2", "P_1"])
        self.assertEqual(input_tokens, 7)
        self.assertEqual(selector.model.calls, 1)
        payload = json.loads(selector.tokenizer.messages[-1]["content"])
        self.assertEqual(payload["candidates"], candidates)
        self.assertEqual(
            [node.get("memory_id") or node.get("parent_id")
             for node in payload["candidates"]],
            ["M_1", "M_2", "P_1"],
        )

    def test_overlength_pool_is_rejected_without_prefilter_or_truncation(self):
        selector = setwise_evidence_selector.SetwiseEvidenceSelector.__new__(
            setwise_evidence_selector.SetwiseEvidenceSelector
        )
        selector.max_length = 6
        selector.tokenizer = _Tokenizer()
        selector.model = _Model()

        with self.assertRaisesRegex(ValueError, "not truncated"):
            selector.select_redundant_ids(
                "question", [{"memory_id": "M_1", "rewrite_content": "fact"}]
            )
        self.assertEqual(selector.model.calls, 0)

    def test_agent_prunes_reported_ids_without_aliases_pairs_or_score_fields(self):
        selector = types.SimpleNamespace()
        selector.inputs = []

        def select(question, candidates):
            selector.inputs.append((question, candidates))
            return ["M_2"], 321

        selector.select_redundant_ids = select
        children = [
            {
                "memory_id": "M_1", "parent_id": "P_1",
                "conversation_time": "2023-05-08",
                "rewrite_content": "answer fact", "fused_score": 0.9,
            },
            {
                "memory_id": "M_2", "parent_id": "P_1",
                "conversation_time": "2023-05-09",
                "rewrite_content": "covered fact", "fused_score": 0.8,
            },
        ]
        parents = [{
            "parent_id": "P_1", "rewrite_content": "complementary summary",
            "fused_score": 0.7,
        }]
        with patch.object(
            setwise_evidence_selector,
            "get_setwise_evidence_selector",
            return_value=selector,
        ):
            package = _Agent().select_eaes_evidence(
                "question", {}, children, parents
            )

        self.assertEqual(len(selector.inputs), 1)
        self.assertEqual(
            selector.inputs[0][1],
            [
                {
                    "memory_id": "M_1", "parent_id": "P_1",
                    "conversation_time": "2023-05-08",
                    "rewrite_content": "answer fact",
                },
                {
                    "memory_id": "M_2", "parent_id": "P_1",
                    "conversation_time": "2023-05-09",
                    "rewrite_content": "covered fact",
                },
                {"parent_id": "P_1", "rewrite_content": "complementary summary"},
            ],
        )
        self.assertEqual(
            package["answer_items"][0]["evidence"][0]["memory_id"], "M_1"
        )
        self.assertEqual(package["selected_parent_memories"], parents)
        self.assertEqual(
            package["selector_diagnostics"]["redundant_ids"], ["M_2"]
        )
        self.assertEqual(
            [node["id"] for node in package["selector_diagnostics"]["selected_nodes"]],
            ["M_1", "P_1"],
        )

    def test_reader_receives_only_retained_set_without_backup_reintroduction(self):
        selector = types.SimpleNamespace(
            select_redundant_ids=lambda _question, _candidates: (["M_2"], 123)
        )
        reader_inputs = []

        def read(messages, **_kwargs):
            reader_inputs.append(json.loads(messages[-1]["content"]))
            return {"answer": "answer"}

        agent = _Agent()
        agent.llm = types.SimpleNamespace(chat_text=read)
        children = [
            {
                "memory_id": "M_1", "parent_id": "P_1",
                "origin": "D1:1", "conversation_time": "2023-05-08",
                "rewrite_content": "answer fact",
            },
            {
                "memory_id": "M_2", "parent_id": "P_1",
                "origin": "D1:2", "conversation_time": "2023-05-09",
                "rewrite_content": "covered fact",
            },
        ]
        with patch.object(
            setwise_evidence_selector,
            "get_setwise_evidence_selector",
            return_value=selector,
        ):
            answer, context, _, trace = agent._read_eaes_candidates(
                "question", {}, children,
                [{"parent_id": "P_1", "rewrite_content": "parent summary"}],
                return_details=True,
            )

        self.assertEqual(answer, "answer")
        self.assertEqual(context, ["D1:1", "origin:P_1"])
        self.assertNotIn("backup_candidates", reader_inputs[0])
        self.assertEqual(
            reader_inputs[0]["evidence_package"]["answer_items"][0]
            ["evidence"][0]["memory_id"],
            "M_1",
        )
        self.assertEqual(trace["redundant_ids"], ["M_2"])


if __name__ == "__main__":
    unittest.main()

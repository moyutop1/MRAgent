"""Selection/control-flow tests: no torch, transformers, network or model weights."""

from contextlib import ExitStack, nullcontext
import copy
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import types
import unittest
from unittest.mock import MagicMock, patch

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from common import config
from llm import evidence_scorer
from eval.evaluate_selector import evaluate_case


class _Tokenizer:
    # Character tokens make serialized-budget assertions transparent.
    def encode(self, text, add_special_tokens=False):
        return list(text)


class _Scorer:
    tokenizer = _Tokenizer()

    def __init__(self, callback=None):
        self.callback = callback or (lambda selected, node: 0.8)
        self.calls = []

    def score_added_value(self, question, selected, candidates):
        self.calls.append(copy.deepcopy((question, selected, candidates)))
        return {"scores": [self.callback(selected, node) for node in candidates],
                "input_tokens": [100] * len(candidates), "forward_batches": 1}


class _Reader:
    def __init__(self, responses=None):
        self.inputs = []
        self.responses = iter(responses or [{"answer": "answer"}])

    def chat_text(self, messages, **kwargs):
        self.inputs.append(json.loads(messages[-1]["content"]))
        return next(self.responses)


class _Agent(EAESMixin):
    def __init__(self, vectors=None, responses=None):
        self.memory = types.SimpleNamespace(
            eaes_notes={key: types.SimpleNamespace(retrieval_embedding=value)
                        for key, value in (vectors or {}).items()},
            eaes_parent_nodes={},
            get_eaes_support_origin=lambda ids: ["parent-origin:" + ids[0]],
        )
        self.llm = _Reader(responses)

    @staticmethod
    def _as_list(value):
        return value if isinstance(value, list) else []


def _child(key, content=None):
    return {"memory_id": key, "rewrite_content": content or key,
            "conversation_time": "2023-05-08", "origin": "origin:" + key}


def _ids(package):
    return [node["memory_id"] for item in package["answer_items"] for node in item["evidence"]]


class LocalSelectorTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in {
            "EVIDENCE_SELECTOR_BACKEND": "local", "DISABLE_EVIDENCE_SELECTOR": False,
            "SELECTOR_TOKEN_BUDGET": 10000, "SELECTOR_REDUNDANCY_WEIGHT": 0.2,
            "SELECTOR_COST_WEIGHT": 0.1, "SELECTOR_MIN_GAIN": 0.1,
            "SELECTOR_SIMILARITY_THRESHOLD": 0.85, "EAES_ROLLBACK_CHECK": False,
        }.items():
            self.stack.enter_context(patch.object(config, name, value))

    def scorer(self, callback=None):
        scorer = _Scorer(callback)
        self.stack.enter_context(patch.object(evidence_scorer, "get_local_evidence_scorer", return_value=scorer))
        return scorer

    def test_rescores_full_selected_set_and_prefers_complement_to_duplicate(self):
        def value(selected, node):
            if node["memory_id"] == "duplicate":
                return 0.9 if not selected else 0.02
            return {"first": 0.95, "complement": 0.8}[node["memory_id"]]
        scorer = self.scorer(value)
        children = [_child(key) for key in ("first", "duplicate", "complement")]
        before = copy.deepcopy(children)
        plan = {"retrieval_phrases": ["view"]}
        package = _Agent().select_eaes_evidence("original question", plan, children)
        self.assertEqual(_ids(package), ["first", "complement"])
        self.assertEqual(len(scorer.calls), 3)
        self.assertEqual([n["memory_id"] for n in scorer.calls[2][1]], ["first", "complement"])
        self.assertEqual(scorer.calls[0][0], "original question")
        self.assertEqual(package["selector_diagnostics"]["stop_reason"], "minimum_gain")
        self.assertEqual(children, before)
        self.assertEqual(plan, {"retrieval_phrases": ["view"]})

    def test_soft_redundancy_does_not_hard_delete_complementary_memory(self):
        self.scorer(lambda selected, node: 0.95)
        children = [_child("a"), _child("b")]
        package = _Agent({"a": [1, 0], "b": [1, 0]}).select_eaes_evidence("q", {}, children)
        self.assertEqual(_ids(package), ["a", "b"])
        self.assertEqual(package["selector_diagnostics"]["steps"][1]["candidates"][0]["redundancy"], 1)

    def test_redundancy_penalty_can_reject_and_be_disabled(self):
        self.scorer(lambda selected, node: 0.25)
        agent = _Agent({"a": [1, 0], "b": [1, 0]})
        children = [_child("a"), _child("b")]
        self.assertEqual(_ids(agent.select_eaes_evidence("q", {}, children)), ["a"])
        with patch.object(config, "SELECTOR_REDUNDANCY_WEIGHT", 0):
            self.assertEqual(_ids(agent.select_eaes_evidence("q", {}, children)), ["a", "b"])

    def test_full_serialized_budget_includes_container_and_rejects_overflow(self):
        self.scorer()
        agent = _Agent()
        children = [_child("a"), _child("b", "long " * 500)]
        one = {"evidence_package": {"answer_items": [{"evidence": [agent._eaes_reader_child_memory(children[0])]}]},
               "parent_memories": []}
        exact_budget = len(json.dumps(one, ensure_ascii=False))
        with patch.object(config, "SELECTOR_TOKEN_BUDGET", exact_budget):
            package = agent.select_eaes_evidence("q", {}, children)
        self.assertEqual(_ids(package), ["a"])
        trace = package["selector_diagnostics"]
        self.assertEqual(trace["selected_evidence_tokens"], exact_budget)
        self.assertEqual(trace["steps"][0]["candidates"][1]["rejected"], "token_budget")

    def test_long_relevant_candidate_loses_on_cost(self):
        self.scorer(lambda selected, node: 0.8)
        with patch.object(config, "SELECTOR_COST_WEIGHT", 1.0):
            package = _Agent().select_eaes_evidence("q", {}, [_child("long", "x" * 4000), _child("short")])
        self.assertEqual(_ids(package)[0], "short")

    def test_relevant_but_redundant_expensive_memory_is_removed(self):
        self.scorer(lambda selected, node: 0.95)
        agent = _Agent({"short": [1, 0], "long": [1, 0]})
        with patch.object(config, "SELECTOR_COST_WEIGHT", 3.0):
            package = agent.select_eaes_evidence("q", {}, [_child("short"), _child("long", "x" * 2700)])
        self.assertEqual(_ids(package), ["short"])
        last = package["selector_diagnostics"]["steps"][-1]["candidates"][0]
        self.assertEqual(last["gain"], 0.95)
        self.assertEqual(last["redundancy"], 1)
        self.assertLessEqual(last["utility"], config.SELECTOR_MIN_GAIN)

    def test_parent_and_child_compete_in_one_loop_with_typed_ids(self):
        scorer = self.scorer(lambda selected, node: 0.01 if selected else (0.99 if "parent_id" in node else 0.8))
        parent = {"parent_id": "same", "rewrite_content": "complete parent"}
        package = _Agent().select_eaes_evidence("q", {}, [_child("same")], [parent])
        self.assertEqual(_ids(package), [])
        self.assertEqual(package["selected_parent_memories"], [parent])
        self.assertEqual(len(scorer.calls[0][2]), 2)
        self.assertEqual(scorer.calls[1][1][0]["parent_id"], "same")

    def test_low_gain_stops_without_fallback_and_duplicates_deduplicated(self):
        scorer = self.scorer(lambda selected, node: 0.01)
        package = _Agent().select_eaes_evidence("q", {}, [_child("a"), _child("a")])
        self.assertEqual(_ids(package), [])
        self.assertEqual(len(scorer.calls[0][2]), 1)

    def test_overlength_is_traced_without_forced_selection(self):
        self.scorer(lambda selected, node: None)
        package = _Agent().select_eaes_evidence("q", {}, [_child("a")])
        self.assertEqual(_ids(package), [])
        self.assertEqual(package["selector_diagnostics"]["stop_reason"], "scorer_input_length")

    def test_invalid_score_is_not_silently_accepted(self):
        self.scorer(lambda selected, node: float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            _Agent().select_eaes_evidence("q", {}, [_child("a")])

    def test_reader_retry_keeps_selection_without_backup_children_or_parents(self):
        self.scorer(lambda selected, node: 0.9 if node.get("memory_id") == "b" else 0.01)
        agent = _Agent(responses=["bad JSON", {"answer": "answer"}])
        result = agent._read_eaes_candidates("q", {}, [_child("a"), _child("b")],
                                             [{"parent_id": "p", "rewrite_content": "p"}], return_details=True)
        self.assertEqual(result[1], ["origin:b"])
        self.assertEqual(result[3]["selected_nodes"][0]["id"], "b")
        self.assertEqual(agent.llm.inputs[0], agent.llm.inputs[1])
        self.assertNotIn("backup_candidates", agent.llm.inputs[0])
        self.assertEqual(agent.llm.inputs[0]["parent_memories"], [])
        payload = {key: agent.llm.inputs[0][key] for key in ("evidence_package", "parent_memories")}
        self.assertEqual(result[3]["selected_evidence_tokens"], len(json.dumps(payload, ensure_ascii=False)))

    def test_reader_receives_empty_selection(self):
        self.scorer(lambda selected, node: 0)
        agent = _Agent()
        _, context, _ = agent._read_eaes_candidates("q", {}, [_child("a")], [])
        self.assertEqual(context, [])
        self.assertEqual(agent.llm.inputs[0]["evidence_package"], {"answer_items": []})

    def test_disabled_path_does_not_load_model_and_context_not_capped(self):
        scorer = self.scorer()
        with patch.object(config, "DISABLE_EVIDENCE_SELECTOR", True):
            result = _Agent()._read_eaes_candidates("q", {}, [_child(str(i)) for i in range(25)], [])
        self.assertEqual(len(result[1]), 25)
        self.assertEqual(scorer.calls, [])
        evidence_scorer.get_local_evidence_scorer.assert_not_called()

    def test_loading_failure_propagates_instead_of_fallback(self):
        with patch.object(evidence_scorer, "get_local_evidence_scorer", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                _Agent()._read_eaes_candidates("q", {}, [_child("a")], [])

    def test_context_preserves_visible_order_not_retrieval_order(self):
        self.scorer(lambda selected, node: 0.8 if node["memory_id"] == "a" else 0.9)
        result = _Agent()._read_eaes_candidates("q", {}, [_child("a"), _child("b")], [])
        self.assertEqual(result[1], ["origin:b", "origin:a"])

    def test_question_details_api_propagates_trace(self):
        self.scorer()
        agent = _Agent()
        agent._retrieve_eaes_first_pass = lambda *args: {
            "query_plan": {}, "final_children": [_child("a")], "selected_parents": []}
        answer, context, trace = agent.answer_question_eaes("q", category=2, return_details=True)
        self.assertEqual(answer, "answer")
        self.assertEqual(context, ["origin:a"])
        self.assertEqual(trace["selected_nodes"][0]["id"], "a")


class LocalScorerContractTests(unittest.TestCase):
    def test_mock_forward_batches_yes_no_scores_and_skips_overlength_without_torch(self):
        scorer = evidence_scorer.LocalEvidenceScorer.__new__(evidence_scorer.LocalEvidenceScorer)
        scorer.batch_size, scorer.max_length = 2, 3
        scorer.no_id, scorer.yes_id = 7, 9
        scorer.forward_options = {"use_cache": False, "logits_to_keep": 1}
        scorer._build_inputs = lambda *args: [[1, 2], [1, 2, 3, 4], [3]]
        scorer.tokenizer = MagicMock()
        scorer.tokenizer.pad.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        scorer.model = MagicMock()
        logits = scorer.model.return_value.logits.__getitem__.return_value
        probabilities = logits.__getitem__.return_value.float.return_value.softmax.return_value
        probabilities.__getitem__.return_value.tolist.return_value = [0.25, 0.75]
        fake_torch = types.SimpleNamespace(inference_mode=nullcontext)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = scorer.score_added_value("q", [], [{}, {}, {}])
        self.assertEqual(result["scores"], [0.25, None, 0.75])
        self.assertEqual(result["input_tokens"], [2, 4, 1])
        self.assertEqual(result["forward_batches"], 1)
        logits.__getitem__.assert_called_once_with((slice(None), [7, 9]))
        self.assertEqual(scorer.tokenizer.pad.call_args.args[0]["input_ids"], [[1, 2], [3]])
        self.assertEqual(scorer.tokenizer.pad.call_args.args[0]["attention_mask"], [[1, 1], [1]])
        self.assertFalse(scorer.model.call_args.kwargs["use_cache"])

    def test_mock_loader_is_offline_causal_lm_with_left_padding_and_shared_pad_id(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 0
        tokenizer.eos_token = "EOS"
        tokenizer.convert_tokens_to_ids.side_effect = lambda label: {"yes": 9, "no": 7}[label]
        tokenizer.encode.side_effect = lambda text, **kwargs: {"yes": [9], "no": [7]}.get(text, [1, 2])

        def set_pad(value):
            tokenizer.pad_token_id = 0
        # A real tokenizer updates pad_token_id when pad_token is assigned.
        type(tokenizer).pad_token = property(lambda self: "EOS", lambda self, value: set_pad(value))
        model = MagicMock()
        model.to.return_value.eval.return_value = model
        model.forward = lambda logits_to_keep=0, **kwargs: None
        fake_transformers = types.SimpleNamespace(AutoTokenizer=MagicMock(), AutoModelForCausalLM=MagicMock())
        fake_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = model
        fake_torch = types.SimpleNamespace(device=lambda device: types.SimpleNamespace(type="cuda"),
                                          cuda=types.SimpleNamespace(is_available=lambda: True),
                                          bfloat16="bf16", float32="fp32")
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            scorer = evidence_scorer.LocalEvidenceScorer("local-model")
        self.assertEqual(model.config.pad_token_id, 0)
        self.assertEqual(scorer.forward_options["logits_to_keep"], 1)
        tokenizer_kwargs = fake_transformers.AutoTokenizer.from_pretrained.call_args.kwargs
        self.assertTrue(tokenizer_kwargs["local_files_only"])
        self.assertEqual(tokenizer_kwargs["padding_side"], "left")
        self.assertTrue(fake_transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["local_files_only"])

    def test_inputs_include_full_selected_content_question_and_suffix_no_scores(self):
        scorer = evidence_scorer.LocalEvidenceScorer.__new__(evidence_scorer.LocalEvidenceScorer)
        scorer.tokenizer = _Tokenizer()
        scorer.prefix_ids = list("PREFIX")
        scorer.suffix_ids = list("SUFFIX")
        scorer.max_length = 2
        rows = scorer._build_inputs("original Q", [_child("a", "selected fact")],
                                    [{**_child("b", "candidate fact"), "retrieval_score": 0.99}])
        text = "".join(rows[0])
        self.assertTrue(text.endswith("SUFFIX"))
        for phrase in ("original Q", "selected fact", "candidate fact"):
            self.assertIn(phrase, text)
        self.assertNotIn("retrieval_score", text)
        self.assertGreater(len(rows[0]), scorer.max_length)

    def test_model_factory_only_initializes_once_across_threads(self):
        with tempfile.TemporaryDirectory() as model_dir, ExitStack() as stack:
            stack.enter_context(patch.object(config, "SELECTOR_MODEL_PATH", model_dir))
            stack.enter_context(patch.object(evidence_scorer, "_SCORERS", {}))
            factory = stack.enter_context(patch.object(evidence_scorer, "LocalEvidenceScorer"))
            with ThreadPoolExecutor(max_workers=4) as pool:
                scorers = list(pool.map(lambda _: evidence_scorer.get_local_evidence_scorer(), range(12)))
            self.assertEqual(factory.call_count, 1)
            self.assertTrue(all(scorer is scorers[0] for scorer in scorers))

    def test_missing_local_directory_rejected_before_loading(self):
        with patch.object(config, "SELECTOR_MODEL_PATH", ""), patch.object(evidence_scorer, "LocalEvidenceScorer") as factory:
            with self.assertRaisesRegex(ValueError, "existing local"):
                evidence_scorer.get_local_evidence_scorer()
            factory.assert_not_called()

    def test_probe_checks_ranking_and_decrease_without_sending_labels(self):
        scorer = _Scorer(lambda selected, node: 0.01 if selected else (0.9 if node["memory_id"] == "a" else 0.1))
        result = evaluate_case(scorer, {"question": "q", "candidates": [_child("a"), _child("b")],
                                       "expanded_selected": [_child("a")], "expected_order": [["a", "b"]],
                                       "expected_decrease": ["a"]})
        self.assertTrue(all(check["passed"] for check in result["checks"]))
        self.assertEqual(scorer.calls[0][0], "q")


if __name__ == "__main__":
    unittest.main()

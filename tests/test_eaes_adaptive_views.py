import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from unittest.mock import patch


dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")
    numpy_module.__spec__ = ModuleSpec("numpy", loader=None)
    sys.modules["numpy"] = numpy_module
sys.modules["numpy"].ndarray = object

if "nltk" not in sys.modules:
    nltk_module = types.ModuleType("nltk")
    stem_module = types.ModuleType("nltk.stem")
    nltk_module.__spec__ = ModuleSpec("nltk", loader=None)
    stem_module.__spec__ = ModuleSpec("nltk.stem", loader=None)

    class _Stemmer:
        @staticmethod
        def stem(value):
            return value

    stem_module.PorterStemmer = _Stemmer
    nltk_module.stem = stem_module
    sys.modules["nltk"] = nltk_module
    sys.modules["nltk.stem"] = stem_module

llm_controller = types.ModuleType("llm.controller")
llm_controller.LLM = object
sys.modules.setdefault("llm.controller", llm_controller)
llm_embeddings = types.ModuleType("llm.embeddings")
llm_embeddings.get_embedding = lambda _texts: []
sys.modules.setdefault("llm.embeddings", llm_embeddings)

from common import config
from agent.retrieval import RetrievalMixin
from memory.controller import MemoryController


class _Store:
    eaes_notes = {}
    eaes_parent_nodes = {}
    episode_events = {}


def _item(node_id, support, rank=1, local_relevance=0.77):
    return {
        "memory_id": node_id,
        "phrase_rank": rank,
        "phrase_similarity": 0.8,
        "phrase_fidelity": 0.9,
        "question_relevance": 0.7,
        "joint_relevance": local_relevance,
        "support_potential": support,
        "phrase": node_id,
        "phrase_index": 0,
    }


class AdaptiveViewTests(unittest.TestCase):
    def test_geometric_relevance_requires_both_question_and_phrase_support(self):
        self.assertAlmostEqual(
            MemoryController._eaes_joint_relevance(0.8, 0.6, 0.7),
            0.8 ** 0.7 * 0.6 ** 0.3,
        )
        self.assertEqual(
            MemoryController._eaes_joint_relevance(0.8, 0.0, 0.7),
            0.0,
        )

    def test_phrase_node_contribution_is_bounded_and_rank_discounted(self):
        top = MemoryController._eaes_phrase_node_contribution(
            0.9, 0.8, 1, 10.0
        )
        tail = MemoryController._eaes_phrase_node_contribution(
            0.9, 0.8, 30, 10.0
        )

        self.assertAlmostEqual(top, 0.72)
        self.assertGreater(top, tail)
        self.assertGreaterEqual(tail, 0.0)

    def test_every_view_is_fused_before_fixed_channel_caps(self):
        controller = MemoryController(_Store())
        phrases = ["view 0", "view 1", "view 2"]
        child_rankings = [
            [_item("A", 0.9)],
            [_item("B", 0.8)],
            [_item("C", 0.7)],
        ]
        parent_rankings = [[{
            **_item(f"unused-{index}", 0.6 - index * 0.1),
            "parent_id": f"P{index}",
        }] for index in range(3)]
        for phrase_index, ranking in enumerate(child_rankings + parent_rankings):
            for item in ranking:
                item["phrase_index"] = phrase_index % len(phrases)

        with (
            patch.object(
                controller, "_eaes_question_relevance_maps",
                return_value=(None, {}, {}),
            ),
            patch.object(
                controller, "rank_eaes_children_per_phrase",
                return_value=child_rankings,
            ),
            patch.object(
                controller, "rank_eaes_parents_per_phrase",
                return_value=parent_rankings,
            ),
            patch.object(
                controller, "_score_eaes_view_rankings",
                return_value=[1.0] * len(phrases),
            ),
            patch.object(config, "EAES_CHILD_CANDIDATE_LIMIT", 2),
            patch.object(config, "PARENT_TOP_K", 1),
        ):
            result = controller.retrieve_eaes_adaptive_views(phrases)

        self.assertEqual(
            result["phrase_retrieval"]["child_selected_phrase_indices"],
            [0, 1, 2],
        )
        self.assertEqual(
            result["phrase_retrieval"]["parent_selected_phrase_indices"],
            [0, 1, 2],
        )
        self.assertEqual(len(result["selected_child_candidates"]), 3)
        self.assertEqual(len(result["adaptive_children"]), 2)
        self.assertEqual(len(result["adaptive_parents"]), 1)
        self.assertTrue(all(
            item["phrase_selected"]
            for ranking in result["child_rankings"] + result["parent_rankings"]
            for item in ranking
        ))

    def test_fusion_preserves_a_strong_single_view_candidate(self):
        rankings = [
            [_item("single", 0.8, 1, 1.0), _item("repeat", 0.1, 8, 0.3)],
            [_item("repeat", 0.1, 8, 0.3)],
            [_item("repeat", 0.1, 8, 0.3)],
        ]

        fused = MemoryController.fuse_eaes_channel_rankings(
            rankings, [0, 1, 2], "memory_id"
        )

        self.assertEqual(fused[0]["memory_id"], "single")
        single = fused[0]
        self.assertAlmostEqual(single["fused_score"], 0.8)
        self.assertGreater(single["fused_score"], fused[1]["fused_score"])

    def test_parent_and_child_fusion_do_not_deduplicate_each_other(self):
        child = MemoryController.fuse_eaes_channel_rankings(
            [[_item("shared", 0.1)]], [0], "memory_id"
        )
        parent_item = {
            "parent_id": "shared",
            "phrase_rank": 1,
            "phrase_similarity": 0.8,
            "phrase_fidelity": 0.9,
            "question_relevance": 0.7,
            "support_potential": 0.1,
            "phrase": "shared",
            "phrase_index": 0,
        }
        parent = MemoryController.fuse_eaes_channel_rankings(
            [[parent_item]], [0], "parent_id"
        )

        self.assertEqual(len(child), 1)
        self.assertEqual(len(parent), 1)


class _GoldEvent:
    origin = "D1:1"


class _GoldNote:
    memory_id = "M_1"
    event_id = "D1:1-1"
    parent_id = "P_1"
    origin = "D1:1"
    rewrite_content = "Caroline adopted a dog."


class _GoldParent:
    parent_id = "P_1"
    child_ids = ["M_1"]
    rewrite_content = "Caroline has a history of caring for dogs."


class _GoldMemory:
    episode_events = {"D1:1-1": _GoldEvent()}
    eaes_event_to_memory = {"D1:1-1": "M_1"}
    eaes_notes = {"M_1": _GoldNote()}
    eaes_parent_nodes = {"P_1": _GoldParent()}

    @classmethod
    def get_eaes_note(cls, memory_id):
        return cls.eaes_notes.get(memory_id)

    @classmethod
    def get_eaes_parent_node(cls, parent_id):
        return cls.eaes_parent_nodes.get(parent_id)


class _GoldController:
    @staticmethod
    def diagnose_eaes_nodes_against_phrases(*_args, **_kwargs):
        phrase_scores = [{
            "phrase_index": index,
            "phrase": f"view {index}",
            "phrase_fidelity": 0.9 - index * 0.01,
            "phrase_node_similarity": 0.8 - index * 0.01,
            "full_channel_rank": index + 1,
            "inside_top30": True,
            "phrase_selected": index == 0,
            "support_potential": 0.05 if index == 0 else 0.0,
            "diagnostic_support_potential": 0.05 - index * 0.005,
            "rank_discount": 11 / (10 + index + 1),
        } for index in range(6)]
        parent_scores = [{
            **{
                key: value for key, value in row.items()
                if key != "inside_top30"
            },
            "inside_top10": row["inside_top30"],
        } for row in phrase_scores]
        return {
            "child": {"M_1": {
                "question_relevance": 0.77,
                "phrase_scores": phrase_scores,
            }},
            "parent": {"P_1": {
                "question_relevance": 0.63,
                "phrase_scores": parent_scores,
            }},
        }


class _GoldAgent(RetrievalMixin):
    memory = _GoldMemory()
    memory_controller = _GoldController()

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    @staticmethod
    def _origin_ids(origin):
        return [origin]


class GoldDiagnosticTests(unittest.TestCase):
    def test_gold_child_exposes_saturating_fusion_and_phrase_parts(self):
        child = {
            "memory_id": "M_1",
            "fused_score": 1.0,
            "fused_score_ratio": 1.0,
            "fused_rank": 1,
            "inside_fused_safety_cap": True,
            "adaptive_k": 1,
        }
        parent = {
            "parent_id": "P_1",
            "fused_score": 1.0,
            "fused_score_ratio": 1.0,
            "fused_rank": 1,
            "inside_fused_safety_cap": True,
            "adaptive_k": 1,
        }
        retrieval = {
            "mode": "eaes",
            "query_plan": {
                "retrieval_phrases": [f"view {index}" for index in range(6)]
            },
            "phrase_retrieval": {
                "child_selected_phrase_indices": [0],
                "parent_selected_phrase_indices": [0],
            },
            "child_rankings": [],
            "parent_rankings": [],
            "child_probe_candidates": [child],
            "prefilter_candidates": [child],
            "initial_candidates": [child],
            "candidates": [child],
            "parent_probe_candidates": [parent],
            "selected_parent_pool": [parent],
            "parent_candidates": [parent],
            "counts": {"adaptive_child_k": 1, "adaptive_parent_k": 1},
        }

        diagnostics = _GoldAgent().diagnose_eaes_gold_memories(
            ["D1:1"], retrieval
        )
        gold = diagnostics["gold_origins"][0]
        child_row = gold["child_nodes"][0]

        self.assertEqual(child_row["question_relevance"], 0.77)
        self.assertEqual(child_row["fused_score"], 1.0)
        self.assertTrue(child_row["inside_fused_safety_cap"])
        self.assertTrue(child_row["inside_direct_child_pool"])
        self.assertEqual(len(child_row["phrase_scores"]), 6)
        self.assertEqual(gold["final_path"], "child_and_parent")
        self.assertEqual(
            gold["parent_nodes"][0]["origin_association"],
            "child_membership_proxy",
        )

    def test_parent_routed_gold_child_has_distinct_final_path(self):
        routed_child = {
            "memory_id": "M_1",
            "event_id": "D1:1-1",
            "origin": "D1:1",
            "parent_id": "P_1",
            "rewrite_content": "Caroline adopted a dog.",
            "retrieval_source": "parent_routed_child",
            "routed_parent_id": "P_1",
            "parent_route_score": 0.72,
            "parent_route_rank": 1,
        }
        parent = {"parent_id": "P_1", "rewrite_content": "Dog history."}
        retrieval = {
            "mode": "eaes",
            "query_plan": {
                "retrieval_phrases": [f"view {index}" for index in range(6)]
            },
            "phrase_retrieval": {
                "child_selected_phrase_indices": [0],
                "parent_selected_phrase_indices": [0],
            },
            "child_rankings": [],
            "parent_rankings": [],
            "child_probe_candidates": [],
            "prefilter_candidates": [],
            "initial_candidates": [],
            "candidates": [],
            "parent_routed_children": [routed_child],
            "selector_input_children": [routed_child],
            "selector_selected_children": [routed_child],
            "parent_probe_candidates": [parent],
            "selected_parent_pool": [parent],
            "parent_candidates": [parent],
            "selector_selected_parents": [],
            "selector": {"enabled": True, "backend": "local"},
            "parent_child_routing": {"enabled": True},
            "counts": {"adaptive_child_k": 0, "adaptive_parent_k": 1},
        }

        diagnostics = _GoldAgent().diagnose_eaes_gold_memories(
            ["D1:1"], retrieval
        )
        gold = diagnostics["gold_origins"][0]

        self.assertEqual(gold["final_path"], "parent_routed_child")
        self.assertEqual(
            gold["selector_input_path"], "parent_routed_child"
        )
        self.assertEqual(
            gold["post_selector_path"], "parent_routed_child"
        )
        self.assertTrue(gold["covered_by_parent_routed_child"])
        self.assertFalse(gold["covered_by_direct_child"])
        self.assertTrue(gold["child_nodes"][0]["inside_parent_route"])


if __name__ == "__main__":
    unittest.main()

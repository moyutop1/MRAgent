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

    def test_meg_selects_complementary_view_and_stops_on_low_gain(self):
        controller = MemoryController(_Store())
        phrases = [f"view {index}" for index in range(6)]
        child_rankings = [
            [_item("A", 0.90), _item("B", 0.80, 2)],
            [_item("A", 0.85), _item("B", 0.75, 2)],
            [_item("C", 0.40)],
            [_item("D", 0.001)],
            [],
            [],
        ]
        for phrase_index, ranking in enumerate(child_rankings):
            for item in ranking:
                item["phrase_index"] = phrase_index

        with (
            patch.object(config, "EAES_MIN_SELECTED_VIEWS", 1),
            patch.object(config, "EAES_MAX_SELECTED_VIEWS", 4),
            patch.object(config, "EAES_VIEW_GAIN_THRESHOLD", 0.15),
        ):
            selection = controller.select_eaes_views_by_meg(
                phrases,
                [0.9] * 6,
                child_rankings,
                [[] for _ in phrases],
            )

        self.assertEqual(
            selection["child_selected_phrase_indices"], [0, 2]
        )
        self.assertEqual(
            selection["parent_selected_phrase_indices"], []
        )
        self.assertEqual(
            selection["child"]["selection_history"][-1]["stop_reason"],
            "below_marginal_gain_threshold",
        )

    def test_parent_and_child_select_phrase_views_independently(self):
        controller = MemoryController(_Store())
        phrases = ["child view", "parent view"]
        child_rankings = [[_item("C", 0.9)], [_item("C", 0.1)]]
        parent_rankings = [
            [{**_item("unused", 0.1), "parent_id": "P"}],
            [{**_item("unused", 0.9), "parent_id": "P"}],
        ]
        with (
            patch.object(config, "EAES_MIN_SELECTED_VIEWS", 1),
            patch.object(config, "EAES_MAX_SELECTED_VIEWS", 1),
        ):
            selection = controller.select_eaes_views_by_meg(
                phrases,
                [0.9, 0.9],
                child_rankings,
                parent_rankings,
            )

        self.assertEqual(selection["child_selected_phrase_indices"], [0])
        self.assertEqual(selection["parent_selected_phrase_indices"], [1])

    def test_fused_score_mass_selects_a_bounded_prefix(self):
        rankings = [
            [_item("A", 0.1, 1), _item("B", 0.08, 5)],
            [_item("A", 0.09, 2), _item("C", 0.07, 6)],
        ]
        fused = MemoryController.fuse_eaes_channel_rankings(
            rankings, [0, 1], "memory_id"
        )
        retained, annotated = MemoryController.select_eaes_adaptive_prefix(
            fused, max_k=2, mass_target=0.75, min_k=1
        )

        self.assertEqual(retained[0]["memory_id"], "A")
        self.assertLessEqual(len(retained), 2)
        self.assertTrue(all("adaptive_score" not in row for row in annotated))
        self.assertTrue(all("fused_score_ratio" in row for row in annotated))
        self.assertTrue(all("normalized_mass" in row for row in annotated))
        self.assertTrue(all("cumulative_mass" in row for row in annotated))
        self.assertEqual(
            [row["inside_adaptive_prefix"] for row in annotated],
            [index < len(retained) for index in range(len(annotated))],
        )

    def test_mass_is_normalized_over_all_fused_candidates_before_cap(self):
        candidates = [
            {"memory_id": "A", "fused_score": 0.4},
            {"memory_id": "B", "fused_score": 0.3},
            {"memory_id": "C", "fused_score": 0.2},
            {"memory_id": "D", "fused_score": 0.1},
        ]

        retained, annotated = MemoryController.select_eaes_adaptive_prefix(
            candidates, max_k=2, mass_target=0.85, min_k=1
        )

        self.assertEqual([row["memory_id"] for row in retained], ["A", "B"])
        self.assertAlmostEqual(annotated[1]["cumulative_mass"], 0.7)
        self.assertFalse(annotated[2]["inside_fused_safety_cap"])
        self.assertAlmostEqual(annotated[-1]["cumulative_mass"], 1.0)

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
            "normalized_mass": 1.0,
            "cumulative_mass": 1.0,
            "inside_fused_safety_cap": True,
            "adaptive_k": 1,
            "rerank_rank": 1,
        }
        parent = {
            "parent_id": "P_1",
            "fused_score": 1.0,
            "fused_score_ratio": 1.0,
            "fused_rank": 1,
            "normalized_mass": 1.0,
            "cumulative_mass": 1.0,
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
        self.assertEqual(child_row["cumulative_mass"], 1.0)
        self.assertTrue(child_row["inside_fused_safety_cap"])
        self.assertTrue(child_row["inside_adaptive_prefix"])
        self.assertEqual(len(child_row["phrase_scores"]), 6)
        self.assertEqual(gold["final_path"], "child_and_parent")
        self.assertEqual(
            gold["parent_nodes"][0]["origin_association"],
            "child_membership_proxy",
        )


if __name__ == "__main__":
    unittest.main()

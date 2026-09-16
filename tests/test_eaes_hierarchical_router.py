import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None


# The routing math does not need an LLM or embedding model. Stub those optional
# imports so this contract test also runs in the lightweight test environment.
if np is not None:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    sys.modules.setdefault("dotenv", dotenv_module)
    if importlib.util.find_spec("openai") is None:
        llm_controller = types.ModuleType("llm.controller")
        llm_controller.LLM = object
        sys.modules.setdefault("llm.controller", llm_controller)
    if importlib.util.find_spec("sentence_transformers") is None:
        llm_embeddings = types.ModuleType("llm.embeddings")
        llm_embeddings.get_embedding = lambda _texts: []
        sys.modules.setdefault("llm.embeddings", llm_embeddings)

    from common import config
    from memory.controller import MemoryController
else:
    config = types.SimpleNamespace()
    MemoryController = None


class _Store:
    def __init__(self):
        self.eaes_notes = {}
        self.eaes_parent_nodes = {}
        self.episode_events = {}


class _Note:
    def __init__(self, index, parent_id="1-1"):
        self.memory_id = f"M_{index}"
        self.event_id = f"E_{index}"
        self.parent_id = parent_id
        self.attribute_paths = []
        self.rewrite_content = f"memory {index}"
        self.origin = f"D1:{index}"

    def to_dict(self, include_raw=False):
        return {
            "memory_id": self.memory_id,
            "event_id": self.event_id,
            "parent_id": self.parent_id,
            "origin": self.origin,
            "rewrite_content": self.rewrite_content,
        }


class _Event:
    def __init__(self, tags):
        self.tag_t = tags


def _parent_rows(count=6, raw_similarity=0.0):
    return [
        {
            "parent_id": f"1-{index}",
            "rewrite_content": f"parent {index}",
            "raw_similarity": raw_similarity,
            "matched_keyword": "topic",
        }
        for index in range(1, count + 1)
    ]


@unittest.skipUnless(np is not None, "numpy is unavailable")
class HierarchicalRouterTests(unittest.TestCase):
    def test_retrieval_tag_validation_allows_one_and_rejects_zero(self):
        store = _Store()
        note = _Note(1)
        store.episode_events[note.event_id] = _Event([
            "Caroline profile.career interest"
        ])
        controller = MemoryController(store)

        self.assertEqual(
            controller._eaes_child_tags(note),
            ["Caroline profile.career interest"],
        )

        store.episode_events[note.event_id].tag_t = []
        with self.assertRaisesRegex(ValueError, "1-4 items"):
            controller._eaes_child_tags(note)

    def test_breadth_changes_dynamic_parent_budget(self):
        controller = MemoryController(_Store())
        children = [
            {"parent_id": f"1-{index}", "base_score": 1.0}
            for index in range(1, 7)
        ]
        with (
            patch.object(
                controller, "score_eaes_parent_candidates",
                return_value=_parent_rows(),
            ),
            patch.object(config, "PARENT_RELEVANCE_FLOOR", -1.0),
            patch.object(config, "PARENT_TOP_K", 6),
        ):
            single, single_diag = controller.route_eaes_parent_candidates(
                {"breadth_value": 0.0, "detail_value": 0.5}, children
            )
            wide, wide_diag = controller.route_eaes_parent_candidates(
                {"breadth_value": 1.0, "detail_value": 0.5}, children
            )

        self.assertEqual(len(single), 4)
        self.assertEqual(len(wide), 6)
        self.assertLess(
            single_diag["target_parent_mass"],
            wide_diag["target_parent_mass"],
        )
        self.assertAlmostEqual(single_diag["child_parent_dispersion"], 1.0)

    def test_router_can_select_zero_when_no_parent_is_eligible(self):
        controller = MemoryController(_Store())
        with (
            patch.object(
                controller, "score_eaes_parent_candidates",
                return_value=_parent_rows(count=2, raw_similarity=0.0),
            ),
            patch.object(config, "PARENT_RELEVANCE_FLOOR", 0.1),
            patch.object(config, "PARENT_TOP_K", 6),
        ):
            selected, diagnostics = controller.route_eaes_parent_candidates(
                {"breadth_value": 0.5, "detail_value": 0.5}, []
            )

        self.assertEqual(selected, [])
        self.assertEqual(diagnostics["selected_parent_k"], 0)

    def test_parent_local_child_retrieval_has_been_removed(self):
        controller = MemoryController(_Store())

        self.assertFalse(hasattr(
            controller, "retrieve_eaes_parent_local_children"
        ))
        self.assertFalse(hasattr(
            controller, "merge_eaes_hierarchical_candidates"
        ))

    def test_parent_routing_does_not_mutate_child_scores(self):
        controller = MemoryController(_Store())
        children = [{
            "memory_id": "G",
            "parent_id": "1-1",
            "base_score": 0.75,
            "candidate_score": 0.75,
        }]
        with (
            patch.object(
                controller, "score_eaes_parent_candidates",
                return_value=_parent_rows(count=1, raw_similarity=0.8),
            ),
            patch.object(config, "PARENT_RELEVANCE_FLOOR", -1.0),
            patch.object(config, "PARENT_TOP_K", 1),
        ):
            controller.route_eaes_parent_candidates(
                {"breadth_value": 0.5, "detail_value": 0.5}, children
            )

        self.assertEqual(children[0]["candidate_score"], 0.75)
        self.assertNotIn("parent_boost", children[0])


if __name__ == "__main__":
    unittest.main()

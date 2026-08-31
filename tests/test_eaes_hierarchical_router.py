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

    def test_parent_local_tag_scan_uses_detail_budget_two_three_four(self):
        store = _Store()
        controller = MemoryController(store)
        for index in range(1, 6):
            note = _Note(index)
            store.eaes_notes[note.memory_id] = note
            store.episode_events[note.event_id] = types.SimpleNamespace(
                tag_t=[f"tag {index}", f"detail {index}"]
            )
            controller._eaes_tag_embedding_cache[note.memory_id] = (
                (f"tag {index}", f"detail {index}"),
                np.asarray([
                    [1.0 / index, 0.0],
                    [0.5 / index, 0.0],
                ]),
            )
        parent = [{"parent_id": "1-1", "posterior_score": 1.0}]
        with (
            patch.object(controller, "_prepare_eaes_tag_embeddings"),
            patch.object(
                controller, "_eaes_phrase_embeddings",
                return_value=np.asarray([[1.0, 0.0]]),
            ),
        ):
            sizes = []
            for detail in (0.0, 0.5, 1.0):
                candidates, diagnostics = (
                    controller.retrieve_eaes_parent_local_children(
                        ["topic"], parent, detail
                    )
                )
                sizes.append((len(candidates), diagnostics["per_parent_k"]))

        self.assertEqual(sizes, [(2, 2), (3, 3), (4, 4)])

    def test_local_children_are_additive_and_use_zero_rrf(self):
        global_children = [{
            "memory_id": "G",
            "parent_id": "1-1",
            "max_phrase_similarity": 0.8,
            "rrf_score": 0.2,
            "candidate_sources": ["global_phrase"],
        }]
        local_children = [{
            "memory_id": "L",
            "parent_id": "1-1",
            "max_phrase_similarity": 0.7,
            "candidate_sources": ["parent_local"],
        }]
        merged, diagnostics = (
            MemoryController.merge_eaes_hierarchical_candidates(
                global_children,
                local_children,
                [{"parent_id": "1-1", "posterior_score": 0.8}],
                limit=60,
            )
        )
        by_id = {item["memory_id"]: item for item in merged}

        self.assertEqual(set(by_id), {"G", "L"})
        self.assertEqual(by_id["L"]["rrf_score"], 0.0)
        self.assertEqual(diagnostics["local_added_ids"], ["L"])
        self.assertGreater(by_id["G"]["parent_boost"], 0.0)


if __name__ == "__main__":
    unittest.main()

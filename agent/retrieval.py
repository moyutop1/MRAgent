import re
from typing import Any, Dict

from common import config


def compact_eaes_retrieval(retrieval):
    """Keep retrieval-only artifacts compact but fully auditable."""
    child_fields = (
        "memory_id", "event_id", "parent_id", "origin", "tag",
        "rewrite_content", "conversation_time", "score", "rank", "retrieval_source",
        "routed_parent_id",
    )
    parent_fields = ("parent_id", "rewrite_content", "score", "rank")

    def project(items, fields):
        return [
            {key: item.get(key) for key in fields if key in item}
            for item in items or []
        ]

    return {
        "mode": "eaes",
        "query": (retrieval.get("query_plan") or {}).get("query"),
        "direct_children": project(
            retrieval.get("initial_candidates"), child_fields
        ),
        "direct_parents": project(
            retrieval.get("parent_candidate_scores"), parent_fields
        ),
        "routed_children": project(
            retrieval.get("parent_routed_children"), child_fields
        ),
        "selected_children": project(retrieval.get("candidates"), child_fields),
        "selected_parents": project(
            retrieval.get("parent_candidates"), parent_fields
        ),
        "counts": retrieval.get("counts") or {},
        "selector": retrieval.get("selector") or {},
        "parent_child_routing": retrieval.get("parent_child_routing") or {},
    }


class RetrievalMixin:
    @staticmethod
    def _unique_keep_order(items):
        return list(dict.fromkeys(item for item in items or [] if item))

    @staticmethod
    def _normalize_evidence_ids(items):
        origins = []
        for item in items or []:
            origins.extend(re.findall(r"D\d+:\d+", str(item)))
        return list(dict.fromkeys(origins))

    def _episode_ids_for_origin(self, origin):
        return [
            event_id for event_id, event in self.memory.episode_events.items()
            if origin in re.findall(r"D\d+:\d+", str(event.origin or ""))
        ]

    def diagnose_eaes_gold_memories(
            self, gold_evidence, retrieval, question_emb=None, window=2
    ):
        """Explain whether each gold origin survived direct/routed/SELECT stages."""
        if retrieval.get("mode") != "eaes":
            return None
        direct_ids = {
            item.get("memory_id") for item in retrieval.get("initial_candidates") or []
        }
        routed_ids = {
            item.get("memory_id")
            for item in retrieval.get("parent_routed_children") or []
        }
        selected_child_ids = {
            item.get("memory_id") for item in retrieval.get("candidates") or []
        }
        selected_parent_ids = {
            item.get("parent_id")
            for item in retrieval.get("parent_candidates") or []
        }
        rows = []
        for origin in self._normalize_evidence_ids(gold_evidence):
            event_ids = self._episode_ids_for_origin(origin)
            memory_ids = [
                self.memory.eaes_event_to_memory[event_id]
                for event_id in event_ids
                if event_id in self.memory.eaes_event_to_memory
            ]
            parent_ids = self._unique_keep_order([
                self.memory.eaes_notes[memory_id].parent_id
                for memory_id in memory_ids
                if self.memory.eaes_notes[memory_id].parent_id
            ])
            rows.append({
                "origin": origin,
                "event_ids": event_ids,
                "memory_ids": memory_ids,
                "parent_ids": parent_ids,
                "in_direct_child": any(mid in direct_ids for mid in memory_ids),
                "in_parent_routed_child": any(mid in routed_ids for mid in memory_ids),
                "selected_as_child": any(
                    mid in selected_child_ids for mid in memory_ids
                ),
                "selected_via_parent": any(
                    pid in selected_parent_ids for pid in parent_ids
                ),
            })
        return {"gold_origins": rows}

    def retrieve_question_evidence(
            self, question: str, category=0, question_emb=None,
            override_question_time=None, lm_current_date=None
    ) -> Dict[str, Any]:
        """Run direct retrieval, Parent routing, and unified SELECT utility."""
        if not config.EAES_MODE:
            raise RuntimeError("Non-EAES retrieval has been removed; run with --eaes.")

        first_pass = self._retrieve_eaes_first_pass(question, question_emb)
        direct_children = first_pass["final_children"]
        parents = first_pass["selected_parents"]
        merged_children, routed_children, route_details = (
            self._prepare_eaes_selector_children(
                question, question_emb, direct_children, parents
            )
        )
        selection = self.select_eaes_evidence(
            question, first_pass["query_plan"], merged_children, parents
        )
        selected_children = [
            evidence
            for item in selection["answer_items"]
            for evidence in item["evidence"]
        ]
        selected_parents = selection["selected_parent_memories"]

        def child_groups(items):
            return [
                re.findall(r"D\d+:\d+", str(item.get("origin") or ""))
                for item in items
            ]

        def parent_groups(items):
            groups = []
            for item in items:
                parent = self.memory.eaes_parent_nodes.get(item.get("parent_id"))
                groups.append([
                    origin
                    for child_id in (parent.child_ids if parent else [])
                    for origin in re.findall(
                        r"D\d+:\d+", str(self.memory.eaes_notes[child_id].origin)
                    )
                ])
            return groups

        stages = {
            "direct_child": child_groups(direct_children),
            "direct_parent": parent_groups(parents),
            "parent_routed_child": child_groups(routed_children),
            "pre_selector": child_groups(merged_children) + parent_groups(parents),
            "post_selector_child": child_groups(selected_children),
            "post_selector_parent": parent_groups(selected_parents),
        }
        stages["post_selector"] = (
            stages["post_selector_child"] + stages["post_selector_parent"]
        )
        stage_origins = {
            name: self._unique_keep_order(
                origin for group in groups for origin in group
            )
            for name, groups in stages.items()
        }
        counts = dict(first_pass["counts"])
        counts.update({
            "routed_child_k": len(routed_children),
            "pre_selector_k": len(merged_children) + len(parents),
            "post_selector_child_k": len(selected_children),
            "post_selector_parent_k": len(selected_parents),
            "post_selector_k": len(selected_children) + len(selected_parents),
        })
        final_ids = (
            [item["memory_id"] for item in selected_children]
            + [item["parent_id"] for item in selected_parents]
        )
        return {
            "mode": "eaes",
            "query_plan": first_pass["query_plan"],
            "routing": first_pass["routing"],
            "counts": counts,
            "initial_candidates": direct_children,
            "parent_candidate_scores": parents,
            "candidates": selected_children,
            "parent_candidates": selected_parents,
            "parent_routed_children": routed_children,
            "selector_input_children": merged_children,
            "retrieved_memory_ids": final_ids,
            "retrieved_event_ids": [item["event_id"] for item in selected_children],
            "retrieved_origins": stage_origins["post_selector"],
            "retrieved_origin_groups": stages["post_selector"],
            "stage_origins": stage_origins,
            "stage_origin_groups": stages,
            "retrieval_k": len(final_ids),
            "child_k": len(selected_children),
            "parent_k": len(selected_parents),
            "selector": selection["selector_diagnostics"],
            "parent_child_routing": route_details,
        }

    def answer_question(
            self, question: str, category=0, question_emb=None,
            override_question_time=None, lm_current_date=None,
            return_details=False
    ):
        if not config.EAES_MODE:
            raise RuntimeError("Non-EAES retrieval has been removed; run with --eaes.")
        return self.answer_question_eaes(
            question,
            category,
            question_emb,
            lm_current_date,
            **({"return_details": True} if return_details else {}),
        )

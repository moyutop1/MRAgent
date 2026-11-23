import json
import re

from common import config
from memory.system import EAESMemoryNote, EAESParentNode
from prompts.prompts import Prompts


class EAESMixin:
    @staticmethod
    def _eaes_query_question(question):
        text = str(question or "").strip()
        for suffix in (" No extra explanations.", " Give reasons with original text."):
            if text.endswith(suffix.strip()):
                text = text[:-len(suffix.strip())].rstrip()
        return text

    @staticmethod
    def _normalize_eaes_parent_id(parent_id):
        return str(parent_id or "").strip()

    @staticmethod
    def _eaes_memory_id(event_id):
        return "M_" + re.sub(r"[^A-Za-z0-9]+", "_", event_id).strip("_")

    def _eaes_build_notes_for_session(self, events, conversation_time, session_id):
        for spec in self._as_list(events.get("sentence")):
            if not isinstance(spec, dict):
                continue
            event_id = spec.get("id")
            event = self.memory.episode_events.get(event_id)
            if event is None:
                continue
            note = EAESMemoryNote(
                memory_id=self._eaes_memory_id(event_id),
                event_id=event_id,
                rewrite_content=spec.get("text") or event.text,
                conversation_time=conversation_time,
                origin=event.origin,
                parent_id=self._normalize_eaes_parent_id(spec.get("parent_id")),
            )
            self.memory.add_eaes_memory_note(note, session_id)

    def _eaes_build_parent_nodes_for_session(self, events, session_id):
        if not config.SEMANTIC_HIERARCHY:
            return
        for spec in self._as_list(events.get("parent_nodes")):
            if not isinstance(spec, dict):
                continue
            parent_id = self._normalize_eaes_parent_id(spec.get("parent_id"))
            rewrite_content = str(spec.get("rewrite_content") or "").strip()
            child_ids = []
            for event_id in self._as_list(spec.get("child_ids")):
                memory_id = self.memory.eaes_event_to_memory.get(event_id)
                child = self.memory.get_eaes_note(memory_id) if memory_id else None
                if child is not None:
                    child.parent_id = parent_id
                    child_ids.append(memory_id)
            if parent_id and rewrite_content and child_ids:
                self.memory.add_eaes_parent_node(
                    EAESParentNode(parent_id, rewrite_content, child_ids),
                    session_id,
                )

    @staticmethod
    def _eaes_reader_child_memory(candidate):
        return {
            "memory_id": candidate.get("memory_id"),
            "parent_id": candidate.get("parent_id"),
            "conversation_time": candidate.get("conversation_time"),
            "rewrite_content": candidate.get("rewrite_content"),
        }

    @staticmethod
    def _eaes_reader_parent_memory(candidate):
        return {
            "parent_id": candidate.get("parent_id"),
            "rewrite_content": candidate.get("rewrite_content"),
        }

    def _eaes_reader_evidence_package(self, selection):
        return {
            "answer_items": [
                {
                    "evidence": [
                        self._eaes_reader_child_memory(candidate)
                        for candidate in item.get("evidence") or []
                    ]
                }
                for item in selection.get("answer_items") or []
            ]
        }

    def _eaes_prediction_context(self, reader_input, candidates):
        candidate_by_id = {
            candidate.get("memory_id"): candidate for candidate in candidates
        }
        context = []
        package = reader_input.get("evidence_package") or {}
        for item in package.get("answer_items") or []:
            for evidence in item.get("evidence") or []:
                candidate = candidate_by_id.get(evidence.get("memory_id"))
                if candidate and candidate.get("origin"):
                    context.append(candidate["origin"])
        for parent in reader_input.get("parent_memories") or []:
            support = self.memory.get_eaes_support_origin([parent.get("parent_id")])
            origins = [
                origin for value in support
                for origin in re.findall(r"D\d+:\d+", str(value))
            ]
            context.append(",".join(dict.fromkeys(origins)))
        return [value for value in dict.fromkeys(context) if value]

    def select_eaes_evidence(
            self, question, _query_plan, candidates, parent_candidates=None
    ):
        """Remove candidates whose absence does not reduce answer utility."""
        if getattr(config, "DISABLE_EVIDENCE_SELECTOR", False):
            children = list(candidates or [])
            parents = list(parent_candidates or [])
            payloads = [
                self._eaes_reader_child_memory(candidate)
                for candidate in children
            ] + [
                self._eaes_reader_parent_memory(candidate)
                for candidate in parents
            ]
            if payloads:
                payload_text = json.dumps(
                    payloads, ensure_ascii=False, separators=(",", ":")
                )
                selected_tokens = max(1, (len(payload_text) + 3) // 4)
            else:
                selected_tokens = 0
            return {
                "answer_items": [
                    {"evidence": [candidate]} for candidate in children
                ],
                "selected_parent_memories": parents,
                "selector_diagnostics": {
                    "enabled": False,
                    "backend": "none",
                    "operation": "SELECT",
                    "selection_direction": "passthrough",
                    "stop_reason": "disabled",
                    "token_budget": None,
                    "budget_enforced": False,
                    "budget_exceeded": False,
                    "candidate_count": len(children) + len(parents),
                    "selected_count": len(children) + len(parents),
                    "selected_tokens": selected_tokens,
                    "decisions": [],
                },
            }

        from llm.unified_utility_judge import get_unified_utility_judge

        judge = get_unified_utility_judge()
        nodes = []
        seen = set()
        for kind, values in (
                ("child", candidates or []),
                ("parent", parent_candidates or [])):
            id_key = "memory_id" if kind == "child" else "parent_id"
            for candidate in values:
                node_id = candidate.get(id_key)
                if not node_id or (kind, node_id) in seen:
                    continue
                seen.add((kind, node_id))
                payload = (
                    self._eaes_reader_child_memory(candidate)
                    if kind == "child"
                    else self._eaes_reader_parent_memory(candidate)
                )
                nodes.append({
                    "kind": kind,
                    "id": node_id,
                    "score": float(candidate.get("score") or 0.0),
                    "candidate": candidate,
                    "payload": payload,
                })
        nodes.sort(key=lambda node: (-node["score"], node["kind"], node["id"]))

        selected = nodes.copy()
        decisions = []
        # Test low-score candidates first so the stronger copy survives redundancy.
        for node in reversed(nodes):
            remaining = [item for item in selected if item is not node]
            utility = judge.estimate(
                operation="SELECT",
                source=node["payload"],
                selected_set=[item["payload"] for item in remaining],
                candidate=node["payload"],
                context={
                    "query": self._eaes_query_question(question),
                    "selection_mode": "backward_elimination",
                },
            )
            decisions.append({
                "kind": node["kind"], "id": node["id"],
                "decision": "KEEP" if utility["keep"] else "DROP",
                "score": node["score"],
                "new_target_unit": utility["new_target_unit"],
                "necessary_constraint": utility["necessary_constraint"],
                "marginal_utility": utility["marginal_utility"],
            })
            if not utility["keep"]:
                selected.remove(node)

        children = [node["candidate"] for node in selected if node["kind"] == "child"]
        parents = [node["candidate"] for node in selected if node["kind"] == "parent"]
        selected_tokens = judge.count_tokens(
            [node["payload"] for node in selected]
        )
        return {
            "answer_items": [{"evidence": [child]} for child in children],
            "selected_parent_memories": parents,
            "selector_diagnostics": {
                "enabled": True,
                "backend": "unified_utility",
                "model": config.UTILITY_MODEL,
                "operation": "SELECT",
                "utility_definition": "unified_target_unit_v1",
                "utility_formula": "I(new_target_unit OR necessary_constraint)",
                "selection_direction": "backward_elimination",
                "token_budget": config.UTILITY_TOKEN_BUDGET,
                "budget_enforced": False,
                "budget_exceeded": selected_tokens > config.UTILITY_TOKEN_BUDGET,
                "candidate_count": len(nodes),
                "selected_count": len(selected),
                "selected_tokens": selected_tokens,
                "decisions": decisions,
            },
        }

    def _prepare_eaes_selector_children(
            self, question, question_emb, direct_children, parent_candidates
    ):
        """Merge direct Child retrieval with query-rescored Parent children."""
        direct_children = list(direct_children or [])
        direct_ids = {
            candidate.get("memory_id") for candidate in direct_children
            if candidate.get("memory_id")
        }
        routed, diagnostics = self.memory_controller.retrieve_eaes_parent_routed_children(
            parent_candidates,
            question_emb=question_emb,
            question_text=self._eaes_query_question(question),
            exclude_memory_ids=direct_ids,
            limit=config.EAES_PARENT_CHILD_ROUTE_LIMIT,
        )
        merged = list(direct_children)
        seen = set(direct_ids)
        for candidate in routed:
            memory_id = candidate.get("memory_id")
            if memory_id and memory_id not in seen:
                seen.add(memory_id)
                merged.append(candidate)
        diagnostics.update({
            "enabled": True,
            "direct_child_k": len(direct_children),
            "routed_child_k": len(merged) - len(direct_children),
            "selector_input_child_k": len(merged),
        })
        return merged, routed, diagnostics

    def _read_eaes_candidates(
            self, question, query_plan, candidates, parent_candidates,
            lm_current_date=None, return_details=False
    ):
        selection = self.select_eaes_evidence(
            question, query_plan, candidates, parent_candidates
        )
        selected_parents = selection["selected_parent_memories"]
        final_input = {
            "question": question,
            "evidence_package": self._eaes_reader_evidence_package(selection),
            "parent_memories": [
                self._eaes_reader_parent_memory(parent)
                for parent in selected_parents
            ],
        }
        if lm_current_date:
            final_input["current_date"] = lm_current_date
        if not final_input["evidence_package"]["answer_items"] and not selected_parents:
            result = ("no information available", [], None)
        else:
            answer_obj = self.llm.chat_text(
                messages=[
                    {"role": "system", "content": Prompts.EAES_FINAL_ANSWER_PROMPT},
                    {"role": "user", "content": json.dumps(final_input, ensure_ascii=False)},
                ],
                model=config.RE_MODEL,
            )
            raw_answer = answer_obj.get("answer") if isinstance(answer_obj, dict) else None
            result = (
                raw_answer if raw_answer is not None else "no information available",
                self._eaes_prediction_context(final_input, candidates),
                raw_answer,
            )
        if return_details:
            return result + (selection["selector_diagnostics"],)
        return result

    def _retrieve_eaes_first_pass(self, question, question_emb=None):
        query = self._eaes_query_question(question)
        children = self.memory_controller.retrieve_eaes_candidates(
            question_emb=question_emb,
            question_text=query,
            limit=config.EAES_CHILD_CANDIDATE_LIMIT,
        )
        parents = self.memory_controller.retrieve_eaes_parent_candidates(
            question_emb=question_emb,
            question_text=query,
            limit=config.PARENT_TOP_K,
        )
        return {
            "query_plan": {"query": query},
            "child_probe_candidates": children,
            "parent_probe_candidates": parents,
            "prefilter_children": children,
            "selected_parent_pool": parents,
            "initial_children": children,
            "final_children": children,
            "selected_parents": parents,
            "routing": {
                "selected_parent_k": len(parents),
                "parent_candidates": parents,
            },
            "initial_retrieval": {
                "direct_child_ids": [item["memory_id"] for item in children],
                "direct_parent_ids": [item["parent_id"] for item in parents],
            },
            "counts": {
                "direct_child_k": len(children),
                "direct_parent_k": len(parents),
                "final_child_k": len(children),
                "final_parent_k": len(parents),
                "final_total_k": len(children) + len(parents),
            },
        }

    def answer_question_eaes(
            self, question, category=0, question_emb=None,
            lm_current_date=None, return_details=False
    ):
        first_pass = self._retrieve_eaes_first_pass(question, question_emb)
        parents = first_pass["selected_parents"]
        children, _, route_details = self._prepare_eaes_selector_children(
            question, question_emb, first_pass["final_children"], parents
        )
        if not children and not parents:
            result = ("no information available", [])
            selector_disabled = getattr(
                config, "DISABLE_EVIDENCE_SELECTOR", False
            )
            details = {
                "enabled": not selector_disabled,
                "backend": "none" if selector_disabled else "unified_utility",
                "stop_reason": "no_candidates",
            }
            return result + (details,) if return_details else result

        reader_result = self._read_eaes_candidates(
            question,
            first_pass["query_plan"],
            children,
            parents,
            lm_current_date=lm_current_date,
            return_details=return_details,
        )
        answer, context = reader_result[:2]
        if return_details:
            details = reader_result[3]
            details["parent_child_routing"] = route_details
            return answer, context, details
        return answer, context

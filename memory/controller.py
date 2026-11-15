import re
import threading

import numpy as np

from common import config
from llm.embeddings import get_embedding
from prompts.schema import check_composite_tag


_EAES_EMBEDDING_LOCK = threading.Lock()


class MemoryController:
    def __init__(self, store, llm=None):
        self.memory = store
        self.llm = llm

    @staticmethod
    def _normalize_embedding_rows(values):
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _eaes_child_tags(self, note):
        event = self.memory.episode_events[note.event_id]
        tags = event.tag_t
        if not isinstance(tags, list) or not 1 <= len(tags) <= 4:
            raise ValueError(f"EAES child {note.event_id} requires 1-4 tags")
        clean_tags = []
        for tag in tags:
            clean = re.sub(r"\s+", " ", str(tag or "")).strip()
            valid, error = check_composite_tag(clean)
            if not valid:
                raise ValueError(
                    f"EAES child {note.event_id} has invalid tag {tag!r}: {error}"
                )
            clean_tags.append(clean)
        return clean_tags

    def _eaes_retrieval_text(self, note):
        tags = "\n".join(self._eaes_child_tags(note))
        return f"TAGS:\n{tags}\nMEMORY:\n{note.rewrite_content}".strip()

    def prepare_eaes_retrieval_embeddings(self):
        pending = [
            note for note in self.memory.eaes_notes.values()
            if note.retrieval_embedding is None
        ]
        if not pending:
            return
        with _EAES_EMBEDDING_LOCK:
            pending = [
                note for note in self.memory.eaes_notes.values()
                if note.retrieval_embedding is None
            ]
            if not pending:
                return
            vectors = self._normalize_embedding_rows(
                get_embedding([self._eaes_retrieval_text(note) for note in pending])
            )
            for note, vector in zip(pending, vectors):
                note.retrieval_embedding = vector

    def prepare_eaes_parent_embeddings(self):
        pending = [
            parent for parent in self.memory.eaes_parent_nodes.values()
            if parent.retrieval_embedding is None
        ]
        if not pending:
            return
        with _EAES_EMBEDDING_LOCK:
            pending = [
                parent for parent in self.memory.eaes_parent_nodes.values()
                if parent.retrieval_embedding is None
            ]
            if not pending:
                return
            vectors = self._normalize_embedding_rows(
                get_embedding([parent.rewrite_content for parent in pending])
            )
            for parent, vector in zip(pending, vectors):
                parent.retrieval_embedding = vector

    def _eaes_question_vector(self, question_emb=None, question_text=None):
        if question_emb is not None:
            return self._normalize_embedding_rows(question_emb)[0]
        return self._normalize_embedding_rows(get_embedding([question_text]))[0]

    def retrieve_eaes_candidates(
            self, question_emb=None, question_text=None, limit=None,
            exclude_memory_ids=None
    ):
        """Direct Child Top-K using the raw question and tag+rewrite embeddings."""
        self.prepare_eaes_retrieval_embeddings()
        question = self._eaes_question_vector(question_emb, question_text)
        excluded = set(exclude_memory_ids or [])
        ranked = []
        for note in self.memory.eaes_notes.values():
            if note.memory_id in excluded:
                continue
            vector = self._normalize_embedding_rows(note.retrieval_embedding)[0]
            score = float(np.dot(vector, question))
            ranked.append({
                **note.to_dict(),
                "tag": self._eaes_child_tags(note),
                "score": score,
                "retrieval_source": "direct_child",
            })
        ranked.sort(key=lambda item: (-item["score"], item["memory_id"]))
        selected = ranked[:limit or config.EAES_CHILD_CANDIDATE_LIMIT]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected

    def retrieve_eaes_parent_candidates(
            self, question_emb=None, question_text=None, limit=None,
            exclude_parent_ids=None
    ):
        """Direct Parent Top-K using the raw question and Parent rewrite."""
        self.prepare_eaes_parent_embeddings()
        question = self._eaes_question_vector(question_emb, question_text)
        excluded = set(exclude_parent_ids or [])
        ranked = []
        for parent in self.memory.eaes_parent_nodes.values():
            if parent.parent_id in excluded:
                continue
            vector = self._normalize_embedding_rows(parent.retrieval_embedding)[0]
            score = float(np.dot(vector, question))
            ranked.append({
                **parent.to_reader_dict(),
                "score": score,
            })
        ranked.sort(key=lambda item: (-item["score"], item["parent_id"]))
        selected = ranked[:limit or config.PARENT_TOP_K]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected

    def retrieve_eaes_parent_routed_children(
            self, parent_candidates, question_emb=None, question_text=None,
            exclude_memory_ids=None, limit=None
    ):
        """Expand Parent links and rescore every routed Child by query cosine."""
        self.prepare_eaes_retrieval_embeddings()
        question = self._eaes_question_vector(question_emb, question_text)
        excluded = set(exclude_memory_ids or [])
        parent_ids = list(dict.fromkeys(
            item.get("parent_id") for item in parent_candidates or []
            if item.get("parent_id")
        ))
        routed = {}
        for parent_id in parent_ids:
            parent = self.memory.eaes_parent_nodes.get(parent_id)
            if parent is None:
                continue
            for memory_id in parent.child_ids:
                if memory_id in excluded:
                    continue
                note = self.memory.eaes_notes[memory_id]
                vector = self._normalize_embedding_rows(note.retrieval_embedding)[0]
                score = float(np.dot(vector, question))
                previous = routed.get(memory_id)
                if previous is None or score > previous["score"]:
                    routed[memory_id] = {
                        **note.to_dict(),
                        "tag": self._eaes_child_tags(note),
                        "score": score,
                        "retrieval_source": "parent_routed_child",
                        "routed_parent_id": parent_id,
                    }
        ranked = sorted(
            routed.values(), key=lambda item: (-item["score"], item["memory_id"])
        )
        route_limit = limit or config.EAES_PARENT_CHILD_ROUTE_LIMIT
        selected = ranked[:route_limit]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected, {
            "source_parent_ids": parent_ids,
            "route_limit": route_limit,
            "eligible_child_k": len(ranked),
            "selected_child_k": len(selected),
            "selected_child_ids": [item["memory_id"] for item in selected],
        }

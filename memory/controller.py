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
            if note.tag_embeddings is None or note.content_embedding is None
        ]
        if not pending:
            return
        with _EAES_EMBEDDING_LOCK:
            pending = [
                note for note in self.memory.eaes_notes.values()
                if note.tag_embeddings is None or note.content_embedding is None
            ]
            if not pending:
                return
            tag_counts = [len(self._eaes_child_tags(note)) for note in pending]
            tag_texts = [
                tag for note in pending for tag in self._eaes_child_tags(note)
            ]
            vectors = self._normalize_embedding_rows(
                get_embedding(
                    tag_texts + [note.rewrite_content for note in pending]
                )
            )
            offset = 0
            for note, count in zip(pending, tag_counts):
                note.tag_embeddings = vectors[offset:offset + count]
                offset += count
            for note, vector in zip(pending, vectors[offset:]):
                note.content_embedding = vector

    def prepare_eaes_fallback_embeddings(self):
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
        # SentenceTransformer is shared by concurrent question workers.
        with _EAES_EMBEDDING_LOCK:
            return self._normalize_embedding_rows(
                get_embedding([question_text])
            )[0]

    def retrieve_eaes_candidates(
            self, cue_text, question_emb=None, question_text=None, limit=None,
            exclude_memory_ids=None
    ):
        """Retrieve mostly by Cue-to-tag similarity plus a small content channel."""
        question = self._eaes_question_vector(question_emb, question_text)
        excluded = set(exclude_memory_ids or [])
        retrieval_limit = limit or config.EAES_CHILD_CANDIDATE_LIMIT
        if not cue_text:
            self.prepare_eaes_fallback_embeddings()
            ranked = []
            for note in self.memory.eaes_notes.values():
                if note.memory_id in excluded:
                    continue
                ranked.append({
                    **note.to_dict(),
                    "tag": self._eaes_child_tags(note),
                    "score": float(np.dot(note.retrieval_embedding, question)),
                    "retrieval_source": "raw_query_fallback",
                })
            ranked.sort(key=lambda item: (-item["score"], item["memory_id"]))
            for rank, item in enumerate(ranked[:retrieval_limit], start=1):
                item["rank"] = rank
            return ranked[:retrieval_limit]

        self.prepare_eaes_retrieval_embeddings()
        cue = self._eaes_question_vector(question_text=cue_text)
        ranked = []
        for note in self.memory.eaes_notes.values():
            if note.memory_id in excluded:
                continue
            tag_score = float(np.max(note.tag_embeddings @ cue))
            content_score = float(np.dot(note.content_embedding, question))
            ranked.append({
                **note.to_dict(),
                "tag": self._eaes_child_tags(note),
                "tag_score": tag_score,
                "content_score": content_score,
            })

        tag_limit = max(1, retrieval_limit * 4 // 5)
        content_limit = max(1, retrieval_limit - tag_limit)
        tag_ranked = sorted(
            ranked, key=lambda item: (-item["tag_score"], item["memory_id"])
        )
        content_ranked = sorted(
            ranked, key=lambda item: (-item["content_score"], item["memory_id"])
        )
        selected_by_id = {}
        for item in tag_ranked[:tag_limit]:
            selected_by_id[item["memory_id"]] = (item, {"cue_tag"})
        for item in content_ranked[:content_limit]:
            memory_id = item["memory_id"]
            if memory_id in selected_by_id:
                selected_by_id[memory_id][1].add("raw_content")
            else:
                selected_by_id[memory_id] = (item, {"raw_content"})
        for item in tag_ranked:
            if len(selected_by_id) >= retrieval_limit:
                break
            selected_by_id.setdefault(
                item["memory_id"], (item, {"cue_tag"})
            )

        selected = []
        for item, sources in selected_by_id.values():
            item["score"] = max(
                item["tag_score"] if "cue_tag" in sources else -1.0,
                item["content_score"] if "raw_content" in sources else -1.0,
            )
            item["retrieval_source"] = "+".join(sorted(sources))
            selected.append(item)
        selected.sort(key=lambda item: (-item["score"], item["memory_id"]))
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
        """Expand Parent links and rescore routed Children by content cosine."""
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
                score = float(np.dot(note.content_embedding, question))
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

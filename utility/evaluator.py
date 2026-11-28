import re
from functools import lru_cache

import numpy as np
import torch

from common import config
from llm.rag_utils import get_embeddings
from utility.model import SetUtilityModel


DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
SLASH_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")
TEXT_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
MONTHS = {
    name: index for index, name in enumerate(
        "january february march april may june july august september october november december".split(),
        start=1,
    )
}


def temporal_features(event_time=None, conversation_time=None):
    values = []
    for value in (event_time, conversation_time):
        text = str(value or "")
        match = DATE_RE.search(text) or SLASH_DATE_RE.search(text)
        if match:
            year, month, day = map(int, match.groups())
        else:
            match = TEXT_DATE_RE.search(text)
            if match:
                day, month_name, year = match.groups()
                year, month, day = int(year), MONTHS[month_name.lower()], int(day)
            else:
                year, month, day = 2000, 0, 0
        values.extend(((year - 2000) / 100, month / 12, day / 31))
    return values


def node_text(node):
    routes = node.get("routes") or [node.get("route") or {}]
    route_text = "; ".join(
        ", ".join(f"{key}={value}" for key, value in route.items())
        for route in routes
    )
    return (
        f"type: {node.get('node_type', 'episode')}; content: {node['content']}; "
        f"event time: {node.get('event_time') or 'unknown'}; "
        f"conversation time: {node.get('conversation_time') or 'unknown'}; "
        f"route: {route_text}"
    )


class UnifiedUtilityEvaluator:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = SetUtilityModel(**checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _features(self, texts, times):
        semantic = get_embeddings(texts)
        temporal = np.asarray(times, dtype=np.float32)
        return np.concatenate((semantic, temporal), axis=1).astype(np.float32)

    def score_write(self, session_text, selected_sessions):
        lines = [line for line in session_text.splitlines() if line.strip()]
        conversation_time = lines[0].removeprefix("time:")
        candidate = self._features(
            lines,
            [temporal_features(conversation_time, conversation_time) for _ in lines],
        )
        context = self._features(
            ["operation: retain this raw session before rewriting"],
            [temporal_features()],
        )
        selected = np.stack(selected_sessions) if selected_sessions else np.empty((0, candidate.shape[1]), np.float32)
        logit = self._forward(candidate[None], selected[None], context[None], 0)[0]
        return float(torch.sigmoid(logit)), candidate.mean(axis=0)

    def select_retrieval(self, question, candidates, coarse_context, threshold):
        if not candidates:
            return [], []
        context_texts = [f"question: {question}"]
        for item in coarse_context["keys_candidates"]:
            context_texts.extend(
                f"cue: {item['key']}; tag: {tag}" for tag in item["tags"]
            )
        context_texts.extend(
            f"topic: {item['topic_id']}; summary: {item['summary']}"
            for item in coarse_context["similar_topic"]
        )
        context = self._features(context_texts, [temporal_features()] * len(context_texts))
        candidate_features = self._features(
            [node_text(node) for node in candidates],
            [temporal_features(node.get("event_time"), node.get("conversation_time")) for node in candidates],
        )

        selected_indices = []
        remaining = list(range(len(candidates)))
        decisions = []
        while remaining:
            selected = candidate_features[selected_indices]
            logits = self._score_candidates(
                candidate_features[remaining][None], selected[None], context[None]
            )[0]
            scores = torch.sigmoid(logits).cpu().numpy()
            best_position = int(scores.argmax())
            best_index = remaining[best_position]
            if scores[best_position] < threshold:
                decisions.extend(
                    {
                        "node_id": candidates[index]["node_id"],
                        "utility": float(score),
                        "keep": False,
                    }
                    for index, score in zip(remaining, scores)
                )
                break
            decisions.append({
                "node_id": candidates[best_index]["node_id"],
                "utility": float(scores[best_position]),
                "keep": True,
            })
            selected_indices.append(best_index)
            remaining.remove(best_index)

        return [candidates[index] for index in selected_indices], decisions

    def _forward(self, candidate, selected, context, operation):
        with torch.inference_mode():
            return self.model(
                torch.as_tensor(candidate, device=self.device),
                torch.as_tensor(selected, device=self.device),
                torch.as_tensor(context, device=self.device),
                torch.full((candidate.shape[0],), operation, dtype=torch.long, device=self.device),
            )

    def _score_candidates(self, candidates, selected, context):
        with torch.inference_mode():
            return self.model.score_candidates(
                torch.as_tensor(candidates, device=self.device),
                torch.as_tensor(selected, device=self.device),
                torch.as_tensor(context, device=self.device),
                torch.ones((candidates.shape[0],), dtype=torch.long, device=self.device),
            )


@lru_cache(maxsize=1)
def get_utility_evaluator():
    return UnifiedUtilityEvaluator(config.UTILITY_CHECKPOINT, config.UTILITY_DEVICE)
